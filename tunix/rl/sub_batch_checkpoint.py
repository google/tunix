# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Sub-batch checkpoint manager for intra-global-step RL pipeline resilience."""

import collections.abc
import dataclasses
import pickle
from typing import Any


from absl import logging
import jax
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_manager as sft_checkpoint_manager
from tunix.sft import checkpoint_options

TrajectoryItem = agent_types.TrajectoryItem
Hashable = collections.abc.Hashable

# Orbax scalar leaves pass through np.asarray. Ints outside this range become
# object-dtype arrays that TensorStore rejects at save time.
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max

# The sub-batch key is a two-part encoding: key = train_steps * KEY_BASE +
# local. LEFT: the optimizer-apply count of the weights this snapshot's buffer
# was accumulated on, read from the trainer at save time -- never reconstructed
# by dividing a micro-step counter. RIGHT: a per-window counter that only
# orders snapshots within their window and resets at every apply.
# Reconciliation for weights restored at T is max{key : key // KEY_BASE == T}.
KEY_BASE = 1_000_000


@dataclasses.dataclass
class SubBatchPreservationPolicy(
    ocp.training.preservation_policies.PreservationPolicy
):
  """Preserves every snapshot of the newest `windows_to_keep` accumulation.

  Retention's requirement is the newest window plus the one before it:
  (the newest is being written; async lag can leave the trainer's durable
  checkpoint one apply behind the newest keys on disk, so restore may land
  in the previous window).

  windows_to_keep: how many applies behind the trainer's durable checkpoint
    may lag. Per-apply trainer cadence needs 2.
  """
  windows_to_keep: int = 2

  def should_preserve(self, checkpoints, *, context):
    del context
    parents = sorted({ck.step // KEY_BASE for ck in checkpoints})
    keep = set(parents[-self.windows_to_keep :])
    return [ck.step // KEY_BASE in keep for ck in checkpoints]


def resolve_sub_batch_checkpointing_defaults(
    options: checkpoint_options.CheckpointingOptions | None = None,
    *,
    run_options: Any = None,
) -> checkpoint_options.TunixCheckpointingOptions:
  """Resolves options with Sub-batch defaults.

  Args:
    options: Partial or complete SUB-BATCH options. every set field is
      respected. None means "derive everything".
    run_options: The run's (trainer-stream) checkpointing options, read only
      the stream-neutral fields (async toggle, async options) are consulted.

  Returns:
    A fully populated `TunixCheckpointingOptions`.
  """
  save_policy = None
  preserve_policy = None
  step_name_format = None
  enable_async = None
  async_options = None
  if options is not None:
    save_policy = options.save_decision_policy
    preserve_policy = options.preservation_policy
    step_name_format = options.step_name_format
    enable_async = options.enable_async_checkpointing
    async_options = options.async_options
  if run_options is not None:
    # getattr, not attribute access: run configs are read protocol-tolerantly
    # (the learner passes whatever checkpointing_options the run carries,
    # which need not define these fields).
    if enable_async is None:
      enable_async = getattr(run_options, "enable_async_checkpointing", None)
    if async_options is None:
      async_options = getattr(run_options, "async_options", None)

  return checkpoint_options.TunixCheckpointingOptions(
      save_decision_policy=(
          save_policy
          or ocp.training.save_decision_policies.FixedIntervalPolicy(interval=1)
      ),
      preservation_policy=(preserve_policy or SubBatchPreservationPolicy()),
      step_name_format=(
          step_name_format or ocp.path.step.standard_name_format()
      ),
      enable_async_checkpointing=(
          True if enable_async is None else enable_async
      ),
      async_options=async_options,
  )


def _normalize_jax_leaves(obj: Any) -> Any:
  """Normalizes jax array leaves to numpy before pickling.

  Single-device jax arrays do pickle, but tying trajectory payloads to
  jax's pickle behavior across versions is an avoidable dependency; numpy's
  is stable. A dedicated walk rather than jax.tree.map: pytree mapping
  SORTS dict keys, which raises on the mixed-type keys env metadata can
  legitimately hold. Traverses exactly plain dict/list/tuple containers and
  dataclasses (where trajectories keep their fields); anything else --
  namedtuples, mapping subclasses, arbitrary objects -- is left to pickle
  as-is, where a nested jax array still pickles natively as the fallback.

  Args:
    obj: The object to normalize.

  Returns:
    The normalized object.
  """
  if isinstance(obj, jax.Array):
    return np.asarray(obj)
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return dataclasses.replace(
        obj,
        **{
            f.name: _normalize_jax_leaves(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        },
    )
  if type(obj) is dict:  # pylint: disable=unidiomatic-typecheck
    return {k: _normalize_jax_leaves(v) for k, v in obj.items()}
  if type(obj) is list:  # pylint: disable=unidiomatic-typecheck
    return [_normalize_jax_leaves(v) for v in obj]
  if type(obj) is tuple:  # pylint: disable=unidiomatic-typecheck
    return tuple(_normalize_jax_leaves(v) for v in obj)
  return obj


def _validate_group_id(group_id: Any) -> Any:
  """Requires group ids that survive the checkpoint round-trip verbatim.

  Args:
    group_id: The group ID to validate.

  Returns:
    The validated group ID.

  Raises:
    ValueError: If the group ID is not a string or integer, or is an integer
      outside of the int64 range.
  """
  if isinstance(group_id, np.integer):
    group_id = int(group_id)
  if isinstance(group_id, int) and not isinstance(group_id, bool):
    if not _INT64_MIN <= group_id <= _INT64_MAX:
      raise ValueError(
          "Sub-batch checkpointing requires group ids within int64 range so"
          f" ledger keys round-trip verbatim; got {group_id!r}. Adjust the"
          " orchestrator's group_key_fn."
      )
    return group_id
  if isinstance(group_id, str):
    return group_id
  raise ValueError(
      "Sub-batch checkpointing requires str or int group ids so ledger keys"
      f" round-trip verbatim; got {type(group_id).__name__}: {group_id!r}."
      " Adjust the orchestrator's group_key_fn."
  )


def _trajectory_item_to_serializable(item: TrajectoryItem) -> dict[str, Any]:
  """Serializes an item as native identity + one pickled payload.

  Identity (group_id / pair_index / start_step) stays native: restore logic
  consumes it directly. Everything else ((traj, metadata), whether traj is a
  Trajectory dataclass or a Token-mode dict) is one pickle.dumps blob
  stored as a 1-D uint8 array: Orbax cannot store bytes, but a uint8 array
  is a plain numeric array it stores natively. Pickle round-trips every
  component exactly (tuples, bytes, datetimes, zero-size arrays, custom
  objects).

  Args:
    item: The TrajectoryItem to serialize.

  Returns:
    A dictionary representing the serializable TrajectoryItem.
  """
  payload = pickle.dumps(
      _normalize_jax_leaves((item.traj, item.metadata)),
      protocol=pickle.HIGHEST_PROTOCOL,
  )
  return {
      "group_id": _validate_group_id(item.group_id),
      "pair_index": item.pair_index,
      "start_step": item.start_step,
      "payload": np.frombuffer(payload, dtype=np.uint8),
  }


def _trajectory_item_from_serializable(data: dict[str, Any]) -> TrajectoryItem:
  """Rebuilds a `TrajectoryItem` from `_trajectory_item_to_serializable`.

  Args:
    data: A dictionary representing the serializable TrajectoryItem.

  Returns:
    A TrajectoryItem.
  """
  traj, metadata = pickle.loads(  # pylint: disable=g-unsafe-pickle-load
      np.asarray(data["payload"], dtype=np.uint8).tobytes()
  )
  return TrajectoryItem(
      group_id=data["group_id"],
      pair_index=data["pair_index"],
      start_step=data["start_step"],
      traj=traj,
      metadata=metadata,
  )


@dataclasses.dataclass
class SubBatchState:
  """Deserialized sub-batch checkpoint.

  Attributes:
    full_batch_size: Batch geometry the snapshot was saved under.
    sub_batch_key: The encoded key this snapshot was saved at (`train_steps *
      KEY_BASE + local`).
    iter_steps: The trainer micro-step count when this snapshot was taken.
    global_step: Global training step the snapshot belongs to.
    grad_accum_steps: `gradient_accumulation_steps` the run was configured with.
      Restore declines on mismatch.
    step_complete: True when this snapshot was taken at the final apply of its
      global step. On resume the learner then starts fresh at `global_step + 1`
      instead of resuming mid-step.
    completed_group_ids: Prompt groups whose rollouts are fully generated (skip
      regeneration on resume).
    trained_trajectory_counts: Map tracking (group_id, pair_index) -> trained
      epoch count.
    active_group_trajectories: Rollouts not yet fully consumed by the trainer,
      re-injected on resume.
    training_state: The live gradient-accumulation state, opaque to the manager.
      Both training modes persist the trainer's `GradientAccumulator` contents
      (`{"grads": <param tree>, "denom": scalar}`); None when the snapshot was
      taken at an apply boundary (the accumulator was just reset -- `denom == 0`
      -- and a restored trainer starts its window with a fresh accumulator
      anyway, so there is nothing to inject).
  """
  sub_batch_key: int
  iter_steps: int
  global_step: int
  grad_accum_steps: int
  step_complete: bool
  completed_group_ids: list[Hashable]
  trained_trajectory_counts: dict[tuple[Hashable, int], int]
  active_group_trajectories: list[TrajectoryItem]
  training_state: Any | None = None
  full_batch_size: int | None = None
  # BENCHMARK instrumentation (temporary, commit 4): wall-clock spent inside
  # the interrupted step at snapshot time, and the run's fixed experiment
  # anchor -- both ride custom_metadata, not the checkpointables schema.
  sub_step_elapsed: float = 0.0
  experiment_start_time: float = 0.0


class SubBatchCheckpointManager(sft_checkpoint_manager.BaseCheckpointManager):
  """Persists the rollout ledger and grad-accum buffer per trainer micro-step."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: checkpoint_options.CheckpointingOptions | None = None,
      *,
      run_options: Any = None,
  ):
    """Initializes the manager.

    Args:
      root_directory: Root directory for sub-batch snapshots. If None, the
        manager is disabled and every method no-ops.
      options: Checkpointing options; fields left None are filled with sub-batch
        defaults (`resolve_sub_batch_checkpointing_defaults`), NOT the
        trainer's defaults, whose save policy would silently skip snapshots.
      run_options: The run's trainer-stream checkpointing options; only
        stream-neutral fields (async toggle, async timeouts) are consulted,
        never its policies (see resolve_sub_batch_checkpointing_defaults).
    """
    super().__init__(
        root_directory=root_directory,
        options=resolve_sub_batch_checkpointing_defaults(
            options, run_options=run_options
        ),
    )

  @property
  def enabled(self) -> bool:
    return self._checkpointer is not None

  def save(
      self,
      train_steps: int,
      local: int,
      *,
      iter_steps: int,
      global_step: int,
      grad_accum_steps: int,
      step_complete: bool,
      completed_group_ids: list[Hashable],
      trained_trajectory_counts: dict[tuple[Hashable, int], int],
      active_group_trajectories: list[TrajectoryItem],
      training_state: Any | None,
      num_generations: int | None = None,
      full_batch_size: int | None = None,
      sub_step_elapsed: float = 0.0,
      experiment_start_time: float = 0.0,
  ) -> None:
    """Persists one snapshot keyed by `train_steps * KEY_BASE + local`.

    `train_steps` is read from the trainer (the apply count of the weights
    the accumulator sits on) and `local` is the caller-maintained count of
    micro-steps completed in the current window (reset at each apply). The
    trainer's `iter_steps` rides in the payload for the resume-time counter
    fixup; it is NOT the key, because under sequence packing the window width
    in micro-steps is data-dependent and `iter_steps // k` would decode the
    wrong parent.

    `num_generations` and `full_batch_size` are recorded in meta as batch
    geometry: the ledger's per-pair accounting and the group-id step
    arithmetic (group_id // full_batch_size) are only valid when the resumed
    run uses the same values. num_generations is checked inside try_restore;
    full_batch_size is returned in `SubBatchState` for the learner to check
    once the dataset reveals the current value (it is not known at restore
    time).

    `training_state` is the live accumulator state for MID-WINDOW snapshots
    and None at apply boundaries.

    The save begins by waiting on the previous async save. Upstream Orbax swaps
    in a fresh v0 checkpointer per save before the previous finalize thread has
    joined, so overlapping saves can mark step N complete while its commit
    futures are orphaned. Serializing saves closes that hole.
    """
    if self._checkpointer is None:
      return

    if not 0 <= local < KEY_BASE:
      raise ValueError(
          f"sub-batch local index {local} outside [0, {KEY_BASE}); the"
          " per-window micro-step counter is not resetting at applies."
      )

    sub_batch_key = train_steps * KEY_BASE + local
    self._checkpointer.wait()
    checkpointables = {
        "meta": {
            "iter_steps": iter_steps,
            "global_step": global_step,
            "grad_accum_steps": grad_accum_steps,
            "step_complete": step_complete,
            "has_training_state": training_state is not None,
            "num_generations": num_generations,
            "full_batch_size": full_batch_size,
        },
        "rollout": {
            "completed_group_ids": [
                _validate_group_id(g) for g in completed_group_ids
            ],
            "active_group_trajectories": [
                _trajectory_item_to_serializable(item)
                for item in active_group_trajectories
            ],
            "trained_trajectory_counts": [
                {
                    "group_id": _validate_group_id(k[0]),
                    "pair_index": k[1],
                    "count": v,
                }
                for k, v in trained_trajectory_counts.items()
            ],
        },
    }
    if training_state is not None:
      checkpointables["state"] = {"training_state": training_state}
    self._save_checkpointables(
        sub_batch_key,
        checkpointables,
        force=True,
        # BENCHMARK instrumentation (temporary): rides custom_metadata so
        # the checkpointables schema is untouched and the fields cost one
        # JSON entry, not a tensor write.
        custom_metadata={
            "sub_step_elapsed": sub_step_elapsed,
            "experiment_start_time": experiment_start_time,
        },
        overwrite=True,
    )

  def purge_steps_above(self, bound: int) -> None:
    """Deletes every snapshot with key > `bound` (dead-lineage cleanup).

    Called on restore: keys above the resume point were written by a run whose
    weights past that point never became durable, so their buffers/ledgers
    describe a divergent lineage. If left on disk, a later restore whose window
    reaches them would inject them onto mismatched weights (silent training
    corruption). The v1 Checkpointer has no public delete; this goes through
    the same `_manager.delete` its own overwrite path uses.
    """
    if self._checkpointer is None:
      return
    stale = sorted(
        ck.step for ck in self._checkpointer.checkpoints if ck.step > bound
    )
    for step in stale:
      try:
        self._checkpointer._manager.delete(step)  # pylint: disable=protected-access
      except FileNotFoundError:
        pass
    if stale:
      logging.info(
          "Sub-batch restore purged %d dead-lineage snapshot(s) above key %d:"
          " %s",
          len(stale),
          bound,
          stale,
      )

  def _report_no_usable_snapshot(self, train_steps: int) -> None:
    """Report when the restore window is empty but sub-batch snapshots survive on disk."""
    assert self._checkpointer is not None
    leftover = sorted(ck.step for ck in self._checkpointer.checkpoints)
    if not leftover:
      return  # genuinely fresh, nothing to report
    latest = max(leftover)
    latest_parent, latest_local = divmod(latest, KEY_BASE)
    meta_desc = "unreadable"
    step_complete = False
    try:
      meta = self._checkpointer.load_checkpointables(
          latest, abstract_checkpointables={"meta": None}
      )["meta"]
      step_complete = bool(meta.get("step_complete", False))
      meta_desc = (
          f"global_step={meta.get('global_step', -1)},"
          f" step_complete={step_complete},"
          f" grad_accum_steps={meta.get('grad_accum_steps', -1)}"
      )
    except Exception:  # pylint: disable=broad-except
      pass
    if step_complete:
      logging.info(
          "Sub-batch restore: no snapshot for train_steps=%d; the latest"
          " surviving snapshot (train_step %d, local %d: %s) marks a cleanly"
          " completed step. Starting fresh; leftovers above this train_step"
          " are purged, older ones age out via retention.",
          train_steps,
          latest_parent,
          latest_local,
          meta_desc,
      )
    else:
      logging.warning(
          "Sub-batch restore: no usable snapshot for the restored"
          " train_steps=%d; %d snapshot(s) survive elsewhere, latest at"
          " train_step %d, local %d (%s). If the previous step completed"
          " at its final apply this is benign. Training skips the remainder of"
          " the last global step (train_step %d) and starts the next step (%d)"
          " fresh from the restored weights (possibly re-doing its rollouts)."
          " Leftovers above this train_step are purged; older ones age out via"
          " retention.",
          train_steps,
          len(leftover),
          latest_parent,
          latest_local,
          meta_desc,
          latest_parent,
          train_steps,
      )

  def _select_step(self, train_steps: int) -> int | None:
    """Picks the snapshot key to restore for restored weights at `train_steps`.

    Valid keys share the parent train_step: `key // KEY_BASE == train_steps`.
    Weights are frozen between applies, so every such snapshot's accumulator
    was built on exactly the restored weights; the latest (max local) loses
    the least work. Keys with a different parent describe different weights
    and are never selected .

    Args:
      train_steps: The trainer's restored apply count (its weight checkpoint's
        step name).

    Returns:
      The snapshot key to restore, or None if no valid snapshot exists.
    """
    assert self._checkpointer is not None
    lo = train_steps * KEY_BASE
    hi = (train_steps + 1) * KEY_BASE
    candidates = [
        ck.step for ck in self._checkpointer.checkpoints if lo <= ck.step < hi
    ]
    return max(candidates) if candidates else None

  def try_restore(
      self,
      train_steps: int,
      grad_accum_steps: int,
      target_training_state: Any = None,
      num_generations: int | None = None,
  ) -> SubBatchState | None:
    """Restores the ledger and grad buffer matching the trainer's restored weights.

    The `state` checkpointable is loaded only when `meta` records that the
    snapshot carries a buffer, and only against a caller-shaped abstract tree.

    Args:
      train_steps: The trainer's restored apply count (its weight checkpoint's
        step name).
      grad_accum_steps: The run's `gradient_accumulation_steps`. Declines the
        restore if it does not match the value the snapshot was saved with.
      target_training_state: Abstract tree for restoring the buffer with the
        correct shapes/shardings. Required when the snapshot carries a buffer.

    Returns:
      The restored `SubBatchState`, or None when no valid snapshot exists for
      this window (fresh start).
    """
    if self._checkpointer is None:
      return None

    self._checkpointer.wait()

    window_start = train_steps * KEY_BASE
    chosen_step = self._select_step(train_steps)
    if chosen_step is None:
      # No usable snapshot: report the surviving evidence FIRST (the purge
      # deletes it).
      self._report_no_usable_snapshot(train_steps)
      self.purge_steps_above(window_start)
      return None

    try:
      meta = self._checkpointer.load_checkpointables(
          chosen_step, abstract_checkpointables={"meta": None}
      )["meta"]
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Sub-batch snapshot at key %d has unreadable meta (torn directory"
          " from an interrupted delete, or an incompatible schema). Declining"
          " restore and purging it so the next restart does not repeat this.",
          chosen_step,
      )
      self.purge_steps_above(chosen_step - 1)
      return None

    if meta.get("grad_accum_steps") != grad_accum_steps:
      logging.warning(
          "Sub-batch snapshot at key %d was saved with grad_accum_steps=%s"
          " but the run is configured with %d; the ledger's window"
          " composition is invalid across this config change. Declining"
          " restore and purging the stale lineage.",
          chosen_step,
          meta.get("grad_accum_steps"),
          grad_accum_steps,
      )
      # Every existing key was placed under the old k; none is valid at or
      # above this run's resume point.
      self.purge_steps_above(window_start - 1)
      return None

    if (
        num_generations is not None
        and meta.get("num_generations") is not None
        and meta["num_generations"] != num_generations
    ):
      logging.warning(
          "Sub-batch snapshot at key %d was saved with num_generations=%s"
          " but the run is configured with %d; the ledger's per-pair"
          " accounting is invalid across this config change. Declining"
          " restore and purging the stale lineage.",
          chosen_step,
          meta.get("num_generations"),
          num_generations,
      )
      self.purge_steps_above(window_start - 1)
      return None

    has_training_state = bool(meta.get("has_training_state", False))
    if has_training_state and target_training_state is None:
      logging.warning(
          "Sub-batch snapshot at key %d carries a grad-accum buffer but no"
          " abstract target was provided to restore it against. Declining"
          " restore and purging the unusable key.",
          chosen_step,
      )
      self.purge_steps_above(chosen_step - 1)
      return None

    abstract_checkpointables: dict[str, Any] = {"rollout": None}
    if has_training_state:
      abstract_checkpointables["state"] = {
          "training_state": target_training_state
      }

    try:
      restored = self._checkpointer.load_checkpointables(
          chosen_step,
          abstract_checkpointables=abstract_checkpointables,
      )
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Sub-batch snapshot at key %d is structurally incompatible with"
          " the current run; declining restore and purging the unusable key.",
          chosen_step,
      )
      self.purge_steps_above(chosen_step - 1)
      return None

    rollout = restored["rollout"]

    # The resume point invalidates everything the crashed run wrote beyond it:
    # its weights past this point never became durable and the rerun diverges.
    self.purge_steps_above(chosen_step)

    # BENCHMARK instrumentation (temporary): the scalars saved in
    # custom_metadata; absent on pre-instrumentation snapshots.
    meta_info = self._checkpointer.checkpointables_metadata(chosen_step)
    custom = meta_info.custom_metadata if meta_info else None
    sub_step_elapsed = 0.0
    experiment_start_time = 0.0
    if isinstance(custom, dict):
      sub_step_elapsed = float(custom.get("sub_step_elapsed", 0.0) or 0.0)
      experiment_start_time = float(
          custom.get("experiment_start_time", 0.0) or 0.0
      )
    return SubBatchState(
        sub_batch_key=chosen_step,
        iter_steps=meta["iter_steps"],
        global_step=meta["global_step"],
        grad_accum_steps=meta["grad_accum_steps"],
        step_complete=bool(meta["step_complete"]),
        completed_group_ids=list(rollout["completed_group_ids"]),
        trained_trajectory_counts={
            (item["group_id"], item["pair_index"]): item["count"]
            for item in rollout["trained_trajectory_counts"]
        },
        active_group_trajectories=[
            _trajectory_item_from_serializable(item)
            for item in rollout["active_group_trajectories"]
        ],
        training_state=(
            restored["state"]["training_state"] if has_training_state else None
        ),
        full_batch_size=meta.get("full_batch_size"),
        sub_step_elapsed=sub_step_elapsed,
        experiment_start_time=experiment_start_time,
    )
