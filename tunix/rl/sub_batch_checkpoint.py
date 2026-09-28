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
from etils import epath
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

# The enable marker lives BESIDE the ledger directory, so losing the entire
# ledger cannot silently turn a restart into a first enable. It records the
# durable trainer step at enrollment: before weights advance, recovery may
# replay from this baseline even if no sub-batch snapshot committed yet.


@dataclasses.dataclass
class SubBatchPreservationPolicy(
    ocp.training.preservation_policies.PreservationPolicy
):
  """Preserves the newest key of every window at or above the floor.

  The floor picks which windows survive: the trainer's last DURABLE apply
  count, as reported by the learner through
  `SubBatchCheckpointManager.set_durable_train_steps` right after it waited
  on the trainer's checkpoint. Nothing reported => no window is dropped (a
  standalone manager never guesses at the trainer's state). In steady state
  the learner reports T just before precommitting (T+1).0, so windows
  {T, T+1} survive.

  Within a surviving window only the newest key is kept: restore selects
  it (`_select_step`), so every older key is dead weight. Each mid-window
  save retires the previous key in orbax's finalize thread instead of the
  whole window being deleted at the next floor move. This is crash-safe
  because orbax deletes only after the new key commits: a crash mid-write
  leaves the previous key, a crash after the commit leaves the new one.

  Orbax evaluates should_preserve synchronously inside save() against this
  live object and deletes in the finalize thread, so the floor must be
  raised BEFORE the save that should observe it.

  durable_train_steps: the floor; None until the owner reports one.
  """
  durable_train_steps: int | None = None

  def should_preserve(self, checkpoints, *, context):
    del context
    by_parent: dict[int, list[int]] = {}
    for ck in checkpoints:
      by_parent.setdefault(ck.step // KEY_BASE, []).append(ck.step % KEY_BASE)
    newest = {p: max(ls) for p, ls in by_parent.items()}
    floor = self.durable_train_steps
    return [
        (floor is None or ck.step // KEY_BASE >= floor)
        and ck.step % KEY_BASE == newest[ck.step // KEY_BASE]
        for ck in checkpoints
    ]


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


def _validate_prompt_id(prompt_id: Any) -> Any:
  """Requires prompt ids that survive the checkpoint round-trip verbatim.

  `prompt_id` is the group's key (the orchestrator's group_key_fn result).

  Args:
    prompt_id: The prompt ID to validate.

  Returns:
    The validated prompt ID.

  Raises:
    ValueError: If the prompt ID is not a string or integer, or is an
      integer outside of the int64 range.
  """
  if isinstance(prompt_id, np.integer):
    prompt_id = int(prompt_id)
  if isinstance(prompt_id, int) and not isinstance(prompt_id, bool):
    if not _INT64_MIN <= prompt_id <= _INT64_MAX:
      raise ValueError(
          "Sub-batch checkpointing requires prompt ids within int64 range so"
          f" ledger keys round-trip verbatim; got {prompt_id!r}. Adjust the"
          " orchestrator's group_key_fn."
      )
    return prompt_id
  if isinstance(prompt_id, str):
    return prompt_id
  raise ValueError(
      "Sub-batch checkpointing requires str or int prompt ids so ledger keys"
      f" round-trip verbatim; got {type(prompt_id).__name__}: {prompt_id!r}."
      " Adjust the orchestrator's group_key_fn."
  )


def _trajectory_item_to_serializable(item: TrajectoryItem) -> dict[str, Any]:
  """Serializes an item as native identity + one pickled payload.

  Identity (prompt_id / group_index / start_step) stays native: restore logic
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
      "prompt_id": _validate_prompt_id(item.prompt_id),
      "group_index": int(item.group_index),
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
      prompt_id=data["prompt_id"],
      group_index=int(data["group_index"]),
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
      Restore raises SubBatchGeometryError on mismatch.
    step_complete: True when this snapshot was taken at the final apply of its
      global step. On resume the learner then starts fresh at `global_step + 1`
      instead of resuming mid-step.
    completed_group_ids: prompt_ids of the groups whose rollouts are fully
      generated (skip regeneration on resume).
    trained_trajectory_counts: Map tracking (prompt_id, group_index) ->
      trained epoch count.
    active_group_trajectories: Rollouts not yet fully consumed by the trainer,
      re-injected on resume.
    training_state: The live gradient-accumulation state, opaque to the manager.
      Both training modes persist the trainer's `GradientAccumulator` contents
      (`{"grads": <param tree>, "denom": scalar}`); None when the snapshot was
      taken at an apply boundary (the accumulator was just reset -- `denom == 0`
      -- and a restored trainer starts its window with a fresh accumulator
      anyway, so there is nothing to inject).
    geometry: Resolved run geometry recorded at save (see save()); the
      learner checks the train-time-only keys (train_micro_batch_size,
      pack_size) in _sb_validate_batch_geometry once the dataset and mesh
      reveal them.
    anchor_step: The trainer's train_steps at the start of the global step
      this snapshot belongs to; its weight checkpoint is the step's
      behavior/anchor policy, which a disaggregated mid-step resume restores
      (see AgenticRLLearner._sb_resync_rollout_weights).
  """
  sub_batch_key: int
  iter_steps: int
  global_step: int
  grad_accum_steps: int
  step_complete: bool
  completed_group_ids: list[Hashable]
  trained_trajectory_counts: dict[tuple[Hashable, int], int]
  active_group_trajectories: list[TrajectoryItem]
  anchor_step: int
  training_state: Any | None = None
  full_batch_size: int | None = None
  # BENCHMARK instrumentation (temporary, commit 4): wall-clock spent inside
  # the interrupted step at snapshot time, and the run's fixed experiment
  # anchor -- both ride custom_metadata, not the checkpointables schema.
  sub_step_elapsed: float = 0.0
  experiment_start_time: float = 0.0
  geometry: dict[str, Any] = dataclasses.field(default_factory=dict)


class SubBatchRestoreError(RuntimeError):
  """try_restore could not produce a matched (weights, ledger) pair.

  Nothing is purged on any raise: the evidence stays for a corrected
  relaunch, or for the operator to move or delete `<root>/sub_batch`
  deliberately (the actor weights under `<root>/actor` stay usable).
  """


class SubBatchLedgerMissingError(SubBatchRestoreError):
  """Trainer checkpoint T > 0 has no ledger with parent T while other
  ledgers exist (retention/lineage bug or a foreign ledger root)."""


class SubBatchUnreadableError(SubBatchRestoreError):
  """The selected snapshot did not load or did not validate: torn
  directory, I/O failure, incompatible tree, or a payload/meta that does
  not decode (the kind is in the message)."""


class SubBatchPurgeError(SubBatchRestoreError):
  """A dead-lineage snapshot could not be deleted. Recovery stops: left on
  disk, its key would later be selected over the rerun's own snapshots."""


class SubBatchGeometryError(ValueError):
  """A restart changed the run's batch geometry.

  The sub-batch resume contract assumes the relaunched process uses the
  same configuration as the crashed one (batch sizes, the accumulation
  window, num_generations, num_iterations, the recorded run geometry). A
  mismatch raises instead of silently adapting, nothing is purged. To
  change geometry deliberately, start a new run root. Removing only the
  ledger does not clear its enrollment record beside the directory.
  """


def _causes(exc: BaseException) -> list[BaseException]:
  """The exception chain (`__cause__`/`__context__`), cycle-safe."""
  seen: list[BaseException] = []
  e: BaseException | None = exc
  while e is not None and all(e is not s for s in seen):
    seen.append(e)
    e = e.__cause__ or e.__context__
  return seen


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
    self._root_directory = root_directory
    self._floor_unset_logged = False

  @property
  def enable_marker_path(self) -> epath.Path | None:
    """The enrollment record, outside the directory it protects."""
    if not self._root_directory:
      return None
    root = epath.Path(self._root_directory)
    return root.parent / f".{root.name}.enabled"

  def mark_enabled(self, train_steps: int = 0) -> None:
    """Records the initial durable trainer step before training starts.

    The record is immutable during a run. An empty or unreadable record
    fails recovery rather than allowing an ambiguous first enable.
    """
    path = self.enable_marker_path
    if path is None:
      return
    if path.exists():
      if self._enabled_at() != train_steps:
        raise SubBatchLedgerMissingError(
            f"Sub-batch enrollment at {path} belongs to another baseline;"
            " restore the matching trainer and ledger, or use a new run root."
        )
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(train_steps))

  def _enabled_at(self) -> int | None:
    path = self.enable_marker_path
    if path is None or not path.exists():
      return None
    try:
      step = int(path.read_text())
      if step < 0:
        raise ValueError("negative baseline")
      return step
    except (OSError, ValueError) as e:
      raise SubBatchUnreadableError(
          f"Cannot read sub-batch enrollment {path}: {e}. Nothing was purged."
      ) from e

  @property
  def enabled(self) -> bool:
    return self._checkpointer is not None

  def set_durable_train_steps(self, train_steps: int) -> None:
    """Records the trainer's last durable apply count; retention keeps the
    newest key of every parent >= it. Call only after the trainer
    checkpoint for `train_steps` is known durable (the learner does so after
    `checkpoint_manager.wait()`) and before the save that should observe it.
    Never lowers the floor. A caller-supplied preservation policy owns
    retention: no-op.
    """
    policy = self._options.preservation_policy
    if isinstance(policy, SubBatchPreservationPolicy):
      policy.durable_train_steps = max(
          policy.durable_train_steps or 0, train_steps
      )

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
      num_iterations: int | None = None,
      mini_batch_size: int | None = None,
      sub_step_elapsed: float = 0.0,
      experiment_start_time: float = 0.0,
      geometry: dict[str, Any] | None = None,
      anchor_step: int | None = None,
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
    geometry: the ledger's per-pair accounting and the prompt-id step
    arithmetic (prompt_id // full_batch_size) are only valid when the resumed
    run uses the same values. num_generations is checked inside try_restore;
    full_batch_size is returned in `SubBatchState` for the learner to check
    once the dataset reveals the current value (it is not known at restore
    time).

    `geometry` is the caller's resolved run geometry beyond the ledger
    arithmetic (packing budget/segment cap, pack_size, micro-batch size,
    objective settings, optional dataset id). Keys are opaque to the
    manager and compared verbatim at restore: a key ABSENT on either side
    is skipped, and a key PRESENT with value None is a value
    (sampler_is=None).

    `training_state` is the live accumulator state for MID-WINDOW snapshots
    and None at apply boundaries.

    `anchor_step` is the trainer's train_steps at the start of the global
    step the snapshot belongs to (the step's behavior policy checkpoint; the
    learner always passes it). Recorded in meta and
    required at restore. None means the snapshot's own parent, exact only
    for a boundary snapshot of a one-apply-per-step run: a convenience for
    tests that plant snapshots, not for callers.

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
    policy = self._options.preservation_policy
    if (
        isinstance(policy, SubBatchPreservationPolicy)
        and policy.durable_train_steps is None
        and not self._floor_unset_logged
    ):
      self._floor_unset_logged = True
      logging.info(
          "Sub-batch save with no durable floor reported: no window is"
          " dropped (newest key each) until set_durable_train_steps is"
          " called."
      )
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
            "num_iterations": num_iterations,
            "mini_batch_size": mini_batch_size,
            "geometry": dict(geometry or {}),
            "anchor_step": int(
                train_steps if anchor_step is None else anchor_step
            ),
        },
        "rollout": {
            "completed_group_ids": [
                _validate_prompt_id(g) for g in completed_group_ids
            ],
            "active_group_trajectories": [
                _trajectory_item_to_serializable(item)
                for item in active_group_trajectories
            ],
            "trained_trajectory_counts": [
                {
                    "prompt_id": _validate_prompt_id(k[0]),
                    "group_index": int(k[1]),
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
    corruption), and _select_step / retention would prefer them over the
    rerun's own keys. The v1 Checkpointer has no public delete; this goes
    through the same `_manager.delete` its own overwrite path uses, then
    removes whatever the deleter left behind. A key that is still on disk
    afterwards raises SubBatchPurgeError: recovery must not continue.

    Raises:
      SubBatchPurgeError: A dead-lineage snapshot could not be removed.
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
      except OSError as e:
        logging.warning(
            "Sub-batch restore: orbax could not delete dead-lineage key %d"
            " (%s); removing its directory directly.",
            step,
            e,
        )
      name_format = self._options.step_name_format if self._options else None
      step_name = name_format.build_name(step) if name_format else str(step)
      step_dir = epath.Path(self._checkpointer.directory) / step_name
      try:
        if step_dir.exists():
          step_dir.rmtree()
        if step_dir.exists():
          raise OSError("directory remains after deletion")
        # delete() may have failed before updating Orbax's cached step list.
        # Refresh it after fallback cleanup before any selection/save can run.
        # pylint: disable-next=protected-access
        manager = self._checkpointer._manager
        if step in manager.all_steps():
          manager.reload()
        if step in manager.all_steps():
          raise OSError("deleted step remains in Orbax's step list")
      except OSError as e:
        raise SubBatchPurgeError(
            f"Sub-batch restore could not remove dead-lineage snapshot key"
            f" {step} ({step_dir}): {e}. Recovery stops here: left on disk,"
            " that key would be selected over the rerun's own snapshots of"
            " its window. Fix the permissions or delete it by hand, then"
            " relaunch."
        ) from e
    if stale:
      logging.info(
          "Sub-batch restore purged %d dead-lineage snapshot(s) above key %d:"
          " %s",
          len(stale),
          bound,
          stale,
      )

  def _describe_missing(self, train_steps: int, leftover: list[int]) -> str:
    """Evidence for SubBatchLedgerMissingError: window `train_steps` is
    empty while `leftover` keys survive. Decodes the latest survivor and
    reads its meta (or reports it unreadable); nothing is purged."""
    assert self._checkpointer is not None
    if not leftover:
      return (
          f"Sub-batch restore: trainer train_steps={train_steps} has no"
          f" matching ledger under {self._root_directory}; the ledger was"
          " lost. Nothing was purged."
      )
    latest = max(leftover)
    latest_parent, latest_local = divmod(latest, KEY_BASE)
    meta_desc = "unreadable"
    try:
      meta = self._checkpointer.load_checkpointables(
          latest, abstract_checkpointables={"meta": None}
      )["meta"]
      meta_desc = (
          f"global_step={meta.get('global_step', -1)},"
          f" step_complete={bool(meta.get('step_complete', False))},"
          f" grad_accum_steps={meta.get('grad_accum_steps', -1)}"
      )
    except Exception:  # pylint: disable=broad-except
      pass
    return (
        f"Sub-batch restore: trainer checkpoint train_steps={train_steps}"
        f" has no ledger with parent {train_steps}, but {len(leftover)}"
        f" snapshot(s) survive under {self._root_directory} (latest:"
        f" train_step {latest_parent}, local {latest_local}: {meta_desc})."
        f" Ledger {train_steps}.0 is written durably before every apply, so"
        " this is a retention/lineage bug or a foreign ledger root; nothing"
        " was purged. Restore the matching ledger before continuing."
    )

  def _invalid_msg(self, key: int, what: str) -> str:
    """Message for SubBatchUnreadableError when the snapshot loaded but its
    meta/payload does not validate: nothing purged, operator decides."""
    assert self._checkpointer is not None
    parent, local = divmod(key, KEY_BASE)
    return (
        f"Sub-batch snapshot key {key} (window {parent}, local {local}, dir"
        f" {self._checkpointer.directory / str(key)}) does not validate:"
        f" {what}. Nothing was purged. Relaunch with the build that wrote"
        " it; restore the matching ledger before continuing."
    )

  def _unreadable_msg(self, key: int, what: str, exc: BaseException) -> str:
    """Message for SubBatchUnreadableError: which key/directory failed to
    load `what`, the kind of failure read off the exception chain, and the
    operator's options. Kinds (orbax 0.12.1): a missing checkpointable is a
    bare KeyError (torn directory); I/O failures and structural drift both
    surface as NoEntryError (a KeyError subclass) with the real cause
    chained, so the bare type is tested first, then OSError in the chain.
    The label is advisory; the class and the no-purge contract do not
    depend on it."""
    assert self._checkpointer is not None
    if type(exc) is KeyError:  # pylint: disable=unidiomatic-typecheck
      kind = "torn directory (checkpointable missing)"
    elif any(isinstance(c, OSError) for c in _causes(exc)):
      kind = "I/O failure"
    else:
      kind = "structurally incompatible tree"
    parent, local = divmod(key, KEY_BASE)
    msg = (
        f"Sub-batch snapshot key {key} (window {parent}, local {local}, dir"
        f" {self._checkpointer.directory / str(key)}) failed to load {what}:"
        f" {kind}. Nothing was purged."
    )
    if kind == "I/O failure":
      return msg + (
          " Fix the I/O problem and relaunch; this key stays the resume"
          " point."
      )
    # A hand-deleted key falls back to the previous key of the SAME window
    # (same weights, fewer trained chunks; _select_step never crosses
    # windows). A precommit key (local 0) has none: the step's remainder
    # is unrecoverable.
    fallback = max(
        (
            ck.step
            for ck in self._checkpointer.checkpoints
            if parent * KEY_BASE <= ck.step < key
        ),
        default=None,
    )
    if local == 0 or fallback is None:
      return msg + (
          f" There is no earlier key in window {parent} to fall back to, so"
          " the remainder of the in-progress step cannot be recovered:"
          " relaunch with the original trainer configuration if the tree"
          " changed, otherwise restore the matching ledger before continuing."
      )
    return msg + (
        f" Delete that directory by hand to fall back to key {fallback} of"
        " the same window (same weights, fewer trained chunks), or relaunch"
        " with the original trainer configuration if the tree changed."
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
      num_iterations: int | None = None,
      mini_batch_size: int | None = None,
      geometry: dict[str, Any] | None = None,
  ) -> SubBatchState | None:
    """Restores the ledger and grad buffer matching the trainer's restored weights.

    The `state` checkpointable is loaded only when `meta` records that the
    snapshot carries a buffer, and only against a caller-shaped abstract tree.

    Args:
      train_steps: The trainer's restored apply count (its weight checkpoint's
        step name).
      grad_accum_steps: The run's `gradient_accumulation_steps`. Raises
        SubBatchGeometryError if it differs from the snapshot's.
      target_training_state: Abstract tree for restoring the buffer with the
        correct shapes/shardings. Required when the snapshot carries a buffer.
      num_generations: The run's `num_generations`. Raises
        SubBatchGeometryError if it differs from the snapshot's (skipped when
        either side is unknown).
      num_iterations: The run's `num_iterations`. Raises
        SubBatchGeometryError if it differs from the snapshot's (skipped when
        either side is unknown).
      mini_batch_size: The run's `mini_batch_size`. Raises
        SubBatchGeometryError if it differs from the snapshot's (skipped when
        either side is unknown).
      geometry: Resolved run geometry of this launch (see save()). Every key
        present on both sides must be equal, None being a value. Mismatch
        raises SubBatchGeometryError.

    Returns:
      The restored `SubBatchState`, or None ONLY when there is nothing to
      resume and starting at the trainer's stamp is a matched pair: a fresh
      anchor (train_steps == 0 with window 0 empty; keys above 0 are dead
      lineage and purged), or the enrollment baseline before any durable
      apply advances the weights. First enable on existing weights records
      that baseline with a WARNING: the old in-progress step is not resumed.

    Raises:
      SubBatchLedgerMissingError: train_steps > 0, window empty, other
        ledgers present.
      SubBatchUnreadableError: The selected snapshot did not load (torn
        directory, I/O failure, incompatible tree) or did not validate
        (missing required meta, undecodable payload).
      SubBatchPurgeError: A dead-lineage snapshot above the resume point
        could not be removed.
      SubBatchGeometryError: The snapshot geometry (e.g. grad_accum_steps,
        num_generations, a `geometry` key) does not match the current run.
      Nothing is purged on any raise; purging happens only after the
      selected snapshot fully validated (dead lineage above the resume key)
      and on the enrollment baseline. Once trainer weights advance beyond
      that baseline, a missing ledger always raises, including when the
      entire ledger directory is absent. The enrollment record is stored
      beside that directory.
    """
    if self._checkpointer is None:
      return None

    self._checkpointer.wait()

    baseline = self._enabled_at()
    chosen_step = self._select_step(train_steps)
    if chosen_step is None:
      leftover = sorted(ck.step for ck in self._checkpointer.checkpoints)
      if baseline is None and not leftover:
        # No prior enrollment: adopt the restored trainer as a baseline.
        # No update may run until this record exists outside the ledger root.
        self.mark_enabled(train_steps)
        baseline = train_steps
        if train_steps > 0:
          logging.warning(
              "Sub-batch checkpointing enabled on an existing run at"
              " train_steps=%d. Its in-progress step is NOT resumed;"
              " stock restart semantics establish this run's baseline.",
              train_steps,
          )
      # A fresh step-0 anchor is also valid before its first ledger, including
      # a crash after the first apply's precommit but before durable weights.
      initial = baseline if baseline is not None else 0
      if train_steps == initial and all(
          key // KEY_BASE > train_steps for key in leftover
      ):
        if leftover and max(leftover) // KEY_BASE > train_steps + 1:
          logging.warning(
              "Sub-batch retention outran trainer baseline %d; replaying"
              " from the baseline and purging dead lineage.", train_steps,
          )
        self.purge_steps_above(train_steps * KEY_BASE)
        return None
      raise SubBatchLedgerMissingError(
          self._describe_missing(train_steps, leftover)
          + f" Enrollment baseline: {baseline!r}; record:"
          f" {self.enable_marker_path}. Restore the matched recovery pair."
      )

    try:
      meta = self._checkpointer.load_checkpointables(
          chosen_step, abstract_checkpointables={"meta": None}
      )["meta"]
    except Exception as e:  # pylint: disable=broad-except
      raise SubBatchUnreadableError(
          self._unreadable_msg(chosen_step, "meta", e)
      ) from e

    for field in ("has_training_state", "anchor_step"):
      if meta.get(field) is None:
        raise SubBatchUnreadableError(
            self._invalid_msg(chosen_step, f"meta lacks {field}")
        )
    has_training_state = bool(meta["has_training_state"])

    # Geometry checks: a mismatch on any of these RAISES (see
    # SubBatchGeometryError). The explicit fields skip when either side is
    # unknown (None); `geometry` keys compare None as a value.
    def _require_same(
        field: str, saved, current, *, skip_none: bool = True
    ) -> None:
      if (skip_none and (saved is None or current is None)) or saved == current:
        return
      raise SubBatchGeometryError(
          f"Sub-batch snapshot at key {chosen_step} was saved with"
          f" {field}={saved!r} but this run is configured with {current!r}."
          " Sub-batch resume requires relaunching with the SAME"
          " configuration; relaunch with the original value, or explicitly"
          " start a new run root to change configuration."
      )

    _require_same(
        "grad_accum_steps", meta.get("grad_accum_steps"), grad_accum_steps
    )
    _require_same(
        "num_generations", meta.get("num_generations"), num_generations
    )
    _require_same(
        "num_iterations", meta.get("num_iterations"), num_iterations
    )
    _require_same(
        "mini_batch_size", meta.get("mini_batch_size"), mini_batch_size
    )
    saved_geo = dict(meta.get("geometry") or {})
    for field, current in (geometry or {}).items():
      if field not in saved_geo:
        continue
      _require_same(field, saved_geo[field], current, skip_none=False)

    if has_training_state and target_training_state is None:
      # A snapshot only carries a buffer if the trainer had a grad
      # accumulator, and a relaunch of the same trainer stack has one too --
      # so a missing target means the trainer stack itself changed across
      # the restart. The alternative to raising, an unshaped load, corrupts
      # the buffer's tuple path keys.
      raise SubBatchGeometryError(
          f"Sub-batch snapshot at key {chosen_step} carries a grad-accum"
          " buffer but the relaunched trainer offers no accumulator target"
          " to restore it into; the trainer stack differs from the run that"
          " saved it. Relaunch with the original configuration."
      )

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
    except Exception as e:  # pylint: disable=broad-except
      raise SubBatchUnreadableError(
          self._unreadable_msg(chosen_step, "rollout/state", e)
      ) from e

    rollout = restored["rollout"]


    # Decode and validate the whole snapshot BEFORE touching the disk: a
    # readable orbax tree is not yet a valid ledger.
    try:
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
      state = SubBatchState(
          sub_batch_key=chosen_step,
          iter_steps=int(meta["iter_steps"]),
          global_step=int(meta["global_step"]),
          grad_accum_steps=int(meta["grad_accum_steps"]),
          step_complete=bool(meta["step_complete"]),
          completed_group_ids=list(rollout["completed_group_ids"]),
          trained_trajectory_counts={
              (item["prompt_id"], int(item["group_index"])): int(item["count"])
              for item in rollout["trained_trajectory_counts"]
          },
          active_group_trajectories=[
              _trajectory_item_from_serializable(item)
              for item in rollout["active_group_trajectories"]
          ],
          training_state=(
              restored["state"]["training_state"]
              if has_training_state
              else None
          ),
          full_batch_size=meta.get("full_batch_size"),
          sub_step_elapsed=sub_step_elapsed,
          experiment_start_time=experiment_start_time,
          geometry=saved_geo,
          anchor_step=int(meta["anchor_step"]),
      )
    except Exception as e:  # pylint: disable=broad-except
      raise SubBatchUnreadableError(
          self._invalid_msg(chosen_step, f"{type(e).__name__}: {e}")
      ) from e

    # The resume point invalidates everything the crashed run wrote beyond it:
    # its weights past this point never became durable and the rerun diverges.
    self.purge_steps_above(chosen_step)
    return state
