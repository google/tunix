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

"""Sub-batch checkpointing: persists a global step's rollouts mid-step."""

from collections.abc import Callable, Hashable, Sequence
from concurrent import futures
import copy
import dataclasses
import itertools
import pickle
import threading
from typing import Any, ParamSpec, TypeVar

from absl import logging
import jax
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.rl.agentic.agents import agent_types

TrajectoryItem = agent_types.TrajectoryItem
_P = ParamSpec("_P")
_T = TypeVar("_T")

# A global step's snapshots are append-only deltas keyed by
# global_step * KEY_BASE + a save counter.
KEY_BASE = 1_000_000


@dataclasses.dataclass(frozen=True)
class BatchGeometry:
  """The batch sizes a global step is trained with.

  Attributes:
    full_batch_size: Prompts per global step.
    mini_batch_size: Prompts per optimizer update.
    train_micro_batch_size: Prompts per micro-batch. None with sequence packing,
      where packs replace micro-batches.
    gradient_accumulation_steps: Micro-batches per optimizer update. None with
      sequence packing.
    num_generations: Rollouts per prompt.
    num_iterations: Epochs over a global step's rollouts.
  """

  full_batch_size: int
  mini_batch_size: int
  train_micro_batch_size: int | None
  gradient_accumulation_steps: int | None
  num_generations: int
  num_iterations: int

  @property
  def updates_per_step(self) -> int:
    """The optimizer updates in a global step."""
    return self.num_iterations * (self.full_batch_size // self.mini_batch_size)


@dataclasses.dataclass(frozen=True)
class SubBatchState:
  """A global step restored from its sub-batch snapshots.

  Attributes:
    global_step: Global step the rollouts belong to.
    geometry: Batch geometry the snapshots were taken under.
    active_groups: The step's rollout groups, in consumption order.
  """

  global_step: int
  geometry: BatchGeometry
  active_groups: list[list[TrajectoryItem]]


@dataclasses.dataclass(frozen=True)
class _Ledger:
  """The payload of one snapshot: rollout groups recorded since the last."""

  geometry: BatchGeometry
  groups: list[list[TrajectoryItem]]


class _LatestTwoSteps(ocp.training.preservation_policies.PreservationPolicy):
  """Keeps the snapshots of the two latest global steps.

  A restart can still resume the previous step: its last weight checkpoint may
  be in flight when the next step starts.
  """

  def should_preserve(
      self,
      checkpoints: Sequence[ocp.training.CheckpointMetadata],
      *,
      context: ocp.training.preservation_policies.PreservationContext,
  ) -> list[bool]:
    del context
    steps = sorted({ck.step // KEY_BASE for ck in checkpoints})[-2:]
    return [ck.step // KEY_BASE in steps for ck in checkpoints]


def _to_numpy(obj: Any) -> Any:
  """Replaces jax arrays with numpy arrays in containers and dataclasses."""
  if isinstance(obj, jax.Array):
    return np.asarray(obj)
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return dataclasses.replace(
        obj,
        **{
            f.name: _to_numpy(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        },
    )
  if type(obj) is dict:  # pylint: disable=unidiomatic-typecheck
    return {k: _to_numpy(v) for k, v in obj.items()}
  if type(obj) is list:  # pylint: disable=unidiomatic-typecheck
    return [_to_numpy(v) for v in obj]
  if type(obj) is tuple:  # pylint: disable=unidiomatic-typecheck
    return tuple(_to_numpy(v) for v in obj)
  return obj


def _snapshot_item(item: TrajectoryItem) -> TrajectoryItem:
  """Copies `item` with numpy arrays in place of jax arrays."""
  snapshot = copy.copy(item)
  snapshot.traj = _to_numpy(item.traj)
  snapshot.metadata = _to_numpy(item.metadata)
  return snapshot


class SubBatchCheckpointManager:
  """Persists the rollout groups of the in-progress global step.

  Each recorded group is saved in the background, in an append-only snapshot of
  its global step, and a restore concatenates the step's snapshots in order.
  Training progress is not saved: the trainer checkpoints only at global-step
  boundaries, and a restart retrains the step from its start on the restored
  rollouts. One writer thread makes every checkpointer call, so neither
  training nor rollout generation waits on storage; a failed save is raised by
  the next `record_rollout_group`, `wait` or `close`. `record_rollout_group`
  and `should_skip_group` may run on the producer thread, the other methods on
  the training thread. A recorded group is pickled later, on the writer
  thread, so it must not be mutated after it is recorded.
  """

  def __init__(self, root_directory: str, *, enable_async: bool = True):
    self._checkpointer = ocp.training.Checkpointer(
        root_directory, preservation_policy=_LatestTwoSteps()
    )
    self._enable_async = enable_async
    # The checkpointer is not thread safe, so only this thread uses it.
    self._writer = futures.ThreadPoolExecutor(max_workers=1)
    self._lock = threading.Lock()
    # (global step, group) records that are not saved yet, and pending saves.
    self._records: list[tuple[int, list[TrajectoryItem]]] = []
    self._saves: list[futures.Future[None]] = []
    self._restored_group_ids: frozenset[Hashable] = frozenset()
    self._geometry: BatchGeometry | None = None
    self._restored_geometry: BatchGeometry | None = None
    # The next snapshot key; only the writer thread uses it.
    self._key = 0

  def _run(
      self, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
  ) -> _T:
    """Runs `fn` on the writer thread and returns its result."""
    return self._writer.submit(fn, *args, **kwargs).result()

  def has_snapshots(self) -> bool:
    """Whether the root holds a snapshot of any global step."""
    return self._run(lambda: bool(self._checkpointer.checkpoints))

  def record_rollout_group(
      self, global_step: int, group: list[TrajectoryItem]
  ) -> None:
    """Saves a generated group of `global_step`, in its training order."""
    with self._lock:
      # A save can finish at any time, so check each one only once.
      done = [save for save in self._saves if save.done()]
      self._saves = [save for save in self._saves if save not in done]
      self._records.append((global_step, group))
      self._saves.append(self._writer.submit(self._save_records))
    for save in done:
      save.result()  # Raises a failed save.

  def should_skip_group(self, group_id: Hashable) -> bool:
    """Whether the group was restored, so it must not be regenerated."""
    return group_id in self._restored_group_ids

  def set_geometry(self, geometry: BatchGeometry) -> None:
    """Records the batch geometry; it must match the restored snapshot's."""
    if (
        self._restored_geometry is not None
        and geometry != self._restored_geometry
    ):
      raise ValueError(
          f"Sub-batch snapshot geometry {self._restored_geometry} does not"
          f" match this run's {geometry}. Relaunch with the original"
          " configuration or a new checkpoint root."
      )
    self._geometry = geometry

  def wait(self) -> None:
    """Blocks until every recorded group is durable."""
    with self._lock:
      saves, self._saves = self._saves, []
    for save in saves:
      save.result()
    self._run(self._checkpointer.wait)

  def _save_records(self) -> None:
    """Saves the groups recorded so far, one snapshot per global step."""
    assert self._geometry is not None, "set_geometry must be called first."
    with self._lock:
      records, self._records = self._records, []
    for global_step, step_records in itertools.groupby(
        records, key=lambda record: record[0]
    ):
      if self._key // KEY_BASE != global_step:
        self._key = global_step * KEY_BASE
      ledger = _Ledger(
          geometry=self._geometry,
          groups=[
              [_snapshot_item(item) for item in group]
              for _, group in step_records
          ],
      )
      payload = pickle.dumps(ledger, protocol=pickle.HIGHEST_PROTOCOL)
      checkpointables = {
          "ledger": {"payload": np.frombuffer(payload, dtype=np.uint8)}
      }
      # Deltas must commit in order, and Orbax swaps in a new checkpointer
      # before it waits on the previous save.
      self._checkpointer.wait()
      if self._enable_async:
        self._checkpointer.save_checkpointables_async(
            self._key, checkpointables, force=True
        )
      else:
        self._checkpointer.save_checkpointables(
            self._key, checkpointables, force=True
        )
      self._key += 1

  def restore(self, global_step: int) -> SubBatchState | None:
    """Loads the snapshots of `global_step`, the step the trainer restarts at.

    Snapshots of later steps followed weights that never became durable and
    are deleted.

    Args:
      global_step: The global step of the restored trainer checkpoint.

    Returns:
      The restored state, or None if the step has no snapshot.
    """
    return self._run(self._restore, global_step)

  def _restore(self, global_step: int) -> SubBatchState | None:
    keys = sorted(ck.step for ck in self._checkpointer.checkpoints)
    for key in keys:
      if key // KEY_BASE > global_step:
        self._checkpointer.delete(key)
    deltas = [key for key in keys if key // KEY_BASE == global_step]
    if not deltas:
      return None
    ledgers = [self._load(key) for key in deltas]
    state = SubBatchState(
        global_step=global_step,
        geometry=ledgers[0].geometry,
        active_groups=[group for ledger in ledgers for group in ledger.groups],
    )
    self._restored_group_ids = frozenset(
        group[0].prompt_id for group in state.active_groups
    )
    self._restored_geometry = state.geometry
    self._key = deltas[-1] + 1
    logging.info(
        "Sub-batch restore: global step %d with %d groups from %d snapshots.",
        global_step,
        len(state.active_groups),
        len(deltas),
    )
    return state

  def _load(self, key: int) -> _Ledger:
    payload = self._checkpointer.load_checkpointables(key)["ledger"]["payload"]
    return pickle.loads(  # pylint: disable=g-unsafe-pickle-load
        np.asarray(payload, dtype=np.uint8).tobytes()
    )

  def close(self) -> None:
    """Saves the recorded groups, then releases the checkpointer."""
    try:
      self.wait()
    finally:
      closed = self._writer.submit(self._checkpointer.close)
      self._writer.shutdown()
      closed.result()
