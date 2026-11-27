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

"""ICI implementation of the weight synchronization transport.

`IciHandler` is the orchestrator-facing facade for `ici_transport`. Raiden's
controller tells source workers to push bytes to destination workers; here
there is no controller and no bytes on a wire. The weights move inside JAX
collectives, which only run when EVERY participating process enters them, so
`transfer` is a fan-out: it turns the registered work units into one
`TransferPlan` and asks every participant to execute it at the same time.

Requirements on the deployment (checked on the workers before anything
moves):

* trainer and rollout workers are separate processes of ONE multi-process JAX
  world (one `jax.distributed` runtime over one TPU slice);
* each of them owns the same number of local devices;
* each rollout replica is a single process.

No weight bytes pass through the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent import futures
import threading
import time
from typing import Any, override

from absl import logging
from tunix.dist_rl.weight_sync import ici_transport
from tunix.dist_rl.weight_sync import weight_sync

TRANSPORT_MODE = "ici"

# `handle.submit(method, *args)`, the blocking call of a worker `ActorHandle`.
WorkerCall = Callable[[Any, str, Any], Any]


def _submit(handle: Any, method: str, argument: Any) -> Any:
  return handle.submit(method, argument)


def leaf_specs(
    metadata: Sequence[weight_sync.WorkUnitMetadata],
) -> tuple[ici_transport.LeafSpec, ...]:
  """Returns the tensors published by `metadata`, sorted by leaf name.

  Sorting guarantees source and destination (and every subsequent round) agree
  on leaf order without exchanging it.

  Args:
    metadata: Registered source or destination work-unit metadata entries.

  Returns:
    Sorted tuple of unique `LeafSpec` entries.

  Raises:
    ValueError: If any variable omits `dtype` or is published twice with
      conflicting shapes or dtypes.
  """
  leaves: dict[str, ici_transport.LeafSpec] = {}
  for unit in metadata:
    for variable in unit.variables:
      if not variable.dtype:
        raise ValueError(
            f"ICI work unit {unit.unit}: variable {variable.name!r} does not"
            " report a dtype"
        )
      leaf = ici_transport.LeafSpec(
          variable.name, tuple(variable.shape), variable.dtype
      )
      if leaves.setdefault(variable.name, leaf) != leaf:
        raise ValueError(
            f"variable {variable.name!r} is published twice with different"
            f" layouts: {leaves[variable.name]} and {leaf}"
        )
  return tuple(leaves[name] for name in sorted(leaves))


def _validate_destination_leaves(
    leaves: tuple[ici_transport.LeafSpec, ...],
    destination: weight_sync.WorkUnitMetadata,
) -> None:
  """Raises `ValueError` unless `destination` matches `leaves` exactly."""
  theirs = leaf_specs([destination])
  if theirs == leaves:
    return
  ours_by_name = {leaf.name: leaf for leaf in leaves}
  their_names = {leaf.name for leaf in theirs}
  differing = [
      f"{leaf.name}: source {ours_by_name.get(leaf.name)}, destination {leaf}"
      for leaf in theirs
      if ours_by_name.get(leaf.name) != leaf
  ] + [
      f"{leaf.name}: missing on destination"
      for leaf in leaves
      if leaf.name not in their_names
  ]
  raise ValueError(
      f"destination {destination.unit} does not match the source's"
      f" leaves ({len(differing)} differ), e.g. {differing[:5]}"
  )


class IciHandler(weight_sync.WeightSyncHandler):
  """Moves weights trainer -> rollouts over the TPU interconnect."""

  def __init__(
      self,
      participants: Callable[[], Sequence[Any]],
      *,
      chunk_bytes: int = ici_transport.DEFAULT_CHUNK_BYTES,
      check_timeout_s: float = 120.0,
      call: WorkerCall = _submit,
  ):
    """Initializes the handler.

    Args:
      participants: Returns the `ActorHandle` of every worker that may take
        part in a transfer (all trainer and rollout workers). Called on every
        transfer, so restarted workers are picked up.
      chunk_bytes: Upper bound on the global bytes one collective moves. It
        bounds scratch device memory on every participant.
      check_timeout_s: Shared deadline across all workers for each pre-transfer
        validation phase. The transfer itself is bounded by the coordinator.
      call: Blocking worker invocation; overridable for tests.

    Raises:
      ValueError: If `chunk_bytes` is not positive.
    """
    if chunk_bytes <= 0:
      raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    self._participants = participants
    self._chunk_bytes = int(chunk_bytes)
    self._check_timeout_s = float(check_timeout_s)
    self._call = call
    self._lock = threading.Lock()
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}

  @property
  @override
  def source_joins_transfer(self) -> bool:
    # The trainer enters the `ppermute` collectives during `transfer`, so it
    # must not start its next step on the same devices until they finish.
    return True

  @property
  @override
  def shares_world(self) -> bool:
    return True

  @override
  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    if metadata.transport_mode != TRANSPORT_MODE:
      raise ValueError(
          f"work unit {metadata.unit} is bound on transport"
          f" {metadata.transport_mode!r}; IciHandler needs {TRANSPORT_MODE!r}."
          " Trainer and rollout must both run weight_sync_config.mode: ici"
      )
    if not metadata.process_indices or not metadata.devices_per_process:
      raise ValueError(
          f"ICI work unit {metadata.unit} must report its JAX process_indices"
          " and devices_per_process"
      )
    if not metadata.variables:
      raise ValueError(f"ICI work unit {metadata.unit} publishes no variables")
    with self._lock:
      self._units[metadata.unit] = metadata

  def _lookup(
      self, units: Sequence[weight_sync.WorkUnitId]
  ) -> list[weight_sync.WorkUnitMetadata]:
    with self._lock:
      missing = [u for u in units if u not in self._units]
      if missing:
        raise KeyError(f"work units were never registered: {missing}")
      return [self._units[u] for u in units]

  def _validate_units(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
  ) -> tuple[
      tuple[ici_transport.LeafSpec, ...],
      list[int],
      list[int],
      int,
  ]:
    """Validates registered units and extracts planning inputs."""
    sources = self._lookup(src_units)
    destinations = self._lookup(dst_units)
    leaves = leaf_specs(sources)
    for destination in destinations:
      _validate_destination_leaves(leaves, destination)
      if len(destination.process_indices) != 1:
        raise ValueError(
            f"destination {destination.unit} spans processes"
            f" {destination.process_indices}; an ICI destination must be one"
            " JAX process"
        )
    all_units = (*sources, *destinations)
    widths = {m.devices_per_process for m in all_units}
    if len(widths) != 1:
      widths_by_unit = {str(m.unit): m.devices_per_process for m in all_units}
      raise ValueError(
          "ICI weight sync needs every trainer and rollout process to own"
          f" the same number of devices; got {widths_by_unit}"
      )
    width = widths.pop()
    assert width is not None  # register_work_unit rejects a missing width.
    source_proc_set: set[int] = set()
    for meta in sources:
      for proc in meta.process_indices:
        source_proc_set.add(int(proc))
    source_processes = sorted(source_proc_set)
    destination_processes = [int(m.process_indices[0]) for m in destinations]
    return leaves, source_processes, destination_processes, width

  def _call_all(
      self, handles: Sequence[Any], method: str, request: Any
  ) -> list[futures.Future[Any]]:
    # Daemon threads, not an executor: after a failure a call may stay blocked
    # in a collective forever, and executor threads are joined at exit.
    return [
        weight_sync.call_in_daemon_thread(self._call, handle, method, request)
        for handle in handles
    ]

  def _check_all(self, handles: Sequence[Any], request: Any) -> list[Any]:
    """Runs `ici_check` on every worker under one shared deadline."""
    deadline = time.monotonic() + self._check_timeout_s
    calls = self._call_all(handles, "ici_check", request)
    pending = set(calls)
    while pending:
      remaining = deadline - time.monotonic()
      done, pending = (
          futures.wait(
              pending, timeout=remaining, return_when=futures.FIRST_EXCEPTION
          )
          if remaining > 0
          else (set(), pending)
      )
      if not done:
        for fut in pending:
          fut.cancel()
        # Re-raised as RuntimeError because the coordinator reads a bare
        # TimeoutError as "the transfer may still be running".
        raise RuntimeError(
            f"ICI pre-transfer check timed out after {self._check_timeout_s}s"
        )
      failed = next(
          (f.exception() for f in done if f.exception() is not None), None
      )
      if failed is not None:
        for fut in pending:
          fut.cancel()
        raise failed
    return [call.result() for call in calls]

  def _discover_world(self, handles: Sequence[Any]) -> list[int]:
    """Queries every live worker for its JAX process index and world view."""
    reports = self._check_all(handles, {"plan": None})
    worlds = {tuple(r["world"]) for r in reports}
    processes = sorted(int(r["process"]) for r in reports)
    if len(worlds) != 1 or processes != list(next(iter(worlds))):
      raise RuntimeError(
          "ICI weight sync needs exactly one live worker per process of one"
          f" shared JAX world; workers report processes {processes} of"
          f" world(s) {sorted(worlds)}. Trainer and rollout workers must"
          " join the same jax.distributed runtime"
      )
    return processes

  def _run_collectives(
      self,
      handles: Sequence[Any],
      request: dict[str, Any],
      req_id: str,
  ) -> list[Any]:
    """Enters the ICI collectives on every worker concurrently."""
    runs = self._call_all(handles, "ici_transfer", request)
    done, _ = futures.wait(runs, return_when=futures.FIRST_EXCEPTION)
    failed = next((f for f in done if f.exception() is not None), None)
    if failed is not None:
      # Survivors may be blocked in a collective the failed worker never
      # entered; whether they wrote anything is unknowable from here.
      raise weight_sync.TransferOutcomeUnknownError(
          f"ICI transfer {req_id} failed on a participant; its peers may"
          f" still be inside the collective: {failed.exception()!r}"
      ) from failed.exception()
    return [f.result() for f in runs]

  @override
  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: str | None = None,
      generation: int | None = None,
  ) -> weight_sync.TransferResult:
    req_id = req_id or ""
    # Fails on mismatched units before any worker is contacted.
    leaves, source_processes, dst_processes, width = self._validate_units(
        src_units, dst_units
    )
    handles = list(self._participants())
    if not handles:
      raise RuntimeError("IciHandler has no workers to run the transfer on")
    started = time.monotonic()
    processes = self._discover_world(handles)
    plan = ici_transport.build_plan(
        leaves,
        source_processes=source_processes,
        destination_processes=dst_processes,
        width=width,
        world_processes=processes,
        chunk_bytes=self._chunk_bytes,
    )
    request = {"plan": plan, "req_id": req_id, "generation": generation}
    # Phase 1: every worker validates its side of the final plan before any
    # process enters a collective.
    self._check_all(handles, request)
    # Phase 2: everyone enters the collectives together.
    timings = self._run_collectives(handles, request, req_id)
    elapsed = time.monotonic() - started
    moved = sum(int(t["bytes"]) for t in timings)
    message = (
        f"{len(plan.leaves)} leaves, {len(plan.chunks)} chunk(s) x"
        f" {len(plan.rounds)} round(s), {moved / 2**30:.2f} GiB to"
        f" {len(plan.destination_processes)} replica(s) in {elapsed:.2f}s"
    )
    logging.info("ICI transfer %s: %s", req_id, message)
    return weight_sync.TransferResult(
        req_id=req_id, success=True, message=message
    )
