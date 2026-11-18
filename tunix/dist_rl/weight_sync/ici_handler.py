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
from typing import Any, Optional

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
  """The tensors a side publishes, sorted by name.

  Sorting is what makes source and destination (and every later round) agree
  on leaf order without exchanging it.
  """
  leaves = {}
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
      check_timeout_s: Deadline for the pre-transfer validation on each
        worker. The transfer itself is bounded by the coordinator.
      call: Blocking worker invocation; overridable for tests.
    """
    if chunk_bytes <= 0:
      raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    self._participants = participants
    self._chunk_bytes = int(chunk_bytes)
    self._check_timeout_s = float(check_timeout_s)
    self._call = call
    self._lock = threading.Lock()
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}

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

  def build_plan(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      world_processes: Sequence[int] | None = None,
  ) -> ici_transport.TransferPlan:
    """The plan for one transfer between registered units."""
    sources = self._lookup(src_units)
    destinations = self._lookup(dst_units)
    leaves = leaf_specs(sources)
    for destination in destinations:
      theirs = leaf_specs([destination])
      if theirs != leaves:
        ours_by_name = {l.name: l for l in leaves}
        differing = [
            f"{l.name}: source {ours_by_name.get(l.name)}, destination {l}"
            for l in theirs
            if ours_by_name.get(l.name) != l
        ] + [
            f"{l.name}: missing on destination"
            for l in leaves
            if l.name not in {t.name for t in theirs}
        ]
        raise ValueError(
            f"destination {destination.unit} does not match the source's"
            f" leaves ({len(differing)} differ), e.g. {differing[:5]}"
        )
      if len(destination.process_indices) != 1:
        raise ValueError(
            f"destination {destination.unit} spans processes"
            f" {destination.process_indices}; an ICI destination must be one"
            " JAX process"
        )
    widths = {m.devices_per_process for m in (*sources, *destinations)}
    if len(widths) != 1:
      raise ValueError(
          "ICI weight sync needs every trainer and rollout process to own"
          " the same number of devices; got"
          f" { {str(m.unit): m.devices_per_process for m in (*sources, *destinations)} }"
      )
    source_processes = sorted({p for m in sources for p in m.process_indices})
    return ici_transport.build_plan(
        leaves,
        source_processes=source_processes,
        destination_processes=[m.process_indices[0] for m in destinations],
        width=widths.pop(),
        world_processes=world_processes,
        chunk_bytes=self._chunk_bytes,
    )

  def _call_all(
      self,
      pool: futures.ThreadPoolExecutor,
      handles: Sequence[Any],
      request: Any,
  ) -> list[Any]:
    """`ici_check` on every worker; nothing has moved, so failures are plain."""
    calls = [
        pool.submit(self._call, handle, "ici_check", request)
        for handle in handles
    ]
    try:
      return [c.result(timeout=self._check_timeout_s) for c in calls]
    except futures.TimeoutError as e:
      # Re-raised as a RuntimeError because the coordinator reads a bare
      # TimeoutError as "the transfer may still be running".
      raise RuntimeError(
          f"ICI pre-transfer check timed out after {self._check_timeout_s}s"
      ) from e

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
  ) -> weight_sync.TransferResult:
    req_id = req_id or ""
    # Fails on mismatched units before any worker is contacted.
    self.build_plan(src_units, dst_units)
    handles = list(self._participants())
    if not handles:
      raise RuntimeError("IciHandler has no workers to run the transfer on")
    started = time.monotonic()
    # Not a `with` block: leaving one joins the pool's threads, and after a
    # failed transfer those may be blocked in a worker call that never returns.
    pool = futures.ThreadPoolExecutor(
        max_workers=len(handles), thread_name_prefix="ici_transfer"
    )
    try:
      # Phase 1, no collectives. First every worker says which process of the
      # JAX world it is: a collective needs ALL of them, so a world process
      # without a live worker means the transfer could only hang.
      reports = self._call_all(pool, handles, {"plan": None})
      worlds = {tuple(r["world"]) for r in reports}
      processes = sorted(int(r["process"]) for r in reports)
      if len(worlds) != 1 or processes != list(next(iter(worlds))):
        raise RuntimeError(
            "ICI weight sync needs exactly one live worker per process of one"
            f" shared JAX world; workers report processes {processes} of"
            f" world(s) {sorted(worlds)}. Trainer and rollout workers must"
            " join the same jax.distributed runtime"
        )
      plan = self.build_plan(src_units, dst_units, world_processes=processes)
      request = {"plan": plan, "req_id": req_id, "generation": generation}
      # Then every worker validates its side of the final plan. A worker
      # failing inside the collectives would strand its peers, so everything
      # that CAN fail early must fail here.
      self._call_all(pool, handles, request)
      # Phase 2: everyone enters the collectives together.
      runs = [
          pool.submit(self._call, handle, "ici_transfer", request)
          for handle in handles
      ]
      done, _ = futures.wait(runs, return_when=futures.FIRST_EXCEPTION)
      failed = next((f for f in done if f.exception() is not None), None)
      if failed is not None:
        # The survivors may be blocked in a collective the failed worker never
        # entered; whether they wrote anything is unknowable from here.
        raise weight_sync.TransferOutcomeUnknownError(
            f"ICI transfer {req_id} failed on a participant; its peers may"
            f" still be inside the collective: {failed.exception()!r}"
        ) from failed.exception()
      timings = [f.result() for f in runs]
    finally:
      pool.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - started
    moved = sum(int(t.get("bytes", 0)) for t in timings if t)
    message = (
        f"{len(plan.leaves)} leaves, {len(plan.chunks)} chunk(s) x"
        f" {len(plan.rounds)} round(s), {moved / 2**30:.2f} GiB to"
        f" {len(plan.destination_processes)} replica(s) in {elapsed:.2f}s"
    )
    logging.info("ICI transfer %s: %s", req_id, message)
    return weight_sync.TransferResult(
        req_id=req_id, success=True, message=message
    )
