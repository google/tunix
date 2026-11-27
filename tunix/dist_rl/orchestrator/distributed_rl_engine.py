# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed compute routing surface (Layer 1) following Orchestrator V2.

Contains:
- WorkerPoolBalancer: Load balancing, queue tracking, and prefix-cache affinity.
- DistributedRLEngine: Worker-backed compute router implementing
AbstractRLEngine.
"""

import asyncio
import bisect
import collections
from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
import contextlib
import inspect
import itertools
import math
import re
from typing import Any
import uuid

from absl import logging
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage
from tunix.dist_rl.common import logging_utils
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import fault_tolerance_manager as fault_tolerance_manager_lib
from tunix.dist_rl.orchestrator import rl_engine_interface
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import function_registry

_summarize_list = logging_utils.summarize_list
# Maximum number of contiguous trajectories per prompt stripe.
_ROLLOUT_PROMPT_STRIPE_SIZE: int = 4
# Lower-bound target on stripes per prompt when capping stripe size for fanout.
_TARGET_STRIPES_PER_PROMPT: int = 4


# Weight sync pushes to serving workers and to late joiners awaiting first sync.
_WEIGHT_SYNC_DESTINATION_STATUSES = (
    worker_registry.AdmissionStatus.ACTIVE,
    worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
)
# Roles whose workers form one SPMD trainer (e.g. a multi-host McJAX trainer):
# every call fans out to all of their workers. Other roles (e.g. REFERENCE)
# are independent replicas, so a call goes to one worker only.
_SPMD_ROLES = (datatypes.Role.ACTOR, datatypes.Role.CRITIC)


def _natural_worker_key(
    info: datatypes.WorkerInfo,
) -> tuple[tuple[int, int | str], ...]:
  """Orders workers by natural `worker_id` (`trainer-2` before `trainer-10`).

  Multi-host trainer replica `r` runs as JAX process `r`, so this is also the
  JAX process order.
  """
  return tuple(
      (0, int(tok)) if tok.isdigit() else (1, tok)
      for tok in re.split(r"(\d+)", info.worker_id)
      if tok
  )


def _restored_step_signature(result: Any) -> tuple[Any, Any] | None:
  """Returns the `(step, global_step)` a restore result resumes from.

  Reads the result as `resume_from_checkpoint` does: a non-mapping result is a
  fresh run, and either key may be absent (legacy checkpoints carry only
  `step`), which then compares as None.

  Args:
    result: One worker's `restore_checkpoint` result.

  Returns:
    The comparable step pair, or None for a fresh run.
  """
  if not isinstance(result, Mapping):
    return None
  return (result.get("step"), result.get("global_step"))


def _response_to_trajectory_item(resp: Any) -> datatypes.TrajectoryItem:
  """Converts a worker rollout response to a TrajectoryItem."""
  if not isinstance(resp, datatypes.RolloutResponse):
    raise TypeError(f"Unsupported response type: {type(resp)}")

  if resp.payload is not None:
    return resp.payload

  if resp.error is not None:
    metadata = dict(resp.metadata) if resp.metadata else {}
    prompt_id = metadata.get("prompt_id", "")
    group_index = metadata.get("group_index", 0)
    metadata["error"] = str(resp.error)
    return datatypes.TrajectoryItem(
        prompt_id=prompt_id,
        group_index=group_index,
        traj={
            "prompt_tokens": np.zeros(0, dtype=np.int32),
            "conversation_tokens": np.zeros(0, dtype=np.int32),
            "conversation_masks": np.zeros(0, dtype=np.float32),
            "old_logprobs": np.zeros(0, dtype=np.float32),
            "status": datatypes.TrajectoryStatus.FAILED,
            "trajectory_reward": 0.0,
        },
        metadata=metadata,
    )

  raise ValueError("RolloutResponse payload is None.")


def _group_refs_by_worker_instance(
    refs: Iterable[datatypes.RoutedExpertsRef],
) -> dict[tuple[str, str], list[datatypes.RoutedExpertsRef]]:
  """Groups `RoutedExpertsRef`s into `(worker_id, instance_id) -> [ref, ...]`."""
  by_instance: dict[tuple[str, str], list[datatypes.RoutedExpertsRef]] = {}
  for ref in refs:
    by_instance.setdefault((ref.worker_id, ref.instance_id), []).append(ref)
  return by_instance


_RolloutTargetEntry = tuple[
    datatypes.WorkerInfo,
    remote_execution.ActorHandle,
    worker_registry.AdmissionStatus,
]


def _resolve_active_staging_handle(
    wid: str,
    ref_instance_id: str,
    targets_by_wid: Mapping[str, _RolloutTargetEntry],
) -> tuple[remote_execution.ActorHandle | None, str]:
  """Resolves an `ACTIVE` rollout handle matching `ref_instance_id`.

  Args:
    wid: Logical rollout worker identifier.
    ref_instance_id: Process `instance_id` stamped on the staged reference.
    targets_by_wid: Registry snapshot keyed by `worker_id`.

  Returns:
    `(handle, "")` when `wid` is currently `ACTIVE` with `instance_id ==
    ref_instance_id`, or `(None, reason)` explaining why `wid` cannot serve
    staged references from `ref_instance_id`.
  """
  entry = targets_by_wid.get(wid)
  if entry is None:
    return (
        None,
        (
            f"Rollout worker {wid!r} was evicted before staged routed_experts"
            " could be fetched."
        ),
    )
  info, handle, status = entry
  if status is not worker_registry.AdmissionStatus.ACTIVE:
    return (
        None,
        (
            f"Rollout worker {wid!r} is in status {status.value!r} (not"
            " ACTIVE); staged routed_experts from its prior instance are"
            " unavailable."
        ),
    )
  if info.instance_id != ref_instance_id:
    return (
        None,
        (
            f"Rollout worker {wid!r} restarted with"
            f" instance_id={info.instance_id!r} != {ref_instance_id!r} before"
            " staged routed_experts could be fetched."
        ),
    )
  return handle, ""


def _validate_staging_overcommit_factor(factor: float) -> float:
  """Validates `overcommit_factor` as a finite float >= 1.0."""
  if (
      isinstance(factor, bool)
      or not isinstance(factor, (int, float))
      or not math.isfinite(factor)
      or factor < 1.0
  ):
    raise ValueError(
        f"overcommit_factor must be a finite float >= 1.0, got {factor!r}."
    )
  return float(factor)


def _worker_staging_slots(
    info: datatypes.WorkerInfo,
    worst_case_bytes: int,
    overcommit_factor: float,
) -> int:
  """Computes orchestrator-side staging slot capacity for one worker."""
  raw_max = info.resources.get("staging_max_bytes")
  if raw_max is None:
    raise ValueError(
        f"Rollout worker {info.worker_id!r} is missing"
        " 'staging_max_bytes' in info.resources."
    )
  physical_slots = int(raw_max) // worst_case_bytes
  if physical_slots < 1:
    raise RuntimeError(
        f"Rollout worker {info.worker_id!r} has staging_max_bytes={raw_max}"
        f" < worst_case_bytes={worst_case_bytes} (0 physical staging slots)."
        " Increase RolloutConfig.staging_max_bytes."
    )
  return int(physical_slots * overcommit_factor)


def _validate_staging_slots(
    targets: Sequence[tuple[datatypes.WorkerInfo, Any]],
    *,
    worst_case_bytes: int,
    required_slots: int,
    overcommit_factor: float,
) -> None:
  """Validates active rollout workers satisfy the Relay staging slot bound."""
  total_slots = sum(
      _worker_staging_slots(info, worst_case_bytes, overcommit_factor)
      for info, _ in targets
  )
  if total_slots < required_slots:
    bound_desc = (
        "required deadlock-free lower bound"
        if overcommit_factor == 1.0
        else "required lower bound"
    )
    raise RuntimeError(
        f"Active rollout pool has only {total_slots} worst-case routed_experts"
        f" staging slots (worst_case_bytes={worst_case_bytes},"
        f" overcommit_factor={overcommit_factor:g}) across {len(targets)}"
        f" worker(s), below {bound_desc} {required_slots}. Increase"
        " RolloutConfig.staging_max_bytes or rollout_replicas (or set"
        " staging_overcommit_factor > 1.0 if average trajectories are shorter"
        " than max_response_length, noting that relaxes the strict"
        " deadlock-free guarantee)."
    )


def _apply_request_staging_defaults(
    requests: Sequence[datatypes.RolloutRequest],
    *,
    stage_routed_experts: bool,
    default_max_response_length: int | None,
) -> None:
  """Applies Relay `stage_routed_experts` and validates response bounds."""
  for req in requests:
    if stage_routed_experts:
      req.stage_routed_experts = True
    if (
        req.max_response_length is None
        and default_max_response_length is not None
    ):
      req.max_response_length = default_max_response_length
    if (
        stage_routed_experts
        and default_max_response_length is not None
        and (
            req.max_response_length is None
            or not 0 < req.max_response_length <= default_max_response_length
        )
    ):
      raise ValueError(
          "RolloutRequest.max_response_length must be in (0,"
          f" {default_max_response_length}] when stage_routed_experts=True,"
          f" got {req.max_response_length!r}."
      )


def _pick_stripe_slice(
    worker_loads: Sequence[int],
    max_slots: Sequence[int | None],
    rr_cursor: int,
    unit_sizes: Sequence[int],
) -> tuple[int, int]:
  """Selects `(worker_idx, unit_count)` for the next rollout stripe."""
  num_workers = len(worker_loads)
  scored: list[tuple[int, float, int, int, int]] = []
  for w, cap in enumerate(max_slots):
    load = worker_loads[w]
    rr_dist = (w - rr_cursor) % num_workers
    if cap is None:
      scored.append((0, 0.0, load, rr_dist, w))
    else:
      scored.append((int(load >= cap), load / cap, load, rr_dist, w))
  best_w = min(scored)[-1]
  cap = max_slots[best_w]
  if cap is None or worker_loads[best_w] >= cap:
    return best_w, len(unit_sizes)
  fit = bisect.bisect_right(
      list(itertools.accumulate(unit_sizes)), cap - worker_loads[best_w]
  )
  return best_w, max(1, fit)


def _assign_rollout_stripes(
    requests: Sequence[datatypes.RolloutRequest],
    actors: Sequence[remote_execution.ActorHandle],
    worker_loads: list[int],
    max_slots: Sequence[int | None],
    last_routed_actor: remote_execution.ActorHandle | None,
) -> tuple[
    list[remote_execution.ActorHandle | None],
    remote_execution.ActorHandle | None,
]:
  """Assigns prompt stripes across rollout workers by slot occupancy."""
  num_workers = len(actors)
  target_actors: list[remote_execution.ActorHandle | None] = [None] * len(
      requests
  )
  if num_workers <= 1 or not requests:
    return target_actors, last_routed_actor

  prompt_traj_buckets: dict[str, dict[str, list[int]]] = {}
  for idx, req in enumerate(requests):
    traj_buckets = prompt_traj_buckets.setdefault(str(req.prompt_id), {})
    traj_buckets.setdefault(req.traj_id, []).append(idx)

  min_stripes = min(num_workers, _TARGET_STRIPES_PER_PROMPT)
  stripes: list[list[list[int]]] = []
  for traj_buckets in prompt_traj_buckets.values():
    units = list(traj_buckets.values())
    step = min(_ROLLOUT_PROMPT_STRIPE_SIZE, max(1, len(units) // min_stripes))
    stripes.extend(units[i : i + step] for i in range(0, len(units), step))

  rr_cursor = (
      (actors.index(last_routed_actor) + 1) % num_workers
      if last_routed_actor in actors
      else 0
  )
  best_w = 0
  for pending in stripes:
    while pending:
      best_w, take = _pick_stripe_slice(
          worker_loads, max_slots, rr_cursor, [len(u) for u in pending]
      )
      chosen, pending = pending[:take], pending[take:]
      req_indices = tuple(itertools.chain.from_iterable(chosen))
      worker_loads[best_w] += len(req_indices)
      rr_cursor = (best_w + 1) % num_workers
      for req_idx in req_indices:
        target_actors[req_idx] = actors[best_w]
  return target_actors, actors[best_w]


class DistributedRLEngine(rl_engine_interface.AbstractRLEngine):
  """Worker-backed compute router dispatching RPCs across role pools."""

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      weight_sync_coordinator: Any = None,
      *,
      fault_tolerance_manager: (
          fault_tolerance_manager_lib.FaultToleranceManager | None
      ) = None,
  ):
    """Initializes DistributedRLEngine."""
    self._registry = registry
    self._fault_tolerance_manager = (
        fault_tolerance_manager
        or fault_tolerance_manager_lib.FaultToleranceManager(registry)
    )
    self._fault_tolerance_manager.check_rollout_capacity()
    self._rollout_session = task_session.TaskSession(
        self._rollout_handles,
        poll_retry_budget_s=(
            self._fault_tolerance_manager.config.effective_poll_retry_budget_s
        ),
        fault_tolerance_config=(
            self._fault_tolerance_manager.task_session_fault_tolerance_config()
        ),
    )
    self._policy_version = 0
    self._last_routed_actor: remote_execution.ActorHandle | None = None
    self._weight_sync_coordinator = weight_sync_coordinator
    self._restored_checkpoint_metadata: dict[str, Any] | None = None
    self._inflight_fwd_bwd: dict[
        datatypes.Role | str, set[asyncio.Future[None]]
    ] = collections.defaultdict(set)
    self._unreleased_staged_refs: collections.Counter[tuple[str, str]] = (
        collections.Counter()
    )
    self._staging_worst_case_bytes: int | None = None
    self._required_staging_slots: int = 0
    self._staging_overcommit_factor: float = 1.0
    self._evicted_release_skips_total: int = 0
    self._rollout_epoch: int = 0
    self._expected_role_counts: dict[datatypes.Role | str, int] = {}
    self._spmd_locks_loop: asyncio.AbstractEventLoop | None = None
    self._spmd_locks_by_handle: dict[int, asyncio.Lock] = {}

  def _rollout_handles(self) -> list[remote_execution.ActorHandle]:
    return self._registry.handles(datatypes.Role.ROLLOUT)

  def _rollout_targets_by_wid(self) -> dict[str, _RolloutTargetEntry]:
    return {
        info.worker_id: (info, handle, status)
        for info, handle, status in self._registry.targets_with_status(
            datatypes.Role.ROLLOUT, statuses=None
        )
    }

  def _role_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns active handles for `role` in deterministic SPMD order."""
    targets = self._registry.targets(
        role, statuses=(worker_registry.AdmissionStatus.ACTIVE,)
    )
    return [
        handle
        for _, handle in sorted(
            targets, key=lambda item: _natural_worker_key(item[0])
        )
    ]

  def _require_role_handles(
      self,
      role: datatypes.Role | str,
      *,
      missing_error: str | None = None,
  ) -> list[remote_execution.ActorHandle]:
    """Returns the `role` handles a call fans out to.

    All active workers of an SPMD role (`_SPMD_ROLES`), else only the first.

    Args:
      role: Worker role.
      missing_error: Error message when no `role` worker is active.

    Raises:
      ValueError: If no `role` worker is active.
      RuntimeError: If an SPMD role has fewer active workers than before.
    """
    workers = self._role_handles(role)
    if not workers:
      raise ValueError(
          missing_error or f"No worker registered for role {role}"
      )
    if role not in _SPMD_ROLES:
      return workers[:1]
    expected = self._expected_role_counts.get(role)
    if expected is None or len(workers) > expected:
      self._expected_role_counts[role] = len(workers)
    elif len(workers) < expected:
      raise RuntimeError(
          f"SPMD role {role} shrunk from {expected} active worker(s) to"
          f" {len(workers)}; aborting to prevent collective deadlock."
      )
    return workers

  def _role_handle(
      self, role: datatypes.Role | str
  ) -> remote_execution.ActorHandle | None:
    handles = self._role_handles(role)
    return handles[0] if handles else None

  async def _invoke_role_workers(
      self,
      workers: Sequence[remote_execution.ActorHandle],
      method_name: str,
      *,
      return_all: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Invokes `method_name` concurrently on `workers`; returns first or all.

    Fan-outs to more than one worker (one SPMD program on several hosts) are
    serialized per worker in FIFO order, so every host receives the same call
    sequence even when several callers fan out at once.
    """
    if not workers:
      raise ValueError(f"No workers provided for {method_name}.")
    if len(workers) == 1:
      result = await self._invoke_worker(workers[0], method_name, **kwargs)
      return [result] if return_all else result
    # Each worker runs its calls in arrival order only by accident (a non-FIFO
    # lock), so concurrent fan-outs could reach the SPMD hosts in different
    # orders and leave them in different collectives. Hold every host's lock
    # until all hosts have finished, so each host sees the same call sequence.
    async with contextlib.AsyncExitStack() as stack:
      for lock in self._spmd_locks(workers):
        await stack.enter_async_context(lock)
      return await self._fan_out(
          workers, method_name, return_all=return_all, **kwargs
      )

  def _spmd_locks(
      self, workers: Sequence[remote_execution.ActorHandle]
  ) -> list[asyncio.Lock]:
    """Returns the FIFO locks of `workers`, in one global acquisition order."""
    loop = asyncio.get_running_loop()
    if self._spmd_locks_loop is not loop:
      # An `asyncio.Lock` is bound to the loop it first waited on.
      self._spmd_locks_loop = loop
      self._spmd_locks_by_handle = {}
    return [
        self._spmd_locks_by_handle.setdefault(key, asyncio.Lock())
        for key in sorted({id(w) for w in workers})
    ]

  async def _fan_out(
      self,
      workers: Sequence[remote_execution.ActorHandle],
      method_name: str,
      *,
      return_all: bool,
      **kwargs: Any,
  ) -> Any:
    """Invokes `method_name` concurrently on all `workers`; fails on any."""
    tasks = [
        asyncio.create_task(self._invoke_worker(w, method_name, **kwargs))
        for w in workers
    ]
    pending = set(tasks)
    try:
      # FIRST_EXCEPTION ignores a peer that ended cancelled, so wake on every
      # completion and fail on the first raised or cancelled peer.
      while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
          exc = (
              asyncio.CancelledError() if task.cancelled() else task.exception()
          )
          if exc is not None:
            raise RuntimeError(
                f"SPMD call {method_name!r} failed on one of {len(workers)}"
                f" workers ({type(exc).__name__}: {exc}); cancelled"
                f" {len(pending)} remaining peer task(s) (surviving SPMD"
                " peers may require teardown)."
            ) from exc
    except BaseException:
      for task in tasks:
        if not task.done():
          task.cancel()
          task.add_done_callback(
              lambda t: t.exception() if not t.cancelled() else None
          )
        elif not task.cancelled():
          task.exception()
      raise
    results = [task.result() for task in tasks]
    return results if return_all else results[0]

  @property
  def restored_checkpoint_metadata(self) -> dict[str, Any] | None:
    """Returns the metadata dict from the most recent checkpoint restoration."""
    return self._restored_checkpoint_metadata

  @property
  def evicted_release_skips_total(self) -> int:
    """Returns staged refs skipped during release due to worker eviction."""
    return self._evicted_release_skips_total

  async def _maybe_configure_trainer_target_state(
      self,
      role: datatypes.Role,
  ) -> None:
    """Seeds trainer-side weight sync with the rollout target-state skeleton."""
    trainers = self._role_handles(role)
    # Match the weight sync destinations: a rejoined rollout worker still in
    # PENDING_WEIGHT_SYNC can seed the trainer before its first sync.
    rollout_handles = self._registry.handles(
        datatypes.Role.ROLLOUT, statuses=_WEIGHT_SYNC_DESTINATION_STATUSES
    )
    if not trainers or not rollout_handles:
      return

    rollout = rollout_handles[0]
    try:
      target_state = await self._invoke_worker(rollout, "get_target_state")
      await self._invoke_role_workers(
          trainers, "set_target_state", target_state=target_state
      )
    except (AttributeError, RuntimeError) as exc:
      if isinstance(exc, RuntimeError) and "AttributeError" not in str(exc):
        raise

  async def _invoke_worker(
      self,
      worker: remote_execution.ActorHandle,
      method_name: str,
      **kwargs: Any,
  ) -> Any:
    """Helper invoking method on remote handle."""
    res = worker.asubmit(method_name, **kwargs)
    if inspect.isawaitable(res):
      return await res
    return res

  async def dispatch_rollout_requests(
      self,
      requests: Sequence[datatypes.RolloutRequest],
  ) -> list[str]:
    """Dispatches pre-formed RolloutRequests across rollout workers.

    When multiple rollout workers are active, groups requests by `prompt_id`
    (colocating duplicate `traj_id` entries submitted within the same call, not
    across separate calls), slices each prompt's trajectories into contiguous
    stripes of size `min(_ROLLOUT_PROMPT_STRIPE_SIZE, max(1, num_trajs //
    min(num_workers, _TARGET_STRIPES_PER_PROMPT)))`, and assigns stripes via
    `preferred_actor` using slot-occupancy-aware scoring: when
    `stage_routed_experts` is enabled, occupancy counts `in_flight +
    unreleased_staged_refs` against each worker's worst-case staging slot cap
    (splitting stripes across workers when a worker nears its cap) and breaks
    ties via an actor-identity round-robin cursor. Every submission retains
    `route_key=req.traj_id` so stale-handle fallbacks and fault-tolerant retries
    spread across surviving workers.

    Args:
      requests: Rollout requests to dispatch.

    Returns:
      The list of dispatched `request_id` strings.
    """
    requests = self._build_rollout_requests(requests)
    targets = self._registry.targets(
        datatypes.Role.ROLLOUT,
        statuses=(worker_registry.AdmissionStatus.ACTIVE,),
    )
    inflight_by_actor = dict(self._rollout_session.worker_inflight_loads())
    stage_any = any(req.stage_routed_experts for req in requests)
    actors = [handle for _, handle in targets]
    worker_loads: list[int] = []
    max_slots: list[int | None] = []
    for info, handle in targets:
      inflight = inflight_by_actor.get(handle, 0)
      occupied = inflight + (
          self._unreleased_staged_refs[(info.worker_id, info.instance_id)]
          if stage_any
          else 0
      )
      worker_loads.append(occupied)
      if stage_any and self._staging_worst_case_bytes is not None:
        max_slots.append(
            _worker_staging_slots(
                info,
                self._staging_worst_case_bytes,
                self._staging_overcommit_factor,
            )
        )
      else:
        max_slots.append(None)
    num_workers = len(actors)
    logging.info(
        "Dispatching %d rollout request(s) across %d worker(s).",
        len(requests),
        num_workers,
    )
    for req in requests:
      logging.debug(
          "Dispatched rollout request (prompt_id=%s, group_index=%d,"
          " request_id=%s).",
          req.prompt_id,
          req.group_index,
          req.request_id,
      )

    target_actors, self._last_routed_actor = _assign_rollout_stripes(
        requests,
        actors,
        worker_loads,
        max_slots,
        self._last_routed_actor,
    )

    await asyncio.gather(*(
        self._rollout_session.submit(
            req.request_id,
            "generate",
            requests=[req],
            route_key=req.traj_id,
            preferred_actor=actor,
        )
        for req, actor in zip(requests, target_actors, strict=True)
    ))

    return [r.request_id for r in requests]

  def _build_rollout_requests(
      self,
      prompts: Sequence[Any],
      *,
      num_generations: int = 1,
      policy_version: int = 0,
      route_metadata: Mapping[str, Any] | None = None,
  ) -> list[datatypes.RolloutRequest]:
    """Validates prompts and constructs typed RolloutRequests with lineage attached."""
    base_metadata = dict(route_metadata or {})
    version = policy_version

    rollout_reqs: list[datatypes.RolloutRequest] = []
    req_task_names: list[str] = []
    for idx, p in enumerate(prompts):
      if isinstance(p, datatypes.RolloutRequest):
        if p.prompt_id is None or p.prompt_id == "":
          raise ValueError(
              f"RolloutRequest at index {idx} (id='{p.request_id}') lacks"
              " 'prompt_id'. Every request must provide a non-empty"
              " 'prompt_id'."
          )
        rollout_reqs.append(p)
        req_task_names.append("")
        continue

      if (
          "generation_kwargs" in p
          if isinstance(p, Mapping)
          else hasattr(p, "generation_kwargs")
      ):
        raise ValueError(
            f"Prompt at index {idx} sets 'generation_kwargs'; sampling settings"
            " come from the rollout workers' RolloutConfig."
        )
      item_metadata = dict(getattr(p, "metadata", {}) or {})
      if isinstance(p, Mapping):
        item_metadata.update(dict(p.get("metadata", {}) or {}))

      if isinstance(p, Mapping):
        prompt_id = p.get("prompt_id")
      else:
        prompt_id = getattr(p, "prompt_id", None)

      if prompt_id is None or prompt_id == "":
        raise ValueError(
            f"Prompt at index {idx} lacks 'prompt_id'. Every prompt item "
            "dispatched to DistributedRLEngine must provide a unique, "
            "collision-free 'prompt_id' (as an attribute or dict key)."
        )

      prompt_id = str(prompt_id)
      item_task_name = (
          p["task_name"] if isinstance(p, Mapping) and "task_name" in p else ""
      )
      if not isinstance(item_task_name, str):
        raise TypeError(
            f"Prompt at index {idx} has non-string 'task_name' of type"
            f" {type(item_task_name).__name__}."
        )
      raw_prompt = (
          p.get("prompt", p)
          if isinstance(p, Mapping)
          else getattr(p, "prompt", p)
      )
      max_turns = getattr(p, "max_turns", 10)
      if isinstance(p, Mapping):
        max_turns = p.get("max_turns", max_turns)

      max_response_length = getattr(p, "max_response_length", None)
      if isinstance(p, Mapping):
        max_response_length = p.get("max_response_length", max_response_length)

      for group_index in range(num_generations):
        request_metadata = dict(base_metadata)
        request_metadata.update(item_metadata)
        request_metadata["group_index"] = group_index
        request_metadata["num_generations"] = num_generations
        if isinstance(request_metadata.get("env_config"), Mapping):
          env_config = dict(request_metadata["env_config"])
          env_config["group_index"] = group_index
          env_config["num_generations"] = num_generations
          env_config["policy_version"] = version
          request_metadata["env_config"] = env_config

        epoch_suffix = (
            f"_e{self._rollout_epoch}" if self._rollout_epoch > 0 else ""
        )
        rollout_reqs.append(
            datatypes.RolloutRequest(
                request_id=(
                    f"req_{prompt_id}_g{group_index}_v{version}{epoch_suffix}"
                ),
                prompt=raw_prompt,
                prompt_id=prompt_id,
                group_index=group_index,
                target_policy_version=version,
                max_turns=max_turns,
                max_response_length=max_response_length,
                metadata=request_metadata,
            )
        )
        req_task_names.append(item_task_name)

    prompt_ids = list(dict.fromkeys(req.prompt_id for req in rollout_reqs))
    logging.info(
        "Created rollout requests for %d prompts (num_generations=%d,"
        " total_requests=%d, policy_version=%d). Prompt IDs: %s",
        len(prompts),
        num_generations,
        len(rollout_reqs),
        version,
        _summarize_list(prompt_ids),
    )
    for req, req_task_name in zip(rollout_reqs, req_task_names, strict=True):
      if req.metadata is None:
        req.metadata = {}  # pyrefly: ignore[bad-assignment]
      if req.metadata.get("lineage") is None:
        lineage_ctx = lineage.LineageContext(
            tracking_id=req.traj_id,
            task_name=req_task_name,
            parent_tracking_ids=[str(req.prompt_id)],
        )
        dispatch_attrs: dict[str, Any] = {
            "policy_version": req.target_policy_version,
            "group_index": req.group_index,
        }
        if "batch_idx" in req.metadata:
          dispatch_attrs["batch_idx"] = req.metadata["batch_idx"]
        lineage_ctx.add_event(
            component="engine.dispatch",
            operation="rollout",
            attributes=dispatch_attrs,
        )
        req.metadata["lineage"] = lineage_ctx

    return rollout_reqs

  async def dispatch_rollouts(
      self,
      prompts: Sequence[Any],
      *,
      num_generations: int = 1,
      policy_version: int = 0,
      route_metadata: Mapping[str, Any] | None = None,
      stage_routed_experts: bool = False,
      max_response_length: int | None = None,
  ) -> list[str]:
    """Dispatches rollout requests across workers, constructing RolloutRequests internally.

    Every prompt item in `prompts` MUST have a unique, collision-free
    `prompt_id` attribute or dict key (missing prompt IDs raise a `ValueError`),
    and may optionally provide a string `task_name` dict key to stamp onto its
    `LineageContext` (non-string `task_name` values raise a `TypeError`).
    """
    rollout_reqs = self._build_rollout_requests(
        prompts,
        num_generations=num_generations,
        policy_version=policy_version,
        route_metadata=route_metadata,
    )
    if stage_routed_experts or max_response_length is not None:
      _apply_request_staging_defaults(
          rollout_reqs,
          stage_routed_experts=stage_routed_experts,
          default_max_response_length=max_response_length,
      )
    return await self.dispatch_rollout_requests(rollout_reqs)

  async def poll_rollouts(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> list[datatypes.TrajectoryItem]:
    """Concurrently long-polls completed rollout responses across all workers."""
    completed: list[datatypes.TrajectoryItem] = []
    for res, exc in await self._rollout_session.poll_completed(
        timeout_s=timeout_s
    ):
      if exc is not None:
        raise exc
      if res is None:
        continue
      items = res if isinstance(res, list) else [res]
      for it in items:
        if isinstance(it, dict):
          it = datatypes.RolloutResponse(**it)
        traj_item = _response_to_trajectory_item(it)
        ref = traj_item.routed_experts_ref
        if ref is not None:
          self._unreleased_staged_refs[(ref.worker_id, ref.instance_id)] += 1
        logging.debug(
            "Received rollout response (prompt_id=%s, group_index=%d).",
            traj_item.prompt_id,
            traj_item.group_index,
        )
        completed.append(traj_item)
    self._fault_tolerance_manager.check_rollout_capacity()
    return completed

  async def generate(
      self,
      prompts: Sequence[Any],
      route_metadata: Mapping[str, Any] | None = None,
  ) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    rollout_handles = self._rollout_handles()
    if not rollout_handles:
      raise ValueError("DistributedRLEngine has no registered rollout workers.")

    logging.info(
        "Generating rollouts for %d prompt(s)/request(s) across %d"
        " worker(s)...",
        len(prompts),
        len(rollout_handles),
    )
    requests = self._build_rollout_requests(
        prompts,
        policy_version=self._policy_version,
        route_metadata=route_metadata,
    )
    if not requests:
      return []

    results = await asyncio.gather(
        *(
            self._rollout_session.route(req.traj_id).asubmit(
                "generate",
                requests=[req],
            )
            for req in requests
        )
    )
    raw_items = [
        item
        for sublist in results
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    logging.info(
        "Completed synchronous generation of %d trajectory item(s).",
        len(raw_items),
    )
    items = [_response_to_trajectory_item(it) for it in raw_items]
    for it in items:
      logging.debug(
          "Generated rollout response (prompt_id=%s, group_index=%d).",
          it.prompt_id,
          it.group_index,
      )
    return items

  async def score(
      self,
      role: datatypes.Role,
      items: Sequence[Any],
      **kwargs: Any,
  ) -> list[float]:
    """Routes reward / PRM scoring requests to InferenceWorker pool."""
    worker = self._role_handle(role)
    if worker is None:
      raise ValueError(f"No inference worker registered for role {role}")
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Scoring %d item(s) on %s worker...",
        len(items),
        role_name,
    )
    return await self._invoke_worker(worker, "score", items=items, **kwargs)

  async def per_token_logps(
      self,
      role: datatypes.Role,
      items: Any,
      **kwargs: Any,
  ) -> Any:
    """Evaluates reference model or actor logprobs on a padded batch/request."""
    workers = self._require_role_handles(
        role,
        missing_error=(
            f"No worker registered for per_token_logps with role {role}"
        ),
    )
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Evaluating per-token log probabilities on %d %s worker(s)...",
        len(workers),
        role_name,
    )
    return await self._invoke_role_workers(
        workers, "per_token_logps", items=items, **kwargs
    )

  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
  ) -> Any:
    """Executes atomic gradient accumulation / update on TrainerWorker(s)."""
    workers = self._require_role_handles(role)
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Executing train_step on %d %s worker(s) (accumulate_gradients=%s,"
        " apply_optimizer=%s)...",
        len(workers),
        role_name,
        accumulate_gradients,
        apply_optimizer,
    )
    metadata = dict(getattr(payload, "metadata", {}) or {})
    request = datatypes.TrainRequest(
        request_id=f"train_req_{uuid.uuid4().hex[:8]}",
        payload=payload,
        metadata=metadata,
    )
    inflight_set = self._inflight_fwd_bwd[role]
    prior_fwd_bwd = tuple(inflight_set) if apply_optimizer else ()
    fwd_bwd_done: asyncio.Future[None] = (
        asyncio.get_running_loop().create_future()
    )
    inflight_set.add(fwd_bwd_done)
    try:
      fwd_bwd_result = await self._invoke_role_workers(
          workers, "fwd_bwd", request=request
      )
    except BaseException as exc:
      if not fwd_bwd_done.done():
        prior_exc = (
            RuntimeError(
                "Prior in-flight fwd_bwd was cancelled before optimizer update."
            )
            if isinstance(exc, asyncio.CancelledError)
            else exc
        )
        fwd_bwd_done.set_exception(prior_exc)
        fwd_bwd_done.exception()
      if apply_optimizer:
        inflight_set.discard(fwd_bwd_done)
        inflight_set.difference_update(prior_fwd_bwd)
      raise
    else:
      if not fwd_bwd_done.done():
        fwd_bwd_done.set_result(None)
      inflight_set.discard(fwd_bwd_done)
    if not apply_optimizer:
      return fwd_bwd_result
    try:
      for prior in prior_fwd_bwd:
        await asyncio.shield(prior)
    finally:
      inflight_set.difference_update(prior_fwd_bwd)
    update_results = await self._invoke_role_workers(
        workers, "update", return_all=True
    )
    steps = [res[0] for res in update_results]
    if len(set(steps)) != 1:
      raise RuntimeError(
          f"SPMD {role_name} workers disagreed on train_step after update:"
          f" {steps}."
      )
    train_step, metrics = update_results[0]
    return {
        "fwd_bwd": fwd_bwd_result,
        "updated": True,
        "train_step": train_step,
        "accumulated": accumulate_gradients,
        "metrics": metrics,
    }

  async def get_metrics(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      **kwargs: Any,
  ) -> (
      datatypes.MetricsBuffer
      | Sequence[datatypes.MetricsBuffer]
      | dict[str, Any]
      | None
  ):
    """Retrieves step metrics from the worker(s) registered for `role`.

    For `Role.ROLLOUT`, returns `[]` under `FaultToleranceConfig.enabled=True`
    when zero rollout workers are currently `ACTIVE` mid-run, skips rollout
    workers failing with transport or closed-handle errors, and propagates
    `RemoteCallError` application errors (when `enabled=False`, worker and
    transport errors fail fast).

    Args:
      role: Worker role whose metrics should be queried.
      **kwargs: Additional keyword arguments forwarded to worker `get_metrics`.

    Returns:
      Aggregated list of per-worker metric payloads for `Role.ROLLOUT`, or the
      single worker's metric payload for singleton roles.
    """
    if role == datatypes.Role.ROLLOUT:
      rollout_handles = self._rollout_handles()
      if not rollout_handles:
        if self._fault_tolerance_manager.config.enabled and (
            self._fault_tolerance_manager.peak_active_rollout_workers > 0
            or self._registry.handles(datatypes.Role.ROLLOUT, statuses=None)
        ):
          return []
        raise ValueError(f"No rollout workers registered for role {role}")
      results = await asyncio.gather(*(
          self._invoke_worker_ft_tolerant(
              w,
              "get_metrics",
              skip_msg="Skipping get_metrics on unreachable rollout worker",
              **kwargs,
          )
          for w in rollout_handles
      ))
      return [r for ok, r in results if ok and r is not None]
    else:
      workers = self._require_role_handles(role)
      return await self._invoke_role_workers(workers, "get_metrics", **kwargs)

  def _configure_trainer_workers(
      self,
      workers: Sequence[remote_execution.ActorHandle],
      role_name: str,
      loss_fn: Any,
      gen_fn: Any,
      dummy_payload: Any,
      precompile: bool,
  ) -> None:
    """Configures and optionally precompiles all SPMD trainer workers."""

    def _configure_one(worker: remote_execution.ActorHandle) -> None:
      # Both setters clear the trainer's JIT cache, so the ahead-of-time
      # compile must run strictly after them.
      worker.submit("with_loss_fn", loss_fn, has_aux=True)
      worker.submit("with_gen_model_input_fn", gen_fn)
      if not precompile:
        return
      # TODO(tunix-dev): Move to the worker lifecycle (e.g., the `initialize`
      # step) so pre-compilation runs during the standard lifecycle `compile`
      # phase instead of re-compiling here.
      logging.info(
          "Precompiling trainer step executables on %s worker...",
          role_name,
      )
      worker.submit("compile", dummy_payload)

    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if len(workers) == 1 and loop is None:
      _configure_one(workers[0])
      return

    # Multi-host SPMD trainers must enter `compile` concurrently so cross-host
    # lowering/collectives run in lockstep. Avoid `with ThreadPoolExecutor` so
    # an early failure on one worker does not block `shutdown(wait=True)` on
    # peers stuck in SPMD compile.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(workers))
    futs = [pool.submit(_configure_one, w) for w in workers]
    try:
      for fut in concurrent.futures.as_completed(futs):
        fut.result()
    except BaseException:
      pool.shutdown(wait=False, cancel_futures=True)
      raise
    pool.shutdown(wait=True)

  def configure_worker(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      *,
      algo_config: algorithm_config.AlgorithmConfig,
      assembler: batch_assembly.BatchAssembler[Any],
      **kwargs: Any,
  ) -> None:
    """Configures worker(s) under the specified role with algorithm or runtime settings.

    For trainer roles this registers the loss and model-input functions, then
    precompiles the trainer step executables against a dummy microbatch built
    by `assembler.build_dummy_payload(...)`, so the first real train step does
    not stall on XLA compilation.

    Args:
      role: Target worker role to configure (defaults to `Role.ACTOR`).
      algo_config: Algorithm configuration used to resolve the loss and
        model-input functions.
      assembler: Batch assembler used to build the dummy microbatch for
        ahead-of-time compilation.
      **kwargs: Role-specific configuration options (for trainer roles:
        `can_fuse_agreement_in_loss: bool = True`, `precompile: bool = True`,
        `return_routed_experts: bool = False`, and `pad_id` / `eos_id`.
        `StandardRLProgram` passes both from its `BatchConfig`; direct callers
        may omit them, in which case `pad_id` is read from the assembler and
        `eos_id` defaults to `pad_id`).
    """
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    if algo_config is None:
      raise ValueError(
          f"algo_config is required to configure worker for role {role_name}"
      )
    if assembler is None:
      raise ValueError(
          f"assembler is required to configure worker for role {role_name}"
      )
    match role:
      case datatypes.Role.ACTOR | datatypes.Role.CRITIC:
        workers = self._require_role_handles(
            role,
            missing_error=(
                f"No trainer worker registered for role {role_name}"
            ),
        )
        logging.info(
            "Auto-configuring trainer loss and model input fn on %d %s"
            " worker(s)...",
            len(workers),
            role_name,
        )
        pad_id = kwargs.get("pad_id", assembler.pad_id)
        eos_id = kwargs.get("eos_id", pad_id)
        can_fuse_agreement_in_loss = bool(
            kwargs.get("can_fuse_agreement_in_loss", True)
        )
        precompile = bool(kwargs.get("precompile", True))
        return_routed_experts = bool(kwargs.get("return_routed_experts", False))
        if role == datatypes.Role.CRITIC:
          # TODO(sizhi): The PPO critic is not supported end to end yet:
          # `create_trainer_payloads` never fills `old_values` and
          # `build_dummy_payload` carries no `returns` / `old_values`, so the
          # value loss cannot be precompiled or trained. This branch only wires
          # the loss and model-input functions.
          if not isinstance(algo_config, algorithm_config.PPOConfig):
            raise ValueError(
                "Configuring a CRITIC trainer worker requires a PPOConfig"
                f" (with value_loss_fn), got {type(algo_config).__name__}."
            )
          loss_fn = function_registry.get_value_loss_fn(
              algo_config.value_loss_fn
          )
          gen_fn = algo_core.build_critic_model_input_fn(
              algo_config,
              pad_id=pad_id,
              eos_id=eos_id,
          )
        else:
          loss_fn = function_registry.get_policy_loss_fn(
              algo_config.policy_loss_fn
          )
          gen_fn = algo_core.build_gen_model_input_fn(
              algo_config,
              pad_id=pad_id,
              eos_id=eos_id,
          )
        use_rollout_logps = algo_config.use_rollout_logps
        sampler_agreement_applied = (
            use_rollout_logps and not can_fuse_agreement_in_loss
        )
        with_sampler_is_weights = sampler_agreement_applied and (
            algo_config.sampler_is is not None
            or algo_config.sampler_rs is not None
        )
        dummy_payload = None
        if precompile:
          routed_experts_encode_mode = (
              algo_config.routed_experts_encode_mode
              if return_routed_experts
              else "auto"
          )
          dummy_payload = assembler.build_dummy_payload(
              with_old_per_token_logps=use_rollout_logps,
              with_ref_per_token_logps=algo_config.requires_reference_kl,
              with_sampler_is_weights=with_sampler_is_weights,
              sampler_agreement_applied=sampler_agreement_applied,
              with_routed_experts=return_routed_experts,
              routed_experts_encode_mode=routed_experts_encode_mode,
          )
        self._configure_trainer_workers(
            workers,
            role_name,
            loss_fn,
            gen_fn,
            dummy_payload,
            precompile,
        )

      case datatypes.Role.ROLLOUT:
        if not self._rollout_handles():
          raise ValueError("No rollout workers registered on engine.")
        logging.info("Configuring rollout workers...")

      case datatypes.Role.REFERENCE:
        worker = self._role_handle(role)
        if worker is None:
          raise ValueError(
              f"No inference worker registered for role {role_name}"
          )
        logging.info("Configuring reference inference worker...")

      case _:
        raise ValueError(f"Unsupported role for configure_worker: {role_name}")

  async def prepare_rollout_policy(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      sync_weights: bool = True,
      policy_version: int | None = None,
      **kwargs: Any,
  ) -> int | None:
    """Bootstraps the rollout-visible policy state before step 0."""
    del kwargs
    self._require_role_handles(role)

    await self._maybe_configure_trainer_target_state(role)

    if not sync_weights:
      return None
    target_policy_version = (
        self._policy_version if policy_version is None else policy_version
    )
    return await self.sync_weights(
        role=role, policy_version=target_policy_version
    )

  def routed_experts_spec(
      self,
      *,
      encode_mode: datatypes.RoutedExpertsEncodeMode = "auto",
      num_experts: int | None = None,
  ) -> batch_assembly.RoutedExpertsSpec:
    """Resolves unified MoE routing spec across active rollout workers."""
    targets = self._registry.targets(
        datatypes.Role.ROLLOUT,
        statuses=(worker_registry.AdmissionStatus.ACTIVE,),
    )
    if not targets:
      raise ValueError(
          "Cannot resolve routed_experts_spec: no active rollout workers."
      )
    canonical_shape: tuple[int, int] | None = None
    canonical_wid = ""
    for info, _ in targets:
      raw_shape = info.resources.get("routed_experts_shape")
      if raw_shape is None:
        raise ValueError(
            f"Rollout worker {info.worker_id!r} reported"
            " routed_experts_shape=None while return_routed_experts=True."
        )
      shape = (int(raw_shape[0]), int(raw_shape[1]))
      if canonical_shape is None:
        canonical_shape = shape
        canonical_wid = info.worker_id
      elif shape != canonical_shape:
        raise ValueError(
            f"Rollout worker {info.worker_id!r} reported"
            f" routed_experts_shape={shape}, disagreeing with"
            f" {canonical_wid!r} ({canonical_shape})."
        )
    assert canonical_shape is not None
    encoding = datatypes.resolve_routed_experts_wire_encoding(
        encode_mode, num_experts
    )
    return batch_assembly.RoutedExpertsSpec(
        num_layers=canonical_shape[0],
        top_k=canonical_shape[1],
        encoding=encoding,
    )

  def configure_staging_liveness(
      self,
      *,
      worst_case_bytes: int,
      required_slots: int,
      overcommit_factor: float = 1.0,
  ) -> None:
    """Validates and installs cluster-wide Relay staging liveness bounds."""
    if isinstance(worst_case_bytes, bool) or worst_case_bytes <= 0:
      raise ValueError(
          f"worst_case_bytes must be a positive int, got {worst_case_bytes!r}."
      )
    if isinstance(required_slots, bool) or required_slots < 0:
      raise ValueError(
          f"required_slots must be a non-negative int, got {required_slots!r}."
      )
    factor = _validate_staging_overcommit_factor(overcommit_factor)
    if required_slots > 0:
      targets = self._registry.targets(
          datatypes.Role.ROLLOUT,
          statuses=(worker_registry.AdmissionStatus.ACTIVE,),
      )
      _validate_staging_slots(
          targets,
          worst_case_bytes=worst_case_bytes,
          required_slots=required_slots,
          overcommit_factor=factor,
      )
      if self._fault_tolerance_manager.config.enabled and targets:
        slot_counts = sorted(
            _worker_staging_slots(info, worst_case_bytes, factor)
            for info, _ in targets
        )
        ratio = self._fault_tolerance_manager.config.min_active_rollout_ratio
        n_workers = len(slot_counts)
        if ratio is None:
          k_min = 1
        else:
          k_min = next(
              (k for k in range(1, n_workers + 1) if k / n_workers >= ratio),
              n_workers,
          )
        min_surviving_slots = sum(slot_counts[:k_min])
        if min_surviving_slots < required_slots:
          logging.warning(
              "Fault tolerance allows shrinking to %d/%d active rollout"
              " worker(s), which provides at most %d worst-case staging slots"
              " (< required %d). Surviving workers may block in"
              " RolloutStagingBuffer.reserve() until replacement workers rejoin"
              " or min_active_rollout_grace_s (%.1fs) elapses; consider"
              " increasing RolloutConfig.staging_max_bytes or"
              " staging_overcommit_factor.",
              k_min,
              n_workers,
              min_surviving_slots,
              required_slots,
              self._fault_tolerance_manager.config.min_active_rollout_grace_s,
          )
    self._staging_worst_case_bytes = worst_case_bytes
    self._required_staging_slots = required_slots
    self._staging_overcommit_factor = factor

    def _count_staging_slots(
        active_targets: Sequence[
            tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]
        ],
    ) -> int:
      return sum(
          _worker_staging_slots(info, worst_case_bytes, factor)
          for info, _ in active_targets
      )

    self._fault_tolerance_manager.set_staging_slot_floor(
        _count_staging_slots if required_slots > 0 else None,
        required_slots,
    )

  async def fetch_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> rl_engine_interface.RoutedExpertsFetchResult:
    """Pulls staged `routed_experts` arrays concurrently per worker instance."""
    if not refs:
      return rl_engine_interface.RoutedExpertsFetchResult(
          fetched={}, failed_refs=frozenset()
      )
    ft_enabled = self._fault_tolerance_manager.config.enabled
    by_instance = _group_refs_by_worker_instance(dict.fromkeys(refs))
    targets_by_wid = self._rollout_targets_by_wid()
    failed: list[datatypes.RoutedExpertsRef] = []
    worker_calls: list[
        tuple[
            str,
            str,
            remote_execution.ActorHandle,
            list[datatypes.RoutedExpertsRef],
        ]
    ] = []
    for (wid, ref_inst), group_refs in by_instance.items():
      handle, err_msg = _resolve_active_staging_handle(
          wid, ref_inst, targets_by_wid
      )
      if handle is None:
        if not ft_enabled:
          raise RuntimeError(err_msg)
        logging.warning(
            "%s Marking %d staged ref(s) failed.", err_msg, len(group_refs)
        )
        failed.extend(group_refs)
        continue
      worker_calls.append((wid, ref_inst, handle, group_refs))

    fetched: dict[datatypes.RoutedExpertsRef, np.ndarray] = {}
    if worker_calls:
      results = await asyncio.gather(*(
          self._invoke_worker_ft_tolerant(
              handle,
              "fetch_staged",
              skip_msg=(
                  f"Rollout worker {wid!r} failed during fetch_staged;"
                  f" marking {len(group_refs)} ref(s) failed"
              ),
              ref_ids=[r.ref_id for r in group_refs],
              expected_instance_id=ref_inst,
          )
          for wid, ref_inst, handle, group_refs in worker_calls
      ))
      for (wid, ref_inst, _, group_refs), (ok, worker_map) in zip(
          worker_calls, results, strict=True
      ):
        if not ok:
          failed.extend(group_refs)
          continue
        if worker_map is None:
          if not ft_enabled:
            raise RuntimeError(
                f"Rollout worker {wid!r} rejected stale instance_id for"
                f" {len(group_refs)} staged ref(s)."
            )
          logging.warning(
              "Rollout worker %r (expected instance_id=%r) rejected stale"
              " instance_id; marking %d staged ref(s) failed.",
              wid,
              ref_inst,
              len(group_refs),
          )
          failed.extend(group_refs)
          continue
        for ref in group_refs:
          if ref.ref_id not in worker_map:
            raise KeyError(
                f"Rollout worker {wid!r} fetch_staged omitted"
                f" ref_id={ref.ref_id}."
            )
          fetched[ref] = worker_map[ref.ref_id]

    return rl_engine_interface.RoutedExpertsFetchResult(
        fetched=fetched,
        failed_refs=frozenset(failed),
    )

  async def _invoke_worker_ft_tolerant(
      self,
      handle: remote_execution.ActorHandle,
      method_name: str,
      *,
      skip_msg: str,
      **kwargs: Any,
  ) -> tuple[bool, Any]:
    """Invokes `method_name`, returning `(False, None)` on FT transport faults."""
    try:
      return True, await self._invoke_worker(handle, method_name, **kwargs)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      if not self._fault_tolerance_manager.config.enabled or not (
          remote_execution.is_transport_error(exc, handle=handle)
      ):
        raise
      logging.warning("%s (%s)", skip_msg, exc)
      return False, None

  async def release_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> None:
    """Releases staged `routed_experts` entries grouped by worker instance."""
    if not refs:
      return
    by_instance = _group_refs_by_worker_instance(dict.fromkeys(refs))
    for (wid, ref_inst), group_refs in by_instance.items():
      held = self._unreleased_staged_refs[(wid, ref_inst)]
      if held < len(group_refs):
        raise RuntimeError(
            f"Cannot release {len(group_refs)} staged routed_experts ref(s)"
            f" on worker {wid!r} (instance_id={ref_inst!r}, e.g."
            f" ref_id={group_refs[0].ref_id}): only {held} unreleased ref(s)"
            " tracked."
        )
    for key, group_refs in by_instance.items():
      remaining = self._unreleased_staged_refs[key] - len(group_refs)
      if remaining > 0:
        self._unreleased_staged_refs[key] = remaining
      else:
        del self._unreleased_staged_refs[key]

    targets_by_wid = self._rollout_targets_by_wid()
    calls: list[tuple[str, remote_execution.ActorHandle, list[int]]] = []
    for (wid, ref_inst), group_refs in by_instance.items():
      ref_ids = [r.ref_id for r in group_refs]
      handle, _ = _resolve_active_staging_handle(wid, ref_inst, targets_by_wid)
      if handle is None:
        self._evicted_release_skips_total += len(ref_ids)
        logging.warning(
            "Skipping release_staged for %d ref(s) on evicted or replaced"
            " rollout worker %r (instance_id=%r,"
            " evicted_release_skips_total=%d).",
            len(ref_ids),
            wid,
            ref_inst,
            self._evicted_release_skips_total,
        )
        continue
      calls.append((wid, handle, ref_ids))

    if calls:
      results = await asyncio.gather(*(
          self._invoke_worker_ft_tolerant(
              handle,
              "release_staged",
              skip_msg=(
                  f"Skipping release_staged for {len(ref_ids)} ref(s) on"
                  f" unreachable rollout worker {wid!r}"
              ),
              ref_ids=ref_ids,
          )
          for wid, handle, ref_ids in calls
      ))
      for (_, _, ref_ids), (ok, _) in zip(calls, results, strict=True):
        if not ok:
          self._evicted_release_skips_total += len(ref_ids)

  async def clear_all_staged_routed_experts(self) -> None:
    """Abandons in-flight rollouts and clears staged routing on workers."""
    self._rollout_epoch += 1
    await self._rollout_session.abandon_in_flight()
    self._unreleased_staged_refs.clear()
    self._staging_worst_case_bytes = None
    self._required_staging_slots = 0
    self._staging_overcommit_factor = 1.0
    self._fault_tolerance_manager.set_staging_slot_floor(None, 0)
    targets = self._registry.targets(
        datatypes.Role.ROLLOUT, statuses=_WEIGHT_SYNC_DESTINATION_STATUSES
    )
    if targets:
      await asyncio.gather(*(
          self._invoke_worker_ft_tolerant(
              handle,
              "clear_staged",
              skip_msg=(
                  "Skipping clear_staged on unreachable rollout worker"
                  f" {info.worker_id!r}"
              ),
          )
          for info, handle in targets
      ))

  async def sync_weights(  # pyrefly: ignore[bad-override]
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
      policy_version: int | None = None,
      *,
      min_retained_policy_version: int | None = None,
      source_prepared: asyncio.Event | None = None,
  ) -> int:
    """Runs one weight sync round through the coordinator."""
    del role, target_roles
    if self._weight_sync_coordinator is None:
      raise RuntimeError(
          "sync_weights needs a coordinator; construct the engine with"
          " weight_sync_coordinator."
      )
    next_policy_version = (
        self._policy_version + 1 if policy_version is None else policy_version
    )
    # TODO(b/577214994): Wait up to min_active_rollout_grace_s when fault
    # tolerance is enabled and zero rollout destinations are currently in
    # ACTIVE or PENDING_WEIGHT_SYNC.
    logging.info(
        "Synchronizing weights (target policy_version=%d)...",
        next_policy_version,
    )
    result = await self._weight_sync_coordinator.sync(
        policy_version=next_policy_version,
        destination_statuses=_WEIGHT_SYNC_DESTINATION_STATUSES,
        min_retained_policy_version=min_retained_policy_version,
        source_prepared=source_prepared,
    )
    for r in result.workers:
      if r.phase != "committed":
        continue
      if self._registry.set_status(
          r.worker_id,
          worker_registry.AdmissionStatus.ACTIVE,
          expected_handle=r.handle,
          expected_status=worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
      ):
        self._fault_tolerance_manager.record_rejoin_promoted()
        logging.info(
            "Promoted synced rollout worker %r to %s.",
            r.worker_id,
            worker_registry.AdmissionStatus.ACTIVE.value,
        )
    self._fault_tolerance_manager.check_rollout_capacity()
    self._policy_version = result.policy_version
    logging.info(
        "Weight synchronization complete (policy_version=%d).",
        self._policy_version,
    )
    return result.policy_version

  def fault_tolerance_metrics(self) -> dict[str, float]:
    """Returns rollout fleet gauges and cumulative fault-tolerance counters."""
    metrics = self._fault_tolerance_manager.metrics(
        rollout_session=self._rollout_session
    )
    metrics["ft/rollout/evicted_release_skips_total"] = float(
        self._evicted_release_skips_total
    )
    return metrics

  async def save_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> Any:
    """Requests the trainer worker(s) for `role` to save a checkpoint."""
    workers = self._require_role_handles(role)
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Saving checkpoint on %d %s worker(s)...",
        len(workers),
        role_name,
    )
    return await self._invoke_role_workers(
        workers, "save_checkpoint", metadata=metadata, step=step, force=force
    )

  async def _restore_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      step: int | None = None,
  ) -> Any:
    role_name = role.name
    workers = self._require_role_handles(
        role,
        missing_error=f"No trainer worker registered for role {role_name}",
    )
    results = await self._invoke_role_workers(
        workers, "restore_checkpoint", return_all=True, step=step
    )
    if len(results) > 1:
      signatures = [_restored_step_signature(res) for res in results]
      if len(set(signatures)) != 1:
        raise RuntimeError(
            f"SPMD {role_name} workers disagreed on restored checkpoint step:"
            f" {signatures}."
        )
    return results[0]

  async def resume_from_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
  ) -> int:
    """Restores the trainer checkpoint for `role` and aligns engine policy state.

    Args:
      role: Trainer role whose checkpoint should be restored.

    Returns:
      The restored step, or 0 when starting fresh.
    """
    self._restored_checkpoint_metadata = None
    self._inflight_fwd_bwd[role].clear()
    metadata = await self._restore_checkpoint(role=role)
    if metadata is None:
      return 0
    # A malformed result could hide a real checkpoint; never train from the
    # base weights on a guess.
    if not isinstance(metadata, Mapping):
      raise TypeError(
          f"restore_checkpoint returned {type(metadata).__name__}, not a"
          " mapping; cannot tell whether a checkpoint was restored."
      )
    metadata = dict(metadata)
    try:
      restored_optimizer_step = int(metadata.get("step", 0) or 0)
      restored_step = int(
          metadata.get("global_step", restored_optimizer_step) or 0
      )
    except (TypeError, ValueError) as exc:
      raise ValueError(
          "restore_checkpoint returned a non-integer step"
          f" {metadata.get('global_step', metadata.get('step'))!r}."
      ) from exc
    if restored_step <= 0:
      logging.info("No checkpoint to resume from; starting from step 0.")
      return 0

    self._restored_checkpoint_metadata = dict(metadata)
    recorded_version = metadata.get("policy_version")
    has_off_policy_state = (
        "committed_prompt_ids" in metadata or "skipped_prompt_ids" in metadata
    )
    if has_off_policy_state and recorded_version is not None:
      restored_policy_version = int(recorded_version)
    else:
      # Resume at the step boundary; the policy version tracks the restored
      # step. New checkpoints record optimizer and global steps separately.
      # Legacy checkpoints have only `step`, for which both values are
      # identical.
      restored_policy_version = restored_step
      if recorded_version is not None and recorded_version != restored_step:
        logging.warning(
            "Checkpoint recorded mid-step policy_version=%s; resuming at the"
            " step-boundary value %d",
            recorded_version,
            restored_step,
        )
    self._policy_version = restored_policy_version
    logging.info(
        "Resuming from checkpoint: global_step=%d optimizer_step=%d "
        "policy_version=%d. Metadata: %s",
        restored_step,
        restored_optimizer_step,
        restored_policy_version,
        metadata,
    )
    return restored_step

  async def close(self) -> None:
    """Closes the rollout execution session and cancels any pending polling tasks."""
    self._inflight_fwd_bwd.clear()
    self._fault_tolerance_manager.set_staging_slot_floor(None, 0)
    await self._rollout_session.close()
