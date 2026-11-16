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
import inspect
import itertools
import math
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


_UNREACHABLE_WORKER_ERRORS = (
    remote_execution.TransientTransportError,
    ConnectionError,
    TimeoutError,
    KeyError,
)


def _group_refs_by_worker(
    refs: Iterable[datatypes.RoutedExpertsRef],
) -> dict[str, list[int]]:
  """Groups `RoutedExpertsRef`s into `worker_id -> [ref_id, ...]`."""
  by_worker: dict[str, list[int]] = {}
  for ref in refs:
    by_worker.setdefault(ref.worker_id, []).append(ref.ref_id)
  return by_worker


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
    self._unreleased_staged_refs: collections.Counter[str] = (
        collections.Counter()
    )
    self._staging_worst_case_bytes: int | None = None
    self._required_staging_slots: int = 0
    self._staging_overcommit_factor: float = 1.0
    self._evicted_release_skips_total: int = 0
    self._rollout_epoch: int = 0

  def _rollout_handles(self) -> list[remote_execution.ActorHandle]:
    return self._registry.handles(datatypes.Role.ROLLOUT)

  def _role_handle(
      self, role: datatypes.Role | str
  ) -> remote_execution.ActorHandle | None:
    handles = self._registry.handles(role)
    return handles[0] if handles else None

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
    trainer = self._role_handle(role)
    # Match the weight sync destinations: a rejoined rollout worker still in
    # PENDING_WEIGHT_SYNC can seed the trainer before its first sync.
    rollout_handles = self._registry.handles(
        datatypes.Role.ROLLOUT, statuses=_WEIGHT_SYNC_DESTINATION_STATUSES
    )
    if trainer is None or not rollout_handles:
      return

    rollout = rollout_handles[0]
    try:
      target_state = await self._invoke_worker(rollout, "get_target_state")
      await self._invoke_worker(
          trainer, "set_target_state", target_state=target_state
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
          self._unreleased_staged_refs[info.worker_id] if stage_any else 0
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
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.RolloutRequest]:
    """Validates prompts and constructs typed RolloutRequests with lineage attached."""
    base_metadata = {
        **(route_metadata or {}),
        **(kwargs.get("metadata") or {}),
    }
    base_generation_kwargs = (
        generation_args.as_kwargs() if generation_args else {}
    )
    version = kwargs.get("policy_version", policy_version)

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
        for k, v in base_generation_kwargs.items():
          p.generation_kwargs.setdefault(k, v)
        rollout_reqs.append(p)
        req_task_names.append("")
        continue

      item_metadata = dict(getattr(p, "metadata", {}) or {})
      item_generation_kwargs = dict(getattr(p, "generation_kwargs", {}) or {})
      if isinstance(p, Mapping):
        item_metadata.update(dict(p.get("metadata", {}) or {}))
        item_generation_kwargs.update(
            dict(p.get("generation_kwargs", {}) or {})
        )

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

        generation_kwargs = dict(base_generation_kwargs)
        generation_kwargs.update(item_generation_kwargs)

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
                generation_kwargs=generation_kwargs,
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
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      stage_routed_experts: bool = False,
      max_response_length: int | None = None,
      **kwargs: Any,
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
        generation_args=generation_args,
        route_metadata=route_metadata,
        **kwargs,
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
        if traj_item.routed_experts_ref is not None:
          self._unreleased_staged_refs[
              traj_item.routed_experts_ref.worker_id
          ] += 1
        logging.debug(
            "Received rollout response (prompt_id=%s, group_index=%d).",
            traj_item.prompt_id,
            traj_item.group_index,
        )
        completed.append(traj_item)
    self._fault_tolerance_manager.check_rollout_capacity()
    self._check_staging_liveness()
    return completed

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    rollout_handles = self._rollout_handles()
    if not rollout_handles:
      raise ValueError("DistributedRLEngine has no registered rollout workers.")

    if kwargs:
      raise TypeError(
          "Unexpected generate kwargs: "
          f"{sorted(kwargs)}. Use generation_args=GenerationArgs(...) for "
          "sampling parameters."
      )

    logging.info(
        "Generating rollouts for %d prompt(s)/request(s) across %d"
        " worker(s)...",
        len(prompts),
        len(rollout_handles),
    )
    requests = self._build_rollout_requests(
        prompts,
        policy_version=self._policy_version,
        generation_args=generation_args,
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
    worker = self._role_handle(role)
    if worker is None:
      raise ValueError(
          f"No worker registered for per_token_logps with role {role}"
      )
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Evaluating per-token log probabilities on %s worker...",
        role_name,
    )
    return await self._invoke_worker(
        worker, "per_token_logps", items=items, **kwargs
    )

  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
  ) -> Any:
    """Executes atomic gradient accumulation / update on TrainerWorker."""
    worker = self._role_handle(role)
    if worker is None:
      raise ValueError(f"No trainer worker registered for role {role}")
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Executing train_step on %s worker (accumulate_gradients=%s,"
        " apply_optimizer=%s)...",
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
      fwd_bwd_result = await self._invoke_worker(
          worker, "fwd_bwd", request=request
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
    train_step, metrics = await self._invoke_worker(worker, "update")
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
    """Retrieves step metrics from the worker(s) registered for the specified role."""
    if role == datatypes.Role.ROLLOUT:
      rollout_handles = self._rollout_handles()
      if not rollout_handles:
        raise ValueError(f"No rollout workers registered for role {role}")
      tasks = [
          self._invoke_worker(w, "get_metrics", **kwargs)
          for w in rollout_handles
      ]
      results = await asyncio.gather(*tasks, return_exceptions=True)
      return [  # pyrefly: ignore[bad-return]
          r for r in results if not isinstance(r, Exception) and r is not None
      ]
    else:
      worker = self._role_handle(role)
      if worker is None:
        raise ValueError(f"No worker registered for role {role}")
      return await self._invoke_worker(worker, "get_metrics", **kwargs)

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
        worker = self._role_handle(role)
        if worker is None:
          raise ValueError(f"No trainer worker registered for role {role_name}")
        logging.info(
            "Auto-configuring trainer loss and model input fn on %s worker...",
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

        def _configure():
          assert worker is not None
          # Both setters clear the trainer's JIT cache, so the ahead-of-time
          # compile must run strictly after them.
          worker.submit("with_loss_fn", loss_fn, has_aux=True)
          worker.submit("with_gen_model_input_fn", gen_fn)
          if not precompile:
            return
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
          # TODO(tunix-dev): Move to the worker lifecycle (e.g., the
          # `initialize` step) so pre-compilation runs during the standard
          # lifecycle `compile` phase instead of re-compiling here.
          logging.info(
              "Precompiling trainer step executables on %s worker...",
              role_name,
          )
          worker.submit("compile", dummy_payload)

        try:
          loop = asyncio.get_running_loop()
        except RuntimeError:
          loop = None

        if loop is not None and loop.is_running():
          with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_configure).result()
        else:
          _configure()

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
    trainer = self._role_handle(role)
    if trainer is None:
      raise ValueError(f"No trainer worker registered for role {role}")

    await self._maybe_configure_trainer_target_state(role)

    if not sync_weights:
      return None
    target_policy_version = (
        self._policy_version if policy_version is None else policy_version
    )
    return await self.sync_weights(
        role=role, policy_version=target_policy_version
    )

  def _check_staging_liveness(self) -> None:
    """Verifies active rollout workers satisfy the installed staging bound."""
    if (
        self._staging_worst_case_bytes is None
        or self._required_staging_slots <= 0
    ):
      return
    targets = self._registry.targets(
        datatypes.Role.ROLLOUT,
        statuses=(worker_registry.AdmissionStatus.ACTIVE,),
    )
    _validate_staging_slots(
        targets,
        worst_case_bytes=self._staging_worst_case_bytes,
        required_slots=self._required_staging_slots,
        overcommit_factor=self._staging_overcommit_factor,
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
    if self._fault_tolerance_manager.config.enabled:
      logging.warning(
          "Orchestrator Relay does not preserve rollout-worker fault tolerance:"
          " staged routed_experts reside in rollout worker RAM, so evicting a"
          " rollout worker with pending refs will fail fetch_routed_experts."
      )
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
    self._staging_worst_case_bytes = worst_case_bytes
    self._required_staging_slots = required_slots
    self._staging_overcommit_factor = factor

  async def fetch_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> dict[datatypes.RoutedExpertsRef, np.ndarray]:
    """Pulls staged `routed_experts` arrays concurrently per `worker_id`.

    Note: Staged entries reside in rollout worker RAM; if a worker is evicted
    before its refs are fetched, this fails fast with `RuntimeError`.
    """
    if not refs:
      return {}
    by_worker = _group_refs_by_worker(dict.fromkeys(refs))
    worker_calls: list[tuple[str, remote_execution.ActorHandle, list[int]]] = []
    for wid, ref_ids in by_worker.items():
      try:
        handle = self._registry.handle(wid)
      except KeyError as exc:
        raise RuntimeError(
            f"Rollout worker {wid!r} was evicted before staged routed_experts"
            " could be fetched."
        ) from exc
      worker_calls.append((wid, handle, ref_ids))
    responses = await asyncio.gather(*(
        self._invoke_worker(handle, "fetch_staged", ref_ids=ref_ids)
        for _, handle, ref_ids in worker_calls
    ))
    ref_by_key = {(r.worker_id, r.ref_id): r for r in refs}
    fetched: dict[datatypes.RoutedExpertsRef, np.ndarray] = {}
    for (wid, _, ref_ids), worker_map in zip(
        worker_calls, responses, strict=True
    ):
      for rid in ref_ids:
        if rid not in worker_map:
          raise KeyError(
              f"Rollout worker {wid!r} fetch_staged omitted ref_id={rid}."
          )
        fetched[ref_by_key[(wid, rid)]] = worker_map[rid]
    return fetched

  async def _invoke_worker_ft_tolerant(
      self,
      handle: remote_execution.ActorHandle,
      method_name: str,
      *,
      skip_msg: str,
      **kwargs: Any,
  ) -> bool:
    """Invokes `method_name`, skipping unreachable workers when FT is enabled."""
    try:
      await self._invoke_worker(handle, method_name, **kwargs)
      return True
    except _UNREACHABLE_WORKER_ERRORS as exc:
      if not self._fault_tolerance_manager.config.enabled:
        raise
      logging.warning("%s (%s)", skip_msg, exc)
      return False

  async def release_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> None:
    """Releases staged `routed_experts` entries grouped by `worker_id`."""
    if not refs:
      return
    by_worker = _group_refs_by_worker(dict.fromkeys(refs))
    for wid, ref_ids in by_worker.items():
      held = self._unreleased_staged_refs[wid]
      if held < len(ref_ids):
        raise RuntimeError(
            f"Cannot release {len(ref_ids)} staged routed_experts ref(s) on"
            f" worker {wid!r} (e.g. ref_id={ref_ids[0]}): only {held}"
            " unreleased ref(s) tracked."
        )
    for wid, ref_ids in by_worker.items():
      remaining = self._unreleased_staged_refs[wid] - len(ref_ids)
      if remaining > 0:
        self._unreleased_staged_refs[wid] = remaining
      else:
        del self._unreleased_staged_refs[wid]
    calls: list[tuple[str, remote_execution.ActorHandle, list[int]]] = []
    for wid, ref_ids in by_worker.items():
      try:
        handle = self._registry.handle(wid)
      except KeyError:
        self._evicted_release_skips_total += len(ref_ids)
        logging.warning(
            "Skipping release_staged for %d ref(s) on evicted rollout worker"
            " %r (evicted_release_skips_total=%d).",
            len(ref_ids),
            wid,
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
      for (_, _, ref_ids), ok in zip(calls, results, strict=True):
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
    handles = self._registry.handles(
        datatypes.Role.ROLLOUT, statuses=_WEIGHT_SYNC_DESTINATION_STATUSES
    )
    if handles:
      await asyncio.gather(*(
          self._invoke_worker_ft_tolerant(
              handle,
              "clear_staged",
              skip_msg="Skipping clear_staged on unreachable rollout worker",
          )
          for handle in handles
      ))

  async def sync_weights(  # pyrefly: ignore[bad-override]
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
      policy_version: int | None = None,
      *,
      min_retained_policy_version: int | None = None,
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
    logging.info(
        "Synchronizing weights (target policy_version=%d)...",
        next_policy_version,
    )
    result = await self._weight_sync_coordinator.sync(
        policy_version=next_policy_version,
        destination_statuses=_WEIGHT_SYNC_DESTINATION_STATUSES,
        min_retained_policy_version=min_retained_policy_version,
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
    self._check_staging_liveness()
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
    """Requests the trainer worker for `role` to save a checkpoint."""
    worker = self._role_handle(role)
    if worker is None:
      raise ValueError(f"No trainer worker registered for role {role}")
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Saving checkpoint on %s worker...",
        role_name,
    )
    return await self._invoke_worker(
        worker, "save_checkpoint", metadata=metadata, step=step, force=force
    )

  async def _restore_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      step: int | None = None,
  ) -> Any:
    role_name = role.name
    worker = self._role_handle(role)
    if worker is None:
      raise ValueError(f"No trainer worker registered for role {role_name}")
    return await self._invoke_worker(worker, "restore_checkpoint", step=step)

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
    if not isinstance(metadata, Mapping):
      if metadata is not None:
        logging.warning(
            "restore_checkpoint returned %s, not a mapping; starting from"
            " fresh run.",
            type(metadata).__name__,
        )
      return 0
    metadata = dict(metadata)
    try:
      restored_optimizer_step = int(metadata.get("step", 0) or 0)
      restored_step = int(
          metadata.get("global_step", restored_optimizer_step) or 0
      )
    except (TypeError, ValueError):
      logging.warning(
          "restore_checkpoint returned a non-integer step %r; starting from"
          " fresh run.",
          metadata.get("global_step", metadata.get("step")),
      )
      return 0
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
    await self._rollout_session.close()
