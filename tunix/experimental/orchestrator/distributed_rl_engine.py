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
import collections
from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import inspect
import time
from typing import Any
import uuid

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import lineage
from tunix.experimental.common import logging_utils
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.weight_sync import weight_sync_coordinator as weight_sync_coordinator_lib
from tunix.experimental.worker import remote_execution

_summarize_list = logging_utils.summarize_list


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
    policy_version = int(metadata.get("policy_version", 0) or 0)
    metadata["error"] = str(resp.error)
    return datatypes.TrajectoryItem(
        prompt_id=prompt_id,
        group_index=group_index,
        policy_version=policy_version,
        traj={
            "status": datatypes.TrajectoryStatus.FAILED,
            "trajectory_reward": 0.0,
            "prompt_tokens": np.zeros(0, dtype=np.int32),
            "conversation_tokens": np.zeros(0, dtype=np.int32),
            "conversation_masks": np.zeros(0, dtype=np.float32),
            "old_logprobs": np.zeros(0, dtype=np.float32),
        },
        metadata=metadata,
    )

  raise ValueError("RolloutResponse payload is None.")


class DistributedRLEngine(rl_engine_interface.AbstractRLEngine):
  """Worker-backed compute router dispatching RPCs across role pools."""

  def __init__(
      self,
      rollout_workers: Sequence[remote_execution.ActorHandle],
      trainer_workers: Mapping[datatypes.Role, remote_execution.ActorHandle],
      inference_workers: (
          Mapping[datatypes.Role, remote_execution.ActorHandle] | None
      ) = None,
      weight_sync_coordinator: Any = None,
      on_worker_evicted: (
          Callable[[remote_execution.ActorHandle, BaseException | None], None]
          | None
      ) = None,
      max_concurrent_rollouts_per_worker: int | None = None,
      rollout_worker_capacities: (
          Mapping[remote_execution.ActorHandle, int] | None
      ) = None,
      rollout_task_timeout_s: float | None = None,
      fault_tolerance_config: datatypes.RolloutFaultToleranceConfig | None = (
          None
      ),
      max_zero_worker_wait_s: float | None = None,
  ):
    # Legacy scalar arguments take precedence over the consolidated config;
    # `with_overrides` validates the merged result.
    ft_cfg = (
        fault_tolerance_config or datatypes.RolloutFaultToleranceConfig()
    ).with_overrides(
        max_in_flight_per_worker=max_concurrent_rollouts_per_worker,
        task_timeout_s=rollout_task_timeout_s,
        max_zero_worker_wait_s=max_zero_worker_wait_s,
    )
    self._fault_tolerance_config = ft_cfg
    self._max_zero_worker_wait_s = ft_cfg.max_zero_worker_wait_s
    self._zero_worker_since: float | None = None
    self._zero_worker_seconds_total: float = 0.0
    self._terminal_failed_trajectories_total: int = 0
    self._weights_consistent: bool = True
    self._pending_sync_lock: asyncio.Lock = asyncio.Lock()
    self._actor_busy_count: int = 0
    self._actor_weights_dirty: bool = False

    self._trainer_workers = dict(trainer_workers)
    self._inference_workers = dict(inference_workers or {})
    self._policy_version = 0
    self._restored_next_batch_idx = 0
    self._weight_sync_coordinator = weight_sync_coordinator
    self._rollout_pool = remote_execution.RoutingActorPool(
        list(rollout_workers)
    )
    # Least-loaded, not hash-by-traj_id: each rollout request is one whole
    # episode, and episode lengths vary by 10x, so hashing left some workers
    # with twice the episodes of others and stretched the batch tail.
    self._rollout_session = remote_execution.PoolExecutionSession(
        self._rollout_pool,
        config=remote_execution.PoolSessionConfig(
            evict_on_failure=ft_cfg.enabled and ft_cfg.evict_on_failure,
            retry_on_worker_failure=(
                ft_cfg.enabled and ft_cfg.retry_on_worker_failure
            ),
            max_task_retries=ft_cfg.max_task_retries,
            on_worker_evicted=on_worker_evicted,
            max_in_flight_per_worker=ft_cfg.max_in_flight_per_worker,
            worker_max_in_flight=rollout_worker_capacities,
            has_pending_workers_fn=self._has_pending_rollout_workers,
            task_timeout_s=ft_cfg.task_timeout_s,
            retain_pending_on_zero_workers=(
                ft_cfg.enabled and ft_cfg.max_zero_worker_wait_s > 0
            ),
        ),
        least_loaded=True,
    )

  def _has_pending_rollout_workers(self) -> bool:
    if self._weight_sync_coordinator is None:
      return False
    return bool(self._weight_sync_coordinator.has_pending_destinations())

  @property
  def _rollout_workers(self) -> list[remote_execution.ActorHandle]:
    return self._rollout_pool.actors

  @property
  def policy_version(self) -> int:
    return self._policy_version

  @property
  def weight_sync_coordinator(self) -> Any:
    return self._weight_sync_coordinator

  @property
  def fault_tolerance_config(self) -> datatypes.RolloutFaultToleranceConfig:
    return self._fault_tolerance_config

  @property
  def weights_consistent(self) -> bool:
    return self._weights_consistent

  @property
  def max_zero_worker_wait_s(self) -> float:
    return self._max_zero_worker_wait_s

  def set_max_zero_worker_wait_s(self, wait_s: float) -> None:
    """Updates the bounded wait timeout when zero rollout workers are active."""
    if wait_s < 0:
      raise ValueError("max_zero_worker_wait_s must be non-negative")
    self._max_zero_worker_wait_s = float(wait_s)
    self._rollout_session.set_retain_pending_on_zero_workers(
        self._fault_tolerance_config.enabled
        and self._max_zero_worker_wait_s > 0
    )

  def _current_zero_worker_seconds(self) -> float:
    total = self._zero_worker_seconds_total
    if self._zero_worker_since is not None:
      total += max(0.0, time.monotonic() - self._zero_worker_since)
    return total

  @property
  def fault_tolerance_metrics(self) -> dict[str, int | float]:
    """Returns minimal fault-tolerance observability counters and gauges."""
    return {
        "rollout_worker_evictions_total": self._rollout_session.evictions_total,
        "rollout_retries_total": self._rollout_session.retries_total,
        "terminal_failed_trajectories_total": (
            self._terminal_failed_trajectories_total
        ),
        "zero_worker_seconds": self._current_zero_worker_seconds(),
    }

  def add_rollout_worker(
      self,
      handle: remote_execution.ActorHandle,
      *,
      max_in_flight: int | None = None,
  ) -> None:
    """Adds a rollout worker handle to the active pool and execution session."""
    self._rollout_session.add_actor(handle, max_in_flight=max_in_flight)

  def remove_rollout_worker(
      self,
      handle: remote_execution.ActorHandle,
      exc: BaseException | None = None,
  ) -> bool:
    """Evicts a rollout worker handle and re-queues any in-flight tasks.

    Returns:
      True if `handle` was still a member of the rollout pool, False if it had
      already been evicted (the call is then a no-op).
    """
    return self._rollout_session.remove_actor(handle, exc=exc)

  @property
  def restored_next_batch_idx(self) -> int:
    """First untrained prompt-batch index from the last restored checkpoint."""
    return self._restored_next_batch_idx

  @property
  def max_concurrent_rollouts_per_worker(self) -> int | None:
    return self._rollout_session.max_in_flight_per_worker

  @property
  def rollout_task_timeout_s(self) -> float | None:
    return self._rollout_session.task_timeout_s

  def set_rollout_task_timeout_s(
      self, rollout_task_timeout_s: float | None
  ) -> None:
    """Updates the per-task execution timeout for in-flight rollout requests."""
    self._rollout_session.set_task_timeout_s(rollout_task_timeout_s)

  def _assert_weights_consistent(self) -> None:
    """Raises RuntimeError if rollout weights are inconsistent or coordinator is poisoned."""
    if not self._weights_consistent:
      raise RuntimeError(
          "Cannot dispatch rollout requests while rollout policy weights are"
          " inconsistent after an uncommitted weight sync round."
      )
    poisoned = (
        self._weight_sync_coordinator.poisoned
        if self._weight_sync_coordinator is not None
        else None
    )
    if isinstance(poisoned, str) and poisoned:
      raise RuntimeError(
          "Cannot dispatch rollout requests while WeightSyncCoordinator is"
          " poisoned."
      )

  async def _wait_for_healthy_rollout_worker(
      self, poll_interval_s: float = 0.02
  ) -> None:
    """Waits up to `max_zero_worker_wait_s` for at least one healthy rollout worker."""
    self._assert_weights_consistent()
    await self.sync_pending_weights()
    if self._rollout_workers:
      if self._zero_worker_since is not None:
        self._zero_worker_seconds_total += max(
            0.0, time.monotonic() - self._zero_worker_since
        )
        self._zero_worker_since = None
      return

    if not self._fault_tolerance_config.enabled:
      raise datatypes.NoHealthyRolloutWorkersError(
          "No active rollout workers registered on DistributedRLEngine."
      )

    wait_start = time.monotonic()
    if self._zero_worker_since is None:
      self._zero_worker_since = wait_start
      logging.warning(
          "[rollout-ft] action=zero_workers_wait max_zero_worker_wait_s=%.2f",
          self._max_zero_worker_wait_s,
      )

    try:
      while not self._rollout_workers:
        self._assert_weights_consistent()
        await self.sync_pending_weights()
        if self._rollout_workers:
          break
        elapsed = time.monotonic() - wait_start
        if elapsed >= self._max_zero_worker_wait_s:
          raise datatypes.NoHealthyRolloutWorkersError(
              "No healthy rollout workers available after waiting"
              f" {elapsed:.2f}s"
              f" (max_zero_worker_wait_s={self._max_zero_worker_wait_s}s)."
          )
        sleep_s = min(
            poll_interval_s, max(0.001, self._max_zero_worker_wait_s - elapsed)
        )
        await asyncio.sleep(sleep_s)
    finally:
      if self._zero_worker_since is not None:
        self._zero_worker_seconds_total += max(
            0.0, time.monotonic() - self._zero_worker_since
        )
        self._zero_worker_since = None

  async def _maybe_configure_trainer_target_state(
      self,
      role: datatypes.Role,
  ) -> None:
    """Seeds trainer-side weight sync with the rollout target-state skeleton."""
    trainer = self._trainer_workers.get(role)
    if trainer is None or not self._rollout_workers:
      return

    rollout = self._rollout_workers[0]
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
    """Dispatches pre-formed RolloutRequests across rollout workers."""
    self._assert_weights_consistent()
    await self._wait_for_healthy_rollout_worker()
    requests = self._build_rollout_requests(requests)
    logging.info(
        "Dispatching %d rollout request(s) across %d worker(s).",
        len(requests),
        len(self._rollout_workers),
    )
    for req in requests:
      logging.debug(
          "Dispatched rollout request (prompt_id=%s, group_index=%d,"
          " request_id=%s).",
          req.prompt_id,
          req.group_index,
          req.request_id,
      )
    await asyncio.gather(*(
        self._rollout_session.submit(
            req.request_id,
            "generate",
            requests=[req],
            route_key=req.traj_id,
        )
        for req in requests
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
      priority: int = 0,
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
    for idx, p in enumerate(prompts):
      if isinstance(p, datatypes.RolloutRequest):
        if p.prompt_id is None or p.prompt_id == "":
          raise ValueError(
              f"RolloutRequest at index {idx} (id='{p.request_id}') lacks"
              " 'prompt_id'. Every request must provide a non-empty"
              " 'prompt_id'."
          )
        rollout_reqs.append(p)
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

        rollout_reqs.append(
            datatypes.RolloutRequest(
                request_id=f"req_{prompt_id}_g{group_index}_v{version}",
                prompt=raw_prompt,
                prompt_id=prompt_id,
                group_index=group_index,
                target_policy_version=version,
                generation_kwargs=generation_kwargs,
                max_turns=max_turns,
                max_response_length=max_response_length,
                exact_token_continuity=kwargs.get(
                    "exact_token_continuity", True
                ),
                priority=priority,
                metadata=request_metadata,
            )
        )

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
    for req in rollout_reqs:
      if req.metadata is None:  # pyrefly: ignore[comparison-with-never]
        req.metadata = {}  # pyrefly: ignore[bad-assignment]
      if req.metadata.get("lineage") is None:
        lineage_ctx = lineage.LineageContext(
            tracking_id=req.traj_id,
            parent_tracking_ids=[str(req.prompt_id)],
        )
        lineage_ctx.add_event(
            component="engine.dispatch",
            operation="rollout",
            attributes={
                "policy_version": req.target_policy_version,
                "group_index": req.group_index,
            },
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
      priority: int = 0,
      **kwargs: Any,
  ) -> list[str]:
    """Dispatches rollout requests across workers, constructing RolloutRequests internally.

    Every prompt item in `prompts` MUST have a unique, collision-free
    `prompt_id`
    attribute or dict key. Missing prompt IDs raise a ValueError.

    `priority` is stamped on every request built here; lower values are served
    first by samplers that schedule by priority.
    """
    rollout_reqs = self._build_rollout_requests(
        prompts,
        num_generations=num_generations,
        policy_version=policy_version,
        generation_args=generation_args,
        route_metadata=route_metadata,
        priority=priority,
        **kwargs,
    )
    return await self.dispatch_rollout_requests(rollout_reqs)

  def _drain_completed_and_failed(
      self, raw_completed: Sequence[tuple[Any, Exception | None]]
  ) -> list[datatypes.TrajectoryItem]:
    """Converts raw session completions and terminal failures into TrajectoryItems."""
    completed: list[datatypes.TrajectoryItem] = []
    for res, exc in raw_completed:
      if exc is not None:
        logging.error("Failed polling rollout worker: %s", exc)
        continue
      if res is None:
        continue
      items = res if isinstance(res, list) else [res]
      for it in items:
        if isinstance(it, dict):
          it = datatypes.RolloutResponse(**it)
        traj_item = _response_to_trajectory_item(it)
        traj_status = (
            traj_item.traj.get("status")
            if isinstance(traj_item.traj, Mapping)
            else None
        )
        if traj_status in (
            datatypes.TrajectoryStatus.FAILED,
            "FAILED",
        ):
          self._terminal_failed_trajectories_total += 1
        logging.debug(
            "Received rollout response (prompt_id=%s, group_index=%d).",
            traj_item.prompt_id,
            traj_item.group_index,
        )
        completed.append(traj_item)

    for _, payload, task_exc in self._rollout_session.pop_failed_tasks():
      # Rollout tasks are always submitted as generate(requests=[req]).
      _, _, orig_kwargs = payload
      for req in orig_kwargs["requests"]:
        self._terminal_failed_trajectories_total += 1
        logging.warning(
            "[rollout-ft] action=terminal_fail request_id=%s prompt_id=%s"
            " group_index=%d policy_version=%d reason=%r",
            req.request_id,
            req.prompt_id,
            req.group_index,
            req.target_policy_version,
            task_exc,
        )
        err_resp = datatypes.RolloutResponse(
            request_id=req.request_id,
            status="FAILED",
            error=datatypes.ErrorInfo(
                error_type=type(task_exc).__name__,
                message=str(task_exc),
            ),
            metadata={
                **(dict(req.metadata) if req.metadata is not None else {}),
                "prompt_id": req.prompt_id,
                "group_index": req.group_index,
                "policy_version": req.target_policy_version,
            },
        )
        completed.append(_response_to_trajectory_item(err_resp))
    return completed

  async def poll_rollouts(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> list[datatypes.TrajectoryItem]:
    """Concurrently long-polls completed rollout responses across all workers."""
    await self.sync_pending_weights()
    if (
        not self._rollout_workers
        and not self._rollout_session.has_pending_or_completed_work()
    ):
      return []

    raw_completed = await self._rollout_session.poll_completed(
        timeout_s=timeout_s
    )
    if (
        not raw_completed
        and self._rollout_session.pending_count > 0
        and self._has_pending_rollout_workers()
    ):
      await self.sync_pending_weights()
      raw_completed = await self._rollout_session.poll_completed(
          timeout_s=timeout_s
      )
    completed = self._drain_completed_and_failed(raw_completed)

    if (
        not completed
        and not self._rollout_workers
        and self._rollout_session.in_flight_count > 0
    ):
      await self._wait_for_healthy_rollout_worker()
      raw_completed = await self._rollout_session.poll_completed(
          timeout_s=timeout_s
      )
      completed = self._drain_completed_and_failed(raw_completed)

    if self._rollout_workers and self._zero_worker_since is not None:
      self._zero_worker_seconds_total += max(
          0.0, time.monotonic() - self._zero_worker_since
      )
      self._zero_worker_since = None

    return completed

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    self._assert_weights_consistent()
    await self.sync_pending_weights()
    if not self._rollout_workers:
      raise datatypes.NoHealthyRolloutWorkersError(
          "DistributedRLEngine has no registered rollout workers."
      )

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
        len(self._rollout_workers),
    )
    generation_kwargs = (
        generation_args.as_kwargs() if generation_args is not None else {}
    )
    requests = self._build_rollout_requests(
        prompts,
        policy_version=self._policy_version,
        generation_args=generation_args,
        route_metadata=route_metadata,
    )
    worker_to_requests: dict[Any, list[datatypes.RolloutRequest]] = (
        collections.defaultdict(list)
    )
    for req in requests:
      worker = self._rollout_pool._get_next_actor(
          kwargs={"route_key": req.traj_id}
      )
      worker_to_requests[worker].append(req)

    tasks = [
        self._invoke_worker(
            worker, "generate", requests=w_requests, **generation_kwargs
        )
        for worker, w_requests in worker_to_requests.items()
        if w_requests
    ]
    if not tasks:
      return []

    results = await asyncio.gather(*tasks)
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
    worker = self._inference_workers.get(role)
    if worker is None:
      raise ValueError(f"No inference worker registered for role {role}")
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Scoring %d item(s) on %s worker...",
        len(items),
        role_name,
    )
    return await self._invoke_worker(worker, "score", items=items, **kwargs)

  async def _await_pending_weight_sync(self) -> None:
    """Waits if a mid-step `sync_pending_weights` round is currently in flight."""
    if self._pending_sync_lock.locked():
      async with self._pending_sync_lock:
        pass

  async def per_token_logps(
      self,
      role: datatypes.Role,
      items: Any,
      **kwargs: Any,
  ) -> Any:
    """Evaluates reference model or actor logprobs on a padded batch/request."""
    worker = self._inference_workers.get(role) or self._trainer_workers.get(
        role
    )
    if worker is None:
      raise ValueError(
          f"No worker registered for per_token_logps with role {role}"
      )
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Evaluating per-token log probabilities on %s worker...",
        role_name,
    )
    is_actor_trainer = (
        role == datatypes.Role.ACTOR
        and role not in self._inference_workers
        and role in self._trainer_workers
    )
    if is_actor_trainer:
      await self._await_pending_weight_sync()
      self._actor_busy_count += 1
    try:
      return await self._invoke_worker(
          worker, "per_token_logps", items=items, **kwargs
      )
    finally:
      if is_actor_trainer:
        self._actor_busy_count -= 1

  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
      skip_jit: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Executes atomic gradient accumulation / update on TrainerWorker."""
    worker = self._trainer_workers.get(role)
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
    is_actor = role == datatypes.Role.ACTOR
    if is_actor:
      await self._await_pending_weight_sync()
      self._actor_busy_count += 1
    try:
      fwd_bwd_result = await self._invoke_worker(
          worker,
          "fwd_bwd",
          request=request,
          skip_jit=skip_jit,
          **kwargs,
      )
      if is_actor:
        self._actor_weights_dirty = True
      if not apply_optimizer:
        return fwd_bwd_result
      train_step = await self._invoke_worker(worker, "update")
      return {
          "fwd_bwd": fwd_bwd_result,
          "updated": True,
          "train_step": train_step,
          "accumulated": accumulate_gradients,
      }
    finally:
      if is_actor:
        self._actor_busy_count -= 1

  async def get_metrics(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      **kwargs: Any,
  ) -> (
      exp_metrics.MetricsBuffer
      | Sequence[exp_metrics.MetricsBuffer]
      | dict[str, Any]
      | None
  ):
    """Retrieves step metrics from the worker(s) registered for the specified role."""
    if role == datatypes.Role.ROLLOUT:
      if not self._rollout_workers:
        raise datatypes.NoHealthyRolloutWorkersError(
            f"No rollout workers registered for role {role}"
        )
      tasks = [
          self._invoke_worker(w, "get_metrics", **kwargs)
          for w in self._rollout_workers
      ]
      results = await asyncio.gather(*tasks, return_exceptions=True)
      return [  # pyrefly: ignore[bad-return]
          r for r in results if not isinstance(r, Exception) and r is not None
      ]
    else:
      worker = self._trainer_workers.get(role) or self._inference_workers.get(
          role
      )
      if worker is None:
        raise ValueError(f"No worker registered for role {role}")
      return await self._invoke_worker(worker, "get_metrics", **kwargs)

  def configure_worker(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      *,
      algo: algorithm_adapter.AlgorithmAdapter,
      assembler: batch_assembly.BatchAssembler[Any],
      **kwargs: Any,
  ) -> None:
    """Configures worker(s) under the specified role with algorithm or runtime settings."""
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    if algo is None:
      raise ValueError(
          f"algo is required to configure worker for role {role_name}"
      )
    if assembler is None:
      raise ValueError(
          f"assembler is required to configure worker for role {role_name}"
      )
    match role:
      case datatypes.Role.ACTOR | datatypes.Role.CRITIC:
        worker = self._trainer_workers.get(role)
        if worker is None:
          raise ValueError(f"No trainer worker registered for role {role_name}")
        logging.info(
            "Auto-configuring trainer loss and model input fn on %s worker...",
            role_name,
        )
        pad_id = getattr(assembler, "pad_id", kwargs.get("pad_id", 0))
        eos_id = getattr(assembler, "eos_id", kwargs.get("eos_id", pad_id))
        gen_fn = algo.build_gen_model_input_fn(
            pad_id=pad_id,  # pyrefly: ignore[bad-argument-type]
            eos_id=eos_id,  # pyrefly: ignore[bad-argument-type]
        )

        def _configure():
          assert worker is not None
          worker.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
          worker.submit("with_gen_model_input_fn", gen_fn)

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
        if not self._rollout_workers:
          raise datatypes.NoHealthyRolloutWorkersError(
              "No rollout workers registered on engine."
          )
        logging.info("Configuring rollout workers...")

      case datatypes.Role.REFERENCE:
        worker = self._inference_workers.get(role)
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
    trainer = self._trainer_workers.get(role)
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

  async def sync_weights(  # pyrefly: ignore[bad-override]
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
      policy_version: int | None = None,
      source_staged: asyncio.Event | None = None,
  ) -> int:
    """Runs one weight sync round through the coordinator.

    Args:
      role: Unused; the coordinator knows its sources.
      target_roles: Unused; the coordinator knows its destinations.
      policy_version: Version to push; defaults to the last one plus one.
      source_staged: Set by the coordinator once the source has snapshotted
        this round's weights, so a caller running the round in the background
        knows when the trainer may step again.
    """
    del role, target_roles
    if self._weight_sync_coordinator is None:
      raise RuntimeError(
          "sync_weights needs a coordinator; construct the engine with"
          " weight_sync_coordinator."
      )
    await self._await_pending_weight_sync()
    next_policy_version = (
        self._policy_version + 1 if policy_version is None else policy_version
    )
    logging.info(
        "Synchronizing weights (target policy_version=%d)...",
        next_policy_version,
    )
    sync_kwargs = {}
    if source_staged is not None:
      sync_kwargs["source_staged"] = source_staged
    self._actor_busy_count += 1
    try:
      result = await self._weight_sync_coordinator.sync(
          policy_version=next_policy_version, **sync_kwargs
      )
    except Exception:
      self._weights_consistent = False
      raise
    finally:
      self._actor_busy_count -= 1
    self._actor_weights_dirty = False
    self._weights_consistent = True
    self._policy_version = result.policy_version
    logging.info(
        "Weight synchronization complete (policy_version=%d).",
        self._policy_version,
    )
    return result.policy_version

  async def sync_pending_weights(
      self,
      policy_version: int | None = None,
  ) -> int | None:
    """Proactively syncs weights to any `PENDING_WEIGHT_SYNC` rollout workers.

    Targets only `PENDING_WEIGHT_SYNC` rollout workers so `ACTIVE` rollout
    workers currently generating trajectories are not quiesced or interrupted.
    If the actor trainer worker is currently executing an operation or holds
    uncommitted mid-step weight updates, the catch-up sync is deferred to the
    next step boundary.

    Args:
      policy_version: Optional policy version to push. Defaults to the current
        `self.policy_version`.

    Returns:
      The synced policy version if pending workers were synced, or None if no
      rollout workers were waiting in `PENDING_WEIGHT_SYNC` or the trainer was
      busy.
    """
    if (
        not self._weights_consistent
        or self._actor_busy_count > 0
        or self._actor_weights_dirty
        or self._pending_sync_lock.locked()
        or self._weight_sync_coordinator is None
        or not self._weight_sync_coordinator.has_pending_destinations()
        or self._weight_sync_coordinator.in_flight
        or self._weight_sync_coordinator.poisoned is not None
    ):
      return None
    async with self._pending_sync_lock:
      if (
          not self._weights_consistent
          or self._actor_busy_count > 0
          or self._actor_weights_dirty
          or self._weight_sync_coordinator is None
          or not self._weight_sync_coordinator.has_pending_destinations()
          or self._weight_sync_coordinator.in_flight
          or self._weight_sync_coordinator.poisoned is not None
      ):
        return None
      target_policy_version = (
          self._policy_version if policy_version is None else policy_version
      )
      logging.info(
          "Proactively synchronizing weights to pending rollout worker(s)"
          " (policy_version=%d)...",
          target_policy_version,
      )
      try:
        result = await self._weight_sync_coordinator.sync(
            policy_version=target_policy_version,
            only_pending=True,
        )
      except (weight_sync_coordinator_lib.WeightSyncError, ValueError) as exc:
        if (
            isinstance(exc, weight_sync_coordinator_lib.WeightSyncError)
            and exc.result is not None
            and exc.result.state
            == weight_sync_coordinator_lib.RoundState.UNKNOWN_TRANSFER_STATE
            and self._weight_sync_coordinator.poisoned is not None
        ):
          raise
        if self._weight_sync_coordinator.poisoned is not None:
          self._weight_sync_coordinator.reset_after_recovery()
        # A failed catch-up round (e.g. the pending worker died mid-sync and
        # was evicted) must not interrupt the step: the next end-of-step
        # sync_weights() targets ACTIVE and PENDING_WEIGHT_SYNC workers alike
        # and brings any survivor up to date.
        logging.warning(
            "[rollout-ft] action=pending_sync_deferred policy_version=%d"
            " error=%r; falling back to the next end-of-step sync_weights().",
            target_policy_version,
            exc,
        )
        return None
      return result.policy_version

  async def save_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      metadata: Any = None,
      **kwargs: Any,
  ) -> Any:
    """Requests the trainer worker for `role` to save a checkpoint."""
    worker = self._trainer_workers.get(role)
    if worker is None:
      raise ValueError(f"No trainer worker registered for role {role}")
    role_name = role.value if isinstance(role, datatypes.Role) else str(role)
    logging.info(
        "Saving checkpoint on %s worker...",
        role_name,
    )
    is_actor = role == datatypes.Role.ACTOR
    if is_actor:
      await self._await_pending_weight_sync()
      self._actor_busy_count += 1
    try:
      return await self._invoke_worker(
          worker, "save_checkpoint", metadata=metadata, **kwargs
      )
    finally:
      if is_actor:
        self._actor_busy_count -= 1

  async def _restore_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      **kwargs: Any,
  ) -> Any:
    role_name = role.name
    worker = self._trainer_workers.get(role)
    if worker is None:
      raise ValueError(f"No trainer worker registered for role {role_name}")
    return await self._invoke_worker(worker, "restore_checkpoint", **kwargs)

  async def resume_from_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      resync_rollout_weights: bool = True,
  ) -> int:
    """Restores a checkpoint and realigns the mesh to the restored state.

    See `rl_engine_interface.AbstractRLEngine.resume_from_checkpoint`.
    """
    self._restored_next_batch_idx = 0
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

    try:
      raw_next_batch = metadata.get("next_batch_idx")
      restored_next_batch_idx = (
          int(raw_next_batch)
          if raw_next_batch is not None
          else restored_step
      )
    except (TypeError, ValueError):
      logging.warning(
          "restore_checkpoint returned a non-integer next_batch_idx %r;"
          " falling back to global_step %d.",
          metadata.get("next_batch_idx"),
          restored_step,
      )
      restored_next_batch_idx = restored_step
    self._restored_next_batch_idx = max(restored_step, restored_next_batch_idx)

    # Resume at the step boundary; the policy version tracks the restored step.
    # New checkpoints record optimizer and global steps separately. Legacy
    # checkpoints have only `step`, for which both values are identical.
    restored_policy_version = restored_step
    recorded_version = metadata.get("policy_version")
    # TODO(tunix-dev): this is a force-fit for fully on-policy RL. Remove when
    # async off-policy is supported.
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
    if resync_rollout_weights:
      await self._maybe_configure_trainer_target_state(role)
      synced_version = await self.sync_weights(
          role=role,
          policy_version=restored_policy_version,
      )
      if synced_version != restored_policy_version:
        raise RuntimeError(
            "Resumed policy_version=%d does not match synced version=%d"
            % (restored_policy_version, synced_version)
        )
    else:
      logging.warning(
          "resync_rollout_weights is False. Resumed rollout workers will use"
          " base weights instead of restored checkpoint version %d.",
          restored_policy_version,
      )
    return restored_step

  async def close(self) -> None:
    """Closes the rollout execution session and cancels any pending polling tasks."""
    await self._rollout_session.close()
