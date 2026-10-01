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
from collections.abc import Mapping, Sequence
import concurrent.futures
import inspect
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
from tunix.experimental.orchestrator import fault_tolerance
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.orchestrator import worker_registry
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
    metadata["error"] = str(resp.error)
    return datatypes.TrajectoryItem(
        prompt_id=prompt_id,
        group_index=group_index,
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
      fault_tolerance_config: (
          fault_tolerance.RolloutFaultToleranceConfig | None
      ) = None,
      fault_tolerance_manager: (
          fault_tolerance.RolloutFaultToleranceManager | None
      ) = None,
      registry: worker_registry.WorkerRegistry | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
  ):
    self._rollout_workers = list(rollout_workers)
    self._rollout_pool = remote_execution.RoutingActorPool(
        self._rollout_workers
    )
    # Least-loaded, not hash-by-traj_id: each rollout request is one whole
    # episode, and episode lengths vary by 10x, so hashing left some workers
    # with twice the episodes of others and stretched the batch tail.
    self._rollout_session = remote_execution.PoolExecutionSession(
        self._rollout_pool, least_loaded=True
    )
    self._trainer_workers = dict(trainer_workers)
    self._inference_workers = dict(inference_workers or {})
    self._policy_version = 0
    self._restored_next_batch_idx = 0
    self._weight_sync_coordinator = weight_sync_coordinator
    self._ft_manager = (
        fault_tolerance_manager
        or fault_tolerance.RolloutFaultToleranceManager(
            rollout_pool=self._rollout_pool,
            registry=registry,
            monitor=monitor,
            config=fault_tolerance_config,
        )
    )

  @property
  def fault_tolerance_manager(
      self,
  ) -> fault_tolerance.RolloutFaultToleranceManager:
    return self._ft_manager

  @property
  def rollout_pool(self) -> remote_execution.RoutingActorPool:
    return self._rollout_pool

  @property
  def restored_next_batch_idx(self) -> int:
    """First untrained prompt-batch index from the last restored checkpoint."""
    return self._restored_next_batch_idx

  async def _maybe_configure_trainer_target_state(
      self,
      role: datatypes.Role,
  ) -> None:
    """Seeds trainer-side weight sync with the rollout target-state skeleton."""
    trainer = self._trainer_workers.get(role)
    active_rollouts = self._rollout_pool.actors
    if trainer is None or not active_rollouts:
      return

    rollout = active_rollouts[0]
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

  async def _submit_to_rollout_worker(
      self,
      worker: remote_execution.ActorHandle,
      req: datatypes.RolloutRequest,
  ) -> None:
    await self._rollout_session.submit_to_actor(
        worker,
        req.request_id,
        "generate",
        requests=[req],
    )

  async def _drain_pending_rollouts(self) -> list[str]:
    return await self._ft_manager.drain_retry_queue(
        dispatch_fn=self._submit_to_rollout_worker,
    )

  async def dispatch_rollout_requests(
      self,
      requests: Sequence[datatypes.RolloutRequest],
  ) -> list[str]:
    """Dispatches pre-formed RolloutRequests across rollout workers."""
    if self._ft_manager.config.enable_background_heartbeat:
      self._ft_manager.start_heartbeat_loop()
    await self._ft_manager.wait_for_weight_sync_idle()
    requests = self._build_rollout_requests(requests)
    logging.info(
        "Dispatching %d rollout request(s) across %d worker(s).",
        len(requests),
        len(self._rollout_pool),
    )
    if len(self._rollout_pool) > 0:
      await self._drain_pending_rollouts()

    to_dispatch: list[
        tuple[remote_execution.ActorHandle, datatypes.RolloutRequest]
    ] = []
    for req in requests:
      logging.debug(
          "Dispatched rollout request (prompt_id=%s, group_index=%d,"
          " request_id=%s).",
          req.prompt_id,
          req.group_index,
          req.request_id,
      )
      worker = self._ft_manager.select_worker(req)
      if worker is None:
        self._ft_manager.enqueue_retry(req)
        continue
      self._ft_manager.record_dispatch(req, worker)
      to_dispatch.append((worker, req))

    async def _send(
        worker: remote_execution.ActorHandle,
        req: datatypes.RolloutRequest,
    ) -> bool:
      try:
        await self._submit_to_rollout_worker(worker, req)
        return True
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Dispatch of trajectory %r to worker %r failed (%s); evicting"
            " worker and queueing in _retry_queue.",
            req.traj_id,
            self._ft_manager.worker_id_of(worker),
            exc,
        )
        self._ft_manager.handle_dispatch_failure(
            req, worker, reason=str(exc)
        )
        self._rollout_session.evict_actor_tasks(worker)
        return False

    if to_dispatch:
      send_results = await asyncio.gather(
          *(_send(w, r) for w, r in to_dispatch)
      )
      if not all(send_results):
        if len(self._rollout_pool) > 0:
          await self._drain_pending_rollouts()
        else:
          self._ft_manager.check_zero_worker_timeout()

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
      **kwargs: Any,
  ) -> list[str]:
    """Dispatches rollout requests across workers, constructing RolloutRequests internally.

    Every prompt item in `prompts` MUST have a unique, collision-free
    `prompt_id`
    attribute or dict key. Missing prompt IDs raise a ValueError.
    """
    rollout_reqs = self._build_rollout_requests(
        prompts,
        num_generations=num_generations,
        policy_version=policy_version,
        generation_args=generation_args,
        route_metadata=route_metadata,
        **kwargs,
    )
    return await self.dispatch_rollout_requests(rollout_reqs)

  async def poll_rollouts(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> list[datatypes.TrajectoryItem]:
    """Concurrently long-polls completed rollout responses across all workers."""
    if not self._rollout_pool:
      if (
          self._ft_manager.pending_rejoin_workers
          and self._weight_sync_coordinator is not None
      ):
        logging.info(
            "0 active rollout workers in pool while %d rejoined worker(s)"
            " await weight sync; triggering immediate sync_weights"
            " (policy_version=%d).",
            len(self._ft_manager.pending_rejoin_workers),
            self._policy_version,
        )
        await self.sync_weights(policy_version=self._policy_version)
      if not self._rollout_pool:
        if (
            not self._rollout_workers
            and self._ft_manager.inflight_count() == 0
            and self._ft_manager.retry_queue_size == 0
        ):
          return []
        self._ft_manager.check_zero_worker_timeout()
        return self._ft_manager.evict_timed_out_inflight()

    for actor in self._rollout_pool.actors:
      req_ids = self._ft_manager.inflight_request_ids_for_worker(actor)
      if req_ids:
        self._rollout_session.ensure_actor_polling(actor, req_ids)

    completed: list[datatypes.TrajectoryItem] = []
    for (
        actor,
        res,
        exc,
    ) in await self._rollout_session.poll_completed_with_actors(
        timeout_s=timeout_s
    ):
      if exc is not None:
        if "AdmissionClosedError" in str(exc):
          logging.info(
              "Rollout worker %r closed admission for weight sync (%s);"
              " re-queueing in-flight requests without evicting worker.",
              self._ft_manager.worker_id_of(actor),
              exc,
          )
          self._ft_manager.requeue_inflight_for_worker(actor, front=True)
          self._rollout_session.evict_actor_tasks(actor)
        else:
          logging.error("Failed polling rollout worker: %s", exc)
          self._ft_manager.evict_rollout_worker(actor, reason=str(exc))
          self._rollout_session.evict_actor_tasks(actor)
        continue
      if res is None:
        continue
      items = res if isinstance(res, list) else [res]
      worker_traj_items: list[datatypes.TrajectoryItem] = []
      for it in items:
        if isinstance(it, dict):
          it = datatypes.RolloutResponse(**it)
        traj_item = _response_to_trajectory_item(it)
        logging.debug(
            "Received rollout response (prompt_id=%s, group_index=%d).",
            traj_item.prompt_id,
            traj_item.group_index,
        )
        worker_traj_items.append(traj_item)
      if worker_traj_items:
        completed.extend(
            self._ft_manager.filter_and_complete_responses(
                actor, worker_traj_items
            )
        )

    completed.extend(self._ft_manager.evict_timed_out_inflight())

    if not self._rollout_pool:
      if (
          self._ft_manager.pending_rejoin_workers
          and self._weight_sync_coordinator is not None
      ):
        await self.sync_weights(policy_version=self._policy_version)
      else:
        self._ft_manager.check_zero_worker_timeout()
    elif self._ft_manager.retry_queue_size > 0:
      await self._drain_pending_rollouts()

    return completed

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    if not self._rollout_pool:
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
        len(self._rollout_pool),
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
    worker_to_requests: dict[
        remote_execution.ActorHandle, list[datatypes.RolloutRequest]
    ] = collections.defaultdict(list)
    active_actors = set(self._rollout_pool.actors)
    routed_by_key: dict[str, remote_execution.ActorHandle] = {}
    for req in requests:
      if (
          req.traj_id in routed_by_key
          and routed_by_key[req.traj_id] in active_actors
      ):
        worker = routed_by_key[req.traj_id]
      else:
        selected = self._rollout_pool.select_actor(
            route_key=req.traj_id,
            load_fn=lambda a: len(worker_to_requests[a]),
        )
        if selected is None:
          raise RuntimeError(
              "Failed to select an available rollout worker for trajectory"
              f" {req.traj_id}."
          )
        worker = selected
        routed_by_key[req.traj_id] = worker
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
    return await self._invoke_worker(
        worker, "per_token_logps", items=items, **kwargs
    )

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
    fwd_bwd_result = await self._invoke_worker(
        worker,
        "fwd_bwd",
        request=request,
        skip_jit=skip_jit,
        **kwargs,
    )
    if not apply_optimizer:
      return fwd_bwd_result
    train_step = await self._invoke_worker(worker, "update")
    return {
        "fwd_bwd": fwd_bwd_result,
        "updated": True,
        "train_step": train_step,
        "accumulated": accumulate_gradients,
    }

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
      active_rollouts = self._rollout_pool.actors
      if not active_rollouts:
        raise ValueError(f"No rollout workers registered for role {role}")
      tasks = [
          self._invoke_worker(w, "get_metrics", **kwargs)
          for w in active_rollouts
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
        if not self._rollout_pool:
          raise ValueError("No rollout workers registered on engine.")
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
    self._ft_manager.enter_weight_sync()
    try:
      while True:
        try:
          result = await self._weight_sync_coordinator.sync(
              policy_version=next_policy_version, **sync_kwargs
          )
          break
        except Exception as exc:  # pylint: disable=broad-exception-caught
          sync_res = getattr(exc, "result", None)
          worker_reports = getattr(sync_res, "workers", None)
          evicted_any = False
          if worker_reports:
            for report in worker_reports:
              if getattr(report, "error", "") or getattr(
                  report, "needs_restart", False
              ):
                wid = report.worker_id
                actor = self._rollout_pool.get_actor(
                    wid
                ) or self._ft_manager.pending_rejoin_workers.get(wid)
                if actor is not None:
                  self._ft_manager.evict_rollout_worker(
                      actor,
                      reason=getattr(report, "error", "weight_sync_failed"),
                  )
                  self._rollout_session.evict_actor_tasks(actor)
                  evicted_any = True
          if not evicted_any:
            raise
          self._ft_manager.check_zero_worker_timeout()
          if hasattr(self._weight_sync_coordinator, "reset_after_recovery"):
            self._weight_sync_coordinator.reset_after_recovery()

      self._policy_version = result.policy_version
      synced_workers = getattr(result, "workers", None)
      self._ft_manager.on_weight_sync_completed(
          synced_workers if synced_workers else None
      )
    finally:
      self._ft_manager.exit_weight_sync()

    if self._ft_manager.retry_queue_size > 0 and len(self._rollout_pool) > 0:
      await self._drain_pending_rollouts()
    logging.info(
        "Weight synchronization complete (policy_version=%d).",
        self._policy_version,
    )
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
    return await self._invoke_worker(
        worker, "save_checkpoint", metadata=metadata, **kwargs
    )

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
    await self._ft_manager.stop_heartbeat_loop()
    await self._rollout_session.close()
