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
- DistributedRLEngine: Worker-backed compute router implementing AbstractRLEngine.
"""

import asyncio
from collections.abc import Mapping, Sequence
import inspect
from typing import Any
import uuid

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_engine_interface


def _response_to_trajectory_item(resp: Any) -> datatypes.TrajectoryItem:
  """Converts a worker rollout response to an TrajectoryItem."""
  if isinstance(resp, datatypes.TrajectoryItem):
    return resp

  if isinstance(resp, datatypes.RolloutResponse):
    prompt_id = resp.prompt_id or "default_prompt"
    metadata = dict(resp.metadata) if resp.metadata else {}
    group_id = metadata.get("group_id", prompt_id)
    pair_index = metadata.get("pair_index", 0)
    traj = datatypes.Trajectory(
        reward=resp.env_reward,
        status=(
            datatypes.TrajectoryStatus.SUCCEEDED
            if resp.status == "COMPLETED"
            else datatypes.TrajectoryStatus.FAILED
        ),
    )
    item = datatypes.TrajectoryItem(
        pair_index=pair_index,
        group_id=group_id,
        start_step=0,
        traj=traj,
        metadata=metadata,
        prompt_tokens=resp.prompt_tokens,
        policy_version=resp.policy_version,
    )

    assistant_tokens = []
    assistant_masks = []
    for seg in resp.segments:
      if seg.source == "assistant":
        assistant_tokens.append(seg.tokens)
        assistant_masks.append(seg.loss_mask)
    if assistant_tokens:
      item.completion_tokens = np.concatenate(assistant_tokens)
      item.action_mask = np.concatenate(assistant_masks)
    else:
      item.completion_tokens = np.zeros(0, dtype=np.int32)
      item.action_mask = np.zeros(0, dtype=np.float32)
    return item

  if isinstance(resp, datatypes.Trajectory):
    item = datatypes.TrajectoryItem(
        pair_index=0,
        group_id=getattr(resp, "task", "default_group"),
        start_step=0,
        traj=resp,
        policy_version=getattr(resp, "policy_version", 0),
        prompt_tokens=getattr(resp, "prompt_tokens", np.zeros(0, dtype=np.int32)),
        completion_tokens=getattr(resp, "completion_tokens", np.zeros(0, dtype=np.int32)),
        action_mask=getattr(resp, "action_mask", np.ones(len(getattr(resp, "completion_tokens", [])), dtype=np.float32)),
    )
    return item

  raise TypeError(
      f"Unsupported response type for trajectory conversion: {type(resp)}"
  )


class WorkerPoolBalancer:
  """Load balancing and prefix-cache affinity tracking across worker replicas."""

  def __init__(self, workers: Sequence[Any]):
    self._workers = list(workers)
    self._in_flight: dict[int, int] = {i: 0 for i in range(len(self._workers))}

  def select_worker_for_request(
      self, req: datatypes.RolloutRequest
  ) -> tuple[int, Any]:
    """Selects worker using least-in-flight queue depth or prefix-cache hash affinity."""
    if not self._workers:
      raise ValueError("WorkerPoolBalancer has no registered rollout workers.")

    metadata = req.metadata if req.metadata else {}
    if "prefix_hash" in metadata:
      idx = metadata["prefix_hash"] % len(self._workers)
    else:
      idx = min(self._in_flight, key=self._in_flight.get)
    self._in_flight[idx] += 1
    return idx, self._workers[idx]

  def record_completion(self, worker_idx: int, count: int = 1) -> None:
    """Decrements in-flight count for a worker upon task completion."""
    if worker_idx in self._in_flight:
      self._in_flight[worker_idx] = max(0, self._in_flight[worker_idx] - count)


class DistributedRLEngine(rl_engine_interface.AbstractRLEngine):
  """Worker-backed compute router dispatching RPCs across role pools."""

  def __init__(
      self,
      rollout_workers: Sequence[Any],
      trainer_workers: Mapping[datatypes.Role, Any],
      inference_workers: Mapping[datatypes.Role, Any] | None = None,
  ):
    self._rollout_workers = list(rollout_workers)
    self._balancer = WorkerPoolBalancer(self._rollout_workers)
    self._trainer_workers = dict(trainer_workers)
    self._inference_workers = dict(inference_workers or {})

  async def _invoke_worker(
      self, worker: Any, method_name: str, **kwargs: Any
  ) -> Any:
    """Helper invoking method on remote handle or in-process mock."""
    if hasattr(worker, "asubmit"):
      res = worker.asubmit(method_name, **kwargs)
      if inspect.isawaitable(res):
        return await res
      return res

    method = getattr(worker, method_name, None)
    if method is None:
      raise AttributeError(f"Worker {worker} has no method {method_name}")

    sig = inspect.signature(method)
    call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    res = method(**call_kwargs)
    if inspect.isawaitable(res):
      return await res
    return res

  async def dispatch_rollouts(
      self, prompts: Sequence[Any], **kwargs: Any
  ) -> list[str]:
    """Dispatches rollout requests across workers, constructing RolloutRequests internally if needed."""
    rollout_reqs: list[datatypes.RolloutRequest] = []
    for idx, p in enumerate(prompts):
      if isinstance(p, datatypes.RolloutRequest):
        rollout_reqs.append(p)
      else:
        req_id = (
            kwargs.get("request_id") or f"req_{idx}_{uuid.uuid4().hex[:8]}"
        )
        rollout_reqs.append(
            datatypes.RolloutRequest(
                request_id=req_id,
                prompt=p,
                prompt_id=f"prompt_{idx}",
                target_policy_version=kwargs.get("policy_version", 0),
                metadata=dict(kwargs.get("metadata", {})),
            )
        )

    for req in rollout_reqs:
      _, worker = self._balancer.select_worker_for_request(req)
      if hasattr(worker, "dispatch_task"):
        res = worker.dispatch_task(method_name="generate", requests=[req])
        if inspect.isawaitable(res):
          await res
      elif hasattr(worker, "dispatch"):
        res = worker.dispatch(req)
        if inspect.isawaitable(res):
          await res
      else:
        await self._invoke_worker(worker, "generate", prompts=[req.prompt])

    return [r.request_id for r in rollout_reqs]

  async def poll_rollouts(
      self, timeout_s: float = 0.1
  ) -> list[datatypes.TrajectoryItem]:
    """Concurrently long-polls completed rollout responses across all workers."""
    if not self._rollout_workers:
      return []

    async def _poll_worker(worker: Any) -> Any:
      if hasattr(worker, "poll_responses"):
        fn = worker.poll_responses
      elif hasattr(worker, "poll"):
        fn = worker.poll
      elif hasattr(worker, "asubmit"):
        return await self._invoke_worker(
            worker, "poll_responses", timeout_s=timeout_s
        )
      else:
        return []

      res = (
          fn(timeout_s=timeout_s)
          if "timeout_s" in inspect.signature(fn).parameters
          else fn()
      )
      if inspect.iscoroutine(res):
        res = await res
      return res

    tasks = [_poll_worker(w) for w in self._rollout_workers]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    completed: list[datatypes.TrajectoryItem] = []

    for i, resp in enumerate(responses):
      if isinstance(resp, Exception) or resp is None:
        continue
      unwrap_fn = getattr(resp, "unwrap", None)
      res = (
          unwrap_fn() if callable(unwrap_fn) else getattr(resp, "result", resp)
      )
      if res is not None:
        items = res if isinstance(res, list) else [res]
        self._balancer.record_completion(i, len(items))
        for it in items:
          if isinstance(it, dict):
            it = datatypes.RolloutResponse(**it)
          completed.append(_response_to_trajectory_item(it))
    return completed

  async def generate(self, prompts: Sequence[Any], **kwargs: Any) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    if not self._rollout_workers:
      raise ValueError("DistributedRLEngine has no registered rollout workers.")

    worker_to_prompts: dict[int, list[Any]] = {
        i: [] for i in range(len(self._rollout_workers))
    }
    for idx, p in enumerate(prompts):
      req = (
          p
          if isinstance(p, datatypes.RolloutRequest)
          else datatypes.RolloutRequest(
              request_id=f"gen_{idx}_{uuid.uuid4().hex[:8]}",
              prompt=p,
              target_policy_version=kwargs.get("policy_version", 0),
              metadata=dict(kwargs.get("metadata", {})),
          )
      )
      w_idx, _ = self._balancer.select_worker_for_request(req)
      worker_to_prompts[w_idx].append(p)

    tasks = []
    task_worker_indices = []
    for w_idx, w_prompts in worker_to_prompts.items():
      if w_prompts:
        worker = self._rollout_workers[w_idx]
        tasks.append(
            self._invoke_worker(
                worker, "generate", prompts=w_prompts, **kwargs
            )
        )
        task_worker_indices.append((w_idx, len(w_prompts)))

    if not tasks:
      return []

    results = await asyncio.gather(*tasks)
    for w_idx, count in task_worker_indices:
      self._balancer.record_completion(w_idx, count)

    raw_items = [
        item
        for sublist in results
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return [_response_to_trajectory_item(it) for it in raw_items]

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
    return await self._invoke_worker(worker, "score", items=items, **kwargs)

  async def per_token_logps(
      self,
      role: datatypes.Role,
      items: Sequence[Any],
      **kwargs: Any,
  ) -> Any:
    """Evaluates reference model or actor per-token logprobs."""
    worker = self._inference_workers.get(role) or self._trainer_workers.get(
        role
    )
    if worker is None:
      raise ValueError(
          f"No worker registered for per_token_logps with role {role}"
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
    return await self._invoke_worker(
        worker,
        "fwd_bwd",
        batch=payload,
        accumulate_gradients=accumulate_gradients,
        apply_optimizer=apply_optimizer,
        skip_jit=skip_jit,
        **kwargs,
    )

  async def sync_weights(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
  ) -> int:
    """Executes accelerator-to-accelerator collective weight broadcast."""
    del target_roles
    trainer = self._trainer_workers.get(role)
    if trainer is None:
      return 0
    sync_metadata = await self._invoke_worker(trainer, "prepare_weight_sync")
    tasks = [
        self._invoke_worker(w, "weight_sync", metadata=sync_metadata)
        for w in self._rollout_workers
        if hasattr(w, "weight_sync") or hasattr(w, "asubmit")
    ]
    if tasks:
      await asyncio.gather(*tasks)
    return getattr(sync_metadata, "new_policy_version", 1)
