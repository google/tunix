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
import collections
from collections.abc import Mapping, Sequence
import inspect
from typing import Any
import uuid

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.worker import remote_execution


# TODO: this multi step conversions seem excessive we convert from trajecotry to response then to trajectory item. we should simplify
def _response_to_trajectory_item(resp: Any) -> datatypes.TrajectoryItem:
  """Converts a worker rollout response to an TrajectoryItem."""
  if isinstance(resp, datatypes.TrajectoryItem):
    return resp

  if isinstance(resp, datatypes.RolloutResponse):
    prompt_id = resp.prompt_id or "default_prompt"
    metadata = dict(resp.metadata) if resp.metadata else {}
    group_id = metadata.get("group_id", prompt_id)
    pair_index = metadata.get("pair_index", 0)
    success_statuses = {"COMPLETED", "SUCCEEDED"}
    traj = datatypes.Trajectory(
        reward=resp.env_reward,
        status=(
            datatypes.TrajectoryStatus.SUCCEEDED
            if resp.status in success_statuses
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


class DistributedRLEngine(rl_engine_interface.AbstractRLEngine):
  """Worker-backed compute router dispatching RPCs across role pools."""

  def __init__(
      self,
      rollout_workers: Sequence[remote_execution.ActorHandle],
      trainer_workers: Mapping[datatypes.Role, remote_execution.ActorHandle],
      inference_workers: (
          Mapping[datatypes.Role, remote_execution.ActorHandle] | None
      ) = None,
  ):
    self._rollout_workers = list(rollout_workers)
    self._rollout_pool = remote_execution.RoutingActorPool(
        self._rollout_workers
    )
    self._trainer_workers = dict(trainer_workers)
    self._inference_workers = dict(inference_workers or {})

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

  async def dispatch_rollouts(
      self, prompts: Sequence[Any], **kwargs: Any
  ) -> list[str]:
    """Dispatches rollout requests across workers, constructing RolloutRequests internally if needed."""
    rollout_reqs: list[datatypes.RolloutRequest] = []
    for idx, p in enumerate(prompts):
      # TODO: why do we support sending rollout requests directly? shouldn't this be the engine resposibility?
      if isinstance(p, datatypes.RolloutRequest):
        rollout_reqs.append(p)
      else:
        prompt_ids = kwargs.get("prompt_ids")
        if not prompt_ids or len(prompt_ids) != len(prompts):
          raise ValueError(
              "When passing raw prompts, 'prompt_ids' must be provided in"
              " kwargs and match the length of prompts."
          )
        if "policy_version" not in kwargs:
          raise ValueError(
              "When passing raw prompts, 'policy_version' must be provided in"
              " kwargs."
          )
        # TODO: should we autogenerate request_id?
        req_id = (
            kwargs.get("request_id") or f"req_{idx}_{uuid.uuid4().hex[:8]}"
        )
        rollout_reqs.append(
            datatypes.RolloutRequest(
                request_id=req_id,
                prompt=p,
                prompt_id=prompt_ids[idx],
                target_policy_version=kwargs["policy_version"],
                metadata=dict(kwargs.get("metadata", {})),
            )
        )

    for req in rollout_reqs:
      metadata = req.metadata or {}
      route_key = metadata.get("prefix_hash")
      if route_key is None:
        route_key = req.prompt_id
      worker = self._rollout_pool._get_next_actor(
          kwargs={"route_key": route_key}
      )

      res = worker.dispatch_task(method_name="generate", requests=[req])
      if inspect.isawaitable(res):
        await res

    return [r.request_id for r in rollout_reqs]

  async def poll_rollouts(
      self, timeout_s: float = 0.1
  ) -> list[datatypes.TrajectoryItem]:
    """Concurrently long-polls completed rollout responses across all workers."""
    if not self._rollout_workers:
      return []

    async def _poll_worker(worker: remote_execution.ActorHandle) -> Any:
      return await self._invoke_worker(
          worker, "poll_responses", timeout_s=timeout_s
      )

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
        for it in items:
          if isinstance(it, dict):
            it = datatypes.RolloutResponse(**it)
          completed.append(_response_to_trajectory_item(it))
    return completed

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.TrajectoryItem]:
    """Blocking rollout generation: load-balances prompts across workers and awaits completion."""
    if not self._rollout_workers:
      raise ValueError("DistributedRLEngine has no registered rollout workers.")

    if kwargs:
      raise TypeError(
          "Unexpected generate kwargs: "
          f"{sorted(kwargs)}. Use generation_args=GenerationArgs(...) for "
          "sampling parameters."
      )

    generation_kwargs = (
        generation_args.as_kwargs() if generation_args is not None else {}
    )
    route_metadata_map = dict(route_metadata or {})
    worker_to_prompts: dict[Any, list[Any]] = collections.defaultdict(list)
    worker_to_requests: dict[Any, list[datatypes.RolloutRequest]] = (
        collections.defaultdict(list)
    )
    for p in prompts:
      if isinstance(p, datatypes.RolloutRequest):
        request_metadata = dict(p.metadata or {})
        route_key = request_metadata.get("prefix_hash")
        if route_key is None:
          route_key = p.prompt_id
        worker = self._rollout_pool._get_next_actor(
            kwargs={"route_key": route_key}
        )
        worker_to_requests[worker].append(p)
        continue

      route_key = route_metadata_map.get("prefix_hash")
      if route_key is None:
        route_key = route_metadata_map.get("prompt_id")
      worker = self._rollout_pool._get_next_actor(
          kwargs={"route_key": route_key}
      )
      worker_to_prompts[worker].append(p)

    tasks = []
    for worker, w_requests in worker_to_requests.items():
      if w_requests:
        tasks.append(
            self._invoke_worker(
                worker, "generate", requests=w_requests, **generation_kwargs
            )
        )
    for worker, w_prompts in worker_to_prompts.items():
      if w_prompts:
        tasks.append(
            self._invoke_worker(
                worker, "generate", prompts=w_prompts, **generation_kwargs
            )
        )

    if not tasks:
      return []

    results = await asyncio.gather(*tasks)

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
    fwd_bwd_result = await self._invoke_worker(
        worker,
        "fwd_bwd",
        payload=payload,
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

  async def sync_weights(  # pyrefly: ignore[bad-override]
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
  ) -> int:
    """Executes accelerator-to-accelerator collective weight broadcast."""
    # TODO: integrate with raiden controller instead
    del target_roles
    trainer = self._trainer_workers.get(role)
    if trainer is None:
      return 0
    sync_metadata = await self._invoke_worker(trainer, "prepare_weight_sync")
    if not isinstance(sync_metadata, datatypes.WeightSyncMetadata):
      raise RuntimeError(
          "prepare_weight_sync must return WeightSyncMetadata; got "
          f"{type(sync_metadata).__name__}."
      )
    tasks = [
        self._invoke_worker(w, "weight_sync", metadata=sync_metadata)
        for w in self._rollout_workers
        if hasattr(w, "weight_sync") or hasattr(w, "asubmit")
    ]
    if tasks:
      await asyncio.gather(*tasks)
    return getattr(sync_metadata, "new_policy_version", 1)
