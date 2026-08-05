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

"""DistributedRLEngine implementation for distributed worker routing.

This engine implements ``AbstractRLEngine`` by routing compute primitives
(generate, train / train_step, per_token_logps, sync_weights) to role-based
worker handles (trainer, rollout, inference).
"""

import asyncio
from typing import Any, Mapping, Sequence

from jax import typing as jax_typing
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.worker import inference_worker as inference_worker_lib
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker as rollout_worker_lib
from tunix.experimental.worker import trainer_worker as trainer_worker_lib

ActorHandle = remote_execution.ActorHandle
InferenceWorker = inference_worker_lib.InferenceWorker
RolloutWorker = rollout_worker_lib.RolloutWorker
TrainerWorker = trainer_worker_lib.TrainerWorker


class LoadBalancer:
  """Distributes requests evenly across available rollout workers."""

  # TODO (noghabi): define apis of load balancer better.
  def split_requests(
      self, requests: Sequence[Any], workers: Sequence[Any]
  ) -> list[tuple[Any, list[Any]]]:
    """Splits requests across workers, returning list of (worker, chunk) pairs."""
    if not workers or not requests:
      return []
    num_workers = len(workers)
    chunk_size = (len(requests) + num_workers - 1) // num_workers
    distribution = []
    for i, worker in enumerate(workers):
      chunk = list(requests[i * chunk_size : (i + 1) * chunk_size])
      if chunk:
        distribution.append((worker, chunk))
    return distribution


class DistributedRLEngine(rl_engine_interface.AbstractRLEngine):
  """Worker-backed router implementing AbstractRLEngine.

  Routes stateless compute (generate, train / train_step, per_token_logps,
  sync_weights) to role-specific worker handles while storing shared
  bookkeeping and config fields directly. Supports both explicit worker
  handles and dynamic WorkerRegistry resolution.
  """

  # TODO(noghabi): support WorkerRegistry for dynamic worker discovery instead of explicit worker handles.
  def __init__(
      self,
      rollout_workers: Sequence[ActorHandle] | ActorHandle,
      trainer_workers: Mapping[datatypes.Role, ActorHandle],
      inference_worker: ActorHandle | None = None,
      load_balancer: LoadBalancer | None = None,
  ):
    """Initializes DistributedRLEngine.

    Args:
      rollout_workers: Sequence of worker handles for generating rollouts.
      trainer_workers: Mapping from Role to trainer worker handles.
      inference_worker: Optional worker handle for reference/reward logprobs.
      load_balancer: Optional load balancer for distributing rollout tasks.
    """
    self._load_balancer = load_balancer or LoadBalancer()

    if rollout_workers is None:
      raise ValueError("rollout_workers must not be None or empty.")
    if isinstance(rollout_workers, Sequence):
      raw_rollouts = list(rollout_workers)
    else:
      raw_rollouts = [rollout_workers]
    if not raw_rollouts:
      raise ValueError("rollout_workers must not be None or empty.")
    if not trainer_workers or datatypes.Role.ACTOR not in trainer_workers:
      raise ValueError(
          "trainer_workers must not be empty and must contain an ACTOR role."
      )
    self._rollout_workers: list[ActorHandle] = raw_rollouts
    self._trainer_workers: dict[datatypes.Role, ActorHandle] = dict(
        trainer_workers
    )
    self._inference_worker: ActorHandle | None = inference_worker

  @property
  def rollout_workers(self) -> list[ActorHandle]:
    """Returns active rollout workers."""
    return list(self._rollout_workers)

  def _get_trainer_worker(
      self, role: datatypes.Role = datatypes.Role.ACTOR
  ) -> ActorHandle:
    """Resolves trainer worker for a given role from mapping."""
    if role not in self._trainer_workers:
      raise KeyError(f"No trainer worker registered for role={role}")
    return self._trainer_workers[role]

  def _get_inference_worker(self) -> ActorHandle | None:
    """Resolves inference/reference worker from handle."""
    return self._inference_worker

  async def dispatch_generate(
      self,
      requests: Sequence[datatypes.RolloutRequest] | Any,
  ) -> None:
    """Dispatches rollout requests across rollout workers using fire-and-forget."""
    workers = self.rollout_workers
    for worker, chunk in self._load_balancer.split_requests(requests, workers):
      await worker.dispatch_task(method_name="generate", requests=chunk)

  async def poll_rollouts(self, timeout_s: float = 0.1) -> list[Any]:
    """Polls completed rollout responses across all rollout workers."""
    completed = []
    for worker in self.rollout_workers:
      resp = await worker.poll_responses(timeout_s=timeout_s)
      if resp is not None:
        unwrap_fn = getattr(resp, "unwrap", None)
        res = (
            unwrap_fn()
            if callable(unwrap_fn)
            else getattr(resp, "result", resp)
        )
        if res is not None:
          if isinstance(res, (list, tuple)):
            completed.extend(res)
          else:
            completed.append(res)
    return completed

  async def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    """Generates completions for prompts across rollout workers."""
    workers = self.rollout_workers
    distribution = self._load_balancer.split_requests(prompts, workers)
    tasks = []
    for worker, chunk in distribution:
      task = worker.asubmit(
          "generate",
          prompts=chunk,
          apply_chat_template=apply_chat_template,
          mode=mode,
          micro_batch_size=micro_batch_size,
          trace_tags=trace_tags,
          max_generation_steps=max_generation_steps,
      )
      tasks.append(task)

    results_list = await asyncio.gather(*tasks)
    results = []
    for res in results_list:
      if isinstance(res, (list, tuple)):
        results.extend(res)
      else:
        results.append(res)
    return results

  def train(
      self,
      role: datatypes.Role,
      train_ds: Any,
      eval_ds: Any,
      skip_jit: bool = False,
  ) -> None:
    raise NotImplementedError(
        "The Distributed engine does not support running the full loop. The"
        " outer loop should be defined in the upper layers."
    )

  async def train_step(
      self,
      batch: Any,
      role: datatypes.Role = datatypes.Role.ACTOR,
      skip_jit: bool = False,
  ) -> Any:
    """Runs a single atomic gradient update step over a batch."""
    worker = self._get_trainer_worker(role)
    return await worker.asubmit("fwd_bwd", batch=batch, skip_jit=skip_jit)

  def per_token_logps(
      self,
      role: datatypes.Role,
      prompt_tokens: jax_typing.ArrayLike,
      completion_tokens: jax_typing.ArrayLike,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: jax_typing.ArrayLike | None = None,
      **kwargs: Any,
  ) -> Any:
    """Computes per-token logprobs for the specified role."""
    if role == datatypes.Role.REFERENCE:
      inference_worker = self._get_inference_worker()
      if inference_worker is None:
        raise KeyError(f"No inference worker registered for role={role}")
      import numpy as np  # pylint: disable=g-import-not-at-top

      req = datatypes.LogprobsRequest(
          prompt_tokens=np.asarray(prompt_tokens),
          completion_tokens=np.asarray(completion_tokens),
          temperature=kwargs.get("temperature", 1.0),
          model_role="reference",
      )
      return inference_worker.submit("compute_logprobs", req)

    try:
      _ = self._get_trainer_worker(role)
      raise NotImplementedError(
          f"per_token_logps is not supported by trainer worker for role={role}"
      )
    except KeyError:
      raise KeyError(f"No worker registered for role={role}")

  # TODO: add a proper weigh sync with Raiden controller
  async def sync_weights(self) -> Any:
    """Synchronizes trainer weights to rollout/inference replicas."""
    actor_worker = self._get_trainer_worker(datatypes.Role.ACTOR)
    sync_info = actor_worker.submit("prepare_weight_sync")
    rollout_workers = self.rollout_workers
    for worker in rollout_workers:
      worker.submit("pre_weight_sync", sync_info)
    for worker in rollout_workers:
      worker.submit("weight_sync", sync_info)
    for worker in rollout_workers:
      worker.submit("post_weight_sync", sync_info)
