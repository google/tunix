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

"""Demonstration of an RL training loop using remote_execution and an ActorPool.

Shows how an Orchestrator pulls prompt batches from a random data loader,
submits rollout requests across a pool of worker servers, and processes
trajectories out-of-order using PoolExecutionSession while dynamically
instantiating registered agent and environment pairs locally on worker nodes.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Sequence, Tuple

from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class MockRolloutWorker(abstract_worker.Worker):
  """Mock rollout worker bound to GrpcRemoteExecutionServer for RL loop demo."""

  def __init__(
      self,
      worker_id: str,
      min_delay_s: float = 0.01,
      max_delay_s: float = 0.05,
  ):
    self.worker_id = worker_id
    self._state = datatypes.WorkerState.READY
    self.min_delay_s = min_delay_s
    self.max_delay_s = max_delay_s
    self.policy_version = 0

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self._state = datatypes.WorkerState.STOPPED
    return datatypes.Response()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self._state, policy_version=self.policy_version
    )

  def _create_agent_env_pair(
      self, request: datatypes.RolloutRequest
  ) -> Tuple[Any, Any]:
    """Constructs agent and environment locally using dynamic registry keys."""
    agent_type = request.metadata.get("agent_type", "diagnostic")
    env_type = request.metadata.get("env_type", "k8s")

    agent_cls = registry.AGENT_REGISTRY.get(agent_type)
    env_cls = registry.ENV_REGISTRY.get(env_type)

    task_data = (
        request.prompt
        if isinstance(request.prompt, dict)
        else {"prompt": request.prompt}
    )
    task_data.update(request.metadata)

    agent_kwargs = request.metadata.get("agent_kwargs", {})
    if (
        "system_prompt" not in agent_kwargs
        and "system_prompt" in request.metadata
    ):
      agent_kwargs["system_prompt"] = request.metadata["system_prompt"]

    agent = agent_cls(**agent_kwargs)

    env_kwargs = {
        "task": task_data,
        "group_id": request.group_id,
        **request.metadata.get("env_kwargs", {}),
    }
    env = env_cls(**env_kwargs)
    return agent, env

  async def generate(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    """Simulates rollout generation with asynchronous out-of-order durations."""
    # Variable delay so completions return out-of-order
    delay = random.uniform(self.min_delay_s, self.max_delay_s)
    await asyncio.sleep(delay)

    # Dynamically construct agent and environment using registry
    agent, env = self._create_agent_env_pair(request)
    try:
      obs, _ = env.reset()
      action = agent.step(obs)
      obs, reward, done, info = env.step(action)
      assert done

      return datatypes.RolloutResponse(
          request_id=request.request_id,
          status="COMPLETED",
          env_reward=reward,
          policy_version=self.policy_version,
          metadata={
              "worker_id": self.worker_id,
              "prompt_id": request.prompt_id,
              "observation": obs,
              "action": str(action),
              "pod_id": info.get("pod_id"),
              "delay_s": delay,
          },
      )
    finally:
      env.close()

  def sync_weights(self, policy_version: int) -> datatypes.Response:
    """Simulates updating policy weights on the worker."""
    self.policy_version = policy_version
    return datatypes.Response()


def random_prompt_loader(
    num_batches: int = 3,
    batch_size: int = 4,
) -> List[List[Dict[str, Any]]]:
  """Generates synthetic training batches of prompts for the RL loop."""
  batches = []
  for b in range(num_batches):
    batch = []
    for i in range(batch_size):
      prompt_id = f"prompt_b{b}_i{i}"
      batch.append({
          "prompt_id": prompt_id,
          "prompt": f"Solve task {prompt_id} with tool use",
          "pod_id": f"k8s-pod-{b}-{i}",
          "difficulty": random.choice(["easy", "medium", "hard"]),
          "agent_type": "diagnostic",
          "env_type": "k8s",
      })
    batches.append(batch)
  return batches


async def run_rl_training_loop(
    worker_addresses: Sequence[str],
    num_epochs: int = 2,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
  """Runs a sample RL training loop using remote_execution and ActorPool."""
  pool = remote_execution.RoutingActorPool(worker_addresses)
  data_loader = random_prompt_loader(
      num_batches=num_epochs, batch_size=batch_size
  )
  epoch_metrics = []

  policy_version = 1

  for epoch_idx, batch in enumerate(data_loader):
    logging.info(
        "=== Starting RL Epoch %d (Policy Version %d, Batch Size %d) ===",
        epoch_idx,
        policy_version,
        len(batch),
    )

    completed_rollouts = []
    failed_requests = 0

    # Execute rollout requests across the worker pool with fault isolation
    async with pool.execution_session() as session:
      # 1. Dynamically submit rollout tasks for the batch
      for idx, prompt_item in enumerate(batch):
        req_id = f"epoch_{epoch_idx}_req_{idx}"
        request = datatypes.RolloutRequest(
            request_id=req_id,
            prompt_id=prompt_item["prompt_id"],
            prompt=prompt_item,
            target_policy_version=policy_version,
            metadata={
                "agent_type": prompt_item.get("agent_type", "diagnostic"),
                "env_type": prompt_item.get("env_type", "k8s"),
                "difficulty": prompt_item["difficulty"],
                "pod_id": prompt_item.get("pod_id", "pod-default"),
                "system_prompt": "You are an expert K8s agent.",
            },
        )
        await session.submit(req_id, "generate", request)

      # 2. Consume completed rollouts out-of-order as workers finish
      async for result, exc in session.as_completed():
        if exc is not None:
          logging.warning("Rollout task failed with error: %s", exc)
          failed_requests += 1
          continue

        assert isinstance(result, datatypes.RolloutResponse)
        completed_rollouts.append(result)
        logging.info(
            "  [Completed out-of-order] Request '%s' on worker '%s' | Action:"
            " '%s' | Obs: '%s' | Reward: %.2f",
            result.request_id,
            result.metadata.get("worker_id"),
            result.metadata.get("action"),
            result.metadata.get("observation"),
            result.env_reward,
        )

    # 3. Compute batch statistics and simulate policy update
    avg_reward = (
        sum(r.env_reward for r in completed_rollouts) / len(completed_rollouts)
        if completed_rollouts
        else 0.0
    )
    metrics = {
        "epoch": epoch_idx,
        "policy_version": policy_version,
        "completed": len(completed_rollouts),
        "failed": failed_requests,
        "avg_reward": avg_reward,
    }
    epoch_metrics.append(metrics)
    logging.info(
        "Epoch %d finished: Completed=%d, Failed=%d, Avg Reward=%.3f",
        epoch_idx,
        len(completed_rollouts),
        failed_requests,
        avg_reward,
    )

    # 4. Advance policy version and sync weights across workers
    policy_version += 1
    for actor in pool._actors:
      await actor.asubmit("sync_weights", policy_version)

  return epoch_metrics


async def start_mock_rollout_server(
    port: int,
    worker_id: str = "rollout-worker-1",
    min_delay_s: float = 0.01,
    max_delay_s: float = 0.05,
) -> remote_execution.GrpcRemoteExecutionServer:
  """Starts a MockRolloutWorker gRPC server on the specified port."""
  registry.auto_discover_modules("tunix.experimental.worker.examples")
  worker = MockRolloutWorker(
      worker_id=worker_id,
      min_delay_s=min_delay_s,
      max_delay_s=max_delay_s,
  )
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(port=port)
  return server
