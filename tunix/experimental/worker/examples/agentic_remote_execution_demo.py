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

"""Demonstration of distributed RL rollout workflow with worker-side agent/env instantiation.

Shows how Orchestrator and RolloutWorker communicate over gRPC using
remote_execution primitives (GrpcRemoteExecutionServer, ActorHandle) while
instantiating stateful/unpicklable clients (e.g. K8s client) locally on worker nodes.
"""

import asyncio
import logging
from typing import Any, Dict, Tuple

from tunix.experimental.common import datatypes
from tunix.experimental.rl.agentic import registry
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class DistributedRolloutWorker(abstract_worker.Worker):
  """Worker bound to GrpcRemoteExecutionServer that handles rollout requests."""

  def __init__(
      self,
      worker_id: str,
  ):
    self.worker_id = worker_id
    self._state = datatypes.WorkerState.READY

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
    return datatypes.HealthReport(state=self._state)

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
    if "system_prompt" not in agent_kwargs and "system_prompt" in request.metadata:
      agent_kwargs["system_prompt"] = request.metadata["system_prompt"]

    agent = agent_cls(**agent_kwargs)

    env_kwargs = {
        "task": task_data,
        "group_id": request.prompt_id,
        **request.metadata.get("env_kwargs", {}),
    }
    env = env_cls(**env_kwargs)
    return agent, env

  async def generate(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    """Remote method exposed to Orchestrator over remote_execution RPC."""
    agent, env = self._create_agent_env_pair(request)
    try:
      obs, _ = env.reset()
      action = agent.step(obs)
      obs, reward, done, info = env.step(action)

      return datatypes.RolloutResponse(
          request_id=request.request_id,
          status="COMPLETED",
          env_reward=reward,
          metadata={
              "worker_id": self.worker_id,
              "observation": obs,
              "reward": reward,
          },
      )
    finally:
      env.close()


async def run_worker_node(
    port: int, worker_id: str = "rollout-worker-1"
) -> remote_execution.GrpcRemoteExecutionServer:
  """Spawns the worker RPC server using remote_execution capability."""
  # Auto-discover all modules containing registered agents and environments under worker package
  registry.auto_discover_modules("tunix.experimental.worker.examples")
  worker_instance = DistributedRolloutWorker(worker_id=worker_id)
  server = remote_execution.GrpcRemoteExecutionServer(worker_instance)
  await server.start_serving_async(port=port)
  logging.info(f"[Worker Node] Listening on port {port}...")
  return server


async def run_orchestrator_node(
    worker_address: str,
) -> datatypes.RolloutResponse:
  """Orchestrator reads data, creates RolloutRequest, and calls remote generate."""
  handle = remote_execution.ActorHandle.from_address(worker_address)

  single_example = {"prompt": "Diagnose pod failures", "pod_id": "k8s-pod-101"}
  request = datatypes.RolloutRequest(
      request_id="req_group4_pair0",
      prompt=single_example,
      prompt_id="group_4",
      group_offset_id="0",
      target_policy_version=1,
      metadata={
          "agent_type": "diagnostic",
          "env_type": "k8s",
          "system_prompt": "You are an expert K8s agent.",
      },
  )

  logging.info(
      f"[Orchestrator] Sending RolloutRequest '{request.request_id}' to"
      f" worker at {worker_address}..."
  )

  response: datatypes.RolloutResponse = await handle.asubmit("generate", request)
  return response
