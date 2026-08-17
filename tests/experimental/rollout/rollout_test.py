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

"""Integration and unit tests for Distributed-RL Rollout Subsystem."""

import asyncio
from typing import List
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.common import datatypes
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.rl.agentic import registry
from tunix.experimental.rollout import collector
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker as worker


class RolloutWorkerTest(parameterized.TestCase):
  """End-to-end integration tests for RolloutWorker and Manager."""

  def setUp(self):
    super().setUp()
    self.env_pool = mocks.MockEnvironmentPool(
        pool_size=5, env_factory=registry.ENV_REGISTRY.get("mock_env")
    )
    self.sampler = mocks.MockBaseSamplerImpl(default_delay=0.05)
    self.service = worker.RolloutWorker(
        worker_id="test_worker_01",
        sampler=self.sampler,
        env_pool=self.env_pool,
        agent_factory=registry.AGENT_REGISTRY.get("mock_agent"),
        max_concurrency=10,
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )
    self.server = remote_execution.InProcessRemoteExecutionServer(self.service)
    self.actor_handle = remote_execution.InProcessActorHandle(self.server)
    self.service.start()

  def tearDown(self):
    super().tearDown()
    self.service.stop()

  def test_sampler_config_types(self):
    """Verifies sampler_type='vllm' raises NotImplementedError."""
    from tunix.experimental.rollout import manager as manager_lib  # pylint: disable=g-import-not-at-top

    config_vllm = worker.RolloutConfig(sampler_type="vllm")
    with self.assertRaises(NotImplementedError):
      manager_lib.RolloutManager(config=config_vllm)

  def test_single_trajectory_generation(self):
    """Verifies single multi-turn episode execution."""

    async def _run_test():
      req = datatypes.RolloutRequest(
          prompt_id="prompt_single",
          prompt="Solve task X",
          generation_kwargs={"delay_seconds": 0.02},
          max_turns=5,
      )
      trajectory = await self.actor_handle.asubmit("generate", req)
      self.assertEqual(trajectory.request_id, "traj_prompt_single")
      self.assertNotEmpty(trajectory.segments)

    asyncio.run(_run_test())

  def test_generate_async_option_1(self):
    """Verifies Option 1: asubmit coroutine keeping open connection for result."""

    async def _run_test():
      req = datatypes.RolloutRequest(
          prompt_id="prompt_async",
          prompt="Solve task Y",
          generation_kwargs={"delay_seconds": 0.01, "force_finish": True},
      )
      trajectory = await self.actor_handle.asubmit("generate", req)
      self.assertIsInstance(trajectory, datatypes.RolloutResponse)
      self.assertEqual(trajectory.request_id, "traj_prompt_async")

    asyncio.run(_run_test())

  def test_episode_error_handling(self):
    """Verifies error handling when episode execution raises an exception."""

    async def _run_test():
      req = datatypes.RolloutRequest(
          prompt_id="prompt_error",
          prompt="Solve failing task",
      )
      with mock.patch.object(
          collector.TrajectoryCollectorEngine,
          "run_episode",
          side_effect=RuntimeError("Simulated episode execution error"),
      ):
        res = await self.actor_handle.asubmit("generate", req)
      self.assertIsInstance(res, datatypes.RolloutResponse)
      self.assertEqual(res.request_id, "traj_prompt_error")
      self.assertEqual(res.status, "ERROR")
      self.assertEqual(res.error, "Simulated episode execution error")

    asyncio.run(_run_test())

  def test_out_of_order_batch_completion(self):
    """Verifies that trajectories with varying latency complete strictly out-of-order!"""

    async def _run_test():
      req_a = datatypes.RolloutRequest(
          prompt_id="slow_A",
          prompt="Task A",
          generation_kwargs={"delay_seconds": 0.15},
      )
      req_b = datatypes.RolloutRequest(
          prompt_id="fast_B",
          prompt="Task B",
          generation_kwargs={"delay_seconds": 0.01, "force_finish": True},
      )
      req_c = datatypes.RolloutRequest(
          prompt_id="med_C",
          prompt="Task C",
          generation_kwargs={"delay_seconds": 0.05},
      )

      _ = asyncio.create_task(
          self.actor_handle.asubmit("generate", [req_a, req_b, req_c])
      )

      completed_order: List[str] = []
      stream = self.service.as_completed_stream()
      for _ in range(3):
        traj = await stream.__anext__()
        completed_order.append(traj.request_id)

      self.assertEqual(
          completed_order, ["traj_fast_B", "traj_med_C", "traj_slow_A"]
      )

    asyncio.run(_run_test())

  def test_two_worker_orchestrator_coordination(self):
    """Verifies orchestrator coordinating 2 local rollout workers using open streams without webhooks."""

    async def _run_test():
      service_1 = worker.RolloutWorker(
          worker_id="slice_01",
          sampler=mocks.MockBaseSamplerImpl(
              sampler_name="sampler_slice_1", default_delay=0.02
          ),
          env_pool=mocks.MockEnvironmentPool(
              pool_size=2, env_factory=registry.ENV_REGISTRY.get("mock_env")
          ),
          agent_factory=registry.AGENT_REGISTRY.get("mock_agent"),
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
      )
      server_1 = remote_execution.InProcessRemoteExecutionServer(service_1)
      actor_1 = remote_execution.InProcessActorHandle(server_1)

      service_2 = worker.RolloutWorker(
          worker_id="slice_02",
          sampler=mocks.MockBaseSamplerImpl(
              sampler_name="sampler_slice_2", default_delay=0.04
          ),
          env_pool=mocks.MockEnvironmentPool(
              pool_size=2, env_factory=registry.ENV_REGISTRY.get("mock_env")
          ),
          agent_factory=registry.AGENT_REGISTRY.get("mock_agent"),
          tokenizer=mocks.MockTokenizer(),
          chat_parser=mocks.MockChatParser(),
      )
      server_2 = remote_execution.InProcessRemoteExecutionServer(service_2)
      actor_2 = remote_execution.InProcessActorHandle(server_2)

      req_1 = datatypes.RolloutRequest(
          prompt_id="req_worker_1",
          prompt="Task for slice 1",
          generation_kwargs={
              "delay_seconds": 0.02,
              "force_finish": True,
              "answer": "Solution_A",
          },
      )
      req_2 = datatypes.RolloutRequest(
          prompt_id="req_worker_2",
          prompt="Task for slice 2",
          generation_kwargs={
              "delay_seconds": 0.04,
              "force_finish": True,
              "answer": "Solution_B",
          },
      )

      # Option 1 (asubmit) futures across workers via actor handles
      futures = [
          actor_1.asubmit("generate", req_1),
          actor_2.asubmit("generate", req_2),
      ]

      received_trajectories = {}
      for completed_future in asyncio.as_completed(futures):
        traj = await completed_future
        received_trajectories[traj.request_id] = traj

      self.assertIn("traj_req_worker_1", received_trajectories)
      self.assertIn("traj_req_worker_2", received_trajectories)

      traj_1 = received_trajectories["traj_req_worker_1"]
      traj_2 = received_trajectories["traj_req_worker_2"]

      self.assertNotEmpty(traj_1.segments)
      self.assertNotEmpty(traj_2.segments)

    asyncio.run(_run_test())

  def test_actor_handle_native_invocations(self):
    """Verifies direct Actor-native protocol using ActorHandle and RoutingActorPool."""

    async def _run_test():
      assert self.actor_handle is not None
      handle = self.actor_handle

      # Direct async submission of pre_weight_sync and weight_sync over handle
      metadata = datatypes.WeightSyncMetadata(
          new_policy_version=333,
          transfer_mode="p2p",
          source_endpoints=["trainer:50051"],
          sharding_topology={"mesh": [2, 2]},
      )
      await handle.asubmit("pre_weight_sync", metadata)
      v = await handle.asubmit("weight_sync", metadata)
      self.assertEqual(v, 333)
      await handle.asubmit("post_weight_sync", metadata)

      # Direct coroutine execution of generate over handle
      req = datatypes.RolloutRequest(
          prompt_id="prompt_native_actor",
          prompt="Solve native actor task",
          generation_kwargs={"delay_seconds": 0.01, "force_finish": True},
      )
      traj = await handle.asubmit("generate", req)
      self.assertIsInstance(traj, datatypes.RolloutResponse)
      self.assertEqual(traj.request_id, "traj_prompt_native_actor")

      # Direct RoutingActorPool out-of-order streaming and string URI support
      pool = remote_execution.RoutingActorPool([handle])
      pool.add_actor("grpc://rollout-pod-2:50051")
      tasks = [(req.prompt_id, "generate", (req,), {})]
      results = []
      async for res in pool.as_completed_stream(tasks):
        results.append(res)
      self.assertLen(results, 1)
      self.assertEqual(results[0].request_id, "traj_prompt_native_actor")

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
