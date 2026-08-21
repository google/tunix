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

"""Unit tests for DistributedRLEngine and WorkerPoolBalancer."""

import asyncio
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.worker import remote_execution


class MockActorHandle(mock.MagicMock):
  """A smart mock for ActorHandle that routes asubmit/dispatch_task to logical methods.

  This allows tests to write clean assertions like
  `worker.generate.assert_called_once()` while preserving the strict ActorHandle
  type requirements of the engine.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(spec=remote_execution.ActorHandle, *args, **kwargs)
    # Ensure all mocked methods return awaitables by default
    self.generate = mock.AsyncMock()
    self.poll_responses = mock.AsyncMock()
    self.weight_sync = mock.AsyncMock()
    self.fwd_bwd = mock.AsyncMock()
    self.update = mock.AsyncMock()
    self.prepare_weight_sync = mock.AsyncMock()
    self.release_weight_sync = mock.AsyncMock()
    self.pre_weight_sync = mock.AsyncMock()
    self.post_weight_sync = mock.AsyncMock()
    self.abort_weight_sync = mock.AsyncMock()
    self.score = mock.AsyncMock()
    self.per_token_logps = mock.AsyncMock()

  async def asubmit(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return await method(*args, **kwargs)

  async def dispatch_task(self, method_name: str, *args, **kwargs):
    method = getattr(self, method_name)
    return await method(*args, **kwargs)


class DistributedRLEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_rollout_1 = MockActorHandle()
    self.mock_rollout_2 = MockActorHandle()
    self.mock_actor = MockActorHandle()
    self.mock_ref = MockActorHandle()

    self.engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
    )

  def test_generate_load_balances_across_rollout_workers(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      resp2 = datatypes.RolloutResponse(
          request_id="r2", status="COMPLETED", env_reward=2.0
      )

      self.mock_rollout_1.generate.return_value = [resp1]
      self.mock_rollout_2.generate.return_value = [resp2]

      results = await self.engine.generate(["p1", "p2"])
      self.assertLen(results, 2)
      rewards = {res.traj.reward for res in results}
      self.assertEqual(rewards, {1.0, 2.0})

      # Verify underlying logical methods were called correctly
      self.assertEqual(self.mock_rollout_1.generate.call_count, 1)
      p1 = self.mock_rollout_1.generate.call_args.kwargs["prompts"][0]
      self.assertEqual(p1, "p1")

      self.assertEqual(self.mock_rollout_2.generate.call_count, 1)
      p2 = self.mock_rollout_2.generate.call_args.kwargs["prompts"][0]
      self.assertEqual(p2, "p2")

    asyncio.run(_run())

  def test_generate_uses_explicit_generation_args(self):
    async def _run():
      resp = datatypes.RolloutResponse(
          request_id="r1", status="COMPLETED", env_reward=1.0
      )
      self.mock_rollout_1.generate.return_value = [resp]
      results = await self.engine.generate(
          ["p1"],
          generation_args=datatypes.GenerationArgs(
              max_generation_steps=8,
              temperature=0.5,
              return_logprobs=False,
          ),
      )
      self.assertLen(results, 1)
      self.mock_rollout_1.generate.assert_called_once_with(
          prompts=["p1"],
          max_generation_steps=8,
          temperature=0.5,
          return_logprobs=False,
      )
    asyncio.run(_run())

  def test_generate_rejects_legacy_generation_kwargs(self):
    async def _run():
      with self.assertRaisesRegex(TypeError, "GenerationArgs"):
        await self.engine.generate(["p1"], temperature=0.5)
    asyncio.run(_run())

  def test_generate_routes_rollout_requests(self):
    async def _run():
      request = datatypes.RolloutRequest(
          request_id="r1",
          prompt="p1",
          prompt_id="prompt_1",
          generation_kwargs={"max_generation_steps": 8},
          metadata={"prefix_hash": 0},
      )
      resp = datatypes.RolloutResponse(
          request_id="r1",
          prompt_id="prompt_1",
          status="COMPLETED",
          env_reward=1.0,
      )
      self.mock_rollout_1.generate.return_value = [resp]

      results = await self.engine.generate([request])

      self.assertLen(results, 1)
      self.mock_rollout_1.generate.assert_called_once()
      self.assertEqual(
          self.mock_rollout_1.generate.call_args.kwargs["requests"], [request]
      )

    asyncio.run(_run())

  def test_poll_rollouts_aggregates_worker_responses(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(
          request_id="r1",
          status="COMPLETED",
          env_reward=1.0,
      )
      self.mock_rollout_1.poll_responses.return_value = [resp1]
      self.mock_rollout_2.poll_responses.return_value = []
      self.mock_rollout_1.asubmit = mock.AsyncMock(
          side_effect=AssertionError("polling must use ActorHandle.poll_responses")
      )
      self.mock_rollout_2.asubmit = mock.AsyncMock(
          side_effect=AssertionError("polling must use ActorHandle.poll_responses")
      )

      results = await self.engine.poll_rollouts(timeout_s=0.1)
      self.assertEqual(len(results), 1)
      self.assertEqual(results[0].traj.reward, 1.0)

      self.mock_rollout_1.poll_responses.assert_called_once_with(timeout_s=0.1)
      self.mock_rollout_2.poll_responses.assert_called_once_with(timeout_s=0.1)
      self.mock_rollout_1.asubmit.assert_not_called()
      self.mock_rollout_2.asubmit.assert_not_called()

    asyncio.run(_run())

  def test_train_step_routes_to_actor(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = {"loss": 0.5}
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
      )
      self.assertEqual(res, {"loss": 0.5})

      self.mock_actor.fwd_bwd.assert_called_once_with(
          payload=mock_payload,
          skip_jit=False,
      )
      self.mock_actor.update.assert_not_called()

    asyncio.run(_run())

  def test_train_step_applies_optimizer_on_last_microbatch(self):
    async def _run():
      self.mock_actor.fwd_bwd.return_value = datatypes.Response(
          metadata={"queued": True}
      )
      self.mock_actor.update.return_value = 3
      mock_payload = mock.MagicMock(spec=datatypes.RLTrainerPayload)

      res = await self.engine.train_step(
          mock_payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=True,
      )
      self.assertEqual(
          res,
          {
              "fwd_bwd": datatypes.Response(metadata={"queued": True}),
              "updated": True,
              "train_step": 3,
              "accumulated": True,
          },
      )

      self.mock_actor.fwd_bwd.assert_called_once_with(
          payload=mock_payload,
          skip_jit=False,
      )
      self.mock_actor.update.assert_called_once_with()

    asyncio.run(_run())

  def test_sync_weights_delegates_to_coordinator(self):
    async def _run():
      class _FakeResult:
        policy_version = 7

      class _FakeCoordinator:

        def __init__(self):
          self.calls = []

        async def sync(self, policy_version=0, **kwargs):
          self.calls.append(policy_version)
          _FakeResult.policy_version = policy_version
          return _FakeResult

      coordinator = _FakeCoordinator()
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
          trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
          inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
          weight_sync_coordinator=coordinator,
      )
      self.assertEqual(await engine.sync_weights(), 1)
      self.assertEqual(await engine.sync_weights(), 2)
      self.assertEqual(coordinator.calls, [1, 2])

    asyncio.run(_run())

  def test_sync_weights_requires_a_coordinator(self):
    async def _run():
      with self.assertRaises(RuntimeError):
        await self.engine.sync_weights()

    asyncio.run(_run())

  def test_dispatch_rollout_requests_with_prefix_routing(self):
    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="1",
          prompt="p1",
          prompt_id="1",
          metadata={"prefix_hash": 0},
      )
      req2 = datatypes.RolloutRequest(
          request_id="2",
          prompt="p2",
          prompt_id="2",
          metadata={"prefix_hash": 1},
      )

      req_ids = await self.engine.dispatch_rollout_requests([req1, req2])
      self.assertEqual(req_ids, ["1", "2"])

      # Due to deterministic hash logic, req1 -> rollout_1 and req2 -> rollout_2
      self.mock_rollout_1.generate.assert_called_once()
      dispatched_req1 = self.mock_rollout_1.generate.call_args.kwargs[
          "requests"
      ][0]
      self.assertEqual(dispatched_req1.request_id, "1")

      self.mock_rollout_2.generate.assert_called_once()
      dispatched_req2 = self.mock_rollout_2.generate.call_args.kwargs[
          "requests"
      ][0]
      self.assertEqual(dispatched_req2.request_id, "2")

    asyncio.run(_run())

  def test_dispatch_rollouts_delegates_to_dispatch_rollout_requests(self):
    async def _run():
      req1 = datatypes.RolloutRequest(
          request_id="1",
          prompt="p1",
          prompt_id="1",
          metadata={"prefix_hash": 0},
      )
      req2 = datatypes.RolloutRequest(
          request_id="2",
          prompt="p2",
          prompt_id="2",
          metadata={"prefix_hash": 1},
      )

      req_ids = await self.engine.dispatch_rollouts([req1, req2])
      self.assertEqual(req_ids, ["1", "2"])

      self.mock_rollout_1.generate.assert_called_once()
      self.mock_rollout_2.generate.assert_called_once()

    asyncio.run(_run())

  def test_dispatch_rollouts_expands_group_size(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          ["p1", "p2"], group_size=3, policy_version=5
      )
      self.assertLen(req_ids, 6)

      total_dispatched = 0
      for mock_worker in (self.mock_rollout_1, self.mock_rollout_2):
        for call in mock_worker.generate.call_args_list:
          reqs = call.kwargs["requests"]
          total_dispatched += len(reqs)
          for req in reqs:
            self.assertEqual(req.target_policy_version, 5)
            self.assertIn(req.metadata["pair_index"], (0, 1, 2))
      self.assertEqual(total_dispatched, 6)

    asyncio.run(_run())

  def test_dispatch_rollouts_auto_extracts_prompt_and_group_ids(self):
    async def _run():
      dict_item = {
          "prompt": "Solve math",
          "prompt_id": "math_1",
          "metadata": {"group_id": "grp_1", "pair_index": 99},
      }
      req_ids = await self.engine.dispatch_rollouts(
          [dict_item], group_size=2, policy_version=1
      )
      self.assertLen(req_ids, 2)

      all_dispatched = []
      for mock_worker in (self.mock_rollout_1, self.mock_rollout_2):
        for call in mock_worker.generate.call_args_list:
          all_dispatched.extend(call.kwargs["requests"])

      self.assertLen(all_dispatched, 2)
      pair_indices = {req.metadata["pair_index"] for req in all_dispatched}
      self.assertEqual(pair_indices, {0, 1})
      self.assertTrue(all(req.prompt == "Solve math" for req in all_dispatched))
      self.assertTrue(
          all(req.prompt_id == "math_1" for req in all_dispatched)
      )
      self.assertTrue(
          all(req.metadata["group_id"] == "grp_1" for req in all_dispatched)
      )

    asyncio.run(_run())

  def test_dispatch_rollouts_passes_generation_args_and_route_metadata(self):
    async def _run():
      gen_args = datatypes.GenerationArgs(
          temperature=0.7, max_generation_steps=128
      )
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}],
          group_size=1,
          generation_args=gen_args,
          route_metadata={"prefix_hash": "cache_key_1"},
      )
      self.assertLen(req_ids, 1)

      mock_call = (
          self.mock_rollout_1.generate.call_args
          or self.mock_rollout_2.generate.call_args
      )
      dispatched = mock_call.kwargs["requests"][0]
      self.assertEqual(
          dispatched.generation_kwargs,
          {"temperature": 0.7, "max_generation_steps": 128},
      )
      self.assertEqual(dispatched.metadata["prefix_hash"], "cache_key_1")

    asyncio.run(_run())

  def test_dispatch_rollouts_handles_none_metadata(self):
    async def _run():
      req_ids = await self.engine.dispatch_rollouts(
          [{"prompt": "p1", "prompt_id": "p1"}],
          group_size=1,
          metadata=None,
          route_metadata=None,
      )
      self.assertLen(req_ids, 1)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
