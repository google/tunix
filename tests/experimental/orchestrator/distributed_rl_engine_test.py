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
      resp1 = datatypes.RolloutResponse(request_id="r1", status="COMPLETED", env_reward=1.0)
      resp2 = datatypes.RolloutResponse(request_id="r2", status="COMPLETED", env_reward=2.0)

      self.mock_rollout_1.generate.return_value = [resp1]
      self.mock_rollout_2.generate.return_value = [resp2]

      results = await self.engine.generate(["p1", "p2"])
      self.assertEqual(len(results), 2)
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

      results = await self.engine.poll_rollouts(timeout_s=0.1)
      self.assertEqual(len(results), 1)
      self.assertEqual(results[0].traj.reward, 1.0)

      self.mock_rollout_1.poll_responses.assert_called_once_with(timeout_s=0.1)
      self.mock_rollout_2.poll_responses.assert_called_once_with(timeout_s=0.1)

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

  def test_sync_weights_coordination(self):
    async def _run():
      mock_meta = datatypes.WeightSyncMetadata(
          new_policy_version=42,
          transfer_mode="p2p",
      )
      self.mock_actor.prepare_weight_sync.return_value = mock_meta
      self.mock_rollout_1.weight_sync.return_value = None
      self.mock_rollout_2.weight_sync.return_value = None

      ver = await self.engine.sync_weights(role=datatypes.Role.ACTOR)
      self.assertEqual(ver, 42)

      self.mock_actor.prepare_weight_sync.assert_called_once()
      self.mock_rollout_1.weight_sync.assert_called_once_with(metadata=mock_meta)
      self.mock_rollout_2.weight_sync.assert_called_once_with(metadata=mock_meta)

    asyncio.run(_run())

  def test_sync_weights_requires_weight_sync_metadata(self):
    async def _run():
      self.mock_actor.prepare_weight_sync.return_value = datatypes.Response()

      with self.assertRaisesRegex(RuntimeError, "WeightSyncMetadata"):
        await self.engine.sync_weights(role=datatypes.Role.ACTOR)

    asyncio.run(_run())

  def test_balancer_prefix_routing(self):

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

      await self.engine.dispatch_rollouts([req1, req2])

      # Due to deterministic round-robin / hash logic, req1 goes to rollout_1 and req2 goes to rollout_2
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

  def test_dispatch_rollouts_requires_strict_kwargs(self):
    async def _run():
      with self.assertRaisesRegex(ValueError, "prompt_ids' must be provided"):
        await self.engine.dispatch_rollouts(["p1", "p2"], policy_version=0)

      with self.assertRaisesRegex(ValueError, "match the length of prompts"):
        await self.engine.dispatch_rollouts(["p1", "p2"], prompt_ids=["id1"], policy_version=0)

      with self.assertRaisesRegex(ValueError, "policy_version' must be provided"):
        await self.engine.dispatch_rollouts(["p1", "p2"], prompt_ids=["id1", "id2"])

    asyncio.run(_run())

if __name__ == "__main__":
  absltest.main()
