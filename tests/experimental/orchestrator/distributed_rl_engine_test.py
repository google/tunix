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


class DummyRollout:
  async def generate(self, prompts, **kwargs): pass
  async def poll_responses(self, timeout_s=0.1): pass
  async def poll(self, timeout_s=0.1): pass
  async def weight_sync(self, metadata): pass

class DummyTrainer:
  async def fwd_bwd(self, batch, accumulate_gradients, apply_optimizer, skip_jit, **kwargs): pass
  async def prepare_weight_sync(self): pass

class DummyRef:
  async def score(self, items, **kwargs): pass
  async def per_token_logps(self, items, **kwargs): pass

class DistributedRLEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_rollout_1 = mock.create_autospec(DummyRollout, instance=True)
    self.mock_rollout_2 = mock.create_autospec(DummyRollout, instance=True)
    self.mock_actor = mock.create_autospec(DummyTrainer, instance=True)
    self.mock_ref = mock.create_autospec(DummyRef, instance=True)

    self.engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[self.mock_rollout_1, self.mock_rollout_2],
        trainer_workers={datatypes.Role.ACTOR: self.mock_actor},
        inference_workers={datatypes.Role.REFERENCE: self.mock_ref},
    )
    
  async def _mock_invoke_worker(self, worker, method_name, **kwargs):
    method = getattr(worker, method_name)
    return await method(**kwargs)

  def test_generate_load_balances_across_rollout_workers(self):
    async def _run():
      resp1 = datatypes.RolloutResponse(request_id="r1", status="COMPLETED", env_reward=1.0)
      resp2 = datatypes.RolloutResponse(request_id="r2", status="COMPLETED", env_reward=2.0)
      self.mock_rollout_1.generate.return_value = [resp1]
      self.mock_rollout_2.generate.return_value = [resp2]

      results = await self.engine.generate(["p1", "p2"])
      self.assertEqual(len(results), 2)
      self.assertEqual(results[0].traj.reward, 1.0)
      self.assertEqual(results[1].traj.reward, 2.0)
      self.mock_rollout_1.generate.assert_called_once_with(prompts=["p1"])
      self.mock_rollout_2.generate.assert_called_once_with(prompts=["p2"])

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

      # We must let _poll_worker invoke the method naturally to test the real poll_rollouts
      # So we temporarily remove the _invoke_worker mock just for this test, as poll_rollouts doesn't use it for poll_responses if it exists
      
      results = await self.engine.poll_rollouts(timeout_s=0.1)
      self.assertEqual(len(results), 1)
      self.assertEqual(results[0].group_id, "default_prompt")
      self.assertEqual(results[0].traj.reward, 1.0)

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
          batch=mock_payload,
          accumulate_gradients=True,
          apply_optimizer=False,
          skip_jit=False,
      )

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

  def test_balancer_prefix_routing(self):
    req_hash0 = datatypes.RolloutRequest(prompt="p0", metadata={"prefix_hash": 0})
    req_hash1 = datatypes.RolloutRequest(prompt="p1", metadata={"prefix_hash": 1})

    idx0, w0 = self.engine._balancer.select_worker_for_request(req_hash0)
    idx1, w1 = self.engine._balancer.select_worker_for_request(req_hash1)

    self.assertEqual(idx0, 0)
    self.assertIs(w0, self.mock_rollout_1)
    self.assertEqual(idx1, 1)
    self.assertIs(w1, self.mock_rollout_2)


if __name__ == "__main__":
  absltest.main()
