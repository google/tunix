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

"""Tests for DistributedRLEngine."""

import asyncio
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.worker import remote_execution


class DistributedRLEngineTest(absltest.TestCase):

  def test_implements_abstract_rl_engine_protocol(self):
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=mock.MagicMock(spec=remote_execution.ActorHandle),
        trainer_workers={
            datatypes.Role.ACTOR: mock.MagicMock(
                spec=remote_execution.ActorHandle
            )
        },
    )
    self.assertIsInstance(engine, rl_engine_interface.AbstractRLEngine)

  def test_routes_to_single_rollout_worker(self):
    async def _run():
      mock_rollout_worker = mock.MagicMock(spec=remote_execution.ActorHandle)
      mock_rollout_worker.asubmit = mock.AsyncMock(
          return_value=["generated_output"]
      )

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=mock_rollout_worker,
          trainer_workers={
              datatypes.Role.ACTOR: mock.MagicMock(
                  spec=remote_execution.ActorHandle
              )
          },
      )
      result = await engine.generate(prompts=["hello"])

      self.assertEqual(result, ["generated_output"])
      mock_rollout_worker.asubmit.assert_called_once_with(
          "generate",
          prompts=["hello"],
          apply_chat_template=False,
          mode=None,
          micro_batch_size=None,
          trace_tags=None,
          max_generation_steps=None,
      )

    asyncio.run(_run())

  def test_generate_shards_across_multiple_rollout_workers(self):
    async def _run():
      w1 = mock.MagicMock(spec=remote_execution.ActorHandle)
      w1.asubmit = mock.AsyncMock(return_value=["out1"])
      w2 = mock.MagicMock(spec=remote_execution.ActorHandle)
      w2.asubmit = mock.AsyncMock(return_value=["out2"])

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[w1, w2],
          trainer_workers={
              datatypes.Role.ACTOR: mock.MagicMock(
                  spec=remote_execution.ActorHandle
              )
          },
      )
      result = await engine.generate(prompts=["hello_1", "hello_2"])
      self.assertEqual(result, ["out1", "out2"])
      w1.asubmit.assert_called_once_with(
          "generate",
          prompts=["hello_1"],
          apply_chat_template=False,
          mode=None,
          micro_batch_size=None,
          trace_tags=None,
          max_generation_steps=None,
      )
      w2.asubmit.assert_called_once_with(
          "generate",
          prompts=["hello_2"],
          apply_chat_template=False,
          mode=None,
          micro_batch_size=None,
          trace_tags=None,
          max_generation_steps=None,
      )

    asyncio.run(_run())

  def test_init_raises_when_worker_is_none_or_empty(self):
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    with self.assertRaises(ValueError):
      distributed_rl_engine.DistributedRLEngine(
          rollout_workers=None,
          trainer_workers={datatypes.Role.ACTOR: mock_handle},
      )
    with self.assertRaises(ValueError):
      distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[],
          trainer_workers={datatypes.Role.ACTOR: mock_handle},
      )
    with self.assertRaises(ValueError):
      distributed_rl_engine.DistributedRLEngine(
          rollout_workers=mock_handle,
          trainer_workers={},
      )
    with self.assertRaises(ValueError):
      distributed_rl_engine.DistributedRLEngine(
          rollout_workers=mock_handle,
          trainer_workers={datatypes.Role.CRITIC: mock_handle},
      )

  def test_routes_to_trainer_worker(self):
    async def _run():
      mock_trainer_worker = mock.MagicMock(spec=remote_execution.ActorHandle)
      mock_trainer_worker.asubmit = mock.AsyncMock(return_value="step_ok")
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=mock.MagicMock(spec=remote_execution.ActorHandle),
          trainer_workers={datatypes.Role.ACTOR: mock_trainer_worker},
      )
      with self.assertRaises(NotImplementedError):
        engine.train(
            role=datatypes.Role.ACTOR,
            train_ds="train_ds",
            eval_ds="eval_ds",
        )

      res = await engine.train_step(
          batch="mock_batch",
          role=datatypes.Role.ACTOR,
      )
      self.assertEqual(res, "step_ok")
      mock_trainer_worker.asubmit.assert_called_once_with(
          "fwd_bwd",
          batch="mock_batch",
          skip_jit=False,
      )

    asyncio.run(_run())

  def test_sync_weights(self):
    async def _run():
      mock_trainer_worker = mock.MagicMock(spec=remote_execution.ActorHandle)
      mock_trainer_worker.submit.return_value = "sync_info_123"
      mock_rollout_worker = mock.MagicMock(spec=remote_execution.ActorHandle)
      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[mock_rollout_worker],
          trainer_workers={datatypes.Role.ACTOR: mock_trainer_worker},
      )
      await engine.sync_weights()
      mock_trainer_worker.submit.assert_called_once_with(
          "prepare_weight_sync"
      )
      mock_rollout_worker.submit.assert_has_calls([
          mock.call("pre_weight_sync", "sync_info_123"),
          mock.call("weight_sync", "sync_info_123"),
          mock.call("post_weight_sync", "sync_info_123"),
      ])

    asyncio.run(_run())

  def test_dispatch_generate_and_poll_rollouts_with_long_polling(self):
    async def _run():
      w1 = mock.MagicMock(spec=remote_execution.ActorHandle)
      w1.dispatch_task = mock.AsyncMock()
      mock_response = mock.MagicMock()
      mock_response.unwrap.return_value = ["resp1"]
      w1.poll_responses = mock.AsyncMock(return_value=mock_response)

      engine = distributed_rl_engine.DistributedRLEngine(
          rollout_workers=[w1],
          trainer_workers={
              datatypes.Role.ACTOR: mock.MagicMock(
                  spec=remote_execution.ActorHandle
              )
          },
      )
      req = datatypes.RolloutRequest(
          request_id="r1", prompt="p1", prompt_id="p1"
      )
      await engine.dispatch_generate([req])
      w1.dispatch_task.assert_called_once()

      completed = await engine.poll_rollouts(timeout_s=0.5)
      self.assertEqual(completed, ["resp1"])

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()

