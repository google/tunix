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

"""Unit tests for synchronous RLProgram."""

import asyncio
from unittest import mock
from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.orchestrator import rl_program


class RLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock(spec=rl_engine_interface.AbstractRLEngine)
    self.mock_request = datatypes.RolloutRequest(
        request_id="r1",
        prompt_id="prompt1",
        prompt="prompt1",
    )
    mock_item = datatypes.TrajectoryItem(
        pair_index=0,
        group_id="prompt1",
        start_step=0,
        traj=datatypes.Trajectory(
            reward=1.0,
            status=datatypes.TrajectoryStatus.SUCCEEDED,
        ),
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        completion_tokens=np.array([3, 4], dtype=np.int32),
        action_mask=np.array([1, 1], dtype=np.int32),
    )
    self.mock_engine.generate = mock.AsyncMock(return_value=[mock_item])
    self.mock_engine.train_step = mock.AsyncMock(return_value="mock_train_result")
    self.mock_engine.sync_weights = mock.AsyncMock(return_value=1)

    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    mock_payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.full(4, 1.0, dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    self.mock_algo.create_trainer_payloads.return_value = [mock_payload]
    self.mock_algo.requires_reference_kl = False
    self.assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)

  def test_step_once_flow(self):
    begin_calls = []
    end_calls = []

    def on_begin(step):
      begin_calls.append(step)

    def on_end(step, result):
      end_calls.append((step, result))

    program = rl_program.SyncRLProgram(
        engine=self.mock_engine,
        algo=self.mock_algo,
        assembler=self.assembler,
        on_step_begin=on_begin,
        on_step_end=on_end,
    )

    res = program.step_once(prompts=[self.mock_request])

    self.assertEqual(res, "mock_train_result")
    self.mock_engine.generate.assert_called_once_with(
        prompts=[self.mock_request]
    )
    self.mock_algo.create_trainer_payloads.assert_called_once()
    self.mock_engine.train_step.assert_called_once()
    self.mock_engine.sync_weights.assert_called_once_with(role=datatypes.Role.ACTOR)
    self.assertEqual(program.step, 1)

    self.assertEqual(begin_calls, [0])
    self.assertEqual(end_calls, [(1, "mock_train_result")])
    self.assertIsNotNone(program.last_step_result)
    self.assertEqual(program.last_step_result.num_rollouts, 1)
    self.assertEqual(program.last_step_result.num_microbatches, 1)

  def test_step_once_can_skip_weight_sync(self):
    program = rl_program.SyncRLProgram(
        engine=self.mock_engine,
        algo=self.mock_algo,
        assembler=self.assembler,
        sync_weights=False,
    )

    res = program.step_once(prompts=[self.mock_request])

    self.assertEqual(res, "mock_train_result")
    self.mock_engine.sync_weights.assert_not_called()
    self.assertEqual(program.step, 1)
    self.assertIsNotNone(program.last_step_result)
    self.assertEqual(program.last_step_result.policy_version, 1)

  def test_run_accepts_orchestrator_supplied_engine(self):
    program = rl_program.SyncRLProgram(
        algo=self.mock_algo,
        assembler=self.assembler,
        sync_weights=False,
    )

    program.run(
        train_dataset=[[self.mock_request]],
        num_steps=1,
        engine=self.mock_engine,
    )

    self.mock_engine.generate.assert_called_once_with(
        prompts=[self.mock_request]
    )
    self.mock_engine.train_step.assert_called_once()
    self.assertEqual(program.step, 1)

  def test_run_uses_one_event_loop_for_all_async_engine_calls(self):
    loop_ids = []
    item = datatypes.TrajectoryItem(
        pair_index=0,
        group_id="prompt1",
        start_step=0,
        traj=datatypes.Trajectory(
            reward=1.0,
            status=datatypes.TrajectoryStatus.SUCCEEDED,
        ),
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        completion_tokens=np.array([3, 4], dtype=np.int32),
        action_mask=np.array([1, 1], dtype=np.int32),
    )

    class LoopTrackingEngine:

      def __init__(self):
        self.policy_version = 0

      async def generate(self, **kwargs):
        del kwargs
        loop_ids.append(id(asyncio.get_running_loop()))
        return [item]

      async def train_step(self, *args, **kwargs):
        del args, kwargs
        loop_ids.append(id(asyncio.get_running_loop()))
        return "train"

      async def sync_weights(self, **kwargs):
        del kwargs
        loop_ids.append(id(asyncio.get_running_loop()))
        self.policy_version += 1
        return self.policy_version

    program = rl_program.SyncRLProgram(
        engine=LoopTrackingEngine(),
        algo=self.mock_algo,
        assembler=self.assembler,
    )
    program.run(
        train_dataset=[[self.mock_request], [self.mock_request]],
        num_steps=2,
    )
    self.assertEqual(program.step, 2)
    self.assertLen(set(loop_ids), 1)

  def test_reference_logps_are_scored_from_padded_microbatch(self):
    item = datatypes.TrajectoryItem(
        pair_index=0,
        group_id="prompt1",
        start_step=0,
        traj=datatypes.Trajectory(
            reward=1.0,
            status=datatypes.TrajectoryStatus.SUCCEEDED,
        ),
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        completion_tokens=np.array([3, 4], dtype=np.int32),
        action_mask=np.array([1, 1], dtype=np.float32),
    )
    trainer_payload = datatypes.RLTrainerPayload(
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_mask=np.ones(4, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.ones(4, dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
    )
    algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    algo.create_trainer_payloads.return_value = [trainer_payload]
    algo.requires_reference_kl = True

    engine = mock.MagicMock(spec=rl_engine_interface.AbstractRLEngine)
    engine.generate = mock.AsyncMock(return_value=[item])
    engine.sync_weights = mock.AsyncMock(return_value=1)
    seen_logps_batches = []
    seen_train_batches = []

    async def _per_token_logps(role, items, **kwargs):
      del kwargs
      self.assertEqual(role, datatypes.Role.REFERENCE)
      seen_logps_batches.append(items)
      np.testing.assert_array_equal(
          np.asarray(items.prompt_ids),
          np.array([[0, 0, 1, 2]], dtype=np.int32),
      )
      np.testing.assert_array_equal(
          np.asarray(items.completion_ids),
          np.array([[3, 4, 0]], dtype=np.int32),
      )
      return np.full(np.asarray(items.completion_ids).shape, 0.25, dtype=np.float32)

    async def _train_step(payload, **kwargs):
      del kwargs
      seen_train_batches.append(payload)
      np.testing.assert_array_equal(
          np.asarray(payload.prompt_ids),
          np.asarray(seen_logps_batches[0].prompt_ids),
      )
      np.testing.assert_array_equal(
          np.asarray(payload.completion_ids),
          np.asarray(seen_logps_batches[0].completion_ids),
      )
      np.testing.assert_allclose(
          np.asarray(payload.ref_per_token_logps),
          np.full((1, 3), 0.25, dtype=np.float32),
      )
      return "mock_train_result"

    engine.per_token_logps = mock.AsyncMock(side_effect=_per_token_logps)
    engine.train_step = mock.AsyncMock(side_effect=_train_step)
    program = rl_program.SyncRLProgram(
        engine=engine,
        algo=algo,
        reward_fns=[lambda _: 1.0],
        assembler=batch_assembly.GRPOTrainExampleAssembler(
            batch_size=1,
            max_prompt_length=4,
            max_response_length=3,
            pad_id=0,
        ),
    )

    res = program.step_once(prompts=[self.mock_request])

    self.assertEqual(res, "mock_train_result")
    algo.create_trainer_payloads.assert_called_once_with(
        [item], rewards=[1.0]
    )
    engine.per_token_logps.assert_awaited_once()
    engine.train_step.assert_awaited_once()
    self.assertLen(seen_logps_batches, 1)
    self.assertLen(seen_train_batches, 1)

  def test_eval_step_once_flow(self):
    program = rl_program.SyncRLProgram(
        engine=self.mock_engine,
        algo=self.mock_algo,
        assembler=self.assembler,
    )
    eval_request = datatypes.RolloutRequest(
        request_id="eval_r1",
        prompt_id="eval_prompt",
        prompt="eval_prompt",
    )
    res = program.eval_step_once(prompts=[eval_request])

    self.assertLen(res, 1)
    self.mock_engine.generate.assert_called_once_with(prompts=[eval_request])
    self.mock_algo.create_trainer_payloads.assert_called_once()


if __name__ == "__main__":
  absltest.main()
