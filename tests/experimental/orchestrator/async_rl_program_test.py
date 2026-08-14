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

"""Tests for AsyncRLProgram and StandardRLProgram."""

import asyncio
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import async_rl_program
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine


def _create_rollout_response(
    request_id: str,
    prompt_id: str,
    group_id: str,
    pair_index: int = 0,
    policy_version: int = 0,
    reward: float = 1.0,
) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id,
      prompt_id=prompt_id,
      status="COMPLETED",
      env_reward=reward,
      policy_version=policy_version,
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=np.array([3, 4], dtype=np.int32),
              loss_mask=np.array([1, 1], dtype=np.int32),
          )
      ],
      metadata={
          "group_id": group_id,
          "pair_index": pair_index,
      },
  )


class AsyncRLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock(spec=distributed_rl_engine.DistributedRLEngine)
    self.mock_engine.dispatch_rollouts = mock.AsyncMock()
    self.mock_engine.train_step = mock.AsyncMock(return_value="step_done")
    async def _mock_poll(*args, **kwargs):
      await asyncio.sleep(0.01)
      return []

    self.mock_engine.sync_weights = mock.AsyncMock(return_value=1)
    self.mock_engine.poll_rollouts = mock.AsyncMock(side_effect=_mock_poll)
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.group_size = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_turns = 1
    self.mock_algo.max_packed_len = 16
    self.mock_algo.requires_reference_kl = False

    mock_payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        loss_mask=np.array([0, 0, 1, 1], dtype=np.float32),
        advantages=np.full(4, 1.0, dtype=np.float32),
        action_mask=np.array([0, 0, 1, 1], dtype=np.float32),
    )
    self.mock_algo.create_trainer_payloads.return_value = [mock_payload, mock_payload]
    self.assembler = batch_assembly.SequencePackedBatchAssembler(max_packed_len=16)

  def test_initialization(self):
    program = async_rl_program.StandardRLProgram(
        dataset=["prompt_1"],
        algo=self.mock_algo,
        reward_fns=[lambda x: 1.0],
        assembler=self.assembler,
    )
    self.assertEqual(program.step, 0)
    self.assertEqual(program.group_size, 2)
    self.assertEqual(program.mini_batch_size, 1)
    self.assertIsNotNone(program.raw_q)
    self.assertIsNotNone(program.scored_q)

  def test_run_async_four_stages_with_long_polling(self):
    async def _run():
      poll_results = [
          [
              distributed_rl_engine._response_to_trajectory_item(_create_rollout_response(
                  "req_0_0", "prompt_0", "group_0", pair_index=0
              )),
              distributed_rl_engine._response_to_trajectory_item(_create_rollout_response(
                  "req_0_1", "prompt_0", "group_0", pair_index=1
              )),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll

      begin_steps = []
      end_steps = []

      def on_begin(step):
        begin_steps.append(step)

      def on_end(step, result):
        end_steps.append((step, result))

      program = async_rl_program.StandardRLProgram(
          dataset=["prompt_data_0"],
          algo=self.mock_algo,
          reward_fns=[lambda x: 1.0],
          assembler=self.assembler,
          on_step_begin=on_begin,
          on_step_end=on_end,
      )

      await program.run_async(self.mock_engine, num_steps=1)

      self.assertEqual(program.step, 1)
      self.assertEqual(begin_steps, [0])
      self.assertEqual(end_steps, [(1, "step_done")])
      self.assertEqual(self.mock_engine.dispatch_rollouts.call_count, 2)
      self.mock_engine.train_step.assert_called_once()
      self.mock_engine.sync_weights.assert_called_once_with(role=datatypes.Role.ACTOR)

    asyncio.run(_run())

  def test_stage_exception_aborts_queue_and_propagates(self):
    class FailingProgram(async_rl_program.StandardRLProgram):

      async def rollout_dispatch_stage(self, engine):
        del engine
        raise RuntimeError("Rollout worker cluster down!")

    async def _run():
      prog = FailingProgram(
          dataset=["prompt"],
          algo=self.mock_algo,
          assembler=self.assembler,
      )
      with self.assertRaises(RuntimeError) as cm:
        await prog.run_async(self.mock_engine, num_steps=1)
      self.assertIn("Rollout worker cluster down!", str(cm.exception))

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
