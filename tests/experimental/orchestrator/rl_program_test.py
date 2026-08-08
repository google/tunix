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
    mock_resp = datatypes.RolloutResponse(
        request_id="r1",
        status="COMPLETED",
        env_reward=1.0,
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        segments=[
            datatypes.TokenSegment(
                source="assistant",
                tokens=np.array([3, 4], dtype=np.int32),
                loss_mask=np.array([1, 1], dtype=np.int32),
            )
        ],
    )
    self.mock_engine.generate = mock.AsyncMock(return_value=[mock_resp])
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

    res = program.step_once(prompts=["prompt1"])

    self.assertEqual(res, "mock_train_result")
    self.mock_engine.generate.assert_called_once_with(prompts=["prompt1"])
    self.mock_algo.create_trainer_payloads.assert_called_once()
    self.mock_engine.train_step.assert_called_once()
    self.mock_engine.sync_weights.assert_called_once_with(role=datatypes.Role.ACTOR)
    self.assertEqual(program.step, 1)

    self.assertEqual(begin_calls, [0])
    self.assertEqual(end_calls, [(1, "mock_train_result")])

  def test_eval_step_once_flow(self):
    program = rl_program.SyncRLProgram(
        engine=self.mock_engine,
        algo=self.mock_algo,
        assembler=self.assembler,
    )
    res = program.eval_step_once(prompts=["eval_prompt"])

    self.assertLen(res, 1)
    self.mock_engine.generate.assert_called_once_with(prompts=["eval_prompt"])
    self.mock_algo.create_trainer_payloads.assert_called_once()


if __name__ == "__main__":
  absltest.main()
