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

"""Tests for inprocess_workers."""

from unittest import mock
from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import inprocess_workers


class InProcessWorkersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock()
    self.mock_engine.cluster_config.training_config.compute_logps_micro_batch_size = (
        4
    )

  def test_trainer_worker_train(self):
    worker = inprocess_workers.InProcessTrainerWorker(self.mock_engine)
    worker.train(
        datatypes.Role.ACTOR,
        train_ds="chunks_data",
        eval_ds="eval_data",
        skip_jit=True,
    )
    self.mock_engine.update_actor.assert_called_once_with(
        "chunks_data", "eval_data", True
    )

    worker.train(
        datatypes.Role.CRITIC,
        train_ds="chunks_data",
        eval_ds="eval_data",
        skip_jit=False,
    )
    self.mock_engine.update_critic.assert_called_once_with(
        "chunks_data", "eval_data", False
    )

  def test_trainer_worker_per_token_logps(self):
    worker = inprocess_workers.InProcessTrainerWorker(self.mock_engine)
    self.mock_engine.per_token_logps.return_value = "logp_result"

    result = worker.per_token_logps(
        prompt_ids="prompts",
        completion_ids="completions",
        pad_id=0,
        eos_id=1,
    )

    self.assertEqual(result, "logp_result")
    self.mock_engine.per_token_logps.assert_called_once_with(
        datatypes.Role.ACTOR,
        prompt_tokens="prompts",
        completion_tokens="completions",
        pad_id=0,
        eos_id=1,
        micro_batch_size=4,
    )

  def test_trainer_worker_sync_weights(self):
    worker = inprocess_workers.InProcessTrainerWorker(self.mock_engine)
    worker.sync_weights()
    self.mock_engine.sync_weights.assert_called_once()

  def test_rollout_worker_generate(self):
    worker = inprocess_workers.InProcessRolloutWorker(self.mock_engine)
    self.mock_engine.generate.return_value = "rollout_output"

    result = worker.generate(
        prompts="prompts_data",
        apply_chat_template=True,
        mode="eval",
        micro_batch_size=2,
        trace_tags={"tag": "val"},
        max_generation_steps=100,
    )

    self.assertEqual(result, "rollout_output")
    self.mock_engine.generate.assert_called_once_with(
        prompts="prompts_data",
        apply_chat_template=True,
        mode="eval",
        micro_batch_size=2,
        trace_tags={"tag": "val"},
        max_generation_steps=100,
    )

  def test_rollout_worker_generate_default_args(self):
    worker = inprocess_workers.InProcessRolloutWorker(self.mock_engine)
    self.mock_engine.generate.return_value = "rollout_output"

    result = worker.generate(prompts="prompts_data")

    self.assertEqual(result, "rollout_output")
    self.mock_engine.generate.assert_called_once_with(
        prompts="prompts_data",
        apply_chat_template=False,
        mode=None,
        micro_batch_size=None,
        trace_tags=None,
        max_generation_steps=None,
    )

  def test_rollout_worker_sync_weights(self):
    worker = inprocess_workers.InProcessRolloutWorker(self.mock_engine)
    worker.sync_weights()
    self.mock_engine.sync_weights.assert_called_once()

  def test_inference_worker_per_token_logps(self):
    worker = inprocess_workers.InProcessInferenceWorker(self.mock_engine)
    self.mock_engine.per_token_logps.return_value = "ref_logp_result"

    result = worker.per_token_logps(
        prompt_ids="prompts",
        completion_ids="completions",
        pad_id=0,
        eos_id=1,
    )

    self.assertEqual(result, "ref_logp_result")
    self.mock_engine.per_token_logps.assert_called_once_with(
        datatypes.Role.REFERENCE,
        prompt_tokens="prompts",
        completion_tokens="completions",
        pad_id=0,
        eos_id=1,
        micro_batch_size=4,
    )


if __name__ == "__main__":
  absltest.main()
