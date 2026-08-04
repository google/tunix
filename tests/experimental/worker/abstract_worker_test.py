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

"""Tests for worker lifecycle transitions across all worker implementations."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.common import datatypes
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.worker import inference_worker
from tunix.experimental.worker import rollout_worker
from tunix.experimental.worker import trainer_worker

WorkerState = datatypes.WorkerState


class DummyScoringCore:

  def get_ref_per_token_logps(self, *args, **kwargs):
    pass

  def get_rewards(self, *args, **kwargs):
    pass


class DummyTrainer:

  def compile(self, *args, **kwargs):
    pass

  def close(self):
    pass


class AbstractWorkerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="inference_worker",
          worker_cls=inference_worker.InferenceWorker,
          module_path=(
              "tunix.experimental.worker.inference_worker.datatypes.Response"
          ),
          kwargs=dict(
              core=DummyScoringCore(),
              worker_id="w1",
              pad_id=0,
              eos_id=1,
          ),
      ),
      dict(
          testcase_name="rollout_worker",
          worker_cls=rollout_worker.RolloutWorker,
          module_path=(
              "tunix.experimental.worker.rollout_worker.datatypes.Response"
          ),
          kwargs=dict(
              worker_id="w2",
              tokenizer=mocks.MockTokenizer(),
              chat_parser=mocks.MockChatParser(),
          ),
      ),
      dict(
          testcase_name="trainer_worker",
          worker_cls=trainer_worker.TrainerWorker,
          module_path=(
              "tunix.experimental.worker.trainer_worker.datatypes.Response"
          ),
          kwargs=dict(trainer_factory=DummyTrainer, worker_id="w3"),
      ),
  )
  def test_lifecycle_state_transitions(self, worker_cls, module_path, kwargs):
    worker = worker_cls(**kwargs)
    self.assertEqual(worker.state, WorkerState.PENDING)

    with mock.patch(module_path) as mock_response:
      # Hook into the Response() creation inside the try-block to check state
      def verify_initializing():
        self.assertEqual(worker.state, WorkerState.INITIALIZING)
        return mock.DEFAULT

      mock_response.side_effect = verify_initializing
      worker.initialize()
      self.assertEqual(worker.state, WorkerState.READY)

      def verify_compiling():
        self.assertEqual(worker.state, WorkerState.COMPILING)
        return mock.DEFAULT

      mock_response.side_effect = verify_compiling
      worker.compile(None)
      self.assertEqual(worker.state, WorkerState.READY)

    worker.stop()
    self.assertEqual(worker.state, WorkerState.STOPPED)


if __name__ == "__main__":
  absltest.main()
