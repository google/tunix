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

"""Tests for the AgenticGRPOOrchestrator seam routing."""

from absl.testing import absltest
from tunix.experimental.orchestrator import agentic_grpo_orchestrator
from tunix.rl.agentic import agentic_grpo_learner


class _FakeTrainerWorker:
  """Records train() calls."""

  def __init__(self):
    self.calls = []

  def train(self, chunks, eval_ds, skip_jit):
    self.calls.append((chunks, eval_ds, skip_jit))


class AgenticGRPOOrchestratorTest(absltest.TestCase):

  def test_reuses_the_agentic_grpo_learner_loop(self):
    # The orchestrator inherits train() and the seam methods from the learner.
    self.assertTrue(
        issubclass(
            agentic_grpo_orchestrator.AgenticGRPOOrchestrator,
            agentic_grpo_learner.GRPOLearner,
        )
    )

  def test_train_micro_batch_routes_to_trainer_worker(self):
    # Bypass GRPOLearner.__init__ (needs a full RLCluster) to exercise the seam
    # override in isolation.
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    worker = _FakeTrainerWorker()
    orchestrator._trainer_worker = worker

    orchestrator._train_micro_batch(["chunk0", "chunk1"], "eval_ds", False)

    self.assertEqual(worker.calls, [(["chunk0", "chunk1"], "eval_ds", False)])


if __name__ == "__main__":
  absltest.main()
