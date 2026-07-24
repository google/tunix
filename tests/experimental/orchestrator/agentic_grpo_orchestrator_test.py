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

import contextlib

from absl.testing import absltest
from tunix.experimental.orchestrator import agentic_grpo_orchestrator
from tunix.rl.agentic import agentic_grpo_learner


class _FakeTrainerWorker:
  """Records train() calls."""

  def __init__(self):
    self.calls = []

  def train(self, chunks, eval_ds, skip_jit):
    self.calls.append((chunks, eval_ds, skip_jit))


class _FakeWeightSync:
  """Records sync() calls."""

  def __init__(self):
    self.syncs = 0

  def sync(self):
    self.syncs += 1


class _FakePerf:
  all_devices = ()

  def span(self, *args, **kwargs):
    return contextlib.nullcontext()


class _FakeCluster:
  """Minimal cluster exercising the in-process weight-sync fallback."""

  def __init__(self):
    self.perf_v2 = _FakePerf()
    self.global_steps = 0
    self.synced = 0

  def sync_weights(self):
    self.synced += 1


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

  def test_sync_weights_routes_to_weight_sync_handle(self):
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    weight_sync = _FakeWeightSync()
    orchestrator._weight_sync = weight_sync

    orchestrator._sync_weights()

    self.assertEqual(weight_sync.syncs, 1)

  def test_sync_weights_falls_back_to_in_process_cluster(self):
    # With no handle, the seam reuses the inherited in-process sync path.
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    orchestrator._weight_sync = None
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    orchestrator._sync_weights()

    self.assertEqual(cluster.synced, 1)


if __name__ == "__main__":
  absltest.main()
