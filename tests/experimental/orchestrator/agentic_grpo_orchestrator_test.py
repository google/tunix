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
import types

from absl.testing import absltest
from tunix.experimental.orchestrator import agentic_grpo_orchestrator
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner


class _FakeTrainerWorker:
  """Records train() calls."""

  def __init__(self):
    self.calls = []

  def train(self, chunks, eval_ds, skip_jit):
    self.calls.append((chunks, eval_ds, skip_jit))


class _FakeScoringTrainerWorker:
  """Trainer worker that also exposes per_token_logps() (actor scoring)."""

  def __init__(self):
    self.calls = []

  def per_token_logps(self, **kwargs):
    self.calls.append(kwargs)
    return "actor_logps"


class _FakeMetricsTrainerWorker:
  """Trainer worker that trains and exposes a drainable metrics buffer."""

  def __init__(self, metrics):
    self.calls = []
    self.drained = 0
    self._metrics = metrics

  def train(self, chunks, eval_ds, skip_jit):
    self.calls.append((chunks, eval_ds, skip_jit))

  def drain_metrics(self):
    self.drained += 1
    return self._metrics


class _FakeWeightSync:
  """Records sync() calls."""

  def __init__(self):
    self.syncs = 0

  def sync(self):
    self.syncs += 1


class _FakeRolloutWorker:
  """Records generate() calls."""

  def __init__(self):
    self.calls = []

  def generate(self, **kwargs):
    self.calls.append(kwargs)
    return "worker_output"


class _FakeInferenceWorker:
  """Records per_token_logps() calls."""

  def __init__(self):
    self.calls = []

  def per_token_logps(self, **kwargs):
    self.calls.append(kwargs)
    return "worker_logps"


class _FakePerf:
  all_devices = ()

  def span(self, *args, **kwargs):
    return contextlib.nullcontext()


class _FakeCluster:
  """Minimal cluster exercising the in-process fallback paths."""

  def __init__(self):
    self.perf_v2 = _FakePerf()
    self.global_steps = 0
    self.synced = 0
    self.generate_calls = []
    self.ref_calls = []
    self.actor_calls = []
    self.buffered_metrics = []
    self.cluster_config = types.SimpleNamespace(
        training_config=types.SimpleNamespace(
            compute_logps_micro_batch_size=2
        )
    )

  def sync_weights(self):
    self.synced += 1

  def buffer_metrics(self, metrics, **kwargs):
    self.buffered_metrics.append(metrics)

  def generate(self, **kwargs):
    self.generate_calls.append(kwargs)
    return "cluster_output"

  def get_ref_per_token_logps(self, **kwargs):
    self.ref_calls.append(kwargs)
    return "ref_logps"

  def get_actor_per_token_logps(self, **kwargs):
    self.actor_calls.append(kwargs)
    return "actor_logps_inprocess"


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

  def test_train_micro_batch_pumps_metrics_from_worker(self):
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    worker = _FakeMetricsTrainerWorker({"loss": (0.5, None)})
    orchestrator._trainer_worker = worker
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    orchestrator._train_micro_batch(["c"], None, False)

    self.assertEqual(worker.calls, [(["c"], None, False)])
    self.assertEqual(worker.drained, 1)
    self.assertEqual(cluster.buffered_metrics, [{"loss": (0.5, None)}])

  def test_train_micro_batch_skips_metrics_without_drain(self):
    # A trainer worker that only implements train() (no drain_metrics) leaves the
    # shared logger untouched from the orchestrator side.
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    orchestrator._trainer_worker = _FakeTrainerWorker()
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    orchestrator._train_micro_batch(["c"], None, False)

    self.assertEqual(cluster.buffered_metrics, [])

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

  def test_generate_routes_to_rollout_worker(self):
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    worker = _FakeRolloutWorker()
    orchestrator._rollout_worker = worker

    out = orchestrator._generate(
        prompts=["p0"],
        apply_chat_template=True,
        trace_tags={"group_id": 1},
        max_generation_steps=None,
    )

    self.assertEqual(out, "worker_output")
    self.assertLen(worker.calls, 1)
    self.assertEqual(worker.calls[0]["prompts"], ["p0"])
    self.assertTrue(worker.calls[0]["apply_chat_template"])

  def test_generate_falls_back_to_in_process_cluster(self):
    # With no handle, the seam reuses the inherited in-process generate path
    # (always TRAIN mode, as in the base learner).
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    orchestrator._rollout_worker = None
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    out = orchestrator._generate(
        prompts=["p0"],
        apply_chat_template=True,
        trace_tags={},
        max_generation_steps=3,
    )

    self.assertEqual(out, "cluster_output")
    self.assertLen(cluster.generate_calls, 1)
    self.assertEqual(
        cluster.generate_calls[0]["mode"], rl_cluster_lib.Mode.TRAIN
    )

  def test_ref_per_token_logps_routes_to_inference_worker(self):
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    worker = _FakeInferenceWorker()
    orchestrator._inference_worker = worker

    out = orchestrator._ref_per_token_logps("pids", "cids", 0, 1)

    self.assertEqual(out, "worker_logps")
    self.assertLen(worker.calls, 1)
    self.assertEqual(worker.calls[0]["prompt_ids"], "pids")
    self.assertEqual(worker.calls[0]["completion_ids"], "cids")
    self.assertEqual(worker.calls[0]["pad_id"], 0)
    self.assertEqual(worker.calls[0]["eos_id"], 1)

  def test_ref_per_token_logps_falls_back_to_in_process_reference(self):
    # With no handle, the seam reuses the inherited in-process reference role,
    # deriving micro_batch_size from the cluster config as the base does.
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    orchestrator._inference_worker = None
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    out = orchestrator._ref_per_token_logps("pids", "cids", 0, 1)

    self.assertEqual(out, "ref_logps")
    self.assertLen(cluster.ref_calls, 1)
    self.assertEqual(cluster.ref_calls[0]["prompt_tokens"], "pids")
    self.assertEqual(cluster.ref_calls[0]["micro_batch_size"], 2)

  def test_actor_per_token_logps_routes_to_trainer_worker(self):
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    worker = _FakeScoringTrainerWorker()
    orchestrator._trainer_worker = worker

    out = orchestrator._actor_per_token_logps("pids", "cids", 0, 1)

    self.assertEqual(out, "actor_logps")
    self.assertLen(worker.calls, 1)
    self.assertEqual(worker.calls[0]["prompt_ids"], "pids")
    self.assertEqual(worker.calls[0]["completion_ids"], "cids")
    self.assertEqual(worker.calls[0]["pad_id"], 0)
    self.assertEqual(worker.calls[0]["eos_id"], 1)

  def test_actor_per_token_logps_falls_back_when_worker_lacks_capability(self):
    # A trainer worker that only implements train() (no per_token_logps) reuses
    # the in-process actor role, deriving micro_batch_size like the base.
    orchestrator = object.__new__(
        agentic_grpo_orchestrator.AgenticGRPOOrchestrator
    )
    orchestrator._trainer_worker = _FakeTrainerWorker()
    cluster = _FakeCluster()
    orchestrator.rl_cluster = cluster

    out = orchestrator._actor_per_token_logps("pids", "cids", 0, 1)

    self.assertEqual(out, "actor_logps_inprocess")
    self.assertLen(cluster.actor_calls, 1)
    self.assertEqual(cluster.actor_calls[0]["prompt_tokens"], "pids")
    self.assertEqual(cluster.actor_calls[0]["micro_batch_size"], 2)


if __name__ == "__main__":
  absltest.main()
