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

"""Routing/delegation tests for OrchestratorRLEngine (no real compute)."""

from absl.testing import absltest
from types import SimpleNamespace
from tunix.experimental.orchestrator import orchestrator_rl_engine
from tunix.rl import rl_cluster as rl_cluster_lib


class _FakeBaseEngine:
  """Records the calls the orchestrator engine would fall back to."""

  def __init__(self):
    self.global_steps = 0
    self.generate_calls = []
    self.update_actor_calls = []
    self.update_critic_calls = []
    self.eval_actor_calls = []
    self.sync_calls = 0
    self.ref_calls = []
    self.actor_calls = []
    self.buffered_metrics = []
    # Arbitrary non-primitive surface delegated via __getattr__.
    self.cluster_config = "CLUSTER_CONFIG"
    self.rollout = "ROLLOUT"

  def get_rollout_config(self, mode):
    del mode
    return SimpleNamespace(temperature=0.75)

  def generate(self, *args):
    self.generate_calls.append(args)
    return "base_generate"

  def update_actor(self, *args):
    self.update_actor_calls.append(args)

  def eval_actor(self, *args):
    self.eval_actor_calls.append(args)

  def update_critic(self, *args):
    self.update_critic_calls.append(args)

  def sync_weights(self):
    self.sync_calls += 1

  def get_ref_per_token_logps(self, **kwargs):
    self.ref_calls.append(kwargs)
    return "base_ref"

  def get_actor_per_token_logps(self, **kwargs):
    self.actor_calls.append(kwargs)
    return "base_actor"

  def buffer_metrics(self, metrics, **kwargs):
    self.buffered_metrics.append(metrics)


class _FakeTrainerWorker:

  def __init__(self, with_logps=False):
    self.train_calls = []
    self.logps_calls = []
    if with_logps:
      self.per_token_logps = self._per_token_logps

  def train(self, chunks, eval_ds, skip_jit):
    self.train_calls.append((chunks, eval_ds, skip_jit))

  def _per_token_logps(self, **kwargs):
    self.logps_calls.append(kwargs)
    return "worker_actor"


class _FakeStepTrainerWorker:

  def __init__(self):
    self.fwd_bwd_calls = []
    self.update_calls = []
    self.eval_calls = []
    self.configure_calls = []

  def configure_grpo_loss(self, **kwargs):
    self.configure_calls.append(kwargs)
    return {"configured_loss": "grpo"}

  def fwd_bwd(self, chunk):
    self.fwd_bwd_calls.append(chunk)

  def update(self, skip_jit=False):
    self.update_calls.append(skip_jit)
    return 7

  def run_eval(self, eval_ds):
    self.eval_calls.append(eval_ds)


class _FakeRolloutWorker:

  def __init__(self):
    self.calls = []

  def generate(self, **kwargs):
    self.calls.append(kwargs)
    return "worker_generate"


class _FakeInferenceWorker:

  def __init__(self):
    self.calls = []

  def per_token_logps(self, **kwargs):
    self.calls.append(kwargs)
    return "worker_ref"


class _FakeWeightSync:

  def __init__(self):
    self.syncs = 0

  def sync(self):
    self.syncs += 1


class OrchestratorRlEngineTest(absltest.TestCase):

  def test_delegates_unrouted_surface_to_base(self):
    base = _FakeBaseEngine()
    cluster = orchestrator_rl_engine.OrchestratorRLEngine(base)
    # Arbitrary attributes delegate through to the base engine.
    self.assertEqual(cluster.cluster_config, "CLUSTER_CONFIG")
    self.assertEqual(cluster.rollout, "ROLLOUT")
    cluster.buffer_metrics({"loss": (1.0, None)})
    self.assertEqual(base.buffered_metrics, [{"loss": (1.0, None)}])

  def test_global_steps_reads_and_writes_the_base(self):
    base = _FakeBaseEngine()
    cluster = orchestrator_rl_engine.OrchestratorRLEngine(base)
    self.assertEqual(cluster.global_steps, 0)
    cluster.global_steps += 5
    self.assertEqual(base.global_steps, 5)
    self.assertEqual(cluster.global_steps, 5)

  def test_generate_routes_then_falls_back(self):
    base = _FakeBaseEngine()
    worker = _FakeRolloutWorker()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, rollout_worker=worker
    )
    self.assertEqual(routed.generate(["p"]), "worker_generate")
    self.assertLen(worker.calls, 1)
    self.assertEmpty(base.generate_calls)

    fallback = orchestrator_rl_engine.OrchestratorRLEngine(base)
    self.assertEqual(fallback.generate(["p"]), "base_generate")
    self.assertLen(base.generate_calls, 1)

  def test_update_actor_routes_then_falls_back(self):
    base = _FakeBaseEngine()
    worker = _FakeTrainerWorker()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, trainer_worker=worker
    )
    routed.update_actor(["c"], None, False)
    self.assertEqual(worker.train_calls, [(["c"], None, False)])
    self.assertEmpty(base.update_actor_calls)

    fallback = orchestrator_rl_engine.OrchestratorRLEngine(base)
    fallback.update_actor(["c"], None, False)
    self.assertEqual(base.update_actor_calls, [(["c"], None, False)])

  def test_update_actor_routes_to_fwd_bwd_update_api(self):
    base = _FakeBaseEngine()
    worker = _FakeStepTrainerWorker()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, trainer_worker=worker
    )
    self.assertEqual(routed.update_actor(["c0", "c1"], "eval", True), 7)
    self.assertEqual(worker.fwd_bwd_calls, ["c0", "c1"])
    self.assertEqual(worker.update_calls, [True])
    self.assertEqual(worker.eval_calls, ["eval"])
    self.assertEmpty(base.update_actor_calls)

  def test_eval_actor_routes_then_falls_back(self):
    base = _FakeBaseEngine()
    worker = _FakeStepTrainerWorker()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, trainer_worker=worker
    )
    routed.eval_actor(["eval"])
    self.assertEqual(worker.eval_calls, [["eval"]])
    self.assertEmpty(base.eval_actor_calls)

    fallback = orchestrator_rl_engine.OrchestratorRLEngine(base)
    fallback.eval_actor(["eval"])
    self.assertEqual(base.eval_actor_calls, [(["eval"],)])

  def test_sync_weights_routes_then_falls_back(self):
    base = _FakeBaseEngine()
    weight_sync = _FakeWeightSync()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, weight_sync=weight_sync
    )
    routed.sync_weights()
    self.assertEqual(weight_sync.syncs, 1)
    self.assertEqual(base.sync_calls, 0)

    fallback = orchestrator_rl_engine.OrchestratorRLEngine(base)
    fallback.sync_weights()
    self.assertEqual(base.sync_calls, 1)

  def test_ref_logps_routes_to_inference_then_falls_back(self):
    base = _FakeBaseEngine()
    inference = _FakeInferenceWorker()
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, inference_worker=inference
    )
    self.assertEqual(
        routed.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "worker_ref",
    )
    self.assertLen(inference.calls, 1)
    self.assertEqual(inference.calls[0]["temperature"], 0.75)
    self.assertEmpty(base.ref_calls)

    fallback = orchestrator_rl_engine.OrchestratorRLEngine(base)
    self.assertEqual(
        fallback.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "base_ref",
    )
    self.assertLen(base.ref_calls, 1)

  def test_actor_logps_routes_when_worker_capable_else_base(self):
    base = _FakeBaseEngine()
    capable = _FakeTrainerWorker(with_logps=True)
    routed = orchestrator_rl_engine.OrchestratorRLEngine(
        base, trainer_worker=capable
    )
    self.assertEqual(
        routed.get_actor_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "worker_actor",
    )
    self.assertLen(capable.logps_calls, 1)
    self.assertEmpty(base.actor_calls)

    # Trainer worker without per_token_logps -> in-process fallback.
    incapable = _FakeTrainerWorker(with_logps=False)
    fallback = orchestrator_rl_engine.OrchestratorRLEngine(
        base, trainer_worker=incapable
    )
    self.assertEqual(
        fallback.get_actor_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "base_actor",
    )
    self.assertLen(base.actor_calls, 1)

  def test_remote_only_routes_without_base_engine(self):
    trainer = _FakeStepTrainerWorker()
    rollout = _FakeRolloutWorker()
    inference = _FakeInferenceWorker()
    weight_sync = _FakeWeightSync()
    cluster_config = SimpleNamespace(
        rollout_config={
            rl_cluster_lib.Mode.TRAIN: SimpleNamespace(temperature=0.25),
        }
    )
    cluster = orchestrator_rl_engine.OrchestratorRLEngine(
        trainer_worker=trainer,
        rollout_worker=rollout,
        inference_worker=inference,
        weight_sync=weight_sync,
        cluster_config=cluster_config,
        pad_id=0,
        eos_id=2,
    )

    self.assertTrue(cluster.has_trainer_worker)
    self.assertEqual(cluster.rollout.pad_id(), 0)
    self.assertEqual(cluster.rollout.eos_id(), 2)
    self.assertEqual(
        cluster.get_rollout_config(rl_cluster_lib.Mode.TRAIN).temperature,
        0.25,
    )
    self.assertEqual(cluster.generate(["p"]), "worker_generate")
    self.assertEqual(cluster.update_actor(["chunk"], None, False), 7)
    self.assertEqual(
        cluster.configure_grpo_loss(num_generations=2),
        {"configured_loss": "grpo"},
    )
    self.assertEqual(
        cluster.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "worker_ref",
    )
    self.assertEqual(inference.calls[0]["temperature"], 0.25)
    cluster.sync_weights()
    self.assertEqual(weight_sync.syncs, 1)


if __name__ == "__main__":
  absltest.main()
