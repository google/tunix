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

"""Tests for the RLOrchestrator primitive API and the GRPO algorithm adapter."""

from absl.testing import absltest
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import rl_orchestrator
from tunix.rl import function_registry
from tunix.rl.agentic import agentic_grpo_learner


def _grpo_adapter():
  return algorithm_adapter.GRPOAdapter(
      agentic_grpo_learner.GRPOConfig(num_generations=2)
  )


class _FakeEngine:
  """Records the primitive calls the orchestrator delegates to."""

  def __init__(self):
    self.global_steps = 0
    self.calls = {}
    self.buffered = []
    self.buffered_async = []
    self.cluster_config = "CFG"
    self.perf_v2 = "PERF"
    self.rollout = "ROLLOUT"
    self.actor_trainer = "ACTOR_TRAINER"
    self.tokenizer = "TOK"
    self.r2m = "R2M"

  def buffer_metrics(self, metrics, **kwargs):
    self.buffered.append((metrics, kwargs))

  def buffer_metrics_async(self, metrics, **kwargs):
    self.buffered_async.append((metrics, kwargs))

  def get_rollout_config(self, mode):
    return ("ROLLOUT_CONFIG", mode)

  def generate(self, *args):
    self.calls["generate"] = args
    return "GEN"

  def update_actor(self, *args):
    self.calls["update_actor"] = args

  def sync_weights(self):
    self.calls["sync_weights"] = True

  def get_ref_per_token_logps(self, **kwargs):
    self.calls["ref"] = kwargs
    return "REF"

  def get_actor_per_token_logps(self, **kwargs):
    self.calls["actor"] = kwargs
    return "ACTOR"


class _FakeRemoteEngine(_FakeEngine):

  has_trainer_worker = True

  def __init__(self):
    super().__init__()
    self.rollout = SimpleNamespace(pad_id=lambda: 0, eos_id=lambda: 1)
    self.cluster_config = SimpleNamespace(
        training_config=SimpleNamespace(compute_logps_chunk_size=4)
    )
    self.actor_trainer = _FakeActorTrainer()

  def get_rollout_config(self, mode):
    del mode
    return SimpleNamespace(temperature=0.5)


class _FakeActorTrainer:

  def __init__(self):
    self.loss_fn = None
    self.has_aux = None
    self.gen_model_input_fn = None
    self.is_managed_externally = False

  def with_loss_fn(self, loss_fn, has_aux=False):
    self.loss_fn = loss_fn
    self.has_aux = has_aux
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn):
    self.gen_model_input_fn = gen_model_input_fn
    return self


def _make(cluster=None, algorithm=None):
  cluster = cluster or _FakeEngine()
  algorithm = algorithm or _grpo_adapter()
  return rl_orchestrator.RLOrchestrator(cluster, algorithm), cluster


class RlOrchestratorTest(absltest.TestCase):

  def test_generate_delegates_to_cluster(self):
    orch, cluster = _make()
    self.assertEqual(orch.generate(["p"]), "GEN")
    self.assertEqual(cluster.calls["generate"][0], ["p"])

  def test_train_step_delegates_to_update_actor(self):
    orch, cluster = _make()
    orch.train_step(["chunk"], eval_ds="ev", skip_jit=True)
    self.assertEqual(cluster.calls["update_actor"], (["chunk"], "ev", True))

  def test_sync_weights_delegates(self):
    orch, cluster = _make()
    orch.sync_weights()
    self.assertTrue(cluster.calls["sync_weights"])

  def test_reference_logps_delegates(self):
    orch, cluster = _make()
    self.assertEqual(
        orch.reference_logps("p", "c", pad_id=0, eos_id=2, micro_batch_size=4),
        "REF",
    )
    self.assertEqual(cluster.calls["ref"]["prompt_tokens"], "p")
    self.assertEqual(cluster.calls["ref"]["pad_id"], 0)
    self.assertEqual(cluster.calls["ref"]["micro_batch_size"], 4)

  def test_actor_logps_delegates(self):
    orch, cluster = _make()
    self.assertEqual(orch.actor_logps("p", "c", pad_id=0, eos_id=2), "ACTOR")
    self.assertEqual(cluster.calls["actor"]["completion_tokens"], "c")
    self.assertEqual(cluster.calls["actor"]["eos_id"], 2)

  def test_global_steps_reads_and_writes_cluster(self):
    orch, cluster = _make()
    self.assertEqual(orch.global_steps, 0)
    orch.global_steps += 3
    self.assertEqual(cluster.global_steps, 3)
    self.assertEqual(orch.global_steps, 3)

  def test_compute_advantages_matches_registry_estimator(self):
    # The adapter must reuse the shared estimator (not reimplement it), so the
    # orchestrator stays numerically identical to the agentic GRPO learner.
    orch, _ = _make()
    rewards = jnp.array([0.0, 1.0, 2.0, 3.0])
    out = orch.compute_advantages(rewards, num_generations=2)
    expected = function_registry.get_advantage_estimator("grpo")(
        rewards=rewards, num_generations=2
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))
    self.assertEqual(np.asarray(out).shape, (4,))

  def test_metrics_and_accessors_delegate_to_cluster(self):
    orch, cluster = _make()
    orch.buffer_metrics({"a": 1}, mode="train")
    orch.buffer_metrics_async({"b": 2}, step=3)
    self.assertEqual(cluster.buffered, [({"a": 1}, {"mode": "train"})])
    self.assertEqual(cluster.buffered_async, [({"b": 2}, {"step": 3})])
    self.assertEqual(orch.cluster_config, "CFG")
    self.assertEqual(orch.perf_v2, "PERF")
    self.assertEqual(orch.rollout, "ROLLOUT")
    self.assertEqual(orch.actor_trainer, "ACTOR_TRAINER")
    self.assertEqual(orch.tokenizer, "TOK")
    self.assertEqual(orch.r2m, "R2M")
    self.assertEqual(
        orch.get_rollout_config("train"), ("ROLLOUT_CONFIG", "train")
    )

  def test_exposes_cluster_and_algorithm(self):
    cluster = _FakeEngine()
    algorithm = _grpo_adapter()
    orch = rl_orchestrator.RLOrchestrator(cluster, algorithm)
    self.assertIs(orch.cluster, cluster)
    self.assertIs(orch.algorithm, algorithm)
    self.assertIsInstance(algorithm, algorithm_adapter.AlgorithmAdapter)

  def test_configure_trainer_uses_remote_safe_grpo_config(self):
    cluster = _FakeRemoteEngine()
    algorithm = _grpo_adapter()
    orch = rl_orchestrator.RLOrchestrator(cluster, algorithm)

    orch.configure_trainer()

    self.assertIsNotNone(cluster.actor_trainer.loss_fn)
    self.assertIsNotNone(cluster.actor_trainer.gen_model_input_fn)
    self.assertTrue(cluster.actor_trainer.has_aux)
    self.assertTrue(cluster.actor_trainer.is_managed_externally)
    self.assertEqual(algorithm.algo_config.temperature, 0.5)


if __name__ == "__main__":
  absltest.main()
