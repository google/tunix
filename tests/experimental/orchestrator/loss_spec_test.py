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

"""A loss can be described to a trainer that runs somewhere else."""

import types
from typing import Any

from absl.testing import absltest
import cloudpickle
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import loss_spec as loss_spec_lib
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rpc_workers
from tunix.experimental.worker import remote_execution
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner


class _Trainer:
  """Records the wiring installed on it."""

  def __init__(self):
    self.loss_fn = None
    self.has_aux = None
    self.gen_model_input_fn = None
    self.is_managed_externally = False

  def with_loss_fn(self, loss_fn, has_aux=False):
    self.loss_fn = loss_fn
    self.has_aux = has_aux
    return self

  def with_gen_model_input_fn(self, fn):
    self.gen_model_input_fn = fn
    return self


class _Cluster:
  """The reads the adapter makes while describing the loss."""

  def __init__(self):
    self.actor_trainer = _Trainer()
    self.rollout = types.SimpleNamespace(pad_id=lambda: 0, eos_id=lambda: 2)
    self.cluster_config = types.SimpleNamespace(
        training_config=types.SimpleNamespace(compute_logps_chunk_size=8)
    )

  def get_rollout_config(self, mode):
    del mode
    return types.SimpleNamespace(temperature=0.6)


class _RemoteTrainer:
  """A served trainer that builds its own loss from a description."""

  def __init__(self):
    self.trainer = _Trainer()
    self.received = None

  def configure_loss(self, spec):
    self.received = spec
    spec.install_on(self.trainer)


def _config() -> agentic_grpo_learner.GRPOConfig:
  return agentic_grpo_learner.GRPOConfig(
      num_generations=2, num_iterations=1, beta=0.0, max_response_length=8
  )


class LossSpecTest(absltest.TestCase):

  def _spec(self) -> loss_spec_lib.LossSpec:
    return algorithm_adapter.GRPOAdapter(_config()).loss_spec(_Cluster())

  def test_the_description_captures_what_the_loss_needs(self):
    spec = self._spec()

    self.assertEqual(spec.policy_loss_fn, "grpo")
    self.assertEqual(spec.pad_id, 0)
    self.assertEqual(spec.eos_id, 2)
    self.assertEqual(spec.compute_logps_chunk_size, 8)

  def test_the_description_survives_the_wire(self):
    """A closure would not; that is the whole reason this exists."""
    spec = self._spec()

    restored = cloudpickle.loads(cloudpickle.dumps(spec))

    self.assertEqual(restored.policy_loss_fn, spec.policy_loss_fn)
    self.assertEqual(restored.pad_id, spec.pad_id)
    self.assertIsNotNone(restored.build_loss_fn())

  def test_installing_it_wires_the_trainer(self):
    trainer = _Trainer()

    self._spec().install_on(trainer)

    self.assertIsNotNone(trainer.loss_fn)
    self.assertTrue(trainer.has_aux)
    self.assertTrue(trainer.is_managed_externally)
    adapted = trainer.gen_model_input_fn("payload")
    self.assertEqual(adapted["train_example"], "payload")
    self.assertIn("algo_config", adapted)

  def test_configuring_without_a_handle_installs_locally(self):
    base = _Cluster()
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)

    algorithm_adapter.GRPOAdapter(_config()).configure_trainer(cluster)

    self.assertIsNotNone(base.actor_trainer.loss_fn)

  def test_a_trainer_handle_receives_the_description_instead(self):
    base = _Cluster()
    remote = _RemoteTrainer()
    handle = rpc_workers.RemoteTrainerWorker(
        remote_execution.InProcessActorHandle(
            remote_execution.InProcessRemoteExecutionServer(instance=remote)
        ),
        worker_id="trainer",
    )
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=handle
    )

    algorithm_adapter.GRPOAdapter(_config()).configure_trainer(cluster)

    # The far side built the loss; nothing was installed on the local trainer.
    self.assertIsNotNone(remote.received)
    self.assertIsNotNone(remote.trainer.loss_fn)
    self.assertTrue(remote.trainer.is_managed_externally)
    self.assertIsNone(base.actor_trainer.loss_fn)

  def test_the_same_description_builds_the_same_loss_on_both_sides(self):
    spec = self._spec()

    here = spec.build_loss_fn()
    there = cloudpickle.loads(cloudpickle.dumps(spec)).build_loss_fn()

    self.assertEqual(here.__name__, there.__name__)
    self.assertEqual(
        here.__defaults__[0].policy_loss_fn,
        there.__defaults__[0].policy_loss_fn,
    )


if __name__ == "__main__":
  absltest.main()
