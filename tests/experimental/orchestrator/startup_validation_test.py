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

"""Tests for orchestrator startup validation pipeline."""

from absl.testing import absltest
import optax
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.rl import algorithm_config
from tunix.rl import rl_cluster


class _MockValidator:

  def __init__(self, errors: list[str]):
    self._errors = errors

  def validate(
      self,
      registry: worker_registry.WorkerRegistry,
      alg_config: algorithm_config.AlgorithmConfig,
      training_config: rl_cluster.RLTrainingConfig,
  ) -> list[str]:
    del registry, alg_config, training_config
    return list(self._errors)


class StartupValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.registry = worker_registry.WorkerRegistry()
    self.alg_config = algorithm_config.AlgorithmConfig()
    self.training_config = rl_cluster.RLTrainingConfig(
        actor_optimizer=optax.identity(),
        eval_every_n_steps=10,
        mini_batch_size=8,
        train_micro_batch_size=2,
    )

  def test_default_empty_validators_pass(self):
    startup_validation.validate_startup(
        self.registry,
        self.alg_config,
        self.training_config,
        validators=(),
    )

  def test_passing_validator_does_not_raise(self):
    startup_validation.validate_startup(
        self.registry,
        self.alg_config,
        self.training_config,
        validators=(_MockValidator([]),),
    )

  def test_failing_validator_raises_error(self):
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          self.registry,
          self.alg_config,
          self.training_config,
          validators=(_MockValidator(["mock failure"]),),
      )
    self.assertEqual(ctx.exception.errors, ["mock failure"])
    self.assertIn("mock failure", str(ctx.exception))

  def test_multiple_validators_aggregate_errors(self):
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          self.registry,
          self.alg_config,
          self.training_config,
          validators=(
              _MockValidator(["failure 1"]),
              _MockValidator([]),
              _MockValidator(["failure 2", "failure 3"]),
          ),
      )
    self.assertEqual(
        ctx.exception.errors,
        ["failure 1", "failure 2", "failure 3"],
    )


if __name__ == "__main__":
  absltest.main()
