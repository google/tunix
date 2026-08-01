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

"""Startup validation for the orchestrator control plane.

Fail-fast cross-checks run once, before the loop starts, against the worker
descriptions in the registry. This replaces scattered config `__post_init__`
checks with an extensible pipeline of category-based validators that reconcile
the run geometry against what the workers actually report through `info()`.

All failures across all validators are collected and raised together so one
run surfaces every misconfiguration at once.
"""

from typing import Protocol
from tunix.experimental.orchestrator import worker_registry
from tunix.rl import algorithm_config
from tunix.rl import rl_cluster


class StartupValidationError(ValueError):
  """Raised when startup validation finds one or more misconfigurations."""

  def __init__(self, errors: list[str]):
    self.errors = list(errors)
    joined = "\n  - ".join(self.errors)
    super().__init__(f"startup validation failed:\n  - {joined}")


class StartupValidator(Protocol):
  """Protocol for a startup validation category using existing Tunix configs."""

  def validate(
      self,
      registry: worker_registry.WorkerRegistry,
      alg_config: algorithm_config.AlgorithmConfig,
      training_config: rl_cluster.RLTrainingConfig,
  ) -> list[str]:
    """Returns a list of error messages; empty list if all checks pass."""
    ...


# Default validation pipeline executed at startup.
DEFAULT_VALIDATORS: tuple[StartupValidator, ...] = ()


def validate_startup(
    registry: worker_registry.WorkerRegistry,
    alg_config: algorithm_config.AlgorithmConfig,
    training_config: rl_cluster.RLTrainingConfig,
    *,
    validators: tuple[StartupValidator, ...] = DEFAULT_VALIDATORS,
) -> None:
  """Validates the run geometry against the registered workers.

  Args:
    registry: The populated worker registry.
    alg_config: The algorithm configuration containing sequence lengths and
      generation params.
    training_config: The training configuration containing batch sizes and
      lattice definitions.
    validators: Optional custom validation pipeline; defaults to
      `DEFAULT_VALIDATORS`.

  Raises:
    StartupValidationError: If any check fails; carries every failure message.
  """
  errors: list[str] = []
  for validator in validators:
    errors.extend(validator.validate(registry, alg_config, training_config))

  if errors:
    raise StartupValidationError(errors)
