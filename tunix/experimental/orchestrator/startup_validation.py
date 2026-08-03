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
from tunix.rl import utils as rl_utils


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


class RunGeometryValidator:
  """Validates positive integer shapes and batch/group lattice divisibility."""

  def validate(
      self,
      registry: worker_registry.WorkerRegistry,
      alg_config: algorithm_config.AlgorithmConfig,
      training_config: rl_cluster.RLTrainingConfig,
  ) -> list[str]:
    del registry
    errors: list[str] = []

    # 1. Positivity & integer type check for key geometry fields.
    for name, value in [
        (
            "max_response_length",
            getattr(alg_config, "max_response_length", 1024),
        ),
        ("num_generations", getattr(alg_config, "num_generations", 1)),
        ("mini_batch_size", getattr(training_config, "mini_batch_size", None)),
    ]:
      try:
        rl_utils.is_positive_integer(value, name)
      except ValueError as e:
        errors.append(str(e))

    mini_batch_size = getattr(training_config, "mini_batch_size", None)
    num_generations = getattr(alg_config, "num_generations", 1)

    # 2. Batch/group lattice divisibility.
    if (
        isinstance(mini_batch_size, int)
        and isinstance(num_generations, int)
        and mini_batch_size > 0
        and num_generations > 0
    ):
      if mini_batch_size % num_generations != 0:
        errors.append(
            f"mini_batch_size {mini_batch_size} is not divisible by "
            f"num_generations (group_size) {num_generations}"
        )

    # 3. Batch/micro-batch divisibility.
    if isinstance(mini_batch_size, int) and mini_batch_size > 0:
      for mbs_name, mbs_val in [
          ("train_micro_batch_size", training_config.train_micro_batch_size),
          (
              "rollout_micro_batch_size",
              training_config.rollout_micro_batch_size,
          ),
          (
              "compute_logps_micro_batch_size",
              training_config.compute_logps_micro_batch_size,
          ),
      ]:
        if mbs_val is not None:
          try:
            rl_utils.is_positive_integer(mbs_val, mbs_name)
            if mini_batch_size % mbs_val != 0:
              errors.append(
                  f"mini_batch_size {mini_batch_size} is not divisible by "
                  f"{mbs_name} {mbs_val}"
              )
          except ValueError as e:
            errors.append(str(e))

    # 4. GRPO group size constraint.
    algo_variant = getattr(alg_config, "algo_variant", "")
    if algo_variant in ("grpo", "gspo-token") and (
        not isinstance(num_generations, int) or num_generations <= 1
    ):
      errors.append(
          f"num_generations (group_size) must be > 1 for {algo_variant!r}, "
          f"got {num_generations}"
      )

    return errors


# Default validation pipeline executed at startup.
DEFAULT_VALIDATORS: tuple[StartupValidator, ...] = (RunGeometryValidator(),)


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
