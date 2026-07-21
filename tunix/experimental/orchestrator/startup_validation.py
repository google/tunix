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
checks with a single place that reconciles the run geometry against what the
workers actually report through `info()`:

  * single-controller guard (v1 assumes one process per worker);
  * tokenizer agreement (all workers must share one tokenizer hash);
  * batch/group lattice (global batch divisible by group size);
  * trainer data-sharding fit (micro-batch divisible by the fsdp size, so packed
    rows do not defeat FSDP).

All failures are collected and raised together so one run surfaces every
misconfiguration at once.
"""

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry


class StartupValidationError(ValueError):
  """Raised when startup validation finds one or more misconfigurations."""

  def __init__(self, errors: list[str]):
    self.errors = list(errors)
    joined = "\n  - ".join(self.errors)
    super().__init__(f"startup validation failed:\n  - {joined}")


def validate_startup(
    registry: worker_registry.WorkerRegistry,
    shape_config: datatypes.ShapeConfig,
    *,
    group_size: int,
    global_batch_size: int,
    require_single_process: bool = True,
) -> None:
  """Validates the run geometry against the registered workers.

  Args:
    registry: The populated worker registry.
    shape_config: Declared shapes for the run.
    group_size: Number of samples per group (G); the group-relative unit.
    global_batch_size: Number of samples per global training batch.
    require_single_process: Enforce the v1 single-controller assumption
      (`process_count == 1` for every worker).

  Raises:
    StartupValidationError: If any check fails; carries every failure message.
  """
  errors: list[str] = []
  infos = registry.infos()

  if not infos:
    raise StartupValidationError(["no workers registered"])

  if shape_config.max_prompt_length <= 0:
    errors.append(
        f"shape_config.max_prompt_length must be positive, got "
        f"{shape_config.max_prompt_length}"
    )
  if shape_config.max_response_tokens <= 0:
    errors.append(
        f"shape_config.max_response_tokens must be positive, got "
        f"{shape_config.max_response_tokens}"
    )

  if group_size <= 0:
    errors.append(f"group_size must be positive, got {group_size}")
  if global_batch_size <= 0:
    errors.append(
        f"global_batch_size must be positive, got {global_batch_size}"
    )
  if group_size > 0 and global_batch_size > 0 and global_batch_size % group_size:
    errors.append(
        f"global_batch_size {global_batch_size} is not divisible by group_size "
        f"{group_size}"
    )

  if require_single_process:
    for info in infos:
      process_count = int(info.resources.get("process_count", 1))
      if process_count != 1:
        errors.append(
            f"worker {info.worker_id!r}: process_count={process_count}, "
            "expected 1 (single-controller v1)"
        )

  # Tokenizer agreement: every worker that declares a hash must agree.
  hashes = {
      info.worker_id: info.resources["tokenizer_hash"]
      for info in infos
      if "tokenizer_hash" in info.resources
  }
  if len(set(hashes.values())) > 1:
    errors.append(f"tokenizer hash mismatch across workers: {hashes}")

  # Trainer data-sharding fit: micro-batch sizes must divide the fsdp size so
  # packed rows keep enough rows to shard.
  for info in infos:
    if "trainer" not in info.roles:
      continue
    fsdp_size = info.resources.get("fsdp_size")
    if fsdp_size is None:
      continue
    fsdp_size = int(fsdp_size)
    for micro_batch_size in shape_config.micro_batch_sizes:
      if fsdp_size > 0 and micro_batch_size % fsdp_size:
        errors.append(
            f"trainer {info.worker_id!r}: micro_batch_size {micro_batch_size} "
            f"is not divisible by fsdp_size {fsdp_size}"
        )

  if errors:
    raise StartupValidationError(errors)
