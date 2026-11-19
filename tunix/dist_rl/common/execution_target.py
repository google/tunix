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

"""Typed execution target contract (`<framework>:<accelerator>`) and cluster invariants."""

from __future__ import annotations

import dataclasses
import enum

from tunix.common import datatypes as common_datatypes


class Framework(enum.StrEnum):
  """ML framework used by a distributed RL node role."""

  JAX = "jax"
  TORCH = "torch"


class Accelerator(enum.StrEnum):
  """Hardware accelerator family used by a distributed RL node role."""

  TPU = "tpu"
  GPU = "gpu"


class Placement(enum.StrEnum):
  """Physical placement mode across distributed RL worker roles."""

  COLOCATED = "colocated"
  DISAGGREGATED = "disaggregated"


@dataclasses.dataclass(frozen=True)
class ExecutionTarget:
  """Immutable `<framework>:<accelerator>` execution target for a worker role."""

  framework: Framework
  accelerator: Accelerator

  @classmethod
  def parse(cls, spec: str) -> ExecutionTarget:
    """Parses a `<framework>:<accelerator>` execution target string."""
    framework_raw, sep, accelerator_raw = spec.strip().lower().partition(":")
    if not sep or ":" in accelerator_raw:
      raise ValueError(
          f"Invalid execution target {spec!r}: expected"
          " '<framework>:<accelerator>' (e.g. 'jax:tpu', 'jax:gpu',"
          " 'torch:tpu')."
      )
    if framework_raw not in Framework:
      valid = ", ".join(repr(f.value) for f in Framework)
      raise ValueError(
          f"Unsupported framework {framework_raw!r} in execution target"
          f" {spec!r}; expected one of: {valid}."
      )
    if accelerator_raw not in Accelerator:
      valid = ", ".join(repr(a.value) for a in Accelerator)
      raise ValueError(
          f"Unsupported accelerator {accelerator_raw!r} in execution target"
          f" {spec!r}; expected one of: {valid}."
      )
    return cls(
        framework=Framework(framework_raw),
        accelerator=Accelerator(accelerator_raw),
    )

  def __str__(self) -> str:
    return f"{self.framework}:{self.accelerator}"


JAX_TPU = ExecutionTarget(Framework.JAX, Accelerator.TPU)
JAX_GPU = ExecutionTarget(Framework.JAX, Accelerator.GPU)
TORCH_TPU = ExecutionTarget(Framework.TORCH, Accelerator.TPU)
TORCH_GPU = ExecutionTarget(Framework.TORCH, Accelerator.GPU)

SUPPORTED_EXECUTION_TARGETS: frozenset[ExecutionTarget] = frozenset({
    JAX_TPU,
    JAX_GPU,
    TORCH_TPU,
})


@dataclasses.dataclass(frozen=True)
class ClusterTargets:
  """Resolved execution targets and topology invariants for a cluster run.

  Validates cross-role placement and transport invariants at construction time:
  - `Placement.COLOCATED`: Requires identical `ExecutionTarget` (`framework` and
    `accelerator`) across `trainer`, `rollout`, and `inference`.
  - `WeightSyncMode.FS`: File/object-store staged transport (`fs`) valid for any
    disaggregated `(trainer, rollout)` pair; orthogonal to checkpoint
    serialization format (`orbax` vs `safetensors`) and storage backend (`gcs`,
    `GOOGLE_INTERNAL_STOAGE_PATH`, or `local` POSIX paths resolved via `etils.epath.Path`).
  """

  trainer: ExecutionTarget = JAX_TPU
  rollout: ExecutionTarget = JAX_TPU
  inference: ExecutionTarget = JAX_TPU
  placement: Placement = Placement.DISAGGREGATED
  weight_sync_mode: common_datatypes.WeightSyncMode = (
      common_datatypes.WeightSyncMode.NONE
  )

  def __post_init__(self) -> None:
    for role, target in (
        ("trainer", self.trainer),
        ("rollout", self.rollout),
        ("inference", self.inference),
    ):
      if target not in SUPPORTED_EXECUTION_TARGETS:
        supported = ", ".join(sorted(str(t) for t in SUPPORTED_EXECUTION_TARGETS))
        raise ValueError(
            f"Unsupported execution target '{target}' for role {role!r};"
            f" supported targets are: {supported}."
        )
    if self.placement == Placement.COLOCATED and not (
        self.trainer == self.rollout == self.inference
    ):
      raise ValueError(
          "Placement 'colocated' requires identical execution targets"
          " (framework and accelerator) across trainer, rollout, and"
          f" inference, got trainer='{self.trainer}',"
          f" rollout='{self.rollout}', inference='{self.inference}'."
      )
    sync_checks = {
        common_datatypes.WeightSyncMode.IN_MEMORY: (
            self.trainer == self.rollout,
            "weight_sync_mode='in_memory' requires homogeneous trainer and"
            f" rollout execution targets, got trainer='{self.trainer}' and"
            f" rollout='{self.rollout}'.",
        ),
        common_datatypes.WeightSyncMode.RAIDEN: (
            self.trainer == JAX_TPU and self.rollout == JAX_TPU,
            "weight_sync_mode='raiden' requires both trainer and rollout"
            f" execution targets to be '{JAX_TPU}', got"
            f" trainer='{self.trainer}' and rollout='{self.rollout}'.",
        ),
        common_datatypes.WeightSyncMode.NCCL: (
            self.trainer.accelerator == Accelerator.GPU
            and self.rollout.accelerator == Accelerator.GPU,
            "weight_sync_mode='nccl' requires GPU accelerators on both trainer"
            f" and rollout, got trainer='{self.trainer}' and"
            f" rollout='{self.rollout}'.",
        ),
    }
    if (
        self.weight_sync_mode in sync_checks
        and not sync_checks[self.weight_sync_mode][0]
    ):
      raise ValueError(sync_checks[self.weight_sync_mode][1])
