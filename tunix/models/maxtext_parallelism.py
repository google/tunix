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

"""Pipeline-parallel configuration helpers for MaxText-backed models."""

import dataclasses
import math
from typing import Any, Sequence

from tunix.utils import mesh as mesh_lib


# Keep this in the same order as MaxText's default ``mesh_axes``. Including
# singleton axes matters because MaxText logical partition rules may refer to
# them even when their parallelism degree is one.
MAXTEXT_MESH_AXIS_NAMES = (
    "diloco",
    "data",
    "stage",
    "fsdp",
    "fsdp_transpose",
    "context",
    "context_autoregressive",
    "tensor",
    "tensor_sequence",
    "expert",
    "autoregressive",
)


@dataclasses.dataclass(frozen=True, slots=True)
class MaxTextPipelineConfig:
  """Single-slice MaxText pipeline and tensor parallelism configuration.

  This helper deliberately configures only ICI parallelism. Multi-slice DCN
  layouts and custom MaxText mesh rules should continue to be configured
  directly in MaxText.

  Attributes:
    pipeline_parallelism: Number of pipeline stages (the ``stage`` mesh axis).
    tensor_parallelism: Tensor-parallel degree within each pipeline stage.
    data_parallelism: Data-parallel degree.
    fsdp_parallelism: FSDP degree.
    num_layers_per_pipeline_stage: Decoder layers executed by a stage during
      one pipeline repeat.
    num_pipeline_microbatches: Number of microbatches in a forward pass. When
      set, it must be a multiple of ``pipeline_parallelism``.
    pipeline_parallel_layers: Optional number of decoder layers assigned to
      the pipeline. ``None`` lets MaxText use all decoder layers.
    pipeline_delay_activation_forwarding: Whether MaxText should delay
      activation forwarding to expose communication/computation overlap.
    pipeline_fsdp_ag_once: Whether MaxText should all-gather FSDP weights once
      before pipeline execution.
    pipeline_fsdp_ag_per_repeat: Whether MaxText should prefetch FSDP weights
      before every circular-pipeline repeat.
  """

  pipeline_parallelism: int
  tensor_parallelism: int = 1
  data_parallelism: int = 1
  fsdp_parallelism: int = 1
  num_layers_per_pipeline_stage: int = 1
  num_pipeline_microbatches: int | None = None
  pipeline_parallel_layers: int | None = None
  pipeline_delay_activation_forwarding: bool = False
  pipeline_fsdp_ag_once: bool = False
  pipeline_fsdp_ag_per_repeat: bool = False

  def __post_init__(self):
    degrees = {
        "pipeline_parallelism": self.pipeline_parallelism,
        "tensor_parallelism": self.tensor_parallelism,
        "data_parallelism": self.data_parallelism,
        "fsdp_parallelism": self.fsdp_parallelism,
        "num_layers_per_pipeline_stage": self.num_layers_per_pipeline_stage,
    }
    for name, value in degrees.items():
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    if self.pipeline_parallelism < 2:
      raise ValueError(
          "pipeline_parallelism must be at least 2; use ordinary MaxText "
          "parallelism when no pipeline stages are required."
      )

    if self.num_pipeline_microbatches is not None:
      if (
          not isinstance(self.num_pipeline_microbatches, int)
          or isinstance(self.num_pipeline_microbatches, bool)
          or self.num_pipeline_microbatches <= 0
      ):
        raise ValueError(
            "num_pipeline_microbatches must be a positive integer or None, "
            f"got {self.num_pipeline_microbatches!r}."
        )
      if self.num_pipeline_microbatches % self.pipeline_parallelism:
        raise ValueError(
            "num_pipeline_microbatches must be divisible by "
            f"pipeline_parallelism ({self.pipeline_parallelism}), got "
            f"{self.num_pipeline_microbatches}."
        )
      if (
          self.pipeline_delay_activation_forwarding
          and self.num_pipeline_microbatches < 2 * self.pipeline_parallelism
      ):
        raise ValueError(
            "pipeline_delay_activation_forwarding requires at least twice "
            "as many microbatches as pipeline stages."
        )

    if self.pipeline_parallel_layers is not None:
      if (
          not isinstance(self.pipeline_parallel_layers, int)
          or isinstance(self.pipeline_parallel_layers, bool)
          or self.pipeline_parallel_layers <= 0
      ):
        raise ValueError(
            "pipeline_parallel_layers must be a positive integer or None, "
            f"got {self.pipeline_parallel_layers!r}."
        )
      layers_per_repeat = (
          self.pipeline_parallelism * self.num_layers_per_pipeline_stage
      )
      if self.pipeline_parallel_layers % layers_per_repeat:
        raise ValueError(
            "pipeline_parallel_layers must be divisible by pipeline stages "
            "times layers per stage; got "
            f"{self.pipeline_parallel_layers} % {layers_per_repeat}."
        )

    if self.pipeline_fsdp_ag_once and self.pipeline_fsdp_ag_per_repeat:
      raise ValueError(
          "pipeline_fsdp_ag_once and pipeline_fsdp_ag_per_repeat are "
          "mutually exclusive."
      )

  @property
  def required_device_count(self) -> int:
    """Number of accelerator devices required by this ICI layout."""
    return math.prod((
        self.pipeline_parallelism,
        self.tensor_parallelism,
        self.data_parallelism,
        self.fsdp_parallelism,
    ))

  @property
  def mesh_axis_shapes(self) -> tuple[int, ...]:
    """MaxText-compatible mesh shape, including singleton logical axes."""
    axis_sizes = {
        "data": self.data_parallelism,
        "stage": self.pipeline_parallelism,
        "fsdp": self.fsdp_parallelism,
        "tensor": self.tensor_parallelism,
    }
    return tuple(axis_sizes.get(axis, 1) for axis in MAXTEXT_MESH_AXIS_NAMES)

  def create_mesh(self, devices: Sequence[Any] | None = None):
    """Creates a MaxText-compatible JAX mesh for this configuration."""
    return mesh_lib.create_mesh(
        self.mesh_axis_shapes,
        MAXTEXT_MESH_AXIS_NAMES,
        devices=devices,
    )

  def validate_mesh(self, mesh: Any) -> None:
    """Validates that an existing mesh implements this exact ICI layout."""
    shape = getattr(mesh, "shape", None)
    if shape is None or not hasattr(shape, "get"):
      raise ValueError("mesh must expose a mapping-like shape attribute.")

    missing_axes = [
        axis for axis in MAXTEXT_MESH_AXIS_NAMES if shape.get(axis) is None
    ]
    if missing_axes:
      raise ValueError(
          "MaxText pipeline meshes must include all logical mesh axes, even "
          f"when their size is one. Missing axes: {missing_axes}."
      )

    mismatches = {
        axis: {"expected": expected, "actual": int(shape.get(axis))}
        for axis, expected in zip(
            MAXTEXT_MESH_AXIS_NAMES, self.mesh_axis_shapes
        )
        if int(shape.get(axis)) != expected
    }
    if mismatches:
      raise ValueError(
          f"mesh shape does not match MaxTextPipelineConfig: {mismatches}."
      )

    mesh_size = math.prod(int(shape.get(axis)) for axis in shape)
    if mesh_size != self.required_device_count:
      raise ValueError(
          f"mesh uses {mesh_size} devices, but this configuration requires "
          f"{self.required_device_count}."
      )

  def validate_batch_size(self, global_batch_size: int) -> None:
    """Validates pipeline microbatch divisibility for a global batch."""
    if not isinstance(global_batch_size, int) or global_batch_size <= 0:
      raise ValueError(
          "global_batch_size must be a positive integer, got "
          f"{global_batch_size!r}."
      )
    if (
        self.num_pipeline_microbatches is not None
        and global_batch_size % self.num_pipeline_microbatches
    ):
      raise ValueError(
          f"global_batch_size ({global_batch_size}) must be divisible by "
          "num_pipeline_microbatches "
          f"({self.num_pipeline_microbatches})."
      )

  def as_maxtext_kwargs(self) -> dict[str, Any]:
    """Returns validated ``AutoModel.from_pretrained`` MaxText overrides."""
    overrides: dict[str, Any] = {
        "ici_pipeline_parallelism": self.pipeline_parallelism,
        "ici_tensor_parallelism": self.tensor_parallelism,
        "ici_data_parallelism": self.data_parallelism,
        "ici_fsdp_parallelism": self.fsdp_parallelism,
        "num_layers_per_pipeline_stage": self.num_layers_per_pipeline_stage,
        "pipeline_delay_activation_forwarding": (
            self.pipeline_delay_activation_forwarding
        ),
        "pipeline_fsdp_ag_once": self.pipeline_fsdp_ag_once,
        "pipeline_fsdp_ag_per_repeat": self.pipeline_fsdp_ag_per_repeat,
    }
    if self.num_pipeline_microbatches is not None:
      overrides["num_pipeline_microbatches"] = self.num_pipeline_microbatches
    if self.pipeline_parallel_layers is not None:
      overrides["pipeline_parallel_layers"] = self.pipeline_parallel_layers
    return overrides
