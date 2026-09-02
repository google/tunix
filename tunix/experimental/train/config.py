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

"""Training configuration for experimental trainers."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Tuple
import orbax.checkpoint as ocp
from tunix.perf import metrics as perf_metrics
from tunix.sft import metrics_logger
from tunix.sft import profiler

MetricsLoggerOptions = metrics_logger.MetricsLoggerOptions
ProfilerOptions = profiler.ProfilerOptions
PerfMetricsOptions = perf_metrics.PerfMetricsOptions


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
  """Canonical configuration for experimental step-level trainers.

  Standard fields cover core loop and infrastructure behaviors.
  Any unknown keyword arguments passed to `__init__` are automatically
  captured into `engine_kwargs`, enabling arbitrary engine-specific
  configuration without schema modifications.
  """

  # 1. Step & Cadence Controls
  eval_every_n_steps: int = 0
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None
  max_inflight_computations: int = 2

  # 2. Checkpointing
  checkpoint_root_directory: str | None = None
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # 3. Telemetry & Metrics
  metrics_logging_options: MetricsLoggerOptions | None = None
  profiler_options: ProfilerOptions | None = None
  perf_metrics_options: PerfMetricsOptions | None = None
  metrics_prefix: str = ""
  pbar_description: str | None = "Training"

  # 4. Distributed Sharding & Sequence Packing
  data_sharding_axis: Tuple[str, ...] = ("fsdp",)
  max_seq_token_per_tpu: int | None = None
  max_segments_per_packed_row: int | None = None

  # 5. Generic Engine-Specific Configuration Store
  engine_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __init__(
      self,
      *,
      eval_every_n_steps: int = 0,
      max_steps: int | None = None,
      gradient_accumulation_steps: int | None = None,
      checkpoint_root_directory: str | None = None,
      checkpointing_options: ocp.CheckpointManagerOptions | None = None,
      metrics_logging_options: MetricsLoggerOptions | None = None,
      profiler_options: ProfilerOptions | None = None,
      perf_metrics_options: PerfMetricsOptions | None = None,
      data_sharding_axis: Tuple[str, ...] = ("fsdp",),
      max_inflight_computations: int = 2,
      metrics_prefix: str = "",
      pbar_description: str | None = "Training",
      max_seq_token_per_tpu: int | None = None,
      max_segments_per_packed_row: int | None = None,
      engine_kwargs: Mapping[str, Any] | None = None,
      **extra_kwargs: Any,
  ):
    self.eval_every_n_steps = eval_every_n_steps
    self.max_steps = max_steps
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.checkpoint_root_directory = checkpoint_root_directory
    self.checkpointing_options = checkpointing_options
    self.metrics_logging_options = metrics_logging_options
    self.profiler_options = profiler_options
    self.perf_metrics_options = perf_metrics_options
    self.data_sharding_axis = tuple(data_sharding_axis)
    self.max_inflight_computations = max_inflight_computations
    self.metrics_prefix = metrics_prefix
    self.pbar_description = pbar_description
    self.max_seq_token_per_tpu = max_seq_token_per_tpu
    self.max_segments_per_packed_row = max_segments_per_packed_row

    # Merge explicit engine_kwargs and any extra arbitrary kwargs passed to __init__
    self.engine_kwargs = dict(engine_kwargs or {})
    self.engine_kwargs.update(extra_kwargs)

  def get(self, key: str, default: Any = None) -> Any:
    """Retrieves an attribute from standard fields first, then engine_kwargs."""
    if hasattr(self, key):
      val = getattr(self, key)
      return default if val is None else val
    if "engine_kwargs" in self.__dict__ and key in self.engine_kwargs:
      return self.engine_kwargs[key]
    return default

  def get_with_default(self, key: str, default: Any) -> Any:
    """Retrieves an attribute with fallback default (matches legacy PeftTrainer API)."""
    return self.get(key, default=default)

  def __getattr__(self, name: str) -> Any:
    """Allows accessing engine_kwargs directly as attributes (e.g. config.model_name)."""
    if "engine_kwargs" in self.__dict__ and name in self.engine_kwargs:
      return self.engine_kwargs[name]
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )

  def to_dict(self) -> dict[str, Any]:
    """Returns a flat/merged dictionary of all standard and engine-specific fields."""
    d = {
        "eval_every_n_steps": self.eval_every_n_steps,
        "max_steps": self.max_steps,
        "gradient_accumulation_steps": self.gradient_accumulation_steps,
        "checkpoint_root_directory": self.checkpoint_root_directory,
        "checkpointing_options": self.checkpointing_options,
        "metrics_logging_options": self.metrics_logging_options,
        "profiler_options": self.profiler_options,
        "perf_metrics_options": self.perf_metrics_options,
        "data_sharding_axis": self.data_sharding_axis,
        "max_inflight_computations": self.max_inflight_computations,
        "metrics_prefix": self.metrics_prefix,
        "pbar_description": self.pbar_description,
        "max_seq_token_per_tpu": self.max_seq_token_per_tpu,
        "max_segments_per_packed_row": self.max_segments_per_packed_row,
    }
    d.update(self.engine_kwargs)
    return d
