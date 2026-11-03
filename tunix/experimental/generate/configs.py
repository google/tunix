# Copyright 2026 The Tunix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration dataclasses for the v2 continuous batching generation engine."""

import dataclasses

import jax


@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheGeometry:
  """The shape and reach of one KV cache.

  Attributes:
    num_kv_heads: The number of KV heads in the model.
    head_dim: The dimension of each KV head.
    window_size: The window size of the KV cache. None if the cache is global.
  """

  num_kv_heads: int
  head_dim: int
  window_size: int | None = None

  def __post_init__(self):
    positive_checks = {
        "num_kv_heads": self.num_kv_heads,
        "head_dim": self.head_dim,
    }
    if self.window_size is not None:
      positive_checks["window_size"] = self.window_size

    for field_name, value in positive_checks.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheConfig:
  """Raw configuration parameters for KV Cache allocation and sharding.

  Attributes:
    max_device_size_gib: The maximum number of GiB to allocate on each device.
    max_host_size_gib: The maximum number of GiB to allocate on the host.
    page_size: The number of tokens per page.
    enable_prefix_caching: Whether pages may be shared between requests with a
      common prefix.
    dtype: The data type of KV cache entries.
    dp_axis: The mesh axis to shard pages across.
    tp_axis: The mesh axis to shard KV heads across.
    mesh: The mesh to shard the KV caches over. Required if `dp_axis` or
      `tp_axis` is set.
  """

  max_device_size_gib: float
  max_host_size_gib: float = 0.0
  page_size: int = 16
  enable_prefix_caching: bool = True
  dtype: jax.typing.DTypeLike
  dp_axis: str | None = None
  tp_axis: str | None = None
  mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    positive_checks = {
        "page_size": self.page_size,
        "max_device_size_gib": self.max_device_size_gib,
    }
    for field_name, value in positive_checks.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")

    if self.max_host_size_gib < 0:
      raise ValueError(
          f"max_host_size_gib cannot be negative, got {self.max_host_size_gib}."
      )

    if not (self.dp_axis or self.tp_axis):
      return
    if self.mesh is None:
      raise ValueError(
          "mesh is required when dp_axis or tp_axis is set, got "
          f"dp_axis={self.dp_axis!r}, tp_axis={self.tp_axis!r}. Pages are "
          "allocated outside any mesh context, so the mesh cannot be inferred."
      )
    if self.dp_axis == self.tp_axis:
      raise ValueError(
          "dp_axis and tp_axis must be different, got"
          f" {self.dp_axis!r} for both."
      )
    for axis in (self.dp_axis, self.tp_axis):
      if axis and axis not in self.mesh.shape:
        raise ValueError(
            f"Axis {axis!r} is not in the mesh axes {tuple(self.mesh.shape)}."
        )

  @property
  def dp_size(self) -> int:
    """The number of data parallel replicas, i.e. the size of `dp_axis`."""
    if not self.dp_axis or self.mesh is None:
      return 1
    return self.mesh.shape[self.dp_axis]

  @property
  def tp_size(self) -> int:
    """The tensor parallel size, i.e. the size of `tp_axis`."""
    if not self.tp_axis or self.mesh is None:
      return 1
    return self.mesh.shape[self.tp_axis]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
  """Configuration for the scheduler.

  Attributes:
    max_num_batched_tokens: The maximum number of tokens that can be scheduled
      in a single batch.
    max_num_seqs: The maximum number of sequences that can be scheduled in a
      single batch.
    max_chunked_prefill_length: The maximum size of a chunked prefill request
      in tokens.
    num_scheduler_steps: The number of model forward passes to execute per
      engine step.
  """

  max_num_batched_tokens: int
  max_num_seqs: int
  max_chunked_prefill_length: int
  num_scheduler_steps: int = 1

  def __post_init__(self):
    is_positive = {
        "max_num_batched_tokens": self.max_num_batched_tokens,
        "max_num_seqs": self.max_num_seqs,
        "max_chunked_prefill_length": self.max_chunked_prefill_length,
        "num_scheduler_steps": self.num_scheduler_steps,
    }
    for field_name, value in is_positive.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")

    if self.max_chunked_prefill_length > self.max_num_batched_tokens:
      raise ValueError(
          "max_chunked_prefill_length must be less than or equal to "
          f"max_num_batched_tokens. Got {self.max_chunked_prefill_length} "
          f"and {self.max_num_batched_tokens}."
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelRunnerConfig:
  """Configuration for the model runner.

  Attributes:
    max_top_k: The maximum `top_k` value allowed in `SamplingParams`.
      <= 0 to set no limit.
    mesh: The device mesh the transformer runs on.
    return_logprobs: Whether to return logprobs.
    return_logits: Whether to return logits.
    num_scheduler_steps: The number of model forward passes to execute per
      engine step.
    seed: The engine's single sampling seed. Each step's key is derived from it,
      and every row samples independently under that key.
  """

  max_top_k: int
  mesh: jax.sharding.Mesh
  return_logprobs: bool = False
  return_logits: bool = False
  num_scheduler_steps: int = 1
  seed: int = 0

  def __post_init__(self):
    if self.num_scheduler_steps < 1:
      raise ValueError(
          "num_scheduler_steps must be positive, got"
          f" {self.num_scheduler_steps}."
      )
