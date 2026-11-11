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

"""Utilities for the v2 sampler."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import jax
import jax.numpy as jnp
from tunix.common import configs as common_configs
from tunix.common import datatypes
from tunix.experimental.generate import configs
from tunix.experimental.generate import request as request_lib

_GIB = 1 << 30


def _device_hbm_limit_bytes(device: Any) -> int:
  """Returns the total HBM bytes per JAX device from its `device_kind`."""
  kind = device.device_kind
  if "TPU" not in kind:
    raise ValueError(
        f"Device {device} does not report HBM memory_stats() and has "
        f"unsupported device_kind={kind!r}."
    )
  if kind.endswith(" lite"):
    kind = kind[: -len(" lite")] + "e"
  if kind in ("TPU v5p", "TPU v5"):
    return 95 * _GIB
  if kind == "TPU v5e":
    return 16 * _GIB
  if kind in ("TPU v6e", "TPU v4"):
    return 32 * _GIB
  if kind.startswith("TPU7") or kind == "TPU v7":
    return 96 * _GIB
  if kind.startswith("TPU8") or kind == "TPU v8i":
    return 144 * _GIB
  raise ValueError(f"Unknown TPU device_kind={kind!r} on {device}.")


def _live_arrays_hbm_usage_bytes(devices: list[Any]) -> dict[Any, int]:
  """Computes per-device HBM bytes in use from `jax.live_arrays()` (Pathways)."""
  used: dict[Any, int] = defaultdict(int)
  seen_buffers: set[int] = set()
  for array in jax.live_arrays():
    if array.is_deleted():
      continue
    for shard in array.addressable_shards:
      buf_id = id(shard.data)
      if buf_id not in seen_buffers:
        seen_buffers.add(buf_id)
        used[shard.data.device] += shard.data.nbytes
  return {d: used[d] for d in devices}


def resolve_max_device_size_gib(
    hbm_utilization: float,
    mesh: jax.sharding.Mesh,
) -> float:
  """Computes available per-device KV cache GiB from an HBM utilization cap.

  Matches vLLM's `determine_available_memory`: budgets `int(bytes_limit *
  hbm_utilization)` total HBM per device and subtracts `bytes_in_use` (e.g.
  already-allocated model weights), returning the minimum across local devices
  in GiB. Falls back to `jax.live_arrays()` + `device_kind` HBM limits when
  running under Pathways (where IFRT proxy devices do not implement
  `memory_stats()`).

  Args:
    hbm_utilization: Fraction of total device HBM (`bytes_limit`) to budget, in
      `(0, 1]`.
    mesh: The JAX mesh whose local devices are queried.

  Returns:
    The minimum available GiB per device across queried devices.

  Raises:
    ValueError: If `hbm_utilization` is out of `(0, 1]`, no local devices are
      available, `memory_stats()` does not report `bytes_limit` /
      `bytes_in_use`, or `bytes_in_use` already exceeds the utilization cap.
  """
  if not 0.0 < hbm_utilization <= 1.0:
    raise ValueError(
        "rollout_vanilla_v2_hbm_utilization must be in (0, 1], got"
        f" {hbm_utilization}."
    )
  devices = list(mesh.local_devices)
  if not devices:
    raise ValueError("No addressable JAX devices found to query HBM stats.")

  available_per_device: list[int] = []
  live_used_by_device: dict[Any, int] | None = None
  for device in devices:
    try:
      stats = device.memory_stats()
    except jax.errors.JaxRuntimeError:
      stats = None
    if not stats or "bytes_limit" not in stats or "bytes_in_use" not in stats:
      if live_used_by_device is None:
        live_used_by_device = _live_arrays_hbm_usage_bytes(devices)
      stats = {
          "bytes_limit": _device_hbm_limit_bytes(device),
          "bytes_in_use": live_used_by_device[device],
      }
    limit = int(stats["bytes_limit"])
    used = int(stats["bytes_in_use"])
    limit_cap = int(limit * hbm_utilization)
    avail = limit_cap - used
    if avail <= 0:
      raise ValueError(
          f"Insufficient HBM on device {device} for"
          f" rollout_vanilla_v2_hbm_utilization={hbm_utilization}:"
          f" bytes_in_use ({used}) >="
          f" int(bytes_limit ({limit}) * {hbm_utilization}) ({limit_cap})."
      )
    available_per_device.append(avail)

  return min(available_per_device) / _GIB


def to_cache_config(
    rollout_config: common_configs.RolloutConfig,
    dtype: jax.typing.DTypeLike,
    mesh: jax.sharding.Mesh,
) -> configs.CacheConfig:
  """Returns the cache config for the given rollout config, dtype, and mesh."""
  return configs.CacheConfig(
      max_device_size_gib=resolve_max_device_size_gib(
          rollout_config.rollout_vanilla_v2_hbm_utilization, mesh
      ),
      max_host_size_gib=rollout_config.rollout_vanilla_v2_host_size_gib,
      page_size=rollout_config.rollout_vanilla_v2_kv_cache_page_size,
      enable_prefix_caching=rollout_config.rollout_vanilla_v2_enable_prefix_caching,
      dtype=dtype,
  )


def to_scheduler_config(
    rollout_config: common_configs.RolloutConfig,
    cache_geometries: Mapping[str, datatypes.CacheGeometry],
) -> configs.SchedulerConfig:
  """Returns the scheduler config for the given rollout config and geometries."""
  # A sliding window layer must hold the KV of every token in its window while
  # it processes a prefill chunk, so a longer chunk size is truncated to the
  # largest power of 2 within the smallest window.
  window_sizes = [
      g.window_size
      for g in cache_geometries.values()
      if g.window_size is not None
  ]
  max_chunked_prefill_length = (
      rollout_config.rollout_vanilla_v2_chunked_prefill_length
  )
  if window_sizes:
    min_window_size = min(window_sizes)
    max_chunked_prefill_length = min(
        max_chunked_prefill_length,
        1 << (min_window_size.bit_length() - 1),
    )
  return configs.SchedulerConfig(
      max_num_batched_tokens=rollout_config.rollout_vanilla_v2_max_num_batched_tokens,
      max_num_seqs=rollout_config.rollout_vanilla_v2_max_num_seqs,
      max_chunked_prefill_length=max_chunked_prefill_length,
      num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
  )


def to_model_runner_config(
    rollout_config: common_configs.RolloutConfig,
    mesh: jax.sharding.Mesh,
) -> configs.ModelRunnerConfig:
  """Returns the model runner config for the given rollout config and mesh."""
  return configs.ModelRunnerConfig(
      max_top_k=rollout_config.top_k or -1,
      mesh=mesh,
      return_logprobs=rollout_config.return_logprobs,
      return_logits=rollout_config.return_logits,
      num_scheduler_steps=rollout_config.rollout_vanilla_v2_num_scheduler_steps,
      seed=0 if rollout_config.seed is None else int(rollout_config.seed),
  )


def verify_sampling_params(
    request_id: str,
    params: request_lib.SamplingParams,
    rollout_config: common_configs.RolloutConfig,
) -> None:
  """Raises a ValueError if the engine cannot sample with `params`."""
  if params.max_tokens < 1:
    raise ValueError(
        f"Request {request_id} has max_tokens={params.max_tokens}; it must"
        " be positive."
    )
  if params.temperature < 0.0:
    raise ValueError(
        f"Request {request_id} has temperature={params.temperature}; it"
        " must be non-negative."
    )
  if params.top_p is not None and not 0.0 < params.top_p <= 1.0:
    raise ValueError(
        f"Request {request_id} has top_p={params.top_p}; it must be in"
        " (0, 1]."
    )
  if params.top_k is not None:
    max_top_k = rollout_config.top_k or -1
    if max_top_k > 0 and params.top_k > max_top_k:
      raise ValueError(
          f"Request {request_id} has top_k={params.top_k}, over"
          f" max_top_k={max_top_k}."
      )

  unsupported = {
      "cache_salt": params.cache_salt is not None,
      "beam_size": params.beam_size is not None,
      "return_routed_experts": params.return_routed_experts,
      "routed_experts_prompt_start": params.routed_experts_prompt_start != 0,
  }
  for name, is_set in unsupported.items():
    if is_set:
      raise ValueError(
          f"Request {request_id} sets {name}, which the engine does not"
          " support."
      )
  # The model runner config fixes the outputs of every step, so a request
  # must ask for exactly those.
  if params.return_logprobs != rollout_config.return_logprobs:
    raise ValueError(
        f"Request {request_id} has return_logprobs={params.return_logprobs},"
        " but the engine has"
        f" return_logprobs={rollout_config.return_logprobs}."
    )
  if params.return_logits != rollout_config.return_logits:
    raise ValueError(
        f"Request {request_id} has return_logits={params.return_logits},"
        f" but the engine has return_logits={rollout_config.return_logits}."
    )


def cdiv(a: int, b: int) -> int:
  """Returns the ceiling of `a / b` for integers."""
  return -(-a // b)


def next_power_of_2(x: int) -> int:
  """Returns the next power of 2 that is not smaller than `x`."""
  x = int(x)
  if x <= 1:
    return 1
  return 1 << (x - 1).bit_length()


def get_dtype_packing(dtype: jax.typing.DTypeLike) -> int:
  """Returns the number of elements of `dtype` packed into 32 bits."""
  return max(1, 32 // jax.dtypes.itemsize_bits(dtype))


@jax.jit
def jit_scatter_last_column(
    target: jax.Array,
    source: jax.Array,
    target_idxs: jax.Array,
) -> jax.Array:
  """Writes the last column of `source` into `target` at `target_idxs`.

  Out-of-bounds indices in `target_idxs` are dropped.

  Args:
    target: Destination array to update.
    source: 2D array whose last column (`source[:, -1]`) supplies the values.
    target_idxs: 1D index array mapping each row of `source` into the flattened
      `target`.

  Returns:
    A copy of `target` with `source[:, -1]` written at `target_idxs`.
  """
  return (
      target.reshape(-1)
      .at[target_idxs]
      .set(source[:, -1], mode="drop")
      .reshape(target.shape)
  )


def derive_kv_geometry(
    config: configs.CacheConfig,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
  """Derives the packed shape of a KV cache element.

  When `num_kv_heads < tp_size`, the KV head count is padded up to `tp_size` so
  that each tensor-parallel shard holds one replicated KV head.

  Args:
    config: Cache configuration specifying dtype.
    num_kv_heads: Number of KV attention heads before tensor-parallel padding.
    head_dim: Dimension of each attention head.

  Returns:
    A `(packed_kv_dim, kv_packing, head_dim)` tuple representing the packed
    element shape of a single token in the KV cache.

  Raises:
    ValueError: If no JAX mesh is active, if `num_kv_heads` and `tp_size` are
      not divisible by one another, or if `2 * (num_kv_heads // tp_size)`
      (after padding) is not divisible by `kv_packing`.
  """
  mesh = jax.sharding.get_mesh()
  if mesh.empty:
    raise ValueError("An active JAX mesh is required to derive KV geometry.")
  tp_size = mesh.shape["tp"] if "tp" in mesh.shape else 1
  if num_kv_heads < tp_size:
    if tp_size % num_kv_heads != 0:
      raise ValueError(
          f"tp_size ({tp_size}) must be divisible by "
          f"num_kv_heads ({num_kv_heads})."
      )
    num_kv_heads = tp_size
  elif num_kv_heads % tp_size != 0:
    raise ValueError(
        f"num_kv_heads ({num_kv_heads}) must be divisible by "
        f"tp_size ({tp_size})."
    )

  kv_packing = get_dtype_packing(config.dtype)
  if (2 * (num_kv_heads // tp_size)) % kv_packing != 0:
    raise ValueError(
        f"2 * (num_kv_heads // tp_size) ({2 * (num_kv_heads // tp_size)}) must "
        f"be divisible by kv_packing ({kv_packing})."
    )

  packed_kv_dim = (2 * num_kv_heads) // kv_packing
  return (packed_kv_dim, kv_packing, head_dim)


def kv_cache_dp_axes(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> tuple[str, ...]:
  """Returns the mesh axes that index the DP groups (`'dp'` and `'fsdp'`)."""
  return tuple(axis for axis in mesh.axis_names if axis in ("dp", "fsdp"))


def derive_cache_sharding() -> jax.sharding.NamedSharding:
  """Derives the device sharding for a KV cache page pool from the active mesh.

  Returns:
    A `NamedSharding` partitioning pages (`axis=0`) across data-parallel mesh
    axes and packed KV heads (`axis=2`) across `"tp"`.

  Raises:
    ValueError: If no JAX mesh is active.
  """
  mesh = jax.sharding.get_mesh()
  if mesh.empty:
    raise ValueError(
        "An active JAX mesh is required to derive KV cache sharding."
    )

  # The cache has shape:
  #   (num_pages, page_size, packed_kv_dim, kv_packing, head_dim)
  #   i.e. (num_pages, num_elements_per_page, *element_shape)
  # Pages are sharded across DP and KV elements are sharded across TP.
  dp_axes = kv_cache_dp_axes(mesh)
  dp_size = math.prod(mesh.shape[axis] for axis in dp_axes)
  tp_axis = "tp" if "tp" in mesh.shape else None
  sharding_spec = jax.sharding.PartitionSpec(
      dp_axes if dp_size > 1 else None, None, tp_axis, None, None
  )
  return jax.sharding.NamedSharding(mesh, sharding_spec)


def compute_page_limits(
    config: configs.CacheConfig,
    element_shapes: Sequence[tuple[int, int, int]],
    sharding: jax.sharding.NamedSharding,
) -> tuple[int, int]:
  """Calculates the per-cache device and host page capacities.

  Each cache is assigned the same number of device and host pages.

  Args:
    config: Cache capacity configuration.
    element_shapes: The packed element shape of every cache, one entry each.
    sharding: The device sharding of every cache.

  Returns:
    A `(num_device_pages, num_host_pages)` tuple giving the device and host
    page capacities per cache.

  Raises:
    ValueError: If `element_shapes` is empty, or if `config.max_device_size_gib`
      cannot hold at least one page per data-parallel replica across all caches.
  """
  if not element_shapes:
    raise ValueError("element_shapes must not be empty.")

  dtype_bytes = jnp.dtype(config.dtype).itemsize

  # Pages are sharded across DP replicas, so they are allocated in groups of
  # `dp_size`, one page per replica. Each device only stores its shard of a
  # group.
  dp_axes = kv_cache_dp_axes(sharding.mesh)
  dp_size = math.prod(sharding.mesh.shape[axis] for axis in dp_axes)

  group_shapes = [
      sharding.shard_shape((dp_size, config.page_size, *shape))
      for shape in element_shapes
  ]
  bytes_per_device_per_group = dtype_bytes * sum(
      math.prod(shape) for shape in group_shapes
  )

  max_device_bytes = math.floor(config.max_device_size_gib * _GIB)
  num_groups = max_device_bytes // bytes_per_device_per_group
  if num_groups <= 0:
    raise ValueError(
        "Cannot allocate 0 device pages. max_device_size_gib="
        f"{config.max_device_size_gib} ({max_device_bytes} bytes) is smaller "
        f"than the {bytes_per_device_per_group} bytes each device needs to "
        f"hold one page on each of {dp_size} DP replica(s)."
    )
  num_device_pages = num_groups * dp_size

  # Host pages are not sharded, so the host stores every page whole.
  bytes_per_page = (
      dtype_bytes
      * config.page_size
      * sum(math.prod(shape) for shape in element_shapes)
  )
  max_host_bytes = math.floor(config.max_host_size_gib * _GIB)
  num_host_pages = max_host_bytes // bytes_per_page
  return num_device_pages, num_host_pages
