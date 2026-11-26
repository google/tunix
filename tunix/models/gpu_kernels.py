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

"""GPU flash attention and ragged dot dispatchers backed by Tokamax Mosaic GPU."""

import functools
import math
from typing import Literal
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping

GpuAttentionBackend = Literal["cudnn", "mosaic", "triton", "xla", "xla_chunked"]
GpuRaggedDotBackend = Literal[
    "cutedsl", "mosaic", "mosaic_tpu_v2", "triton", "xla"
]

DEFAULT_GPU_ATTN_IMPLEMENTATIONS: tuple[GpuAttentionBackend, ...] = (
    "mosaic",
    "triton",
    "cudnn",
    "xla_chunked",
    "xla",
)
DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS: tuple[GpuRaggedDotBackend, ...] = (
    "mosaic",
    "triton",
    "xla",
)
_RAGGED_CONTRACTING_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def _is_sharded_axis_valid(
    axis: str | tuple[str, ...] | None,
    q_dim: int,
    k_dim: int,
    mesh_shape: dict[str, int],
) -> bool:
  """Returns whether `axis` is a valid multi-device sharding axis for Q and K."""
  if not axis:
    return False
  names = (axis,) if isinstance(axis, str) else tuple(axis)
  if not set(names).issubset(mesh_shape):
    return False
  shards = math.prod(map(mesh_shape.__getitem__, names))
  k_divisible = k_dim == 1 or k_dim % shards == 0
  return shards > 1 and q_dim % shards == 0 and k_divisible


def resolve_q_sharding(
    shd_spec: (
        shd.PartitionSpec | tuple[str | tuple[str, ...] | None, ...] | None
    ),
    query_shape: tuple[int, ...] | None = None,
    key_shape: tuple[int, ...] | None = None,
) -> shd.NamedSharding | None:
  """Resolves `shd_spec` against the active multi-device mesh when present.

  Drops axis names that are absent from the active mesh, have size `<= 1`, or do
  not evenly divide `query_shape` / `key_shape` (Tokamax shares `heads_axis`
  between `q` and `k` unless `k.shape[2] == 1`). Also clears `head_dim_axis`
  (index 3), which Tokamax does not allow to be sharded.

  Args:
    shd_spec: PartitionSpec or 4-tuple of axis names for `[B, T, N, H]`.
    query_shape: Optional query tensor shape `[B, T, N, H]`.
    key_shape: Optional key tensor shape `[B, S, K, H]`.

  Returns:
    Resolved `NamedSharding` when at least one axis is sharded across `> 1`
    devices, or `None` otherwise.
  """
  if shd_spec is None:
    return None
  mesh = shd.get_abstract_mesh()
  if mesh.empty:
    # TODO(tunix-dev): Remove deprecated physical_mesh fallback once callers
    # migrate to jax.set_mesh and shd.get_abstract_mesh().
    mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or mesh.size <= 1:
    return None
  q_dims = query_shape if query_shape is not None else (0, 0, 0, 0)
  k_dims = (
      (key_shape[0], 0, key_shape[2]) if key_shape is not None else (0, 0, 0)
  )
  axes = [
      ax
      if _is_sharded_axis_valid(ax, q_dims[i], k_dims[i], mesh.shape)
      else None
      for i, ax in enumerate(tuple(shd_spec)[:3])
  ] + [None]
  return (
      shd.NamedSharding(mesh, shd.PartitionSpec(*axes)) if any(axes) else None
  )


def _prepare_attention_mask(
    mask: jaxtyping.Array | None,
    segment_ids: jaxtyping.Array | None,
    q_len: int,
    kv_len: int,
    is_causal: bool,
    local_window_size: tuple[int, int] | None = None,
) -> jaxtyping.Array | None:
  """Builds a NaN-safe 4D boolean attention mask for Tokamax GPU kernels.

  When `mask` or `segment_ids` (intersected with `is_causal` and
  `local_window_size`) masks out all keys for a padding query row, online
  softmax in Tokamax GPU kernels (`use_stable_softmax=False` on SM90, Triton,
  cuDNN) divides by `l_i = 0`, producing `NaN` activations and gradients.
  Because `segment_ids` and `mask` are traced `jax.Array`s inside `jax.jit`, a
  rank-1 key mask `[B, 1, 1, S]` cannot simultaneously mask `<pad>` key 0 for
  real queries (`M(q, 0) = False` for `q > 0`) and keep `<pad>` query 0 valid
  (`M(0, 0) = True`). A rank-2 query-by-key mask `[B, 1, T, S]` with a diagonal
  self-attention guard on all-`False` query rows guarantees `l_i > 0` for
  `<pad>` rows while leaving every real query row (`has_valid_key == True`) 100%
  untouched.

  Args:
    mask: Optional 3D `[B, T, S]` or 4D `[B, N, T, S]` boolean attention mask.
    segment_ids: Optional 2D `[B, S]` token segment IDs.
    q_len: Query sequence length `T`.
    kv_len: Key/value sequence length `S`.
    is_causal: Whether causal masking is active.
    local_window_size: Optional `(left_window, right_window)` sliding window.

  Returns:
    4D boolean attention mask of shape `[B, 1|N, T, S]`, or `None` if unmasked.
  """
  if segment_ids is not None and segment_ids.shape[1] != kv_len:
    raise ValueError(
        f"segment_ids sequence length ({segment_ids.shape[1]}) must match "
        f"key sequence length ({kv_len})."
    )
  if mask is not None and mask.ndim not in (3, 4):
    raise ValueError(
        f"mask must be 3D or 4D, got ndim={mask.ndim} with shape={mask.shape}."
    )
  if mask is not None:
    mask = jnp.expand_dims(mask, axis=1) if mask.ndim == 3 else mask
    mask = mask[:, :, :q_len, :kv_len].astype(jnp.bool_)
  if segment_ids is not None:
    seg_mask = (
        segment_ids[:, None, :q_len, None]
        == segment_ids[:, None, None, :kv_len]
    )
    mask = seg_mask if mask is None else (mask & seg_mask)
  if mask is None or q_len > kv_len:
    return mask
  diag = jnp.eye(q_len, kv_len, dtype=jnp.bool_)
  effective_mask = jnp.tril(mask) if is_causal else mask
  if local_window_size is not None:
    left_window, right_window = local_window_size
    dist = jnp.arange(q_len)[:, None] - jnp.arange(kv_len)[None, :]
    effective_mask = (
        effective_mask & (dist <= left_window) & (-dist <= right_window)
    )
  has_valid_key = jnp.any(effective_mask, axis=-1, keepdims=True)
  return jnp.where(has_valid_key, mask, diag)


def supports_mosaic_gpu_attention(device: jax.Device | None = None) -> bool:
  """Returns whether `device` (or default device) supports SM90+ Mosaic attention.

  Tokamax's Pallas Mosaic GPU flash attention targets SM90+ (Hopper/Blackwell)
  hardware TMA; its experimental SM80 `cp.async` path is incompatible with JAX's
  logical-shape `run_scoped` SMEM lowering, so pre-Hopper GPUs (`SM80`/`SM89`)
  dispatch via Triton, cuDNN, and chunked XLA instead.

  Args:
    device: Optional JAX device to inspect (defaults to the active default
      device).

  Returns:
    `True` when `device` is an NVIDIA GPU with compute capability `>= 9.0`.
  """
  if device is None:
    if jax.default_backend() != "gpu":
      return False
    device = jax.local_devices(backend="gpu")[0]
  if device.platform != "gpu" or "NVIDIA" not in device.device_kind:
    return False
  return float(device.compute_capability) >= 9.0


def resolve_attn_implementations(
    implementation: tuple[GpuAttentionBackend, ...],
    *,
    logits_soft_cap: float | None = None,
    mosaic_supported: bool | None = None,
) -> tuple[GpuAttentionBackend, ...]:
  """Filters `implementation` for GPU architecture and `logits_soft_cap` support.

  Args:
    implementation: Ordered candidate Tokamax attention backends.
    logits_soft_cap: Optional tanh soft-cap bound applied to attention logits.
    mosaic_supported: Optional override for `supports_mosaic_gpu_attention()`.

  Returns:
    Filtered non-empty tuple of supported candidate backends.

  Raises:
    NotImplementedError: If no candidate backend remains after filtering.
  """
  allow_mosaic = (
      supports_mosaic_gpu_attention()
      if mosaic_supported is None
      else mosaic_supported
  )
  drop_triton = logits_soft_cap is not None and len(implementation) > 1
  filtered = tuple(
      impl
      for impl in implementation
      if (allow_mosaic or impl != "mosaic")
      and (not drop_triton or impl != "triton")
  )
  if not filtered:
    raise NotImplementedError(
        "No supported GPU attention backend remains after filtering "
        f"implementation={implementation!r} (mosaic_supported={allow_mosaic}, "
        f"logits_soft_cap={logits_soft_cap}). Tokamax Mosaic GPU flash "
        "attention requires an NVIDIA SM90+ (Hopper/Blackwell) GPU."
    )
  return filtered


def gpu_flash_attention(
    query: jaxtyping.Array,
    key: jaxtyping.Array,
    value: jaxtyping.Array,
    *,
    scale: float | None = None,
    is_causal: bool = True,
    mask: jaxtyping.Array | None = None,
    segment_ids: jaxtyping.Array | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    q_sharding: shd.NamedSharding | None = None,
    query_sharding_spec: (
        shd.PartitionSpec | tuple[str | tuple[str, ...] | None, ...] | None
    ) = None,
    implementation: tuple[GpuAttentionBackend, ...] = (
        DEFAULT_GPU_ATTN_IMPLEMENTATIONS
    ),
) -> jaxtyping.Array:
  """Runs flash attention on GPU via Tokamax (Mosaic GPU primary backend).

  Args:
    query: Query tensor of shape `[B, T, N, H]`.
    key: Key tensor of shape `[B, S, K, H]`.
    value: Value tensor of shape `[B, S, K, h]`.
    scale: Attention logit scale factor (`1 / sqrt(head_dim)` when `None`).
    is_causal: Whether to apply causal masking.
    mask: Optional boolean mask of shape `[B, T, S]` or broadcastable to `[B, N,
      T, S]`.
    segment_ids: Optional token segment IDs of shape `[B, T]` (`0` = padding).
      Requires `key.shape[1] == segment_ids.shape[1]`.
    local_window_size: Optional sliding window size `(left_window,
      right_window)` or integer window size `W`.
    logits_soft_cap: Optional tanh soft-cap bound applied to attention logits.
    q_sharding: Optional explicit `NamedSharding` for distributed execution.
    query_sharding_spec: Optional `PartitionSpec` resolved against the active
      multi-device mesh when `q_sharding` is `None`.
    implementation: Ordered tuple of Tokamax backends to try (`"mosaic"` maps to
      `PallasMosaicGpuFlashAttention` on SM90+ Hopper/Blackwell NVIDIA GPUs;
      pre-Hopper SM80/SM89 GPUs automatically route to Triton/cuDNN/XLA).

  Returns:
    Attention output tensor of shape `[B, T, N, h]` matching `query.dtype`.
  """
  # Lazy import mirrors paged_attention.py so TPU/CPU runs avoid GPU deps.
  import tokamax  # pylint: disable=g-import-not-at-top

  if isinstance(local_window_size, int):
    if local_window_size <= 0:
      raise ValueError(
          f"local_window_size must be positive, got {local_window_size}."
      )
    local_window_size = (
        local_window_size - 1,
        0 if is_causal else local_window_size - 1,
    )

  mask = _prepare_attention_mask(
      mask,
      segment_ids,
      query.shape[1],
      key.shape[1],
      is_causal,
      local_window_size,
  )

  if q_sharding is None and query_sharding_spec is not None:
    q_sharding = resolve_q_sharding(
        query_sharding_spec, query_shape=query.shape, key_shape=key.shape
    )

  orig_dtype = query.dtype
  if key.dtype != orig_dtype or value.dtype != orig_dtype:
    key = key.astype(orig_dtype)
    value = value.astype(orig_dtype)
  implementation = resolve_attn_implementations(
      tuple(implementation), logits_soft_cap=logits_soft_cap
  )

  out = tokamax.dot_product_attention(
      query,
      key,
      value,
      mask=mask,
      is_causal=is_causal,
      scale=scale,
      local_window_size=local_window_size,
      logits_soft_cap=logits_soft_cap,
      q_sharding=q_sharding,
      implementation=implementation,
  )
  return out.astype(orig_dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _gpu_ragged_dot_custom_vjp(
    lhs: jaxtyping.Array,
    rhs: jaxtyping.Array,
    group_sizes: jaxtyping.Array,
    out_dtype: jnp.dtype,
    implementation: tuple[GpuRaggedDotBackend, ...],
) -> jaxtyping.Array:
  """Custom VJP wrapper ensuring full SM80/SM90/SM100 backward support."""
  import tokamax  # pylint: disable=g-import-not-at-top

  in_dtype = jnp.result_type(lhs, rhs)
  out = tokamax.ragged_dot(
      lhs.astype(in_dtype),
      rhs.astype(in_dtype),
      group_sizes=group_sizes,
      preferred_element_type=out_dtype,
      implementation=implementation,
  )
  return out.astype(out_dtype)


def _gpu_ragged_dot_fwd(
    lhs: jaxtyping.Array,
    rhs: jaxtyping.Array,
    group_sizes: jaxtyping.Array,
    out_dtype: jnp.dtype,
    implementation: tuple[GpuRaggedDotBackend, ...],
) -> tuple[
    jaxtyping.Array,
    tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array],
]:
  """Forward pass for custom VJP grouped ragged dot."""
  out = _gpu_ragged_dot_custom_vjp(
      lhs, rhs, group_sizes, out_dtype, implementation
  )
  return out, (lhs, rhs, group_sizes)


def _gpu_ragged_dot_bwd(
    out_dtype: jnp.dtype,
    implementation: tuple[GpuRaggedDotBackend, ...],
    residuals: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array],
    dout: jaxtyping.Array,
) -> tuple[jaxtyping.Array, jaxtyping.Array, None]:
  """Backward pass for custom VJP grouped ragged dot."""
  del out_dtype
  import tokamax  # pylint: disable=g-import-not-at-top

  lhs, rhs, group_sizes = residuals
  in_dtype = jnp.result_type(lhs, rhs)
  lhs_c = lhs.astype(in_dtype)
  rhs_c = rhs.astype(in_dtype)
  dout_c = dout.astype(in_dtype)
  dlhs = tokamax.ragged_dot(
      dout_c,
      rhs_c.swapaxes(-1, -2),
      group_sizes=group_sizes,
      preferred_element_type=lhs.dtype,
      implementation=implementation,
  ).astype(lhs.dtype)
  drhs_impl: tuple[GpuRaggedDotBackend, ...] = tuple(
      i for i in implementation if i != "mosaic"
  ) or ("triton", "xla")
  drhs = tokamax.ragged_dot_general(
      lhs_c,
      dout_c,
      group_sizes,
      ragged_dot_dimension_numbers=_RAGGED_CONTRACTING_DOT_DIM_NUMS,
      preferred_element_type=rhs.dtype,
      implementation=drhs_impl,
  ).astype(rhs.dtype)
  return dlhs, drhs, None


_gpu_ragged_dot_custom_vjp.defvjp(_gpu_ragged_dot_fwd, _gpu_ragged_dot_bwd)


def gpu_ragged_dot(
    lhs: jaxtyping.Array,
    rhs: jaxtyping.Array,
    *,
    group_sizes: jaxtyping.Array,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    implementation: tuple[GpuRaggedDotBackend, ...] = (
        DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS
    ),
) -> jaxtyping.Array:
  """Runs grouped MoE matrix multiplication on GPU via Tokamax Mosaic GPU.

  Args:
    lhs: Sorted token activations of shape `[M, K]`.
    rhs: Expert weight tensor of shape `[G, K, N]`.
    group_sizes: Number of tokens routed to each expert of shape `[G]`.
    preferred_element_type: Output element dtype (defaults to
      `jnp.result_type(lhs, rhs)`).
    implementation: Ordered tuple of Tokamax backends (`"mosaic"` maps to
      `PallasMosaicGpuRaggedDot` on NVIDIA GPUs).

  Returns:
    Grouped matrix multiplication output of shape `[M, N]`.
  """
  out_dtype = jnp.dtype(
      jnp.result_type(lhs, rhs)
      if preferred_element_type is None
      else preferred_element_type
  )
  m = lhs.shape[0]
  if m % 8 != 0:
    lhs = jnp.pad(lhs, ((0, (-m) % 8), (0, 0)))
  return _gpu_ragged_dot_custom_vjp(
      lhs, rhs, group_sizes, out_dtype, tuple(implementation)
  )[:m]
