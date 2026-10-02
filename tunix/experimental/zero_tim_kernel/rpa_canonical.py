# Copyright 2026 Google LLC
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
"""Canonical ragged-paged-attention (RPA v3) call-site controls.

Pins the call-site controls for ragged-paged-attention (RPA v3):

* "R2" pinned block sizes (``CANON_RPA_D`` / ``CANON_RPA_P`` /
  ``CANON_RPA_M``).  Unpinned, RPA v3 calls ``get_tuned_block_sizes()``,
  whose result is shape-dependent twice over: the lookup key includes the
  shape, and it clamps with ``min(max_num_tokens, bq)`` -- ``max_num_tokens``
  IS the decode concurrency.  So the online-softmax KV block (the attention
  accumulation granularity) changed with concurrency and with decode vs
  prefill (measured (16, 16) vs (16, 32) at Qwen3-32B@tp4).  The canonical
  configuration pins all three to ``(128, 512, 128, 512)`` (tuple order
  ``(bq_sz, bkv_sz, bq_csz, bkv_csz)``), together with
  ``MIN_TOKEN_BUCKET=256``.
* forced-mixed distribution (P19.X.4, ``CANON_FORCE_MIXED``; diagnostic).
  ``request_distribution = (i, j, k)`` is traced,
  so decode and prefill share one program, but the kernel branches on it at
  runtime (decode-only / prefill-only / mixed grids).  Rewriting it to
  ``[0, 0, k]`` sends every sequence down the mixed branch.
* "R4" differentiable RPA.  The Pallas kernel has no autodiff rule, but the
  training forward must be the very kernel the engine ran.
  ``CANON_RPA_VJP2`` wraps it with ``rpa_diff_chunked`` (forward = kernel
  verbatim, backward = chunked-cache replica with cache cotangents).  The
  older ``CANON_RPA_VJP`` (``rpa_diff``: prefill-only contract, zero KV-cache
  gradient) is kept for prefill-only workloads.
* P59 local-attention validation: inside P59's outer manual ``("data",
"model")``
  map the attention operands are already TP-local, so KV-head replication is
  skipped and only the local shapes are validated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os

from absl import logging
import jax
import jax.numpy as jnp
from tunix.experimental.zero_tim_kernel import rpa_diff
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked

# (RPA v3 kwarg, env var) pairs; tuple order (bq_sz, bkv_sz, bq_csz, bkv_csz).
BLOCK_SIZE_ENVS = (
    ("d_block_sizes", "CANON_RPA_D"),
    ("p_block_sizes", "CANON_RPA_P"),
    ("m_block_sizes", "CANON_RPA_M"),
)
# Canonical block sizes and token bucket.
CANONICAL_BLOCK_SIZES = (128, 512, 128, 512)
CANONICAL_MIN_TOKEN_BUCKET = 256
FORCE_MIXED_ENV = "CANON_FORCE_MIXED"
RPA_VJP2_ENV = "CANON_RPA_VJP2"
RPA_VJP_ENV = "CANON_RPA_VJP"

_RECEIPTS: set[str] = set()


def _log_once(tag: str) -> None:
  if tag not in _RECEIPTS:
    _RECEIPTS.add(tag)
    logging.info("[PATHTRACE] %s", tag)


def parse_block_sizes(value: str | None) -> tuple[int, ...] | None:
  """Parses ``"128,512,128,512"``; unset or empty means "not pinned"."""
  if not value:
    return None
  return tuple(int(token) for token in value.split(","))


def pinned_block_size_kwargs(
    *, use_hd64: bool = False, environ: Mapping[str, str] | None = None
) -> dict[str, tuple[int, ...]]:
  """Returns the RPA v3 ``{d,p,m}_block_sizes`` kwargs pinned by the env.

  Args:
    use_hd64: Whether the head-dim-64 kernel is used (never pinned).
    environ: The environment to read (defaults to ``os.environ``).

  Returns:
    The block-size kwargs to add to the RPA v3 call (possibly empty).
  """
  if use_hd64:
    return {}
  env = os.environ if environ is None else environ
  kwargs = {}
  for key, name in BLOCK_SIZE_ENVS:
    blocks = parse_block_sizes(env.get(name))
    if blocks is not None:
      kwargs[key] = blocks
      _log_once(f"RPA_BLOCK {key}={blocks}")
  return kwargs


def canonical_block_size_kwargs() -> dict[str, tuple[int, ...]]:
  """The canonical bundle's pinned kwargs, independent of the environment."""
  return {key: CANONICAL_BLOCK_SIZES for key, _ in BLOCK_SIZE_ENVS}


def force_mixed_distribution(distribution):
  """Rewrites ``(i, j, k)`` to ``[0, 0, k]``: every sequence takes "mixed"."""
  return jnp.stack([
      jnp.zeros_like(distribution[0]),
      jnp.zeros_like(distribution[1]),
      distribution[2],
  ])


def maybe_force_mixed_distribution(distribution):
  """Applies ``force_mixed_distribution`` iff CANON_FORCE_MIXED=1."""
  if os.environ.get(FORCE_MIXED_ENV) == "1":
    _log_once("FORCE_MIXED distribution -> [0, 0, k]")
    return force_mixed_distribution(distribution)
  return distribution


def differentiable_rpa(
    kernel_fn: Callable,
    *,
    sm_scale: float,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    use_causal_mask: bool = True,
    use_hd64: bool = False,
    has_attention_sink: bool = False,
) -> Callable:
  """Selects the differentiable wrapper of the local RPA v3 call by env.

  Args:
    kernel_fn: The shard-local RPA v3 call ``kernel_fn(q, k, v, kv_cache,
      kv_lens, page_indices, cu_q_lens, distribution)``.
    sm_scale: Softmax scale.
    page_size: Tokens per KV-cache page.
    num_q_heads: Number of query heads (informational for the chunked wrapper,
      which derives local head counts from operand shapes).
    num_kv_heads: Number of KV heads (informational, as above).
    use_causal_mask: Causal masking flag of the RPA call (``rpa_diff`` only).
    use_hd64: Whether the head-dim-64 kernel is used (unsupported).
    has_attention_sink: Whether an attention sink is used (unsupported).

  Returns:
    ``kernel_fn`` wrapped per CANON_RPA_VJP2 / CANON_RPA_VJP, or unchanged.

  Raises:
    NotImplementedError: If a wrapper is requested on an unsupported path.
  """
  if os.environ.get(RPA_VJP2_ENV) == "1":
    if use_hd64 or has_attention_sink:
      raise NotImplementedError(f"{RPA_VJP2_ENV}: v3 default path only")
    _log_once("RPA_VJP2 on (chunked-cache differentiable)")
    return rpa_diff_chunked.make_diff_rpa_chunked_ragged(
        kernel_fn,
        sm_scale=sm_scale,
        page_size=page_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
  if os.environ.get(RPA_VJP_ENV) == "1":
    if use_hd64 or has_attention_sink:
      raise NotImplementedError(
          f"{RPA_VJP_ENV} is only wired for the v3 default RPA path without "
          f"attention sink (use_hd64={use_hd64}, "
          f"attention_sink={has_attention_sink})"
      )
    _log_once("RPA_VJP on")
    return rpa_diff.make_diff_rpa(
        kernel_fn, sm_scale=sm_scale, use_causal_mask=use_causal_mask
    )
  return kernel_fn


def p59_local_attention_context(mesh) -> bool:
  """Returns whether P59 has already sliced attention over DP and TP.

  Args:
    mesh: The live engine mesh.

  Returns:
    True iff CANON_P59_RANK_PARALLEL_BACKWARD=1 and the abstract mesh is the
    manual ("data", "model") map of the same topology as ``mesh``.

  Raises:
    RuntimeError: If the manual context and the engine topology disagree.
  """
  if os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "") != "1":
    return False
  context = jax.sharding.get_abstract_mesh()
  if tuple(context.axis_names) != ("data", "model"):
    return False
  axis_types = dict(zip(context.axis_names, context.axis_types))
  if (
      axis_types.get("data") is not jax.sharding.AxisType.Manual
      or axis_types.get("model") is not jax.sharding.AxisType.Manual
  ):
    return False
  if "data" not in mesh.shape or "model" not in mesh.shape:
    raise RuntimeError("P59 local attention requires engine data/model axes")
  data_size = int(context.shape["data"])
  model_size = int(context.shape["model"])
  if (
      data_size < 1
      or model_size <= 1
      or data_size != int(mesh.shape["data"])
      or model_size != int(mesh.shape["model"])
  ):
    raise RuntimeError("P59 local attention context and engine topology differ")
  return True


def validate_p59_local_attention(q, k, v, kv_cache, tp_size: int) -> None:
  """Validates TP-local RPA operands inside the P59 map (no replication).

  Args:
    q: ``[T, local_q_heads, head_dim]``.
    k: ``[T, local_kv_heads, head_dim]``.
    v: Same shape as ``k``.
    kv_cache: ``[pages, page_size, cache_heads, packing, head_dim]`` with
      ``cache_heads = ceil(2 * local_kv_heads / packing)``.
    tp_size: The TP degree (for the receipt).

  Raises:
    ValueError: On any rank, GQA-ratio or cache-layout mismatch.
  """
  shapes = tuple(tuple(map(int, value.shape)) for value in (q, k, v, kv_cache))
  q_shape, k_shape, v_shape, cache_shape = shapes
  if len(q_shape) != 3 or len(cache_shape) != 5 or k_shape != v_shape:
    raise ValueError(f"P59 local attention rank/shape mismatch: {shapes}")
  if k_shape[0] != q_shape[0] or k_shape[2] != q_shape[2]:
    raise ValueError(f"P59 local attention Q/K/V mismatch: {shapes}")
  if k_shape[1] <= 0 or q_shape[1] % k_shape[1] != 0:
    raise ValueError(f"P59 local attention GQA ratio mismatch: {shapes}")
  packing = cache_shape[3]
  if packing <= 0:
    raise ValueError(f"P59 local attention cache packing mismatch: {shapes}")
  expected_cache_heads = (2 * k_shape[1] + packing - 1) // packing
  if cache_shape[2] != expected_cache_heads or cache_shape[4] != k_shape[2]:
    raise ValueError(f"P59 local attention cache shape mismatch: {shapes}")
  _log_once(
      f"P59_RPA_LOCAL_KV_READY tp={tp_size} local_q_heads={q_shape[1]} "
      f"local_kv_heads={k_shape[1]} cache_heads={cache_shape[2]} "
      f"packing={packing}"
  )
