# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fixed-program tiled log-softmax for canonical rollout/trainer scoring."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


ENV = "CANON_PALLAS_LOGSOFTMAX"
PRODUCTION_M = 256
PRODUCTION_V = 151936
VOCAB_ALIGN = 128
VOCAB_TILE = 1024
TILES_PER_GROUP = 8
SUMMARY_ALIGN = 128


class CanonicalLogSoftmaxError(ValueError):
  pass


def _validate(logits, *, interpret: bool) -> tuple[int, int, int]:
  if os.environ.get(ENV, "") != "1":
    raise CanonicalLogSoftmaxError(f"{ENV}=1 is required")
  if logits.ndim != 2 or logits.dtype != jnp.float32:
    raise CanonicalLogSoftmaxError(
        f"canonical log-softmax requires rank-2 f32, got {logits.shape}/{logits.dtype}"
    )
  m, vocab = map(int, logits.shape)
  if not interpret and (m, vocab) != (PRODUCTION_M, PRODUCTION_V):
    raise CanonicalLogSoftmaxError(
        "TPU canonical log-softmax requires exact production shape "
        f"{(PRODUCTION_M, PRODUCTION_V)}, got {(m, vocab)}"
    )
  padded_vocab = ((vocab + VOCAB_ALIGN - 1) // VOCAB_ALIGN) * VOCAB_ALIGN
  return m, vocab, padded_vocab


def _pallas_log_softmax(logits, *, interpret: bool):
  m, vocab, _ = _validate(logits, interpret=interpret)
  block_rows = 1 if interpret else 8
  if m % block_rows:
    raise CanonicalLogSoftmaxError(
        f"row count {m} must divide TPU block_rows={block_rows}"
    )
  group_width = TILES_PER_GROUP * VOCAB_TILE
  vocab_groups = (vocab + group_width - 1) // group_width
  padded_vocab = vocab_groups * group_width
  if padded_vocab != vocab:
    logits = jnp.pad(
        logits,
        ((0, 0), (0, padded_vocab - vocab)),
        # A finite sentinel avoids (-inf) - (-inf) in fully padded tiles.  Its
        # contribution vanishes exactly when the group summaries are combined.
        constant_values=jnp.finfo(jnp.float32).min,
    )
  grouped_logits = logits.reshape(
      m, vocab_groups, TILES_PER_GROUP, VOCAB_TILE
  )

  # Stage 1 reduces each vocabulary tile independently.  Keeping the vocabulary
  # block bounded is required on TPU: the earlier full-row custom call needed
  # 18.55 MiB of scoped VMEM, above the 16 MiB hardware limit.
  def partial_kernel(x_ref, max_ref, sum_ref):
    x = x_ref[...].astype(jnp.float32)
    tile_max = jnp.max(x, axis=-1)
    tile_sum = jnp.sum(
        jnp.exp(x - tile_max[..., None]), axis=-1, dtype=jnp.float32
    )
    max_ref[...] = jnp.broadcast_to(tile_max[..., None], max_ref.shape)
    sum_ref[...] = jnp.broadcast_to(tile_sum[..., None], sum_ref.shape)

  partial_max, partial_sum = pl.pallas_call(
      partial_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              (m, vocab_groups, TILES_PER_GROUP, SUMMARY_ALIGN), jnp.float32
          ),
          jax.ShapeDtypeStruct(
              (m, vocab_groups, TILES_PER_GROUP, SUMMARY_ALIGN), jnp.float32
          ),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (block_rows, 1, TILES_PER_GROUP, VOCAB_TILE),
                  lambda row, group: (row, group, 0, 0),
              ),
          ],
          out_specs=[
              pl.BlockSpec(
                  (block_rows, 1, TILES_PER_GROUP, SUMMARY_ALIGN),
                  lambda row, group: (row, group, 0, 0),
              ),
              pl.BlockSpec(
                  (block_rows, 1, TILES_PER_GROUP, SUMMARY_ALIGN),
                  lambda row, group: (row, group, 0, 0),
              ),
          ],
          grid=(m // block_rows, vocab_groups),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          allow_input_fusion=(False,),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f"canon_logsoftmax_partial_m{m}_v{vocab}_vp{padded_vocab}",
  )(grouped_logits)

  # Stage 2 combines the tile summaries in a fixed left-to-right vocabulary
  # layout.  This is the only cross-tile reduction and its input is tiny.
  def combine_kernel(max_ref, sum_ref, norm_ref):
    local_max = max_ref[..., 0].astype(jnp.float32)
    local_sum = sum_ref[..., 0].astype(jnp.float32)
    local_max = local_max.reshape(block_rows, -1)
    local_sum = local_sum.reshape(block_rows, -1)
    global_max = jnp.max(local_max, axis=-1)
    global_sum = jnp.sum(
        local_sum * jnp.exp(local_max - global_max[:, None]),
        axis=-1,
        dtype=jnp.float32,
    )
    normalizer = global_max + jnp.log(global_sum)
    norm_ref[...] = jnp.broadcast_to(
        normalizer[:, None, None, None], norm_ref.shape
    )

  log_normalizer = pl.pallas_call(
      combine_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, 1, TILES_PER_GROUP, SUMMARY_ALIGN), jnp.float32
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (block_rows, vocab_groups, TILES_PER_GROUP, SUMMARY_ALIGN),
                  lambda row: (row, 0, 0, 0),
              ),
              pl.BlockSpec(
                  (block_rows, vocab_groups, TILES_PER_GROUP, SUMMARY_ALIGN),
                  lambda row: (row, 0, 0, 0),
              ),
          ],
          out_specs=pl.BlockSpec(
              (block_rows, 1, TILES_PER_GROUP, SUMMARY_ALIGN),
              lambda row: (row, 0, 0, 0),
          ),
          grid=(m // block_rows,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          allow_input_fusion=(False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f"canon_logsoftmax_combine_m{m}_v{vocab}_vp{padded_vocab}",
  )(partial_max, partial_sum)

  # Stage 3 materializes the normalized rows one vocabulary tile at a time.
  def normalize_kernel(x_ref, norm_ref, out_ref):
    normalizer = norm_ref[:, 0, 0, 0].astype(jnp.float32)
    out_ref[...] = (
        x_ref[...].astype(jnp.float32)
        - normalizer[:, None, None, None]
    )

  output = pl.pallas_call(
      normalize_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, vocab_groups, TILES_PER_GROUP, VOCAB_TILE), jnp.float32
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (block_rows, 1, TILES_PER_GROUP, VOCAB_TILE),
                  lambda row, group: (row, group, 0, 0),
              ),
              pl.BlockSpec(
                  (block_rows, 1, TILES_PER_GROUP, SUMMARY_ALIGN),
                  lambda row, _group: (row, 0, 0, 0),
              ),
          ],
          out_specs=pl.BlockSpec(
              (block_rows, 1, TILES_PER_GROUP, VOCAB_TILE),
              lambda row, group: (row, group, 0, 0),
          ),
          grid=(m // block_rows, vocab_groups),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          allow_input_fusion=(False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f"canon_logsoftmax_normalize_m{m}_v{vocab}_vp{padded_vocab}",
  )(grouped_logits, log_normalizer)
  return output.reshape(m, padded_vocab)[:, :vocab]


def log_softmax(logits, *, interpret: bool = False):
  """Returns the fixed-program primal with the analytic log-softmax VJP."""
  _validate(logits, interpret=interpret)

  @jax.custom_vjp
  def op(value):
    return _pallas_log_softmax(value, interpret=interpret)

  def forward(value):
    output = _pallas_log_softmax(value, interpret=interpret)
    return output, output

  def backward(output, cotangent):
    probability = jnp.exp(output)
    cotangent_sum = jnp.sum(cotangent, axis=-1, keepdims=True)
    return (cotangent - probability * cotangent_sum,)

  op.defvjp(forward, backward)
  return op(logits)
