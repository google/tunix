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


def _pallas_normalizer(logits, *, interpret: bool):
  """Stages 1+2: per-tile summaries and the fixed-order row normalizer.

  Shared verbatim by the materializing log_softmax and the gathered
  variant so both consume the bit-identical normalizer.
  """
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
  return (
      m,
      vocab,
      vocab_groups,
      padded_vocab,
      block_rows,
      grouped_logits,
      log_normalizer,
  )


def _pallas_log_softmax(logits, *, interpret: bool):
  (
      m,
      vocab,
      vocab_groups,
      padded_vocab,
      block_rows,
      grouped_logits,
      log_normalizer,
  ) = _pallas_normalizer(logits, interpret=interpret)

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


def gathered_logprobs(logits, token_ids, *, interpret: bool = False):
  """Sampled-token logprob, top-1, and rank without the row materialize.

  Replaces stage 3 plus the stock gather for the max_logprobs=1 rollout
  contract.  Bitwise argument: stage 1+2 are shared verbatim, and every
  comparison and output below operates on x - normalizer computed with
  the same broadcast subtract stage 3 uses, so the sampled logprob, the
  top-1 value/index (lowest index on ties, like jax.lax.top_k), and the
  `>=`-rank match the materialize-then-gather chain bit for bit.  The
  padded-vocab sentinel stays maximally negative after the subtract, so
  it can never win the strict-> top-1 update and always fails the >=
  rank test, exactly as the sliced materialized row excludes it.
  """
  (
      m,
      vocab,
      vocab_groups,
      padded_vocab,
      block_rows,
      grouped_logits,
      log_normalizer,
  ) = _pallas_normalizer(logits, interpret=interpret)
  group_width = TILES_PER_GROUP * VOCAB_TILE

  normalizer = log_normalizer[:, 0, 0, 0]
  token_column = jnp.take_along_axis(
      logits, token_ids.astype(jnp.int32)[:, None], axis=-1
  )
  token_logprob = token_column[:, 0] - normalizer
  token_tile = jnp.broadcast_to(token_logprob[:, None], (m, SUMMARY_ALIGN))

  def gather_kernel(x_ref, norm_ref, token_ref, rank_ref, val_ref, idx_ref):
    group = pl.program_id(1)
    x = x_ref[...].astype(jnp.float32)
    row_normalizer = norm_ref[:, 0, 0, 0].astype(jnp.float32)
    normalized = x - row_normalizer[:, None, None, None]
    flat = normalized.reshape(block_rows, group_width)
    token_value = token_ref[:, 0].astype(jnp.float32)
    rank_part = jnp.sum(
        flat >= token_value[:, None], axis=-1
    ).astype(jnp.int32)
    tile_max = jnp.max(flat, axis=-1)
    lane = jax.lax.broadcasted_iota(jnp.int32, flat.shape, 1)
    at_max = flat == tile_max[:, None]
    local_index = jnp.min(
        jnp.where(at_max, lane, group_width), axis=-1
    ).astype(jnp.int32)
    global_index = group * group_width + local_index

    @pl.when(group == 0)
    def _():
      rank_ref[...] = jnp.broadcast_to(rank_part[:, None], rank_ref.shape)
      val_ref[...] = jnp.broadcast_to(tile_max[:, None], val_ref.shape)
      idx_ref[...] = jnp.broadcast_to(global_index[:, None], idx_ref.shape)

    @pl.when(group != 0)
    def _():
      rank_ref[...] = rank_ref[...] + jnp.broadcast_to(
          rank_part[:, None], rank_ref.shape
      )
      previous_value = val_ref[:, 0]
      previous_index = idx_ref[:, 0]
      better = tile_max > previous_value
      val_ref[...] = jnp.broadcast_to(
          jnp.where(better, tile_max, previous_value)[:, None], val_ref.shape
      )
      idx_ref[...] = jnp.broadcast_to(
          jnp.where(better, global_index, previous_index)[:, None],
          idx_ref.shape,
      )

  rank, top_value, top_index = pl.pallas_call(
      gather_kernel,
      out_shape=(
          jax.ShapeDtypeStruct((m, SUMMARY_ALIGN), jnp.int32),
          jax.ShapeDtypeStruct((m, SUMMARY_ALIGN), jnp.float32),
          jax.ShapeDtypeStruct((m, SUMMARY_ALIGN), jnp.int32),
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
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN),
                  lambda row, _group: (row, 0),
              ),
          ],
          out_specs=[
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN), lambda row, _group: (row, 0)
              ),
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN), lambda row, _group: (row, 0)
              ),
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN), lambda row, _group: (row, 0)
              ),
          ],
          grid=(m // block_rows, vocab_groups),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          allow_input_fusion=(False, False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f"canon_logsoftmax_gather_m{m}_v{vocab}_vp{padded_vocab}",
  )(grouped_logits, log_normalizer, token_tile)

  return (
      token_logprob,
      top_value[:, 0],
      top_index[:, 0],
      rank[:, 0],
  )


def continue_decode_gathered_logprobs(
    logits, token_ids, *, interpret: bool = False
):
  """Run the production-M gather for continue-decode request buckets."""
  value = os.environ.get("CANON_CONTINUE_DECODE", "")
  if not value or not value.isdigit() or not 1 <= int(value) <= 64:
    raise CanonicalLogSoftmaxError(
        "continue-decode row compatibility requires CANON_CONTINUE_DECODE "
        f"in [1, 64], got {value!r}"
    )
  if logits.ndim != 2 or token_ids.ndim != 1:
    raise CanonicalLogSoftmaxError(
        "continue-decode gather expects logits[M,V] and token_ids[M], got "
        f"{logits.shape}/{token_ids.shape}"
    )
  rows = int(logits.shape[0])
  if int(token_ids.shape[0]) != rows:
    raise CanonicalLogSoftmaxError(
        "continue-decode logits/token rows differ: "
        f"{rows} vs {token_ids.shape[0]}"
    )
  if rows == PRODUCTION_M:
    return gathered_logprobs(logits, token_ids, interpret=interpret)
  if rows not in (8, 16, 32):
    raise CanonicalLogSoftmaxError(
        "continue-decode gather admits only request buckets 8/16/32 or "
        f"production M={PRODUCTION_M}, got {rows}"
    )

  # The normalizer, token gather, top-1, and rank are all row-independent.
  # Append inert rows to restore the certified M=256 program, then discard
  # them.  Every real row therefore executes the identical vocabulary tiles,
  # reduction order, subtract, comparisons, and tie-breaking as before.
  padded_logits = jnp.pad(
      logits,
      ((0, PRODUCTION_M - rows), (0, 0)),
      constant_values=jnp.float32(0),
  )
  padded_tokens = jnp.pad(
      token_ids,
      ((0, PRODUCTION_M - rows),),
      constant_values=jnp.int32(0),
  )
  output = gathered_logprobs(
      padded_logits, padded_tokens, interpret=interpret
  )
  print(
      f"[PATHTRACE] CANON_CONTINUE_DECODE gathered-logprobs M-padding "
      f"M={rows} Mp={PRODUCTION_M}",
      flush=True,
  )
  return tuple(item[:rows] for item in output)
