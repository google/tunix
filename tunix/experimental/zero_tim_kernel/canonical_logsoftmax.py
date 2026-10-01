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

"""Fixed-program tiled log-softmax for canonical rollout/trainer scoring ("R5").

``jax.nn.log_softmax`` over a 151936-wide vocabulary lets XLA choose the
reduction tree per shape and per program, so the sampler's logprob of a token
and the learner's recomputed logprob of the same token can differ in the last
bits.  Here the row normalizer is computed by a fixed three-stage Pallas
program: per-1024-column tile max/sum-exp (stage 1), a left-to-right combine
of the tile summaries (stage 2, the only cross-tile reduction) and a
broadcast subtract (stage 3).  Every stage is row-independent, so a short
decode slice run at its own 8-aligned row bucket gives bit-identical rows to
the 256-row production program.
"""

from __future__ import annotations

import os

from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

ENV = "CANON_PALLAS_LOGSOFTMAX"
PRODUCTION_M = 256
PRODUCTION_V = 151936
VOCAB_ALIGN = 128
# Every stage of the canonical log-softmax
# is row-independent (block_rows=8 grids; per-row max/sum/normalize/gather),
# so a shorter row bucket runs the identical per-row arithmetic as the
# 256-row production program.  Measured bitwise on TPU for m=8/32/64/128
# against m=256.  Short decode slices use their own bucket instead of padding
# to M=256; CANON_LOGPROB_M_BUCKET=0 restores the padded program for A/B runs.
ROW_BUCKET_ALIGN = 8
ROW_BUCKET_ENV = "CANON_LOGPROB_M_BUCKET"


def row_bucket_admitted(m: int) -> bool:
  """Whether ``m`` rows is an admitted TPU row bucket."""
  return m % ROW_BUCKET_ALIGN == 0 and ROW_BUCKET_ALIGN <= m <= PRODUCTION_M


def row_bucket_enabled() -> bool:
  """Whether short decode slices run at their own row bucket (default)."""
  return os.environ.get(ROW_BUCKET_ENV, "1") != "0"


def row_bucket(rows: int) -> int:
  """The smallest admitted bucket holding ``rows`` (never above PRODUCTION_M)."""
  bucket = max(
      ROW_BUCKET_ALIGN, -(-rows // ROW_BUCKET_ALIGN) * ROW_BUCKET_ALIGN
  )
  return min(bucket, PRODUCTION_M)


VOCAB_TILE = 1024
TILES_PER_GROUP = 8
SUMMARY_ALIGN = 128


class CanonicalLogSoftmaxError(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when the canonical log-softmax contract is not met.

  The name is kept for parity with the upstream Tunix branch's public API.
  """


def _validate(
    logits, *, interpret: bool, strict_vocab: bool = True
) -> tuple[int, int, int]:
  """Checks the env gate, rank/dtype and (on TPU) the admitted shapes.

  Args:
    logits: Rank-2 float32 logits `[M, V]`.
    interpret: Whether Pallas runs in interpret mode (relaxes the TPU shape
      admission checks).
    strict_vocab: On TPU, require `V == PRODUCTION_V`.

  Returns:
    `(m, vocab, padded_vocab)` with `padded_vocab` rounded up to `VOCAB_ALIGN`.

  Raises:
    CanonicalLogSoftmaxError: If the env gate is off or the contract is not met.
  """
  if os.environ.get(ENV, "") != "1":
    raise CanonicalLogSoftmaxError(f"{ENV}=1 is required")
  if logits.ndim != 2 or logits.dtype != jnp.float32:
    raise CanonicalLogSoftmaxError(
        "canonical log-softmax requires rank-2 f32, got"
        f" {logits.shape}/{logits.dtype}"
    )
  m, vocab = map(int, logits.shape)
  if not interpret and (
      (strict_vocab and vocab != PRODUCTION_V) or not row_bucket_admitted(m)
  ):
    raise CanonicalLogSoftmaxError(
        "TPU canonical log-softmax requires the production vocabulary "
        f"{PRODUCTION_V} and a row bucket (multiple of {ROW_BUCKET_ALIGN} "
        f"up to {PRODUCTION_M}), got {(m, vocab)}"
    )
  padded_vocab = ((vocab + VOCAB_ALIGN - 1) // VOCAB_ALIGN) * VOCAB_ALIGN
  return m, vocab, padded_vocab


def _pallas_normalizer(logits, *, interpret: bool, strict_vocab: bool = True):
  """Stages 1+2: per-tile summaries and the fixed-order row normalizer.

  Shared verbatim by the materializing log_softmax and the gathered
  variant so both consume the bit-identical normalizer.

  Args:
    logits: Rank-2 float32 logits `[M, V]`.
    interpret: Whether Pallas runs in interpret mode.
    strict_vocab: On TPU, require `V == PRODUCTION_V`.

  Returns:
    `(m, vocab, vocab_groups, padded_vocab, block_rows, grouped_logits,
    log_normalizer)`: the static geometry, the group-padded logits reshaped to
    `[M, vocab_groups, TILES_PER_GROUP, VOCAB_TILE]` and the per-row
    log-normalizer `[M, 1, TILES_PER_GROUP, SUMMARY_ALIGN]` (the scalar
    broadcast along the trailing axes).

  Raises:
    CanonicalLogSoftmaxError: If `M` is not a multiple of the TPU block rows.
  """
  m, vocab, _ = _validate(
      logits, interpret=interpret, strict_vocab=strict_vocab
  )
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
  grouped_logits = logits.reshape(m, vocab_groups, TILES_PER_GROUP, VOCAB_TILE)

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


def _pallas_log_softmax(logits, *, interpret: bool, strict_vocab: bool = True):
  """Stages 1+2+3: materializes the normalized `[M, V]` log-probabilities."""
  (
      m,
      vocab,
      vocab_groups,
      padded_vocab,
      block_rows,
      grouped_logits,
      log_normalizer,
  ) = _pallas_normalizer(logits, interpret=interpret, strict_vocab=strict_vocab)

  # Stage 3 materializes the normalized rows one vocabulary tile at a time.
  def normalize_kernel(x_ref, norm_ref, out_ref):
    normalizer = norm_ref[:, 0, 0, 0].astype(jnp.float32)
    out_ref[...] = (
        x_ref[...].astype(jnp.float32) - normalizer[:, None, None, None]
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


def log_softmax(logits, *, interpret: bool = False, strict_vocab: bool = True):
  """Returns the fixed-program primal with the analytic log-softmax VJP."""
  _validate(logits, interpret=interpret, strict_vocab=strict_vocab)

  @jax.custom_vjp
  def op(value):
    return _pallas_log_softmax(
        value, interpret=interpret, strict_vocab=strict_vocab
    )

  def forward(value):
    output = _pallas_log_softmax(
        value, interpret=interpret, strict_vocab=strict_vocab
    )
    return output, output

  def backward(output, cotangent):
    probability = jnp.exp(output)
    cotangent_sum = jnp.sum(cotangent, axis=-1, keepdims=True)
    return (cotangent - probability * cotangent_sum,)

  op.defvjp(forward, backward)
  return op(logits)


def token_logprobs(
    logits,
    token_ids,
    *,
    interpret: bool = False,
    strict_vocab: bool = True,
):
  """Returns per-row target token log-probabilities with fused analytic VJP.

  HBM & Latency Optimization over ``take_along_axis(log_softmax(logits), ...)``:
  1. Forward HBM & Latency:
     Runs only Stages 1+2 (``_pallas_normalizer``) to compute the per-row
     log-normalizer ``normalizer[M]`` in ``float32`` and subtracts it directly
     from the gathered target logit column ``logits[row, token_ids[row]]``.
     This completely skips Stage 3 (``normalize_kernel``) and avoids
     materializing the full ``[M, padded_vocab]`` ``float32`` normalized
     log-probabilities tensor in HBM (saving ~152 MB per ``M=256`` chunk or
     ~24 MB at ``M=64``). Because ``normalize_kernel`` performs the identical
     ``float32`` subtraction ``x - normalizer``, the gathered scalar logprob is
     100% bitwise identical to ``log_softmax(logits)[row, token_ids[row]]``.
  2. Backward (``@jax.custom_vjp``) HBM & Latency:
     Instead of saving a second ``[M, V]`` ``float32`` ``output`` tensor as the
     residual of ``log_softmax`` AND materializing a third ``[M, V]``
     ``float32`` sparse one-hot cotangent tensor from ``jnp.take_along_axis``'s
     VJP, the custom VJP saves only ``(logits, normalizer)`` (where
     ``normalizer`` is a 1D ``float32[M]`` vector of at most 1 KiB) and computes
     ``dL/dlogits`` directly from the 1D upstream cotangent ``cotangent[M]``:

       ``dL/dlogits[r, c] = cotangent[r] *``
       ``(1[c == token_ids[r]] - exp(logits[r, c] - normalizer[r]))``

     eliminating two full ``[M, V]`` ``float32`` intermediate tensors during
     ``Fwd+Bwd``.

  Args:
    logits: Rank-2 float32 logits ``[M, V]``.
    token_ids: Integer target ids ``[M]`` (one per row).
    interpret: Whether Pallas runs in interpret mode.
    strict_vocab: On TPU, require ``V == PRODUCTION_V``.

  Returns:
    ``float32[M]`` log-probabilities of the target tokens, bitwise equal to
    ``log_softmax(logits)[row, token_ids[row]]``.

  Raises:
    CanonicalLogSoftmaxError: If the contract is not met or ``token_ids`` does
      not have shape ``[M]``.
  """
  m, _, _ = _validate(logits, interpret=interpret, strict_vocab=strict_vocab)
  if token_ids.ndim != 1 or int(token_ids.shape[0]) != m:
    raise CanonicalLogSoftmaxError(
        "canonical token_logprobs expects token_ids[M] matching logits[M,V], "
        f"got {logits.shape}/{token_ids.shape}"
    )
  tok_ids_i32 = token_ids.astype(jnp.int32)

  def _primal_and_normalizer(value, tok_ids):
    *_, log_normalizer = _pallas_normalizer(
        value, interpret=interpret, strict_vocab=strict_vocab
    )
    normalizer = log_normalizer[:, 0, 0, 0]
    token_column = jnp.take_along_axis(value, tok_ids[:, None], axis=-1)[:, 0]
    return token_column - normalizer, normalizer

  @jax.custom_vjp
  def op(value, tok_ids):
    token_logprob, _ = _primal_and_normalizer(value, tok_ids)
    return token_logprob

  def forward(value, tok_ids):
    token_logprob, normalizer = _primal_and_normalizer(value, tok_ids)
    return token_logprob, (value, normalizer, tok_ids)

  def backward(residual, cotangent):
    value, normalizer, tok_ids = residual
    probability = jnp.exp(value - normalizer[:, None])
    grad_logits = -probability * cotangent[:, None]
    rows = jnp.arange(m, dtype=jnp.int32)
    grad_logits = grad_logits.at[rows, tok_ids].add(cotangent)
    return grad_logits, None

  op.defvjp(forward, backward)
  return op(logits, tok_ids_i32)


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

  Args:
    logits: Rank-2 float32 logits `[M, V]`.
    token_ids: Integer sampled token ids `[M]`.
    interpret: Whether Pallas runs in interpret mode.

  Returns:
    `(token_logprob, top_value, top_index, rank)`, each of shape `[M]`: the
    sampled token's log-probability, the top-1 log-probability and its vocab
    index, and the 1-based `>=`-rank of the sampled token.
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
    rank_part = jnp.sum(flat >= token_value[:, None], axis=-1).astype(jnp.int32)
    tile_max = jnp.max(flat, axis=-1)
    lane = jax.lax.broadcasted_iota(jnp.int32, flat.shape, 1)
    at_max = flat == tile_max[:, None]
    local_index = jnp.min(jnp.where(at_max, lane, group_width), axis=-1).astype(
        jnp.int32
    )
    global_index = group * group_width + local_index

    @pl.when(group == 0)
    def _init_first_group():
      rank_ref[...] = jnp.broadcast_to(rank_part[:, None], rank_ref.shape)
      val_ref[...] = jnp.broadcast_to(tile_max[:, None], val_ref.shape)
      idx_ref[...] = jnp.broadcast_to(global_index[:, None], idx_ref.shape)

    @pl.when(group != 0)
    def _merge_later_group():
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
  if row_bucket_enabled():
    # B2 knife 1: the bucket runs the same per-row arithmetic as the padded
    # 256-row program (row-independent stages), without the inert rows.
    output = gathered_logprobs(logits, token_ids, interpret=interpret)
    logging.info(
        "[PATHTRACE] %s=1 gathered-logprobs M-bucket M=%d (no padding)",
        ROW_BUCKET_ENV,
        rows,
    )
    return output

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
  output = gathered_logprobs(padded_logits, padded_tokens, interpret=interpret)
  logging.info(
      "[PATHTRACE] CANON_CONTINUE_DECODE gathered-logprobs M-padding "
      "M=%d Mp=%d",
      rows,
      PRODUCTION_M,
  )
  return tuple(item[:rows] for item in output)
