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

"""Fixed-program tiled log-softmax and vocab-parallel scoring ("R5").

``jax.nn.log_softmax`` over a 151936-wide vocabulary lets XLA choose the
reduction tree per shape and per program, so the sampler's logprob of a token
and the learner's recomputed logprob of the same token can differ in the last
bits. Here the row normalizer is computed by a fixed two-stage program:
per-1024-column tile max/sum-exp (stage 1) and a strict left-to-right combine
of the compact tile summaries (stage 2, the only cross-tile reduction),
followed by flag-proof ``float32`` subtraction. On one device both stages run
in a single Pallas launch (the tile summaries stay in VMEM scratch); under
tensor parallelism each rank runs stage 1 on its shard and a small combine
kernel runs stage 2 on the all-gathered summaries. Both forms evaluate the
same per-tile arithmetic and the same stage-2 operand shape, so they agree bit
for bit.

Both ``bfloat16`` and ``float32`` ``[M, V]`` inputs of any positive shape are
accepted directly without environment-variable gates or ``M <= 256`` row caps.
Under tensor parallelism (``vocab_parallel_log_softmax`` /
``vocab_parallel_token_logprobs``), Stage 1 runs locally on each TP shard
whenever ``V / TP >= VOCAB_TILE``: when ``(V / TP) % VOCAB_TILE != 0`` (e.g.
Qwen3 ``V = 151936`` at ``TP = 2 / 4 / 8``), a single 1024-column neighbor halo
exchange (``lax.ppermute``) aligns every rank's slice to the exact global
1024-column tile boundaries so the TP-sharded normalizer remains 100% bitwise
identical to the unsharded program without all-gathering ``[M, V]`` logits.
"""

from __future__ import annotations

import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tunix.zero_tim_kernel import fixed_order_reduce
from tunix.zero_tim_kernel import model_contracts

MeshLike = jax.sharding.Mesh | jax.sharding.AbstractMesh

PRODUCTION_V = 151936
ROW_BUCKET_ALIGN = 8

VOCAB_TILE = 1024
TILES_PER_GROUP = 8
SUMMARY_ALIGN = 128


class CanonicalLogSoftmaxError(ValueError):  # pylint: disable=g-bad-exception-name
  """Raised when the canonical log-softmax operand contract is violated."""


def _as_f32(x: jax.Array) -> jax.Array:
  """Upcasts ``bf16`` via ``hard_upcast`` so XLA excess precision cannot fuse."""
  if x.dtype == jnp.bfloat16:
    return fixed_order_reduce.hard_upcast(x)
  return x


def _validate(logits: jax.Array | jax.ShapeDtypeStruct) -> tuple[int, int]:
  """Checks rank, floating dtype (bf16 or f32), and positive dimensions."""
  if logits.ndim != 2 or logits.dtype not in (jnp.bfloat16, jnp.float32):
    raise CanonicalLogSoftmaxError(
        "canonical log-softmax requires rank-2 bf16 or f32, got"
        f" {logits.shape}/{logits.dtype}"
    )
  m, vocab = map(int, logits.shape)
  if m <= 0 or vocab <= 0:
    raise CanonicalLogSoftmaxError(
        f"canonical log-softmax requires positive (M, V), got {(m, vocab)}"
    )
  return m, vocab


def _combine_rows(l_max: jax.Array, l_sum: jax.Array) -> jax.Array:
  """Stage 2 arithmetic: ``[rows, tiles]`` summaries -> ``[rows]`` normalizer.

  The single cross-tile reduction of the program. Both the standalone combine
  kernel and the fused single-launch kernel call this on identically shaped
  ``[block_rows, summary_width]`` operands, so they produce the same bits.

  Args:
    l_max: Per-tile maxima, ``float32[rows, tiles]``.
    l_sum: Per-tile ``sum(exp(x - max))``, ``float32[rows, tiles]``.

  Returns:
    ``float32[rows]`` log-normalizers ``max + log(sum)``.
  """
  global_max = jnp.max(l_max, axis=-1)
  global_sum = jnp.sum(
      l_sum * jnp.exp(l_max - global_max[:, None]),
      axis=-1,
      dtype=jnp.float32,
  )
  return global_max + jnp.log(global_sum)


def _tiles_per_step(tiles: int) -> int:
  """Tiles one grid step loads: ``<= TILES_PER_GROUP``, minimal step padding.

  The last step may run past the real tiles (those columns are padded /
  dropped), so among ``1..TILES_PER_GROUP`` the width with the least padding
  wins, the widest on ties (fewer grid steps). Widths below 4 are only used
  for ``tiles < 4``: a step costs more than the padded tiles it saves. The
  padded tile count never exceeds ``round_up(tiles, TILES_PER_GROUP)``, so
  the Stage 2 operand width (``_summary_width``) is the same as with
  ``TILES_PER_GROUP``-wide steps.

  Args:
    tiles: Number of real 1024-column tiles.

  Returns:
    Tiles per grid step.
  """
  if tiles <= TILES_PER_GROUP:
    return tiles
  candidates = range(4, TILES_PER_GROUP + 1)
  return min(
      candidates,
      key=lambda tps: (model_contracts.round_up(tiles, tps) - tiles, -tps),
  )


def _summary_width(tiles: int) -> int:
  """Summary columns: ``tiles`` padded to a multiple of ``TILES_PER_GROUP``.

  Padding columns hold ``(finfo(float32).min, VOCAB_TILE)``: the summary of a
  fully masked tile, which contributes ``VOCAB_TILE * exp(min - max) == 0``.

  Args:
    tiles: Number of real 1024-column tiles.

  Returns:
    The padded column count.
  """
  return model_contracts.round_up(tiles, TILES_PER_GROUP)


def _tile_stage(
    logits_2d: jax.Array,
    *,
    block_rows: int,
    tiles: int,
    vocab_limit: int | None,
    col_start: jax.Array | None,
    finalize: bool,
    interpret: bool,
    name: str,
) -> tuple[jax.Array, ...]:
  """Stage 1 (+ Stage 2 when ``finalize``) in one Pallas launch.

  Grid ``(row blocks, vocab steps)`` with the vocab axis ``"arbitrary"``: each
  step loads ``_tiles_per_step(tiles)`` 1024-column tiles, computes every
  tile's ``(max, sum_exp)`` over its ``[rows_per_step, 1024]`` slice and
  stores it in a compact ``[rows, _summary_width(tiles)]`` VMEM scratch
  (pre-filled with the masked-tile summary). The summaries never touch HBM:

  * ``finalize=True`` (unsharded normalizer): the last vocab step runs Stage 2
    (``_combine_rows``) on ``block_rows``-row slices of the scratch -- the
    operand shape of the standalone combine kernel -- and writes the
    ``[mp, SUMMARY_ALIGN]`` normalizer tile: one launch instead of two.
  * ``finalize=False`` (vocab-parallel Stage 1 of one TP rank): the last step
    writes the lane-dense ``[mp, width]`` summaries for the cross-rank
    all-gather (no ``SUMMARY_ALIGN`` broadcast).

  Each tile's reduction is a lane reduction over its own 1024 columns and the
  Stage 2 operand keeps the combine kernel's block shape, so the normalizer
  does not depend on the step width, the row count or whether the combine is
  fused (``canonical_logsoftmax_test`` checks the fused path against the TP
  combine path bitwise; on TPU the single-device program reproduces the
  TP=4 program and the pre-fusion fingerprint at ``V = 151936``).

  Args:
    logits_2d: ``[mp, cols]`` logits, ``mp % (TILES_PER_GROUP * block_rows) ==
      0`` and ``cols`` a multiple of ``_tiles_per_step(tiles) * VOCAB_TILE``
      with ``cols >= tiles * VOCAB_TILE``.
    block_rows: Rows per Stage 2 slice (8 on TPU, 1 in interpret mode).
    tiles: Number of real 1024-column tiles.
    vocab_limit: Columns ``>= vocab_limit`` (in global coordinates) are masked
      to ``finfo(float32).min``; ``None`` if no column needs masking.
    col_start: Optional traced global column of ``logits_2d[:, 0]`` (halo
      aligned TP shards); ``None`` means column 0.
    finalize: Whether to fuse Stage 2 and return the normalizer tile.
    interpret: Run the kernel in Pallas interpret mode.
    name: Kernel name.

  Returns:
    ``(norm_2d[mp, SUMMARY_ALIGN],)`` if ``finalize`` else ``(p_max, p_sum)``
    of shape ``[mp, tiles]``.
  """
  mp, cols = map(int, logits_2d.shape)
  rows_per_step = TILES_PER_GROUP * block_rows
  tiles_per_step = _tiles_per_step(tiles)
  step_cols = tiles_per_step * VOCAB_TILE
  if cols % step_cols or cols // VOCAB_TILE < tiles:
    raise ValueError(
        f"_tile_stage needs cols % {step_cols} == 0 and cols >="
        f" {tiles * VOCAB_TILE}, got cols={cols}"
    )
  steps = cols // step_cols
  width = _summary_width(steps * tiles_per_step)
  f32_min = jnp.finfo(jnp.float32).min

  def kernel(meta_ref, x_ref, *refs):
    if finalize:
      out_ref, max_sc, sum_sc = refs
    else:
      max_out, sum_out, max_sc, sum_sc = refs
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _init():
      max_sc[...] = jnp.full(max_sc.shape, f32_min, jnp.float32)
      sum_sc[...] = jnp.full(sum_sc.shape, float(VOCAB_TILE), jnp.float32)

    lane = jax.lax.broadcasted_iota(jnp.int32, (rows_per_step, width), 1)
    cur_max = max_sc[...]
    cur_sum = sum_sc[...]
    for k in range(tiles_per_step):
      x = x_ref[:, k * VOCAB_TILE : (k + 1) * VOCAB_TILE].astype(jnp.float32)
      if vocab_limit is not None:
        global_cols = (
            meta_ref[0]
            + (j * tiles_per_step + k) * VOCAB_TILE
            + jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
        )
        x = jnp.where(global_cols < vocab_limit, x, f32_min)
      tile_max = jnp.max(x, axis=-1)
      tile_sum = jnp.sum(
          jnp.exp(x - tile_max[:, None]), axis=-1, dtype=jnp.float32
      )
      hit = lane == j * tiles_per_step + k
      cur_max = jnp.where(hit, tile_max[:, None], cur_max)
      cur_sum = jnp.where(hit, tile_sum[:, None], cur_sum)
    max_sc[...] = cur_max
    sum_sc[...] = cur_sum

    @pl.when(j == steps - 1)
    def _finish():
      if finalize:
        for r in range(0, rows_per_step, block_rows):
          normalizer = _combine_rows(
              max_sc[r : r + block_rows, :], sum_sc[r : r + block_rows, :]
          )
          out_ref[r : r + block_rows, :] = jnp.broadcast_to(
              normalizer[:, None], (block_rows, SUMMARY_ALIGN)
          )
      else:
        max_out[...] = max_sc[...]
        sum_out[...] = sum_sc[...]

  if finalize:
    out_shape = jax.ShapeDtypeStruct((mp, SUMMARY_ALIGN), jnp.float32)
    out_specs = pl.BlockSpec(
        (rows_per_step, SUMMARY_ALIGN), lambda i, j, _meta: (i, 0)
    )
  else:
    summary = jax.ShapeDtypeStruct((mp, width), jnp.float32)
    out_shape = (summary, summary)
    out_specs = [
        pl.BlockSpec((rows_per_step, width), lambda i, j, _meta: (i, 0)),
        pl.BlockSpec((rows_per_step, width), lambda i, j, _meta: (i, 0)),
    ]
  meta = (
      jnp.zeros((1,), jnp.int32)
      if col_start is None
      else jnp.reshape(col_start.astype(jnp.int32), (1,))
  )
  result = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec(
                  (rows_per_step, step_cols), lambda i, j, _meta: (i, j)
              ),
          ],
          out_specs=out_specs,
          grid=(mp // rows_per_step, steps),
          scratch_shapes=[
              pltpu.VMEM((rows_per_step, width), jnp.float32),
              pltpu.VMEM((rows_per_step, width), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          allow_input_fusion=(False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=name,
  )(meta, logits_2d)
  if finalize:
    return (result,)
  p_max, p_sum = result
  return p_max[:, :tiles], p_sum[:, :tiles]


def _run_combine_stage(
    all_max: jax.Array,
    all_sum: jax.Array,
    *,
    block_rows: int,
    interpret: bool,
    name: str,
) -> jax.Array:
  """Stage 2: combines compact ``[mp, total_tiles]`` summaries into ``[mp, 128]``."""
  mp, total_tiles = map(int, all_max.shape)
  padded_tiles = _summary_width(total_tiles)
  if padded_tiles != total_tiles:
    all_max = jnp.pad(
        all_max,
        ((0, 0), (0, padded_tiles - total_tiles)),
        constant_values=jnp.finfo(jnp.float32).min,
    )
    all_sum = jnp.pad(
        all_sum,
        ((0, 0), (0, padded_tiles - total_tiles)),
        constant_values=jnp.float32(VOCAB_TILE),
    )

  def combine_kernel(max_ref, sum_ref, norm_ref):
    normalizer = _combine_rows(
        max_ref[...].astype(jnp.float32), sum_ref[...].astype(jnp.float32)
    )
    norm_ref[...] = jnp.broadcast_to(normalizer[:, None], norm_ref.shape)

  return pl.pallas_call(
      combine_kernel,
      out_shape=jax.ShapeDtypeStruct((mp, SUMMARY_ALIGN), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_rows, padded_tiles), lambda row: (row, 0)),
              pl.BlockSpec((block_rows, padded_tiles), lambda row: (row, 0)),
          ],
          out_specs=pl.BlockSpec(
              (block_rows, SUMMARY_ALIGN), lambda row: (row, 0)
          ),
          grid=(mp // block_rows,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          allow_input_fusion=(False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=name,
  )(all_max, all_sum)


def _pallas_normalizer_2d(
    logits: jax.Array,
    *,
    interpret: bool,
) -> tuple[int, int, int, jax.Array]:
  """Runs fused Stages 1+2 on unsharded ``logits[M, V]`` -> ``norm_2d[mp, 128]``."""
  m, vocab = _validate(logits)
  block_rows = 1 if interpret else ROW_BUCKET_ALIGN
  rows_per_step = TILES_PER_GROUP * block_rows
  mp = model_contracts.round_up(m, rows_per_step)
  total_tiles = (vocab + VOCAB_TILE - 1) // VOCAB_TILE
  step_cols = _tiles_per_step(total_tiles) * VOCAB_TILE
  padded_vocab = model_contracts.round_up(vocab, step_cols)

  padded_logits = logits
  if mp != m or padded_vocab != vocab:
    padded_logits = jnp.pad(
        logits,
        ((0, mp - m), (0, padded_vocab - vocab)),
        constant_values=jnp.zeros((), dtype=logits.dtype),
    )

  (norm_2d,) = _tile_stage(
      padded_logits,
      block_rows=block_rows,
      tiles=total_tiles,
      vocab_limit=vocab if padded_vocab != vocab else None,
      col_start=None,
      finalize=True,
      interpret=interpret,
      name=f"canon_logsoftmax_fused_m{mp}_v{vocab}_t{total_tiles}",
  )
  return m, vocab, block_rows, norm_2d


def _pallas_normalizer(
    logits: jax.Array,
    *,
    interpret: bool,
) -> jax.Array:
  """Returns 1D ``float32[M]`` fixed-order row log-normalizers."""
  m, _, _, norm_2d = _pallas_normalizer_2d(logits, interpret=interpret)
  return norm_2d[:m, 0]


def log_softmax(
    logits: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
  """Returns the fixed-program ``float32[M, V]`` primal with analytic VJP."""
  _validate(logits)
  in_dtype = logits.dtype

  @jax.custom_vjp
  def op(value: jax.Array) -> jax.Array:
    normalizer = _pallas_normalizer(value, interpret=interpret)
    return _as_f32(value) - normalizer[:, None]

  def forward(value: jax.Array):
    output = op(value)
    return output, output

  def backward(output: jax.Array, cotangent: jax.Array):
    cot_f32 = cotangent.astype(jnp.float32)
    probability = jnp.exp(output)
    cotangent_sum = jnp.sum(cot_f32, axis=-1, keepdims=True)
    return ((cot_f32 - probability * cotangent_sum).astype(in_dtype),)

  op.defvjp(forward, backward)
  return op(logits)


def token_logprobs(
    logits: jax.Array,
    token_ids: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
  """Returns per-row target token log-probabilities with fused analytic VJP."""
  m, _ = _validate(logits)
  if token_ids.ndim != 1 or int(token_ids.shape[0]) != m:
    raise CanonicalLogSoftmaxError(
        "canonical token_logprobs expects token_ids[M] matching logits[M,V], "
        f"got {logits.shape}/{token_ids.shape}"
    )
  in_dtype = logits.dtype
  tok_ids_i32 = token_ids.astype(jnp.int32)

  def _primal_and_normalizer(
      value: jax.Array, tok_ids: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    normalizer = _pallas_normalizer(value, interpret=interpret)
    token_column = _as_f32(
        jnp.take_along_axis(value, tok_ids[:, None], axis=-1)[:, 0]
    )
    return token_column - normalizer, normalizer

  @jax.custom_vjp
  def op(value: jax.Array, tok_ids: jax.Array) -> jax.Array:
    token_logprob, _ = _primal_and_normalizer(value, tok_ids)
    return token_logprob

  def forward(value: jax.Array, tok_ids: jax.Array):
    token_logprob, normalizer = _primal_and_normalizer(value, tok_ids)
    return token_logprob, (value, normalizer, tok_ids)

  def backward(residual, cotangent: jax.Array):
    value, normalizer, tok_ids = residual
    cot_f32 = cotangent.astype(jnp.float32)
    probability = jnp.exp(value.astype(jnp.float32) - normalizer[:, None])
    grad_logits = -probability * cot_f32[:, None]
    rows = jnp.arange(value.shape[0], dtype=jnp.int32)
    grad_logits = grad_logits.at[rows, tok_ids].add(cot_f32)
    return grad_logits.astype(in_dtype), None

  op.defvjp(forward, backward)
  return op(logits, tok_ids_i32)


def gathered_logprobs(
    logits: jax.Array,
    token_ids: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Sampled-token logprob, top-1, and 1-based ``>=``-rank without row materialize."""
  m, vocab, block_rows, norm_2d = _pallas_normalizer_2d(
      logits, interpret=interpret
  )
  if token_ids.ndim != 1 or int(token_ids.shape[0]) != m:
    raise CanonicalLogSoftmaxError(
        "canonical gathered_logprobs expects token_ids[M] matching logits[M,V],"
        f" got {logits.shape}/{token_ids.shape}"
    )
  mp = int(norm_2d.shape[0])
  group_width = TILES_PER_GROUP * VOCAB_TILE
  vocab_groups = (vocab + group_width - 1) // group_width
  padded_vocab = vocab_groups * group_width

  normalizer = norm_2d[:m, 0]
  token_column = _as_f32(
      jnp.take_along_axis(
          logits, token_ids.astype(jnp.int32)[:, None], axis=-1
      )[:, 0]
  )
  token_logprob = token_column - normalizer

  if mp != m:
    padded_token_lp = jnp.pad(
        token_logprob, ((0, mp - m),), constant_values=jnp.float32(0.0)
    )
  else:
    padded_token_lp = token_logprob
  token_tile = jnp.broadcast_to(padded_token_lp[:, None], (mp, SUMMARY_ALIGN))
  if mp != m or padded_vocab != vocab:
    padded_logits = jnp.pad(
        logits,
        ((0, mp - m), (0, padded_vocab - vocab)),
        constant_values=jnp.zeros((), dtype=logits.dtype),
    )
  else:
    padded_logits = logits

  def gather_kernel(x_ref, norm_ref, token_ref, rank_ref, val_ref, idx_ref):
    group = pl.program_id(1)
    x = x_ref[...].astype(jnp.float32)
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    global_col = group * group_width + lane
    if padded_vocab != vocab:
      x = jnp.where(global_col < vocab, x, jnp.finfo(jnp.float32).min)
    norm_val = norm_ref[:, 0].astype(jnp.float32)
    flat = x - norm_val[:, None]
    token_value = token_ref[:, 0].astype(jnp.float32)
    rank_part = jnp.sum(flat >= token_value[:, None], axis=-1).astype(jnp.int32)
    tile_max = jnp.max(flat, axis=-1)
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
          jax.ShapeDtypeStruct((mp, SUMMARY_ALIGN), jnp.int32),
          jax.ShapeDtypeStruct((mp, SUMMARY_ALIGN), jnp.float32),
          jax.ShapeDtypeStruct((mp, SUMMARY_ALIGN), jnp.int32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (block_rows, group_width), lambda row, group: (row, group)
              ),
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN), lambda row, _group: (row, 0)
              ),
              pl.BlockSpec(
                  (block_rows, SUMMARY_ALIGN), lambda row, _group: (row, 0)
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
          grid=(mp // block_rows, vocab_groups),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          allow_input_fusion=(False, False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f"canon_logsoftmax_gather_m{mp}_v{vocab}_vp{padded_vocab}",
  )(padded_logits, norm_2d, token_tile)

  return (
      token_logprob,
      top_value[:m, 0],
      top_index[:m, 0],
      rank[:m, 0],
  )


def vocab_parallel_logsoftmax_admitted(vocab: int, tp_size: int) -> bool:
  """Whether ``[M, V]`` logits sharded ``V/TP`` per rank admit vocab-parallel scoring.

  Every rank needs at least one ``VOCAB_TILE = 1024`` slice so that at most one
  neighbor's 1024-column halo suffices to align all ranks onto the global
  1024-column tile boundaries. Admits Qwen3 ``V = 151936`` at ``TP = 2 / 4 /
  8``.

  Args:
    vocab: Full vocabulary width ``V``.
    tp_size: Tensor-parallel degree along the vocabulary axis.

  Returns:
    ``True`` if ``vocab_parallel_normalizer`` reproduces ``_pallas_normalizer``
    bit for bit.
  """
  return (
      tp_size > 1
      and vocab > 0
      and vocab % tp_size == 0
      and (vocab // tp_size) >= VOCAB_TILE
  )


def vocab_parallel_normalizer(
    logits_local: jax.Array,
    *,
    tp_axis: str,
    tp_size: int,
    interpret: bool = False,
) -> jax.Array:
  """Stages 1+2 normalizer on TP-sharded ``logits_local[m, v_local]``.

  Bitwise identical to ``_pallas_normalizer`` on the full ``[m, tp_size *
  v_local]`` logits:
  - When ``v_local % VOCAB_TILE == 0``, each rank's slice already starts and
    ends on a global 1024-column tile boundary, so Stage 1 runs with zero
    neighbor exchange.
  - When ``v_local % VOCAB_TILE != 0`` (e.g. ``V = 151936`` at ``TP = 2/4/8``),
    rank ``r`` owns global 1024-column tiles ``[t_start_r, t_end_r)`` where
    ``t_start_r = ceil(r * v_local / 1024)``. Because ``v_local >= 1024``, rank
    ``r`` needs at most the first 1024 columns of rank ``r + 1``
    (``lax.ppermute`` of ``[mp, 1024]`` — 2 KB/row in ``bf16``) to assemble its
    exact global 1024-column tiles before running Stage 1, and columns past
    ``vocab`` on the final rank are masked to ``finfo(float32).min`` in-kernel.

  Args:
    logits_local: Local ``[m, v_local]`` logits slice (``bf16`` or ``f32``).
    tp_axis: Tensor-parallel mesh axis name.
    tp_size: Tensor-parallel degree along ``tp_axis``.
    interpret: Whether to execute Pallas in CPU interpret mode.

  Returns:
    1D ``float32[m]`` row log-normalizers.
  """
  m, v_local = _validate(logits_local)
  if tp_size <= 1 or v_local < VOCAB_TILE:
    raise ValueError(
        f"vocab-parallel normalizer needs v_local >= {VOCAB_TILE} (and"
        f" tp_size > 1), got v_local={v_local}, tp_size={tp_size}; gate on"
        " vocab_parallel_logsoftmax_admitted()"
    )
  block_rows = 1 if interpret else ROW_BUCKET_ALIGN
  rows_per_step = TILES_PER_GROUP * block_rows
  mp = model_contracts.round_up(m, rows_per_step)
  if mp != m:
    logits_local = jnp.pad(
        logits_local,
        ((0, mp - m), (0, 0)),
        constant_values=jnp.zeros((), dtype=logits_local.dtype),
    )

  vocab = tp_size * v_local
  total_tiles = (vocab + VOCAB_TILE - 1) // VOCAB_TILE

  if v_local % VOCAB_TILE == 0:
    max_tiles_local = v_local // VOCAB_TILE
    counts = [max_tiles_local] * tp_size
    aligned_local = logits_local
    vocab_limit = None
    col_start = None
  else:
    tile_starts = [
        (r * v_local + VOCAB_TILE - 1) // VOCAB_TILE for r in range(tp_size)
    ]
    tile_ends = tile_starts[1:] + [total_tiles]
    counts = [end - start for start, end in zip(tile_starts, tile_ends)]
    max_tiles_local = max(counts)
    slice_width = max_tiles_local * VOCAB_TILE
    next_head = lax.ppermute(
        logits_local[:, :VOCAB_TILE],
        tp_axis,
        perm=[(src, (src - 1) % tp_size) for src in range(tp_size)],
    )
    min_ext_width = max(
        tile_starts[r] * VOCAB_TILE - r * v_local + slice_width
        for r in range(tp_size)
    )
    extra_pad = max(0, min_ext_width - (v_local + VOCAB_TILE))
    parts = [logits_local, next_head]
    if extra_pad > 0:
      parts.append(jnp.zeros((mp, extra_pad), dtype=logits_local.dtype))
    ext_local = jnp.concatenate(parts, axis=1)

    axis_idx = lax.axis_index(tp_axis)
    t_start_col = (
        (axis_idx * v_local + VOCAB_TILE - 1) // VOCAB_TILE
    ) * VOCAB_TILE
    col_offset = t_start_col - axis_idx * v_local
    aligned_local = lax.dynamic_slice_in_dim(
        ext_local, col_offset, slice_width, axis=1
    )
    if vocab % VOCAB_TILE != 0:
      vocab_limit = vocab
      col_start = t_start_col
    else:
      vocab_limit = None
      col_start = None

  # Step padding (`_tiles_per_step`): the padded tiles are computed and
  # sliced off by `_tile_stage`; tiles of a rank past its own `count` are
  # dropped by `valid_tile_indices` after the all-gather.
  step_cols = _tiles_per_step(max_tiles_local) * VOCAB_TILE
  step_padded = model_contracts.round_up(
      max_tiles_local * VOCAB_TILE, step_cols
  )
  if step_padded != max_tiles_local * VOCAB_TILE:
    aligned_local = jnp.pad(
        aligned_local,
        ((0, 0), (0, step_padded - max_tiles_local * VOCAB_TILE)),
        constant_values=jnp.zeros((), dtype=aligned_local.dtype),
    )
  valid_indices_list: list[int] = []
  for r, count in enumerate(counts):
    valid_indices_list.extend(r * max_tiles_local + k for k in range(count))
  valid_tile_indices = (
      None
      if len(valid_indices_list) == tp_size * max_tiles_local
      else jnp.asarray(valid_indices_list, dtype=jnp.int32)
  )

  p_max, p_sum = _tile_stage(
      aligned_local,
      block_rows=block_rows,
      tiles=max_tiles_local,
      vocab_limit=vocab_limit,
      col_start=col_start,
      finalize=False,
      interpret=interpret,
      name=f"canon_logsoftmax_vp_stage1_m{mp}_vl{v_local}_t{max_tiles_local}",
  )
  local_summary = jnp.stack([p_max, p_sum])  # [2, mp, max_tiles_local]
  all_summary = lax.all_gather(local_summary, tp_axis, axis=2, tiled=True)
  if valid_tile_indices is not None:
    all_summary = jnp.take(all_summary, valid_tile_indices, axis=2)

  norm_2d = _run_combine_stage(
      all_summary[0],
      all_summary[1],
      block_rows=block_rows,
      interpret=interpret,
      name=f"canon_logsoftmax_vp_combine_m{mp}_t{total_tiles}",
  )
  return norm_2d[:m, 0]


def _resolve_row_sharding(
    mesh: MeshLike,
    data_axes: tuple[str, ...],
    num_shards: int | None,
) -> tuple[tuple[str, ...] | None, int]:
  """Resolves row-axis ``PartitionSpec`` entry and validates ``num_shards``."""
  mesh_shards = math.prod((int(mesh.shape[a]) for a in data_axes), start=1)
  if num_shards is not None and num_shards != mesh_shards:
    raise ValueError(
        f"num_shards={num_shards} does not match mesh.shape over"
        f" data_axes={data_axes} ({mesh_shards})"
    )
  return (data_axes if (data_axes and mesh_shards > 1) else None), mesh_shards


def vocab_parallel_log_softmax(
    flat_logits: jax.Array,
    *,
    mesh: MeshLike,
    tp_axis: str,
    tp_size: int,
    data_axes: tuple[str, ...] = (),
    num_shards: int | None = None,
    interpret: bool = False,
    exact_residual_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Vocab-parallel ``log_softmax`` over ``P(data_axes, tp_axis)`` ``flat_logits[M, V]``."""
  m, vocab = _validate(flat_logits)
  if not vocab_parallel_logsoftmax_admitted(vocab, tp_size):
    raise ValueError(
        f"vocab_parallel_log_softmax needs v_local >= {VOCAB_TILE} (and"
        f" vocab % tp_size == 0, tp_size > 1), got vocab={vocab},"
        f" tp_size={tp_size}"
    )
  row_spec, resolved_shards = _resolve_row_sharding(mesh, data_axes, num_shards)
  mp = (
      model_contracts.round_up(m, resolved_shards)
      if row_spec is not None
      else m
  )
  if mp != m:
    padded_logits = jnp.pad(
        flat_logits,
        ((0, mp - m), (0, 0)),
        constant_values=jnp.zeros((), dtype=flat_logits.dtype),
    )
  else:
    padded_logits = flat_logits
  in_dtype = flat_logits.dtype

  def _fwd_local(logits_local: jax.Array):
    normalizer = vocab_parallel_normalizer(
        logits_local,
        tp_axis=tp_axis,
        tp_size=tp_size,
        interpret=interpret,
    )
    out_local = _as_f32(logits_local) - normalizer[:, None]
    logits_res = (
        logits_local.astype(exact_residual_dtype)
        if exact_residual_dtype is not None
        else logits_local
    )
    return out_local, (logits_res, normalizer)

  def _bwd_local(res_local, cot_local: jax.Array):
    logits_res, normalizer = res_local
    logits_f32_local = logits_res.astype(jnp.float32)
    cot_f32_local = cot_local.astype(jnp.float32)
    prob_local = jnp.exp(logits_f32_local - normalizer[:, None])
    cot_sum = lax.psum(jnp.sum(cot_f32_local, axis=-1, keepdims=True), tp_axis)
    grad_f32 = cot_f32_local - prob_local * cot_sum
    return grad_f32.astype(in_dtype)

  fwd_shmap = jax.shard_map(
      _fwd_local,
      mesh=mesh,
      in_specs=(P(row_spec, tp_axis),),
      out_specs=(P(row_spec, tp_axis), (P(row_spec, tp_axis), P(row_spec))),
      check_vma=False,
  )
  bwd_shmap = jax.shard_map(
      _bwd_local,
      mesh=mesh,
      in_specs=((P(row_spec, tp_axis), P(row_spec)), P(row_spec, tp_axis)),
      out_specs=P(row_spec, tp_axis),
      check_vma=False,
  )

  @jax.custom_vjp
  def op(logits_in: jax.Array) -> jax.Array:
    out, _ = fwd_shmap(logits_in)
    return out

  def _fwd(logits_in: jax.Array):
    return fwd_shmap(logits_in)

  def _bwd(residual, cotangent: jax.Array):
    return (bwd_shmap(residual, cotangent),)

  op.defvjp(_fwd, _bwd)
  out = op(padded_logits)
  return out[:m, :] if mp != m else out


def vocab_parallel_token_logprobs(
    flat_logits: jax.Array,
    flat_tokens: jax.Array,
    *,
    mesh: MeshLike,
    tp_axis: str,
    tp_size: int,
    data_axes: tuple[str, ...] = (),
    num_shards: int | None = None,
    interpret: bool = False,
    exact_residual_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Vocab-parallel ``token_logprobs`` over ``P(data_axes, tp_axis)`` ``flat_logits[M, V]``."""
  m, vocab = _validate(flat_logits)
  if flat_tokens.ndim != 1 or int(flat_tokens.shape[0]) != m:
    raise CanonicalLogSoftmaxError(
        "vocab_parallel_token_logprobs expects flat_tokens[M] matching"
        f" flat_logits[M,V], got {flat_logits.shape}/{flat_tokens.shape}"
    )
  if not vocab_parallel_logsoftmax_admitted(vocab, tp_size):
    raise ValueError(
        f"vocab_parallel_token_logprobs needs v_local >= {VOCAB_TILE} (and"
        f" vocab % tp_size == 0, tp_size > 1), got vocab={vocab},"
        f" tp_size={tp_size}"
    )
  v_local = vocab // tp_size
  row_spec, resolved_shards = _resolve_row_sharding(mesh, data_axes, num_shards)
  mp = (
      model_contracts.round_up(m, resolved_shards)
      if row_spec is not None
      else m
  )
  tok_i32 = flat_tokens.astype(jnp.int32)
  if mp != m:
    padded_logits = jnp.pad(
        flat_logits,
        ((0, mp - m), (0, 0)),
        constant_values=jnp.zeros((), dtype=flat_logits.dtype),
    )
    padded_tokens = jnp.pad(
        tok_i32, ((0, mp - m),), constant_values=jnp.int32(0)
    )
  else:
    padded_logits = flat_logits
    padded_tokens = tok_i32
  in_dtype = flat_logits.dtype

  def _fwd_local(logits_local: jax.Array, tok_1d: jax.Array):
    normalizer = vocab_parallel_normalizer(
        logits_local,
        tp_axis=tp_axis,
        tp_size=tp_size,
        interpret=interpret,
    )
    axis_idx = lax.axis_index(tp_axis)
    v_start = axis_idx * v_local
    local_tok = tok_1d - v_start
    in_shard = (local_tok >= 0) & (local_tok < v_local)
    safe_tok = jnp.where(in_shard, local_tok, 0)
    gathered = _as_f32(
        jnp.take_along_axis(logits_local, safe_tok[:, None], axis=-1)[:, 0]
    )
    token_col = lax.psum(
        jnp.where(in_shard, gathered, jnp.float32(0.0)), tp_axis
    )
    token_logprob = token_col - normalizer
    logits_res = (
        logits_local.astype(exact_residual_dtype)
        if exact_residual_dtype is not None
        else logits_local
    )
    return token_logprob, (logits_res, normalizer, tok_1d)

  def _bwd_local(res_local, cot_1d: jax.Array):
    logits_res, normalizer, tok_1d = res_local
    logits_f32_local = logits_res.astype(jnp.float32)
    cot_f32 = cot_1d.astype(jnp.float32)
    prob_local = jnp.exp(logits_f32_local - normalizer[:, None])
    grad_local = -prob_local * cot_f32[:, None]
    axis_idx = lax.axis_index(tp_axis)
    v_start = axis_idx * v_local
    local_tok = tok_1d - v_start
    in_shard = (local_tok >= 0) & (local_tok < v_local)
    safe_tok = jnp.where(in_shard, local_tok, 0)
    rows = jnp.arange(logits_f32_local.shape[0], dtype=jnp.int32)
    grad_local = grad_local.at[rows, safe_tok].add(
        jnp.where(in_shard, cot_f32, jnp.float32(0.0))
    )
    return grad_local.astype(in_dtype)

  fwd_shmap = jax.shard_map(
      _fwd_local,
      mesh=mesh,
      in_specs=(P(row_spec, tp_axis), P(row_spec)),
      out_specs=(
          P(row_spec),
          (P(row_spec, tp_axis), P(row_spec), P(row_spec)),
      ),
      check_vma=False,
  )
  bwd_shmap = jax.shard_map(
      _bwd_local,
      mesh=mesh,
      in_specs=(
          (P(row_spec, tp_axis), P(row_spec), P(row_spec)),
          P(row_spec),
      ),
      out_specs=P(row_spec, tp_axis),
      check_vma=False,
  )

  @jax.custom_vjp
  def op(logits_in: jax.Array, tok_in: jax.Array) -> jax.Array:
    out, _ = fwd_shmap(logits_in, tok_in)
    return out

  def _fwd(logits_in: jax.Array, tok_in: jax.Array):
    return fwd_shmap(logits_in, tok_in)

  def _bwd(residual, cotangent: jax.Array):
    return bwd_shmap(residual, cotangent), None

  op.defvjp(_fwd, _bwd)
  out = op(padded_logits, padded_tokens)
  return out[:m] if mp != m else out
