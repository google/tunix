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
"""Fixed-tile, batch-invariant Pallas TPU matmul (Zero-TIM "P22.XE").

Why this kernel exists: XLA's ``jnp.dot`` picks a different tiling/reduction
strategy per operand shape, so the value of one output row depends on M (the
number of rows in the batch).  Decode (M=1..8) and prefill/scoring (M=bucket)
therefore disagree by a few ULPs.  This kernel fixes the per-element
contraction order: every output element is ``sum_q x[i, q*BK:(q+1)*BK] @
y[q*BK:(q+1)*BK, j]`` accumulated left-to-right in f32 over ``q`` and rounded
to bf16 exactly once, independently of M, of the row position, and of the
contents of any other row.  ``block_m``/``block_n`` only choose which output
tile a program writes, so they are bitwise neutral; ``block_k`` is the only
tile that enters the numerics and is never changed by the policy.
"""

from __future__ import annotations

import os

from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

BM = 128
BN = 256
BK = 256
# The (128, 256, 256) tiles ran every
# production shape at 55-59 TFLOP/s (14% of v5p bf16 peak).  With the k-block
# accumulation order (BK) unchanged, wider output tiles are bitwise identical
# per element and 2-4x faster ([128,4096]x[4096,6144] 118 -> 58 us at
# (128,2048|1024,256); [512,4096]x[4096,6144] 439 -> 107 us at (256,1024,256);
# every variant bitwise == production).  Small block_m (8-64) is slower
# (weight streaming), so rows are never sub-tiled.
TILE_POLICY = "v2"
_TILE_RECEIPTS: set[tuple[int, int, int]] = set()
# The fixed tile triples the P22.XF model contracts pass explicitly
# (model_contracts.py: tp1/tp2/1.7B 128/256/256, tp8 128/128/128).
# A caller passing one of these gets the shape policy; any other explicit
# tiles are honored as written.
CONTRACT_TILES = frozenset({(BM, BN, BK), (128, 128, 128)})
# Output-column tile candidates in preference order (bitwise-neutral: bn never
# enters an element's contraction order).  See tile_policy.
BLOCK_N_CANDIDATES = (1024, 1280, 640, 512)

# Compiler-only knob: producer fusion changes materialization, not values.
INPUT_FUSION_ENV = "CANON_PALLAS_INPUT_FUSION"
# P66/P67 manual-axis (VMA) typing knobs.  They only matter while tracing
# inside a checked ``shard_map`` over the P59 ("data", "model") manual mesh.
P66_CHECK_VMA_ENV = "CANON_P66_P59_CHECK_VMA"
P67_SCOPE_ENV = "CANON_P67_P66_VMA_P59_ONLY"
P59_RANK_PARALLEL_BACKWARD_ENV = "CANON_P59_RANK_PARALLEL_BACKWARD"


def tile_policy(
    m: int, k: int, n: int, tiles: tuple[int, int, int] = (BM, BN, BK)
) -> tuple[int, int, int]:
  """Return (block_m, block_n, block_k) for a [m,k]@[k,n] bf16 matmul.

  block_m 256 when m divides by 256 else the caller's; block_n the first of
  BLOCK_N_CANDIDATES dividing n else the caller's; block_k always the
  caller's, so the per-element accumulation order never changes (the tp8
  contract accumulates in 128-wide k blocks).  1280 and 640 exist for the
  Qwen3-4B TP-local widths (2560-multiples: o_proj/down_proj N=2560, and
  the contract-padded MLP width 2432->2560, 1216->1280); they divide none of
  the 8B/1.7B widths, so those shapes keep the 1024/512 choice
  (o_proj 22.9->15.4 us, down_proj 42.5->24.9 us at M=256, bitwise == 128/512).

  Args:
    m: Rows of the left operand.
    k: Contracted extent (unused; kept for signature parity).
    n: Columns of the right operand.
    tiles: The caller's (block_m, block_n, block_k).

  Returns:
    The (block_m, block_n, block_k) the kernel runs with.
  """
  del k  # The contraction tile is never changed by the policy.
  block_m, block_n, block_k = tiles
  if m % 256 == 0:
    block_m = 256
  for candidate in BLOCK_N_CANDIDATES:
    if n % candidate == 0:
      block_n = candidate
      break
  return block_m, block_n, block_k


def _tile_receipt(m: int, k: int, n: int, tiles: tuple[int, int, int]) -> None:
  key = (m, k, n)
  if key in _TILE_RECEIPTS:
    return
  _TILE_RECEIPTS.add(key)
  logging.info(
      "[PATHTRACE] matmul_tile_policy=%s M=%d K=%d N=%d bm=%d bn=%d bk=%d",
      TILE_POLICY,
      m,
      k,
      n,
      tiles[0],
      tiles[1],
      tiles[2],
  )


def p67_p59_vma_context() -> bool:
  """Return whether a P67-scoped VMA mutation is inside P59's outer map.

  P66's compatibility flag is process-wide, while the engine's serving and
  trainer programs share that process.  The repair candidate must therefore
  preserve the historical serving program and retain VMA annotations only
  while P59 is tracing its exact manual DP/TP pullback.
  """
  scope = os.environ.get(P67_SCOPE_ENV, "0")
  if scope not in ("0", "1"):
    raise ValueError(f"{P67_SCOPE_ENV} must be exactly 0 or 1")
  if scope != "1":
    return True
  if os.environ.get(P59_RANK_PARALLEL_BACKWARD_ENV, "") != "1":
    return False
  context = jax.sharding.get_abstract_mesh()
  if tuple(context.axis_names) != ("data", "model"):
    return False
  axis_types = dict(zip(context.axis_names, context.axis_types))
  return (
      axis_types.get("data") is jax.sharding.AxisType.Manual
      and axis_types.get("model") is jax.sharding.AxisType.Manual
      # The exact FrozenLake singleton P59 carrier keeps a manual data
      # axis even though its size is one.  Output VMA metadata is still
      # mandatory for the outer checked shard_map in that geometry.
      and int(context.shape["data"]) >= 1
      and int(context.shape["model"]) > 1
  )


def _vma_active() -> bool:
  return os.environ.get(P66_CHECK_VMA_ENV, "0") == "1" and p67_p59_vma_context()


def p66_vma_align_operands(*values):
  """Give operands of a local Pallas operation one common VMA type.

  JAX dot primitives require all operands to have matching varying manual
  axes.  A pcast from replicated to varying is a runtime identity; its
  transpose supplies the psum that the replicated operand needs.  P66 only
  admits plain-varying state here, never an already reduced/unreduced value.

  Args:
    *values: The operands of one local Pallas call.

  Returns:
    The operands, each pcast to the union of their varying manual axes when
    the P66 VMA mode is active; otherwise the operands unchanged.
  """
  if not _vma_active():
    return values
  mats = tuple(jax.typeof(value).mat for value in values)
  if any(mat.unreduced or mat.reduced for mat in mats):
    raise ValueError(
        "P66 Pallas VMA does not admit reduced/unreduced operands: "
        + ", ".join(str(mat) for mat in mats)
    )
  varying = frozenset().union(*(mat.varying for mat in mats))
  aligned = []
  for value, mat in zip(values, mats, strict=True):
    for axis in sorted(varying - mat.varying):
      value = jax.lax.pcast(value, axis, to="varying")
    aligned.append(value)
  return tuple(aligned)


def p66_vma_output_manual_axis_type(*values):
  """Return the explicit Pallas output VMA type for aligned operands."""
  if not _vma_active():
    return None
  mats = tuple(jax.typeof(value).mat for value in values)
  if any(mat.unreduced or mat.reduced for mat in mats):
    raise ValueError(
        "P66 Pallas VMA output does not admit reduced/unreduced operands: "
        + ", ".join(str(mat) for mat in mats)
    )
  varying = frozenset().union(*(mat.varying for mat in mats))
  return jax.sharding.ManualAxisType(varying=varying)


def input_fusion_enabled() -> bool:
  """Whether Pallas producer (input) fusion is requested (compile-only knob)."""
  return os.environ.get(INPUT_FUSION_ENV, "") == "1"


def matmul(
    x,
    y,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = BM,
    block_n: int = BN,
    block_k: int = BK,
):
  """Compute bf16 [M,K] @ [K,N] with fixed BM/BN/BK and f32 accumulation."""
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(f"P22.XE expects rank-2 inputs, got {x.shape}, {y.shape}")
  m, k = map(int, x.shape)
  ky, n = map(int, y.shape)
  if k != ky:
    raise ValueError(f"P22.XE contracted dimensions differ: {k} vs {ky}")
  if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
    raise ValueError(f"P22.XE requires bf16 inputs, got {x.dtype}, {y.dtype}")
  if (block_m, block_n, block_k) in CONTRACT_TILES:
    # Callers that pass a contract's fixed tiles (or nothing) get the
    # shape policy; explicit non-contract tiles are honored as written.
    block_m, block_n, block_k = tile_policy(
        m, k, n, (block_m, block_n, block_k)
    )
    _tile_receipt(m, k, n, (block_m, block_n, block_k))
  if min(block_m, block_n, block_k) <= 0:
    raise ValueError("P22.XE block sizes must be positive")
  if m % block_m or n % block_n or k % block_k:
    raise ValueError(
        "P22.XE shape must divide BM/BN/BK="
        f"{block_m}/{block_n}/{block_k}, got {(m, k, n)}"
    )
  x, y = p66_vma_align_operands(x, y)

  def _kernel(x_ref, y_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

    # The (i, j) output block is written back
    # only after its last k step, so store it once there instead of on
    # every k step.  The f32 accumulation sequence and the single final
    # bf16 cast are unchanged, so every element is bitwise the
    # per-step-store form; only the per-k-step VMEM store goes away.
    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _finalize():
      out_ref[...] = acc_ref[...].astype(out_ref.dtype)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, n),
          jnp.bfloat16,
          manual_axis_type=p66_vma_output_manual_axis_type(x, y),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, block_k), lambda i, _j, q: (i, q)),
              pl.BlockSpec((block_k, block_n), lambda _i, j, q: (q, j)),
          ],
          out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, _q: (i, j)),
          grid=(m // block_m, n // block_n, k // block_k),
          scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          # P56.4.7: producer fusion changes materialization, not
          # values -- the fused producers are elementwise-exact and
          # the kernel reads identical operand values either way.
          allow_input_fusion=(
              (True, True) if input_fusion_enabled() else (False, False)
          ),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_matmul_bm{block_m}_bn{block_n}_bk{block_k}",
  )(x, y)
