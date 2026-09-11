#!/usr/bin/env python3
"""Fixed-feature-order bf16 RMSNorm custom call for additive P22.XH."""

from __future__ import annotations

import os

from p22xh_contract import BF, BM, validate_shape
from p22_pallas_matmul import p66_vma_align_operands
from p22_pallas_matmul import p66_vma_output_manual_axis_type


# tasks/zero_tim_perf3 A: the contract row block BM=8 ran the per-head norms
# (rows = tokens x heads, e.g. 8192 x 128 in the 512-token prefill program) as
# 1024 grid steps of fixed overhead, ~30% of that program's device time.  The
# feature-block reduction (BF, left-to-right f32) is the numerics; the row block
# only decides how many independent rows share a grid step, so wider row blocks
# are bitwise identical (scratch/zero_tim_perf3/probe/a1_rmsnorm_probe.log,
# 2026-09-11: bm 8/64/256/512 all == BM8 on 12 shapes; [8192,128] 242 -> 58 us,
# [8192,4096] 272 -> 84 us).  bm=512 exhausts scoped VMEM at F=4096, so the
# policy stops at 256 and at the probed feature widths.
ROW_TILES_ENV = "CANON_PALLAS_RMSNORM_TILES"
ROW_TILE_WIDE = 256
ROW_TILE_MAX_FEATURES = 4096
_ROW_RECEIPTS: set[tuple[int, int]] = set()


def row_tile(m: int, f: int) -> int:
    """Return block_m for an [m, f] rmsnorm: 256 when m divides by 256 and f is
    within the probed width, else the contract BM.  v1 always returns BM."""
    if os.environ.get(ROW_TILES_ENV, "v2") == "v1":
        return BM
    if m % ROW_TILE_WIDE == 0 and f <= ROW_TILE_MAX_FEATURES:
        return ROW_TILE_WIDE
    return BM


def _row_receipt(m: int, f: int, block_m: int) -> None:
    key = (m, f)
    if key in _ROW_RECEIPTS:
        return
    _ROW_RECEIPTS.add(key)
    print(
        f"[PATHTRACE] {ROW_TILES_ENV}={os.environ.get(ROW_TILES_ENV, 'v2')} "
        f"rows={m} F={f} bm={block_m} bf={BF}",
        flush=True,
    )


def _imports():
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    return jax, jnp, pl, pltpu


def canonical_rmsnorm(x, weight, *, epsilon: float):
    """Declared P22.XH semantic: fixed BF-block f32 accumulation, bf16 output."""
    jax, jnp, _pl, _pltpu = _imports()
    m, f = validate_shape(x.shape, weight.shape)
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XH requires bf16 inputs, got {x.dtype}, {weight.dtype}")
    if not float(epsilon) > 0.0:
        raise ValueError(f"P22.XH epsilon must be positive, got {epsilon}")
    xf = x.astype(jnp.float32).reshape(m, f // BF, BF)
    wf = weight.astype(jnp.float32)

    # Python-static expansion preserves the registered left-to-right BF-block
    # order and avoids a dynamic_slice primitive, which Mosaic TPU does not
    # lower inside a Pallas kernel in this exact build.
    sumsq = jnp.zeros((m,), jnp.float32)
    for q in range(f // BF):
        block = xf[:, q, :]
        sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
    inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
    return (x.astype(jnp.float32) * inv[:, None] * wf[None, :]).astype(jnp.bfloat16)


def rmsnorm(
    x,
    weight,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
):
    """Apply the declared P22.XH semantic to TP-local rank-2 bf16 arrays."""
    jax, jnp, pl, pltpu = _imports()
    m, f = validate_shape(x.shape, weight.shape)
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XH requires bf16 inputs, got {x.dtype}, {weight.dtype}")
    if not float(epsilon) > 0.0:
        raise ValueError(f"P22.XH epsilon must be positive, got {epsilon}")
    x, weight = p66_vma_align_operands(jax, x, weight)
    block_m = row_tile(m, f)
    _row_receipt(m, f, block_m)

    def _kernel(x_ref, weight_ref, out_ref):
        xb = x_ref[...].astype(jnp.float32).reshape(block_m, f // BF, BF)
        wb = weight_ref[...].astype(jnp.float32)

        sumsq = jnp.zeros((block_m,), dtype=jnp.float32)
        for q in range(f // BF):
            block = xb[:, q, :]
            sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
        inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
        out_ref[...] = (
            x_ref[...].astype(jnp.float32) * inv[:, None] * wb[None, :]
        ).astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (m, f),
            jnp.bfloat16,
            manual_axis_type=p66_vma_output_manual_axis_type(
                jax, x, weight
            ),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, f), lambda i: (i, 0)),
                pl.BlockSpec((f,), lambda _i: (0,)),
            ],
            out_specs=pl.BlockSpec((block_m, f), lambda i: (i, 0)),
            grid=(m // block_m,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            # P56.4.7: producer fusion changes materialization, not
            # values (elementwise-exact producers; layout is not value).
            allow_input_fusion=(
                (True, True)
                if os.environ.get("CANON_PALLAS_INPUT_FUSION", "") == "1"
                else (False, False)
            ),
            shape_invariant_numerics=shape_invariant_numerics,
        ),
        interpret=interpret,
        name=f"canon_rmsnorm_bm{block_m}_bf{BF}_f{f}",
    )(x, weight)
