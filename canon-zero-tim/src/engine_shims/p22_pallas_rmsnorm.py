#!/usr/bin/env python3
"""Fixed-feature-order bf16 RMSNorm custom call for additive P22.XH."""

from __future__ import annotations

import os

from p22xh_contract import BF, BM, validate_shape


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

    def _kernel(x_ref, weight_ref, out_ref):
        xb = x_ref[...].astype(jnp.float32).reshape(BM, f // BF, BF)
        wb = weight_ref[...].astype(jnp.float32)

        sumsq = jnp.zeros((BM,), dtype=jnp.float32)
        for q in range(f // BF):
            block = xb[:, q, :]
            sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
        inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
        out_ref[...] = (
            x_ref[...].astype(jnp.float32) * inv[:, None] * wb[None, :]
        ).astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, f), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((BM, f), lambda i: (i, 0)),
                pl.BlockSpec((f,), lambda _i: (0,)),
            ],
            out_specs=pl.BlockSpec((BM, f), lambda i: (i, 0)),
            grid=(m // BM,),
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
        name=f"canon_rmsnorm_bm{BM}_bf{BF}_f{f}",
    )(x, weight)
