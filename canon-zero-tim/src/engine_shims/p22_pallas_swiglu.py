#!/usr/bin/env python3
"""Fixed-tile bf16 SwiGLU custom call used only by additive P22.XG."""

from __future__ import annotations

from p22xg_contract import BF, BM, validate_shape


def _imports():
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    return jax, jnp, pl, pltpu


def swiglu(gate, up, *, interpret: bool = False,
           shape_invariant_numerics: bool = True):
    """Return `silu(gate) * up` for TP-local bf16 rank-2 arrays."""
    jax, jnp, pl, pltpu = _imports()
    m, f = validate_shape(gate.shape, up.shape)
    if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XG requires bf16 inputs, got {gate.dtype}, {up.dtype}")

    def _kernel(g_ref, u_ref, out_ref):
        g = g_ref[...]
        u = u_ref[...]
        out_ref[...] = (jax.nn.silu(g) * u).astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, f), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
                pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
            grid=(m // BM, f // BF),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            allow_input_fusion=(False, False),
            shape_invariant_numerics=shape_invariant_numerics,
        ),
        interpret=interpret,
        name="canon_swiglu_bm128_bf256",
    )(gate, up)

