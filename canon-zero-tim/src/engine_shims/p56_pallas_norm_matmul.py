#!/usr/bin/env python3
"""Fused P22.XH rmsnorm prologue + P22.XE matmul in one custom call.

P56.4.6: the decode layer computes rmsnorm(x) only to feed the very next
projection matmul, paying a [M, F] bf16 round-trip through HBM plus one
extra kernel launch per site.  This kernel normalizes the row block once
into VMEM scratch with the verbatim P22.XH arithmetic (f32 promotion,
BF-blocked left-to-right sumsq, rsqrt(mean + eps), f32 scale by gamma,
bf16 cast) and then runs the verbatim P22.XE tile loop (bf16 tiles, f32
accumulator, same BK order) against that scratch.  Every per-element
chain is the one the two-kernel chain executes, so outputs are bitwise
equal; only the intermediate's residency changes (VMEM scratch instead
of an HBM tensor).
"""

from __future__ import annotations

import os

from p22xh_contract import BF


def _imports():
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    return jax, jnp, pl, pltpu


def norm_matmul(
    x,
    gamma,
    y,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 256,
):
    """Compute p22 matmul(p22 rmsnorm(x, gamma), y) as one custom call."""
    jax, jnp, pl, pltpu = _imports()
    if x.ndim != 2 or y.ndim != 2 or gamma.ndim != 1:
        raise ValueError(
            f"P56.4.6 expects x[M,K] gamma[K] y[K,N], got "
            f"{x.shape}, {gamma.shape}, {y.shape}"
        )
    m, k = map(int, x.shape)
    ky, n = map(int, y.shape)
    if k != ky or int(gamma.shape[0]) != k:
        raise ValueError(
            f"P56.4.6 contracted dims differ: x{x.shape} gamma{gamma.shape} "
            f"y{y.shape}"
        )
    if (
        x.dtype != jnp.bfloat16
        or y.dtype != jnp.bfloat16
        or gamma.dtype != jnp.bfloat16
    ):
        raise ValueError(
            f"P56.4.6 requires bf16 inputs, got {x.dtype}/{gamma.dtype}/"
            f"{y.dtype}"
        )
    if not float(epsilon) > 0.0:
        raise ValueError(f"P56.4.6 epsilon must be positive, got {epsilon}")
    if m % block_m or n % block_n or k % block_k or k % BF:
        raise ValueError(
            "P56.4.6 shape must divide BM/BN/BK/BF="
            f"{block_m}/{block_n}/{block_k}/{BF}, got {(m, k, n)}"
        )

    def _kernel(x_ref, gamma_ref, y_ref, out_ref, acc_ref, normed_ref):
        # P4.6b: the n dimension lives INSIDE the kernel as a static tile
        # loop, so the row block and its normalized scratch stream from
        # HBM once per (i, q) instead of once per (i, j, q).  Each
        # n-tile's dot keeps the exact (block_m, block_k) x (block_k,
        # block_n) shape and the same sequential q order, so every
        # output element's accumulation chain is unchanged.
        q = pl.program_id(1)

        @pl.when(q == 0)
        def _norm():
            # Verbatim P22.XH: f32 promote, BF-blocked left-to-right sumsq
            # (python-static expansion), rsqrt(mean + eps), f32 gamma
            # scale, bf16 cast.  Row-independent, so the 128-row batch
            # changes nothing per row.
            xf = x_ref[...].astype(jnp.float32).reshape(block_m, k // BF, BF)
            sumsq = jnp.zeros((block_m,), dtype=jnp.float32)
            for b in range(k // BF):
                block = xf[:, b, :]
                sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
            inv = jax.lax.rsqrt(sumsq / jnp.float32(k) + jnp.float32(epsilon))
            normed_ref[...] = (
                x_ref[...].astype(jnp.float32)
                * inv[:, None]
                * gamma_ref[...].astype(jnp.float32)[None, :]
            ).astype(jnp.bfloat16)

        @pl.when(q == 0)
        def _init():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # Verbatim P22.XE tile step on the scratch rows: bf16 tiles, f32
        # accumulator, the same sequential BK order; n tiles unrolled
        # statically with the identical per-tile dot shape.
        xk = normed_ref[:, pl.ds(q * block_k, block_k)]
        for j in range(n // block_n):
            lo = j * block_n
            acc_ref[:, lo:lo + block_n] = acc_ref[:, lo:lo + block_n] + jnp.dot(
                xk,
                y_ref[:, lo:lo + block_n],
                preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, k), lambda i, _q: (i, 0)),
                pl.BlockSpec((k,), lambda _i, _q: (0,)),
                pl.BlockSpec((block_k, n), lambda _i, q: (q, 0)),
            ],
            out_specs=pl.BlockSpec((block_m, n), lambda i, _q: (i, 0)),
            grid=(m // block_m, k // block_k),
            scratch_shapes=[
                pltpu.VMEM((block_m, n), jnp.float32),
                pltpu.VMEM((block_m, k), jnp.bfloat16),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            # q must run in grid order: the row block's normalize happens
            # at its first visit and later steps read the scratch.
            dimension_semantics=("parallel", "arbitrary"),
            allow_input_fusion=(False, False, False),
            shape_invariant_numerics=shape_invariant_numerics,
        ),
        interpret=interpret,
        name=(
            f"canon_norm_matmul_bm{block_m}_bn{block_n}_bk{block_k}_f{k}"
        ),
    )(x, gamma, y)


def continue_decode_norm_matmul(
    x,
    gamma,
    y,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 256,
):
    """Use the certified separate coats for continue-decode request buckets."""
    value = os.environ.get("CANON_CONTINUE_DECODE", "")
    if not value or not value.isdigit() or not 1 <= int(value) <= 64:
        raise ValueError(
            "P57 small-M compatibility requires CANON_CONTINUE_DECODE in "
            f"[1, 64], got {value!r}"
        )
    if x.ndim != 2 or gamma.ndim != 1 or y.ndim != 2:
        raise ValueError(
            f"P57 small-M compatibility expects x[M,K] gamma[K] y[K,N], "
            f"got {x.shape}, {gamma.shape}, {y.shape}"
        )
    m = int(x.shape[0])
    if m <= 0 or m >= block_m or m % 8:
        raise ValueError(
            f"P57 small-M compatibility requires 0 < M < {block_m} and "
            f"M divisible by 8, got {m}"
        )

    # The fused 128-row primal is not shape invariant at every production
    # width (the CPU gate caught a d_weight byte delta at M32/K2048/N512).
    # Keep the continue-decode loop, but for its 8/16/32 request buckets run
    # the exact already-certified coats: fixed-BF rmsnorm on the real rows,
    # then the XI padded matmul.  This preserves both the primal operations
    # and the custom-VJP boundary/cotangent casts byte for byte.
    from p22_pallas_rmsnorm import rmsnorm as pallas_rmsnorm
    from p22xi_padded_matmul import matmul as padded_matmul
    from p22xk_vjp_ops import matmul as coated_matmul
    from p22xk_vjp_ops import rmsnorm as coated_rmsnorm

    normalized = coated_rmsnorm(
        x,
        gamma,
        epsilon=epsilon,
        forward=lambda a, g: pallas_rmsnorm(
            a,
            g,
            epsilon=epsilon,
            interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
        ),
    )
    result = coated_matmul(
        normalized,
        y,
        forward=lambda a, b: padded_matmul(
            a,
            b,
            interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        ),
    )
    print(
        f"[PATHTRACE] CANON_CONTINUE_DECODE norm_matmul separate-coat "
        f"fallback M={m} Mp={block_m}",
        flush=True,
    )
    return result
