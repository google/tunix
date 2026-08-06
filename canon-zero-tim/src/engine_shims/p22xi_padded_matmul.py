"""P22.XI additive M-row compatibility wrapper around the unchanged P22.XE matmul."""

from __future__ import annotations

from p22_pallas_matmul import BM, matmul as base_matmul
from p22xi_contract import preflight


def matmul(x, y, *, interpret: bool = False, shape_invariant_numerics: bool = True):
    import jax.numpy as jnp

    preflight(require_enabled=True)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"P22.XI expects rank-2 inputs, got {x.shape}, {y.shape}")
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"P22.XI requires positive M, got {m}")
    mp = ((m + BM - 1) // BM) * BM
    if mp == m:
        return base_matmul(
            x, y, interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
        )
    padded = jnp.pad(x, ((0, mp - m), (0, 0)), constant_values=0)
    out = base_matmul(
        padded, y, interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )
    return out[:m]
