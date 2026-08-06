"""P22.XJ row-padding wrapper around the unchanged P22.XG SwiGLU kernel."""

from __future__ import annotations

from p22_pallas_swiglu import swiglu as base_swiglu
from p22xg_contract import BF, BM
from p22xj_contract import preflight


def swiglu(gate, up, *, interpret: bool = False,
           shape_invariant_numerics: bool = True):
    import jax.numpy as jnp

    preflight(require_enabled=True)
    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError(f"P22.XJ gate/up shapes differ: {gate.shape} vs {up.shape}")
    if gate.ndim != 2:
        raise ValueError(f"P22.XJ expects rank-2 inputs, got {gate.shape}")
    m, f = map(int, gate.shape)
    if m <= 0 or f % BF:
        raise ValueError(f"P22.XJ requires positive M and F%{BF}=0, got {(m, f)}")
    if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XJ requires bf16 inputs, got {gate.dtype}, {up.dtype}")
    mp = ((m + BM - 1) // BM) * BM
    if mp == m:
        return base_swiglu(
            gate, up, interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
        )
    pad = ((0, mp - m), (0, 0))
    gate_padded = jnp.pad(gate, pad, constant_values=0)
    up_padded = jnp.pad(up, pad, constant_values=0)
    out = base_swiglu(
        gate_padded, up_padded, interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )
    return out[:m]

