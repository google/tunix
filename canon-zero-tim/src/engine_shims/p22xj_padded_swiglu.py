"""P22.XJ model-pinned padding around the unchanged P22.XG SwiGLU kernel."""

from __future__ import annotations

from p22_pallas_swiglu import swiglu as base_swiglu
from p22xg_contract import BF, BM
from p22xj_contract import preflight


def padded_feature_extent(feature: int) -> int:
    """Returns an admitted BF-aligned extent for one TP-local model width."""
    import p22xf_contract as model_contract

    feature = int(feature)
    if feature <= 0:
        raise ValueError(f"P22.XJ requires positive feature width, got {feature}")
    if feature % BF == 0:
        return feature

    padding = getattr(model_contract, "SWIGLU_FEATURE_PADDING", {})
    if not isinstance(padding, dict):
        raise ValueError("P22.XJ model SWIGLU_FEATURE_PADDING must be a dict")
    padded = padding.get(feature)
    if (
        not isinstance(padded, int)
        or isinstance(padded, bool)
        or padded <= feature
        or padded % BF
    ):
        raise ValueError(
            f"P22.XJ feature width F={feature} is not admitted by the "
            f"model-pinned BF={BF} padding contract: {padding!r}"
        )
    return padded


def swiglu(gate, up, *, interpret: bool = False,
           shape_invariant_numerics: bool = True):
    import jax.numpy as jnp

    preflight(require_enabled=True)
    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError(f"P22.XJ gate/up shapes differ: {gate.shape} vs {up.shape}")
    if gate.ndim != 2:
        raise ValueError(f"P22.XJ expects rank-2 inputs, got {gate.shape}")
    m, f = map(int, gate.shape)
    if m <= 0:
        raise ValueError(f"P22.XJ requires positive M, got {(m, f)}")
    fp = padded_feature_extent(f)
    if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XJ requires bf16 inputs, got {gate.dtype}, {up.dtype}")
    mp = ((m + BM - 1) // BM) * BM
    if mp == m and fp == f:
        return base_swiglu(
            gate, up, interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
        )
    pad = ((0, mp - m), (0, fp - f))
    gate_padded = jnp.pad(gate, pad, constant_values=0)
    up_padded = jnp.pad(up, pad, constant_values=0)
    out = base_swiglu(
        gate_padded, up_padded, interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )
    return out[:m, :f]
