"""P22.XI model-pinned shape padding around the unchanged P22.XE matmul."""

from __future__ import annotations

from p22_pallas_matmul import BK as DEFAULT_BK
from p22_pallas_matmul import BM
from p22_pallas_matmul import BN as DEFAULT_BN
from p22_pallas_matmul import matmul as base_matmul
from p22xi_contract import preflight


def _padded_extent(width: int, tile: int, mapping, *, axis: str) -> int:
    width = int(width)
    tile = int(tile)
    if width <= 0 or tile <= 0:
        raise ValueError(
            f"P22.XI requires positive {axis}/tile, got {width}/{tile}"
        )
    if width % tile == 0:
        return width
    if not isinstance(mapping, dict):
        raise ValueError(f"P22.XI model {axis} padding contract must be a dict")
    padded = mapping.get(width)
    if (
        not isinstance(padded, int)
        or isinstance(padded, bool)
        or padded <= width
        or padded % tile
    ):
        raise ValueError(
            f"P22.XI {axis}={width} is not admitted by the model-pinned "
            f"tile={tile} padding contract: {mapping!r}"
        )
    return padded


def padded_matmul_extents(
    k: int,
    n: int,
    *,
    block_k: int = DEFAULT_BK,
    block_n: int = DEFAULT_BN,
) -> tuple[int, int]:
    """Returns admitted TP-local K/N extents for the selected model overlay."""
    import p22xf_contract as model_contract

    kp = _padded_extent(
        k,
        block_k,
        getattr(model_contract, "MATMUL_K_PADDING", {}),
        axis="K",
    )
    np = _padded_extent(
        n,
        block_n,
        getattr(model_contract, "MATMUL_N_PADDING", {}),
        axis="N",
    )
    return kp, np


def matmul(
    x,
    y,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    **kwargs,
):
    import jax.numpy as jnp

    preflight(require_enabled=True)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"P22.XI expects rank-2 inputs, got {x.shape}, {y.shape}")
    m, k = map(int, x.shape)
    ky, n = map(int, y.shape)
    if k != ky:
        raise ValueError(f"P22.XI contracted dimensions differ: {k} vs {ky}")
    if m <= 0:
        raise ValueError(f"P22.XI requires positive M, got {m}")
    block_k = int(kwargs.get("block_k", DEFAULT_BK))
    block_n = int(kwargs.get("block_n", DEFAULT_BN))
    kp, np = padded_matmul_extents(
        k, n, block_k=block_k, block_n=block_n
    )
    mp = ((m + BM - 1) // BM) * BM
    if mp == m and kp == k and np == n:
        return base_matmul(
            x,
            y,
            interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
            **kwargs,
        )
    x_padded = jnp.pad(x, ((0, mp - m), (0, kp - k)), constant_values=0)
    y_padded = jnp.pad(y, ((0, kp - k), (0, np - n)), constant_values=0)
    out = base_matmul(
        x_padded,
        y_padded,
        interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
        **kwargs,
    )
    return out[:m, :n]
