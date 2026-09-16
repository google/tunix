"""P22.XI model-pinned shape padding around the unchanged P22.XE matmul."""

from __future__ import annotations

from p22_pallas_matmul import BK as DEFAULT_BK
from p22_pallas_matmul import BLOCK_N_CANDIDATES
from p22_pallas_matmul import BM
from p22_pallas_matmul import CONTRACT_TILES
from p22_pallas_matmul import BN as DEFAULT_BN
from p22_pallas_matmul import matmul as base_matmul
from p22xi_contract import preflight


# tasks/deepswe_4b_perf phase1 1a.  A TP-local width that no policy candidate
# divides (Qwen3-4B MLP: 2432 at TP4, 1216 at TP8) used to run at the
# contract's 128-wide output tile.  When the model contract already admits a
# zero-padded width for it, that padded width can carry a wide tile: padding
# columns are sliced off and every real column's contraction order is
# unchanged, so the result is bitwise identical (k3b_4b_bench: gate/up
# 98.1 -> 37.7 us at TP4, 57.1 -> 28.2 us at TP8, M=256).  Only contract
# tiles trigger this; explicit tiles are honored as written.
WIDE_N_CANDIDATES = (1280, 640, 512)
WIDE_N_MAX_PADDING_NUM, WIDE_N_MAX_PADDING_DEN = 11, 10  # np <= 1.1 * n
_WIDE_N_RECEIPTS: set[tuple[int, int, int]] = set()


def wide_n_tiles(m: int, k: int, n: int, tiles, mapping):
    """Return (np, block_n) when a contract-padded width admits a wide tile, else None."""
    block_m, block_n, block_k = tiles
    if (block_m, block_n, block_k) not in CONTRACT_TILES:
        return None
    if any(n % candidate == 0 for candidate in BLOCK_N_CANDIDATES):
        return None
    if not isinstance(mapping, dict):
        return None
    padded = mapping.get(int(n))
    if not isinstance(padded, int) or isinstance(padded, bool) or padded <= n:
        return None
    if padded * WIDE_N_MAX_PADDING_DEN > n * WIDE_N_MAX_PADDING_NUM:
        return None
    for candidate in WIDE_N_CANDIDATES:
        if padded % candidate == 0:
            return padded, candidate
    return None


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
    import p22xf_contract as model_contract

    wide = wide_n_tiles(
        mp, kp, n, (int(kwargs.get("block_m", BM)), block_n, block_k),
        getattr(model_contract, "MATMUL_N_PADDING", {}),
    )
    if wide is not None:
        np, wide_bn = wide
        wide_bm = 256 if mp % 256 == 0 else int(kwargs.get("block_m", BM))
        key = (m, k, n)
        if key not in _WIDE_N_RECEIPTS:
            _WIDE_N_RECEIPTS.add(key)
            print(
                f"[PATHTRACE] matmul_wide_n_padding=v2 M={m} K={k} N={n}->{np} "
                f"bm={wide_bm} bn={wide_bn} bk={block_k}",
                flush=True,
            )
        x_padded = jnp.pad(x, ((0, mp - m), (0, kp - k)), constant_values=0)
        y_padded = jnp.pad(y, ((0, kp - k), (0, np - n)), constant_values=0)
        wide_kwargs = {**kwargs, "block_m": wide_bm, "block_n": wide_bn, "block_k": block_k}
        out = base_matmul(
            x_padded,
            y_padded,
            interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
            **wide_kwargs,
        )
        return out[:m, :n]
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
