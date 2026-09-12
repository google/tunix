"""Shape policy of the canonical pallas matmul tiles (tasks/zero_tim_perf2 B2 knife 2).

Pure-host checks: the policy only ever widens block_m/block_n, keeps the
caller's block_k (the per-element accumulation order), falls back to the
caller's tiles when nothing wider divides, and fires for every fixed tile
triple a model contract passes.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "src" / "engine_shims"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_spec = importlib.util.spec_from_file_location("p22mm", SHIMS / "p22_pallas_matmul.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

TP2 = (128, 256, 256)
TP8 = (128, 128, 128)


@pytest.mark.parametrize(
    "shape,tiles,expected",
    [
        # tp2 / tp1 / 1.7B contracts (128, 256, 256): unchanged from ec083acb.
        ((512, 4096, 6144), TP2, (256, 1024, 256)),
        ((256, 6144, 4096), TP2, (256, 1024, 256)),
        ((128, 4096, 6144), TP2, (128, 1024, 256)),
        ((128, 4096, 512), TP2, (128, 512, 256)),
        ((512, 4096, 2048), TP2, (256, 1024, 256)),
        ((128, 4096, 768), TP2, (128, 256, 256)),
        # tp8 contract (128, 128, 128): block_k stays 128; k/v (N=128) keep 128.
        ((512, 4096, 1536), TP8, (256, 512, 128)),
        ((128, 4096, 1536), TP8, (128, 512, 128)),
        ((128, 1536, 4096), TP8, (128, 1024, 128)),
        ((512, 512, 4096), TP8, (256, 1024, 128)),
        ((128, 4096, 512), TP8, (128, 512, 128)),
        ((128, 4096, 128), TP8, (128, 128, 128)),
        ((512, 4096, 128), TP8, (256, 128, 128)),
    ],
)
def test_policy_widens_m_and_n_only(shape, tiles, expected):
    m, k, n = shape
    got = mm.tile_policy(m, k, n, tiles)
    assert got == expected
    assert got[2] == tiles[2]
    assert m % got[0] == 0 and n % got[1] == 0 and k % got[2] == 0


def test_default_tiles_are_the_tp2_contract():
    assert mm.tile_policy(512, 4096, 6144) == mm.tile_policy(512, 4096, 6144, TP2)


def test_every_contract_fixed_tiles_trigger_the_policy():
    contracts = sorted((SHIMS / "models").glob("*/p22xf_contract.py"))
    assert contracts, "no model contracts found"
    for path in contracts:
        text = path.read_text()
        tiles = tuple(
            int(re.search(rf"^{name} = (\d+)$", text, re.M).group(1))
            for name in ("BM", "BN", "BK")
        )
        assert tiles in mm.CONTRACT_TILES, f"{path.parent.name} passes {tiles}"
