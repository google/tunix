"""Row-tile policy of the canonical rmsnorm kernel (tasks/zero_tim_perf3 A).

Pure-host checks: the policy only widens the row block (the parallel axis) to
256 for shapes inside the probed envelope, never changes BF (the reduction
block), keeps the contract BM elsewhere, and is fully disabled by v1.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "src" / "engine_shims"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(SHIMS))  # the module imports its sibling contract modules by name
_spec = importlib.util.spec_from_file_location("p22rms", SHIMS / "p22_pallas_rmsnorm.py")
rms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rms)


@pytest.fixture(autouse=True)
def _v2(monkeypatch):
    monkeypatch.delenv(rms.ROW_TILES_ENV, raising=False)


@pytest.mark.parametrize(
    "rows,features,expected",
    [
        (8, 128, 8),          # decode x-norm rows below the wide block: contract BM
        (128, 4096, 8),       # decode x-norm (128 tokens): not a multiple of 256
        (512, 128, 256),      # decode k-norm
        (2048, 128, 256),     # decode q-norm / prefill k-norm
        (8192, 128, 256),     # prefill q-norm
        (512, 4096, 256),     # prefill x-norm
        (256, 4096, 256),     # trainer x-norm
        (256, 8192, 8),       # feature width outside the probed envelope
        (264, 128, 8),        # multiple of 8 but not of 256
    ],
)
def test_row_tile_policy(rows, features, expected):
    got = rms.row_tile(rows, features)
    assert got == expected
    assert rows % got == 0
    assert got % rms.BM == 0


def test_v1_restores_the_contract_block(monkeypatch):
    monkeypatch.setenv(rms.ROW_TILES_ENV, "v1")
    assert rms.row_tile(8192, 128) == rms.BM
    assert rms.row_tile(512, 4096) == rms.BM


def test_reduction_block_is_untouched():
    assert rms.BF == 128
    assert rms.ROW_TILE_WIDE % rms.BM == 0


def test_contract_validation_still_requires_bm_multiples():
    with pytest.raises(ValueError):
        rms.validate_shape((12, 128), (128,))
    assert rms.validate_shape((8192, 128), (128,)) == (8192, 128)
