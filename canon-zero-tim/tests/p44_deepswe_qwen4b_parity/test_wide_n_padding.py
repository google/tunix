"""P22.XI wide output tile through the Qwen3-4B contract padding (tasks/deepswe_4b_perf 1a).

Host-only: pure policy checks on wide_n_tiles, and an interpret-mode bitwise check that the
padded wide-tile matmul equals the previous 128-wide production tiles column for column.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "src" / "engine_shims"
TP4 = SHIMS / "models" / "qwen4b_tp4"
TP8 = SHIMS / "models" / "qwen4b"

_spec = importlib.util.spec_from_file_location("p22mm_wide", SHIMS / "p22_pallas_matmul.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _contract(path):
    spec = importlib.util.spec_from_file_location(f"wide_contract_{path.parent.name}", path / "p22xf_contract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses need the defining module registered
    spec.loader.exec_module(module)
    return module


def _wide():
    # wide_n_tiles lives in the P22.XI wrapper, which imports p22_pallas_matmul by name.
    sys.path.insert(0, str(SHIMS))
    try:
        sys.modules.pop("p22xi_padded_matmul", None)
        spec = importlib.util.spec_from_file_location("p22xi_wide", SHIMS / "p22xi_padded_matmul.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


CONTRACT = (128, 128, 128)


@pytest.mark.parametrize(
    "overlay,n,expected",
    [
        (TP4, 2432, (2560, 1280)),   # gate/up at TP4
        (TP4, 256, None),            # k/v: no contract padding entry
        (TP4, 2560, None),           # o/down N: policy divides it directly
        (TP4, 1024, None),           # q: policy divides it directly
        (TP8, 1216, (1280, 1280)),   # gate/up at TP8
        (TP8, 128, None),            # k/v at TP8
    ],
)
def test_wide_n_tiles_follow_the_contract_padding(overlay, n, expected):
    wide = _wide()
    mapping = _contract(overlay).MATMUL_N_PADDING
    assert wide.wide_n_tiles(256, 2560, n, CONTRACT, mapping) == expected


def test_wide_n_tiles_only_for_contract_tiles_and_bounded_padding():
    wide = _wide()
    assert wide.wide_n_tiles(256, 2560, 2432, (256, 128, 128), {2432: 2560}) is None
    assert wide.wide_n_tiles(256, 2560, 1000, CONTRACT, {1000: 1280}) is None   # 28% padding
    assert wide.wide_n_tiles(256, 2560, 2432, CONTRACT, {}) is None
    assert wide.wide_n_tiles(256, 2560, 2432, CONTRACT, {2432: 2432}) is None


_INTERPRET = textwrap.dedent(
    """
    import os, sys
    import numpy as np
    sys.path[:0] = [os.environ["WIDE_OVERLAY"], os.environ["WIDE_SHIMS"]]
    import jax, jax.numpy as jnp
    from p22_pallas_matmul import matmul as base_matmul
    from p22xi_padded_matmul import matmul as padded_matmul
    m, k, n = (int(v) for v in os.environ["WIDE_SHAPE"].split(","))
    x = jnp.asarray(((np.arange(m * k) % 5) - 2).reshape(m, k) / 64, jnp.bfloat16)
    y = jnp.asarray(((np.arange(k * n) % 7) - 3).reshape(k, n) / 128, jnp.bfloat16)
    out = padded_matmul(x, y, interpret=True, block_m=128, block_n=128, block_k=128)
    mp = -(-m // 128) * 128
    xr = jnp.pad(x, ((0, mp - m), (0, 0)))
    ref = base_matmul(xr, y, interpret=True, block_m=256 if mp % 256 == 0 else 128, block_n=128, block_k=128)[:m]
    a = np.asarray(out.astype(jnp.float32)).view(np.uint32); b = np.asarray(ref.astype(jnp.float32)).view(np.uint32)
    assert out.shape == (m, n), out.shape
    assert np.array_equal(a, b), int((a != b).sum())
    print("WIDE_BITWISE_OK", m, k, n)
    """
)


@pytest.mark.parametrize("overlay,shape", [(TP4, "256,2560,2432"), (TP4, "8,2560,2432"), (TP8, "256,2560,1216")])
def test_interpret_wide_tile_is_bitwise_identical_to_contract_tiles(overlay, shape):
    import inspect
    import jax
    if "manual_axis_type" not in inspect.signature(jax.ShapeDtypeStruct.__init__).parameters:
        pytest.skip("the P22 kernel needs the pinned image's JAX (ShapeDtypeStruct manual_axis_type); run in-image")
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "WIDE_OVERLAY": str(overlay),
        "WIDE_SHIMS": str(SHIMS),
        "WIDE_SHAPE": shape,
        "CANON_PALLAS_MPAD": "1",
        "CANON_FIXED_AR": "1",
        "CANON_FIXED_AR_EMBED": "1",
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_PALLAS_SWIGLU": "1",
        "CANON_QWEN3_TP_SIZE": "4" if overlay is TP4 else "8",
    }
    for name in ("CANON_PALLAS_MATMUL", "CANON_PALLAS_MATERIALIZE", "CANON_CUT", "CANON_TAIL", "CANON_POSTRPA_M", "P16_NUM_LAYERS"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", _INTERPRET], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "WIDE_BITWISE_OK" in result.stdout
    assert "matmul_wide_n_padding=v2" in result.stdout, result.stdout[-1500:]
