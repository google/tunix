"""The page-blockwise chunked RPA backward matches autodiff of the replica (tasks/zero_tim_perf3 E2c).

Host CPU, f32.  Reference = jax.vjp through make_diff_rpa_chunked(...)._replica on the
same operands; the blockwise path loops over the dynamic number of prefix pages.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "src" / "engine_shims"
sys.path.insert(0, str(SHIMS))
_spec = importlib.util.spec_from_file_location("rpa_diff_chunked", SHIMS / "rpa_diff_chunked.py")
rdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdc)

NQ, NKV, HD, PAGE, NP, T = 8, 4, 32, 16, 12, 32  # nkv != 2 disambiguates the [np, PAGE, nkv, 2, hd] layout
SM = 1.0 / np.sqrt(HD)


def _case(seed, ctx, q_len, n_tbl):
  rng = np.random.default_rng(seed)
  mk = lambda *s: jnp.asarray(rng.normal(size=s).astype(np.float32) * 0.3)
  q, k, v = mk(T, NQ, HD), mk(T, NKV, HD), mk(T, NKV, HD)
  cache = mk(NP, PAGE, NKV, 2, HD)
  tbl = jnp.asarray(rng.permutation(NP)[:n_tbl].astype(np.int32))
  g_out = mk(T, NQ, HD)
  kv_len = jnp.int32(ctx + q_len)
  return q, k, v, cache, kv_len, jnp.int32(q_len), tbl, g_out


@pytest.mark.parametrize("ctx,q_len,n_tbl", [(40, 32, 6), (0, 32, 6), (48, 20, 6), (79, 32, 8), (16, 5, 4)])
def test_blockwise_matches_replica_autodiff(ctx, q_len, n_tbl):
  d = rdc.make_diff_rpa_chunked(lambda *a: None, sm_scale=SM, page_size=PAGE, num_q_heads=NQ, num_kv_heads=NKV)
  q, k, v, cache, kv_len, q_len_a, tbl, g_out = _case(ctx * 7 + q_len, ctx, q_len, n_tbl)
  row_ok = jnp.arange(T) < q_len
  g_out = jnp.where(row_ok[:, None, None], g_out, 0.0)

  def f(q_, k_, v_, c_):
    return d._replica(q_, k_, v_, c_, kv_len, q_len_a, tbl)

  _, vjp = jax.vjp(f, q, k, v, cache)
  ref = vjp(g_out)
  got = jax.jit(d._blockwise_vjp)(q, k, v, cache, kv_len, q_len_a, tbl, g_out)
  for name, a, b in zip(("dq", "dk", "dv", "dcache"), got, ref):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, name
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5, err_msg=name)
  # invalid query rows carry no gradient
  assert np.all(np.asarray(got[0])[q_len:] == 0)


def test_switch_default_off(monkeypatch):
  monkeypatch.delenv(rdc.BLOCKWISE_VJP_ENV, raising=False)
  assert not rdc.blockwise_vjp_enabled()
  monkeypatch.setenv(rdc.BLOCKWISE_VJP_ENV, "1")
  assert rdc.blockwise_vjp_enabled()
  monkeypatch.setenv(rdc.BLOCKWISE_VJP_ENV, "x")
  with pytest.raises(ValueError):
    rdc.blockwise_vjp_enabled()
