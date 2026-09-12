"""CANON_MATMUL_VJP_PLAIN: plain-dot projection VJPs match the K-block replica pullback (E2d)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CANON_PALLAS_ALL_PROJ", "1")
os.environ.setdefault("CANON_FIXED_AR", "1")
for name, val in (("CANON_QWEN3_HIDDEN_SIZE", "4096"), ("CANON_QWEN3_INTERMEDIATE_SIZE", "12288"),
                  ("CANON_QWEN3_NUM_ATTENTION_HEADS", "32"), ("CANON_QWEN3_NUM_KV_HEADS", "8"),
                  ("CANON_QWEN3_HEAD_DIM", "128"), ("CANON_QWEN3_TP_SIZE", "2")):
  os.environ.setdefault(name, val)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")  # the checked-shard_map test
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
SHIMS = ROOT / "src" / "engine_shims"
sys.path.insert(0, str(SHIMS / "models" / "qwen8b_tp2"))
sys.path.insert(0, str(SHIMS))
_spec = importlib.util.spec_from_file_location("p22xk_vjp_ops", SHIMS / "p22xk_vjp_ops.py")
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)


@pytest.mark.parametrize("m,k,n", [(256, 512, 256), (128, 1024, 512), (64, 256, 768)])
def test_plain_pullback_matches_replica(m, k, n):
  key = jax.random.PRNGKey(m + k + n)
  a = (jax.random.normal(key, (m, k)) * 0.5).astype(jnp.bfloat16)
  b = (jax.random.normal(jax.random.PRNGKey(1), (k, n)) * 0.05).astype(jnp.bfloat16)
  cot = (jax.random.normal(jax.random.PRNGKey(2), (m, n)) * 0.1).astype(jnp.bfloat16)
  _, pullback = jax.vjp(ops.canonical_matmul, a, b)
  da_ref, db_ref = pullback(cot)
  da, db = ops.plain_matmul_pullback(a, b, cot)
  assert da.dtype == a.dtype and db.dtype == b.dtype and da.shape == a.shape and db.shape == b.shape
  np.testing.assert_allclose(np.asarray(da, np.float32), np.asarray(da_ref, np.float32), rtol=2e-2, atol=2e-3)
  np.testing.assert_allclose(np.asarray(db, np.float32), np.asarray(db_ref, np.float32), rtol=2e-2, atol=2e-3)
  # bf16 outputs: the two orders agree to well within bf16 resolution on the norm
  for x, y in ((da, da_ref), (db, db_ref)):
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    assert np.linalg.norm(x - y) / np.linalg.norm(y) < 5e-3


def test_switch(monkeypatch):
  monkeypatch.delenv(ops.PLAIN_VJP_ENV, raising=False)
  assert ops.plain_matmul_vjp_enabled()  # Phase H: the plain pullback is the default
  monkeypatch.setenv(ops.PLAIN_VJP_ENV, "0")
  assert not ops.plain_matmul_vjp_enabled()  # explicit A/B path: the K-block replica pullback
  monkeypatch.setenv(ops.PLAIN_VJP_ENV, "1")
  assert ops.plain_matmul_vjp_enabled()
  monkeypatch.setenv(ops.PLAIN_VJP_ENV, "2")
  with pytest.raises(ValueError):
    ops.plain_matmul_vjp_enabled()


def test_plain_pullback_keeps_the_tp_psum_inside_a_checked_shard_map(monkeypatch):
  """Column-parallel projection under check_vma=True: dX must come back invariant over model with the TP sum applied."""
  from jax.sharding import Mesh, PartitionSpec as P
  devices = jax.devices()
  if len(devices) < 2:
    pytest.skip("needs --xla_force_host_platform_device_count=2")
  mesh = Mesh(np.array(devices[:2]).reshape(1, 2), axis_names=("data", "model"))
  m, k, n = 64, 256, 512
  a = (jax.random.normal(jax.random.PRNGKey(3), (m, k)) * 0.5).astype(jnp.bfloat16)
  b = (jax.random.normal(jax.random.PRNGKey(4), (k, n)) * 0.05).astype(jnp.bfloat16)
  cot = (jax.random.normal(jax.random.PRNGKey(5), (m, n)) * 0.1).astype(jnp.bfloat16)
  # the coat's own custom_vjp shape (primal = the Pallas forward, bwd = the plain pullback) without its preflight
  @jax.custom_vjp
  def coat(x, y):
    return jnp.dot(x, y, preferred_element_type=jnp.bfloat16)

  coat.defvjp(lambda x, y: (coat(x, y), (x, y)), lambda res, ct: ops.plain_matmul_pullback(res[0], res[1], ct))

  def local(a_local, b_local, cot_local):
    # the activation is replicated over model, the weight shard and cotangent vary over it
    _, pullback = jax.vjp(coat, a_local, b_local)
    da, db = pullback(cot_local)
    return da, db

  mapped = jax.shard_map(local, mesh=mesh, in_specs=(P(None, None), P(None, "model"), P(None, "model")),
                         out_specs=(P(None, None), P(None, "model")), check_vma=True)
  da, db = mapped(a, b, cot)
  da_ref = jnp.dot(cot, b.T, preferred_element_type=jnp.float32)
  db_ref = jnp.dot(a.T, cot, preferred_element_type=jnp.float32)
  np.testing.assert_allclose(np.asarray(da, np.float32), np.asarray(da_ref), rtol=2e-2, atol=2e-3)
  np.testing.assert_allclose(np.asarray(db, np.float32), np.asarray(db_ref), rtol=2e-2, atol=2e-3)
