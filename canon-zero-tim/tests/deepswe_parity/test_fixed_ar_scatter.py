"""tasks/deepswe_4b_perf P2b: the all_to_all fixed-order TP sum is bitwise the gather form.

The scatter form moves 2/TP of the bytes of CANON_FIXED_AR_GATHER=1 but must add
the identical operands in the identical rank order.  The numeric gate runs the
installed shim helpers under shard_map on eight forced CPU devices (TP 2/4/8,
bf16 and f32, the contract row counts 8/16/256 plus a non-divisible fallback) and
compares raw bytes rank by rank and against an unsharded sequential sum.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
SHIMS = ROOT / "canon-zero-tim" / "src" / "engine_shims"
PROFILES = ROOT / "canon-zero-tim" / "cluster" / "profiles"

NUMERIC_SCRIPT = textwrap.dedent(
    '''
    import importlib.util, os, pathlib, sys, tempfile, types
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("CANON_PALLAS_ALL_PROJ", None)
    import numpy as np, jax, jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    shims = pathlib.Path(sys.argv[1])
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "linear_patched.py").write_text(
        "import jax\\nimport jax.numpy as jnp\\n_CANON_TP_AXIS = 'model'\\n"
        "_CANON_MESH = None\\n_CANON_FIXED_SPECS = {}\\n"
        "class JaxEinsum:\\n    def __call__(self, inputs):\\n        return inputs\\n")
    root = types.ModuleType("canon_shim_root"); root.resolve = lambda name: str(tmp / name)
    sys.modules["canon_shim_root"] = root
    sys.path.insert(0, str(shims / "models" / "qwen4b_tp4"))
    spec = importlib.util.spec_from_file_location("linear_p22xf_scatter_probe", shims / "linear_p22xf.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    devices = np.array(jax.devices()); assert devices.size == 8, devices
    def mapped(fn, mesh):
        try:
            return jax.jit(shard_map(fn, mesh=mesh, in_specs=(P("model"),), out_specs=P("model"), check_vma=False))
        except TypeError:
            return jax.jit(shard_map(fn, mesh=mesh, in_specs=(P("model"),), out_specs=P("model"), check_rep=False))
    cases = 0; failures = []
    for tp in (2, 4, 8):
        mesh = Mesh(devices.reshape(8 // tp, tp), ("data", "model"))
        for dtype in (jnp.bfloat16, jnp.float32):
            for (m, n) in ((8, 2560), (16, 1280), (256, 2560), (24, 640), (12, 640)):
                key = jax.random.PRNGKey(tp * 100000 + m * 10 + n)
                parts = (jax.random.normal(key, (tp, m, n), jnp.float32) * 3.0).astype(dtype)
                gather = mapped(lambda x, tp=tp: mod._fixed_order_tp_sum_gather(x[0], "model", tp), mesh)(parts)
                scatter = mapped(lambda x, tp=tp: mod._fixed_order_tp_sum_scatter(x[0], "model", tp), mesh)(parts)
                ref = parts[0]
                for p in range(1, tp):
                    ref = ref + parts[p]
                g = np.asarray(gather).reshape(tp, m, n); s = np.asarray(scatter).reshape(tp, m, n); r = np.asarray(ref)
                for rank in range(tp):
                    cases += 1
                    if g[rank].tobytes() != r.tobytes():
                        failures.append(f"gather rank{rank} != sequential ref tp={tp} dtype={dtype.__name__} m={m} n={n}")
                    if s[rank].tobytes() != r.tobytes():
                        failures.append(f"scatter rank{rank} != sequential ref tp={tp} dtype={dtype.__name__} m={m} n={n}")
    if failures:
        print("\\n".join(failures[:20])); print(f"SCATTER_BITWISE_FAIL failures={len(failures)} cases={cases}"); sys.exit(1)
    print(f"SCATTER_BITWISE_OK cases={cases}")
    '''
)


def _load_shim_with_stub_base(tmp: Path):
  (tmp / "linear_patched.py").write_text(
      "import jax\nimport jax.numpy as jnp\n_CANON_TP_AXIS = 'model'\n"
      "_CANON_MESH = None\n_CANON_FIXED_SPECS = {}\n"
      "class JaxEinsum:\n  def __call__(self, inputs):\n    return inputs\n"
  )
  root = types.ModuleType("canon_shim_root")
  root.resolve = lambda name: str(tmp / name)
  sys.modules["canon_shim_root"] = root
  sys.path.insert(0, str(SHIMS / "models" / "qwen4b_tp4"))
  os.environ.pop("CANON_PALLAS_ALL_PROJ", None)
  spec = importlib.util.spec_from_file_location(
      "linear_p22xf_scatter_flags", SHIMS / "linear_p22xf.py"
  )
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class _Mesh:

  def __init__(self, tp):
    self.shape = {"model": tp, "data": 1}


def test_scatter_flag_is_validated_and_requires_the_gather_form(monkeypatch):
  with tempfile.TemporaryDirectory() as tmp:
    module = _load_shim_with_stub_base(Path(tmp))
    module.base._CANON_MESH = _Mesh(4)
    module.base._CANON_FIXED_SPECS = {"mn,np->mp": ((None, "model"), ("model", None))}
    site = types.SimpleNamespace(family="down_proj")
    monkeypatch.setenv("CANON_FIXED_AR_GATHER", "1")
    monkeypatch.setenv("CANON_FIXED_AR_SCATTER", "2")
    with pytest.raises(RuntimeError, match="CANON_FIXED_AR_SCATTER must be unset/0/1"):
      module._contract_parallel(site, "mn,np->mp", None, None, "layers.0")
    monkeypatch.setenv("CANON_FIXED_AR_GATHER", "0")
    monkeypatch.setenv("CANON_FIXED_AR_SCATTER", "1")
    with pytest.raises(RuntimeError, match="requires CANON_FIXED_AR_GATHER=1"):
      module._contract_parallel(site, "mn,np->mp", None, None, "layers.0")


def test_source_receipt_and_profiles_carry_the_scatter_form():
  source = (SHIMS / "linear_p22xf.py").read_text()
  assert "scatter-ordered-sum" in source
  assert "all_to_all(" in source and "tiled=True" in source
  for name in ("qwen3-4b-dp1-tp4-deepswe-zero.env", "qwen3-4b-dp8-tp8-deepswe-v1-hp.env"):
    profile = (PROFILES / name).read_text()
    assert "export CANON_FIXED_AR_GATHER=1\n" in profile
    assert "export CANON_FIXED_AR_SCATTER=1\n" in profile
  assert "  CANON_FIXED_AR_SCATTER \\\n" in (PROFILES / "qwen3-4b-dp8-tp8-deepswe-v1-hp.env").read_text()
  flags = (ROOT / "canon-zero-tim" / "FLAGS.md").read_text()
  assert "| `CANON_FIXED_AR_SCATTER` |" in flags and "\nCANON_FIXED_AR_SCATTER\n" in flags


def test_scatter_form_is_bitwise_the_gather_form_on_eight_cpu_devices():
  env = {**os.environ, "JAX_PLATFORMS": "cpu"}
  env.pop("XLA_FLAGS", None)
  result = subprocess.run(
      [sys.executable, "-c", NUMERIC_SCRIPT, str(SHIMS)],
      env=env, capture_output=True, text=True, timeout=900,
  )
  assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
  assert "SCATTER_BITWISE_OK" in result.stdout, result.stdout[-2000:]
