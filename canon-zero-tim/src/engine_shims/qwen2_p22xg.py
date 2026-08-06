"""Additive/default-off Qwen2 MLP wrapper for P22.XG fixed-tile SwiGLU."""

from __future__ import annotations

import importlib.util
import os

from p22xg_contract import preflight


BASE_PATH = __import__("canon_shim_root").resolve('qwen2_patched.py')
value = os.environ.get("CANON_PALLAS_SWIGLU", "")
if value not in ("", "1"):
    raise RuntimeError(f"CANON_PALLAS_SWIGLU must be unset or 1, got {value!r}")

spec = importlib.util.spec_from_file_location("_canon_qwen2_p22xg_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load qwen2 base from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_mlp_call = base.Qwen2MLP.__call__


if value == "1":
    from p22_pallas_swiglu import swiglu as pallas_swiglu
    preflight(require_enabled=True)


def _p22xg_mlp_call(self, x):
    if value != "1":
        return original_mlp_call(self, x)
    expected_act = base.modeling_flax_utils.ACT2FN["silu"]
    if self.act_fn is not expected_act:
        raise RuntimeError(
            f"P22.XG supports exact silu only, got {getattr(self.act_fn, '__name__', self.act_fn)!r}"
        )
    gate = self.gate_proj(x)
    up = self.up_proj(x)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    import tpu_inference.layers.jax.linear as linear_module

    mesh = linear_module._CANON_MESH
    axis = linear_module._CANON_TP_AXIS
    if mesh is None:
        raise RuntimeError("P22.XG canonical mesh is unset")

    def local(g_local, u_local):
        out = pallas_swiglu(g_local, u_local)
        print(
            f"[PATHTRACE] CANON_PALLAS_SWIGLU=1 M={g_local.shape[0]} "
            f"Flocal={g_local.shape[1]}", flush=True
        )
        return out

    try:
        mapped = shard_map(local, mesh=mesh,
                           in_specs=(P(None, axis), P(None, axis)),
                           out_specs=P(None, axis), check_vma=False)
    except TypeError:
        mapped = shard_map(local, mesh=mesh,
                           in_specs=(P(None, axis), P(None, axis)),
                           out_specs=P(None, axis), check_rep=False)
    fused = mapped(gate, up)
    return self.down_proj(fused)


base.Qwen2MLP.__call__ = _p22xg_mlp_call

for name, obj in vars(base).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[name] = obj

