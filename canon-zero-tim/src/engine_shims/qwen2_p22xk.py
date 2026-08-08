"""Additive P22.XK Qwen2 shim: promoted padded SwiGLU primal + canonical VJP."""

from __future__ import annotations

import importlib.util
import os

from p22xk_contract import preflight
from p22xk_vjp_ops import swiglu as canonical_vjp_swiglu


BASE_PATH = __import__("canon_shim_root").resolve('qwen2_p22xj.py')
spec = importlib.util.spec_from_file_location("_canon_qwen2_p22xk_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load P22.XJ qwen2 module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
_p22xk_layer_override = os.environ.pop("P16_NUM_LAYERS", None)
try:
    spec.loader.exec_module(base)
finally:
    if _p22xk_layer_override is not None:
        os.environ["P16_NUM_LAYERS"] = _p22xk_layer_override
_p22xk_qwen2_module = base
preflight(require_enabled=True)

_forward = base.P22XJ_XG_MODULE.pallas_swiglu


def traced_canonical_vjp_swiglu(
    gate,
    up,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    **kwargs,
):
    def forward(g, u):
        layer_override = os.environ.pop("P16_NUM_LAYERS", None)
        try:
            return _forward(
                g,
                u,
                interpret=interpret,
                shape_invariant_numerics=shape_invariant_numerics,
                **kwargs,
            )
        finally:
            if layer_override is not None:
                os.environ["P16_NUM_LAYERS"] = layer_override

    print(
        f"[PATHTRACE] CANON_PALLAS_CANONICAL_VJP=1 op=swiglu "
        f"M={int(gate.shape[0])} F={int(gate.shape[1])}",
        flush=True,
    )
    return canonical_vjp_swiglu(gate, up, forward=forward)


base.P22XJ_XG_MODULE.pallas_swiglu = traced_canonical_vjp_swiglu
for name, obj in vars(_p22xk_qwen2_module).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__", "base"}:
        globals()[name] = obj

P22XK_QWEN2_BASE = _p22xk_qwen2_module
P22XK_SWIGLU_FORWARD = _forward
P22XK_LAYER_OVERRIDE_BRIDGED = _p22xk_layer_override is not None
P22XK_SWIGLU_ACTIVE = (
    _p22xk_qwen2_module.P22XJ_XG_MODULE.pallas_swiglu is traced_canonical_vjp_swiglu
)
