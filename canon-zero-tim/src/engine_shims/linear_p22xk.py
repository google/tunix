"""Additive P22.XK linear shim: promoted Pallas matmul primal + canonical VJP."""

from __future__ import annotations

import importlib.util
import os

from p22xk_contract import preflight
from p22xk_vjp_ops import matmul as canonical_vjp_matmul


BASE_PATH = __import__("canon_shim_root").resolve('linear_p22xi.py')
spec = importlib.util.spec_from_file_location("_canon_linear_p22xk_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load P22.XI linear module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
_p22xk_layer_override = os.environ.pop("P16_NUM_LAYERS", None)
try:
    spec.loader.exec_module(base)
finally:
    if _p22xk_layer_override is not None:
        os.environ["P16_NUM_LAYERS"] = _p22xk_layer_override
_p22xk_linear_module = base
preflight(require_enabled=True)
if _p22xk_layer_override is not None:
    print(
        f"P22.XK LAYER-OVERRIDE IMPORT BRIDGE PASS value={_p22xk_layer_override}",
        flush=True,
    )

_forward = base.P22XI_XF_MODULE.pallas_matmul


def traced_canonical_vjp_matmul(
    x,
    y,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
):
    def forward(a, b):
        layer_override = os.environ.pop("P16_NUM_LAYERS", None)
        try:
            return _forward(
                a,
                b,
                interpret=interpret,
                shape_invariant_numerics=shape_invariant_numerics,
            )
        finally:
            if layer_override is not None:
                os.environ["P16_NUM_LAYERS"] = layer_override

    print(
        f"[PATHTRACE] CANON_PALLAS_CANONICAL_VJP=1 op=matmul "
        f"M={int(x.shape[0])} K={int(x.shape[1])} N={int(y.shape[1])}",
        flush=True,
    )
    return canonical_vjp_matmul(x, y, forward=forward)


base.P22XI_XF_MODULE.pallas_matmul = traced_canonical_vjp_matmul

for name, obj in vars(_p22xk_linear_module).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__", "base"}:
        globals()[name] = obj

# P22.XI is itself a nested module.  The engine writes the live mesh onto the
# installed (outermost) linear module, so bridge that state before the inherited
# JaxEinsum wrapper forwards it to P22.XF.
_p22xk_einsum_call = _p22xk_linear_module.JaxEinsum.__call__


def _p22xk_einsum_with_mesh(self, inputs):
    _p22xk_linear_module._CANON_MESH = _CANON_MESH
    _p22xk_linear_module._CANON_TP_AXIS = _CANON_TP_AXIS
    return _p22xk_einsum_call(self, inputs)


_p22xk_linear_module.JaxEinsum.__call__ = _p22xk_einsum_with_mesh
JaxEinsum = _p22xk_linear_module.JaxEinsum

P22XK_LINEAR_BASE = _p22xk_linear_module
P22XK_MATMUL_FORWARD = _forward
P22XK_LAYER_OVERRIDE_BRIDGED = _p22xk_layer_override is not None
P22XK_MATMUL_ACTIVE = (
    _p22xk_linear_module.P22XI_XF_MODULE.pallas_matmul is traced_canonical_vjp_matmul
)
