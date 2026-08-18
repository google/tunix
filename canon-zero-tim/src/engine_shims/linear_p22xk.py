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
    **kwargs,
):
    def forward(a, b):
        layer_override = os.environ.pop("P16_NUM_LAYERS", None)
        try:
            return _forward(
                a,
                b,
                interpret=interpret,
                shape_invariant_numerics=shape_invariant_numerics,
                **kwargs,
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

_p38_fixed_lm_head_value = os.environ.get("CANON_P38_FIXED_LM_HEAD", "")
if _p38_fixed_lm_head_value not in ("", "0", "1"):
    raise RuntimeError(
        "CANON_P38_FIXED_LM_HEAD must be unset, 0, or 1, got "
        f"{_p38_fixed_lm_head_value!r}"
    )

if _p38_fixed_lm_head_value == "1":
    from p38_fixed_lm_head import fixed_lm_head as _p38_fixed_lm_head
    from p38_fixed_lm_head import preflight as _p38_fixed_lm_head_preflight

    _p38_fixed_lm_head_preflight(require_enabled=True)
    _p38_original_lm_head_call = _p22xk_linear_module.JaxLmHead.__call__

    def _p38_fixed_lm_head_call(self, inputs):
        if self.einsum_str != "TD,DV->TV":
            raise RuntimeError(
                f"P38 fixed lm_head equation mismatch: {self.einsum_str!r}"
            )
        if not str(self.prefix).endswith("lm_head"):
            raise RuntimeError(
                f"P38 fixed lm_head prefix mismatch: {self.prefix!r}"
            )
        return _p38_fixed_lm_head(
            inputs,
            self.weight.value,
            mesh=_CANON_MESH,
            tp_axis=_CANON_TP_AXIS,
            local_matmul=traced_canonical_vjp_matmul,
        )

    _p22xk_linear_module.JaxLmHead.__call__ = _p38_fixed_lm_head_call
    JaxLmHead = _p22xk_linear_module.JaxLmHead

P22XK_LINEAR_BASE = _p22xk_linear_module
P22XK_MATMUL_FORWARD = _forward
P22XK_LAYER_OVERRIDE_BRIDGED = _p22xk_layer_override is not None
P22XK_MATMUL_ACTIVE = (
    _p22xk_linear_module.P22XI_XF_MODULE.pallas_matmul is traced_canonical_vjp_matmul
)
P38_FIXED_LM_HEAD_ACTIVE = _p38_fixed_lm_head_value == "1"
