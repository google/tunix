"""Additive P22.XJ shim: P22.XG SwiGLU with M-row padding compatibility."""

from __future__ import annotations

import importlib.util

from p22xj_contract import preflight
from p22xj_padded_swiglu import swiglu as padded_swiglu


BASE_PATH = __import__("canon_shim_root").resolve('qwen2_p22xg.py')
spec = importlib.util.spec_from_file_location("_canon_qwen2_p22xj_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load P22.XG qwen2 module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
preflight(require_enabled=True)


def traced_padded_swiglu(gate, up, *, interpret=False,
                         shape_invariant_numerics=True):
    m = int(gate.shape[0])
    mp = ((m + 127) // 128) * 128
    print(
        f"[PATHTRACE] CANON_PALLAS_SWIGLU_MPAD=1 M={m} Mp={mp} "
        f"padded={int(mp != m)}",
        flush=True,
    )
    return padded_swiglu(
        gate, up, interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )


_p22xj_xg_module = base
_p22xj_xg_module.pallas_swiglu = traced_padded_swiglu
for name, obj in vars(_p22xj_xg_module).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[name] = obj

P22XJ_XG_MODULE = _p22xj_xg_module
P22XJ_PADDED_ACTIVE = P22XJ_XG_MODULE.pallas_swiglu is traced_padded_swiglu

