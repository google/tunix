"""Additive P22.XI shim: P22.XF projections with M-row padding compatibility."""

from __future__ import annotations

import importlib.util

from p22xi_contract import preflight
from p22xi_padded_matmul import matmul as padded_matmul


BASE_PATH = __import__("canon_shim_root").resolve('linear_p22xf.py')
spec = importlib.util.spec_from_file_location("_canon_linear_p22xi_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load P22.XF linear module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
preflight(require_enabled=True)


def traced_padded_matmul(x, y, *, interpret=False, shape_invariant_numerics=True):
    m = int(x.shape[0])
    mp = ((m + 127) // 128) * 128
    print(
        f"[PATHTRACE] CANON_PALLAS_MPAD=1 M={m} Mp={mp} padded={int(mp != m)}",
        flush=True,
    )
    return padded_matmul(
        x, y, interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )


_p22xi_xf_module = base
_p22xi_xf_module.pallas_matmul = traced_padded_matmul
for name, obj in vars(_p22xi_xf_module).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[name] = obj

# The engine stashes the live mesh on the *installed* linear module.  P22.XI
# wraps P22.XF as a nested module, so without this bridge P22.XF keeps the
# import-time ``None`` even though this outer module has the real mesh.  Sync
# immediately before every einsum call; doing it at import time is too early.
_p22xi_xf_einsum_call = _p22xi_xf_module.JaxEinsum.__call__


def _p22xi_sync_mesh():
    _p22xi_xf_module._CANON_MESH = _CANON_MESH
    return _p22xi_xf_module._CANON_MESH


def _p22xi_einsum_call(self, inputs):
    _p22xi_sync_mesh()
    return _p22xi_xf_einsum_call(self, inputs)


_p22xi_xf_module.JaxEinsum.__call__ = _p22xi_einsum_call

# Export explicit post-copy sentinels.  The P22.XF module itself exports a
# lower-level name called `base`, so checking `linear.base` is not a valid
# wiring test after the symbol copy above.
P22XI_XF_MODULE = _p22xi_xf_module
P22XI_PADDED_ACTIVE = P22XI_XF_MODULE.pallas_matmul is traced_padded_matmul
