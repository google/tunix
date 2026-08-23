"""Additive P22.XK Qwen3 shim: promoted all-RMSNorm primal + canonical VJP."""

from __future__ import annotations

import importlib.util

from p22xk_contract import preflight
from p22xk_vjp_ops import rmsnorm as canonical_vjp_rmsnorm
from xprof_labels import enabled as xprof_labels_enabled
from xprof_labels import layer_tag as xprof_layer_tag
from xprof_labels import scope as xprof_scope


BASE_PATH = __import__("canon_shim_root").resolve('qwen3_p22xh.py')
spec = importlib.util.spec_from_file_location("_canon_qwen3_p22xk_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load P22.XH qwen3 module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
_p22xk_qwen3_module = base
preflight(require_enabled=True)

if not hasattr(base, "pallas_rmsnorm"):
    raise RuntimeError("P22.XK requires the enabled P22.XH pallas_rmsnorm forward")
_forward = base.pallas_rmsnorm


def traced_canonical_vjp_rmsnorm(
    x,
    weight,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    **kwargs,
):
    def forward(a, w):
        return _forward(
            a,
            w,
            epsilon=epsilon,
            interpret=interpret,
            shape_invariant_numerics=shape_invariant_numerics,
            **kwargs,
        )

    print(
        f"[PATHTRACE] CANON_PALLAS_CANONICAL_VJP=1 op=rmsnorm "
        f"M={int(x.shape[0])} F={int(x.shape[1])}",
        flush=True,
    )
    return canonical_vjp_rmsnorm(x, weight, epsilon=epsilon, forward=forward)


base.pallas_rmsnorm = traced_canonical_vjp_rmsnorm

if xprof_labels_enabled():
    # The continue-decode loop compiles the full model into jit__decode_core.
    # These scopes survive that inlining, giving its otherwise anonymous HLO
    # operations a compact model/layer/component stack in XProf.
    _xprof_decoder_call = base.Qwen3DecoderLayer.__call__
    _xprof_attention_call = base.Qwen3Attention.__call__
    _xprof_mlp_call = base.Qwen3MLP.__call__

    def _labeled_decoder_call(self, *args, **kwargs):
        tag = xprof_layer_tag(self.self_attn.q_proj.prefix)
        with xprof_scope(tag):
            return _xprof_decoder_call(self, *args, **kwargs)

    def _labeled_attention_call(self, *args, **kwargs):
        xprof_layer_tag(self.q_proj.prefix)
        with xprof_scope("attention"):
            return _xprof_attention_call(self, *args, **kwargs)

    def _labeled_mlp_call(self, *args, **kwargs):
        xprof_layer_tag(self.gate_proj.prefix)
        with xprof_scope("mlp"):
            return _xprof_mlp_call(self, *args, **kwargs)

    base.Qwen3DecoderLayer.__call__ = _labeled_decoder_call
    base.Qwen3Attention.__call__ = _labeled_attention_call
    base.Qwen3MLP.__call__ = _labeled_mlp_call
    print(
        "[CANON_XPROF_LABELS] rollout layer/attention/mlp scopes installed",
        flush=True,
    )

for name, obj in vars(_p22xk_qwen3_module).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__", "base"}:
        globals()[name] = obj

P22XK_QWEN3_BASE = _p22xk_qwen3_module
P22XK_RMSNORM_FORWARD = _forward
P22XK_RMSNORM_ACTIVE = (
    _p22xk_qwen3_module.pallas_rmsnorm is traced_canonical_vjp_rmsnorm
)
