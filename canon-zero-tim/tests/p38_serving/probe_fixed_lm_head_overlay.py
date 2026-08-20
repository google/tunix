#!/usr/bin/env python3
"""Attest an installed Qwen3 TP4 fixed-lm-head hook in the pinned image."""

from __future__ import annotations

import argparse
import os
import types


parser = argparse.ArgumentParser()
parser.add_argument("--hidden-size", type=int, choices=(2048, 4096), default=4096)
args = parser.parse_args()


for name in (
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_P38_FIXED_LM_HEAD",
):
  os.environ[name] = "1"

model_env = {
    2048: (6144, 16, "qwen1p7b"),
    4096: (12288, 32, "qwen8b"),
}
intermediate_size, attention_heads, model_name = model_env[args.hidden_size]
os.environ.update({
    "CANON_QWEN3_HIDDEN_SIZE": str(args.hidden_size),
    "CANON_QWEN3_INTERMEDIATE_SIZE": str(intermediate_size),
    "CANON_QWEN3_NUM_ATTENTION_HEADS": str(attention_heads),
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": "4",
})

import linear_p22xk as linear  # noqa: E402
import p22xf_contract as model  # noqa: E402
import p38_fixed_lm_head as fixed  # noqa: E402
from tpu_inference.layers.jax import embed as embed_module  # noqa: E402


if not linear.P22XK_MATMUL_ACTIVE:
  raise AssertionError("installed P22.XK matmul wrapper is inactive")
if not linear.P38_FIXED_LM_HEAD_ACTIVE:
  raise AssertionError("installed P38 fixed lm_head wrapper is inactive")
if linear.JaxLmHead.__call__.__name__ != "_p38_fixed_lm_head_call":
  raise AssertionError("JaxLmHead does not point at the P38 hook")
if not linear.P38_FIXED_TIED_HEAD_ACTIVE:
  raise AssertionError("installed P38 tied-embedding head wrapper is inactive")
if embed_module.JaxEmbed.decode.__name__ != "_p38_fixed_tied_head_decode":
  raise AssertionError("JaxEmbed.decode does not point at the P38 tied hook")


class _FakeWeight:
  @property
  def T(self):
    return "transposed-embed-weight"


captured = {}


def _fake_fixed(inputs, weight, **kwargs):
  captured.update(inputs=inputs, weight=weight, **kwargs)
  return "fixed-tied-output"


linear._p38_fixed_lm_head = _fake_fixed
linear._CANON_MESH = "test-mesh"
linear._CANON_TP_AXIS = "model"
fake_embed = types.SimpleNamespace(
    weight=types.SimpleNamespace(value=_FakeWeight())
)
fake_inputs = object()
if embed_module.JaxEmbed.decode(fake_embed, fake_inputs) != "fixed-tied-output":
  raise AssertionError("P38 tied hook did not return the fixed-head result")
if captured != {
    "inputs": fake_inputs,
    "weight": "transposed-embed-weight",
    "mesh": "test-mesh",
    "tp_axis": "model",
    "local_matmul": linear.traced_canonical_vjp_matmul,
    "endpoint": "tied_embed",
}:
  raise AssertionError(f"P38 tied hook contract drifted: {captured!r}")
if model.TP_SIZE != 4 or model.MATMUL_N_PADDING != {37984: 38144}:
  raise AssertionError("Qwen3 TP4 lm_head padding contract is absent")
if fixed.REQUEST_M != (8, 16, 32, 64, 128, 256):
  raise AssertionError(f"fixed lm_head request buckets drifted: {fixed.REQUEST_M}")
if fixed.LEARNER_M != (4096,):
  raise AssertionError(f"fixed lm_head learner rows drifted: {fixed.LEARNER_M}")
model.preflight(require_enabled=True)
fixed.validate_global_contract(
    (16, args.hidden_size),
    (args.hidden_size, fixed.VOCAB),
    "bfloat16",
    "bfloat16",
    tp_size=4,
)

print(
    "P38_FIXED_LM_HEAD_EXACT_IMAGE_PASS "
    f"chain=linear_p22xk model={model_name} tp=4 K={args.hidden_size} "
    "request_M=8,16,32,64,128,256 "
    "learner_M=4096 fixed_M=256 "
    "local_N=37984 fixed_N=38144 endpoints=untied_lm_head,tied_embed",
    flush=True,
)
