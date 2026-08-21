#!/usr/bin/env python3
"""Attest one installed Qwen3 fixed-lm-head hook in the pinned image."""

from __future__ import annotations

import argparse
import os
import types


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hidden-size", type=int, choices=(2048, 2560, 4096, 5120), default=4096
)
parser.add_argument("--tp-size", type=int, choices=(4, 8), default=None)
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
    (2048, 4): (6144, 16, "qwen1p7b", "tied_embed", 37984, 38144),
    (4096, 4): (12288, 32, "qwen8b", "untied_lm_head", 37984, 38144),
    (4096, 8): (12288, 32, "qwen8b_tp8", "untied_lm_head", 18992, 19200),
    (2560, 8): (9728, 32, "qwen4b", "tied_embed", 18992, 19200),
    (5120, 8): (25600, 64, "qwen32b", "untied_lm_head", 18992, 19200),
}
tp_size = args.tp_size or (4 if args.hidden_size in (2048, 4096) else 8)
(
    intermediate_size,
    attention_heads,
    model_name,
    endpoint,
    local_vocab,
    padded_local_vocab,
) = model_env[(args.hidden_size, tp_size)]
os.environ.update({
    "CANON_QWEN3_HIDDEN_SIZE": str(args.hidden_size),
    "CANON_QWEN3_INTERMEDIATE_SIZE": str(intermediate_size),
    "CANON_QWEN3_NUM_ATTENTION_HEADS": str(attention_heads),
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": str(tp_size),
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
fake_inputs = object()
if endpoint == "tied_embed":
  fake_endpoint = types.SimpleNamespace(
      weight=types.SimpleNamespace(value=_FakeWeight())
  )
  result = embed_module.JaxEmbed.decode(fake_endpoint, fake_inputs)
  expected_weight = "transposed-embed-weight"
else:
  fake_endpoint = types.SimpleNamespace(
      einsum_str="TD,DV->TV",
      prefix="model.lm_head",
      weight=types.SimpleNamespace(value="untied-lm-head-weight"),
  )
  result = linear.JaxLmHead.__call__(fake_endpoint, fake_inputs)
  expected_weight = "untied-lm-head-weight"
if result != "fixed-tied-output":
  raise AssertionError(f"P38 {endpoint} hook did not return the fixed-head result")
if captured != {
    "inputs": fake_inputs,
    "weight": expected_weight,
    "mesh": "test-mesh",
    "tp_axis": "model",
    "local_matmul": linear.traced_canonical_vjp_matmul,
    "endpoint": endpoint,
}:
  raise AssertionError(f"P38 {endpoint} hook contract drifted: {captured!r}")
if model.TP_SIZE != tp_size:
  raise AssertionError(f"Qwen3 {model_name} TP width drifted: {model.TP_SIZE}")
if model.MATMUL_N_PADDING.get(local_vocab) != padded_local_vocab:
  raise AssertionError(
      f"Qwen3 {model_name} lm_head padding contract is absent: "
      f"{model.MATMUL_N_PADDING}"
  )
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
    tp_size=tp_size,
)
geometry = fixed.resolve_geometry(args.hidden_size, tp_size, endpoint=endpoint)
if (
    geometry.local_vocab != local_vocab
    or geometry.padded_local_vocab != padded_local_vocab
):
  raise AssertionError(f"fixed lm_head geometry drifted: {geometry}")

print(
    "P38_FIXED_LM_HEAD_EXACT_IMAGE_PASS "
    f"chain=linear_p22xk model={model_name} tp={tp_size} K={args.hidden_size} "
    "request_M=8,16,32,64,128,256 "
    "learner_M=4096 fixed_M=256 "
    f"local_N={local_vocab} fixed_N={padded_local_vocab} endpoint={endpoint}",
    flush=True,
)
