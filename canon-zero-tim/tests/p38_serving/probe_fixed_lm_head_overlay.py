#!/usr/bin/env python3
"""Attest the installed Qwen3-8B TP4 fixed-lm-head hook in the pinned image."""

from __future__ import annotations

import os


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

os.environ.update({
    "CANON_QWEN3_HIDDEN_SIZE": "4096",
    "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
    "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": "4",
})

import linear_p22xk as linear  # noqa: E402
import p22xf_contract as model  # noqa: E402
import p38_fixed_lm_head as fixed  # noqa: E402


if not linear.P22XK_MATMUL_ACTIVE:
  raise AssertionError("installed P22.XK matmul wrapper is inactive")
if not linear.P38_FIXED_LM_HEAD_ACTIVE:
  raise AssertionError("installed P38 fixed lm_head wrapper is inactive")
if linear.JaxLmHead.__call__.__name__ != "_p38_fixed_lm_head_call":
  raise AssertionError("JaxLmHead does not point at the P38 hook")
if model.TP_SIZE != 4 or model.MATMUL_N_PADDING != {37984: 38144}:
  raise AssertionError("Qwen3-8B TP4 lm_head padding contract is absent")
if fixed.REQUEST_M != (8, 16, 32, 64, 128, 256):
  raise AssertionError(f"fixed lm_head request buckets drifted: {fixed.REQUEST_M}")
if fixed.LEARNER_M != (4096,):
  raise AssertionError(f"fixed lm_head learner rows drifted: {fixed.LEARNER_M}")
model.preflight(require_enabled=True)

print(
    "P38_FIXED_LM_HEAD_EXACT_IMAGE_PASS "
    "chain=linear_p22xk model=qwen8b tp=4 request_M=8,16,32,64,128,256 "
    "learner_M=4096 fixed_M=256 "
    "local_N=37984 fixed_N=38144",
    flush=True,
)
