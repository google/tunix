#!/usr/bin/env python3
"""Import the installed Qwen3-1.7B TP1 chain under its exact environment."""

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
):
  os.environ[name] = "1"

os.environ.update({
    "CANON_QWEN3_HIDDEN_SIZE": "2048",
    "CANON_QWEN3_INTERMEDIATE_SIZE": "6144",
    "CANON_QWEN3_NUM_ATTENTION_HEADS": "16",
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": "1",
})

import linear_p22xk as linear  # noqa: E402
import p22xf_contract as model  # noqa: E402


if not linear.P22XK_MATMUL_ACTIVE:
  raise AssertionError("installed P22.XK matmul wrapper is inactive")
if model.TP_SIZE != 1 or len(model.SITES) != 7:
  raise AssertionError("installed Qwen3-1.7B TP1 model contract is not active")
if model.MATMUL_N_PADDING != {151936: 152064}:
  raise AssertionError("installed Qwen3-1.7B TP1 lm-head padding is wrong")
model.preflight(require_enabled=True)

print(
    "P59_QWEN1P7B_TP1_IMPORT_PASS "
    "chain=linear_p22xk model=qwen1p7b_tp1 tp=1 sites=7",
    flush=True,
)
