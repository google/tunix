#!/usr/bin/env python3
"""Fail-fast import identity gate for the Qwen3-8B TP2 projection family."""

from __future__ import annotations

import hashlib
from pathlib import Path

import p22xf_contract
from tunix.rl import canonical_qwen3_adapter


expected = (
    ("q_proj", 4096, 2048),
    ("k_proj", 4096, 512),
    ("v_proj", 4096, 512),
    ("o_proj", 2048, 4096),
    ("gate_proj", 4096, 6144),
    ("up_proj", 4096, 6144),
    ("down_proj", 6144, 4096),
)
actual = tuple(
    (site.family, site.k_local, site.n_local)
    for site in p22xf_contract.SITES
)
if p22xf_contract.TP_SIZE != 2 or actual != expected:
  raise RuntimeError(
      "Qwen3-8B TP2 projection contract import mismatch: "
      f"tp={getattr(p22xf_contract, 'TP_SIZE', None)} "
      f"sites={actual} file={p22xf_contract.__file__}"
  )
p22xf_contract.validate_manifest(p22xf_contract.SITES)
if canonical_qwen3_adapter._canonical_logprob_bucket() != 512:
  raise RuntimeError("P32 adapter did not admit the global M512 contract")
source = Path(p22xf_contract.__file__).read_bytes()
print(
    "P32_QWEN8B_TP2_IMPORT_GATE_PASS "
    f"file={p22xf_contract.__file__} "
    f"sha256={hashlib.sha256(source).hexdigest()} sites=7/7 global_m=512",
    flush=True,
)
