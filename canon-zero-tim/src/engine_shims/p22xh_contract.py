#!/usr/bin/env python3
"""Fail-closed contract for the additive P22.XH all-RMSNorm experiment."""

from __future__ import annotations

import os


ENV = "CANON_PALLAS_ALL_RMSNORM"
BM = 8
BF = 128
CONFLICTS = (
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_POSTRPA_M",
    "CANON_CUT",
    "CANON_TAIL",
)


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise SystemExit(f"P22.XH preflight: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise SystemExit(f"P22.XH preflight: {ENV}=1 required")
    if value != "1":
        return
    required = {
        "CANON_PALLAS_SWIGLU": "1",
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
        "CANON_FIXED_AR_EMBED": "1",
    }
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, expected in required.items()
        if os.environ.get(name, "") != expected
    ]
    if wrong:
        raise SystemExit("P22.XH preflight: required canonical env missing: " + ", ".join(wrong))
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise SystemExit("P22.XH preflight: conflicting diagnostics: " + ",".join(active))


def validate_shape(shape_x, shape_weight) -> tuple[int, int]:
    if len(shape_x) != 2:
        raise ValueError(f"P22.XH kernel expects rank-2 x, got {shape_x}")
    if len(shape_weight) != 1:
        raise ValueError(f"P22.XH kernel expects rank-1 weight, got {shape_weight}")
    m, f = map(int, shape_x)
    fw = int(shape_weight[0])
    if f != fw:
        raise ValueError(f"P22.XH feature/weight mismatch: {f} vs {fw}")
    if m % BM or f % BF:
        raise ValueError(f"P22.XH shape must divide BM/BF={BM}/{BF}, got {(m, f)}")
    return m, f

