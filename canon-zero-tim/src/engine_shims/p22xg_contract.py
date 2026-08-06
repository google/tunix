#!/usr/bin/env python3
"""Fail-closed contract for the additive P22.XG SwiGLU custom call."""

from __future__ import annotations

import os


ENV = "CANON_PALLAS_SWIGLU"
BM = 128
BF = 256
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
        raise SystemExit(f"P22.XG preflight: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise SystemExit(f"P22.XG preflight: {ENV}=1 required")
    if value != "1":
        return
    required = {
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
        "CANON_FIXED_AR_EMBED": "1",
    }
    wrong = [f"{name}={os.environ.get(name)!r}" for name, expected in required.items()
             if os.environ.get(name, "") != expected]
    if wrong:
        raise SystemExit("P22.XG preflight: required canonical env missing: " + ", ".join(wrong))
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise SystemExit("P22.XG preflight: conflicting diagnostics: " + ",".join(active))


def validate_shape(shape_gate, shape_up) -> tuple[int, int]:
    if tuple(shape_gate) != tuple(shape_up):
        raise ValueError(f"P22.XG gate/up shapes differ: {shape_gate} vs {shape_up}")
    if len(shape_gate) != 2:
        raise ValueError(f"P22.XG expects rank-2 local arrays, got {shape_gate}")
    m, f = map(int, shape_gate)
    if m % BM or f % BF:
        raise ValueError(f"P22.XG shape must divide BM/BF={BM}/{BF}, got {(m, f)}")
    return m, f

