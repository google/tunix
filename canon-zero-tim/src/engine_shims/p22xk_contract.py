#!/usr/bin/env python3
"""Fail-closed contract for P22.XK canonical custom-VJP promotion."""

from __future__ import annotations

import os


ENV = "CANON_PALLAS_CANONICAL_VJP"
REQUIRED = {
    "CANON_PALLAS_ALL_PROJ": "1",
    "CANON_PALLAS_SWIGLU": "1",
    "CANON_PALLAS_ALL_RMSNORM": "1",
    "CANON_PALLAS_MPAD": "1",
    "CANON_PALLAS_SWIGLU_MPAD": "1",
    "CANON_FIXED_AR": "1",
    "CANON_FIXED_AR_EMBED": "1",
}
CONFLICTS = ("CANON_CUT", "CANON_TAIL", "CANON_POSTRPA_M")


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise SystemExit(f"P22.XK preflight: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise SystemExit(f"P22.XK preflight: {ENV}=1 required")
    if value != "1":
        return
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, expected in REQUIRED.items()
        if os.environ.get(name, "") != expected
    ]
    if wrong:
        raise SystemExit("P22.XK preflight: required canonical env missing: " + ", ".join(wrong))
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise SystemExit("P22.XK preflight: conflicting diagnostics: " + ",".join(active))

