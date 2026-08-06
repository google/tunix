"""Fail-closed contract for additive P22.XJ SwiGLU M-row compatibility."""

from __future__ import annotations

import os


ENV = "CANON_PALLAS_SWIGLU_MPAD"
REQUIRES = (
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_ALL_PROJ",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
)
CONFLICTS = (
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_POSTRPA_M",
    "CANON_CUT",
    "CANON_TAIL",
    "P16_NUM_LAYERS",
)


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise RuntimeError(f"P22.XJ: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise RuntimeError(f"P22.XJ: {ENV}=1 required")
    if value != "1":
        return
    missing = [name for name in REQUIRES if os.environ.get(name, "") != "1"]
    if missing:
        raise RuntimeError("P22.XJ: missing required full-stack env: " + ",".join(missing))
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise RuntimeError("P22.XJ: conflicting diagnostics: " + ",".join(active))

