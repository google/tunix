"""Fail-closed dependency contract for P22.XI row-padding compatibility."""

from __future__ import annotations

import os


ENV = "CANON_PALLAS_MPAD"
REQUIRES = (
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_SWIGLU",
)
OPTIONAL_XH = "CANON_PALLAS_ALL_RMSNORM"
CONFLICTS = (
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_CUT",
    "CANON_TAIL",
    "CANON_POSTRPA_M",
    "P16_NUM_LAYERS",
)


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise RuntimeError(f"P22.XI: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise RuntimeError(f"P22.XI: {ENV}=1 required")
    if value != "1":
        return
    xh = os.environ.get(OPTIONAL_XH, "")
    if xh not in ("", "1"):
        raise RuntimeError(f"P22.XI: {OPTIONAL_XH} must be unset or 1, got {xh!r}")
    missing = [name for name in REQUIRES if os.environ.get(name, "") != "1"]
    if missing:
        raise RuntimeError("P22.XI: missing required full-stack env: " + ",".join(missing))
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise RuntimeError("P22.XI: conflicting diagnostics: " + ",".join(active))
