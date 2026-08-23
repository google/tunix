"""Opt-in JAX name-stack labels for zero-TIM XProf captures."""

from __future__ import annotations

import contextlib
import os
import re


ENV = "CANON_XPROF_LABELS"


def enabled(environ=None) -> bool:
    """Returns whether labels are enabled, rejecting typoed modes."""
    source = os.environ if environ is None else environ
    value = source.get(ENV, "")
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{ENV} must be unset/0/1, got {value!r}")
    return value == "1"


def scope(name: str):
    """Returns a JAX name scope when enabled and an exact no-op otherwise."""
    if not enabled():
        return contextlib.nullcontext()
    if not re.fullmatch(r"[a-z0-9_./-]+", name):
        raise RuntimeError(f"invalid XProf operation scope {name!r}")
    import jax

    return jax.named_scope(name)


def layer_tag(prefix: str) -> str:
    """Maps a live model prefix to the compact lNN label vocabulary."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(prefix))
    if match is None:
        raise RuntimeError(f"XProf layer scope requires a layer prefix, got {prefix!r}")
    return f"l{int(match.group(1)):02d}"
