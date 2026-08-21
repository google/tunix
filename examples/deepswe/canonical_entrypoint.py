#!/usr/bin/env python3
"""Initializes Pathways once before importing the canonical DeepSWE program."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
  if os.environ.get("CANON_P34_DEEPSWE", "") != "1":
    raise RuntimeError("canonical DeepSWE entrypoint requires CANON_P34_DEEPSWE=1")
  if os.environ.get("CANON_PATHWAYS_INITIALIZED", ""):
    raise RuntimeError("Pathways was initialized before the canonical entrypoint")
  if "proxy" in os.environ.get("JAX_PLATFORMS", "").split(","):
    import pathwaysutils

    pathwaysutils.initialize()
  os.environ["CANON_PATHWAYS_INITIALIZED"] = "1"
  print("[P34.PATHWAYS] initialized_once=1 before_jax=1", flush=True)
  # This wrapper is launched by the signed JobSet as a file path, for example
  # ``python3 /app/examples/deepswe/canonical_entrypoint.py``.  In that mode
  # Python adds only /app/examples/deepswe to sys.path; it does not add /app,
  # so the package-qualified target below is otherwise undiscoverable.  Derive
  # the repository root from this file instead of relying on the caller's cwd
  # or an externally supplied PYTHONPATH.
  repository_root = str(Path(__file__).resolve().parents[2])
  if repository_root not in sys.path:
    sys.path.insert(0, repository_root)
  runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")


if __name__ == "__main__":
  main()
