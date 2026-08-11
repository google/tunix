#!/usr/bin/env python3
"""Initializes Pathways once before importing the canonical DeepSWE program."""

from __future__ import annotations

import os
import runpy


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
  runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")


if __name__ == "__main__":
  main()
