#!/usr/bin/env python3
"""Import the pinned stock engine without a canonical overlay on sys.path."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


FORBIDDEN_ENV = (
    "CANON_SHIM_ROOT",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_ALL_PROJ",
)
MODULES = (
    "tpu_inference.layers.common.attention_interface",
    "tpu_inference.layers.jax.embed",
    "tpu_inference.layers.jax.linear",
    "tpu_inference.runner.tpu_runner",
    "tpu_inference.models.jax.qwen3",
    "tpu_inference.models.jax.qwen2",
)


def main() -> int:
  leaked = {name: os.environ[name] for name in FORBIDDEN_ENV if name in os.environ}
  if leaked:
    raise RuntimeError(f"stock engine import received canonical env: {sorted(leaked)}")
  paths = []
  for name in MODULES:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if "canon-state" in path.parts or "engine_shims" in path.parts:
      raise RuntimeError(f"stock engine import resolved through overlay: {name}={path}")
    paths.append(str(path))
  print(
      "P57_STOCK_ENGINE_IMPORT_PASS "
      f"modules={len(paths)} root={Path(paths[0]).parents[3]}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
