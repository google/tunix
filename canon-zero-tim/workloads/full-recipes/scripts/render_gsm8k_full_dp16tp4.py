#!/usr/bin/env python3
"""Render one optimized GSM8K DP16xTP4 full-training JobSet."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]
_RENDERER_PATH = _SCRIPT_DIR / "render_three_full_recipes.py"


def _load_renderer():
  spec = importlib.util.spec_from_file_location(
      "v1_phase4_gsm8k_full_renderer", _RENDERER_PATH
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load renderer: {_RENDERER_PATH}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--run-id", required=True)
  parser.add_argument(
      "--base",
      type=Path,
      default=_REPO_ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
  )
  args = parser.parse_args()
  renderer = _load_renderer()
  renderer.render_gsm8k_full(
      source_commit=args.source_commit,
      output_dir=args.output_dir,
      run_id=args.run_id,
      base_path=args.base,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
