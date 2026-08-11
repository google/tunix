#!/usr/bin/env python3
"""Verify a P38 capsule and emit its mask-derived R0/R1 schedule contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_module(repo: Path):
  module_path = repo / "tunix/rl/p38_frozenlake_replay.py"
  spec = importlib.util.spec_from_file_location(
      "p38_frozenlake_replay", module_path
  )
  module = importlib.util.module_from_spec(spec)
  if spec.loader is None:
    raise RuntimeError(f"cannot load P38 replay module: {module_path}")
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--capsule", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--row-index", type=int, default=0)
  parser.add_argument("--local-m", type=int, default=256)
  args = parser.parse_args()
  repo = Path(__file__).resolve().parents[4]
  replay = _load_module(repo)
  capsule = replay.load_verified_capsule(args.capsule)
  if args.row_index < 0 or args.row_index >= len(capsule.rows):
    raise ValueError(f"row index is out of range: {args.row_index}")
  row = capsule.rows[args.row_index]
  schedules = (
      replay.build_r0_mask_derived_schedule(row, local_m=args.local_m),
      replay.build_r1_continuous_decode_schedule(row, local_m=args.local_m),
      replay.build_fixed_chunk_reference_schedule(
          row, local_m=args.local_m
      ),
  )
  report = replay.schedules_report(capsule, schedules)
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
