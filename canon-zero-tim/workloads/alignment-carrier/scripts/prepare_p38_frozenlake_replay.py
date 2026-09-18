#!/usr/bin/env python3
"""Verify a P38 capsule and emit its E0-lite mask-derived schedule contract.

This tool does not consume the production incident ledger and therefore cannot
construct strict E0. Its output is a bounded counterfactual only.
"""

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
  parser.add_argument("--source-row", type=int)
  parser.add_argument("--local-m", type=int, default=256)
  args = parser.parse_args()
  repo = Path(__file__).resolve().parents[4]
  replay = _load_module(repo)
  capsule = replay.load_verified_capsule(args.capsule)
  row_index = args.row_index
  if args.source_row is not None:
    matches = [
        index for index, row in enumerate(capsule.rows)
        if row.source_row == args.source_row
    ]
    if len(matches) != 1:
      raise ValueError(
          f"source row must identify exactly one capsule row: "
          f"source_row={args.source_row} matches={matches}"
      )
    row_index = matches[0]
  if row_index < 0 or row_index >= len(capsule.rows):
    raise ValueError(f"row index is out of range: {row_index}")
  row = capsule.rows[row_index]
  schedules = (
      replay.build_r0_mask_derived_schedule(row, local_m=args.local_m),
      replay.build_r1_continuous_decode_schedule(row, local_m=args.local_m),
      replay.build_fixed_chunk_reference_schedule(
          row, local_m=args.local_m
      ),
  )
  report = replay.schedules_report(capsule, schedules)
  report["selected_row_index"] = row_index
  report["selected_source_row"] = row.source_row
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
