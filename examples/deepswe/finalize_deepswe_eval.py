#!/usr/bin/env python3
"""Fail-closed finalizer for the complete P46 DeepSWE evaluation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepswe_eval_artifacts import finalize_campaign


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--summary-json", nargs="+", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  result = finalize_campaign(args.summary_json, args.output_dir)
  print(
      "P46_EVAL_CAMPAIGN_PASS "
      f"tasks={result['tasks']} n_sample={result['n_sample']} "
      f"valid_trajectories={result['valid_trajectories']} "
      f"logical_shards={result['logical_shards']} "
      f"summary_sha256={result['summary_sha256']}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
