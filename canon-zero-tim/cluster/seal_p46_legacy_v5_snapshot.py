#!/usr/bin/env python3
"""Seals a terminal P46 trajectory-v5 staging copy for reviewed adoption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "deepswe"))
sys.path.insert(0, str(ROOT / "canon-zero-tim" / "cluster"))

import deepswe_eval_artifacts as artifacts  # pylint: disable=wrong-import-position
import render_p34_jobset as p34  # pylint: disable=wrong-import-position


def _task_order(whitelist: Path) -> list[str]:
  if artifacts.sha256_file(whitelist) != p34.P34_CLEAN_WHITELIST_SHA256:
    raise ValueError("P46 clean whitelist SHA-256 mismatch")
  keys = []
  with whitelist.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      if not line.strip():
        continue
      record = json.loads(line)
      key = record.get("docker_image")
      if not isinstance(key, str) or not key:
        raise ValueError(f"whitelist line {line_number} lacks docker_image")
      keys.append(key)
  if len(keys) != p34.P34_CLEAN_ROWS or len(set(keys)) != len(keys):
    raise ValueError("P46 clean whitelist cardinality/uniqueness mismatch")
  return sorted(keys)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--snapshot-dir", required=True)
  parser.add_argument("--sampling-source-commit", required=True)
  parser.add_argument("--topology", choices=("64", "128"), required=True)
  parser.add_argument(
      "--whitelist", default=str(ROOT / p34.P34_CLEAN_WHITELIST)
  )
  args = parser.parse_args()

  snapshot = Path(args.snapshot_dir).resolve()
  whitelist = Path(args.whitelist).resolve()
  task_order = _task_order(whitelist)
  # Paths and client image were not recorded by trajectory-v5 and are excluded
  # from stable source semantics. They remain syntactically valid placeholders
  # solely so EvalConfig can enforce every semantic P46 field.
  config = artifacts.EvalConfig(
      model_id="Qwen/Qwen3-4B-Instruct-2507",
      model_path="/legacy-source/unrecorded-model-path",
      dataset_name=p34.P34_DATASET_NAME,
      dataset_revision=p34.P34_DATASET_REVISION,
      dataset_split=p34.P34_DATASET_SPLIT,
      dataset_rows=p34.P34_DATASET_ROWS,
      whitelist_path=str(whitelist),
      whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
      whitelist_rows=p34.P34_CLEAN_ROWS,
      source_commit=args.sampling_source_commit,
      harness_commit=args.sampling_source_commit,
      client_image="legacy-source.invalid/tunix@sha256:" + "0" * 64,
      topology=args.topology,
      resume_tag="legacy-source-seal",
  )
  result = artifacts.seal_legacy_v5_snapshot(
      snapshot, config=config, allowed_task_keys=task_order
  )
  print(
      "P46_LEGACY_V5_SEAL_PASS "
      f"snapshot={snapshot} records={result['records']} "
      f"manifest_sha256={result['snapshot_manifest_sha256']} "
      "contract_sha256="
      f"{result['legacy_source_contract_sha256']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
