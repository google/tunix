#!/usr/bin/env python3
"""Prepare a self-hashed M15 classifier-input checkpoint before classification.

The observer payload is already durable in immutable wide shards.  This
checkpoint preserves the remaining host-only inputs needed to reassemble and
rerun the classifier if analysis code fails after the rollout pod exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class M15ClassifierInputError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise M15ClassifierInputError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"{label} is absent")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as error:
    raise M15ClassifierInputError(f"{label} is invalid") from error
  _require(isinstance(value, dict), f"{label} is not an object")
  return value


def checkpoint(directory: Path, *, arm: str) -> dict[str, Any]:
  _require(arm in ("off", "on"), "invalid M15 APC arm")
  _require(directory.is_dir(), "assembled M15 round directory is absent")
  manifest = directory / "CLASSIFIER_INPUT_SHA256SUMS"
  receipt_path = directory / "CLASSIFIER_INPUT_RECEIPT.json"
  _require(not manifest.exists() and not receipt_path.exists(),
           "classifier-input checkpoint already exists")

  round_receipt = _json(directory / "ROUND_INPUT_RECEIPT.json", "round receipt")
  _require(
      round_receipt.get("schema") == "m15-wide-sealed-input-v1"
      and round_receipt.get("status") == "PASS"
      and isinstance(round_receipt.get("diagnostic_round"), int)
      and 0 <= int(round_receipt["diagnostic_round"]) < 8
      and isinstance(round_receipt.get("record_pairs"), int)
      and int(round_receipt["record_pairs"]) > 0
      and isinstance(round_receipt.get("shards"), list)
      and bool(round_receipt["shards"])
      and re.fullmatch(
          r"[0-9a-f]{40}", str(round_receipt.get("expected_source_commit", ""))
      ) is not None
      and round_receipt.get("runtime_source_commit")
      == round_receipt.get("expected_source_commit"),
      "round receipt contract drifted",
  )
  alignment_path = directory / "pre-alignment.jsonl"
  _require(alignment_path.is_file() and alignment_path.stat().st_size > 0,
           "pre-alignment input is absent")
  try:
    alignment_rows = [
        json.loads(line)
        for line in alignment_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
  except (json.JSONDecodeError, OSError) as error:
    raise M15ClassifierInputError("pre-alignment input is invalid") from error
  _require(len(alignment_rows) == 1, "classifier checkpoint requires one alignment row")
  alignment = alignment_rows[0]
  round_index = int(round_receipt["diagnostic_round"])
  _require(int(alignment.get("diagnostic_round", -1)) == round_index,
           "alignment round differs from assembled receipt")
  ab_bytes = int(
      alignment.get("boundaries", {})
      .get("S_decode_vs_S_prefill", {})
      .get("differing_bytes", -1)
  )
  _require(ab_bytes >= 0, "alignment A-B byte count is invalid")

  names = [
      "ROUND_INPUT_RECEIPT.json",
      "m15-replay-envelope.jsonl",
      "pre-alignment.jsonl",
  ]
  capsule = directory / "mismatch-capsule.npz"
  if capsule.is_file() and capsule.stat().st_size > 0:
    names.append(capsule.name)
  _require((ab_bytes == 0) == (capsule.name not in names),
           "mismatch capsule presence disagrees with A-B verdict")
  names = sorted(names)
  for name in names:
    path = directory / name
    _require(path.is_file() and path.stat().st_size > 0,
             f"classifier input is absent: {name}")
  replay_rows = []
  try:
    replay_rows = [
        json.loads(line)
        for line in (directory / "m15-replay-envelope.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
  except (json.JSONDecodeError, OSError) as error:
    raise M15ClassifierInputError("replay input is invalid") from error
  _require(
      bool(replay_rows)
      and all(
          row.get("schema") == "m15-apc-serving-envelope-v1"
          and int(row.get("diagnostic_round", -1)) == round_index
          for row in replay_rows
      ),
      "replay input round contract drifted",
  )

  manifest.write_text(
      "".join(f"{_sha256(directory / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  receipt = {
      "schema": "m15-wide-classifier-input-v1",
      "status": "prepared-for-durable-upload",
      "arm": arm,
      "diagnostic_round": round_index,
      "a_b_differing_bytes": ab_bytes,
      "files": names,
      "manifest_sha256": _sha256(manifest),
      "record_pairs": int(round_receipt["record_pairs"]),
      "shards": round_receipt["shards"],
      "expected_source_commit": round_receipt.get("expected_source_commit"),
      "runtime_source_commit": round_receipt.get("runtime_source_commit"),
      "claim_ceiling": (
          "CLASSIFIER_INPUT_ONLY / OBSERVER_VALUES_REMAIN_IN_VERIFIED_SHARDS / "
          "NO_NUMERICAL_CLASSIFICATION"
      ),
  }
  receipt_path.write_text(
      json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return receipt


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  receipt = checkpoint(args.directory, arm=args.arm)
  _require(
      args.output.resolve()
      == (args.directory / "CLASSIFIER_INPUT_RECEIPT.json").resolve(),
      "output must be the classifier receipt inside the assembled round",
  )
  print(json.dumps(receipt, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
