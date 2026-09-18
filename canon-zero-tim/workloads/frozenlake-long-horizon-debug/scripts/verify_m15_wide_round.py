#!/usr/bin/env python3
"""Verify that the terminal M15 artifacts came from one sealed shard union."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


class VerificationError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise VerificationError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def verify(
    *,
    round_directory: Path,
    classification: Path,
    bundle: Path,
    expected_commit: str,
    runtime_commit: str,
) -> dict[str, object]:
  _require(round_directory.is_dir(),
           f"sealed round directory is absent: {round_directory}")
  _require(expected_commit == runtime_commit and len(expected_commit) == 40,
           "runtime source does not match rendered source")
  expected_names = {
      "ROUND_INPUT_RECEIPT.json",
      "p38_seam.classification.json",
      "m15_wide_seam_bundle.tar",
  }
  manifest = round_directory / "WIDE_SHA256SUMS"
  _require(manifest.is_file() and manifest.stat().st_size > 0,
           "wide-round SHA manifest is absent")
  rows: dict[str, str] = {}
  for line in manifest.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(separator == "  " and len(digest) == 64,
             f"invalid wide-round manifest row: {line!r}")
    _require(name in expected_names and name not in rows,
             f"unexpected or duplicate wide-round member: {name!r}")
    rows[name] = digest
  _require(set(rows) == expected_names,
           f"wide-round manifest membership drifted: {sorted(rows)}")
  for name, digest in rows.items():
    member = round_directory / name
    _require(member.is_file() and _sha256(member) == digest,
             f"wide-round member failed SHA: {name}")

  receipt = json.loads(
      (round_directory / "ROUND_INPUT_RECEIPT.json").read_text(encoding="utf-8")
  )
  _require(receipt.get("schema") == "m15-wide-sealed-input-v1"
           and receipt.get("status") == "PASS",
           "wide-round input receipt is not PASS")
  _require(receipt.get("expected_source_commit") == expected_commit
           and receipt.get("runtime_source_commit") == runtime_commit,
           "wide-round source receipt drifted")
  _require(int(receipt.get("record_pairs", 0)) > 0
           and int(receipt.get("replay_records", 0)) > 0
           and len(receipt.get("shards", ())) > 0,
           "wide-round input receipt is empty")
  diagnostic_round = int(receipt.get("diagnostic_round", -1))
  _require(0 <= diagnostic_round < 8,
           "wide-round input diagnostic round is invalid")

  classification_record = json.loads(
      (round_directory / "p38_seam.classification.json").read_text(
          encoding="utf-8"
      )
  )
  _require(classification_record.get("status") == "PASS"
           and int(classification_record.get("diagnostic_round", -1))
           == diagnostic_round,
           "wide-round classification diagnostic round drifted")

  completion = json.loads(
      (round_directory / "WIDE_ROUND_COMPLETE.json").read_text(encoding="utf-8")
  )
  _require(completion.get("schema") == "m15-wide-round-completion-v1"
           and completion.get("status") == "classified-and-uploaded",
           "wide-round completion receipt is not terminal")
  _require(completion.get("expected_source_commit") == expected_commit
           and completion.get("runtime_source_commit") == runtime_commit,
           "wide-round completion source drifted")
  _require(int(completion.get("diagnostic_round", -1)) == diagnostic_round,
           "wide-round completion diagnostic round drifted")
  _require(completion.get("classification")
           == classification_record.get("classification"),
           "wide-round completion classification drifted")
  _require(completion.get("manifest_sha256") == _sha256(manifest),
           "wide-round completion manifest SHA drifted")
  _require(int(completion.get("record_pairs", -1)) == receipt["record_pairs"]
           and completion.get("shards") == receipt["shards"],
           "wide-round completion input inventory drifted")

  canonical_classification = round_directory / "p38_seam.classification.json"
  canonical_bundle = round_directory / "m15_wide_seam_bundle.tar"
  _require(classification.is_file()
           and classification.read_bytes() == canonical_classification.read_bytes(),
           "published classification differs from sealed-round output")
  _require(bundle.is_file() and bundle.read_bytes() == canonical_bundle.read_bytes(),
           "published bundle differs from sealed-round output")
  return {
      "diagnostic_round": diagnostic_round,
      "classification": completion["classification"],
      "record_pairs": receipt["record_pairs"],
      "replay_records": receipt["replay_records"],
      "shards": len(receipt["shards"]),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--round-directory", required=True, type=Path)
  parser.add_argument("--classification", required=True, type=Path)
  parser.add_argument("--bundle", required=True, type=Path)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--runtime-commit", required=True)
  args = parser.parse_args()
  receipt = verify(
      round_directory=args.round_directory,
      classification=args.classification,
      bundle=args.bundle,
      expected_commit=args.expected_commit,
      runtime_commit=args.runtime_commit,
  )
  print(
      "[M15.WIDE.ROUND] VERIFIED "
      f"shards={receipt['shards']} pairs={receipt['record_pairs']} "
      f"replay_records={receipt['replay_records']} "
      f"classification={receipt['classification']}"
  )
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (VerificationError, json.JSONDecodeError, OSError) as error:
    print(f"[M15.WIDE.ROUND] RED {error}", file=sys.stderr)
    raise SystemExit(2) from error
