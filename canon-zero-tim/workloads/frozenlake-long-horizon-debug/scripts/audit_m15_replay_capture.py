#!/usr/bin/env python3
"""Verify an extracted M15 capture and emit a small GCS return bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


class AuditError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise AuditError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing JSON: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise AuditError(f"invalid JSON {path}: {exc}") from exc
  _require(isinstance(value, dict), f"JSON is not an object: {path}")
  return value


def _verify_manifest(path: Path, allowed_root: Path) -> int:
  _require(path.is_file() and path.stat().st_size > 0, f"missing manifest: {path}")
  count = 0
  root = allowed_root.resolve()
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    parts = line.split("  ", 1)
    _require(len(parts) == 2, f"manifest line {line_number} is invalid: {path}")
    digest, name = parts
    _require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"manifest line {line_number} SHA is invalid")
    target = (path.parent / name).resolve()
    _require(target == root or root in target.parents, f"manifest path escapes capture root: {name}")
    _require(target.is_file() and target.stat().st_size > 0, f"manifest member is absent: {name}")
    _require(_sha256(target) == digest, f"manifest member SHA drifted: {name}")
    count += 1
  _require(count > 0, f"manifest is empty: {path}")
  return count


def audit(
    *,
    root_dir: Path,
    capture_dir: Path,
    source_gcs_uri: str,
    output_dir: Path,
) -> dict[str, Any]:
  _require(not output_dir.exists(), f"refusing to overwrite output: {output_dir}")
  prefix = "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
  _require(source_gcs_uri.startswith(prefix) and source_gcs_uri.endswith("/attempt-0"), "source GCS URI is outside the registered P38 bucket")
  root_members = _verify_manifest(root_dir / "SHA256SUMS", root_dir)
  markers = {
      name: _json(root_dir / name)
      for name in ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json")
  }
  _require(len({value.get("source_commit") for value in markers.values()}) == 1, "GCS marker source commits disagree")
  _require(all(value.get("prefix") == source_gcs_uri for value in markers.values()), "GCS marker prefix drifted")
  serving = _json(root_dir / "serving-classification.json")
  _require(serving.get("verdict") == "PASS", "serving classification is not PASS")
  target = _json(capture_dir / "m15_apc_target.classification.json")
  status = target.get("status")
  _require(status in ("CONTROL_GREEN", "TARGET_NOT_REPRODUCED", "FRESH_TARGET_RED_FROZEN"), f"inadmissible M15 status: {status}")
  _require((capture_dir / "m15_producer_unit.npz").is_file(), "full producer unit is absent")
  _require((capture_dir / "m15_replay_envelope.jsonl").is_file(), "serving replay envelope is absent")

  nested_manifest_members = 0
  full_contract = None
  if status == "FRESH_TARGET_RED_FROZEN":
    first_dir = capture_dir / "m15_first_red_replay"
    full_dir = capture_dir / "m15_full_replay_carrier"
    nested_manifest_members += _verify_manifest(first_dir / "SHA256SUMS", capture_dir)
    nested_manifest_members += _verify_manifest(full_dir / "SHA256SUMS", capture_dir)
    full_contract = _json(full_dir / "replay_contract.json")
    _require(full_contract.get("status") == "FULL_REPLAY_CARRIER_FROZEN", "full replay carrier is not frozen")
    _require(full_contract.get("source_commit") == next(iter({value["source_commit"] for value in markers.values()})), "full replay source commit drifted")
  else:
    _require(not (capture_dir / "m15_full_replay_carrier").exists(), "clean run unexpectedly contains a red replay carrier")

  output_dir.mkdir(parents=True, exist_ok=False)
  small_sources = {
      "PREFLIGHT.json": root_dir / "PREFLIGHT.json",
      "COLLECTED.json": root_dir / "COLLECTED.json",
      "COMPLETE.json": root_dir / "COMPLETE.json",
      "root-SHA256SUMS": root_dir / "SHA256SUMS",
      "serving-classification.json": root_dir / "serving-classification.json",
      "m15-classification.json": capture_dir / "m15_apc_target.classification.json",
  }
  if status == "FRESH_TARGET_RED_FROZEN":
    small_sources.update({
        "first-red-contract.json": capture_dir / "m15_first_red_replay/first_red_contract.json",
        "first-red-SHA256SUMS": capture_dir / "m15_first_red_replay/SHA256SUMS",
        "replay-contract.json": capture_dir / "m15_full_replay_carrier/replay_contract.json",
        "request-row-joins.jsonl": capture_dir / "m15_full_replay_carrier/request_row_joins.jsonl",
        "replay-SHA256SUMS": capture_dir / "m15_full_replay_carrier/SHA256SUMS",
    })
  for name, source in small_sources.items():
    _require(source.is_file() and source.stat().st_size > 0, f"small return source is absent: {source}")
    shutil.copyfile(source, output_dir / name)

  raw_log = root_dir / "run.log"
  selected_log = output_dir / "selected-markers.log"
  prefixes = (
      "[CANON_ALIGN_PRE]",
      "[CANON_" "ALIGN_PRE_JSON]",
      "[CANON_" "APC_M15",
      "[P3_APC_CONFIG]",
      "[CANON_P38] CONTROLLED_EXIT",
      "[run] FATAL",
  )
  with raw_log.open("r", encoding="utf-8", errors="replace") as source, selected_log.open("x", encoding="utf-8") as target_log:
    for line in source:
      if any(prefix in line for prefix in prefixes) or "Prefix cache hit rate" in line:
        target_log.write(line)
    target_log.flush()
    os.fsync(target_log.fileno())

  receipt = {
      "schema": "m15-apc-gcs-return-receipt-v1",
      "status": status,
      "source_gcs_uri": source_gcs_uri,
      "source_commit": markers["COMPLETE.json"]["source_commit"],
      "root_manifest_members": root_members,
      "nested_manifest_members": nested_manifest_members,
      "large_artifacts_retained_in_gcs": {
          "serving_capture_tar": f"{source_gcs_uri}/serving-capture.tar",
          "run_log": f"{source_gcs_uri}/run.log",
      },
      "full_replay": None if full_contract is None else {
          "producer_rows": full_contract["producer_rows"],
          "serving_call_count": full_contract["serving_call_count"],
          "request_count": full_contract["request_count"],
          "first_red": full_contract["first_red"],
      },
      "claim_ceiling": (
          "FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN"
          if full_contract is not None
          else "TARGET_OBSERVATION_ONLY"
      ),
  }
  receipt_path = output_dir / "RETURN_RECEIPT.json"
  with receipt_path.open("x", encoding="utf-8") as target_file:
    target_file.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    target_file.flush()
    os.fsync(target_file.fileno())
  members = sorted(path for path in output_dir.iterdir() if path.name != "SHA256SUMS")
  with (output_dir / "SHA256SUMS").open("x", encoding="utf-8") as manifest:
    for path in members:
      manifest.write(f"{_sha256(path)}  {path.name}\n")
    manifest.flush()
    os.fsync(manifest.fileno())
  return receipt


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root-dir", required=True, type=Path)
  parser.add_argument("--capture-dir", required=True, type=Path)
  parser.add_argument("--source-gcs-uri", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = audit(
        root_dir=args.root_dir,
        capture_dir=args.capture_dir,
        source_gcs_uri=args.source_gcs_uri,
        output_dir=args.output_dir,
    )
  except (OSError, KeyError, TypeError, ValueError, AuditError) as exc:
    print(json.dumps({"status": "INCONCLUSIVE", "error": str(exc)}, sort_keys=True))
    raise SystemExit(2) from exc
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
