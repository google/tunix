#!/usr/bin/env python3
"""Build one self-contained, immutable P38 diagnostic-round directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _copy(source: Path, destination: Path) -> None:
  _require(source.is_file() and source.stat().st_size > 0,
           f"required round artifact is missing or empty: {source}")
  shutil.copyfile(source, destination)


def _filter_jsonl(source: Path, destination: Path, round_index: int) -> int:
  _require(source.is_file(), f"round JSONL source is absent: {source}")
  selected: list[str] = []
  for line_number, line in enumerate(
      source.read_text(encoding="utf-8").splitlines(), start=1
  ):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(
          f"invalid JSONL record in {source}:{line_number}"
      ) from exc
    if int(record.get("diagnostic_round", -1)) == round_index:
      selected.append(json.dumps(record, sort_keys=True))
  _require(selected, f"no round {round_index} records in {source}")
  destination.write_text("\n".join(selected) + "\n", encoding="utf-8")
  return len(selected)


def _copy_record_pairs(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    round_index: int,
    expected_schema: str,
) -> int:
  selected = 0
  for json_path in sorted(source_dir.glob(f"{prefix}_*.json")):
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if int(record.get("diagnostic_round", -1)) != round_index:
      continue
    _require(record.get("schema") == expected_schema,
             f"record schema drifted: {json_path.name}")
    npz_path = json_path.with_suffix(".npz")
    _require(npz_path.is_file(), f"paired NPZ is absent: {npz_path.name}")
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    _require(digest == record.get("npz_sha256"),
             f"paired NPZ SHA failed: {npz_path.name}")
    _copy(json_path, output_dir / json_path.name)
    _copy(npz_path, output_dir / npz_path.name)
    selected += 1
  return selected


def stage(args: argparse.Namespace) -> dict:
  _require(0 <= args.round < 8, "diagnostic round must be in [0, 8)")
  _require(not args.output.exists(), f"round stage already exists: {args.output}")
  args.output.mkdir(parents=True, mode=0o700)

  capsule = Path(f"{args.capsule.with_suffix('')}.round-{args.round:06d}.npz")
  _copy(capsule, args.output / "mismatch-capsule.npz")
  _copy(args.run_log, args.output / "run.log")
  pre_alignment_records = _filter_jsonl(
      args.pre_alignment, args.output / "pre-alignment.jsonl", args.round
  )
  journal_records = _filter_jsonl(
      args.request_journal, args.output / "request-journal.jsonl", args.round
  )
  incident_records = _filter_jsonl(
      args.incident_ledger, args.output / "incident-ledger.jsonl", args.round
  )
  seam_records = _copy_record_pairs(
      args.observer_dir,
      args.output,
      "p38_seam",
      args.round,
      "p38-seam-fingerprint-v1",
  )
  if args.require_seam:
    _require(seam_records > 0, f"round {args.round} has no seam records")
  kv_records = _copy_record_pairs(
      args.observer_dir,
      args.output,
      "p38_kv_observer",
      args.round,
      "p38-live-kv-prefix-table-v1",
  )
  if args.require_kv:
    _require(kv_records > 0, f"round {args.round} has no KV observer records")
  tail_records = _copy_record_pairs(
      args.observer_dir,
      args.output,
      "p38_tail",
      args.round,
      "p38-tail-values-v1",
  )
  if args.require_tail:
    _require(tail_records > 0, f"round {args.round} has no tail records")

  record = {
      "diagnostic_round": args.round,
      "incident_records": incident_records,
      "journal_records": journal_records,
      "kv_records": kv_records,
      "pre_alignment_records": pre_alignment_records,
      "schema": "canon-p38-round-stage-v1",
      "seam_records": seam_records,
      "tail_records": tail_records,
  }
  (args.output / "ROUND_INVENTORY.json").write_text(
      json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return record


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--round", required=True, type=int)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--pre-alignment", required=True, type=Path)
  parser.add_argument("--capsule", required=True, type=Path)
  parser.add_argument("--request-journal", required=True, type=Path)
  parser.add_argument("--incident-ledger", required=True, type=Path)
  parser.add_argument("--observer-dir", required=True, type=Path)
  parser.add_argument("--require-seam", action="store_true")
  parser.add_argument("--require-kv", action="store_true")
  parser.add_argument("--require-tail", action="store_true")
  args = parser.parse_args()
  result = stage(args)
  print(
      "[P38.ROUND] STAGED "
      f"round={result['diagnostic_round']} seam={result['seam_records']} "
      f"kv={result['kv_records']} tail={result['tail_records']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
