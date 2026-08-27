#!/usr/bin/env python3
"""Assemble the classifier input exclusively from sealed M15 shard stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from stage_m15_wide_shard import (
    M15WideShardError,
    _candidate_pairs,
    _manifest,
    _require,
    _sealed_members,
    _sha256,
)


def _link_or_copy(source: Path, destination: Path) -> None:
  shutil.copyfile(source, destination)


def _copy_alignment(source: Path, output: Path, round_index: int) -> dict[str, Any]:
  selected = []
  for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    record = json.loads(line)
    if int(record.get("diagnostic_round", -1)) == round_index:
      selected.append(record)
  _require(len(selected) == 1,
           f"expected one pre-alignment row for round {round_index}, got {len(selected)}")
  output.write_text(json.dumps(selected[0], sort_keys=True) + "\n", encoding="utf-8")
  return selected[0]


def _copy_replay(source: Path, output: Path, round_index: int) -> int:
  _require(source.is_file() and source.stat().st_size > 0,
           f"M15 replay ledger is absent or empty: {source}")
  count = 0
  with source.open(encoding="utf-8") as input_stream, output.open(
      "x", encoding="utf-8"
  ) as output_stream:
    for line_number, line in enumerate(input_stream, start=1):
      if not line.strip():
        continue
      record = json.loads(line)
      _require(record.get("schema") == "m15-apc-serving-envelope-v1",
               f"replay schema drifted at line {line_number}")
      _require(int(record.get("diagnostic_round", -1)) == round_index,
               f"replay round drifted at line {line_number}")
      output_stream.write(json.dumps(record, sort_keys=True) + "\n")
      count += 1
  _require(count > 0, "M15 replay ledger contains no records")
  return count


def assemble(
    *,
    live_directory: Path,
    shard_root: Path,
    output: Path,
    round_index: int,
    pre_alignment: Path,
    capsule: Path,
    replay_ledger: Path,
    observer_mode: str,
    expected_commit: str,
    runtime_commit: str,
) -> dict[str, Any]:
  _require(observer_mode in ("layer", "full"), "invalid M15 observer mode")
  _require(expected_commit == runtime_commit and len(expected_commit) == 40,
           "runtime source does not match the rendered source")
  _require(not output.exists(), f"round input output already exists: {output}")
  sealed = _sealed_members(
      shard_root,
      round_index,
      expected_commit=expected_commit,
      runtime_commit=runtime_commit,
      verify_payload=True,
  )
  _require(sealed, "M15 wide observer produced no sealed records")
  unsealed_pairs = _candidate_pairs(
      live_directory, round_index, sealed_members=sealed
  )
  _require(not unsealed_pairs,
           "complete live observer records remain outside sealed shards")
  for npz_path in live_directory.glob("p38_seam_*.npz"):
    _require(npz_path.name in sealed
             or npz_path.with_suffix(".json").is_file(),
             f"orphan seam NPZ remains at final seal: {npz_path.name}")
  for npz_path in live_directory.glob("p38_tail_*.npz"):
    _require(npz_path.name in sealed
             or npz_path.with_suffix(".json").is_file(),
             f"orphan tail NPZ remains at final seal: {npz_path.name}")

  output.mkdir(parents=True, mode=0o700)
  shard_receipts = []
  copied: set[str] = set()
  for shard_dir in sorted(path for path in shard_root.iterdir() if path.is_dir()):
    if not (shard_dir / "SHARD_COMPLETE.json").is_file():
      continue
    inventory = json.loads(
        (shard_dir / "SHARD_INVENTORY.json").read_text(encoding="utf-8")
    )
    rows = _manifest(shard_dir / "SHA256SUMS")
    manifest_sha = _sha256(shard_dir / "SHA256SUMS")
    for name in inventory["files"]:
      _require(name not in copied, f"duplicate sealed record during assembly: {name}")
      _require(rows.get(name) == _sha256(shard_dir / name),
               f"sealed shard member changed before assembly: {name}")
      _link_or_copy(shard_dir / name, output / name)
      copied.add(name)
    shard_receipts.append({
        "sequence": int(inventory["sequence"]),
        "record_pairs": int(inventory["record_pairs"]),
        "payload_bytes": int(inventory["payload_bytes"]),
        "manifest_sha256": manifest_sha,
    })

  alignment = _copy_alignment(
      pre_alignment, output / "pre-alignment.jsonl", round_index
  )
  replay_records = _copy_replay(
      replay_ledger, output / "m15-replay-envelope.jsonl", round_index
  )
  ab_bytes = int(
      alignment.get("boundaries", {})
      .get("S_decode_vs_S_prefill", {})
      .get("differing_bytes", -1)
  )
  capsule_names = []
  round_capsule = Path(f"{capsule.with_suffix('')}.round-{round_index:06d}.npz")
  capsule_source = round_capsule if round_capsule.is_file() else capsule
  if ab_bytes > 0:
    _require(capsule_source.is_file() and capsule_source.stat().st_size > 0,
             "red M15 round lacks a mismatch capsule")
    capsule_name = "mismatch-capsule.npz"
    _link_or_copy(capsule_source, output / capsule_name)
    capsule_names.append(capsule_name)
  else:
    _require(not capsule_source.exists() or capsule_source.stat().st_size == 0,
             "exact M15 round unexpectedly has a mismatch capsule")

  receipt = {
      "schema": "m15-wide-sealed-input-v1",
      "status": "PASS",
      "diagnostic_round": round_index,
      "observer_mode": observer_mode,
      "expected_source_commit": expected_commit,
      "runtime_source_commit": runtime_commit,
      "record_files": len(copied),
      "record_pairs": len(copied) // 2,
      "replay_records": replay_records,
      "capsules": capsule_names,
      "shards": shard_receipts,
  }
  (output / "ROUND_INPUT_RECEIPT.json").write_text(
      json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return receipt


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--live-directory", required=True, type=Path)
  parser.add_argument("--shard-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--round", required=True, type=int)
  parser.add_argument("--pre-alignment", required=True, type=Path)
  parser.add_argument("--capsule", required=True, type=Path)
  parser.add_argument("--replay-ledger", required=True, type=Path)
  parser.add_argument("--observer-mode", choices=("layer", "full"), required=True)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--runtime-commit", required=True)
  args = parser.parse_args()
  receipt = assemble(
      live_directory=args.live_directory,
      shard_root=args.shard_root,
      output=args.output,
      round_index=args.round,
      pre_alignment=args.pre_alignment,
      capsule=args.capsule,
      replay_ledger=args.replay_ledger,
      observer_mode=args.observer_mode,
      expected_commit=args.expected_commit,
      runtime_commit=args.runtime_commit,
  )
  print(
      "[M15.WIDE.ROUND] INPUT_READY "
      f"round={receipt['diagnostic_round']} shards={len(receipt['shards'])} "
      f"pairs={receipt['record_pairs']}"
  )
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (M15WideShardError, json.JSONDecodeError, OSError) as error:
    print(f"[M15.WIDE.ROUND] RED {error}", file=sys.stderr)
    raise SystemExit(2) from error
