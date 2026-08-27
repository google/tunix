#!/usr/bin/env python3
"""Stage one bounded immutable shard of completed M15 observer records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


_PREFIX_SCHEMAS = {
    "p38_seam": "p38-seam-fingerprint-v1",
    "p38_tail": "p38-tail-values-v1",
}


class M15WideShardError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise M15WideShardError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _manifest(path: Path) -> dict[str, str]:
  _require(path.is_file() and path.stat().st_size > 0,
           f"shard manifest is absent or empty: {path}")
  rows: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(separator == "  " and len(digest) == 64,
             f"invalid shard manifest row: {line!r}")
    _require(name and "/" not in name and name not in rows,
             f"unsafe or duplicate shard member: {name!r}")
    rows[name] = digest
  return rows


def _sealed_members(
    shard_root: Path,
    round_index: int,
    *,
    expected_commit: str,
    runtime_commit: str,
    verify_payload: bool,
) -> set[str]:
  sealed: set[str] = set()
  if not shard_root.exists():
    return sealed
  for shard_dir in sorted(path for path in shard_root.iterdir() if path.is_dir()):
    completion_path = shard_dir / "SHARD_COMPLETE.json"
    if not completion_path.is_file():
      continue
    inventory_path = shard_dir / "SHARD_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    _require(inventory.get("schema") == "m15-wide-observer-shard-v1",
             f"shard inventory schema drifted: {inventory_path}")
    _require(int(inventory.get("diagnostic_round", -1)) == round_index,
             f"shard round drifted: {inventory_path}")
    sequence = int(inventory.get("sequence", -1))
    _require(sequence >= 0 and shard_dir.name == f"{sequence:06d}",
             f"shard sequence/path drifted: {inventory_path}")
    rows = _manifest(shard_dir / "SHA256SUMS")
    expected = set(inventory.get("files", ())) | {"SHARD_INVENTORY.json"}
    _require(set(rows) == expected,
             f"shard manifest inventory drifted: {inventory_path}")
    if verify_payload:
      for name, digest in rows.items():
        member = shard_dir / name
        _require(member.is_file() and _sha256(member) == digest,
                 f"local sealed shard member failed SHA: {member}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _require(
        completion.get("schema") == "m15-wide-observer-shard-completion-v1"
        and completion.get("status") == "sealed-uploaded-verified"
        and completion.get("claim_ceiling")
        == "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
        and int(completion.get("sequence", -1)) == int(inventory["sequence"])
        and int(completion.get("diagnostic_round", -1)) == round_index
        and int(completion.get("record_pairs", -1)) == int(
            inventory["record_pairs"]
        )
        and completion.get("manifest_sha256") == _sha256(
            shard_dir / "SHA256SUMS"
        )
        and completion.get("expected_source_commit") == expected_commit
        and completion.get("runtime_source_commit") == runtime_commit,
        f"shard completion receipt drifted: {completion_path}",
    )
    for name in inventory.get("files", ()):
      _require(name not in sealed, f"observer member appears in two shards: {name}")
      sealed.add(name)
  return sealed


def _candidate_pairs(
    directory: Path,
    round_index: int,
    *,
    sealed_members: set[str],
) -> list[dict[str, Any]]:
  pairs = []
  for prefix, schema in _PREFIX_SCHEMAS.items():
    for json_path in sorted(directory.glob(f"{prefix}_*.json")):
      npz_path = json_path.with_suffix(".npz")
      sealed_pair = (
          json_path.name in sealed_members,
          npz_path.name in sealed_members,
      )
      _require(sealed_pair[0] == sealed_pair[1],
               f"only half an observer pair is sealed: "
               f"{(json_path.name, npz_path.name)}")
      # The sealed copy is authoritative. Avoid rehashing immutable multi-GiB
      # history on every periodic snapshot tick.
      if sealed_pair[0]:
        continue
      record = json.loads(json_path.read_text(encoding="utf-8"))
      if int(record.get("diagnostic_round", -1)) != round_index:
        continue
      index = int(record.get("record_index", -1))
      _require(
          record.get("schema") == schema
          and index >= 0
          and json_path.name == f"{prefix}_{index:06d}.json",
          f"invalid M15 observer record: {json_path.name}",
      )
      _require(npz_path.is_file() and npz_path.stat().st_size > 0,
               f"published observer JSON lacks its NPZ: {npz_path}")
      _require(_sha256(npz_path) == record.get("npz_sha256"),
               f"observer NPZ SHA failed: {npz_path}")
      pairs.append({
          "prefix": prefix,
          "record_index": index,
          "json": json_path,
          "npz": npz_path,
          "bytes": json_path.stat().st_size + npz_path.stat().st_size,
      })
  return sorted(pairs, key=lambda item: (item["prefix"], item["record_index"]))


def _link_or_copy(source: Path, destination: Path) -> None:
  shutil.copyfile(source, destination)


def stage(
    *,
    directory: Path,
    shard_root: Path,
    output: Path,
    round_index: int,
    sequence: int,
    max_records: int,
    max_bytes: int,
    expected_commit: str,
    runtime_commit: str,
) -> dict[str, Any] | None:
  _require(directory.is_dir(), f"observer directory is absent: {directory}")
  _require(0 <= round_index < 8, "diagnostic round must be in [0,8)")
  _require(sequence >= 0, "shard sequence must be nonnegative")
  _require(1 <= max_records <= 256, "shard record cap must be in [1,256]")
  _require(1024 * 1024 <= max_bytes <= 512 * 1024 * 1024,
           "shard byte cap must be in [1MiB,512MiB]")
  _require(not output.exists(), f"shard output already exists: {output}")

  _require(expected_commit == runtime_commit and len(expected_commit) == 40,
           "runtime source does not match the rendered source")
  sealed = _sealed_members(
      shard_root,
      round_index,
      expected_commit=expected_commit,
      runtime_commit=runtime_commit,
      verify_payload=False,
  )
  selected = []
  selected_bytes = 0
  for pair in _candidate_pairs(
      directory, round_index, sealed_members=sealed
  ):
    _require(pair["bytes"] <= max_bytes,
             "single observer pair exceeds shard byte cap: "
             f"{(pair['json'].name, pair['npz'].name)}")
    if selected and (
        len(selected) >= max_records
        or selected_bytes + pair["bytes"] > max_bytes
    ):
      break
    selected.append(pair)
    selected_bytes += pair["bytes"]
  if not selected:
    return None

  output.mkdir(parents=True, mode=0o700)
  files = []
  records = []
  for pair in selected:
    for key in ("json", "npz"):
      source = pair[key]
      destination = output / source.name
      _link_or_copy(source, destination)
      files.append(source.name)
    records.append({
        "prefix": pair["prefix"],
        "record_index": pair["record_index"],
        "bytes": pair["bytes"],
    })
  files.sort()
  inventory = {
      "schema": "m15-wide-observer-shard-v1",
      "status": "STAGED",
      "diagnostic_round": round_index,
      "sequence": sequence,
      "record_pairs": len(records),
      "payload_bytes": selected_bytes,
      "files": files,
      "records": records,
  }
  inventory_path = output / "SHARD_INVENTORY.json"
  inventory_path.write_text(
      json.dumps(inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  manifest_names = sorted([*files, inventory_path.name])
  (output / "SHA256SUMS").write_text(
      "".join(f"{_sha256(output / name)}  {name}\n" for name in manifest_names),
      encoding="ascii",
  )
  rows = _manifest(output / "SHA256SUMS")
  _require(set(rows) == set(manifest_names), "staged shard manifest drifted")
  for name, digest in rows.items():
    _require(_sha256(output / name) == digest,
             f"staged shard member failed SHA: {name}")
  return inventory


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--shard-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--round", required=True, type=int)
  parser.add_argument("--sequence", required=True, type=int)
  parser.add_argument("--max-records", default=32, type=int)
  parser.add_argument("--max-bytes", default=256 * 1024 * 1024, type=int)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--runtime-commit", required=True)
  args = parser.parse_args()
  result = stage(
      directory=args.directory,
      shard_root=args.shard_root,
      output=args.output,
      round_index=args.round,
      sequence=args.sequence,
      max_records=args.max_records,
      max_bytes=args.max_bytes,
      expected_commit=args.expected_commit,
      runtime_commit=args.runtime_commit,
  )
  if result is None:
    print("[M15.WIDE.SHARD] EMPTY")
    return 3
  print(
      "[M15.WIDE.SHARD] STAGED "
      f"sequence={result['sequence']} pairs={result['record_pairs']} "
      f"bytes={result['payload_bytes']}"
  )
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except M15WideShardError as error:
    print(f"[M15.WIDE.SHARD] RED {error}", file=sys.stderr)
    raise SystemExit(2) from error
