#!/usr/bin/env python3
"""Create a byte-preserving, red-point-complete P38 seam evidence subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

import classify_p38_seam as seam


class ReductionError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ReductionError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
  _require(path.is_file(), f"source SHA256SUMS is absent: {path}")
  entries = []
  for line_number, raw in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    parts = raw.split(maxsplit=1)
    _require(len(parts) == 2 and len(parts[0]) == 64,
             f"invalid source manifest line {line_number}")
    relative = parts[1].lstrip("*")
    candidate = Path(relative)
    _require(
        relative
        and not candidate.is_absolute()
        and ".." not in candidate.parts,
        f"unsafe source manifest path: {relative}",
    )
    entries.append((parts[0], relative))
  _require(entries, "source SHA256SUMS is empty")
  return entries


def _verify_source_manifest(source: Path) -> tuple[Path, int]:
  manifest = source / "SHA256SUMS"
  entries = _manifest_entries(manifest)
  expected_paths = {relative for _, relative in entries}
  actual_paths = {
      path.relative_to(source).as_posix()
      for path in source.rglob("*")
      if path.is_file() and path.name not in ("LIVE.json", "SHA256SUMS")
  }
  _require(actual_paths == expected_paths,
           "source snapshot file inventory differs from SHA256SUMS")
  for expected, relative in entries:
    target = source / relative
    _require(target.is_file(), f"source manifest file is absent: {relative}")
    _require(_sha256(target) == expected,
             f"source manifest SHA failed: {relative}")
  return manifest, len(entries)


def _key(round_index: int, prefix: bytes, arm: str) -> tuple[int, bytes, str]:
  return int(round_index), bytes(prefix), arm


def _key_json(key: tuple[int, bytes, str]) -> dict[str, Any]:
  return {
      "diagnostic_round": int(key[0]),
      "token_prefix_sha256": key[1].decode("ascii"),
      "arm": key[2],
  }


def _scan_records(
    source: Path,
    mode: str,
    required: set[tuple[int, bytes, str]],
) -> tuple[
    dict[tuple[int, bytes, str], list[int]],
    dict[int, tuple[Path, Path]],
    int,
]:
  matches = {key: [] for key in required}
  selected: dict[int, tuple[Path, Path]] = {}
  json_paths = sorted(source.glob("p38_seam_*.json"))
  _require(json_paths, "source snapshot has no P38 seam JSON records")
  seen_indices = set()
  for json_path in json_paths:
    record = json.loads(json_path.read_text(encoding="utf-8"))
    _require(record.get("schema") == "p38-seam-fingerprint-v1",
             f"invalid seam schema: {json_path.name}")
    _require(record.get("observer_mode") == mode,
             f"seam mode drifted: {json_path.name}")
    index = int(record.get("record_index", -1))
    _require(index >= 0 and index not in seen_indices,
             f"invalid or duplicate seam record index: {json_path.name}")
    seen_indices.add(index)
    _require(json_path.name == f"p38_seam_{index:06d}.json",
             f"seam JSON identity drifted: {json_path.name}")
    arm = record.get("arm")
    diagnostic_round = int(record.get("diagnostic_round", -1))
    _require(arm in ("A", "B") and 0 <= diagnostic_round < 8,
             f"seam provenance drifted: {json_path.name}")
    if not any(
        key[0] == diagnostic_round and key[2] == arm for key in required
    ):
      continue
    npz_path = source / f"p38_seam_{index:06d}.npz"
    _require(npz_path.is_file(), f"seam NPZ is absent: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"seam NPZ SHA failed: {npz_path.name}")
    with np.load(npz_path, allow_pickle=False) as archive:
      _require("token_prefix_sha256" in archive.files,
               f"seam prefix hashes are absent: {npz_path.name}")
      hashes = np.asarray(archive["token_prefix_sha256"]).reshape(-1)
    hit = False
    for prefix in hashes:
      candidate = _key(diagnostic_round, bytes(prefix), arm)
      if candidate in matches:
        matches[candidate].append(index)
        hit = True
    if hit:
      selected[index] = (json_path, npz_path)
  return matches, selected, len(json_paths)


def _copy_file(source: Path, target: Path) -> dict[str, Any]:
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, target)
  shutil.copystat(source, target)
  _require(_sha256(source) == _sha256(target),
           f"byte-preserving copy failed: {source.name}")
  return {
      "path": target.as_posix(),
      "sha256": _sha256(target),
      "bytes": target.stat().st_size,
  }


def _count_completed_rounds(source: Path) -> tuple[int, int]:
  pre_alignment = source / "pre-alignment.jsonl"
  run_log = source / "run.log"
  report_lines = 0
  if pre_alignment.is_file():
    for raw in pre_alignment.read_text(encoding="utf-8").splitlines():
      if raw.strip():
        json.loads(raw)
        report_lines += 1
  terminal_markers = 0
  if run_log.is_file():
    terminal_markers = run_log.read_text(
        encoding="utf-8", errors="replace").count(
            "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD")
  return report_lines, terminal_markers


def _write_output_manifest(output: Path) -> None:
  manifest = output / "SHA256SUMS"
  lines = []
  for path in sorted(output.rglob("*")):
    if path.is_file() and path != manifest:
      lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
  manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reduce(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  source = args.source_dir.resolve()
  output = args.output_dir.resolve()
  _require(source.is_dir(), f"source directory is absent: {source}")
  _require(
      args.source_gcs_uri.startswith(
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/")
      and "/attempt-0/live/" in args.source_gcs_uri,
      "source GCS URI is outside the registered P38 live-snapshot root",
  )
  _require(not output.exists(), f"output directory already exists: {output}")
  output.mkdir(parents=True)
  selected_dir = output / "selected"
  capsules_dir = output / "capsules"
  selected_dir.mkdir()
  capsules_dir.mkdir()

  source_manifest, source_file_count = _verify_source_manifest(source)
  source_manifest_sha = _sha256(source_manifest)
  capsules = [path.resolve() for path in args.capsule]
  _require(capsules and all(path.is_file() for path in capsules),
           "one or more immutable round capsules are absent")
  _require(all(path.parent == source for path in capsules),
           "immutable round capsules must come from the source snapshot")
  red_points = seam._red_points(capsules)
  required = {
      _key(point["diagnostic_round"], point["token_prefix_sha256"], arm)
      for point in red_points for arm in ("A", "B")
  }
  matches, selected_records, source_record_count = _scan_records(
      source, args.mode, required)
  unmatched = sorted(
      (key for key, values in matches.items() if not values),
      key=lambda value: (value[0], value[1], value[2]),
  )
  ambiguous = sorted(
      (key for key, values in matches.items() if len(values) > 1),
      key=lambda value: (value[0], value[1], value[2]),
  )
  selection_complete = not unmatched and not ambiguous

  selected_files = []
  if selection_complete:
    selected_indices = sorted({values[0] for values in matches.values()})
  else:
    selected_indices = sorted(selected_records)
  for index in selected_indices:
    json_path, npz_path = selected_records[index]
    for path in (json_path, npz_path):
      info = _copy_file(path, selected_dir / path.name)
      info["path"] = f"selected/{path.name}"
      info["source_path"] = path.name
      info["record_index"] = index
      selected_files.append(info)

  capsule_files = []
  capsule_rounds = []
  for path in capsules:
    with np.load(path, allow_pickle=False) as archive:
      metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
    diagnostic_round = int(metadata.get("diagnostic_round", -1))
    _require(diagnostic_round not in capsule_rounds,
             f"duplicate capsule round: {diagnostic_round}")
    capsule_rounds.append(diagnostic_round)
    info = _copy_file(path, capsules_dir / path.name)
    info["path"] = f"capsules/{path.name}"
    info["source_path"] = path.name
    info["diagnostic_round"] = diagnostic_round
    capsule_files.append(info)

  live_path = source / "LIVE.json"
  _require(live_path.is_file(), "source live snapshot has no LIVE.json")
  _copy_file(live_path, output / "SOURCE_LIVE.json")
  _copy_file(source_manifest, output / "SOURCE_SHA256SUMS")
  completed_rounds, terminal_markers = _count_completed_rounds(source)
  run_contract_complete = (
      completed_rounds == args.expected_rounds and terminal_markers == 1)
  reduction_manifest = {
      "schema": "p38-seam-reduction-v1",
      "status": "selected" if selection_complete else "inconclusive",
      "source_gcs_uri": args.source_gcs_uri.rstrip("/"),
      "source_snapshot_manifest_sha256": source_manifest_sha,
      "source_snapshot_files": source_file_count,
      "source_seam_records": source_record_count,
      "observer_mode": args.mode,
      "expected_rounds": args.expected_rounds,
      "completed_pre_alignment_rounds": completed_rounds,
      "terminal_precheck_markers": terminal_markers,
      "run_contract_complete": run_contract_complete,
      "capsule_rounds": sorted(capsule_rounds),
      "red_points": len(red_points),
      "required_arm_keys": len(required),
      "matched_arm_keys": sum(1 for values in matches.values()
                              if len(values) == 1),
      "selection_complete": selection_complete,
      "unmatched_keys": [_key_json(key) for key in unmatched],
      "ambiguous_keys": [
          {**_key_json(key), "record_indices": matches[key]}
          for key in ambiguous
      ],
      "selected_directory": "selected",
      "selected_record_indices": selected_indices,
      "selected_files": selected_files,
      "capsules": capsule_files,
      "claim_ceiling": (
          "This is a byte-preserving derived subset of a live snapshot. It "
          "does not manufacture missing diagnostic rounds or terminal markers."
      ),
  }
  reduction_path = output / "REDUCTION_MANIFEST.json"
  _write_json(reduction_path, reduction_manifest)

  classification = None
  if selection_complete:
    reduced_capsules = [capsules_dir / path.name for path in capsules]
    classification = seam.classify(
        selected_dir,
        reduced_capsules,
        args.mode,
        reduction_manifest=reduction_path,
    )
    _write_json(output / "classification.json", classification)

  if not selection_complete:
    verdict = "INCONCLUSIVE_REDUCTION_JOIN"
    exit_code = 4
  elif not run_contract_complete:
    verdict = "INCONCLUSIVE_PARTIAL_RUN"
    exit_code = 0
  else:
    verdict = "PASS"
    exit_code = 0
  verdict_record = {
      "schema": "p38-seam-reduction-verdict-v1",
      "verdict": verdict,
      "selection_complete": selection_complete,
      "run_contract_complete": run_contract_complete,
      "classification": (
          classification.get("classification") if classification else None),
      "red_points": len(red_points),
      "joined_red_points": (
          classification.get("joined_red_points") if classification else 0),
  }
  _write_json(output / "verdict.json", verdict_record)
  (output / "PACKAGING.txt").write_text(
      "\n".join((
          "p38 seam reduction 2026-08 v1",
          f"source_gcs_uri: {args.source_gcs_uri.rstrip('/')}",
          f"source_seam_records: {source_record_count}",
          f"selected_records: {len(selected_indices)}",
          f"red_points: {len(red_points)}",
          f"selection_complete: {str(selection_complete).lower()}",
          f"run_contract_complete: {str(run_contract_complete).lower()}",
          f"verdict: {verdict}",
      )) + "\n",
      encoding="utf-8",
  )
  total_bytes = sum(
      path.stat().st_size for path in output.rglob("*") if path.is_file())
  _require(total_bytes <= args.max_output_bytes,
           f"reduced evidence exceeds byte ceiling: {total_bytes}")
  _write_output_manifest(output)
  return verdict_record, exit_code


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--source-gcs-uri", required=True)
  parser.add_argument("--capsule", type=Path, action="append", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--mode", choices=("layer", "full"), required=True)
  parser.add_argument("--expected-rounds", type=int, default=3)
  parser.add_argument("--max-output-bytes", type=int, default=90_000_000)
  args = parser.parse_args()
  try:
    verdict, exit_code = reduce(args)
  except (ReductionError, seam.SeamError, ValueError, OSError) as error:
    print(f"[P38.REDUCE] REFUSING: {error}", file=sys.stderr)
    return 2
  print(
      "[P38.REDUCE] COMPLETE "
      f"verdict={verdict['verdict']} red_points={verdict['red_points']} "
      f"joined_red_points={verdict['joined_red_points']} "
      f"output={args.output_dir}",
      flush=True,
  )
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
