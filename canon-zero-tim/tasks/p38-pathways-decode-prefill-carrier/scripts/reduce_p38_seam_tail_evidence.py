#!/usr/bin/env python3
"""Reduce one immutable P38 round to alias-audited seam and tail evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

import classify_p38_seam as seam
import reduce_p38_seam_evidence as base


class SeamTailReductionError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SeamTailReductionError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _round_number(source_gcs_uri: str) -> int:
  match = re.fullmatch(
      r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
      r"[^/]+/attempt-0/rounds/([0-9]{6})",
      source_gcs_uri.rstrip("/"),
  )
  _require(match is not None,
           "source GCS URI is outside the immutable P38 round hierarchy")
  return int(match.group(1))


def _listing_names(path: Path, source_gcs_uri: str) -> set[str]:
  _require(path.is_file(), f"object listing is absent: {path}")
  prefix = source_gcs_uri.rstrip("/") + "/"
  names = set()
  for raw in path.read_text(encoding="utf-8").splitlines():
    value = raw.strip().rstrip(":")
    _require(value.startswith(prefix),
             f"object listing escaped the source round: {value}")
    relative = value[len(prefix):]
    candidate = Path(relative)
    _require(
        relative and not candidate.is_absolute() and ".." not in candidate.parts,
        f"unsafe object-listing path: {relative}",
    )
    _require(relative not in names,
             f"duplicate object-listing path: {relative}")
    names.add(relative)
  _require(names, "object listing is empty")
  return names


def _verify_round_source(args: argparse.Namespace) -> dict[str, Any]:
  source = args.source_dir.resolve()
  _require(source.is_dir(), f"source directory is absent: {source}")
  diagnostic_round = _round_number(args.source_gcs_uri)
  _require(diagnostic_round == args.expected_diagnostic_round,
           "source URI diagnostic round differs from the contract")

  manifest = source / "SHA256SUMS"
  entries = base._manifest_entries(manifest)
  expected_paths = {relative for _, relative in entries}
  actual_paths = {
      path.relative_to(source).as_posix()
      for path in source.rglob("*")
      if path.is_file()
      and path.name not in ("ROUND_COMPLETE.json", "SHA256SUMS")
  }
  _require(actual_paths == expected_paths,
           "source round file inventory differs from SHA256SUMS")
  for expected, relative in entries:
    target = source / relative
    _require(target.is_file(), f"source manifest file is absent: {relative}")
    _require(_sha256(target) == expected,
             f"source manifest SHA failed: {relative}")

  manifest_sha = _sha256(manifest)
  _require(manifest_sha == args.expected_manifest_sha256,
           "source manifest SHA differs from the registered contract")
  complete_path = source / "ROUND_COMPLETE.json"
  inventory_path = source / "ROUND_INVENTORY.json"
  _require(complete_path.is_file(), "source ROUND_COMPLETE.json is absent")
  _require(inventory_path.is_file(), "source ROUND_INVENTORY.json is absent")
  complete = json.loads(complete_path.read_text(encoding="utf-8"))
  inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
  _require(
      complete.get("schema") == "canon-p38-round-completion-v1"
      and int(complete.get("diagnostic_round", -1)) == diagnostic_round
      and complete.get("status") == "sealed-and-verified"
      and complete.get("manifest_sha256") == manifest_sha,
      "source round completion contract drifted",
  )
  _require(complete.get("source_commit") == args.expected_source_commit,
           "source round commit differs from the registered contract")
  _require(
      inventory.get("schema") == "canon-p38-round-stage-v1"
      and int(inventory.get("diagnostic_round", -1)) == diagnostic_round,
      "source round inventory contract drifted",
  )

  seam_json = {path.stem for path in source.glob("p38_seam_*.json")}
  seam_npz = {path.stem for path in source.glob("p38_seam_*.npz")}
  tail_json = {path.stem for path in source.glob("p38_tail_*.json")}
  tail_npz = {path.stem for path in source.glob("p38_tail_*.npz")}
  _require(seam_json == seam_npz and tail_json == tail_npz,
           "source seam/tail JSON-NPZ pairing drifted")
  _require(
      len(seam_json) == int(inventory.get("seam_records", -1))
      == args.expected_seam_records,
      "source seam-record count differs from the contract",
  )
  _require(
      len(tail_json) == int(inventory.get("tail_records", -1))
      == args.expected_tail_records,
      "source tail-record count differs from the contract",
  )
  _require(len(entries) == args.expected_manifest_files,
           "source manifest file count differs from the contract")

  listing_names = _listing_names(args.object_listing, args.source_gcs_uri)
  _require(
      listing_names == expected_paths | {"ROUND_COMPLETE.json", "SHA256SUMS"},
      "source object listing differs from the sealed source inventory",
  )
  _require(len(listing_names) == args.expected_object_count,
           "source object count differs from the contract")
  return {
      "diagnostic_round": diagnostic_round,
      "manifest": manifest,
      "manifest_sha256": manifest_sha,
      "manifest_files": len(entries),
      "object_count": len(listing_names),
      "complete": complete,
      "complete_path": complete_path,
      "inventory": inventory,
      "inventory_path": inventory_path,
      "seam_records": len(seam_json),
      "tail_records": len(tail_json),
  }


def _tail_payload_sha256(
    *,
    position: int,
    source_token_id: int,
    target_id: int,
    logit_row_index: int,
    checkpoint_names: list[str],
    values: np.ndarray,
) -> str:
  return seam._tail_numeric_payload_sha256(
      position=position,
      source_token_id=source_token_id,
      target_id=target_id,
      logit_row_index=logit_row_index,
      checkpoint_names=checkpoint_names,
      values=values,
  )


def _scan_tail_records(
    source: Path,
    required: set[tuple[int, bytes, str]],
) -> tuple[
    dict[tuple[int, bytes, str], list[dict[str, Any]]],
    dict[int, tuple[Path, Path]],
    int,
]:
  matches = {key: [] for key in required}
  matching_records: dict[int, tuple[Path, Path]] = {}
  json_paths = sorted(source.glob("p38_tail_*.json"))
  _require(json_paths, "source round has no P38 terminal-tail JSON records")
  seen_indices = set()
  expected_arrays = {
      "row_indices", "positions", "token_ids", "request_ordinals",
      "token_prefix_sha256", "logit_row_indices", "target_ids",
      "tail_values",
  }
  for json_path in json_paths:
    record = json.loads(json_path.read_text(encoding="utf-8"))
    _require(record.get("schema") == "p38-tail-values-v1",
             f"invalid terminal-tail schema: {json_path.name}")
    index = int(record.get("record_index", -1))
    _require(index >= 0 and index not in seen_indices,
             f"invalid or duplicate terminal-tail index: {json_path.name}")
    seen_indices.add(index)
    _require(json_path.name == f"p38_tail_{index:06d}.json",
             f"terminal-tail JSON identity drifted: {json_path.name}")
    arm = record.get("arm")
    diagnostic_round = int(record.get("diagnostic_round", -1))
    _require(arm in ("A", "B") and 0 <= diagnostic_round < 8,
             f"terminal-tail provenance drifted: {json_path.name}")
    if not any(
        key[0] == diagnostic_round and key[2] == arm for key in required
    ):
      continue
    npz_path = source / f"p38_tail_{index:06d}.npz"
    _require(npz_path.is_file(), f"terminal-tail NPZ is absent: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"terminal-tail NPZ SHA failed: {npz_path.name}")
    with np.load(npz_path, allow_pickle=False) as archive:
      _require(set(archive.files) == expected_arrays,
               f"terminal-tail array inventory drifted: {npz_path.name}")
      arrays = {name: np.asarray(archive[name]) for name in archive.files}
    rows = arrays["row_indices"].reshape(-1)
    positions = arrays["positions"].reshape(-1)
    source_tokens = arrays["token_ids"].reshape(-1)
    request_ordinals = arrays["request_ordinals"].reshape(-1)
    hashes = arrays["token_prefix_sha256"].reshape(-1)
    logit_rows = arrays["logit_row_indices"].reshape(-1)
    target_ids = arrays["target_ids"].reshape(-1)
    values = arrays["tail_values"]
    checkpoint_names = [str(value) for value in record.get(
        "checkpoint_names", ())]
    _require(tuple(checkpoint_names) == seam._TAIL_CHECKPOINTS,
             f"terminal-tail checkpoints drifted: {json_path.name}")
    _require(
        rows.size == positions.size == source_tokens.size
        == request_ordinals.size == hashes.size == logit_rows.size
        == target_ids.size == values.shape[0]
        and values.shape[1:] == (len(checkpoint_names),),
        f"terminal-tail row geometry drifted: {npz_path.name}",
    )
    requests = record.get("requests", [])
    _require(isinstance(requests, list),
             f"terminal-tail request metadata drifted: {json_path.name}")
    hit = False
    for row_offset, prefix in enumerate(hashes):
      key = base._key(diagnostic_round, bytes(prefix), arm)
      if key not in matches:
        continue
      request_ordinal = int(request_ordinals[row_offset])
      request = None
      if requests:
        _require(0 <= request_ordinal < len(requests),
                 f"terminal-tail request ordinal drifted: {npz_path.name}")
        _require(isinstance(requests[request_ordinal], dict),
                 f"terminal-tail request entry drifted: {json_path.name}")
        request = requests[request_ordinal]
      value_row = np.asarray(values[row_offset])
      matches[key].append({
          "record_index": index,
          "row_offset": row_offset,
          "row_index": int(rows[row_offset]),
          "position": int(positions[row_offset]),
          "source_token_id": int(source_tokens[row_offset]),
          "target_id": int(target_ids[row_offset]),
          "logit_row_index": int(logit_rows[row_offset]),
          "request_ordinal": request_ordinal,
          "call_index": int(record.get("call_index", -1)),
          "program_path": record.get("program_path"),
          "request": request,
          "checkpoint_names": checkpoint_names,
          "tail_value_sha256": base._array_sha256(value_row),
          "numeric_payload_sha256": _tail_payload_sha256(
              position=int(positions[row_offset]),
              source_token_id=int(source_tokens[row_offset]),
              target_id=int(target_ids[row_offset]),
              logit_row_index=int(logit_rows[row_offset]),
              checkpoint_names=checkpoint_names,
              values=value_row,
          ),
      })
      hit = True
    if hit:
      matching_records[index] = (json_path, npz_path)
  return matches, matching_records, len(json_paths)


def _resolve_matches(
    matches: dict[tuple[int, bytes, str], list[dict[str, Any]]],
) -> dict[str, Any]:
  entries = []
  unmatched = []
  conflicts = []
  aliases = []
  selected_indices = set()
  for key in sorted(matches, key=lambda value: (value[0], value[1], value[2])):
    candidates = sorted(matches[key], key=lambda value: (
        value["record_index"], value["row_offset"]))
    if not candidates:
      resolution = "missing"
      selected = None
      unmatched.append(base._key_json(key))
    else:
      payloads = {value["numeric_payload_sha256"] for value in candidates}
      if len(candidates) == 1:
        resolution = "unique"
      elif len(payloads) == 1:
        resolution = "equivalent_alias"
      else:
        resolution = "payload_conflict"
      selected = candidates[0] if resolution in (
          "unique", "equivalent_alias") else None
      if selected is not None:
        selected_indices.add(int(selected["record_index"]))
      if resolution == "equivalent_alias":
        aliases.append({
            **base._key_json(key),
            "candidate_count": len(candidates),
            "selected": selected,
            "aliases": candidates[1:],
        })
      elif resolution == "payload_conflict":
        conflicts.append({
            **base._key_json(key),
            "candidate_count": len(candidates),
            "candidates": candidates,
        })
    entries.append({
        **base._key_json(key),
        "resolution": resolution,
        "selected": selected,
        "candidates": candidates,
    })
  return {
      "join_entries": entries,
      "unmatched_keys": unmatched,
      "payload_conflict_keys": conflicts,
      "equivalent_alias_keys": aliases,
      "selected_record_indices": sorted(selected_indices),
      "matched_keys": sum(
          entry["resolution"] in ("unique", "equivalent_alias")
          for entry in entries),
      "selection_complete": not unmatched and not conflicts,
  }


def _copy_record_set(
    records: dict[int, tuple[Path, Path]],
    destination: Path,
    *,
    kind: str,
    selected_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
  files = []
  for index in sorted(records):
    if selected_indices is not None and index not in selected_indices:
      continue
    for path in records[index]:
      info = base._copy_file(path, destination / path.name)
      info["path"] = f"{destination.name}/{path.name}"
      info["source_path"] = path.name
      info["record_index"] = index
      info["observer_kind"] = kind
      files.append(info)
  return files


def reduce(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
  source = args.source_dir.resolve()
  output = args.output_dir.resolve()
  _require(not output.exists(), f"output directory already exists: {output}")
  _require(re.fullmatch(r"[0-9a-f]{40}", args.analysis_source_commit) is not None,
           "analysis source commit must be a full lowercase SHA")
  source_fact = _verify_round_source(args)
  output.mkdir(parents=True)
  records_dir = output / "records"
  candidates_dir = output / "candidates"
  capsules_dir = output / "capsules"
  records_dir.mkdir()
  candidates_dir.mkdir()
  capsules_dir.mkdir()

  capsule = args.capsule.resolve()
  _require(capsule.is_file() and capsule.parent == source,
           "immutable mismatch capsule must come from the source round")
  red_points = seam._red_points([capsule])
  _require(len(red_points) == args.expected_red_points,
           "capsule red-point count differs from the contract")
  required = {
      base._key(point["diagnostic_round"], point["token_prefix_sha256"], arm)
      for point in red_points for arm in ("A", "B")
  }
  _require({key[0] for key in required} == {source_fact["diagnostic_round"]},
           "capsule diagnostic round differs from the source round")

  seam_matches, seam_records, source_seam_count = base._scan_records(
      source, args.mode, required)
  tail_matches, tail_records, source_tail_count = _scan_tail_records(
      source, required)
  _require(source_seam_count == source_fact["seam_records"],
           "scanned seam-record count differs from source inventory")
  _require(source_tail_count == source_fact["tail_records"],
           "scanned tail-record count differs from source inventory")
  seam_resolution = _resolve_matches(seam_matches)
  tail_resolution = _resolve_matches(tail_matches)
  selection_complete = (
      seam_resolution["selection_complete"]
      and tail_resolution["selection_complete"])

  candidate_files = _copy_record_set(
      seam_records, candidates_dir, kind="seam")
  candidate_files.extend(_copy_record_set(
      tail_records, candidates_dir, kind="tail"))
  seam_selected = set(seam_resolution["selected_record_indices"])
  tail_selected = set(tail_resolution["selected_record_indices"])
  record_files = _copy_record_set(
      seam_records, records_dir, kind="seam", selected_indices=seam_selected)
  tail_record_files = _copy_record_set(
      tail_records, records_dir, kind="tail", selected_indices=tail_selected)

  capsule_info = base._copy_file(capsule, capsules_dir / capsule.name)
  capsule_info["path"] = f"capsules/{capsule.name}"
  capsule_info["source_path"] = capsule.name
  capsule_info["diagnostic_round"] = source_fact["diagnostic_round"]
  capsule_path = capsules_dir / capsule.name

  base._copy_file(args.object_listing.resolve(), output / "OBJECT_LISTING.txt")
  base._copy_file(source_fact["complete_path"],
                  output / "SOURCE_ROUND_COMPLETE.json")
  base._copy_file(source_fact["inventory_path"],
                  output / "SOURCE_ROUND_INVENTORY.json")
  base._copy_file(source_fact["manifest"], output / "SOURCE_SHA256SUMS")
  (output / "analysis_source_commit.txt").write_text(
      args.analysis_source_commit + "\n", encoding="utf-8")
  classifier_path = Path(seam.__file__).resolve()
  (output / "classifier_source.sha256").write_text(
      f"{_sha256(classifier_path)}  {classifier_path.name}\n",
      encoding="utf-8",
  )
  selection = {
      "schema": "p38-immutable-round-selection-v1",
      "selection_complete": True,
      "source_kind": "immutable-round",
      "selected_snapshot": f"{source_fact['diagnostic_round']:06d}",
      "selected_source_gcs_uri": args.source_gcs_uri.rstrip("/"),
      "selected_capsule_rounds": [source_fact["diagnostic_round"]],
      "minimum_capsule_rounds": 1,
      "listing_sha256": _sha256(args.object_listing),
      "source_manifest_sha256": source_fact["manifest_sha256"],
  }
  _write_json(output / "SNAPSHOT_SELECTION.json", selection)

  combined_unmatched = [
      {"observer": "seam", **entry}
      for entry in seam_resolution["unmatched_keys"]
  ] + [
      {"observer": "tail", **entry}
      for entry in tail_resolution["unmatched_keys"]
  ]
  combined_conflicts = [
      {"observer": "seam", **entry}
      for entry in seam_resolution["payload_conflict_keys"]
  ] + [
      {"observer": "tail", **entry}
      for entry in tail_resolution["payload_conflict_keys"]
  ]
  ambiguity = {
      "schema": "p38-seam-tail-ambiguity-audit-v1",
      "required_arm_keys": len(required),
      "selection_complete": selection_complete,
      "seam": seam_resolution,
      "tail": tail_resolution,
      "unmatched_keys": combined_unmatched,
      "payload_conflict_keys": combined_conflicts,
      "interpretation": (
          "Aliases are admitted only when every registered provenance field "
          "and numerical payload byte is identical. Conflicts retain all "
          "candidate source files and remain fail-closed."
      ),
  }
  _write_json(output / "AMBIGUITY_AUDIT.json", ambiguity)

  completed_rounds, terminal_markers = base._count_completed_rounds(source)
  run_contract_complete = (
      completed_rounds == args.expected_rounds and terminal_markers == 1)
  manifest = {
      "schema": "p38-seam-reduction-v2",
      "status": "selected" if selection_complete else "inconclusive",
      "require_tail": True,
      "source_gcs_uri": args.source_gcs_uri.rstrip("/"),
      "source_commit": source_fact["complete"]["source_commit"],
      "analysis_source_commit": args.analysis_source_commit,
      "classifier_source_sha256": _sha256(classifier_path),
      "snapshot_selection": "SNAPSHOT_SELECTION.json",
      "snapshot_selection_sha256": _sha256(
          output / "SNAPSHOT_SELECTION.json"),
      "source_snapshot_manifest_sha256": source_fact["manifest_sha256"],
      "source_round_complete_sha256": _sha256(
          output / "SOURCE_ROUND_COMPLETE.json"),
      "source_round_inventory_sha256": _sha256(
          output / "SOURCE_ROUND_INVENTORY.json"),
      "object_listing_sha256": _sha256(output / "OBJECT_LISTING.txt"),
      "source_snapshot_files": source_fact["manifest_files"],
      "source_object_count": source_fact["object_count"],
      "source_seam_records": source_seam_count,
      "source_tail_records": source_tail_count,
      "observer_mode": args.mode,
      "expected_rounds": args.expected_rounds,
      "completed_pre_alignment_rounds": completed_rounds,
      "terminal_precheck_markers": terminal_markers,
      "run_contract_complete": run_contract_complete,
      "capsule_rounds": [source_fact["diagnostic_round"]],
      "red_points": len(red_points),
      "required_arm_keys": len(required),
      "matched_arm_keys": seam_resolution["matched_keys"],
      "matched_seam_keys": seam_resolution["matched_keys"],
      "matched_tail_keys": tail_resolution["matched_keys"],
      "selection_complete": selection_complete,
      "unmatched_keys": combined_unmatched,
      "ambiguous_keys": combined_conflicts,
      "equivalent_alias_keys": seam_resolution["equivalent_alias_keys"],
      "tail_equivalent_alias_keys": tail_resolution[
          "equivalent_alias_keys"],
      "ambiguity_audit": "AMBIGUITY_AUDIT.json",
      "join_entries": seam_resolution["join_entries"],
      "tail_join_entries": tail_resolution["join_entries"],
      "records_directory": "records",
      "candidates_directory": "candidates",
      "candidate_record_files": candidate_files,
      "selected_record_indices": seam_resolution[
          "selected_record_indices"],
      "selected_tail_record_indices": tail_resolution[
          "selected_record_indices"],
      "record_files": record_files,
      "tail_record_files": tail_record_files,
      "capsules": [capsule_info],
      "claim_ceiling": (
          "This is one analysis-grade immutable round. It does not "
          "manufacture the missing diagnostic rounds or terminal marker."
      ),
  }
  manifest_path = output / "REDUCTION_MANIFEST.json"
  _write_json(manifest_path, manifest)

  classification = None
  classifier_rc = 4
  classifier_stderr = "classifier not run: seam/tail reduction join incomplete\n"
  if selection_complete:
    try:
      classification = seam.classify(
          records_dir,
          [capsule_path],
          args.mode,
          reduction_manifest=manifest_path,
          require_tail=True,
      )
      _write_json(output / "classification.json", classification)
      classifier_rc = 0
      classifier_stderr = ""
    except (seam.SeamError, OSError, ValueError) as error:
      classifier_rc = 2
      classifier_stderr = f"{type(error).__name__}: {error}\n"
  (output / "classifier.rc").write_text(
      f"{classifier_rc}\n", encoding="utf-8")
  (output / "classifier.stdout").write_text(
      json.dumps(classification, sort_keys=True) + "\n"
      if classification is not None else "",
      encoding="utf-8",
  )
  (output / "classifier.stderr").write_text(
      classifier_stderr, encoding="utf-8")

  if not selection_complete:
    verdict = "INCONCLUSIVE_REDUCTION_JOIN"
    exit_code = 4
  elif classification is None:
    verdict = "INCONCLUSIVE_REMOTE_CLASSIFICATION"
    exit_code = 5
  elif not run_contract_complete:
    verdict = "INCONCLUSIVE_PARTIAL_RUN"
    exit_code = 0
  else:
    verdict = "PASS"
    exit_code = 0
  verdict_record = {
      "schema": "p38-seam-tail-reduction-verdict-v1",
      "verdict": verdict,
      "selection_complete": selection_complete,
      "run_contract_complete": run_contract_complete,
      "classifier_rc": classifier_rc,
      "classification": (
          classification.get("classification") if classification else None),
      "red_points": len(red_points),
      "joined_red_points": (
          classification.get("joined_red_points") if classification else 0),
      "matched_seam_keys": seam_resolution["matched_keys"],
      "matched_tail_keys": tail_resolution["matched_keys"],
  }
  _write_json(output / "verdict.json", verdict_record)
  (output / "PACKAGING.txt").write_text(
      "\n".join((
          "P38s18r2 Round 0 alias-aware seam-tail reduction v2.",
          f"source_gcs_uri: {args.source_gcs_uri.rstrip('/')}",
          f"source_seam_records: {source_seam_count}",
          f"source_tail_records: {source_tail_count}",
          f"candidate_files: {len(candidate_files)}",
          f"red_points: {len(red_points)}",
          f"matched_seam_keys: {seam_resolution['matched_keys']}",
          f"matched_tail_keys: {tail_resolution['matched_keys']}",
          f"seam_alias_keys: {len(seam_resolution['equivalent_alias_keys'])}",
          f"tail_alias_keys: {len(tail_resolution['equivalent_alias_keys'])}",
          f"payload_conflict_keys: {len(combined_conflicts)}",
          f"classifier_rc: {classifier_rc}",
          f"verdict: {verdict}",
      )) + "\n",
      encoding="utf-8",
  )
  base._write_output_manifest(output)
  total_bytes = sum(
      path.stat().st_size for path in output.rglob("*") if path.is_file())
  _require(total_bytes <= args.max_output_bytes,
           f"reduced evidence exceeds byte ceiling: {total_bytes}")
  return verdict_record, exit_code


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-dir", type=Path, required=True)
  parser.add_argument("--source-gcs-uri", required=True)
  parser.add_argument("--object-listing", type=Path, required=True)
  parser.add_argument("--capsule", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--mode", choices=("layer",), required=True)
  parser.add_argument("--analysis-source-commit", required=True)
  parser.add_argument("--expected-source-commit", required=True)
  parser.add_argument("--expected-manifest-sha256", required=True)
  parser.add_argument("--expected-diagnostic-round", type=int, required=True)
  parser.add_argument("--expected-seam-records", type=int, required=True)
  parser.add_argument("--expected-tail-records", type=int, required=True)
  parser.add_argument("--expected-object-count", type=int, required=True)
  parser.add_argument("--expected-manifest-files", type=int, required=True)
  parser.add_argument("--expected-red-points", type=int, required=True)
  parser.add_argument("--expected-rounds", type=int, default=3)
  parser.add_argument("--max-output-bytes", type=int, default=180_000_000)
  args = parser.parse_args()
  try:
    verdict, exit_code = reduce(args)
  except (SeamTailReductionError, base.ReductionError, seam.SeamError,
          OSError, ValueError) as error:
    print(f"[P38.SEAM_TAIL.REDUCE] REFUSING: {error}", file=sys.stderr)
    return 2
  print(
      "[P38.SEAM_TAIL.REDUCE] COMPLETE "
      f"verdict={verdict['verdict']} red_points={verdict['red_points']} "
      f"matched_seam_keys={verdict['matched_seam_keys']} "
      f"matched_tail_keys={verdict['matched_tail_keys']} "
      f"joined_red_points={verdict['joined_red_points']} "
      f"output={args.output_dir}",
      flush=True,
  )
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
