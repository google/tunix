#!/usr/bin/env python3
"""Audit immutable P38s22 evidence beside GCS and emit a compact receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any

import numpy as np


SCHEMA = "p38s22-offsite-audit-contract-v1"
RESULT_SCHEMA = "p38s22-offsite-audit-v1"
_SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class AuditError(ValueError):
  """Fail-closed P38s22 evidence error."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise AuditError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0,
           f"required JSON is absent or empty: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise AuditError(f"invalid JSON: {path}") from exc
  _require(isinstance(value, dict), f"JSON root is not an object: {path}")
  return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
  _require(path.is_file() and path.stat().st_size > 0,
           f"required {label} JSONL is absent or empty: {path}")
  records: list[dict[str, Any]] = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as exc:
      raise AuditError(
          f"invalid {label} JSONL line {line_number}: {path}"
      ) from exc
    _require(isinstance(record, dict),
             f"{label} JSONL line {line_number} is not an object: {path}")
    records.append(record)
  _require(records, f"required {label} JSONL has no records: {path}")
  return records


def _load_manifest(
    path: Path, *, require_sorted: bool = True
) -> list[tuple[str, str]]:
  _require(path.is_file(), f"SHA256SUMS is absent: {path}")
  records: list[tuple[str, str]] = []
  seen: set[str] = set()
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    match = _SHA_LINE.fullmatch(line)
    _require(match is not None,
             f"invalid SHA256SUMS line {line_number}: {line!r}")
    digest, name = match.groups()
    relative = PurePosixPath(name)
    _require(
        not relative.is_absolute()
        and name not in (".", "..", "SHA256SUMS")
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"unsafe manifest member: {name!r}",
    )
    _require(name not in seen, f"duplicate manifest member: {name}")
    seen.add(name)
    records.append((name, digest))
  _require(records, f"SHA256SUMS is empty: {path}")
  if require_sorted:
    _require([name for name, _ in records] == sorted(seen),
             f"SHA256SUMS is not sorted: {path}")
  return records


def _verify_manifest(
    root: Path, manifest: Path, *, require_sorted: bool = True
) -> list[str]:
  records = _load_manifest(manifest, require_sorted=require_sorted)
  for name, expected in records:
    candidate = root / name
    _require(candidate.is_file() and not candidate.is_symlink(),
             f"manifest member is absent or unsafe: {name}")
    _require(_sha256(candidate) == expected,
             f"manifest SHA failed: {name}")
  return [name for name, _ in records]


def _load_archive_module(script_dir: Path):
  module_path = script_dir / "p38_evidence_archive.py"
  spec = importlib.util.spec_from_file_location(
      "p38s22_evidence_archive", module_path)
  _require(spec is not None and spec.loader is not None,
           "cannot load deterministic evidence archive tool")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _byte_diff(left: np.ndarray, right: np.ndarray) -> int:
  left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(-1)
  right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(-1)
  return int(np.count_nonzero(left_bytes != right_bytes))


def _capsule_counts(path: Path) -> dict[str, Any]:
  with np.load(path, allow_pickle=False) as archive:
    required = {"action_mask", "s_decode", "s_prefill", "t_old"}
    _require(required <= set(archive.files),
             f"capsule array inventory is incomplete: {path.name}")
    mask = np.asarray(archive["action_mask"], dtype=np.bool_)
    a = np.asarray(archive["s_decode"])[mask]
    b = np.asarray(archive["s_prefill"])[mask]
    c = np.asarray(archive["t_old"])[mask]
  _require(a.dtype == b.dtype == c.dtype == np.dtype("float32"),
           f"capsule logprob dtype drifted: {path.name}")
  ab_mask = a != b
  bc_mask = b != c
  return {
      "selected_action_elements": int(a.size),
      "a_b_differing_elements": int(np.count_nonzero(ab_mask)),
      "a_b_differing_bytes": _byte_diff(a, b),
      "a_b_max_abs": (
          float(np.max(np.abs(a[ab_mask] - b[ab_mask])))
          if np.any(ab_mask) else 0.0
      ),
      "b_c_differing_elements": int(np.count_nonzero(bc_mask)),
      "b_c_differing_bytes": _byte_diff(b, c),
      "b_c_max_abs": (
          float(np.max(np.abs(b[bc_mask] - c[bc_mask])))
          if np.any(bc_mask) else 0.0
      ),
  }


def _boundary(record: dict[str, Any], name: str) -> dict[str, Any]:
  boundaries = record.get("boundaries")
  _require(isinstance(boundaries, dict) and isinstance(boundaries.get(name), dict),
           f"pre-alignment boundary is absent: {name}")
  return boundaries[name]


def _observer_pair_count(
    root: Path,
    prefix: str,
    schema: str,
    round_index: int,
) -> int:
  json_paths = sorted(root.glob(f"{prefix}_*.json"))
  npz_paths = sorted(root.glob(f"{prefix}_*.npz"))
  expected_npz_names = {path.with_suffix(".npz").name for path in json_paths}
  _require(expected_npz_names == {path.name for path in npz_paths},
           f"round {round_index} {prefix} observer JSON/NPZ inventory differs")
  count = 0
  for json_path in json_paths:
    record = _load_json(json_path)
    _require(record.get("schema") == schema,
             f"round {round_index} observer schema drifted: {json_path.name}")
    _require(int(record.get("diagnostic_round", -1)) == round_index,
             f"round {round_index} observer identity drifted: {json_path.name}")
    npz_path = json_path.with_suffix(".npz")
    _require(npz_path.is_file(),
             f"round {round_index} observer NPZ is absent: {npz_path.name}")
    _require(record.get("npz_sha256") == _sha256(npz_path),
             f"round {round_index} observer NPZ SHA drifted: {npz_path.name}")
    count += 1
  return count


def _copy_receipt(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, destination)


def _seal(output: Path) -> None:
  members = sorted(
      path.relative_to(output).as_posix()
      for path in output.rglob("*")
      if path.is_file() and path.name != "SHA256SUMS"
  )
  _require(members, "offsite audit produced no files")
  (output / "SHA256SUMS").write_text(
      "".join(f"{_sha256(output / name)}  {name}\n" for name in members),
      encoding="utf-8",
  )
  _verify_manifest(output, output / "SHA256SUMS")


def _terminal_status(reference: Path) -> dict[str, Any]:
  classification = reference / "p38_terminal.classification.json"
  raw_json = sorted(reference.glob("p38_terminal_*.json"))
  raw_npz = sorted(reference.glob("p38_terminal_*.npz"))
  admitted = classification.is_file() and bool(raw_json) and bool(raw_npz)
  return {
      "classification_present": classification.is_file(),
      "raw_json_records": len(raw_json),
      "raw_npz_records": len(raw_npz),
      "admitted": admitted,
      "reason": (
          "raw_terminal_inputs_and_classification_present"
          if admitted else
          "P38s22_forbade_terminal_observer_and_returned_no_raw_terminal_inputs"
      ),
  }


def audit(args: argparse.Namespace) -> dict[str, Any]:
  contract = _load_json(args.contract)
  _require(contract.get("schema") == SCHEMA, "audit contract schema drifted")
  expected_rounds = contract.get("expected_rounds")
  _require(isinstance(expected_rounds, list) and len(expected_rounds) == 3,
           "audit contract must define exactly three rounds")
  expected_by_round = {
      int(item["diagnostic_round"]): item for item in expected_rounds
  }
  _require(set(expected_by_round) == {0, 1, 2},
           "audit contract round identities drifted")

  root_files = args.source_root / "files"
  root_manifest = root_files / "SHA256SUMS"
  root_names = _verify_manifest(
      root_files, root_manifest, require_sorted=False)
  _require(_sha256(root_manifest) == contract["expected_root_manifest_sha256"],
           "root manifest SHA differs from the registered run")
  collected = _load_json(args.source_root / "COLLECTED.json")
  complete = _load_json(args.source_root / "COMPLETE.json")
  preflight = _load_json(args.source_root / "PREFLIGHT.json")
  for marker, schema, status in (
      (preflight, "canon-p38-gcs-preflight-v1", "writable"),
      (collected, "canon-p38-gcs-collection-v1", "collected"),
      (complete, "canon-p38-gcs-completion-v1", "postflight-accepted"),
  ):
    _require(marker.get("schema") == schema, f"root marker schema drifted: {schema}")
    _require(marker.get("status") == status, f"root marker status drifted: {schema}")
    _require(marker.get("source_commit") == contract["expected_source_commit"],
             f"root marker source commit drifted: {schema}")
    _require(str(marker.get("attempt")) == contract["expected_attempt"],
             f"root marker attempt drifted: {schema}")
    _require(marker.get("prefix") == contract["source_gcs_uri"],
             f"root marker source prefix drifted: {schema}")
  _require(collected.get("jobset") == contract["expected_jobset"],
           "COLLECTED jobset drifted")
  _require(complete.get("manifest_sha256") == _sha256(root_manifest),
           "COMPLETE root manifest SHA drifted")

  required_root = {
      "run.log", "pre-alignment.jsonl", "mismatch-capsule.npz",
      "mismatch-capsule.round-000000.npz",
      "mismatch-capsule.round-000001.npz",
      "mismatch-capsule.round-000002.npz",
  }
  _require(required_root <= set(root_names),
           f"root manifest lacks required files: {sorted(required_root - set(root_names))}")
  _require("terminal-classification.json" not in root_names,
           "P38s22 root unexpectedly contains terminal classification")

  pre_records: dict[int, dict[str, Any]] = {}
  for record in _load_jsonl(
      root_files / "pre-alignment.jsonl", label="root pre-alignment"
  ):
    round_index = record.get("diagnostic_round")
    _require(isinstance(round_index, int) and round_index not in pre_records,
             f"invalid or duplicate pre-alignment round: {round_index!r}")
    pre_records[round_index] = record
  _require(set(pre_records) == {0, 1, 2},
           f"pre-alignment rounds drifted: {sorted(pre_records)}")

  run_log = (root_files / "run.log").read_text(encoding="utf-8", errors="replace")
  preset = contract["expected_mm_algo_preset"]
  _require(f"[PATHTRACE] CANON_MM_ALGO on preset={preset}" in run_log,
           "CANON_MM_ALGO PATHTRACE is absent")
  _require(run_log.count("[CANON_P38] PRECHECK_ROUND_COMPLETE ") == 3,
           "run log does not contain exactly three round-complete markers")
  _require("[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0" in run_log,
           "controlled no-backward exit marker is absent")
  _require("[CANON_P38_TERMINAL_DISCRIMINATOR_INIT]" not in run_log and
           "[CANON_P38_TERMINAL_DISCRIMINATOR_RECORD]" not in run_log,
           "terminal discriminator unexpectedly ran in P38s22")
  rendered = (
      args.reference_evidence / "rendered-stock.yaml"
  ).read_text(encoding="utf-8")
  _require("name: CANON_P38_TERMINAL_DISCRIMINATOR" not in rendered,
           "rendered P38s22 contract unexpectedly enabled terminal evidence")

  archive_module = _load_archive_module(Path(__file__).resolve().parent)
  round_results = []
  total_actions = 0
  total_ab_elements = 0
  total_ab_bytes = 0
  total_bc_elements = 0
  total_bc_bytes = 0
  for round_index in range(3):
    expected = expected_by_round[round_index]
    round_dir = args.round_root / f"{round_index:06d}"
    marker = _load_json(round_dir / "ROUND_COMPLETE.json")
    manifest = round_dir / "SHA256SUMS"
    archive = round_dir / "ROUND_ARCHIVE.tar"
    _require(marker.get("schema") == "canon-p38-round-completion-v1",
             f"round {round_index} marker schema drifted")
    _require(marker.get("status") == "sealed-and-verified",
             f"round {round_index} is not sealed-and-verified")
    _require(marker.get("source_commit") == contract["expected_source_commit"],
             f"round {round_index} source commit drifted")
    _require(str(marker.get("attempt")) == contract["expected_attempt"],
             f"round {round_index} attempt drifted")
    _require(int(marker.get("diagnostic_round", -1)) == round_index,
             f"round {round_index} marker identity drifted")
    _require(marker.get("archive_name") == "ROUND_ARCHIVE.tar" and
             marker.get("transport") == "single-deterministic-tar-v1",
             f"round {round_index} archive transport drifted")
    actual_archive_sha = _sha256(archive)
    _require(marker.get("archive_sha256") == actual_archive_sha,
             f"round {round_index} archive SHA receipt differs from the object")
    _require(marker.get("manifest_sha256") == _sha256(manifest),
             f"round {round_index} manifest SHA receipt differs from the object")
    count, verified_sha = archive_module.verify_archive(
        archive, marker["archive_sha256"])
    _require(verified_sha == actual_archive_sha,
             f"round {round_index} archive verifier SHA drifted")
    _require(int(marker.get("logical_file_count", -1)) == count,
             f"round {round_index} logical file count drifted")
    extracted = round_dir / "extracted"
    _require(not extracted.exists(), f"round {round_index} extraction already exists")
    archive_module.extract_archive(archive, extracted)
    _require((extracted / "SHA256SUMS").read_bytes() == manifest.read_bytes(),
             f"round {round_index} remote/archived manifests differ")
    members = _verify_manifest(extracted, extracted / "SHA256SUMS")
    required_members = set(contract["required_round_members"])
    _require(required_members <= set(members),
             f"round {round_index} lacks staged evidence members: "
             f"{sorted(required_members - set(members))}")
    inventory = _load_json(extracted / "ROUND_INVENTORY.json")
    _require(inventory.get("schema") == "canon-p38-round-stage-v1",
             f"round {round_index} inventory schema drifted")
    _require(int(inventory.get("diagnostic_round", -1)) == round_index,
             f"round {round_index} inventory identity drifted")
    _require(int(inventory.get("pre_alignment_records", -1)) == 1,
             f"round {round_index} pre-alignment inventory drifted")
    _require(int(inventory.get("journal_records", 0)) > 0 and
             int(inventory.get("incident_records", 0)) > 0,
             f"round {round_index} journal/incident inventory is empty")
    _require(inventory.get("journal_scope") == "cumulative-unscoped",
             f"round {round_index} request-journal scope drifted")
    round_pre_records = _load_jsonl(
        extracted / "pre-alignment.jsonl", label="round pre-alignment")
    _require(len(round_pre_records) == 1 and
             round_pre_records[0] == pre_records[round_index],
             f"round {round_index} staged/root pre-alignment records differ")
    journal_records = _load_jsonl(
        extracted / "request-journal.jsonl", label="request journal")
    _require(len(journal_records) == int(inventory["journal_records"]),
             f"round {round_index} request-journal inventory differs from JSONL")
    _require(all(record.get("schema") == "p38-request-journal-v1"
                 for record in journal_records),
             f"round {round_index} request-journal schema drifted")
    incident_records = _load_jsonl(
        extracted / "incident-ledger.jsonl", label="incident ledger")
    _require(len(incident_records) == int(inventory["incident_records"]),
             f"round {round_index} incident inventory differs from JSONL")
    _require(all(
        record.get("schema") == "p38-incident-ledger-v1" and
        int(record.get("diagnostic_round", -1)) == round_index
        for record in incident_records
    ), f"round {round_index} incident-ledger scope/schema drifted")
    observer_counts = {
        "kv_records": _observer_pair_count(
            extracted, "p38_kv_observer", "p38-live-kv-prefix-table-v1",
            round_index),
        "seam_records": _observer_pair_count(
            extracted, "p38_seam", "p38-seam-fingerprint-v1", round_index),
        "tail_records": _observer_pair_count(
            extracted, "p38_tail", "p38-tail-values-v1", round_index),
        "terminal_records": _observer_pair_count(
            extracted, "p38_terminal", "p38-terminal-discriminator-v1",
            round_index),
    }
    for name, observed in observer_counts.items():
      _require(int(inventory.get(name, -1)) == observed,
               f"round {round_index} {name} inventory differs from files")
    _require(observer_counts["kv_records"] > 0,
             f"round {round_index} required KV evidence is empty")
    _require(observer_counts["seam_records"] == 0 and
             observer_counts["tail_records"] == 0 and
             observer_counts["terminal_records"] == 0,
             f"round {round_index} no-terminal-observer inventory drifted")
    capsule = extracted / "mismatch-capsule.npz"
    capsule_sha = _sha256(capsule)
    _require(capsule_sha == expected["capsule_sha256"],
             f"round {round_index} capsule SHA differs from registered evidence")
    root_capsule = root_files / f"mismatch-capsule.round-{round_index:06d}.npz"
    _require(_sha256(root_capsule) == capsule_sha,
             f"round {round_index} root and sealed capsules differ")
    counts = _capsule_counts(capsule)
    record = pre_records[round_index]
    ab = _boundary(record, "S_decode_vs_S_prefill")
    bc = _boundary(record, "S_prefill_vs_T_old")
    _require(int(record.get("N_action", -1)) == expected["n_action"],
             f"round {round_index} N_action drifted")
    for observed, field in (
        (ab.get("differing_elements"), "a_b_differing_elements"),
        (ab.get("differing_bytes"), "a_b_differing_bytes"),
        (bc.get("differing_elements"), "b_c_differing_elements"),
        (bc.get("differing_bytes"), "b_c_differing_bytes"),
    ):
      _require(int(observed) == int(expected[field]),
               f"round {round_index} pre-alignment {field} drifted")
    _require(float(ab.get("max_abs")) == float(expected["a_b_max_abs"]),
             f"round {round_index} A-B max_abs drifted")
    for field in (
        "a_b_differing_elements", "a_b_differing_bytes",
        "b_c_differing_elements", "b_c_differing_bytes",
    ):
      _require(int(counts[field]) == int(expected[field]),
               f"round {round_index} capsule {field} drifted")
    _require(float(counts["a_b_max_abs"]) == float(expected["a_b_max_abs"]),
             f"round {round_index} capsule A-B max_abs drifted")
    round_result = {
        "diagnostic_round": round_index,
        "n_action": int(record["N_action"]),
        "a_b_differing_elements": int(ab["differing_elements"]),
        "a_b_differing_bytes": int(ab["differing_bytes"]),
        "a_b_max_abs": float(ab["max_abs"]),
        "b_c_differing_elements": int(bc["differing_elements"]),
        "b_c_differing_bytes": int(bc["differing_bytes"]),
        "archive_sha256": actual_archive_sha,
        "archive_logical_files": count,
        "capsule_sha256": capsule_sha,
    }
    round_results.append(round_result)
    total_actions += round_result["n_action"]
    total_ab_elements += round_result["a_b_differing_elements"]
    total_ab_bytes += round_result["a_b_differing_bytes"]
    total_bc_elements += round_result["b_c_differing_elements"]
    total_bc_bytes += round_result["b_c_differing_bytes"]

  terminal = _terminal_status(args.reference_evidence)
  _require(not contract.get("forbid_terminal_observer") or not terminal["admitted"],
           "P38s22 cannot admit terminal evidence from the no-observer arm")
  returned_receipts = []
  for round_index, result in enumerate(round_results):
    path = args.reference_evidence / f"ROUND_COMPLETE.round-{round_index:06d}.json"
    if not path.is_file():
      returned_receipts.append({"diagnostic_round": round_index, "present": False})
      continue
    value = _load_json(path)
    returned_receipts.append({
        "diagnostic_round": round_index,
        "present": True,
        "matches_remote_marker": (
            path.read_bytes()
            == (args.round_root / f"{round_index:06d}/ROUND_COMPLETE.json").read_bytes()
        ),
        "archive_sha_equals_capsule_sha": (
            value.get("archive_sha256") == result["capsule_sha256"]
        ),
    })

  verdict = (
      "GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED"
      if total_ab_elements > 0 and total_bc_elements == 0 else
      "UNEXPECTED_NUMERICAL_OUTCOME"
  )
  _require(verdict == "GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED",
           "P38s22 numerical decision differs from the preregistered table")
  return {
      "schema": RESULT_SCHEMA,
      "status": "PASS",
      "verdict": verdict,
      "claim_ceiling": (
          "P38s22_rejects_BF16_BF16_F32_as_a_causal_repair;_lm_head_interval_"
          "localization_remains_inherited_from_admitted_P38s21_evidence"
      ),
      "analysis_source_commit": args.analysis_source_commit,
      "contract_sha256": _sha256(args.contract),
      "tool_sha256": {
          "auditor": _sha256(Path(__file__).resolve()),
          "archive": _sha256(Path(__file__).resolve().with_name(
              "p38_evidence_archive.py")),
          "wrapper": _sha256(Path(__file__).resolve().with_name(
              "run_p38s22_offsite_audit.sh")),
      },
      "source_uri_sha256": hashlib.sha256(
          contract["source_gcs_uri"].encode("utf-8")).hexdigest(),
      "root_manifest_sha256": _sha256(root_manifest),
      "rounds": round_results,
      "totals": {
          "n_action": total_actions,
          "a_b_differing_elements": total_ab_elements,
          "a_b_differing_bytes": total_ab_bytes,
          "b_c_differing_elements": total_bc_elements,
          "b_c_differing_bytes": total_bc_bytes,
      },
      "terminal_classification": terminal,
      "returned_receipts": returned_receipts,
      "next_gate": "dedicated_fixed_tile_Pallas_lm_head_onehost_then_P38s23",
  }


def _write_output(args: argparse.Namespace, result: dict[str, Any]) -> None:
  _require(not args.output.exists(), f"output already exists: {args.output}")
  args.output.mkdir(parents=True, mode=0o700)
  receipts = args.output / "receipts"
  receipts.mkdir()
  for name in ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json"):
    source = args.source_root / name
    if source.is_file():
      _copy_receipt(source, receipts / name)
  root_manifest = args.source_root / "files/SHA256SUMS"
  if root_manifest.is_file():
    _copy_receipt(root_manifest, receipts / "ROOT_SHA256SUMS")
  for round_index in range(3):
    source = args.round_root / f"{round_index:06d}/ROUND_COMPLETE.json"
    manifest = args.round_root / f"{round_index:06d}/SHA256SUMS"
    if source.is_file():
      _copy_receipt(source, receipts / f"ROUND_COMPLETE.round-{round_index:06d}.json")
    if manifest.is_file():
      _copy_receipt(manifest, receipts / f"ROUND_SHA256SUMS.round-{round_index:06d}")
  (args.output / "AUDIT.json").write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  verdict = {
      "schema": "p38s22-offsite-verdict-v1",
      "status": result["status"],
      "verdict": result["verdict"],
      "claim_ceiling": result.get("claim_ceiling", "none"),
      "next_gate": result.get("next_gate", "repair_offsite_evidence_only"),
  }
  (args.output / "verdict.json").write_text(
      json.dumps(verdict, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  totals = result.get("totals", {})
  (args.output / "SUMMARY.txt").write_text(
      "\n".join((
          f"status={result['status']}",
          f"verdict={result['verdict']}",
          f"n_action={totals.get('n_action', 'unknown')}",
          f"a_b_differing_elements={totals.get('a_b_differing_elements', 'unknown')}",
          f"a_b_differing_bytes={totals.get('a_b_differing_bytes', 'unknown')}",
          f"b_c_differing_elements={totals.get('b_c_differing_elements', 'unknown')}",
          f"b_c_differing_bytes={totals.get('b_c_differing_bytes', 'unknown')}",
          f"next_gate={result.get('next_gate', 'repair_offsite_evidence_only')}",
      )) + "\n",
      encoding="utf-8",
  )
  _seal(args.output)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--contract", required=True, type=Path)
  parser.add_argument("--source-root", required=True, type=Path)
  parser.add_argument("--round-root", required=True, type=Path)
  parser.add_argument("--reference-evidence", required=True, type=Path)
  parser.add_argument("--analysis-source-commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = audit(args)
    rc = 0
  # Evidence parsers (NumPy, tarfile, JSON, and filesystem code) use different
  # exception families. Preserve every such failure as a sealed INCONCLUSIVE
  # receipt; only process-level interrupts bypass this boundary.
  except Exception as exc:  # pylint: disable=broad-exception-caught
    result = {
        "schema": RESULT_SCHEMA,
        "status": "INCONCLUSIVE",
        "verdict": "OFFSITE_EVIDENCE_AUDIT_FAILED",
        "failure": str(exc),
        "analysis_source_commit": args.analysis_source_commit,
        "contract_sha256": _sha256(args.contract) if args.contract.is_file() else None,
        "tool_sha256": {
            "auditor": _sha256(Path(__file__).resolve()),
            "archive": _sha256(Path(__file__).resolve().with_name(
                "p38_evidence_archive.py")),
            "wrapper": _sha256(Path(__file__).resolve().with_name(
                "run_p38s22_offsite_audit.sh")),
        },
        "claim_ceiling": "none",
        "next_gate": "repair_or_return_the_named_GCS_artifact_without_TPU_relaunch",
    }
    rc = 4
  _write_output(args, result)
  print(
      "[P38S22.OFFSITE] COMPLETE "
      f"status={result['status']} verdict={result['verdict']} "
      f"output={args.output} rc={rc}"
  )
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
