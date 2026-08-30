#!/usr/bin/env python3
"""Recover Attempt-20 treatment round 0 from its durable classifier input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


class Attempt20Round0RecoveryError(RuntimeError):
  """Raised when the recovered checkpoint cannot support a classification."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt20Round0RecoveryError(message)


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
    raise Attempt20Round0RecoveryError(f"{label} is invalid") from error
  _require(isinstance(value, dict), f"{label} is not an object")
  return value


def _load_archive_module(canon: Path):
  path = (
      canon
      / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
      "p38_evidence_archive.py"
  )
  spec = importlib.util.spec_from_file_location("p38_evidence_archive", path)
  _require(spec is not None and spec.loader is not None,
           "evidence archive verifier cannot be loaded")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _copy(source: Path, destination: Path) -> None:
  _require(source.is_file() and source.stat().st_size > 0,
           f"recovered output is absent: {source.name}")
  shutil.copyfile(source, destination)


def _load_source_classifier(path: Path):
  spec = importlib.util.spec_from_file_location(
      "attempt20_archived_kv_classifier", path
  )
  _require(spec is not None and spec.loader is not None,
           "archived classifier cannot be loaded for failure audit")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _request_sha256(value: Any) -> str:
  return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _safe_comparison(value: dict[str, Any]) -> dict[str, Any]:
  result = dict(value)
  for field in ("source_a_request_id", "clean_b_request_id"):
    raw = result.pop(field, "")
    result[f"{field}_sha256"] = _request_sha256(raw)
  return result


def _prefix_audit(
    target: np.ndarray, history: np.ndarray
) -> dict[str, Any]:
  target = np.asarray(target, dtype=np.int32).reshape(-1)
  history = np.asarray(history, dtype=np.int32).reshape(-1)
  common = min(target.size, history.size)
  unequal = np.flatnonzero(target[:common] != history[:common])
  if unequal.size:
    first = int(unequal[0])
    target_token = int(target[first])
    history_token = int(history[first])
  elif target.size != history.size:
    first = common
    target_token = int(target[first]) if first < target.size else None
    history_token = int(history[first]) if first < history.size else None
  else:
    first = -1
    target_token = None
    history_token = None
  return {
      "target_tokens": int(target.size),
      "history_tokens": int(history.size),
      "longest_common_prefix_tokens": (
          int(first) if first >= 0 else int(target.size)
      ),
      "first_mismatch_index": first,
      "target_token_at_first_mismatch": target_token,
      "history_token_at_first_mismatch": history_token,
      "target_prefix_equal": bool(
          target.size <= history.size
          and np.array_equal(target, history[:target.size])
      ),
      "target_token_sha256": hashlib.sha256(
          np.ascontiguousarray(target, dtype="<i8").tobytes()
      ).hexdigest(),
      "history_token_sha256": hashlib.sha256(
          np.ascontiguousarray(history, dtype="<i8").tobytes()
      ).hexdigest(),
  }


def _classifier_failure_audit(
    *, extracted: Path, classifier_path: Path, capsule: Path
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
  """Builds a bounded receipt without treating unjoined KV data as red."""
  classifier = _load_source_classifier(classifier_path)
  try:
    records = classifier._load_records(extracted)  # pylint: disable=protected-access
    pairs = classifier._pair_records(records)  # pylint: disable=protected-access
    diagnostic_round, histories = classifier._load_capsule_histories(  # pylint: disable=protected-access
        capsule
    )
    comparisons = [
        _safe_comparison(classifier._compare_pair(live, clean))  # pylint: disable=protected-access
        for live, clean in pairs
    ]
  except Exception as error:  # The archived module owns its exception type.
    raise Attempt20Round0RecoveryError(
        "archived classifier inputs cannot support a bounded failure audit"
    ) from error

  matrix = []
  exact_prefixes = 0
  red_joins = 0
  for live, _clean in pairs:
    target = np.asarray(live["arrays"]["token_ids"], dtype=np.int32)
    for history in histories:
      row = _prefix_audit(target, history["tokens"])
      row.update({
          "diagnostic_round": int(live["diagnostic_round"]),
          "source_a_record_index": int(live["record_index"]),
          "source_a_request_id_sha256": _request_sha256(
              live["request_id"]
          ),
          "capsule_source_row": int(history["source_row"]),
          "capsule": str(history["capsule"]),
      })
      mismatch_at_or_before_target = any(
          int(history["prompt_length"]) + int(position) <= target.size
          for position in history["mismatch_positions"]
      )
      row["red_position_at_or_before_target"] = bool(
          mismatch_at_or_before_target
      )
      exact_prefixes += int(row["target_prefix_equal"])
      red_joins += int(
          row["target_prefix_equal"] and mismatch_at_or_before_target
      )
      matrix.append(row)

  status = (
      "TOKEN_HISTORY_JOIN_MISMATCH"
      if exact_prefixes == 0
      else "INVALID_OR_CLASSIFIER_FAILED"
  )
  audit = {
      "schema": "m15-attempt20-token-join-audit-v1",
      "status": status,
      "diagnostic_round": diagnostic_round,
      "observer_pairs": len(pairs),
      "capsule_histories": len(histories),
      "matrix_rows": len(matrix),
      "exact_target_prefix_candidates": exact_prefixes,
      "red_join_candidates": red_joins,
      "minimum_longest_common_prefix_tokens": min(
          row["longest_common_prefix_tokens"] for row in matrix
      ),
      "maximum_longest_common_prefix_tokens": max(
          row["longest_common_prefix_tokens"] for row in matrix
      ),
      "rows": matrix,
  }
  return status, audit, comparisons


def _write_failure_output(
    *, output: Path, analysis_source: str, expected_source: str,
    receipt_path: Path, manifest: Path, receipt: dict[str, Any],
    round_input: dict[str, Any], classifier_log: Path, classifier_rc: int,
    status: str, token_join_audit: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
  report = {
      "schema": "m15-attempt20-on-round0-offline-recovery-v1",
      "status": status,
      "analysis_source_commit": analysis_source,
      "target_source_commit": expected_source,
      "arm": "on",
      "diagnostic_round": 0,
      "failure_stage": "source-bound-classifier",
      "classifier_returncode": classifier_rc,
      "classifier_log_sha256": _sha256(classifier_log),
      "a_b_differing_bytes": round_input["a_b_differing_bytes"],
      "a_b_differing_elements": round_input["a_b_differing_elements"],
      "b_c_differing_bytes": round_input["b_c_differing_bytes"],
      "b_c_differing_elements": round_input["b_c_differing_elements"],
      "classifier_input_archive_sha256": receipt["archive_sha256"],
      "classifier_input_manifest_sha256": receipt["manifest_sha256"],
      "classification": None,
      "classification_available": False,
      "round_input_recovered": True,
      "unbound_observer_comparisons": comparisons,
      "token_join_audit": token_join_audit,
      "b_full_reset_runtime_receipt_available": False,
      "all_num_cached_tokens_zero_runtime_receipt_available": False,
      "rounds_recovered": [],
      "three_round_verdict": False,
      "terminal_pair_complete": False,
      "target_rerun": False,
      "numerical_repair_authorized": False,
      "remote_mutation": False,
      "claim_ceiling": (
          "NO_CLASSIFICATION / INCONCLUSIVE / NO_TARGET_PASS / "
          "B_RESET_RUNTIME_RECEIPT_UNAVAILABLE / "
          "NO_NUMERICAL_REPAIR_AUTHORIZATION"
      ),
  }
  output.mkdir(mode=0o700)
  _copy(classifier_log, output / "raw_classifier_error.log")
  _copy(
      receipt_path,
      output / "on.round-000000.classifier-input-receipt.json",
  )
  _copy(
      manifest,
      output / "on.round-000000.classifier-input-sha256sums",
  )
  report_path = output / "ATTEMPT20_ON_R0_RECOVERY.json"
  report_path.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  names = sorted(path.name for path in output.iterdir() if path.is_file())
  (output / "SHA256SUMS").write_text(
      "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  return report


def recover(
    *,
    archive: Path,
    manifest: Path,
    receipt_path: Path,
    expected_source: str,
    analysis_source: str,
    scratch: Path,
    output: Path,
) -> dict[str, Any]:
  """Verify, extract, classify, and compact one durable treatment checkpoint."""
  for label, value in (
      ("expected source", expected_source),
      ("analysis source", analysis_source),
  ):
    _require(re.fullmatch(r"[0-9a-f]{40}", value) is not None,
             f"{label} is not one full lowercase SHA")
  _require(scratch.is_dir(), "scratch directory is absent")
  _require(not output.exists(), "recovery output already exists")
  _require(archive.is_file() and archive.stat().st_size > 0,
           "classifier-input archive is absent")
  _require(manifest.is_file() and manifest.stat().st_size > 0,
           "classifier-input manifest is absent")

  receipt = _json(receipt_path, "classifier-input receipt")
  _require(
      receipt.get("schema") == "m15-e0-kv-classifier-input-receipt-v1"
      and receipt.get("status")
          == "uploaded-readback-verified-before-classification"
      and receipt.get("arm") == "on"
      and receipt.get("diagnostic_round") == 0
      and receipt.get("source_commit") == expected_source
      and receipt.get("runtime_source_commit") == expected_source
      and receipt.get("kv_records") == 16
      and receipt.get("kv_pairs") == 8
      and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", "")))
          is not None
      and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("manifest_sha256", "")))
          is not None,
      "classifier-input receipt contract drifted",
  )
  _require(_sha256(archive) == receipt["archive_sha256"],
           "classifier-input archive SHA differs from receipt")
  _require(_sha256(manifest) == receipt["manifest_sha256"],
           "classifier-input manifest SHA differs from receipt")

  canon = Path(__file__).resolve().parents[3]
  archive_module = _load_archive_module(canon)
  extracted = scratch / "classifier-input"
  try:
    count, archive_sha = archive_module.extract_archive(archive, extracted)
  except (OSError, ValueError) as error:
    raise Attempt20Round0RecoveryError(
        "classifier-input archive verification or extraction failed"
    ) from error
  _require(count > 0 and archive_sha == receipt["archive_sha256"],
           "classifier-input archive identity drifted during extraction")
  _require((extracted / "SHA256SUMS").read_bytes() == manifest.read_bytes(),
           "remote manifest differs from the archived manifest")

  round_input = _json(extracted / "ROUND_INPUT.json", "round input")
  _require(
      round_input.get("schema") == "m15-e0-kv-round-input-v1"
      and round_input.get("status") == "STAGED_FOR_CLASSIFIER_CHECKPOINT"
      and round_input.get("arm") == "on"
      and round_input.get("diagnostic_round") == 0
      and round_input.get("expected_source_commit") == expected_source
      and round_input.get("runtime_source_commit") == expected_source
      and round_input.get("kv_records") == 16
      and round_input.get("kv_pairs") == 8
      and round_input.get("b_c_differing_bytes") == 0
      and round_input.get("b_c_differing_elements") == 0
      and type(round_input.get("a_b_differing_bytes")) is int
      and type(round_input.get("a_b_differing_elements")) is int
      and round_input["a_b_differing_bytes"] >= 0
      and round_input["a_b_differing_elements"] >= 0
      and ((round_input["a_b_differing_bytes"] == 0)
           == (round_input["a_b_differing_elements"] == 0))
      and receipt.get("a_b_differing_bytes")
          == round_input["a_b_differing_bytes"],
      "round input contract drifted",
  )

  classifier_path = extracted / "classify_p38_kv_observer.py"
  runtime = _json(extracted / "CLASSIFIER_RUNTIME.json", "classifier runtime")
  _require(
      runtime.get("schema") == "m15-e0-kv-classifier-runtime-v2"
      and runtime.get("status") == "source-bound"
      and runtime.get("runtime_source_commit") == expected_source
      and runtime.get("path") == classifier_path.name
      and runtime.get("sha256") == _sha256(classifier_path),
      "archived classifier runtime is not source-bound",
  )
  replay = extracted / "m15-replay-envelope.jsonl"
  pre_alignment = extracted / "pre-alignment.jsonl"
  _require(replay.is_file() and replay.stat().st_size > 0,
           "archived replay ledger is absent")
  _require(pre_alignment.is_file() and pre_alignment.stat().st_size > 0,
           "archived pre-alignment input is absent")

  red = round_input["a_b_differing_bytes"] > 0
  capsule = extracted / "mismatch-capsule.npz"
  _require((capsule.is_file() and capsule.stat().st_size > 0) == red,
           "mismatch capsule presence disagrees with A-B")
  classification_path = scratch / "on.round-000000.kv-observer-classification.json"
  classifier_log = scratch / "classifier.log"
  command = [
      sys.executable,
      classifier_path.name,
      "--directory", ".",
  ]
  if red:
    command.extend([
        "--capsule", capsule.name,
        "--require-red-join",
        "--replay-ledger", replay.name,
    ])
  command.extend(["--output", str(classification_path)])
  environment = dict(os.environ)
  environment["JAX_PLATFORMS"] = "cpu"
  environment["PYTHONPATH"] = (
      str(canon.parent)
      + (":" + environment["PYTHONPATH"]
         if environment.get("PYTHONPATH") else "")
  )
  with classifier_log.open("xb") as log:
    completed = subprocess.run(
        command,
        cwd=extracted,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
  if completed.returncode != 0:
    status, token_join_audit, comparisons = _classifier_failure_audit(
        extracted=extracted,
        classifier_path=classifier_path,
        capsule=capsule,
    )
    return _write_failure_output(
        output=output,
        analysis_source=analysis_source,
        expected_source=expected_source,
        receipt_path=receipt_path,
        manifest=manifest,
        receipt=receipt,
        round_input=round_input,
        classifier_log=classifier_log,
        classifier_rc=completed.returncode,
        status=status,
        token_join_audit=token_join_audit,
        comparisons=comparisons,
    )

  classification = _json(classification_path, "offline classification")
  comparisons = classification.get("comparisons", [])
  _require(
      classification.get("schema") == "p38-live-kv-classification-v2"
      and classification.get("status") == "PASS"
      and classification.get("records") == 16
      and classification.get("pairs") == 8
      and isinstance(comparisons, list)
      and len(comparisons) == 8
      and {row.get("diagnostic_round") for row in comparisons} == {0}
      and classification.get("source_inputs", {}).get("classifier", {}).get(
          "sha256") == runtime["sha256"],
      "offline classification contract drifted",
  )
  outcome = classification.get("classification")
  if red:
    _require(
        outcome in {
            "live_kv_fingerprint_equal_on_red_row",
            "live_kv_fingerprint_differs_on_red_row",
        }
        and classification.get("source_request_binding", {}).get("status")
            == "UNIQUE_FUTURE_PREFIX_BINDING",
        "red treatment round lacks a uniquely bound mechanism classification",
    )
  else:
    _require(
        outcome == "observer_pairs_valid_red_join_pending"
        and classification.get("source_request_binding") is None,
        "exact treatment round classification drifted",
    )

  target_lengths = {int(row.get("target_seq_len", -1)) for row in comparisons}
  _require(len(target_lengths) == 1 and next(iter(target_lengths)) > 0,
           "treatment aliases do not share one target prefix length")
  target_seq_len = next(iter(target_lengths))
  selected_index = None
  binding = classification.get("source_request_binding")
  if isinstance(binding, dict):
    selected_index = int(binding["selected_source_a_record_index"])
  selected_comparison = next(
      (
          row for row in comparisons
          if selected_index is not None
          and int(row["source_a_record_index"]) == selected_index
      ),
      comparisons[0],
  )
  selected_json = extracted / (
      f"p38_kv_observer_{int(selected_comparison['source_a_record_index']):04d}_a.json"
  )
  selected_record = _json(selected_json, "selected A observer record")
  geometry = {
      "block_size": selected_record.get("block_size"),
      "cache_shape": selected_record.get("cache_shape"),
      "layer_count": selected_record.get("layer_count"),
      "layer_indices": selected_record.get("layer_indices", [0]),
      "logical_pages": selected_record.get("logical_pages"),
      "observer_pages": selected_record.get("observer_pages"),
      "target_seq_len": target_seq_len,
  }

  statuses = {
      "observer_pairs_valid_red_join_pending": "ROUND0_TARGET_NON_REPRODUCTION",
      "live_kv_fingerprint_equal_on_red_row": (
          "ROUND0_LIVE_KV_FINGERPRINT_EQUAL"
      ),
      "live_kv_fingerprint_differs_on_red_row": (
          "ROUND0_LIVE_KV_FINGERPRINT_DIFFERS"
      ),
  }
  report = {
      "schema": "m15-attempt20-on-round0-offline-recovery-v1",
      "status": statuses[outcome],
      "analysis_source_commit": analysis_source,
      "target_source_commit": expected_source,
      "arm": "on",
      "diagnostic_round": 0,
      "a_b_differing_bytes": round_input["a_b_differing_bytes"],
      "a_b_differing_elements": round_input["a_b_differing_elements"],
      "b_c_differing_bytes": 0,
      "b_c_differing_elements": 0,
      "b_full_reset_runtime_receipt_available": False,
      "all_num_cached_tokens_zero_runtime_receipt_available": False,
      "classification": outcome,
      "classification_available": True,
      "classification_sha256": _sha256(classification_path),
      "classifier_input_archive_sha256": receipt["archive_sha256"],
      "classifier_input_manifest_sha256": receipt["manifest_sha256"],
      "first_difference": selected_comparison.get("first_difference"),
      "observer_geometry": geometry,
      "rounds_recovered": [0],
      "three_round_verdict": False,
      "terminal_pair_complete": False,
      "target_rerun": False,
      "numerical_repair_authorized": False,
      "remote_mutation": False,
      "claim_ceiling": (
          "ANALYSIS_GRADE_TREATMENT_ROUND0_ONLY / NO_3_OF_3 / "
          "B_RESET_RUNTIME_RECEIPT_UNAVAILABLE / NO_TARGET_PASS / "
          "NO_NUMERICAL_REPAIR_AUTHORIZATION"
      ),
  }

  output.mkdir(mode=0o700)
  _copy(
      classification_path,
      output / "on.round-000000.kv-observer-classification.json",
  )
  _copy(
      receipt_path,
      output / "on.round-000000.classifier-input-receipt.json",
  )
  _copy(
      manifest,
      output / "on.round-000000.classifier-input-sha256sums",
  )
  report_path = output / "ATTEMPT20_ON_R0_RECOVERY.json"
  report_path.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  names = sorted(path.name for path in output.iterdir() if path.is_file())
  (output / "SHA256SUMS").write_text(
      "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  return report


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--archive", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--receipt", required=True, type=Path)
  parser.add_argument("--expected-source", required=True)
  parser.add_argument("--analysis-source", required=True)
  parser.add_argument("--scratch", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    report = recover(
        archive=args.archive,
        manifest=args.manifest,
        receipt_path=args.receipt,
        expected_source=args.expected_source,
        analysis_source=args.analysis_source,
        scratch=args.scratch,
        output=args.output,
    )
  except (Attempt20Round0RecoveryError, OSError) as error:
    print(f"[M15.E0U.ON-R0] INVALID {error}", file=sys.stderr)
    return 2
  print(
      (
          "[M15.E0U.ON-R0] LOCAL_CLASSIFICATION_COMPLETE "
          if report["classification_available"]
          else "[M15.E0U.ON-R0] FAILURE_AUDIT_COMPLETE "
      )
      + f"status={report['status']} "
      + f"classification={report['classification'] or 'NONE'} "
      + "rounds="
      + ("1 " if report["classification_available"] else "0 ")
      + "three_round_verdict=0 numerical_repair_authorized=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
