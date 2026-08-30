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
  _require(completed.returncode == 0,
           "archived source-bound classifier failed")

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
      "[M15.E0U.ON-R0] LOCAL_CLASSIFICATION_COMPLETE "
      f"status={report['status']} classification={report['classification']} "
      "rounds=1 three_round_verdict=0 numerical_repair_authorized=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
