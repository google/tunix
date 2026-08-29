#!/usr/bin/env python3
"""Fail-closed intake review for the official Attempt-18 E0 compact return."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_RE = re.compile(r"[0-9a-f]{40}")
_EXPECTED_FILES = {
    "E0_KV_RETURN.json",
    "off.kv-observer-classification.json",
    "on.kv-observer-classification.json",
    "SHA256SUMS",
}
_CLASSIFIER_CLAIM_LEVEL = (
    "bit-level-diagnostic-fingerprint-not-full-kv-bytes"
)
_RETURN_CLAIM = (
    "The KV result is a diagnostic fingerprint over the uniquely bound "
    "red request, not a collision-free proof of all KV bytes."
)


class ReturnReviewError(RuntimeError):
  """Raised when the compact return cannot support an E0 verdict."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ReturnReviewError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _load_canonical_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON root is not an object: {path.name}")
  encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
  _require(
      path.read_text(encoding="utf-8") == encoded,
      f"JSON is not canonical official-wrapper output: {path.name}",
  )
  return value


def _manifest(root: Path) -> dict[str, str]:
  path = root / "SHA256SUMS"
  entries: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    fields = line.split("  ", 1)
    _require(len(fields) == 2, "return manifest line is malformed")
    digest, name = fields
    _require(_SHA_RE.fullmatch(digest) is not None, "manifest SHA256 is invalid")
    _require(Path(name).name == name and name not in entries,
             "manifest member name is invalid or duplicated")
    entries[name] = digest
  expected = _EXPECTED_FILES - {"SHA256SUMS"}
  _require(set(entries) == expected, "official return manifest inventory drifted")
  for name, digest in entries.items():
    _require(_sha256(root / name) == digest,
             f"official return member failed SHA256: {name}")
  return entries


def _review_classifier(
    path: Path, *, require_red_binding: bool
) -> dict[str, Any]:
  value = _load_canonical_json(path)
  _require(
      value.get("schema") == "p38-live-kv-classification-v2"
      and value.get("status") == "PASS"
      and value.get("records") == 16
      and value.get("pairs") == 8,
      f"classifier inventory is incomplete: {path.name}",
  )
  comparisons = value.get("comparisons")
  _require(isinstance(comparisons, list) and len(comparisons) == 8,
           f"classifier comparisons are incomplete: {path.name}")
  for comparison in comparisons:
    valid = comparison.get("valid_tokens")
    _require(
        comparison.get("target_seq_len") == 1226
        and valid == [16] * 76 + [10]
        and isinstance(comparison.get("fingerprint_equal"), bool),
        f"classifier prefix geometry drifted: {path.name}",
    )
  source_indices = [item.get("source_a_record_index") for item in comparisons]
  clean_indices = [item.get("clean_b_record_index") for item in comparisons]
  _require(
      all(isinstance(index, int) for index in source_indices + clean_indices)
      and len(set(source_indices)) == 8
      and len(set(clean_indices)) == 8,
      f"classifier pair identities are incomplete: {path.name}",
  )
  source_inputs = value.get("source_inputs") or {}
  classifier_input = source_inputs.get("classifier") or {}
  observer_records = source_inputs.get("observer_records")
  _require(
      _SHA_RE.fullmatch(str(classifier_input.get("sha256", ""))) is not None
      and isinstance(observer_records, list)
      and len(observer_records) == 16,
      f"classifier source inputs are incomplete: {path.name}",
  )
  for record in observer_records:
    _require(
        _SHA_RE.fullmatch(str(record.get("json_sha256", ""))) is not None
        and _SHA_RE.fullmatch(str(record.get("npz_sha256", ""))) is not None,
        f"observer source digest is invalid: {path.name}",
    )
  a_records = {
      record.get("record_index") for record in observer_records
      if record.get("arm") == "A"
  }
  b_records = {
      record.get("record_index") for record in observer_records
      if record.get("arm") == "B"
  }
  _require(
      len(a_records) == len(b_records) == 8
      and set(source_indices) == a_records
      and set(clean_indices) == b_records
      and all(
          record.get("valid_tokens") == [16] * 76 + [10]
          for record in observer_records
      ),
      f"observer record identity/geometry drifted: {path.name}",
  )
  _require(
      value.get("claim_level") == _CLASSIFIER_CLAIM_LEVEL
      and any(
          "does not mathematically prove full KV byte equality" in str(item)
          for item in value.get("claim_ceiling", ())
      ),
      f"classifier claim ceiling drifted: {path.name}",
  )

  binding = value.get("source_request_binding")
  if not require_red_binding:
    _require(
        value.get("classification") == "observer_pairs_valid_red_join_pending"
        and binding is None,
        "APC-off control unexpectedly claims a red-row mechanism verdict",
    )
    _require(
        value.get("red_joins") == []
        and source_inputs.get("capsules") == []
        and "replay_ledger" not in source_inputs,
        "APC-off control unexpectedly carries red-source provenance",
    )
    return value

  _require(isinstance(binding, dict), "treatment classifier lacks request binding")
  candidates = binding.get("candidates")
  selected_index = binding.get("selected_source_a_record_index")
  horizon = binding.get("required_elimination_horizon")
  proof = binding.get("selected_proof_prefix_tokens")
  _require(
      binding.get("schema") == "m15-kv-source-request-binding-v1"
      and binding.get("status") == "UNIQUE_FUTURE_PREFIX_BINDING"
      and binding.get("diagnostic_round") == 0
      and binding.get("source_row") == 217
      and binding.get("anchor_prefix_tokens") == 1226
      and isinstance(selected_index, int)
      and isinstance(horizon, int)
      and isinstance(proof, int)
      and proof >= horizon
      and isinstance(candidates, list)
      and len(candidates) == 8,
      "treatment source-request binding is truncated or invalid",
  )
  statuses = [candidate.get("status") for candidate in candidates]
  _require(
      statuses.count("FUTURE_PREFIX_MATCH") == 1
      and statuses.count("FUTURE_PREFIX_CONFLICT") == 7,
      "treatment source-request candidates are not uniquely eliminated",
  )
  selected = [
      candidate for candidate in candidates
      if candidate.get("source_a_record_index") == selected_index
  ]
  conflicts = [
      candidate for candidate in candidates
      if candidate.get("status") == "FUTURE_PREFIX_CONFLICT"
  ]
  _require(
      len(selected) == 1
      and selected[0].get("status") == "FUTURE_PREFIX_MATCH"
      and binding.get("selected_request_id") == selected[0].get("request_id")
      and selected[0].get("matching_prefix_lengths")
      and max(selected[0]["matching_prefix_lengths"]) == proof
      and all(candidate.get("conflicting_prefix_lengths") for candidate in conflicts)
      and max(min(candidate["conflicting_prefix_lengths"])
              for candidate in conflicts) == horizon
      and {candidate.get("source_a_record_index") for candidate in candidates}
      == set(source_indices),
      "treatment selected request/proof horizon is not self-consistent",
  )
  capsules = source_inputs.get("capsules")
  replay = source_inputs.get("replay_ledger") or {}
  _require(
      isinstance(capsules, list)
      and capsules
      and all(_SHA_RE.fullmatch(str(item.get("sha256", ""))) is not None
              for item in capsules)
      and binding.get("capsule") in {
          item.get("path") for item in capsules
      }
      and _SHA_RE.fullmatch(str(replay.get("sha256", ""))) is not None,
      "treatment classifier lacks capsule/replay provenance",
  )
  red_joins = value.get("red_joins")
  _require(
      isinstance(red_joins, list)
      and len(red_joins) == 8
      and {item.get("source_a_record_index") for item in red_joins}
      == set(source_indices)
      and all(
          item.get("diagnostic_round") == 0
          and item.get("source_row") == 217
          and item.get("target_seq_len") == 1226
          for item in red_joins
      ),
      "treatment red joins are incomplete or drifted",
  )
  return value


def review(root: Path, expected_source: str, raw_log: Path | None = None) -> dict[str, Any]:
  _require(_SOURCE_RE.fullmatch(expected_source) is not None,
           "expected runtime source is not one full SHA")
  _require(root.is_dir(), "official return directory is absent")
  inventory = {path.name for path in root.iterdir()}
  _require(inventory == _EXPECTED_FILES, "official return file inventory drifted")
  manifest = _manifest(root)
  report = _load_canonical_json(root / "E0_KV_RETURN.json")
  _require(
      report.get("schema") == "m15-attempt18-e0-kv-return-v1"
      and report.get("source_commit") == expected_source
      and report.get("target_executed") is True
      and report.get("remote_mutation") is False
      and report.get("numerical_repair_authorized") is False
      and report.get("claim_ceiling") == _RETURN_CLAIM,
      "E0 return identity or claim ceiling drifted",
  )
  arms = report.get("arms")
  _require(isinstance(arms, dict) and set(arms) == {"off", "on"},
           "E0 return arms are incomplete")
  classifiers = {
      arm: _review_classifier(
          root / f"{arm}.kv-observer-classification.json",
          require_red_binding=(arm == "on" and arms[arm].get(
              "a_b_differing_bytes", 0) > 0),
      )
      for arm in ("off", "on")
  }
  for arm in ("off", "on"):
    row = arms[arm]
    classifier_path = root / f"{arm}.kv-observer-classification.json"
    digest = row.get("kv_classification_sha256")
    comparisons = classifiers[arm]["comparisons"]
    all_equal = all(item.get("fingerprint_equal") is True
                    for item in comparisons)
    execution = row.get("execution_receipts") or {}
    _require(
        _SHA_RE.fullmatch(str(digest)) is not None
        and digest == _sha256(classifier_path)
        and digest == manifest[classifier_path.name]
        and _SHA_RE.fullmatch(str(row.get("root_manifest_sha256", "")))
        is not None
        and row.get("kv_classification")
        == classifiers[arm].get("classification")
        and row.get("kv_all_pairs_equal") == all_equal
        and row.get("source_request_binding")
        == classifiers[arm].get("source_request_binding")
        and _SHA_RE.fullmatch(str(execution.get("run_log_sha256", "")))
        is not None
        and execution.get("runtime_source_exact") is True
        and execution.get("b_full_reset") is True
        and execution.get("all_num_cached_tokens_zero") is True
        and execution.get("zero_backward") is True
        and execution.get("zero_optimizer_commit") is True
        and row.get("b_c_differing_bytes") == 0
        and isinstance(row.get("n_action"), int)
        and row.get("n_action") > 0,
        f"E0 return arm summary does not match classifier: {arm}",
    )
  _require(
      arms["off"].get("a_b_differing_bytes") == 0
      and arms["off"].get("a_b_differing_elements") == 0
      and arms["off"].get("kv_all_pairs_equal") is True,
      "APC-off control is not exact and observer-neutral",
  )
  if arms["on"].get("a_b_differing_bytes") == 0:
    expected_status = "TARGET_NON_REPRODUCTION"
  elif classifiers["on"].get("classification") == (
      "live_kv_fingerprint_differs_on_red_row"
  ):
    expected_status = "LIVE_KV_FINGERPRINT_DIFFERS"
  elif classifiers["on"].get("classification") == (
      "live_kv_fingerprint_equal_on_red_row"
  ):
    expected_status = "LIVE_KV_FINGERPRINT_EQUAL"
  else:
    raise ReturnReviewError("treatment classifier has no admitted E0 outcome")
  _require(report.get("status") == expected_status,
           "E0 return status does not match the arm classifiers")

  if raw_log is not None:
    _require(raw_log.is_file(), "official return raw log is absent")
    text = raw_log.read_text(encoding="utf-8")
    _require(
        f"M15_E0_KV_RETURN_PASS status={expected_status}" in text
        and f"[M15.E0.KV.RETURN] COMPLETE status={expected_status}" in text
        and "[M15.E0.KV.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
        in text,
        "official return terminal markers are incomplete",
    )
  return {
      "status": expected_status,
      "source_commit": expected_source,
      "inventory_members": 3,
      "classifier_files": 2,
      "manifest_sha256": _sha256(root / "SHA256SUMS"),
      "control_a_b": arms["off"]["a_b_differing_bytes"],
      "treatment_a_b": arms["on"]["a_b_differing_bytes"],
      "b_c": 0,
      "claim": "diagnostic-fingerprint-only",
      "numerical_repair_authorized": False,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--return-dir", required=True, type=Path)
  parser.add_argument("--expected-source", required=True)
  parser.add_argument("--raw-log", type=Path)
  args = parser.parse_args()
  result = review(args.return_dir, args.expected_source, args.raw_log)
  print(
      "M15_E0_RETURN_INTAKE_PASS "
      f"status={result['status']} source={result['source_commit']} "
      f"inventory={result['inventory_members']} "
      f"classifiers={result['classifier_files']} "
      f"manifest_sha256={result['manifest_sha256']} "
      "claim=diagnostic-fingerprint-only numerical_repair_authorized=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
