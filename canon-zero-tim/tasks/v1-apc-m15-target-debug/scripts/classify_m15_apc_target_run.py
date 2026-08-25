#!/usr/bin/env python3
"""Fail-closed verdict for a bounded M15 APC target capture.

This classifier does not decide why A and B differ.  It decides whether a
fresh DP8xTP8 run is a valid APC-off control, a sufficiently representative
clean APC-on treatment, or a replayable APC-on red.  A red is replayable only
when the checked-in P38 capture classifier joined the mismatch capsule to an
exact incident call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any


class ClassificationError(RuntimeError):
  pass


_REQUIRED_COMMAND = {
    "mesh_dp": "8",
    "mesh_tp": "8",
    "batch_size": "32",
    "mini_batch_size": "32",
    "num_generations": "8",
    "max_prompt_length": "4096",
    "max_response_length": "8192",
    "max_concurrency": "256",
    "vllm_max_num_seqs": "32",
    "vllm_max_num_batched_tokens": "256",
    "env_max_steps": "15",
    "temperature": "0.7",
    "top_k": "0",
    "top_p": "1.0",
    "seed": "42",
    "p57_workload_candidate": "m15",
    "p57_data_split": "main",
}
_FIRST_RED_PREFIX = 1226
_HISTORICAL_DEPTH = 1686
_MIN_REPRESENTATIVE_HIT_RATE = 80.0
_SOURCE_RE = re.compile(r"^\[sync\] HEAD=([0-9a-f]{40})$", re.MULTILINE)
_HIT_RE = re.compile(r"Prefix cache hit rate:\s*([0-9.]+)%")


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ClassificationError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ClassificationError(f"invalid {label}: {path}: {exc}") from exc
  _require(isinstance(value, dict), f"{label} is not a JSON object")
  return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing pre-alignment report: {path}")
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ClassificationError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    _require(isinstance(value, dict), f"pre-alignment row {number} is not an object")
    rows.append(value)
  _require(rows, "pre-alignment report contains no records")
  return rows


def _command_fields(raw: str) -> dict[str, str]:
  lines = [line for line in raw.splitlines() if line.startswith("[run] cmd: ")]
  _require(len(lines) == 1, f"expected one run command, found {len(lines)}")
  fields: dict[str, str] = {}
  for token in shlex.split(lines[0].removeprefix("[run] cmd: ")):
    if token.startswith("--") and "=" in token:
      name, value = token[2:].split("=", 1)
      fields[name] = value
  return fields


def _boundary(record: dict[str, Any], name: str) -> dict[str, Any]:
  value = record.get("boundaries", {}).get(name)
  _require(isinstance(value, dict), f"missing {name} boundary")
  _require(value.get("valid") is True, f"{name} shape contract is invalid")
  _require(value.get("finite") is True, f"{name} contains non-finite values")
  _require(int(value.get("differing_bytes", -1)) >= 0, f"{name} byte count is invalid")
  _require(int(value.get("differing_elements", -1)) >= 0, f"{name} element count is invalid")
  return value


def _capture_contract(
    report: dict[str, Any],
    *,
    expected_source_commit: str,
    capsule_path: Path | None,
    require_join: bool,
) -> dict[str, Any]:
  _require(report.get("verdict") == "PASS", "serving capture classifier is not PASS")
  _require(report.get("scope") == "p38-serving-capture", "serving capture scope drifted")
  _require(report.get("program_path") == "standard", "capture did not use standard serving path")
  _require(report.get("source_commit") == expected_source_commit, "capture source commit drifted")
  _require(int(report.get("request_journal_records", 0)) > 0, "request journal is empty")
  _require(int(report.get("incident_ledger_records", 0)) > 0, "incident ledger is empty")
  _require(bool(report.get("records")), "serving capture has no strata records")
  capsule = report.get("mismatch_capsule")
  joins = report.get("incident_exact_joins", [])
  if require_join:
    _require(capsule_path is not None, "red run supplied no mismatch capsule")
    _require(capsule_path.is_file(), f"mismatch capsule is absent: {capsule_path}")
    _require(isinstance(capsule, dict), "capture report did not attest its mismatch capsule")
    _require(capsule.get("sha256") == _sha256(capsule_path), "mismatch capsule SHA drifted")
    _require(bool(joins), "red run has no exact incident-to-capsule join")
    _require(not report.get("incident_missing_joins"), "red run has unjoined mismatch positions")
  else:
    _require(capsule is None, "exact run unexpectedly produced a mismatch capsule")
    _require(not joins, "exact run unexpectedly reports mismatch joins")
  return {
      "sha256": None,
      "request_journal_records": int(report["request_journal_records"]),
      "incident_ledger_records": int(report["incident_ledger_records"]),
      "incident_exact_joins": len(joins),
      "joined_source_rows": sorted({int(item["source_row"]) for item in joins}),
      "prefix_bounds": report.get("prefix_bounds"),
  }


def classify(
    *,
    raw_path: Path,
    report_path: Path,
    capture_classification_path: Path,
    arm: str,
    expected_source_commit: str,
    capsule_path: Path | None = None,
) -> dict[str, Any]:
  _require(arm in ("off", "on"), f"invalid APC arm: {arm}")
  _require(re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is not None,
           "expected source commit must be a full lowercase SHA")
  _require(raw_path.is_file() and raw_path.stat().st_size > 0, f"missing raw log: {raw_path}")
  raw = raw_path.read_text(encoding="utf-8", errors="replace")
  source_commits = _SOURCE_RE.findall(raw)
  _require(source_commits == [expected_source_commit],
           f"runtime source receipt drifted: {source_commits}")

  command = _command_fields(raw)
  wrong_command = {
      name: command.get(name)
      for name, expected in _REQUIRED_COMMAND.items()
      if command.get(name) != expected
  }
  _require(not wrong_command, f"M15 command geometry drifted: {wrong_command}")

  enabled = 1 if arm == "on" else 0
  apc_marker = (
      f"[P3_APC_CONFIG] enabled={enabled} workload=frozenlake "
      "reader=train_frozenlake_qwen3"
  )
  _require(raw.count(apc_marker) == 1, "APC runtime marker count drifted")
  _require(
      f"[P3_APC_CONFIG] enabled={1 - enabled} " not in raw,
      "opposite APC arm marker is present",
  )
  _require(
      raw.count(
          "[VLLM.LOGPROB_REQUEST] return_logprobs=1 sampled=1 "
          "prompt=None host_extraction=enabled"
      ) == 1,
      "A sampled-logprob request contract is absent or duplicated",
  )
  _require(
      raw.count(
          "[CAN" "ON_APC_M15_A_CONTRACT] prompt_logprobs=None logprobs=1 "
          "skip_reading_prefix_cache=False"
      ) == 1,
      "A cache-readable request receipt is absent or duplicated",
  )
  _require(
      raw.count(
          "[CAN" "ON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
          "all_num_cached_tokens_zero=True"
      ) >= 1,
      "B full-reset receipt is absent",
  )
  _require(
      raw.count(
          f"[CAN" f"ON_APC_M15_TARGET_CONTRACT] arm={arm} topology=DP8xTP8 "
          "workload=m15/main backward=0 optimizer_commits=0"
      ) == 1,
      "M15 target-debug identity is absent or duplicated",
  )
  _require(
      raw.count("[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0") == 1,
      "controlled zero-commit exit receipt is absent or duplicated",
  )
  _require("OPTIMIZER_COMMIT" not in raw, "optimizer commit marker appeared")

  records = _load_jsonl(report_path)
  ab_bytes = []
  ab_elements = []
  bc_bytes = []
  rounds = []
  min_prefixes = []
  max_prefixes = []
  rows_reaching_depth = []
  for index, record in enumerate(records):
    _require(int(record.get("step", -1)) == 0, f"record {index} is not step zero")
    _require(int(record.get("N_action", 0)) > 0, f"record {index} has no action tokens")
    ab = _boundary(record, "S_decode_vs_S_prefill")
    bc = _boundary(record, "S_prefill_vs_T_old")
    _require(int(bc["differing_bytes"]) == 0, "B-C changed; run is not APC-isolated")
    ab_bytes.append(int(ab["differing_bytes"]))
    ab_elements.append(int(ab["differing_elements"]))
    bc_bytes.append(int(bc["differing_bytes"]))
    rounds.append(int(record.get("diagnostic_round", -1)))
    geometry = record.get("action_geometry", {})
    _require(geometry.get("valid") is True, f"record {index} action geometry is invalid")
    min_prefixes.append(int(geometry.get("min_logical_kv_prefix_length", -1)))
    max_prefixes.append(int(geometry.get("max_logical_kv_prefix_length", -1)))
    rows_reaching_depth.append(int(geometry.get("rows_reaching_1686", -1)))
    hashes = record.get("hashes", {})
    _require(
        all(hashes.get(name) for name in (
            "S_decode", "S_prefill", "T_old", "tokens", "action_mask",
            "policy_version",
        )),
        f"record {index} identity hashes are incomplete",
    )
  _require(rounds == list(range(len(records))), f"diagnostic rounds are not ordered: {rounds}")

  hit_rates = [float(value) for value in _HIT_RE.findall(raw)]
  if arm == "on":
    _require(hit_rates and max(hit_rates) > 0.0, "APC-on run observed no cache hit")
  else:
    _require(not hit_rates or max(hit_rates) == 0.0, "APC-off control reported a cache hit")

  is_red = any(value > 0 for value in ab_bytes)
  capture = _load_json(capture_classification_path, "serving capture classification")
  capture_summary = _capture_contract(
      capture,
      expected_source_commit=expected_source_commit,
      capsule_path=capsule_path,
      require_join=is_red,
  )
  capture_summary["sha256"] = _sha256(capture_classification_path)

  if arm == "off":
    _require(not is_red, "APC-off control has an A-B byte difference")
    status = "CONTROL_GREEN"
  elif is_red:
    status = "FRESH_TARGET_RED_FROZEN"
  else:
    _require(min(min_prefixes) <= _FIRST_RED_PREFIX,
             "APC-on exact run did not cover the historical first-red prefix")
    _require(max(max_prefixes) >= _HISTORICAL_DEPTH,
             "APC-on exact run did not reach the historical deep band")
    _require(max(rows_reaching_depth) > 0,
             "APC-on exact run has no rows at the historical deep band")
    _require(max(hit_rates) >= _MIN_REPRESENTATIVE_HIT_RATE,
             "APC-on exact run did not reproduce representative cache occupancy")
    status = "TARGET_NOT_REPRODUCED"

  return {
      "schema": "m15-apc-target-run-classification-v1",
      "status": status,
      "arm": arm,
      "source_commit": expected_source_commit,
      "records": len(records),
      "diagnostic_rounds": rounds,
      "a_b_differing_bytes": ab_bytes,
      "a_b_differing_elements": ab_elements,
      "b_c_differing_bytes": bc_bytes,
      "min_logical_kv_prefix_length": min(min_prefixes),
      "max_logical_kv_prefix_length": max(max_prefixes),
      "max_rows_reaching_1686": max(rows_reaching_depth),
      "prefix_cache_hit_rates_percent": hit_rates,
      "capture": capture_summary,
      "artifacts": {
          "raw_sha256": _sha256(raw_path),
          "pre_alignment_sha256": _sha256(report_path),
          "mismatch_capsule_sha256": (
              _sha256(capsule_path)
              if capsule_path is not None and capsule_path.is_file()
              else None
          ),
      },
      "claim": (
          "fresh target red with exact incident joins; mechanism not localized"
          if status == "FRESH_TARGET_RED_FROZEN"
          else "bounded target observation; not an APC repair or certification"
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", required=True, type=Path)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--capture-classification", required=True, type=Path)
  parser.add_argument("--mismatch-capsule", type=Path)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--expected-source-commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite classification: {args.output}")
  try:
    result = classify(
        raw_path=args.raw,
        report_path=args.report,
        capture_classification_path=args.capture_classification,
        arm=args.arm,
        expected_source_commit=args.expected_source_commit,
        capsule_path=args.mismatch_capsule,
    )
  except (ClassificationError, OSError, ValueError) as exc:
    result = {
        "schema": "m15-apc-target-run-classification-v1",
        "status": "INCONCLUSIVE",
        "error": str(exc),
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(result, sort_keys=True))
  if result["status"] == "INCONCLUSIVE":
    raise SystemExit(2)


if __name__ == "__main__":
  main()
