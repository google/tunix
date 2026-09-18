#!/usr/bin/env python3
"""Fail-closed matched-workload comparison for Phase3 APC performance arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


class PairError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise PairError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
  _require(path.is_file(), f"missing JSON artifact: {path}")
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
  return value


def _read_records(path: Path) -> list[dict]:
  _require(path.is_file(), f"missing pre-alignment report: {path}")
  records = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  _require(len(records) == 3, f"expected three records in {path}")
  _require(
      [record.get("diagnostic_round") for record in records] == [0, 1, 2],
      f"diagnostic rounds drifted in {path}",
  )
  return records


def _parse_raw(path: Path) -> dict:
  _require(path.is_file(), f"missing raw log: {path}")
  text = path.read_text(encoding="utf-8", errors="replace")
  _require(
      text.count(
          "[P3.APC] performance_contract=greedy-matched-v1 "
          "temperature=0.0 max_concurrency=1"
      ) == 1,
      f"greedy matched-workload marker drifted in {path}",
  )
  rounds = []
  rollout_seconds = []
  rescore_seconds = []
  rollout_re = re.compile(
      r"\[PERF\] stage=rollout_generate seconds=([0-9.]+) rows=([0-9]+)"
  )
  rescore_re = re.compile(
      r"\[PERF\] step=[0-9]+ stage=rescore_b seconds=([0-9.]+) rows=([0-9]+)"
  )
  round_re = re.compile(
      r"\[CANON_P38\] PRECHECK_ROUND_COMPLETE round=([123])/3"
  )
  for line in text.splitlines():
    rollout_match = rollout_re.search(line)
    if rollout_match:
      _require(int(rollout_match.group(2)) == 1, "rollout row geometry drifted")
      rollout_seconds.append(float(rollout_match.group(1)))
    rescore_match = rescore_re.search(line)
    if rescore_match:
      _require(int(rescore_match.group(2)) == 4, "B-rescore row geometry drifted")
      rescore_seconds.append(float(rescore_match.group(1)))
    round_match = round_re.search(line)
    if round_match:
      expected = len(rounds) + 1
      _require(int(round_match.group(1)) == expected, "raw round order drifted")
      _require(rollout_seconds, f"round {expected - 1} has no rollout timing")
      _require(
          len(rescore_seconds) == 1,
          f"round {expected - 1} has {len(rescore_seconds)} B-rescore timings",
      )
      rounds.append({
          "round": expected - 1,
          "rollout_calls": len(rollout_seconds),
          "rollout_seconds": rollout_seconds,
          "rollout_total_seconds": round(sum(rollout_seconds), 6),
          "rescore_b_seconds": rescore_seconds[0],
      })
      rollout_seconds = []
      rescore_seconds = []
  _require(len(rounds) == 3, f"raw log did not complete three rounds: {path}")
  _require(not rollout_seconds and not rescore_seconds, "unsealed PERF samples")
  elapsed_matches = re.findall(
      r"\[P3\.APC\] docker_exit=42 elapsed_seconds=([0-9]+)", text
  )
  _require(len(elapsed_matches) == 1, "controlled wall-time marker drifted")
  return {
      "rounds": rounds,
      "elapsed_seconds": int(elapsed_matches[0]),
      "raw_sha256": _sha256(path),
  }


def compare(
    control_raw: Path,
    control_report: Path,
    control_classification: Path,
    apc_raw: Path,
    apc_report: Path,
    apc_classification: Path,
) -> dict:
  control_class = _read_json(control_classification)
  apc_class = _read_json(apc_classification)
  _require(control_class.get("status") == "CONTROL_GREEN",
           "control arm is not CONTROL_GREEN")
  _require(control_class.get("expect_apc") is False,
           "control classification is not APC-off")
  _require(apc_class.get("status") == "GB_GC_CERTIFICATION_GREEN",
           "APC arm is not GB_GC_CERTIFICATION_GREEN")
  _require(apc_class.get("expect_apc") is True,
           "APC classification is not APC-on")

  control_records = _read_records(control_report)
  apc_records = _read_records(apc_report)
  hash_fields = (
      "tokens", "action_mask", "policy_version", "S_decode", "S_prefill",
      "T_old",
  )
  round_inputs = []
  for index, (control, apc) in enumerate(zip(control_records, apc_records)):
    _require(control.get("verdict") == apc.get("verdict") == "PASS",
             f"round {index} alignment verdict is not PASS in both arms")
    _require(control.get("N_action") == apc.get("N_action"),
             f"round {index} N_action differs across arms")
    equal = {
        field: control.get("hashes", {}).get(field)
        == apc.get("hashes", {}).get(field)
        for field in hash_fields
    }
    _require(all(equal.values()),
             f"round {index} input/value hashes differ across arms: {equal}")
    round_inputs.append({
        "round": index,
        "N_action": int(control["N_action"]),
        "hashes_equal": equal,
    })

  control_perf = _parse_raw(control_raw)
  apc_perf = _parse_raw(apc_raw)
  comparisons = []
  for index, (control, apc) in enumerate(
      zip(control_perf["rounds"], apc_perf["rounds"])
  ):
    _require(
        control["rollout_calls"] == apc["rollout_calls"],
        f"round {index} rollout call count differs across arms",
    )
    baseline = control["rollout_total_seconds"]
    treatment = apc["rollout_total_seconds"]
    comparisons.append({
        "round": index,
        "N_action": round_inputs[index]["N_action"],
        "rollout_calls": control["rollout_calls"],
        "control_rollout_seconds": baseline,
        "apc_rollout_seconds": treatment,
        "delta_seconds": round(treatment - baseline, 6),
        "speedup_percent": round(100.0 * (baseline - treatment) / baseline, 3),
        "control_rescore_b_seconds": control["rescore_b_seconds"],
        "apc_rescore_b_seconds": apc["rescore_b_seconds"],
    })

  # Round 0 carries first-call compilation.  The pre-registered performance
  # decision uses rounds 1+2, while retaining round 0 and full wall time.
  control_steady = sum(item["control_rollout_seconds"] for item in comparisons[1:])
  apc_steady = sum(item["apc_rollout_seconds"] for item in comparisons[1:])
  steady_improvements = [item["speedup_percent"] > 0 for item in comparisons[1:]]
  decision = (
      "KEEP_ONEHOST_PROXY"
      if apc_steady < control_steady and all(steady_improvements)
      else "REVERT_ONEHOST_PROXY"
  )
  return {
      "schema": "phase3-apc-matched-perf-pair-v1",
      "status": "MATCHED_INPUTS",
      "decision": decision,
      "claim": "one-host greedy DP1xTP4 proxy only; profile runs cannot replace this timing pair",
      "round_inputs": round_inputs,
      "round_comparisons": comparisons,
      "steady_rounds": [1, 2],
      "control_steady_rollout_seconds": round(control_steady, 6),
      "apc_steady_rollout_seconds": round(apc_steady, 6),
      "steady_speedup_percent": round(
          100.0 * (control_steady - apc_steady) / control_steady, 3
      ),
      "control_elapsed_seconds": control_perf["elapsed_seconds"],
      "apc_elapsed_seconds": apc_perf["elapsed_seconds"],
      "artifact_sha256": {
          "control_raw": control_perf["raw_sha256"],
          "control_report": _sha256(control_report),
          "control_classification": _sha256(control_classification),
          "apc_raw": apc_perf["raw_sha256"],
          "apc_report": _sha256(apc_report),
          "apc_classification": _sha256(apc_classification),
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--control-raw", type=Path, required=True)
  parser.add_argument("--control-report", type=Path, required=True)
  parser.add_argument("--control-classification", type=Path, required=True)
  parser.add_argument("--apc-raw", type=Path, required=True)
  parser.add_argument("--apc-report", type=Path, required=True)
  parser.add_argument("--apc-classification", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite comparison: {args.output}")
  try:
    result = compare(
        args.control_raw,
        args.control_report,
        args.control_classification,
        args.apc_raw,
        args.apc_report,
        args.apc_classification,
    )
  except (PairError, json.JSONDecodeError, OSError, ZeroDivisionError) as exc:
    result = {
        "schema": "phase3-apc-matched-perf-pair-v1",
        "status": "INCONCLUSIVE_INPUT_MISMATCH",
        "error": str(exc),
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  if result["status"] != "MATCHED_INPUTS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
