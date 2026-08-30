#!/usr/bin/env python3
"""Classify one E0v exact-TiTO one-host arm without assuming its A-B outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


class OnehostArmError(RuntimeError):
  """Raised when one-host arm evidence violates its strict carrier contract."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise OnehostArmError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(*, raw_path: Path, report_path: Path, arm: str) -> dict:
  _require(arm in ("off", "on"), "APC arm must be off or on")
  _require(raw_path.is_file(), "raw log is absent")
  _require(report_path.is_file(), "pre-alignment report is absent")
  raw = raw_path.read_text(encoding="utf-8", errors="strict")
  records = [
      json.loads(line)
      for line in report_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  _require(len(records) == 3, "one-host arm requires exactly three records")
  enabled = arm == "on"
  marker = (
      f"[P3_APC_CONFIG] enabled={int(enabled)} "
      "workload=frozenlake reader=train_frozenlake_qwen3"
  )
  opposite = (
      f"[P3_APC_CONFIG] enabled={int(not enabled)} "
      "workload=frozenlake reader=train_frozenlake_qwen3"
  )
  _require(raw.count(marker) == 1, "APC runtime marker count drifted")
  _require(opposite not in raw, "opposite APC runtime marker is present")
  hit_rates = [
      float(value)
      for value in re.findall(r"Prefix cache hit rate:\s*([0-9.]+)%", raw)
  ]
  if enabled:
    _require(hit_rates and max(hit_rates) > 0.0, "APC-on observed no cache hit")

  b_marker = (
      "[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
      "all_num_cached_tokens_zero=True"
  )
  round_prefix = "[CANON_P38] PRECHECK_ROUND_COMPLETE "
  current_round = 0
  b_receipts = [0, 0, 0]
  for line in raw.splitlines():
    if line == b_marker:
      _require(current_round < 3, "B full-reset receipt appeared after round 3")
      b_receipts[current_round] += 1
    elif line.startswith(round_prefix):
      current_round += 1
  _require(current_round == 3, "raw diagnostic round markers are incomplete")
  _require(all(value > 0 for value in b_receipts),
           "each round requires a B full-reset/zero-cached-token receipt")

  a_b = []
  b_c = []
  rounds = []
  actions = []
  for index, record in enumerate(records):
    _require(record.get("verdict") == "PASS", f"alignment verdict failed in round {index}")
    _require(record.get("blocking_reds") == [], f"blocking red present in round {index}")
    n_action = int(record.get("N_action", 0))
    _require(n_action > 0, f"round {index} has no action rows")
    actions.append(n_action)
    rounds.append(int(record.get("diagnostic_round", -1)))
    boundaries = record.get("boundaries", {})
    for name, destination in (
        ("S_decode_vs_S_prefill", a_b),
        ("S_prefill_vs_T_old", b_c),
    ):
      boundary = boundaries.get(name, {})
      _require(boundary.get("valid") is True, f"{name} shape is invalid in round {index}")
      _require(boundary.get("finite") is True, f"{name} is non-finite in round {index}")
      destination.append(int(boundary.get("differing_bytes", -1)))
    _require(b_c[-1] == 0, f"B-C changed in round {index}")
    hashes = record.get("hashes", {})
    masked = record.get("masked_hashes", {})
    _require(
        all(hashes.get(key) for key in (
            "S_decode", "S_prefill", "T_old", "tokens", "action_mask", "policy_version"
        )),
        f"hash attestation is incomplete in round {index}",
    )
    _require(
        all(masked.get(key) for key in ("S_decode", "S_prefill", "T_old")),
        f"masked hash attestation is incomplete in round {index}",
    )
  _require(rounds == [0, 1, 2], "diagnostic round sequence drifted")
  if arm == "off":
    _require(a_b == [0, 0, 0], "APC-off control A-B is red")
    status = "CONTROL_GREEN"
  else:
    status = "TREATMENT_EXACT" if a_b == [0, 0, 0] else "TREATMENT_RED"

  return {
      "schema": "m15-e0v-tito-onehost-arm-classification-v1",
      "status": status,
      "arm": arm,
      "records": 3,
      "diagnostic_rounds": rounds,
      "actions": actions,
      "a_b_differing_bytes": a_b,
      "b_c_differing_bytes": b_c,
      "prefix_cache_hit_rates_percent": hit_rates,
      "max_prefix_cache_hit_rate_percent": max(hit_rates) if hit_rates else None,
      "b_full_reset_receipt_counts": b_receipts,
      "raw_sha256": _sha256(raw_path),
      "pre_alignment_sha256": _sha256(report_path),
      "first_red_localized": False,
      "target_executed": False,
      "numerical_repair_authorized": False,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", required=True, type=Path)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  try:
    report = classify(raw_path=args.raw, report_path=args.report, arm=args.arm)
  except (OSError, UnicodeError, json.JSONDecodeError, OnehostArmError) as error:
    print(f"[M15.E0V.ONEHOST.ARM] INCONCLUSIVE {error}")
    return 2
  args.output.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(
      "[M15.E0V.ONEHOST.ARM] CLASSIFIED "
      f"arm={report['arm']} status={report['status']} "
      f"A-B={report['a_b_differing_bytes']} B-C={report['b_c_differing_bytes']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
