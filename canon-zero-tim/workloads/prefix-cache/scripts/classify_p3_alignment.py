#!/usr/bin/env python3
"""Classify the bounded Phase3 APC reproduction without relaxing alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


class ClassificationError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ClassificationError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(
    raw_path: Path,
    report_path: Path,
    expect_apc: bool,
    purpose: str = "reproduction",
) -> dict:
  _require(
      purpose in ("reproduction", "certification"),
      f"unknown classification purpose: {purpose}",
  )
  _require(raw_path.is_file(), f"raw log is absent: {raw_path}")
  _require(report_path.is_file(), f"pre-alignment report is absent: {report_path}")
  raw = raw_path.read_text(encoding="utf-8", errors="replace")
  records = [
      json.loads(line)
      for line in report_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  _require(records, "pre-alignment report contains no records")

  expected_marker = (
      f"[P3_APC_CONFIG] enabled={int(expect_apc)} "
      "workload=frozenlake reader=train_frozenlake_qwen3"
  )
  _require(raw.count(expected_marker) == 1, "APC runtime marker count drifted")
  opposite_marker = (
      f"[P3_APC_CONFIG] enabled={int(not expect_apc)} "
      "workload=frozenlake reader=train_frozenlake_qwen3"
  )
  _require(opposite_marker not in raw, "opposite APC arm marker is present")

  hit_rates = [
      float(value)
      for value in re.findall(r"Prefix cache hit rate:\s*([0-9.]+)%", raw)
  ]
  if expect_apc:
    _require(hit_rates, "APC-on run emitted no prefix-cache hit-rate metric")
    _require(max(hit_rates) > 0.0, "APC-on run observed no positive cache hit")

  ab_differing = []
  bc_differing = []
  rounds = []
  for record in records:
    _require(int(record.get("N_action", 0)) > 0, "record has no action rows")
    boundaries = record.get("boundaries", {})
    ab = boundaries.get("S_decode_vs_S_prefill", {})
    bc = boundaries.get("S_prefill_vs_T_old", {})
    for name, boundary in (("A-B", ab), ("B-C", bc)):
      _require(boundary.get("valid") is True, f"{name} shape contract is invalid")
      _require(boundary.get("finite") is True, f"{name} contains non-finite values")
    ab_differing.append(int(ab.get("differing_bytes", -1)))
    bc_differing.append(int(bc.get("differing_bytes", -1)))
    _require(bc_differing[-1] == 0, "B-C changed; APC reproduction is confounded")
    hashes = record.get("hashes", {})
    masked_hashes = record.get("masked_hashes", {})
    _require(
        all(hashes.get(key) for key in (
            "S_decode", "S_prefill", "T_old", "tokens", "action_mask",
            "policy_version",
        )),
        "input/value hash attestation is incomplete",
    )
    _require(
        all(masked_hashes.get(key) for key in (
            "S_decode", "S_prefill", "T_old",
        )),
        "masked hash attestation is incomplete",
    )
    rounds.append(int(record.get("diagnostic_round", -1)))

  if expect_apc:
    if purpose == "certification":
      _require(all(value == 0 for value in ab_differing),
               "APC-on certification observed an A-B byte difference")
      _require(len(records) == 3 and rounds == [0, 1, 2],
               "APC-on certification did not complete three ordered rounds")
      status = "GB_GC_CERTIFICATION_GREEN"
    else:
      _require(any(value > 0 for value in ab_differing),
               "APC-on did not reproduce an A-B byte difference")
      status = "REPRODUCED_RED"
  else:
    _require(all(value == 0 for value in ab_differing),
             "APC-off control is not byte-exact")
    _require(len(records) == 3 and rounds == [0, 1, 2],
             "APC-off control did not complete three ordered rounds")
    status = "CONTROL_GREEN"

  return {
      "schema": "phase3-apc-alignment-classification-v1",
      "status": status,
      "expect_apc": expect_apc,
      "purpose": purpose,
      "records": len(records),
      "diagnostic_rounds": rounds,
      "a_b_differing_bytes": ab_differing,
      "b_c_differing_bytes": bc_differing,
      "prefix_cache_hit_rates_percent": hit_rates,
      "max_prefix_cache_hit_rate_percent": max(hit_rates) if hit_rates else None,
      "raw_sha256": _sha256(raw_path),
      "pre_alignment_sha256": _sha256(report_path),
      "claim": (
          "P3.3 one-host multi-round G-B/G-C evidence only; not at-scale certification"
          if purpose == "certification"
          else "P3.1 reproduction only; this is not an APC fix or certification"
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--expect-apc", choices=("0", "1"), required=True)
  parser.add_argument(
      "--purpose",
      choices=("reproduction", "certification"),
      default="reproduction",
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite classification: {args.output}")
  try:
    result = classify(
        args.raw, args.report, args.expect_apc == "1", args.purpose
    )
  except (ClassificationError, json.JSONDecodeError, OSError) as exc:
    result = {
        "schema": "phase3-apc-alignment-classification-v1",
        "status": "INCONCLUSIVE",
        "error": str(exc),
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  if result["status"] == "INCONCLUSIVE":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
