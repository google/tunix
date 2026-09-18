#!/usr/bin/env python3
"""Classify a strict APC-off M15 one-host verify or exact TiTO arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(
    r"^\[CANON_M15_TOKEN_CONTINUITY\] mode=(verify|exact) turn=(\d+) "
    r"verdict=(TOKEN_STREAM_EQUAL|TOKEN_STREAM_DIFFERENT) .*?"
    r"first_mismatch=(-?\d+)"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, reason: str, reasons: list[str]) -> None:
  if not condition:
    reasons.append(reason)


def classify(
    raw_path: Path, report_path: Path, *, mode: str = "verify"
) -> dict[str, Any]:
  if mode not in ("verify", "exact"):
    raise ValueError(f"unsupported M15 token-continuity mode: {mode!r}")
  raw = raw_path.read_text(encoding="utf-8", errors="replace")
  reasons: list[str] = []
  rows = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
  _require(len(rows) == 3, "pre_alignment.round_count", reasons)
  rounds = []
  for expected_round, row in enumerate(rows):
    boundaries = row.get("boundaries", {})
    ab = boundaries.get("S_decode_vs_S_prefill", {})
    bc = boundaries.get("S_prefill_vs_T_old", {})
    _require(row.get("diagnostic_round") == expected_round, f"round.{expected_round}.index", reasons)
    _require(row.get("verdict") == "PASS", f"round.{expected_round}.verdict", reasons)
    _require(ab.get("differing_bytes") == 0, f"round.{expected_round}.A-B", reasons)
    _require(bc.get("differing_bytes") == 0, f"round.{expected_round}.B-C", reasons)
    _require(ab.get("finite") is True and bc.get("finite") is True, f"round.{expected_round}.finite", reasons)
    rounds.append({
        "round": expected_round,
        "action_tokens": row.get("N_action"),
        "a_b_differing_bytes": ab.get("differing_bytes"),
        "b_c_differing_bytes": bc.get("differing_bytes"),
    })

  receipts = []
  for line in raw.splitlines():
    match = TOKEN_RE.search(line)
    if match:
      receipts.append({
          "mode": match.group(1),
          "turn": int(match.group(2)),
          "verdict": match.group(3),
          "first_mismatch": int(match.group(4)),
      })
  _require(len(receipts) >= 3, "token_receipts.coverage", reasons)
  _require(
      all(item["mode"] == mode for item in receipts),
      "token_receipts.mode",
      reasons,
  )
  _require(all(item["turn"] >= 1 for item in receipts), "token_receipts.turn", reasons)
  _require(
      raw.count("[CANON_P38] PRECHECK_ROUND_COMPLETE ") == 3,
      "round_complete.count",
      reasons,
  )
  _require(
      raw.count("[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True all_num_cached_tokens_zero=True") == 3,
      "B_full_reset.count",
      reasons,
  )
  _require(
      raw.count("[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0") == 1,
      "controlled_exit",
      reasons,
  )
  _require("[CANON_ALIGN] verdict=FAIL" not in raw, "alignment_fail_marker", reasons)
  _require("CANON_VLLM_ENABLE_PREFIX_CACHING=1" not in raw, "apc_on_leak", reasons)
  other_mode = "exact" if mode == "verify" else "verify"
  _require(f"mode={other_mode}" not in raw, "other_mode_leak", reasons)

  different = [item for item in receipts if item["verdict"] == "TOKEN_STREAM_DIFFERENT"]
  if mode == "exact":
    _require(not different, "token_receipts.exact_mismatch", reasons)
  if reasons:
    status = "FAIL"
  elif mode == "exact":
    status = "EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS"
  elif different:
    status = "LEGACY_TOKEN_DRIFT"
  else:
    status = "LEGACY_TOKEN_EQUAL"
  return {
      "schema": "canon.m15-onehost-token-continuity.v2",
      "status": status,
      "scope": (
          "Qwen3-8B M15/main DP1xTP4 APC-off "
          + ("rendered-text observer" if mode == "verify" else "exact TiTO")
      ),
      "mode": mode,
      "rounds": rounds,
      "token_receipts": len(receipts),
      "equal_receipts": len(receipts) - len(different),
      "different_receipts": len(different),
      "first_red": different[0] if different else None,
      "backward": 0,
      "optimizer_commits": 0,
      "target_executed": False,
      "target_pass": False,
      "claim": (
          "one-host token transport and strict alignment only; "
          "DP8xTP8 and production admission unverified"
      ),
      "raw_sha256": _sha256(raw_path),
      "pre_alignment_sha256": _sha256(report_path),
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--mode", choices=("verify", "exact"), default="verify")
  args = parser.parse_args()
  result = classify(args.raw, args.report, mode=args.mode)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(
      "[M15.TITO.ONEHOST] "
      f"{result['status']} rounds={len(result['rounds'])} "
      f"receipts={result['token_receipts']} different={result['different_receipts']} "
      "A-B=0 B-C=0 backward=0 optimizer_commits=0 target_executed=0"
  )
  return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
  raise SystemExit(main())
