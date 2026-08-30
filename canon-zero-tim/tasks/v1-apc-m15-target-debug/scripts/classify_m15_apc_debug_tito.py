#!/usr/bin/env python3
"""Fail-closed exact-TiTO postflight for M15 APC layer re-baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")
_TOKEN_PREFIX = "[CANON_M15_TOKEN_CONTINUITY] "
_ROUND_PREFIX = "[CANON_P38] PRECHECK_ROUND_COMPLETE "


class TitoAuditError(RuntimeError):
  """Raised when exact token continuity is absent or ambiguous."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise TitoAuditError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def classify(
    *,
    run_log: Path,
    arm: str,
    expected_rounds: int = 3,
    scope: str = "target",
) -> dict:
  _require(run_log.is_file() and run_log.stat().st_size > 0,
           "run log is absent")
  _require(arm in ("off", "on"), "APC arm must be off or on")
  _require(scope in ("target", "onehost"), "TiTO scope must be target or onehost")
  _require(expected_rounds == 3, "TiTO layer re-baseline requires three rounds")
  text = run_log.read_text(encoding="utf-8", errors="strict")
  env_marker = (
      "[env] M15 APC debug exact TITO enabled mode=exact "
      f"arm={arm} observer=layer rounds=3"
      if scope == "target"
      else "[M15.E0V.ONEHOST] exact TITO enabled mode=exact "
      f"arm={arm} topology=DP1xTP4 rounds=3"
  )
  _require(text.count(env_marker) == 1,
           f"exactly one {scope} TiTO environment receipt is required")

  current_round = 0
  receipt_counts = [0] * expected_rounds
  round_markers = []
  total_receipts = 0
  for line_number, line in enumerate(text.splitlines(), start=1):
    if line.startswith(_TOKEN_PREFIX):
      _require(current_round < expected_rounds,
               "token receipt appeared after the final diagnostic round")
      fields = dict(_FIELD_RE.findall(line.removeprefix(_TOKEN_PREFIX)))
      _require(
          fields.get("mode") == "exact"
          and fields.get("verdict") == "TOKEN_STREAM_EQUAL"
          and fields.get("first_mismatch") == "-1"
          and fields.get("actual_tokens") == fields.get("expected_tokens")
          and fields.get("actual_sha256") == fields.get("expected_sha256")
          and fields.get("actual_token") == "NA"
          and fields.get("expected_token") == "NA"
          and re.fullmatch(r"[0-9a-f]{64}", fields.get("actual_sha256", ""))
              is not None,
          f"non-exact token receipt at line {line_number}",
      )
      receipt_counts[current_round] += 1
      total_receipts += 1
    elif line.startswith(_ROUND_PREFIX):
      fields = dict(_FIELD_RE.findall(line.removeprefix(_ROUND_PREFIX)))
      expected = f"{current_round + 1}/{expected_rounds}"
      _require(fields.get("round") == expected,
               f"diagnostic round sequence drifted at line {line_number}")
      _require(receipt_counts[current_round] > 0,
               f"diagnostic round {current_round} has no exact TiTO receipt")
      _require(fields.get("backward") == "0"
               and fields.get("optimizer_commits") == "0",
               "diagnostic round changed backward/optimizer scope")
      round_markers.append(fields["round"])
      current_round += 1

  _require(current_round == expected_rounds,
           "not all diagnostic rounds completed")
  _require(round_markers == ["1/3", "2/3", "3/3"],
           "diagnostic round markers are incomplete")
  _require(total_receipts == sum(receipt_counts) and total_receipts > 0,
           "exact TiTO receipt accounting drifted")
  return {
      "schema": "m15-apc-debug-tito-postflight-v1",
      "status": "PASS",
      "scope": scope,
      "arm": arm,
      "mode": "exact",
      "observer": "layer" if scope == "target" else None,
      "topology": "DP8xTP8" if scope == "target" else "DP1xTP4",
      "diagnostic_rounds": expected_rounds,
      "round_receipt_counts": receipt_counts,
      "total_exact_equal_receipts": total_receipts,
      "different_or_malformed_receipts": 0,
      "backward": 0,
      "optimizer_commits": 0,
      "program_identity_changed": True,
      "historical_1226_prefix_reused": False,
      "target_pass": False,
      "numerical_repair_authorized": False,
      "run_log_sha256": _sha256(run_log),
      "run_log_bytes": run_log.stat().st_size,
      "run_log_lines": len(text.splitlines()),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--expected-rounds", type=int, default=3)
  parser.add_argument("--scope", choices=("target", "onehost"), default="target")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  try:
    report = classify(
        run_log=args.run_log,
        arm=args.arm,
        expected_rounds=args.expected_rounds,
        scope=args.scope,
    )
  except (OSError, UnicodeError, TitoAuditError) as error:
    print(f"[M15.E0V.TITO] FAIL {error}")
    return 2
  args.output.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(
      "[M15.E0V.TITO] PASS "
      f"scope={report['scope']} arm={report['arm']} "
      f"rounds={report['diagnostic_rounds']} "
      f"exact_equal_receipts={report['total_exact_equal_receipts']} "
      "backward=0 optimizer_commits=0 historical_prefix_reused=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
