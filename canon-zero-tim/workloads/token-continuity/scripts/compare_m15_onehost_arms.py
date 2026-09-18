#!/usr/bin/env python3
"""Compare ordered M15 legacy/exact one-host token and trajectory receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(
    r"^\[CANON_M15_TOKEN_CONTINUITY\] mode=(verify|exact) turn=(\d+) "
    r"verdict=TOKEN_STREAM_EQUAL actual_tokens=(\d+) expected_tokens=(\d+) "
    r"actual_sha256=([0-9a-f]{64}) expected_sha256=([0-9a-f]{64}) "
)


def _token_receipts(root: Path, expected_mode: str) -> list[dict[str, Any]]:
  receipts = []
  for line in (root / "raw.log").read_text(
      encoding="utf-8", errors="replace"
  ).splitlines():
    match = TOKEN_RE.search(line)
    if not match:
      continue
    mode, turn, actual_n, expected_n, actual_sha, expected_sha = match.groups()
    if mode != expected_mode:
      raise ValueError(f"{root}: unexpected receipt mode {mode!r}")
    if actual_n != expected_n or actual_sha != expected_sha:
      raise ValueError(f"{root}: internally unequal receipt at turn {turn}")
    receipts.append({
        "turn": int(turn),
        "tokens": int(actual_n),
        "sha256": actual_sha,
    })
  if not receipts:
    raise ValueError(f"{root}: no token receipts")
  return receipts


def _round_receipts(root: Path) -> list[dict[str, Any]]:
  rows = [
      json.loads(line)
      for line in (root / "pre_alignment.jsonl").read_text().splitlines()
      if line.strip()
  ]
  return [
      {
          "diagnostic_round": row.get("diagnostic_round"),
          "N_action": row.get("N_action"),
          "tokens": row.get("hashes", {}).get("tokens"),
          "action_mask": row.get("hashes", {}).get("action_mask"),
      }
      for row in rows
  ]


def compare(verify_root: Path, exact_root: Path) -> dict[str, Any]:
  verify_tokens = _token_receipts(verify_root, "verify")
  exact_tokens = _token_receipts(exact_root, "exact")
  verify_rounds = _round_receipts(verify_root)
  exact_rounds = _round_receipts(exact_root)
  token_equal = verify_tokens == exact_tokens
  rounds_equal = verify_rounds == exact_rounds
  return {
      "schema": "canon.m15-onehost-verify-exact-comparison.v1",
      "status": "MATCH" if token_equal and rounds_equal else "DIFFERENT",
      "verify_root": str(verify_root),
      "exact_root": str(exact_root),
      "token_receipts": len(verify_tokens),
      "ordered_prompt_receipts_equal": token_equal,
      "round_receipts_equal": rounds_equal,
      "verify_rounds": verify_rounds,
      "exact_rounds": exact_rounds,
      "first_prompt_difference": next(
          (
              index
              for index, pair in enumerate(zip(verify_tokens, exact_tokens))
              if pair[0] != pair[1]
          ),
          None if len(verify_tokens) == len(exact_tokens) else min(
              len(verify_tokens), len(exact_tokens)
          ),
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--verify-root", type=Path, required=True)
  parser.add_argument("--exact-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = compare(args.verify_root, args.exact_root)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(
      "[M15.TITO.CROSS_ARM] "
      f"status={result['status']} receipts={result['token_receipts']} "
      f"prompts_equal={result['ordered_prompt_receipts_equal']} "
      f"rounds_equal={result['round_receipts_equal']}"
  )
  return 0 if result["status"] == "MATCH" else 1


if __name__ == "__main__":
  raise SystemExit(main())
