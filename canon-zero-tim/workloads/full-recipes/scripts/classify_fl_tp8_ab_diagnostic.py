#!/usr/bin/env python3
"""Classify a zero-commit FrozenLake DP8xTP8 pre-backward bisection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(path: Path) -> list[dict]:
  return [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]


def classify(
    *, raw: Path, pre_alignment: Path, workload: str, arm: str, output: Path
) -> dict:
  if arm not in ("p66-off", "serving-scope"):
    raise ValueError("arm must be p66-off or serving-scope")
  text = raw.read_text(encoding="utf-8", errors="replace")
  records = _load_records(pre_alignment)
  errors: list[str] = []
  checked_vma = "0" if arm == "p66-off" else "1"
  p59_only = "0" if arm == "p66-off" else "1"
  marker = (
      f"[V1.FL.AB] profile_resolved arm={arm} workload={workload} "
      f"dp=8 tp=8 checked_vma={checked_vma} vma_p59_only={p59_only} "
      "fixed_ar_gather=1 continue_decode=8 "
      "prefix_cache=0 backward=0 optimizer_commits=0"
  )
  if text.count(marker) != 1:
    errors.append("exact profile marker count must be one")
  if text.count("[CANON_P38] PRECHECK_ROUND_COMPLETE ") != 1:
    errors.append("precheck round count must be one")
  if text.count(
      "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0"
  ) != 1:
    errors.append("controlled zero-commit exit marker count must be one")
  if len(records) != 1:
    errors.append(f"expected one pre-alignment record, got {len(records)}")
    record = {}
  else:
    record = records[0]
  boundaries = record.get("boundaries", {})
  a_b = boundaries.get("S_decode_vs_S_prefill", {})
  b_c = boundaries.get("S_prefill_vs_T_old", {})
  geometry = record.get("action_geometry", {})
  n_action = record.get("N_action", 0)
  if not isinstance(n_action, int) or n_action <= 0:
    errors.append("N_action must be positive")
  if geometry.get("valid") is not True:
    errors.append("action geometry must be valid")
  required_depth = 3936 if workload == "m15" else 1686
  observed_depth = geometry.get("max_logical_kv_prefix_length", -1)
  if not isinstance(observed_depth, int) or observed_depth < required_depth:
    errors.append(
        f"depth is insufficient: observed={observed_depth} required={required_depth}"
    )
  for name, boundary in (("A-B", a_b), ("B-C", b_c)):
    if boundary.get("valid") is not True or boundary.get("finite") is not True:
      errors.append(f"{name} must be valid and finite")
    if not isinstance(boundary.get("differing_bytes"), int):
      errors.append(f"{name} differing_bytes must be an integer")
  if b_c.get("differing_bytes") != 0:
    errors.append("B-C must remain byte-exact")
  if "optimizer_commits=1" in text or "Global step 1 completed" in text:
    errors.append("optimizer activity is forbidden")
  if "[P59.BACKWARD]" in text or "[P66.BACKWARD]" in text:
    errors.append("backward execution is forbidden")
  differing = a_b.get("differing_bytes")
  outcome = (
      "ZERO_TIM_RECOVERED"
      if differing == 0
      else "A_B_RED_REPRODUCED"
      if isinstance(differing, int) and differing > 0
      else "INVALID"
  )
  verdict = "PASS" if not errors and outcome != "INVALID" else "FAIL"
  result = {
      "schema": "v1-fl-tp8-ab-diagnostic-v1",
      "verdict": verdict,
      "outcome": outcome,
      "arm": arm,
      "workload": workload,
      "optimizer_commits": 0,
      "backward": 0,
      "N_action": n_action,
      "max_logical_kv_prefix_length": observed_depth,
      "A_B_differing_bytes": differing,
      "B_C_differing_bytes": b_c.get("differing_bytes"),
      "errors": errors,
  }
  if output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(
      "V1_FL_TP8_AB_CLASSIFICATION "
      f"verdict={verdict} outcome={outcome} arm={arm} workload={workload} "
      f"a_b_bytes={differing} b_c_bytes={b_c.get('differing_bytes')} "
      "backward=0 optimizer_commits=0",
      flush=True,
  )
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", required=True, type=Path)
  parser.add_argument("--pre-alignment", required=True, type=Path)
  parser.add_argument("--workload", required=True, choices=("p45", "m15"))
  parser.add_argument("--arm", required=True, choices=("p66-off", "serving-scope"))
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = classify(
      raw=args.raw,
      pre_alignment=args.pre_alignment,
      workload=args.workload,
      arm=args.arm,
      output=args.output,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
