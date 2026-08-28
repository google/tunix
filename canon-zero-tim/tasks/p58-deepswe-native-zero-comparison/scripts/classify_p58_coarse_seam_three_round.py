#!/usr/bin/env python3
"""Aggregate three independently sealed P58 coarse seam rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_CONTROLLED_EXIT = (
    "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0"
)


def _signature(record: dict[str, Any]) -> tuple[int | None, str]:
  layer = record.get("layer")
  return (None if layer is None else int(layer), str(record["checkpoint"]))


def classify(*, rounds: list[Path], run_log: Path) -> dict[str, Any]:
  reasons: list[str] = []
  records = [json.loads(path.read_text(encoding="utf-8")) for path in rounds]
  if len(records) != 3:
    reasons.append(f"round_count={len(records)}")
  observed_rounds = sorted(
      int(record.get("diagnostic_round", -1)) for record in records
  )
  if observed_rounds != [0, 1, 2]:
    reasons.append(f"diagnostic_rounds={observed_rounds}")
  for index, record in enumerate(records):
    if record.get("schema") != "canon.p58.coarse-seam-round-classification.v1":
      reasons.append(f"round_{index}_schema")
    if record.get("verdict") != "PASS":
      reasons.append(f"round_{index}_verdict={record.get('verdict')}")
    if int(record.get("alignment", {}).get("b_c_differing_bytes", -1)) != 0:
      reasons.append(f"round_{index}_b_c")
    if int(record.get("alignment", {}).get("a_b_differing_bytes", 0)) <= 0:
      reasons.append(f"round_{index}_a_b")

  signature_sets = [
      {_signature(value) for value in record.get("first_difference_signatures", [])}
      for record in records
  ]
  common = set.intersection(*signature_sets) if signature_sets else set()
  selected = None
  if common:
    selected = min(
        common,
        key=lambda item: (10**9 if item[0] is None else item[0], item[1]),
    )
  else:
    reasons.append("no_common_first_red_signature")

  text = run_log.read_text(encoding="utf-8", errors="replace")
  marker_counts = {
      "precheck_round": text.count("[CANON_P38] PRECHECK_ROUND_COMPLETE "),
      "controlled_exit": text.count(_CONTROLLED_EXIT),
  }
  if marker_counts != {"precheck_round": 3, "controlled_exit": 1}:
    reasons.append(f"marker_counts={marker_counts}")
  forbidden = {
      "p59_backward": "[P59.BACKWARD]",
      "p66_backward": "[P66.BACKWARD]",
      "fixed_lm_head_vjp": "CANON_P38_FIXED_LM_HEAD_VJP=1",
      "optimizer_commit": "optimizer_commits=1",
      "global_step_1": "Global step 1 completed",
  }
  present_forbidden = [
      name for name, marker in forbidden.items() if marker in text
  ]
  if present_forbidden:
    reasons.append(f"forbidden_runtime={present_forbidden}")

  verdict = "PASS" if not reasons else "INCONCLUSIVE"
  outcome = (
      "REPRODUCIBLE_COARSE_FIRST_RED_SIGNATURE"
      if verdict == "PASS"
      else "INCONCLUSIVE_NONREPEATING_OR_INVALID_ROUNDS"
  )
  return {
      "schema": "canon.p58.coarse-seam-three-round-classification.v1",
      "verdict": verdict,
      "outcome": outcome,
      "diagnostic_rounds": observed_rounds,
      "round_reports": [str(path) for path in rounds],
      "round_signature_sets": [
          [
              {"layer": layer, "checkpoint": checkpoint}
              for layer, checkpoint in sorted(
                  values,
                  key=lambda item: (
                      10**9 if item[0] is None else item[0], item[1]
                  ),
              )
          ]
          for values in signature_sets
      ],
      "common_first_red_signatures": [
          {"layer": layer, "checkpoint": checkpoint}
          for layer, checkpoint in sorted(
              common,
              key=lambda item: (
                  10**9 if item[0] is None else item[0], item[1]
              ),
          )
      ],
      "selected_signature": (
          {"layer": selected[0], "checkpoint": selected[1]}
          if selected is not None else None
      ),
      "marker_counts": marker_counts,
      "backward": 0,
      "optimizer_commits": 0,
      "reasons": reasons,
      "next_action": (
          "Run the fine observer only inside the selected coarse interval."
          if verdict == "PASS"
          else "Do not refine; inspect round coverage and repeatability first."
      ),
      "claim_ceiling": (
          "A PASS proves a repeated fingerprint boundary on the exact frozen "
          "P58 DP8xTP8 serving/trainer carrier. It does not prove full-tensor "
          "byte equality, root cause, backward, optimizer, or training."
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--round", action="append", required=True, type=Path)
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {args.output}")
  report = classify(rounds=args.round, run_log=args.run_log)
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P58_COARSE_SEAM_THREE_ROUND_CLASSIFICATION "
      f"verdict={report['verdict']} outcome={report['outcome']} "
      f"selected={report['selected_signature']} backward=0 "
      "optimizer_commits=0",
      flush=True,
  )
  return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
  raise SystemExit(main())
