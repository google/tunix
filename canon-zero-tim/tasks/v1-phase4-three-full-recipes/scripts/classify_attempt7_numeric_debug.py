#!/usr/bin/env python3
"""Classify one P62 DP16xTP4 backward-no-commit raw log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_P62_PREFIX = "[P62.NUMERIC] "
_PRE_PREFIX = "[CANON_ALIGN_PRE_JSON] "
_REVERSE_RE = re.compile(
    r"^\[P33\.DP16\] reverse_group_done group=(\d+)/16 .*"
    r"rank_contributions=16 .*pullback_invocations=1 .*"
    r"replicas_exact=1 "
)
_COMMIT_RE = re.compile(r"optimizer_commits=([1-9][0-9]*)")
_EXPECTED_TREE_STAGES = {
    "loss_cotangent",
    "engine_vjp",
    "trainer_rank_local",
    "fixed_dp_reduced",
    "scaled_microgradient",
    "final_accumulator",
}


def _json_lines(text: str, prefix: str) -> list[dict[str, Any]]:
  records = []
  for line_number, line in enumerate(text.splitlines(), 1):
    if not line.startswith(prefix):
      continue
    payload = line[len(prefix) :]
    if prefix == _P62_PREFIX and (
        payload.startswith("admission workload=gsm8k ")
        or payload.startswith("discard_complete optimizer_commits=0 ")
    ):
      continue
    try:
      record = json.loads(payload)
    except json.JSONDecodeError as error:
      raise ValueError(
          f"malformed JSON after {prefix!r} at line {line_number}: {error}"
      ) from error
    record["_line"] = line_number
    records.append(record)
  return records


def _validate_pre_alignment(record: dict[str, Any]) -> list[str]:
  failures = []
  if record.get("verdict") != "PASS":
    failures.append("pre_alignment_not_pass")
  if int(record.get("N_action", 0)) <= 0:
    failures.append("pre_alignment_no_actions")
  context = record.get("context", {})
  if context.get("mesh") != "16,4":
    failures.append("pre_alignment_wrong_mesh")
  if context.get("run_stage") != "backward-no-commit":
    failures.append("pre_alignment_wrong_stage")
  for name in ("S_decode_vs_S_prefill", "S_prefill_vs_T_old"):
    boundary = record.get("boundaries", {}).get(name, {})
    if (
        boundary.get("valid") is not True
        or boundary.get("finite") is not True
        or boundary.get("differing_elements") != 0
        or boundary.get("differing_bytes") != 0
    ):
      failures.append(f"pre_alignment_{name}_changed")
  return failures


def _validate_p62_records(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
  failures = []
  tree_records = []
  loss_records = [
      record
      for record in records
      if record.get("schema") == "canon-p62-loss-scale-v1"
  ]
  if len(loss_records) != 1:
    failures.append(f"loss_scale_receipts={len(loss_records)}/1")
  else:
    loss = loss_records[0]
    exact = {
        "stage": "loss_scale",
        "dp": 16,
        "tp": 4,
        "global_trajectories": 256,
        "local_trajectories": 16,
        "gradient_groups": 16,
        "global_M": 4096,
        "local_M": 256,
        "expected_accumulator_denominator": 16,
        "expected_streamed_multiplier": 0.0625,
        "loss_denominator": 256.0,
        "loss_scale": 0.00390625,
    }
    changed = {
        key: loss.get(key) for key, expected in exact.items()
        if loss.get(key) != expected
    }
    if changed:
      failures.append(f"loss_scale_contract_changed={changed}")
  for record in records:
    if record.get("schema") != "canon-p62-tree-numeric-v1":
      continue
    tree_records.append(record)
    stage = record.get("stage")
    if stage not in _EXPECTED_TREE_STAGES:
      failures.append(f"unknown_tree_stage={stage!r}")
    if record.get("groups") != 16:
      failures.append(
          f"wrong_group_count_at_line_{record['_line']}={record.get('groups')}"
      )
    group = record.get("group")
    if stage == "loss_cotangent":
      if group != -1:
        failures.append(f"loss_cotangent_group={group}")
    elif not isinstance(group, int) or group < 0 or group >= 16:
      failures.append(f"invalid_group_at_line_{record['_line']}={group}")
    if not isinstance(record.get("all_finite"), bool):
      failures.append(f"missing_all_finite_at_line_{record['_line']}")
    if not isinstance(record.get("naive_norm_finite"), bool):
      failures.append(
          f"missing_naive_norm_finite_at_line_{record['_line']}"
      )
  return failures, tree_records


def classify(text: str) -> dict[str, Any]:
  failures = []
  try:
    pre_records = _json_lines(text, _PRE_PREFIX)
    p62_records = _json_lines(text, _P62_PREFIX)
  except ValueError as error:
    return {
        "schema": "canon-p62-classification-v1",
        "verdict": "FATAL_CONTRACT",
        "failures": [str(error)],
    }
  if len(pre_records) != 1:
    failures.append(f"pre_alignment_receipts={len(pre_records)}/1")
  else:
    failures.extend(_validate_pre_alignment(pre_records[0]))
  admission_count = text.count(
      "[P62.NUMERIC] admission workload=gsm8k dp=16 tp=4 "
      "global_trajectories=256 local_trajectories=16 "
      "global_M=4096 local_M=256 optimizer_commits=0"
  )
  if admission_count != 1:
    failures.append(f"p62_admission={admission_count}/1")
  if any(
      "CANON_ALIGN" in line
      and (
          "verdict=FAIL" in line
          or '"verdict":"FAIL"' in line
          or '"verdict": "FAIL"' in line
      )
      for line in text.splitlines()
  ):
    failures.append("real_alignment_fail")
  commit_matches = [int(value) for value in _COMMIT_RE.findall(text)]
  if commit_matches:
    failures.append(f"optimizer_commit_violation={commit_matches}")
  if "[CANON_UPDATE_JSON]" in text or "[P28.G6] UPDATE" in text:
    failures.append("optimizer_update_receipt_present")
  record_failures, tree_records = _validate_p62_records(p62_records)
  failures.extend(record_failures)
  if failures:
    return {
        "schema": "canon-p62-classification-v1",
        "verdict": "FATAL_CONTRACT",
        "failures": failures,
        "p62_records": len(p62_records),
    }

  first_nonfinite = next(
      (record for record in tree_records if not record["all_finite"]), None
  )
  first_naive_overflow = next(
      (
          record for record in tree_records
          if record["all_finite"] and not record["naive_norm_finite"]
      ),
      None,
  )
  reverse_groups = {
      int(match.group(1))
      for line in text.splitlines()
      if (match := _REVERSE_RE.match(line))
  }
  discard_count = text.count(
      "[P62.NUMERIC] discard_complete optimizer_commits=0 "
      "microsteps=16 denominator=16.0"
  )
  final_records = [
      record for record in tree_records
      if record.get("stage") == "final_accumulator"
  ]
  complete = (
      reverse_groups == set(range(1, 17))
      and len(final_records) == 1
      and final_records[0].get("accumulator_denominator") == 16.0
      and discard_count == 1
  )
  common = {
      "schema": "canon-p62-classification-v1",
      "failures": [],
      "pre_alignment_actions": pre_records[0]["N_action"],
      "p62_records": len(p62_records),
      "tree_records": len(tree_records),
      "reverse_groups": sorted(reverse_groups),
      "discard_count": discard_count,
      "optimizer_commits": 0,
  }
  if first_nonfinite is not None:
    return {
        **common,
        "verdict": "ROOT_LOCALIZED_NONFINITE",
        "first_red": {
            key: first_nonfinite.get(key)
            for key in (
                "_line",
                "stage",
                "group",
                "first_nonfinite",
                "first_nonfinite_rank",
                "max_abs",
                "stable_norm",
            )
        },
    }
  if first_naive_overflow is not None:
    return {
        **common,
        "verdict": "FINITE_NAIVE_L2_OVERFLOW",
        "complete": complete,
        "first_red": {
            key: first_naive_overflow.get(key)
            for key in (
                "_line", "stage", "group", "max_abs", "stable_norm"
            )
        },
    }
  if complete:
    return {
        **common,
        "verdict": "ALL_BOUNDARIES_FINITE_NO_COMMIT",
        "complete": True,
        "max_abs_by_record": [
            {
                "stage": record["stage"],
                "group": record["group"],
                "max_abs": record.get("max_abs"),
                "stable_norm": record.get("stable_norm"),
            }
            for record in tree_records
        ],
    }
  return {
      **common,
      "verdict": "INCONCLUSIVE_INCOMPLETE",
      "complete": False,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("raw_log", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = classify(args.raw_log.read_text(encoding="utf-8", errors="replace"))
  rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.output:
    if args.output.exists():
      raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
  print(rendered, end="")
  return 1 if result["verdict"] in {
      "FATAL_CONTRACT", "INCONCLUSIVE_INCOMPLETE"
  } else 0


if __name__ == "__main__":
  raise SystemExit(main())
