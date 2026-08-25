#!/usr/bin/env python3
"""Classify one P64 P45 DP8xTP8 first-red raw log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_PREFIX = "[P64.NUMERIC] "
_PRE_PREFIX = "[" "CANON" "_ALIGN_PRE_JSON] "
_PROFILE_BASE = (
    "profile_resolved workload=frozenlake-dp8-tp8 dp=8 tp=8 "
    "stage=backward-no-commit optimizer_commits=0"
)
_ADMISSION = (
    "admission workload=frozenlake-dp8-tp8 dp=8 tp=8 "
    "global_trajectories=256 local_trajectories=32 "
    "global_M=2048 local_M=256 optimizer_commits=0"
)
_COMMIT_RE = re.compile(r"optimizer_commits=([1-9][0-9]*)")
_STAGE_ORDER = {
    "loss_cotangent": 0,
    "group_input_cotangent": 1,
    "engine_vjp": 2,
    "trainer_rank_local": 3,
    "fixed_dp_reduced": 4,
    "scaled_microgradient": 5,
    "final_accumulator": 6,
}


def _tree_order(record: dict[str, Any]) -> tuple[int, int]:
  stage = str(record.get("stage"))
  group = int(record.get("group", -999))
  if stage == "loss_cotangent":
    return (-1, _STAGE_ORDER[stage])
  return (group, _STAGE_ORDER[stage])


def _json_lines(text: str, prefix: str) -> list[dict[str, Any]]:
  result = []
  for line_number, line in enumerate(text.splitlines(), 1):
    if not line.startswith(prefix):
      continue
    record = json.loads(line[len(prefix) :])
    if not isinstance(record, dict):
      raise ValueError(f"non-object JSON at line {line_number}")
    record["_line"] = line_number
    result.append(record)
  return result


def classify(text: str) -> dict[str, Any]:
  failures: list[str] = []
  profile_matches = re.findall(
      re.escape(_PREFIX + _PROFILE_BASE)
      + r" capsule_mode=(capture|replay)$",
      text,
      flags=re.MULTILINE,
  )
  capsule_mode = profile_matches[0] if len(profile_matches) == 1 else ""
  profile_count = len(profile_matches)
  admission_count = text.count(_PREFIX + _ADMISSION)
  if profile_count != 1:
    failures.append(f"profile={profile_count}/1")
  if admission_count != 1:
    failures.append(f"admission={admission_count}/1")
  capsule_markers = {
      "capture_ready": text.count("[P64.CAPSULE] capture_ready "),
      "diagnostic_replay_ready": text.count(
          "[P64.CAPSULE] diagnostic_replay_ready "
      ),
      "producer_bypass": text.count(
          "[P64.CAPSULE] producer_bypass verdict=PASS "
      ),
      "backward_scope": text.count(
          "[P64.CAPSULE] backward_scope mode=replay groups=1/32 "
      ),
      "model_bound": text.count("[P64.CAPSULE] model_bound mode=capture "),
      "model_verified": text.count(
          "[P64.CAPSULE] model_verified mode=replay "
      ),
      "transport_capture": text.count(
          "[P64.CAPSULE] transport_ready mode=capture "
      ),
      "transport_replay": text.count(
          "[P64.CAPSULE] transport_ready mode=replay "
      ),
  }
  expected_markers = (
      {
          "capture_ready": 1,
          "diagnostic_replay_ready": 0,
          "producer_bypass": 0,
          "backward_scope": 0,
          "model_bound": 1,
          "model_verified": 0,
          "transport_capture": 1,
          "transport_replay": 0,
      }
      if capsule_mode == "capture"
      else {
          "capture_ready": 0,
          "diagnostic_replay_ready": 1,
          "producer_bypass": 1,
          "backward_scope": 1,
          "model_bound": 0,
          "model_verified": 1,
          "transport_capture": 0,
          "transport_replay": 1,
      }
      if capsule_mode == "replay"
      else {}
  )
  wrong_markers = {
      name: f"{capsule_markers[name]}/{expected}"
      for name, expected in expected_markers.items()
      if capsule_markers[name] != expected
  }
  if wrong_markers:
    failures.append(f"capsule_markers={wrong_markers}")
  if _COMMIT_RE.findall(text):
    failures.append("optimizer_commit_violation")
  if "[" "CANON" "_UPDATE_JSON]" in text or "[P28.G6] UPDATE" in text:
    failures.append("optimizer_update_receipt_present")
  if any(
      "CANON_ALIGN" in line and "verdict=FAIL" in line
      for line in text.splitlines()
  ):
    failures.append("real_alignment_fail")
  try:
    pre = _json_lines(text, _PRE_PREFIX)
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
      if not line.startswith(_PREFIX):
        continue
      payload = line[len(_PREFIX) :]
      if payload.startswith(_PROFILE_BASE + " capsule_mode=") or payload == _ADMISSION or payload.startswith(
          "discard_complete "
      ):
        continue
      if not payload.startswith("{"):
        failures.append(f"unknown_text_marker_at_{line_number}")
        continue
      record = json.loads(payload)
      record["_line"] = line_number
      records.append(record)
  except (json.JSONDecodeError, ValueError) as error:
    failures.append(f"malformed_json={error}")
    pre = []
    records = []
  if len(pre) != 1:
    failures.append(f"pre_alignment={len(pre)}/1")
  elif (
      pre[0].get("verdict") != "PASS"
      or pre[0].get("context", {}).get("mesh") != "8,8"
      or pre[0].get("context", {}).get("run_stage")
      != "backward-no-commit"
      or int(pre[0].get("N_action", 0)) <= 0
  ):
    failures.append("pre_alignment_contract")

  loss_records = [
      record for record in records
      if record.get("schema") == "canon-p64-loss-scale-v1"
  ]
  if len(loss_records) != 1:
    failures.append(f"loss_scale={len(loss_records)}/1")
  else:
    loss = loss_records[0]
    exact = {
        "dp": 8,
        "tp": 8,
        "global_trajectories": 256,
        "local_trajectories": 32,
        "gradient_groups": 32,
        "global_M": 2048,
        "local_M": 256,
        "expected_accumulator_denominator": 32,
        "expected_streamed_multiplier": 0.125,
        "loss_denominator": 256.0,
        "loss_scale": 0.00390625,
    }
    changed = {
        name: loss.get(name)
        for name, expected in exact.items()
        if loss.get(name) != expected
    }
    if changed:
      failures.append(f"loss_scale_contract={changed}")
  tree_records = [
      record for record in records
      if record.get("schema") == "canon-p64-tree-numeric-v1"
  ]
  unknown_schemas = sorted({
      str(record.get("schema")) for record in records
      if record.get("schema") not in {
          "canon-p64-loss-scale-v1", "canon-p64-tree-numeric-v1"
      }
  })
  if unknown_schemas:
    failures.append(f"unknown_schemas={unknown_schemas}")
  loss_cotangents = [
      record for record in tree_records
      if record.get("stage") == "loss_cotangent"
  ]
  if len(loss_cotangents) != 1:
    failures.append(f"loss_cotangent={len(loss_cotangents)}/1")
  # The runtime fails closed at the first red boundary.  A later boundary is
  # therefore mandatory only when the preceding boundary was finite.
  elif loss_cotangents[0].get("all_finite") is True:
    group_inputs = [
        record for record in tree_records
        if record.get("stage") == "group_input_cotangent"
    ]
    if len(group_inputs) != 1:
      failures.append(f"group_input_cotangent={len(group_inputs)}/1")
  for record in tree_records:
    if record.get("stage") not in _STAGE_ORDER:
      failures.append(f"unknown_stage={record.get('stage')!r}")
    if not isinstance(record.get("all_finite"), bool):
      failures.append(f"missing_all_finite_at_{record.get('_line')}")
    if record.get("groups") != 32:
      failures.append(f"groups_at_{record.get('_line')}={record.get('groups')}")
    if record.get("stage") in {
        "group_input_cotangent", "engine_vjp", "trainer_rank_local"
    } and record.get("rank_count") != 8:
      failures.append(
          f"rank_count_at_{record.get('_line')}={record.get('rank_count')}"
      )
  identities = [
      (record.get("stage"), record.get("group")) for record in tree_records
  ]
  if len(identities) != len(set(identities)):
    failures.append("duplicate_stage_group")
  ordered = [
      _tree_order(record)
      for record in tree_records
      if record.get("stage") in _STAGE_ORDER
  ]
  if ordered != sorted(ordered):
    failures.append("stage_order_violation")
  if failures:
    return {
        "schema": "canon-p64-p45-classification-v1",
        "verdict": "FATAL_CONTRACT",
        "failures": failures,
    }
  first_nonfinite = next(
      (record for record in tree_records if not record["all_finite"]), None
  )
  if first_nonfinite is not None and first_nonfinite is not tree_records[-1]:
    return {
        "schema": "canon-p64-p45-classification-v1",
        "verdict": "FATAL_CONTRACT",
        "failures": ["receipt_after_first_nonfinite"],
    }
  common = {
      "schema": "canon-p64-p45-classification-v1",
      "optimizer_commits": 0,
      "pre_alignment_actions": pre[0]["N_action"],
      "tree_records": len(tree_records),
      "observed_stages": [record["stage"] for record in tree_records],
      "failures": [],
      "capsule_mode": capsule_mode,
      "evidence_kind": (
          "strict-capture-source"
          if capsule_mode == "capture"
          else "diagnostic-replay-not-certification"
      ),
  }
  if first_nonfinite is not None:
    return {
        **common,
        "verdict": "ROOT_LOCALIZED_NONFINITE",
        "first_red": {
            name: first_nonfinite.get(name)
            for name in (
                "_line", "stage", "group", "first_nonfinite",
                "first_nonfinite_rank", "max_abs", "stable_norm",
            )
        },
    }
  expected_microsteps = 1 if capsule_mode == "replay" else 32
  expected_denominator = 1.0 if capsule_mode == "replay" else 32.0
  discard = text.count(
      _PREFIX
      + "discard_complete optimizer_commits=0 "
      f"microsteps={expected_microsteps} denominator={expected_denominator} "
      f"diagnostic_replay={int(capsule_mode == 'replay')}"
  )
  completion_failures = []
  required_groups = (0,) if capsule_mode == "replay" else (0, 31)
  for group in required_groups:
    for stage in (
        "engine_vjp", "trainer_rank_local", "fixed_dp_reduced",
        "scaled_microgradient",
    ):
      count = sum(
          record.get("stage") == stage and record.get("group") == group
          for record in tree_records
      )
      if count != 1:
        completion_failures.append(f"{stage}.group{group}={count}/1")
  final_count = sum(
      record.get("stage") == "final_accumulator"
      and record.get("group") == required_groups[-1]
      for record in tree_records
  )
  if final_count != 1:
    completion_failures.append(
        f"final_accumulator.group{required_groups[-1]}={final_count}/1"
    )
  if discard == 1 and not completion_failures:
    return {**common, "verdict": "ALL_BOUNDARIES_FINITE_NO_COMMIT"}
  return {
      **common,
      "verdict": "INCONCLUSIVE_INCOMPLETE",
      "reason": "no non-finite boundary and no complete discard terminal",
      "completion_failures": completion_failures,
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
