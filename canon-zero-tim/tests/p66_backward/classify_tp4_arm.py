#!/usr/bin/env python3
"""Fail-closed classifier for one full-depth P66 DP1xTP4 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = (
    "tp4-serial",
    "tp4-p59-old",
    "tp4-p59",
    "tp4-gather-off",
    "tp4-vma-oracle",
)

ORACLE_ENDPOINTS = {"head", "norm", "layer_27", "layer_14", "layer_0", "embed"}
ORACLE_CAPS = {
    "rel_l2": 4.0e-2,
    "one_minus_cos": 3.2e-4,
    "norm_ratio_error": 4.0e-2,
    "sign_mismatch_rate": 2.0e-2,
}


def _valid_row_summary(value: Any, arm: str) -> bool:
  return (
      isinstance(value, dict)
      and value.get("schema") == "canon-p66-row-cotangent-summary-v1"
      and value.get("arm") == arm
      and isinstance(value.get("chunks"), int)
      and value["chunks"] >= 1
      and value.get("layers") == list(range(28))
      and value.get("records") == value["chunks"] * 28
      and isinstance(value.get("padding_row_layer_nonzero"), int)
      and value["padding_row_layer_nonzero"] >= 0
      and isinstance(value.get("padding_row_layer_nonfinite"), int)
      and value["padding_row_layer_nonfinite"] >= 0
  )


def _raw_row_summary(raw: str) -> dict[str, Any]:
  lines = [
      line.split(" ", 1)[1]
      for line in raw.splitlines()
      if line.startswith("[P66.TP4.ROWS.SUMMARY] ")
  ]
  try:
    value = json.loads(lines[-1])
  except (IndexError, json.JSONDecodeError):
    return {}
  return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _valid_oracle_summary(value: Any) -> bool:
  if (
      not isinstance(value, dict)
      or value.get("schema")
      != "canon-p66-same-point-vjp-oracle-summary-v1"
      or value.get("arm") != "tp4-vma-oracle"
      or value.get("verdict") != "PASS"
      or value.get("negative_control_detected") is not True
      or set(value.get("expected_endpoints", ())) != ORACLE_ENDPOINTS
      or set(value.get("observed_endpoints", ())) != ORACLE_ENDPOINTS
      or len(value.get("observed_endpoints", ())) != len(ORACLE_ENDPOINTS)
  ):
    return False
  records = value.get("records")
  if not isinstance(records, list) or len(records) != len(ORACLE_ENDPOINTS):
    return False
  for record in records:
    metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
    if (
        record.get("schema") != "canon-p66-same-point-vjp-oracle-v1"
        or record.get("endpoint") not in ORACLE_ENDPOINTS
        or record.get("verdict") != "PASS"
        or record.get("finite") is not True
        or not isinstance(record.get("leaf_count"), int)
        or record["leaf_count"] <= 0
        or not isinstance(record.get("elements"), int)
        or record["elements"] <= 0
        or record.get("dead_candidate_leaves") != 0
        or record.get("caps") != ORACLE_CAPS
        or set(metrics) != set(ORACLE_CAPS)
        or any(
            not isinstance(metrics[name], (int, float))
            or not np.isfinite(metrics[name])
            or metrics[name] > limit
            for name, limit in ORACLE_CAPS.items()
        )
    ):
      return False
  return len({record["endpoint"] for record in records}) == len(records)


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  values = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if not all(isinstance(value, dict) for value in values):
    raise ValueError(f"{path}: expected JSON objects")
  return values


def classify(
    *,
    arm: str,
    run_log: Path,
    pre_alignment_report: Path,
    alignment_report: Path,
    update_report: Path,
    docker_exit: int,
) -> dict[str, Any]:
  raw = run_log.read_text(encoding="utf-8", errors="replace")
  pre = _load_jsonl(pre_alignment_report) if pre_alignment_report.exists() else []
  align = _load_jsonl(alignment_report) if alignment_report.exists() else []
  row_summary = _raw_row_summary(raw)
  if arm == "tp4-p59-old":
    numeric_lines = [
        line.split(" ", 1)[1]
        for line in raw.splitlines()
        if line.startswith("[P66.TP4.NUMERIC] ")
    ]
    profile_lines = [
        line.split(" ", 1)[1]
        for line in raw.splitlines()
        if line.startswith("[P66.TP4.PROFILE] ")
    ]
    reasons = []
    try:
      engine = json.loads(numeric_lines[-1])
      profile = json.loads(profile_lines[-1])
    except (IndexError, json.JSONDecodeError):
      engine = {}
      profile = {}
      reasons.append("missing_red_receipt")
    red = (
        engine.get("all_finite") is not True
        or (
            isinstance(engine.get("stable_norm"), (int, float))
            and engine["stable_norm"] > 1.0e6
        )
    )
    components = profile.get("components", {})
    if docker_exit == 0 or not red:
      reasons.append("unsafe_arm_did_not_fail_closed")
    if len(pre) != 1 or pre[0].get("verdict") != "PASS":
      reasons.append("pre_alignment")
    if any(row.get("verdict") == "FAIL" for row in align):
      reasons.append("alignment_failure")
    if "Global step " in raw and " completed in " in raw:
      reasons.append("optimizer_commit")
    if len(components) != 31:
      reasons.append("layerwise_profile")
    if not _valid_row_summary(row_summary, arm):
      reasons.append("row_cotangent_summary")
    return {
        "schema": "canon-p66-tp4-arm-classification-v1",
        "verdict": "EXPECTED_RED" if not reasons else "FAIL",
        "arm": arm,
        "zero_tim": {
            "expected_pass": 1,
            "observed_pass": sum(
                row.get("verdict") == "PASS" for row in pre + align
            ),
            "observed_fail": sum(
                row.get("verdict") == "FAIL" for row in pre + align
            ),
        },
        "gradient_norm": None,
        "engine_norm": engine.get("stable_norm"),
        "row_cotangent_summary": row_summary,
        "evidence_sha256": {
            "run_log": _sha256(run_log),
            **({"pre_alignment_report": _sha256(pre_alignment_report)}
               if pre_alignment_report.exists() else {}),
        },
        "reasons": reasons,
    }
  update = _load_json(update_report) if update_report.exists() else {}
  reasons = []
  if docker_exit != 0:
    reasons.append("docker_exit")
  marker = f"[P66.BACKWARD] arm={arm} verdict=PASS commits=0"
  if raw.count(marker) != 1:
    reasons.append("terminal_marker")
  if "verdict=FAIL" in raw:
    reasons.append("raw_alignment_failure")
  if len(pre) != 1 or pre[0].get("verdict") != "PASS":
    reasons.append("pre_alignment")
  if len(align) != 16 or any(row.get("verdict") != "PASS" for row in align):
    reasons.append("alignment")
  if (
      update.get("schema") != "canon-p66-backward-gate-v1"
      or update.get("arm") != arm
      or update.get("verdict") != "PASS"
      or update.get("commits") != 0
      or (update.get("dp_size"), update.get("tp_size")) != (1, 4)
      or update.get("global_trajectories") != 16
      or update.get("gradient_groups") != 16
      or update.get("alignment_verdicts") != ["PASS"] * 16
      or update.get("train_steps_before") != update.get("train_steps_after")
      or any(update.get("state_changed_paths", {}).values())
  ):
    reasons.append("update_contract")
  gradient = update.get("gradient", {})
  if (
      not isinstance(gradient, dict)
      or gradient.get("all_finite") is not True
      or gradient.get("any_nonzero") is not True
      or not isinstance(gradient.get("stable_norm"), (int, float))
      or not np.isfinite(gradient["stable_norm"])
      or not 0.0 < gradient["stable_norm"] <= 1.0e6
  ):
    reasons.append("mapped_gradient")
  engine = update.get("engine_vjp", {})
  if (
      not isinstance(engine, dict)
      or engine.get("all_finite") is not True
      or engine.get("any_nonzero") is not True
      or not isinstance(engine.get("stable_norm"), (int, float))
      or not np.isfinite(engine["stable_norm"])
      or not 0.0 < engine["stable_norm"] <= 1.0e6
  ):
    reasons.append("engine_vjp")
  profile = update.get("layerwise_profile", {})
  components = profile.get("components", {}) if isinstance(profile, dict) else {}
  expected_components = {
      "embed", "norm", "head", *(f"layer_{index}" for index in range(28))
  }
  if (
      profile.get("schema") != "canon-p66-full-depth-profile-v1"
      or profile.get("arm") != arm
      or set(components) != expected_components
      or any(
          not isinstance(value, (int, float))
          or not np.isfinite(value)
          or value < 0.0
          or value > 1.0e6
          for value in components.values()
      )
  ):
    reasons.append("layerwise_profile")
  update_row_summary = update.get("row_cotangent_summary", {})
  if (
      not _valid_row_summary(row_summary, arm)
      or update_row_summary != row_summary
  ):
    reasons.append("row_cotangent_summary")
  for field in ("model_before_sample", "gradient_sample"):
    sample = update.get(field, {})
    if (
        not isinstance(sample, dict)
        or not isinstance(sample.get("leaves"), dict)
        or not sample["leaves"]
        or sample.get("sampled_leaves") != len(sample["leaves"])
    ):
      reasons.append(field)
  if arm in ("tp4-p59", "tp4-gather-off", "tp4-vma-oracle"):
    if "[P66.VMA] outer_check_enabled" not in raw:
      reasons.append("vma_marker")
    if "tp_input_reduction=vma_autodiff_psum" not in raw:
      reasons.append("vma_tp_transpose_marker")
  oracle = update.get("vjp_oracle")
  if arm == "tp4-vma-oracle":
    if (
        not _valid_oracle_summary(oracle)
        or raw.count("[P66.ORACLE.NEGATIVE] detected=1") != 1
        or raw.count("[P66.ORACLE.ENDPOINT] ") != len(ORACLE_ENDPOINTS)
        or raw.count("[P66.ORACLE.SUMMARY] ") != 1
    ):
      reasons.append("vjp_oracle")
  elif oracle is not None:
    reasons.append("unexpected_vjp_oracle")
  if arm == "tp4-p59-old" and "tp_input_reduction=all_gather_rank_order_f32_barrier" not in raw:
    reasons.append("historical_tp_sum_marker")
  return {
      "schema": "canon-p66-tp4-arm-classification-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "arm": arm,
      "zero_tim": {
          "expected_pass": 17,
          "observed_pass": sum(
              row.get("verdict") == "PASS" for row in pre + align
          ),
          "observed_fail": sum(
              row.get("verdict") == "FAIL" for row in pre + align
          ),
      },
      "gradient_norm": gradient.get("stable_norm"),
      "engine_norm": engine.get("stable_norm"),
      "row_cotangent_summary": row_summary,
      "evidence_sha256": {
          "run_log": _sha256(run_log),
          **({"pre_alignment_report": _sha256(pre_alignment_report)}
             if pre_alignment_report.exists() else {}),
          **({"alignment_report": _sha256(alignment_report)}
             if alignment_report.exists() else {}),
          **({"update_report": _sha256(update_report)}
             if update_report.exists() else {}),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=ARMS, required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--docker-exit", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      arm=args.arm,
      run_log=args.run_log,
      pre_alignment_report=args.pre_alignment_report,
      alignment_report=args.alignment_report,
      update_report=args.update_report,
      docker_exit=args.docker_exit,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"P66_TP4_ARM verdict={result['verdict']} arm={args.arm} "
      f"zero_tim={result['zero_tim']['observed_pass']}/"
      f"{result['zero_tim']['expected_pass']} "
      f"fail={result['zero_tim']['observed_fail']}"
  )
  return 0 if result["verdict"] in ("PASS", "EXPECTED_RED") else 1


if __name__ == "__main__":
  raise SystemExit(main())
