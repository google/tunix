#!/usr/bin/env python3
"""Classify the P57 stock stochastic rollout-calibration receipt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from examples.frozenlake import p57_workloads


_MARKER = "[CANON_P57_CALIBRATION_JSON] "
_MODE = "stochastic"
_GENERATIONS = 8
_TEMPERATURE = 0.7
_RECIPE_ORDER = ("m10", "m15", "m20")
_SAFE_STATUSES = {"SUCCEEDED", "MAX_STEPS_REACHED"}
_SELECTION_TIE_ORDER = ("m15", "m10", "m20")
_PHYSICAL_MAX_PROMPT_LENGTH = 16_384
_PHYSICAL_MAX_RESPONSE_LENGTH = 16_384
_ABSENT_SWITCHES = (
    "CANON_FIXED_AR", "CANON_FIXED_AR_EMBED", "CANON_RPA_D",
    "CANON_RPA_P", "CANON_RPA_M", "CANON_LOGPROB_M",
    "CANON_PALLAS_ALL_PROJ", "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU", "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD", "CANON_PALLAS_CANONICAL_VJP",
)
_ZERO_SWITCHES = (
    "CANON_RPA_VJP2", "CANON_VJP2_MAX_SEQS",
    "CANON_PROMPT_PROCESSED_LOGPROBS", "CANON_PALLAS_LOGSOFTMAX",
    "CANON_ENGINE_MODULE_C", "CANON_KV_UNIFIED",
    "CANON_P32_TRAIN_ADMITTED", "CANON_P32_DP_REDUCTION_ADMITTED",
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED", "CANON_P32_DP16_SEGMENTED",
    "CANON_FROZENLAKE_L3", "CANON_FROZENLAKE_P27",
    "CANON_P28_SEGMENTED_FORWARD", "CANON_P28_SEGMENTED_VJP",
    "CANON_P28_SEGMENTED_TRAIN", "CANON_P28_G6_UPDATE",
    "CANON_P28_BATCHED_REPORT", "CANON_P29_FULL_TRAIN",
    "CANON_ALIGNMENT_GATE", "CANON_ALIGNMENT_GATE_ONLY",
    "CANON_ALIGNMENT_UPDATE_CANARY", "CANON_ALIGNMENT_TRAIN",
    "CANON_PRE_ALIGN_GATE", "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
    "CANON_P38_FIXED_LM_HEAD",
)
_ZERO_TIM_OFF_ATTESTATION = {
    "regime": "stock-fast",
    "absent_switches": list(_ABSENT_SWITCHES),
    "zero_switches": list(_ZERO_SWITCHES),
    "canonical_excess_precision_pin": False,
}


def _load(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8", errors="replace")
  try:
    value = json.loads(text)
  except json.JSONDecodeError:
    matches = [
        line[len(_MARKER):]
        for line in text.splitlines()
        if line.startswith(_MARKER)
    ]
    if len(matches) != 1:
      raise ValueError(
          f"expected one P57 calibration marker, found {len(matches)}"
      )
    value = json.loads(matches[0])
  if not isinstance(value, dict):
    raise ValueError("P57 calibration receipt must be a JSON object")
  return value


def _percentile(values: list[int], fraction: float) -> int:
  if not values:
    raise ValueError("cannot compute a percentile of an empty sequence")
  ordered = sorted(values)
  index = max(0, math.ceil(fraction * len(ordered)) - 1)
  return ordered[index]


def _summarize(
    result: dict[str, Any],
    *,
    recipe_name: str,
    mode: str,
    generations: int,
) -> tuple[dict[str, Any], list[str]]:
  reasons = []
  spec = p57_workloads.recipe(recipe_name)
  expected_recipe = {
      "name": spec.name,
      "min_grid_side": spec.min_grid_side,
      "max_grid_side": spec.max_grid_side,
      "max_turns": spec.max_turns,
      "context_hard_cap": spec.context_hard_cap,
      "frozen_probability": spec.frozen_probability,
      "eligible": spec.eligible,
  }
  if result.get("recipe") != expected_recipe:
    reasons.append(f"{mode}/{recipe_name}: recipe contract drifted")
  records = result.get("records")
  if not isinstance(records, list):
    return {}, [f"{mode}/{recipe_name}: records are absent"]
  expected_records = 100 * generations
  if (
      result.get("prompts") != 100
      or result.get("generations") != generations
      or result.get("trajectories") != expected_records
      or len(records) != expected_records
  ):
    reasons.append(
        f"{mode}/{recipe_name}: trajectory coverage drifted "
        f"prompts={result.get('prompts')} generations={result.get('generations')} "
        f"trajectories={result.get('trajectories')} records={len(records)}"
    )
  if (
      result.get("train_steps_before") != result.get("train_steps_after")
  ):
    reasons.append(f"{mode}/{recipe_name}: training steps changed")

  groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
  statuses = Counter()
  rewards = []
  turns = []
  prompt_lengths = []
  context_lengths = []
  completion_lengths = []
  assistant_token_total = 0
  invalid_action_total = 0
  ineffective_action_total = 0
  seen_pairs = set()
  index_counts = Counter()
  for record in records:
    if not isinstance(record, dict):
      reasons.append(f"{mode}/{recipe_name}: non-object trajectory record")
      continue
    try:
      group_id = int(record["group_id"])
      pair_index = int(record["pair_index"])
      reward = float(record["reward"])
      status = str(record["status"])
      p57_index = int(record["p57_index"])
      grid_side = int(record["grid_side"])
      turn_count = int(record["turns"])
      prompt_tokens = int(record["prompt_tokens"])
      context_tokens = int(record["context_tokens"])
      completion_tokens = int(record["completion_tokens"])
      assistant_tokens = int(record["assistant_tokens"])
      invalid_actions = int(record["invalid_actions"])
      ineffective_actions = int(record["ineffective_actions"])
      map_sha = str(record["map_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
      reasons.append(f"{mode}/{recipe_name}: malformed trajectory: {exc}")
      continue
    if not math.isfinite(reward):
      reasons.append(f"{mode}/{recipe_name}: nonfinite reward")
    if pair_index not in range(generations):
      reasons.append(f"{mode}/{recipe_name}: invalid pair index {pair_index}")
    if (group_id, pair_index) in seen_pairs:
      reasons.append(f"{mode}/{recipe_name}: duplicate group/pair")
    seen_pairs.add((group_id, pair_index))
    if p57_index not in range(100):
      reasons.append(f"{mode}/{recipe_name}: invalid P57 index {p57_index}")
    if grid_side not in spec.grid_sides():
      reasons.append(f"{mode}/{recipe_name}: grid side {grid_side} drifted")
    if len(map_sha) != 64:
      reasons.append(f"{mode}/{recipe_name}: invalid map SHA")
    if turn_count < 0 or turn_count > spec.max_turns:
      reasons.append(f"{mode}/{recipe_name}: turn count {turn_count} drifted")
    if (
        prompt_tokens < 0
        or context_tokens < 0
        or completion_tokens < 0
        or assistant_tokens < 0
        or invalid_actions < 0
        or ineffective_actions < 0
    ):
      reasons.append(f"{mode}/{recipe_name}: negative token length")
    groups[group_id].append(record)
    index_counts[p57_index] += 1
    statuses[status] += 1
    rewards.append(reward)
    turns.append(turn_count)
    prompt_lengths.append(prompt_tokens)
    context_lengths.append(context_tokens)
    completion_lengths.append(completion_tokens)
    assistant_token_total += assistant_tokens
    invalid_action_total += invalid_actions
    ineffective_action_total += ineffective_actions

  if len(groups) != 100 or any(len(group) != generations for group in groups.values()):
    reasons.append(
        f"{mode}/{recipe_name}: group coverage drifted "
        f"groups={len(groups)} sizes={sorted({len(v) for v in groups.values()})}"
    )
  if set(index_counts) != set(range(100)) or any(
      count != generations for count in index_counts.values()
  ):
    reasons.append(f"{mode}/{recipe_name}: map-index coverage drifted")
  invalid_status_count = sum(
      count for status, count in statuses.items() if status not in _SAFE_STATUSES
  )
  solved = sum(value > 0.1 for value in rewards)
  all_solved = all_failed = mixed = 0
  for group in groups.values():
    group_solved = [float(record["reward"]) > 0.1 for record in group]
    if all(group_solved):
      all_solved += 1
    elif not any(group_solved):
      all_failed += 1
    else:
      mixed += 1
  summary = {
      "solve_rate": solved / len(rewards) if rewards else None,
      "all_solved_group_ratio": all_solved / len(groups) if groups else None,
      "all_failed_group_ratio": all_failed / len(groups) if groups else None,
      "mixed_group_ratio": mixed / len(groups) if groups else None,
      "nonzero_advantage_ratio": (
          mixed * generations / len(rewards) if rewards else None
      ),
      "status_counts": dict(sorted(statuses.items())),
      "invalid_status_count": invalid_status_count,
      "invalid_action_count": invalid_action_total,
      "ineffective_action_count": ineffective_action_total,
      "turns_p50": _percentile(turns, 0.50) if turns else None,
      "turns_p90": _percentile(turns, 0.90) if turns else None,
      "turns_p95": _percentile(turns, 0.95) if turns else None,
      "turns_p99": _percentile(turns, 0.99) if turns else None,
      "turns_max": max(turns) if turns else None,
      "prompt_tokens_p99": _percentile(prompt_lengths, 0.99) if prompt_lengths else None,
      "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
      "context_tokens_p50": _percentile(context_lengths, 0.50) if context_lengths else None,
      "context_tokens_p90": _percentile(context_lengths, 0.90) if context_lengths else None,
      "context_tokens_p95": _percentile(context_lengths, 0.95) if context_lengths else None,
      "context_tokens_p99": _percentile(context_lengths, 0.99) if context_lengths else None,
      "context_tokens_max": max(context_lengths) if context_lengths else None,
      "completion_tokens_p95": _percentile(completion_lengths, 0.95) if completion_lengths else None,
      "completion_tokens_p99": _percentile(completion_lengths, 0.99) if completion_lengths else None,
      "completion_tokens_max": max(completion_lengths) if completion_lengths else None,
      "physical_prompt_cap_hits": sum(
          value >= _PHYSICAL_MAX_PROMPT_LENGTH for value in prompt_lengths
      ),
      "physical_response_cap_hits": sum(
          value >= _PHYSICAL_MAX_RESPONSE_LENGTH for value in completion_lengths
      ),
      "context_hard_cap": spec.context_hard_cap,
      "context_cap_exceeded": sum(
          value > spec.context_hard_cap for value in context_lengths
      ),
      "wall_seconds": result.get("wall_seconds"),
      "sampled_assistant_tokens": assistant_token_total,
      "sampled_assistant_tokens_per_second": (
          assistant_token_total / float(result["wall_seconds"])
          if isinstance(result.get("wall_seconds"), (int, float))
          and float(result["wall_seconds"]) > 0
          else None
      ),
  }
  return summary, reasons


def classify(stochastic_path: Path) -> dict[str, Any]:
  reasons = []
  try:
    receipt = _load(stochastic_path)
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    receipt = {}
    reasons.append(f"stochastic receipt unreadable: {exc}")
  expected = {
      "schema": "p57-frozenlake-stock-rollout-calibration-v2",
      "arm": "mismatch",
      "inference_regime": "stock-fast",
      "zero_tim_off_attestation": _ZERO_TIM_OFF_ATTESTATION,
      "fixed_lm_head": "0",
      "mode": _MODE,
      "temperature": _TEMPERATURE,
      "generations": _GENERATIONS,
      "recipe_order": list(_RECIPE_ORDER),
      "physical_max_prompt_length": _PHYSICAL_MAX_PROMPT_LENGTH,
      "physical_max_response_length": _PHYSICAL_MAX_RESPONSE_LENGTH,
      "backward_calls": 0,
      "optimizer_commits": 0,
      "checkpoint_writes": 0,
  }
  wrong = {
      key: receipt.get(key)
      for key, value in expected.items()
      if receipt.get(key) != value
  }
  if wrong:
    reasons.append(f"stochastic receipt contract drifted: {wrong}")
  if (
      receipt.get("train_steps_before") != receipt.get("train_steps_after")
      or receipt.get("global_steps_before") != receipt.get("global_steps_after")
  ):
    reasons.append("stochastic receipt mutated training state")
  results = receipt.get("results")
  if not isinstance(results, dict) or tuple(results) != _RECIPE_ORDER:
    reasons.append("stochastic recipe result inventory drifted")
  source_commit = receipt.get("source_commit")
  if not isinstance(source_commit, str) or len(source_commit) != 40:
    reasons.append("calibration receipt lacks one full source commit")

  summaries: dict[str, dict[str, Any]] = {}
  if isinstance(results, dict):
    for recipe_name in _RECIPE_ORDER:
      result = results.get(recipe_name, {})
      summary, local_reasons = _summarize(
          result,
          recipe_name=recipe_name,
          mode=_MODE,
          generations=_GENERATIONS,
      )
      summaries[recipe_name] = summary
      reasons.extend(local_reasons)
      dataset_sha = result.get("dataset_eval_sha256")
      if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        reasons.append(f"{recipe_name}: dataset SHA is invalid")

  eligible = []
  eligibility = {}
  if not reasons:
    for recipe_name in _RECIPE_ORDER:
      spec = p57_workloads.recipe(recipe_name)
      stochastic = summaries[recipe_name]
      checks = {
          "stochastic_band_15_35": 0.15 <= stochastic["solve_rate"] <= 0.35,
          "mixed_groups_ge_25": stochastic["mixed_group_ratio"] >= 0.25,
          "nonzero_advantage_ge_25": (
              stochastic["nonzero_advantage_ratio"] >= 0.25
          ),
          "no_invalid_status": stochastic["invalid_status_count"] == 0,
          "no_context_cap_excess": stochastic["context_cap_exceeded"] == 0,
          "no_physical_cap_hit": (
              stochastic["physical_prompt_cap_hits"] == 0
              and stochastic["physical_response_cap_hits"] == 0
          ),
      }
      eligibility[recipe_name] = checks
      if all(checks.values()):
        eligible.append(recipe_name)

  selected = None
  selection = "INVALID"
  recommendation = None
  if not reasons:
    if eligible:
      selected = min(
          eligible,
          key=lambda name: (
              round(abs(summaries[name]["solve_rate"] - 0.20), 12),
              _SELECTION_TIE_ORDER.index(name),
          ),
      )
      selection = f"FREEZE_{selected.upper()}"
    else:
      recommendation = min(
          _SELECTION_TIE_ORDER,
          key=lambda name: (
              round(abs(summaries[name]["solve_rate"] - 0.20), 12),
              _SELECTION_TIE_ORDER.index(name),
          ),
      )
      selection = "REVIEW_NO_ELIGIBLE_RECIPE"

  return {
      "schema": "p57-frozenlake-stock-calibration-classification-v3",
      "verdict": "PASS" if not reasons else "FAIL",
      "selection": selection,
      "selected_recipe": selected,
      "review_recommendation": recommendation,
      "eligible_recipes": eligible,
      "eligibility": eligibility,
      "summaries": summaries,
      "reasons": reasons,
      "source_commit": source_commit,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stochastic", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = classify(args.stochastic)
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "[P57.CALIBRATION.CLASSIFIER] "
      f"verdict={result['verdict']} selection={result['selection']} "
      f"output={args.output}",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
