# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Layered parity reports for DeepSWE reward-only evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


L2_SCHEMA = "canon.p46.deepswe-eval.reward-only-l2.v1"
L3_SCHEMA = "canon.p46.deepswe-eval.reward-only-l3.v1"


def _tokens(value: Sequence[int], *, arm: str) -> tuple[int, ...]:
  if not isinstance(value, (list, tuple)) or not value:
    raise ValueError(f"{arm} token stream must be a nonempty sequence")
  if any(not isinstance(item, int) or item < 0 for item in value):
    raise ValueError(f"{arm} token stream contains a malformed token id")
  return tuple(value)


def classify_l2_tokens(
    logprob_arm: Sequence[int], reward_only_arm: Sequence[int]
) -> dict[str, Any]:
  """Classifies token identity as expected but explicitly non-blocking."""
  with_logprobs = _tokens(logprob_arm, arm="logprob")
  reward_only = _tokens(reward_only_arm, arm="reward_only")
  common = 0
  for left, right in zip(with_logprobs, reward_only):
    if left != right:
      break
    common += 1
  identical = with_logprobs == reward_only
  return {
      "schema": L2_SCHEMA,
      "hard_gate_pass": True,
      "classification": (
          "IDENTICAL_OBSERVER" if identical else "LAW1_SUFFIX_DIVERGENCE"
      ),
      "identical": identical,
      "common_prefix_tokens": common,
      "first_divergence_index": None if identical else common,
      "logprob_arm_tokens": len(with_logprobs),
      "reward_only_arm_tokens": len(reward_only),
  }


def _exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
  discordant = left_only + right_only
  if discordant == 0:
    return 1.0
  tail = min(left_only, right_only)
  probability = sum(
      math.comb(discordant, k) for k in range(tail + 1)
  ) / (2**discordant)
  return min(1.0, 2.0 * probability)


def classify_l3_paired_solve_rate(
    pairs: Sequence[Mapping[str, Any]], *, alpha: float = 0.05
) -> dict[str, Any]:
  """Runs an exact paired-binomial gate over matching task/sample identities."""
  if not pairs:
    raise ValueError("L3 requires at least one paired trajectory")
  if not 0.0 < alpha < 1.0:
    raise ValueError("L3 alpha must be between zero and one")
  identities: set[str] = set()
  both_solved = both_failed = logprob_only = reward_only_only = 0
  for pair in pairs:
    identity = str(pair.get("identity", ""))
    if not identity or identity in identities:
      raise ValueError("L3 identities must be nonempty and unique")
    identities.add(identity)
    left = pair.get("logprob_solved")
    right = pair.get("reward_only_solved")
    if not isinstance(left, bool) or not isinstance(right, bool):
      raise ValueError("L3 solve values must be booleans")
    if left and right:
      both_solved += 1
    elif left:
      logprob_only += 1
    elif right:
      reward_only_only += 1
    else:
      both_failed += 1
  pvalue = _exact_mcnemar_pvalue(logprob_only, reward_only_only)
  n = len(pairs)
  left_solved = both_solved + logprob_only
  right_solved = both_solved + reward_only_only
  return {
      "schema": L3_SCHEMA,
      "alpha": alpha,
      "pairs": n,
      "both_solved": both_solved,
      "both_failed": both_failed,
      "logprob_only_solved": logprob_only,
      "reward_only_only_solved": reward_only_only,
      "logprob_solve_ratio": left_solved / n,
      "reward_only_solve_ratio": right_solved / n,
      "solve_ratio_delta": (right_solved - left_solved) / n,
      "exact_mcnemar_pvalue": pvalue,
      "verdict": "PASS" if pvalue >= alpha else "FAIL",
  }


def _load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
  records: list[dict[str, Any]] = []
  for path in paths:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
      if not line.strip():
        continue
      record = json.loads(line)
      if not isinstance(record, dict):
        raise ValueError(f"non-object record at {path}:{line_number}")
      records.append(record)
  return records


def _contains_numeric(value: Any) -> bool:
  if isinstance(value, bool):
    return False
  if isinstance(value, (int, float)):
    return True
  if isinstance(value, Mapping):
    return any(_contains_numeric(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return any(_contains_numeric(item) for item in value)
  return False


def _has_numeric_logprobs(value: Any) -> bool:
  if isinstance(value, Mapping):
    for key, item in value.items():
      if "logprob" in str(key).lower().replace("_", ""):
        if _contains_numeric(item):
          return True
      if _has_numeric_logprobs(item):
        return True
  elif isinstance(value, (list, tuple)):
    return any(_has_numeric_logprobs(item) for item in value)
  return False


def build_l3_report(
    observer_records: Sequence[Mapping[str, Any]],
    reward_only_records: Sequence[Mapping[str, Any]],
    *,
    observer_wall_secs: float,
    reward_only_wall_secs: float,
) -> dict[str, Any]:
  """Builds the exact 64-chip 1-task x N16 promotion report."""
  if observer_wall_secs <= 0 or reward_only_wall_secs <= 0:
    raise ValueError("L3 wall times must be positive")

  def index(
      records: Sequence[Mapping[str, Any]], *, mode: str
  ) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
      if record.get("trajectory_mode") != mode:
        raise ValueError(f"L3 {mode} provenance mismatch")
      if record.get("valid") is not True:
        raise ValueError("L3 requires 16 valid records in each arm")
      identity = (str(record.get("task_key", "")), record.get("sample_index"))
      if not identity[0] or not isinstance(identity[1], int):
        raise ValueError("L3 record identity is malformed")
      if identity in result:
        raise ValueError("L3 record identities must be unique")
      result[identity] = record
    return result

  observer = index(
      observer_records, mode="observer_with_sampled_logprobs"
  )
  reward_only = index(
      reward_only_records, mode="reward_only_no_logprobs"
  )
  if len(observer) != 16 or set(observer) != set(reward_only):
    raise ValueError("L3 requires the same exact 16 task/sample identities")
  if len({identity[0] for identity in observer}) != 1:
    raise ValueError("L3 canary must cover exactly one task at N16")
  sampled_by = {
      str(item.get("sampled_by", ""))
      for item in [*observer.values(), *reward_only.values()]
  }
  if len(sampled_by) != 1 or not next(iter(sampled_by)).startswith("stock@"):
    raise ValueError("L3 arms must use one exact stock sampler SHA")
  if not all(
      _has_numeric_logprobs(item.get("trajectory"))
      for item in observer.values()
  ):
    raise ValueError("L3 observer arm lacks sampled-token logprobs")
  if any(
      _has_numeric_logprobs(item.get("trajectory"))
      for item in reward_only.values()
  ):
    raise ValueError("L3 reward-only arm contains numeric logprobs")
  pairs = [
      {
          "identity": f"{task}:{sample_index}",
          "logprob_solved": observer[(task, sample_index)].get("solved") is True,
          "reward_only_solved": (
              reward_only[(task, sample_index)].get("solved") is True
          ),
      }
      for task, sample_index in sorted(observer)
  ]
  report = classify_l3_paired_solve_rate(pairs)
  report.update({
      "sampled_by": next(iter(sampled_by)),
      "observer_valid_trajectories_per_hour": 16 * 3600 / observer_wall_secs,
      "reward_only_valid_trajectories_per_hour": (
          16 * 3600 / reward_only_wall_secs
      ),
      "observer_wall_secs": observer_wall_secs,
      "reward_only_wall_secs": reward_only_wall_secs,
  })
  return report


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--observer-jsonl", type=Path, nargs="+", required=True)
  parser.add_argument(
      "--reward-only-jsonl", type=Path, nargs="+", required=True
  )
  parser.add_argument("--observer-wall-secs", type=float, required=True)
  parser.add_argument("--reward-only-wall-secs", type=float, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite L3 report: {args.output}")
  report = build_l3_report(
      _load_jsonl(args.observer_jsonl),
      _load_jsonl(args.reward_only_jsonl),
      observer_wall_secs=args.observer_wall_secs,
      reward_only_wall_secs=args.reward_only_wall_secs,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"P46_REWARD_ONLY_L3_{report['verdict']} pairs={report['pairs']} "
      f"report={args.output}",
      flush=True,
  )
  return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
