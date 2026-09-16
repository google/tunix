# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyzes Tunix distributed RL trajectory logs to verify GRPO loss sanity.

This script inspects recorded trajectory CSV files (one at a time) to evaluate
group-relative advantage signals, completion lengths, and reward variance,
diagnosing whether the observed policy loss and gradient behavior make sense.

By default, if pointed to a directory (or run without arguments), it selects the
most recently modified CSV file in that directory.

Usage examples:
  # 1. Analyze the latest written trajectory run in /tmp/p0_1_env:
  python3 analyze_trajectories.py

  # 2. Analyze a concrete trajectory file from a previous run:
  python3 analyze_trajectories.py /tmp/p0_1_env/trajectory_log_1789512708.csv

  # 3. Analyze the newest file in a custom directory:
  python3 analyze_trajectories.py /path/to/my_trajectories/

  # 4. Correlate trajectory statistics with logged losses in orchestrator.log:
  python3 analyze_trajectories.py /tmp/p0_1_env/trajectory_log_1789512708.csv \\
      --orchestrator_log=orchestrator.log

  # 5. Filter to a specific step and inspect sample completions:
  python3 analyze_trajectories.py --step=0 --samples=3
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_COLUMNS = ("global_step", "prompt_id", "reward")


def parse_int_field(row: Dict[str, Any], field: str) -> int:
  """Parses a required integer field from a row, raising on failure."""
  if field not in row or row[field] is None or row[field] == "":
    raise KeyError(f"Trajectory row is missing required '{field}' field: {row}")
  val_str = str(row[field]).strip()
  if val_str in ("None", "none", "null", "NULL"):
    raise ValueError(f"Required field '{field}' cannot be None in row: {row}")
  try:
    return int(val_str)
  except (ValueError, TypeError) as e:
    raise ValueError(
        f"Invalid integer value for '{field}' ('{row.get(field)}') in row:"
        f" {row}"
    ) from e


def parse_float_field(row: Dict[str, Any], field: str) -> float:
  """Parses a required float field from a row, raising on failure."""
  if field not in row or row[field] is None or row[field] == "":
    raise KeyError(f"Trajectory row is missing required '{field}' field: {row}")
  val_str = str(row[field]).strip()
  if val_str in ("None", "none", "null", "NULL"):
    raise ValueError(f"Required field '{field}' cannot be None in row: {row}")
  try:
    return float(val_str)
  except (ValueError, TypeError) as e:
    raise ValueError(
        f"Invalid float value for '{field}' ('{row.get(field)}') in row:"
        f" {row}"
    ) from e


def parse_optional_token_count(val: Any) -> Optional[int]:
  """Parses optional token count from an integer, stringified list, or None."""
  if val is None:
    return None
  if isinstance(val, (int, float)):
    return int(val)
  val_str = str(val).strip()
  if val_str in ("", "None", "none", "null", "NULL"):
    return None
  if val_str.startswith("[") and val_str.endswith("]"):
    try:
      parsed = json.loads(val_str)
      if isinstance(parsed, list):
        return len(parsed)
    except (json.JSONDecodeError, ValueError):
      items = [x.strip() for x in val_str[1:-1].split(",") if x.strip()]
      return len(items)
  try:
    return int(val_str)
  except ValueError as e:
    raise ValueError(
        f"Invalid value for 'completion_tokens' ('{val_str}'): {e}"
    ) from e


def extract_clean_response_text(val: Any) -> str:
  """Extracts human-readable response text, unwrapping chat structures if present."""
  if not val:
    return ""
  val_str = str(val).strip()
  if val_str in ("None", "none", "null", "NULL"):
    return ""
  if val_str.startswith("[") and ("assistant" in val_str):
    try:
      parsed = json.loads(val_str)
    except Exception:
      try:
        parsed = ast.literal_eval(val_str)
      except Exception:
        parsed = None

    if isinstance(parsed, list):
      for msg in reversed(parsed):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
          content = msg.get("content", "")
          if content:
            return str(content).strip()
  return val_str


def resolve_csv_file(path_or_dir: str) -> Tuple[str, List[str]]:
  """Resolves a concrete CSV file.

  If path_or_dir is a directory, selects the CSV file with the most recent
  modification time.

  Args:
    path_or_dir: Path to a CSV file or directory containing CSV files.

  Returns:
    A tuple of (selected_csv_path, all_found_csv_paths_sorted_newest_first).

  Raises:
    FileNotFoundError: If path does not exist or no CSV files found in dir.
    ValueError: If a file path does not end with .csv or is not a regular file.
  """
  if not os.path.exists(path_or_dir):
    raise FileNotFoundError(f"Path does not exist: {path_or_dir}")

  if os.path.isfile(path_or_dir):
    if not path_or_dir.endswith(".csv"):
      raise ValueError(f"Expected a .csv file, got: {path_or_dir}")
    parent = os.path.dirname(path_or_dir) or "."
    siblings = [
        os.path.join(parent, f)
        for f in os.listdir(parent)
        if f.endswith(".csv") and os.path.isfile(os.path.join(parent, f))
    ]
    siblings.sort(key=os.path.getmtime, reverse=True)
    return path_or_dir, siblings

  if not os.path.isdir(path_or_dir):
    raise ValueError(
        f"Path is neither a regular file nor directory: {path_or_dir}"
    )

  csv_files = [
      os.path.join(path_or_dir, f)
      for f in os.listdir(path_or_dir)
      if f.endswith(".csv") and os.path.isfile(os.path.join(path_or_dir, f))
  ]

  if not csv_files:
    raise FileNotFoundError(
        f"No .csv trajectory log files found in directory: {path_or_dir}"
    )

  # Sort by modification time, newest first
  csv_files.sort(key=os.path.getmtime, reverse=True)
  return csv_files[0], csv_files


def load_trajectories(
    path_or_dir: str,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
  """Loads trajectory rows from a single concrete CSV file (latest if directory).

  Args:
    path_or_dir: Path to a CSV file or directory.

  Returns:
    A tuple of (rows, selected_csv_path, all_available_csv_paths).

  Raises:
    FileNotFoundError: If path or directory has no CSV files.
    ValueError: If CSV file is empty or has no rows.
    KeyError: If CSV is missing required columns.
  """
  selected_file, all_files = resolve_csv_file(path_or_dir)

  rows: List[Dict[str, Any]] = []
  with open(selected_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
      raise ValueError(f"CSV file is empty or has no header: {selected_file}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing_cols:
      raise KeyError(
          f"CSV file '{selected_file}' is missing required column(s):"
          f" {missing_cols}. Found columns: {reader.fieldnames}"
      )

    for r in reader:
      rows.append(r)

  if not rows:
    raise ValueError(
        f"CSV file contains no trajectory rows (only header): {selected_file}"
    )

  return rows, selected_file, all_files


def parse_orchestrator_log(log_path: str) -> Dict[int, Dict[str, float]]:
  """Parses step losses and metrics from orchestrator.log.

  Args:
    log_path: Path to orchestrator.log file.

  Returns:
    Dict mapping step index to parsed metric dictionaries.

  Raises:
    FileNotFoundError: If log_path does not exist.
  """
  if not os.path.exists(log_path):
    raise FileNotFoundError(f"Orchestrator log file not found: {log_path}")

  step_metrics: Dict[int, Dict[str, float]] = collections.defaultdict(dict)
  metric_pattern = re.compile(
      r"(loss|policy_loss|grad_norm|reward_mean|advantage_mean)[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
      re.IGNORECASE,
  )

  with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
      m_step = re.search(
          r"(?:step\s+|global_step\s*[:=]\s*)(\d+)", line, re.IGNORECASE
      )
      if m_step:
        step_idx = int(m_step.group(1))
        for m in metric_pattern.finditer(line):
          k, v = m.group(1).lower(), float(m.group(2))
          step_metrics[step_idx][k] = v

  return dict(step_metrics)


def compute_group_advantages(
    rewards: List[float], epsilon: float = 1e-6
) -> Tuple[List[float], float, float]:
  """Computes GRPO advantages within a single prompt group.

  A_i = (R_i - mean(R)) / (std(R) + eps)

  Args:
    rewards: List of rewards for rollouts of the same prompt.
    epsilon: Small constant for numerical stability.

  Returns:
    A tuple of (advantages, mean_reward, std_reward).

  Raises:
    ValueError: If rewards list is empty.
  """
  n = len(rewards)
  if n == 0:
    raise ValueError("Cannot compute advantages for an empty rewards list.")
  mean_r = sum(rewards) / n
  if n > 1:
    var_r = sum((r - mean_r) ** 2 for r in rewards) / (n - 1)
    std_r = math.sqrt(var_r)
  else:
    std_r = 0.0

  if std_r < 1e-8:
    # All rewards in the group are identical (all 0, all 1, etc.)
    # Advantage is 0 everywhere -> no gradient update signal.
    return [0.0] * n, mean_r, std_r

  advs = [(r - mean_r) / (std_r + epsilon) for r in rewards]
  return advs, mean_r, std_r


def analyze_trajectories(
    rows: List[Dict[str, Any]],
    orchestrator_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    max_response_length: int = 1024,
) -> Dict[str, Any]:
  """Performs statistical analysis of trajectories by step and prompt group.

  Args:
    rows: Trajectory rows loaded from CSV.
    orchestrator_metrics: Optional metrics parsed from orchestrator.log.
    max_response_length: Maximum allowed response token length before
      truncation.

  Returns:
    Dictionary containing step-level and aggregate statistics.

  Raises:
    ValueError: If rows is empty.
    KeyError: If any row is missing required fields (global_step, prompt_id,
      reward).
  """
  if not rows:
    raise ValueError("No trajectory rows provided for analysis.")

  if orchestrator_metrics is None:
    orchestrator_metrics = {}

  steps_data: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
  for r in rows:
    step = parse_int_field(r, "global_step")
    steps_data[step].append(r)

  step_summaries = []
  total_trajectories = len(rows)
  all_rewards = []
  total_active_groups = 0
  total_zero_groups = 0

  for step in sorted(steps_data.keys()):
    items = steps_data[step]
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for it in items:
      pid = it.get("prompt_id")
      if not pid or str(pid).strip() in ("None", "none", "null", "NULL"):
        raise KeyError(
            f"Trajectory row is missing required 'prompt_id' for GRPO"
            f" grouping: {it}"
        )
      groups[pid].append(it)

    step_rewards = []
    step_advantages = []
    active_groups = 0
    zero_variance_groups = 0
    all_zero_groups = 0
    all_one_groups = 0
    completion_lengths = []
    truncated_count = 0

    group_details = []
    for pid, g_items in groups.items():
      g_rewards = [parse_float_field(x, "reward") for x in g_items]
      g_advs, g_mean, g_std = compute_group_advantages(g_rewards)
      step_rewards.extend(g_rewards)
      step_advantages.extend(g_advs)
      all_rewards.extend(g_rewards)

      if g_std > 1e-6:
        active_groups += 1
        total_active_groups += 1
      else:
        zero_variance_groups += 1
        total_zero_groups += 1
        if g_mean < 1e-6:
          all_zero_groups += 1
        elif abs(g_mean - 1.0) < 1e-6:
          all_one_groups += 1

      for it, adv in zip(g_items, g_advs):
        it["computed_advantage"] = adv
        comp_tokens = parse_optional_token_count(it.get("completion_tokens"))
        if comp_tokens is not None:
          completion_lengths.append(comp_tokens)
          if comp_tokens >= max_response_length - 5:
            truncated_count += 1

      raw_q = g_items[0].get("question", "").strip().replace("\n", " ")
      raw_resp = extract_clean_response_text(
          g_items[0].get("completion", "")
      ).strip().replace("\n", " ")
      prompt_snippet = (raw_q[:80] + "...") if len(raw_q) > 80 else raw_q
      response_snippet = (
          (raw_resp[:120] + "...") if len(raw_resp) > 120 else raw_resp
      )

      group_details.append({
          "prompt_id": pid,
          "size": len(g_items),
          "mean_reward": g_mean,
          "std_reward": g_std,
          "rewards": g_rewards,
          "advantages": g_advs,
          "question": prompt_snippet,
          "response": response_snippet,
          "sample_completion": response_snippet,
      })

    num_groups = len(groups)
    mean_reward = (
        sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    )
    active_ratio = (active_groups / num_groups) if num_groups > 0 else 0.0

    if active_groups == 0:
      loss_verdict = "LOSS SHOULD BE ~0.0 (NO UPDATE SIGNAL)"
      loss_explanation = (
          f"All {num_groups} prompt groups had zero reward variance"
          f" ({all_zero_groups} all-wrong, {all_one_groups} all-correct)."
          " Within-group std is 0, making all advantages exactly 0.0. With"
          " BETA=0 (no KL penalty), policy gradient surrogate loss is 0.0."
      )
    else:
      pos_adv = sum(1 for a in step_advantages if a > 0.01)
      neg_adv = sum(1 for a in step_advantages if a < -0.01)
      loss_verdict = "ACTIVE TRAINING SIGNAL (NON-ZERO LOSS)"
      loss_explanation = (
          f"{active_groups}/{num_groups} ({active_ratio*100:.1f}%) groups had"
          " mixed outcomes with reward variance. Produced"
          f" {pos_adv} positive advantage rollouts (encouraged) and"
          f" {neg_adv} negative rollouts (penalized). Policy loss should be"
          " non-zero (typically negative or fluctuating near 0)."
      )

    obs_metrics = orchestrator_metrics.get(step, {})

    step_summaries.append({
        "step": step,
        "num_rollouts": len(items),
        "num_groups": num_groups,
        "mean_group_size": len(items) / num_groups if num_groups else 0,
        "mean_reward": mean_reward,
        "active_groups": active_groups,
        "zero_variance_groups": zero_variance_groups,
        "all_zero_groups": all_zero_groups,
        "all_one_groups": all_one_groups,
        "active_ratio": active_ratio,
        "avg_completion_tokens": (
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else None
        ),
        "truncated_count": (
            truncated_count if completion_lengths else None
        ),
        "loss_verdict": loss_verdict,
        "loss_explanation": loss_explanation,
        "observed_metrics": obs_metrics,
        "groups": group_details,
    })

  overall_mean_reward = (
      sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
  )
  return {
      "total_steps": len(steps_data),
      "total_trajectories": total_trajectories,
      "overall_mean_reward": overall_mean_reward,
      "total_active_groups": total_active_groups,
      "total_zero_groups": total_zero_groups,
      "steps": step_summaries,
  }


def print_report(
    analysis: Dict[str, Any],
    selected_file: str,
    all_files: Optional[List[str]] = None,
    show_samples: int = 2,
    max_step: Optional[int] = None,
) -> None:
  """Prints a comprehensive terminal analysis report."""
  mtime = os.path.getmtime(selected_file)
  mtime_str = datetime.datetime.fromtimestamp(mtime).strftime(
      "%Y-%m-%d %H:%M:%S"
  )

  print("\n" + "=" * 80)
  print(" TUNIX DISTRIBUTED RL TRAJECTORY & LOSS SANITY REPORT")
  print("=" * 80)
  print(f" Analyzed CSV File:         {selected_file}")
  print(f" File Last Modified:        {mtime_str}")
  if all_files and len(all_files) > 1:
    print(
        f" Total CSV Runs in Dir:     {len(all_files)} (Selected latest written"
        " by default)"
    )
  print(f" Total Trajectories Logged: {analysis['total_trajectories']}")
  print(f" Total Steps Recorded:      {analysis['total_steps']}")
  print(f" Overall Mean Reward:       {analysis['overall_mean_reward']:.4f}")
  total_g = analysis["total_active_groups"] + analysis["total_zero_groups"]
  active_pct = (
      (analysis["total_active_groups"] / total_g * 100) if total_g > 0 else 0
  )
  print(
      " Active Gradient Groups:   "
      f" {analysis['total_active_groups']} / {total_g} ({active_pct:.1f}%)"
  )
  print("=" * 80)

  if all_files and len(all_files) > 1:
    print("\n Other CSV runs detected in directory:")
    for f in all_files[:6]:
      if f == selected_file:
        continue
      t_str = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime(
          "%Y-%m-%d %H:%M:%S"
      )
      print(f"   * {os.path.basename(f)} ({t_str})")
    if len(all_files) > 6:
      print(f"   ... and {len(all_files) - 6} more")
    print(
        "   (To analyze an earlier run, pass its path: python3"
        " analyze_trajectories.py <path>)\n"
    )

  for s in analysis["steps"]:
    if max_step is not None and s["step"] > max_step:
      continue

    print(f"--- [GLOBAL STEP {s['step']}] " + "-" * 55)
    print(
        f"  Rollouts: {s['num_rollouts']} | Prompt Groups: {s['num_groups']} |"
        f" Group Size: {s['mean_group_size']:.1f}"
    )
    print(
        f"  Mean Reward (Accuracy): {s['mean_reward']:.4f} "
        f"(All-0: {s['all_zero_groups']}, All-1: {s['all_one_groups']}, Mixed:"
        f" {s['active_groups']})"
    )
    if s["avg_completion_tokens"] is not None:
      print(
          f"  Avg Completion Tokens:  {s['avg_completion_tokens']:.1f} |"
          f" Truncated (hit max length): {s['truncated_count']}"
      )
    else:
      print("  Avg Completion Tokens:  N/A (not logged)")

    if s["observed_metrics"]:
      obs_str = ", ".join(
          f"{k}={v:.5f}" for k, v in s["observed_metrics"].items()
      )
      print(f"  Observed Log Metrics:   {obs_str}")

    print(f"\n  >>> LOSS SANITY VERDICT: {s['loss_verdict']}")
    print(f"      {s['loss_explanation']}\n")

    active_samples = [g for g in s["groups"] if g["std_reward"] > 1e-6]
    zero_samples = [g for g in s["groups"] if g["std_reward"] <= 1e-6]

    if active_samples and show_samples > 0:
      print("  [Sample Active Groups - Where Gradient Updates Occur]:")
      for g in active_samples[:show_samples]:
        print(f"    * Prompt:   \"{g['question']}\"")
        if g.get("response"):
          print(f"      Response: \"{g['response']}\"")
        print(
            f"      Rewards:  {g['rewards']} -> Advantages:"
            f" {[round(a, 2) for a in g['advantages']]}"
        )
      print()

    if zero_samples and show_samples > 0:
      print("  [Sample Zero-Variance Groups - Zero Advantage / Zero Gradient]:")
      for g in zero_samples[:show_samples]:
        print(f"    * Prompt:   \"{g['question']}\"")
        if g.get("response"):
          print(f"      Response: \"{g['response']}\"")
        print(f"      Rewards:  {g['rewards']} (std=0 -> all advantages=0.0)")
      print()

  print("=" * 80)
  print(" LOSS INTERPRETATION CHEAT-SHEET (GRPO with BETA=0):")
  print(" 1. Loss == 0.0:")
  print(
      "    -> Normal when all generations in every group get the same reward"
      " (e.g. all wrong)."
  )
  print(
      "    -> Standard deviation within group is 0, so advantages are 0. With"
      " beta=0, loss is 0.0."
  )
  print(" 2. Loss is Negative (e.g. -0.01 to -0.3):")
  print(
      "    -> Expected in policy gradient minimization (-E[advantage *"
      " log_ratio])."
  )
  print(
      "    -> When correct rollouts have positive advantages, the loss"
      " naturally becomes negative."
  )
  print(" 3. Loss is Positive:")
  print(
      "    -> Occurs when negative advantage rollouts dominate token count, or"
      " during policy ratio shifts."
  )
  print(" 4. Truncation Check:")
  print(
      "    -> If truncated count is high, answers get cut off before final"
      " boxed answer -> 0 reward."
  )
  print("=" * 80 + "\n")


def main() -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Analyze Tunix distributed trajectory logs to verify GRPO loss."
          " Analyzes one concrete CSV file at a time (by default the most"
          " recently written one in /tmp/p0_1_env)."
      )
  )
  parser.add_argument(
      "csv_path",
      nargs="?",
      default="/tmp/p0_1_env",
      help=(
          "Path to a concrete trajectory CSV file or directory (default:"
          " /tmp/p0_1_env). If a directory is provided, the most recently"
          " written CSV is selected automatically."
      ),
  )
  parser.add_argument(
      "--orchestrator_log",
      type=str,
      default=None,
      help="Path to orchestrator.log if available for metric comparison",
  )
  parser.add_argument(
      "--max_response_length",
      type=int,
      default=1024,
      help="Expected max response token length for detecting truncations",
  )
  parser.add_argument(
      "--step",
      type=int,
      default=None,
      help="Filter analysis to a specific global step",
  )
  parser.add_argument(
      "--samples",
      type=int,
      default=2,
      help="Number of sample prompt groups to print per step",
  )
  parser.add_argument(
      "--json_out",
      type=str,
      default=None,
      help="Optional path to output JSON analysis summary",
  )

  args = parser.parse_args()

  try:
    rows, selected_file, all_files = load_trajectories(args.csv_path)

    orch_metrics = {}
    if args.orchestrator_log:
      orch_metrics = parse_orchestrator_log(args.orchestrator_log)

    analysis = analyze_trajectories(
        rows,
        orchestrator_metrics=orch_metrics,
        max_response_length=args.max_response_length,
    )
  except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"Error during analysis: {e}", file=sys.stderr)
    return 1

  print_report(
      analysis,
      selected_file=selected_file,
      all_files=all_files,
      show_samples=args.samples,
      max_step=args.step,
  )

  if args.json_out:
    analysis["analyzed_file"] = selected_file
    with open(args.json_out, "w", encoding="utf-8") as f:
      json.dump(analysis, f, indent=2)
    print(f"JSON analysis saved to: {args.json_out}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
