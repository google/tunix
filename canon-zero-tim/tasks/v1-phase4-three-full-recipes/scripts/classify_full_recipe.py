#!/usr/bin/env python3
"""Fail-closed postflight for one V1 high-performance full-training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any


_RECIPES = {
    "gsm8k": {
        "workload": "gsm8k",
        "updates": 200,
        "dp": 16,
        "tp": 4,
        "apc": False,
        "candidate": "",
        "split": "",
    },
    "p45": {
        "workload": "frozenlake-dp8-tp8",
        "updates": 300,
        "dp": 8,
        "tp": 8,
        "apc": True,
        "candidate": "",
        "split": "",
    },
    "m15": {
        "workload": "frozenlake-dp8-tp8",
        "updates": 300,
        "dp": 8,
        "tp": 8,
        "apc": True,
        "candidate": "m15",
        "split": "main",
    },
}
_PROFILED_STEP = 2
_FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")
_STEP_RE = re.compile(r"Global step (\d+) completed in ([0-9.]+) seconds\.")
_ALIGN_RE = re.compile(
    r"^\[CANON_ALIGN(?:_PRE)?\].*\bverdict=(PASS|FAIL)\b"
)


def _require(condition: bool, reason: str, reasons: list[str]) -> None:
  if not condition:
    reasons.append(reason)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _artifact(path: Path, state: Path) -> dict[str, Any]:
  return {
      "path": str(path.relative_to(state)),
      "bytes": path.stat().st_size,
      "sha256": _sha256(path),
  }


def _json_lines(path: Path) -> list[dict[str, Any]]:
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    if not isinstance(row, dict):
      raise ValueError(f"expected JSON object at {path}:{number}")
    rows.append(row)
  return rows


def _resolved_env(path: Path) -> dict[str, str]:
  values = {}
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.startswith("export "):
      continue
    assignment = line.removeprefix("export ")
    if "=" not in assignment:
      raise ValueError(f"invalid export at {path}:{number}")
    name, encoded = assignment.split("=", 1)
    decoded = shlex.split(encoded)
    if len(decoded) > 1:
      raise ValueError(f"ambiguous export at {path}:{number}")
    values[name] = decoded[0] if decoded else ""
  return values


def _perf_rows(text: str) -> dict[str, list[dict[str, str]]]:
  result: dict[str, list[dict[str, str]]] = {}
  for line in text.splitlines():
    stripped = line.strip()
    if not stripped.startswith("[PERF] "):
      continue
    fields = dict(_FIELD_RE.findall(stripped.removeprefix("[PERF] ")))
    stage = fields.pop("stage", None)
    if stage:
      result.setdefault(stage, []).append(fields)
  return result


def _seconds(row: dict[str, str], label: str, reasons: list[str]) -> float:
  try:
    value = float(row["seconds"])
  except (KeyError, TypeError, ValueError):
    reasons.append(f"{label}.seconds")
    return float("nan")
  if not math.isfinite(value) or value < 0.0:
    reasons.append(f"{label}.seconds")
  return value


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
  if not rows:
    return {}
  keys = tuple(key for key in rows[0] if key != "global_step")
  return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _steady_timing_rows(
    rows: list[dict[str, float]],
    *,
    expected_updates: int,
    p57_eval: dict[str, Any] | None,
) -> tuple[list[dict[str, float]], set[int], list[dict[str, float]]]:
  steady = [
      row for row in rows
      if row["global_step"] >= 2 and row["global_step"] != _PROFILED_STEP
  ]
  direct_eval_enclosing_global_steps = set()
  if p57_eval is not None:
    # Use the producer's explicit receipt instead of inferring a wall-row
    # number from the pre-update policy step. The final post-training eval has
    # no enclosing Global-step timing row and is represented by null.
    direct_eval_enclosing_global_steps = {
        int(receipt["enclosing_global_step"])
        for receipt in p57_eval.get("cycle_receipts", ())
        if receipt.get("enclosing_global_step") is not None
    }
  direct_eval_cycle_excluded = [
      row for row in steady
      if int(row["global_step"]) not in direct_eval_enclosing_global_steps
  ]
  return (
      steady,
      direct_eval_enclosing_global_steps,
      direct_eval_cycle_excluded,
  )


def classify(
    *,
    recipe: str,
    state: Path,
    run_log: Path,
    update_report: Path,
    base_classification: Path,
) -> dict[str, Any]:
  contract = _RECIPES[recipe]
  expected_updates = int(contract["updates"])
  dp_size = int(contract["dp"])
  tp_size = int(contract["tp"])
  local_groups = 256 // dp_size
  expected_alignment_pass = expected_updates * (1 + local_groups)
  reasons: list[str] = []

  env_path = state / "env.sh"
  for path, label in (
      (state, "state"),
      (env_path, "env"),
      (run_log, "run_log"),
      (update_report, "update_report"),
      (base_classification, "base_classification"),
  ):
    _require(path.exists(), f"missing_{label}", reasons)
  if reasons:
    return {"schema": "v1-hp-full-classification-v1", "verdict": "FAIL", "recipe": recipe, "reasons": reasons}

  env = _resolved_env(env_path)
  required_env = {
      "CANON_V1_HP_FULL": "1",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_CONTINUE_DECODE": "8",
      "CANON_FIXED_AR_GATHER": "1",
      "CANON_PALLAS_GATHERED_LOGPROBS": "1",
      "CANON_LOGPROB_STEP_FUSION": "1",
      "CANON_P28_BATCHED_REPORT": "1",
      "CANON_P28_BATCHED_REVERSE": "0",
      "CANON_FUSED_TREE_OPS": "0",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": str(_PROFILED_STEP),
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
      "CANON_XPROF_LABELS": "1",
      "CANON_PERF_TRACE_EXPORT_STEP": str(_PROFILED_STEP),
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "1" if contract["apc"] else "0",
  }
  if recipe == "gsm8k":
    required_env.update({
        "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0",
        "CANON_BATCHED_EVIDENCE": "1",
    })
  else:
    required_env.update({
        "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
        "CANON_BATCHED_EVIDENCE": "0",
        "CANON_P57_TIM_ARM": "zero",
        "CANON_P57_EXPECTED_UPDATES": "300",
        "CANON_P57_WORKLOAD_CANDIDATE": str(contract["candidate"]),
        "CANON_P57_DATA_SPLIT": str(contract["split"]),
        "CANON_P33_ENABLE_EVAL": "1",
        "CANON_P33_DISABLE_EVAL": "0",
        "CANON_P31_ENABLE_EVAL": "1",
        "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL": "0",
    })
  wrong_env = {
      name: env.get(name)
      for name, value in required_env.items()
      if env.get(name) != value
  }
  _require(not wrong_env, f"resolved_env={wrong_env}", reasons)

  p57_eval_classification = state / "p57_inprocess_eval.classification.json"
  p57_eval = None
  if recipe != "gsm8k":
    _require(
        p57_eval_classification.is_file(),
        "missing_p57_inprocess_eval_classification",
        reasons,
    )
    if p57_eval_classification.is_file():
      p57_eval = json.loads(
          p57_eval_classification.read_text(encoding="utf-8")
      )
      _require(
          p57_eval.get("verdict") == "PASS",
          f"p57_eval_verdict={p57_eval.get('verdict')}",
          reasons,
      )
      _require(
          p57_eval.get("expected_updates") == expected_updates,
          "p57_eval_expected_updates",
          reasons,
      )
      _require(
          p57_eval.get("steps") == list(range(0, expected_updates + 1, 50)),
          "p57_eval_steps",
          reasons,
      )
      expected_cycle_receipts = [
          {
              "policy_step": step,
              "enclosing_global_step": (
                  None if step == expected_updates else step + 1
              ),
          }
          for step in range(0, expected_updates + 1, 50)
      ]
      _require(
          p57_eval.get("schema")
          == "p57-inprocess-evaluation-classification-v2",
          "p57_eval_schema",
          reasons,
      )
      _require(
          p57_eval.get("cycle_receipts") == expected_cycle_receipts,
          "p57_eval_cycle_receipts",
          reasons,
      )

  base = json.loads(base_classification.read_text(encoding="utf-8"))
  _require(base.get("verdict") == "PASS", f"base_verdict={base.get('verdict')}", reasons)
  _require(base.get("claim_level") == "strict-zero-tim", "base_claim_level", reasons)
  _require(base.get("expected_updates") == expected_updates, "base_expected_updates", reasons)
  _require(base.get("observed_updates") == expected_updates, "base_observed_updates", reasons)
  _require(base.get("observed_pre_alignments") == expected_updates, "base_pre_alignments", reasons)
  _require(
      base.get("observed_alignments") == expected_updates * local_groups,
      "base_alignments",
      reasons,
  )
  _require(base.get("alignment_warning_records") == 0, "base_alignment_warnings", reasons)
  _require(base.get("pre_alignment_warning_records") == 0, "base_pre_alignment_warnings", reasons)

  text = run_log.read_text(encoding="utf-8", errors="replace")
  align_verdicts = [
      match.group(1)
      for line in text.splitlines()
      if (match := _ALIGN_RE.match(line.strip()))
  ]
  _require(
      align_verdicts.count("PASS") == expected_alignment_pass,
      f"canon_align_pass={align_verdicts.count('PASS')} expected={expected_alignment_pass}",
      reasons,
  )
  _require(
      align_verdicts.count("FAIL") == 0,
      f"canon_align_fail={align_verdicts.count('FAIL')} expected=0",
      reasons,
  )

  updates = _json_lines(update_report)
  _require(len(updates) == expected_updates, f"updates={len(updates)} expected={expected_updates}", reasons)
  for index, update in enumerate(updates):
    expected = {
        "dp_rank_pullbacks_per_transaction": dp_size,
        "dp_pullback_invocations_per_transaction": 1,
        "dp_replicas_exact": True,
    }
    wrong = {
        name: update.get(name)
        for name, value in expected.items()
        if update.get(name) != value
    }
    if wrong:
      reasons.append(f"update[{index}].p59={wrong}")

  marker_counts = {
      "continue_decode": text.count("[P57.CONTINUE_DECODE] on-device decode loop enabled"),
      "gathered_logprobs": text.count("[P56.GATHERED_LOGPROBS] installed"),
      "logprob_step_fusion": text.count("[P56.LOGPROB_STEP_FUSION] active"),
      "fixed_ar_gather": text.count("CANON_FIXED_AR=1 gather-ordered-sum"),
      "xprof_labels": text.count("[CANON_XPROF_LABELS] continue-decode stage callables cached"),
      "p59_head_partition": text.count(
          f"[P59.DP{dp_size}] head_cotangent_partition_ready"
      ),
      "p59_parallel": text.count(f"[P59.DP{dp_size}] gradient_reducer_ready"),
      "xprof_armed": text.count(f"[P51.XPROF] phase=update armed step={_PROFILED_STEP}"),
      "xprof_started": text.count(f"[P51.XPROF] phase=update started step={_PROFILED_STEP}"),
      "xprof_stopped": text.count(f"[P51.XPROF] phase=update stopped step={_PROFILED_STEP + 1}"),
      "perfetto": text.count(f"[V1.PERFETTO] captured training_step={_PROFILED_STEP}"),
  }
  for name in (
      "continue_decode", "gathered_logprobs", "logprob_step_fusion",
      "fixed_ar_gather", "xprof_labels",
  ):
    _require(marker_counts[name] >= 1, f"marker.{name}={marker_counts[name]}", reasons)
  _require(
      marker_counts["p59_head_partition"] >= 1,
      f"marker.p59_head_partition={marker_counts['p59_head_partition']}",
      reasons,
  )
  _require(marker_counts["p59_parallel"] == expected_updates, f"marker.p59_parallel={marker_counts['p59_parallel']} expected={expected_updates}", reasons)
  for name in ("xprof_armed", "xprof_started", "xprof_stopped", "perfetto"):
    _require(marker_counts[name] == 1, f"marker.{name}={marker_counts[name]} expected=1", reasons)

  hit_rates = [
      float(value)
      for value in re.findall(r"Prefix cache hit rate:\s*([0-9.]+)%", text)
  ]
  apc_on = bool(contract["apc"])
  apc_marker = (
      f"[P3_APC_CONFIG] enabled={int(apc_on)} "
      "workload=frozenlake reader=train_frozenlake_qwen3"
  )
  if apc_on:
    _require(text.count(apc_marker) == 1, "apc_runtime_marker", reasons)
    _require(bool(hit_rates) and max(hit_rates) > 0.0, "apc_positive_cache_hit", reasons)
  else:
    _require("[P3_APC_CONFIG] enabled=1" not in text, "unexpected_apc_on", reasons)

  xplanes = sorted((state / "xprof-update").rglob("*.xplane.pb"))
  trace_json = sorted((state / "xprof-update").rglob("*.trace.json.gz"))
  perfetto = sorted((state / "perfetto").rglob("perfetto_trace_v2_*.pb"))
  _require(bool(xplanes), "missing_xplane", reasons)
  _require(bool(trace_json), "missing_trace_json_gz", reasons)
  _require(len(perfetto) == 1, f"perfetto_artifacts={len(perfetto)} expected=1", reasons)
  for label, paths in (("xplane", xplanes), ("trace_json_gz", trace_json), ("perfetto", perfetto)):
    _require(all(path.stat().st_size > 0 for path in paths), f"empty_{label}", reasons)

  perf = _perf_rows(text)
  required_stages = (
      "p32_vag_forward",
      "p32_vag_reverse",
      "segmented_value_and_grad",
      "optimizer_transaction",
      "weight_sync",
  )
  for stage in required_stages:
    _require(len(perf.get(stage, [])) == expected_updates, f"perf.{stage}={len(perf.get(stage, []))} expected={expected_updates}", reasons)
  wall_rows = [(int(step), float(seconds)) for step, seconds in _STEP_RE.findall(text)]
  _require(len(wall_rows) == expected_updates, f"global_steps={len(wall_rows)} expected={expected_updates}", reasons)

  timing_rows = []
  timing_complete = (
      len(wall_rows) == expected_updates
      and len(updates) == expected_updates
      and all(len(perf.get(stage, [])) == expected_updates for stage in required_stages)
  )
  if timing_complete:
    for index, (step, wall) in enumerate(wall_rows):
      elapsed = updates[index].get("elapsed_seconds")
      if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
        reasons.append(f"update[{index}].elapsed_seconds")
        elapsed = float("nan")
      segmented = _seconds(perf["segmented_value_and_grad"][index], f"perf.segmented[{index}]", reasons)
      forward = _seconds(perf["p32_vag_forward"][index], f"perf.forward[{index}]", reasons)
      reverse = _seconds(perf["p32_vag_reverse"][index], f"perf.reverse[{index}]", reasons)
      optimizer = _seconds(perf["optimizer_transaction"][index], f"perf.optimizer[{index}]", reasons)
      sync = _seconds(perf["weight_sync"][index], f"perf.sync[{index}]", reasons)
      system = wall - float(elapsed)
      training_other = float(elapsed) - segmented - optimizer
      if system < -0.05:
        reasons.append(f"timing[{index}].negative_system={system:.6f}")
      if training_other < -0.05:
        reasons.append(f"timing[{index}].negative_training_other={training_other:.6f}")
      timing_rows.append({
          "global_step": float(step),
          "wall_seconds": wall,
          "weight_sync_seconds": sync,
          "cycle_seconds": wall + sync,
          "training_seconds": float(elapsed),
          "system_seconds": max(system, 0.0),
          "system_including_sync_seconds": max(system, 0.0) + sync,
          "segmented_value_and_grad_seconds": segmented,
          "p32_forward_seconds": forward,
          "p32_reverse_seconds": reverse,
          "optimizer_seconds": optimizer,
          "training_other_seconds": max(training_other, 0.0),
      })
  steady, direct_eval_enclosing_steps, direct_eval_cycle_excluded_steady = (
      _steady_timing_rows(
      timing_rows,
      expected_updates=expected_updates,
      p57_eval=p57_eval,
      )
  )
  artifacts = {
      "xplane": [_artifact(path, state) for path in xplanes],
      "trace_json_gz": [_artifact(path, state) for path in trace_json],
      "semantic_perfetto": [_artifact(path, state) for path in perfetto],
  }
  if p57_eval_classification.is_file():
    artifacts["p57_inprocess_eval"] = _artifact(
        p57_eval_classification, state
    )
  return {
      "schema": "v1-hp-full-classification-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "recipe": recipe,
      "workload": contract["workload"],
      "topology": {"dp": dp_size, "tp": tp_size},
      "updates": {"expected": expected_updates, "observed": len(updates)},
      "zero_tim": {
          "expected_pass": expected_alignment_pass,
          "observed_pass": align_verdicts.count("PASS"),
          "observed_fail": align_verdicts.count("FAIL"),
          "claim_level": "strict-zero-tim",
      },
      "p59_acceptance": "ordinary-jax-fp64-gradient-correctness",
      "profiled_step": _PROFILED_STEP,
      "profiled_step_excluded_from_steady_mean": True,
      "runtime_markers": marker_counts,
      "apc": {
          "enabled": apc_on,
          "hit_rates_percent": hit_rates,
          "max_hit_rate_percent": max(hit_rates) if hit_rates else None,
      },
      "p57_inprocess_evaluation": p57_eval,
      "timing": {
          "steps": timing_rows,
          "all_mean": _mean(timing_rows),
          "steady_steps2_plus_excluding_profile_count": len(steady),
          "steady_steps2_plus_excluding_profile_mean": _mean(steady),
          "direct_eval_enclosing_global_steps": sorted(
              direct_eval_enclosing_steps
          ),
          "direct_eval_cycle_excluded_steady_count": len(
              direct_eval_cycle_excluded_steady
          ),
          "direct_eval_cycle_excluded_steady_mean": _mean(
              direct_eval_cycle_excluded_steady
          ),
      },
      "artifacts": artifacts,
      "evidence_sha256": {
          "run_log": _sha256(run_log),
          "update_report": _sha256(update_report),
          "base_classification": _sha256(base_classification),
          "resolved_env": _sha256(env_path),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--recipe", choices=tuple(_RECIPES), required=True)
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--base-classification", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite V1 classification: {args.output}")
  record = classify(
      recipe=args.recipe,
      state=args.state,
      run_log=args.run_log,
      update_report=args.update_report,
      base_classification=args.base_classification,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(
      "V1_HP_FULL_CLASSIFICATION "
      f"verdict={record['verdict']} recipe={args.recipe} "
      f"zero={record.get('zero_tim', {}).get('observed_pass', 0)}/"
      f"{record.get('zero_tim', {}).get('expected_pass', 0)} "
      f"fail={record.get('zero_tim', {}).get('observed_fail', 0)}",
      flush=True,
  )
  print(f"V1_HP_FULL_CLASSIFICATION_JSON {json.dumps(record, sort_keys=True)}", flush=True)
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
