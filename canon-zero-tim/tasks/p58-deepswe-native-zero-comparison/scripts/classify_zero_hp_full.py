#!/usr/bin/env python3
"""Postflight and performance ledger for P58.7 Zero-HP full training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any


_EXPECTED_UPDATES = 1000
_PROFILED_STEP = 2
_FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")
_STEP_RE = re.compile(r"Global step (\d+) completed in ([0-9.]+) seconds\.")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"expected object at {path}:{number}")
    rows.append(value)
  return rows


def _resolved_env(path: Path) -> dict[str, str]:
  values = {}
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.startswith("export "):
      continue
    name, encoded = line.removeprefix("export ").split("=", 1)
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


def _seconds(row: dict[str, str]) -> float:
  value = float(row["seconds"])
  if not math.isfinite(value) or value < 0.0:
    raise ValueError(f"invalid PERF seconds: {row}")
  return value


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
  if not rows:
    return {}
  keys = tuple(key for key in rows[0] if key != "global_step")
  return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def classify(
    *,
    state: Path,
    run_log: Path,
    update_report: Path,
    base_classification: Path,
) -> dict[str, Any]:
  reasons: list[str] = []
  env_path = state / "env.sh"
  required_paths = (env_path, run_log, update_report, base_classification)
  for path in required_paths:
    if not path.is_file() or path.stat().st_size == 0:
      reasons.append(f"missing_or_empty:{path.name}")
  if reasons:
    return {
        "schema": "canon.p58.zero-hp-full.v1",
        "verdict": "FAIL",
        "reasons": reasons,
    }

  env = _resolved_env(env_path)
  required_env = {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env"
      ),
      "CANON_V1_HP_FULL": "1",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": "zero",
      "CANON_P58_EXPECTED_UPDATES": str(_EXPECTED_UPDATES),
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P38_FIXED_LM_HEAD": "1",
      "CANON_CONTINUE_DECODE": "8",
      "CANON_FIXED_AR_GATHER": "1",
      "CANON_PALLAS_GATHERED_LOGPROBS": "1",
      "CANON_LOGPROB_STEP_FUSION": "1",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P28_BATCHED_REPORT": "1",
      "CANON_P28_BATCHED_REVERSE": "0",
      "CANON_BATCHED_EVIDENCE": "0",
      "CANON_FUSED_TREE_OPS": "0",
      "CANON_PALLAS_NORM_MATMUL": "0",
      "CANON_PALLAS_INPUT_FUSION": "0",
      "CANON_SAMPLE_SPLIT_FUSION": "0",
      "CANON_ENGINE_LOGPROB_READBACK": "0",
      "CANON_ANCHOR_OVERLAP": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": str(_PROFILED_STEP),
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
      "CANON_XPROF_LABELS": "1",
      "CANON_PERF_TRACE_EXPORT_STEP": str(_PROFILED_STEP),
  }
  wrong_env = {
      key: env.get(key)
      for key, expected in required_env.items()
      if env.get(key) != expected
  }
  if wrong_env:
    reasons.append(f"resolved_env={wrong_env}")

  base = json.loads(base_classification.read_text(encoding="utf-8"))
  for key, expected in {
      "verdict": "PASS",
      "arm": "zero",
      "stage": "full",
      "expected_commits": _EXPECTED_UPDATES,
      "observed_commits": _EXPECTED_UPDATES,
  }.items():
    if base.get(key) != expected:
      reasons.append(f"base.{key}={base.get(key)!r}")
  if base.get("checks", {}).get("zero_all_boundaries_exact") is not True:
    reasons.append("base.zero_all_boundaries_exact")

  updates = _jsonl(update_report)
  committed = [row for row in updates if row.get("commits") == 1]
  if len(committed) != _EXPECTED_UPDATES:
    reasons.append(f"committed_updates={len(committed)}")
  for index, row in enumerate(committed):
    expected = {
        "contract_name": "p58-qwen4b-tim-128",
        "dp_size": 8,
        "tp_size": 8,
        "dp_rank_pullbacks_per_transaction": 8,
        "dp_pullback_invocations_per_transaction": 1,
        "dp_replicas_exact": True,
        "gradient_finite": True,
        "optimizer_placement": "device-resident",
        "verdict": "PASS",
    }
    wrong = {
        key: row.get(key)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if wrong:
      reasons.append(f"update[{index}].p59={wrong}")

  text = run_log.read_text(encoding="utf-8", errors="replace")
  if re.search(r"^\[CANON_ALIGN(?:_PRE)?\].*verdict=FAIL", text, re.MULTILINE):
    reasons.append("strict_alignment_fail")
  marker_counts = {
      "continue_decode": text.count(
          "[P57.CONTINUE_DECODE] on-device decode loop enabled"
      ),
      "gathered_logprobs": text.count("[P56.GATHERED_LOGPROBS] installed"),
      "step_fusion": text.count("[P56.LOGPROB_STEP_FUSION] active"),
      "fixed_ar_gather": text.count("CANON_FIXED_AR=1 gather-ordered-sum"),
      "labels": text.count(
          "[CANON_XPROF_LABELS] continue-decode stage callables cached"
      ),
      "p59": text.count("[P59.DP8] gradient_reducer_ready"),
      "xprof_armed": text.count(
          f"[P51.XPROF] phase=update armed step={_PROFILED_STEP}"
      ),
      "xprof_started": text.count(
          f"[P51.XPROF] phase=update started step={_PROFILED_STEP}"
      ),
      "xprof_stopped": text.count(
          f"[P51.XPROF] phase=update stopped step={_PROFILED_STEP + 1}"
      ),
      "perfetto": text.count(
          f"[V1.PERFETTO] captured training_step={_PROFILED_STEP}"
      ),
  }
  for name in (
      "continue_decode", "gathered_logprobs", "step_fusion",
      "fixed_ar_gather", "labels",
  ):
    if marker_counts[name] < 1:
      reasons.append(f"marker.{name}={marker_counts[name]}")
  if marker_counts["p59"] != _EXPECTED_UPDATES:
    reasons.append(f"marker.p59={marker_counts['p59']}")
  for name in ("xprof_armed", "xprof_started", "xprof_stopped", "perfetto"):
    if marker_counts[name] != 1:
      reasons.append(f"marker.{name}={marker_counts[name]}")

  xplanes = sorted((state / "xprof-update").rglob("*.xplane.pb"))
  traces = sorted((state / "xprof-update").rglob("*.trace.json.gz"))
  perfetto = sorted((state / "perfetto").rglob("perfetto_trace_v2_*.pb"))
  fixed_head = state / "p38_fixed_lm_head_receipts.json"
  for name, paths in (
      ("xplane", xplanes),
      ("trace_json_gz", traces),
      ("perfetto", perfetto),
  ):
    if not paths or any(path.stat().st_size == 0 for path in paths):
      reasons.append(f"artifact.{name}")
  if len(perfetto) != 1:
    reasons.append(f"artifact.perfetto_count={len(perfetto)}")
  if not fixed_head.is_file() or fixed_head.stat().st_size == 0:
    reasons.append("artifact.fixed_head_receipts")

  perf = _perf_rows(text)
  stages = (
      "p32_vag_forward",
      "p32_vag_reverse",
      "segmented_value_and_grad",
      "optimizer_transaction",
      "weight_sync",
  )
  for stage in stages:
    if len(perf.get(stage, ())) != _EXPECTED_UPDATES:
      reasons.append(f"perf.{stage}={len(perf.get(stage, ()))}")
  wall = [(int(step), float(seconds)) for step, seconds in _STEP_RE.findall(text)]
  if len(wall) != _EXPECTED_UPDATES:
    reasons.append(f"global_steps={len(wall)}")

  timing_rows = []
  timing_ready = (
      len(wall) == len(committed) == _EXPECTED_UPDATES
      and all(len(perf.get(stage, ())) == _EXPECTED_UPDATES for stage in stages)
  )
  if timing_ready:
    for index, (step, wall_seconds) in enumerate(wall):
      training = float(committed[index]["elapsed_seconds"])
      forward = _seconds(perf["p32_vag_forward"][index])
      reverse = _seconds(perf["p32_vag_reverse"][index])
      segmented = _seconds(perf["segmented_value_and_grad"][index])
      optimizer = _seconds(perf["optimizer_transaction"][index])
      sync = _seconds(perf["weight_sync"][index])
      system = wall_seconds - training
      if system < -0.05:
        reasons.append(f"timing[{index}].negative_system={system}")
      timing_rows.append({
          "global_step": float(step),
          "wall_seconds": wall_seconds,
          "training_seconds": training,
          "system_seconds": max(system, 0.0),
          "weight_sync_seconds": sync,
          "cycle_including_sync_seconds": wall_seconds + sync,
          "segmented_value_and_grad_seconds": segmented,
          "p32_forward_seconds": forward,
          "p32_reverse_seconds": reverse,
          "optimizer_seconds": optimizer,
      })
  steady = [
      row for row in timing_rows
      if row["global_step"] >= 2 and row["global_step"] != _PROFILED_STEP
  ]
  return {
      "schema": "canon.p58.zero-hp-full.v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "claim_level": "optimized-strict-zero-tim-dp8-tp8-target",
      "updates": {"expected": _EXPECTED_UPDATES, "observed": len(committed)},
      "p59_acceptance": "ordinary-jax-fp64-gradient-correctness",
      "serial_adamw_trajectory_identity_claimed": False,
      "profiled_step": _PROFILED_STEP,
      "profiled_step_excluded_from_steady_mean": True,
      "runtime_markers": marker_counts,
      "timing": {
          "steps": timing_rows,
          "all_mean": _mean(timing_rows),
          "steady_steps2_plus_excluding_profile_count": len(steady),
          "steady_steps2_plus_excluding_profile_mean": _mean(steady),
      },
      "artifacts": {
          "xplane": [str(path.relative_to(state)) for path in xplanes],
          "trace_json_gz": [str(path.relative_to(state)) for path in traces],
          "perfetto": [str(path.relative_to(state)) for path in perfetto],
          "fixed_head_receipts": (
              str(fixed_head.relative_to(state)) if fixed_head.is_file() else None
          ),
      },
      "evidence_sha256": {
          "env": _sha256(env_path),
          "run_log": _sha256(run_log),
          "updates": _sha256(update_report),
          "base_classification": _sha256(base_classification),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--base-classification", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  result = classify(
      state=args.state,
      run_log=args.run_log,
      update_report=args.update_report,
      base_classification=args.base_classification,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P58_ZERO_HP_FULL_CLASSIFICATION "
      f"verdict={result['verdict']} updates={result['updates']['observed']}/"
      f"{result['updates']['expected']}",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
