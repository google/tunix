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
        "profile": "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env",
        "updates": 200,
        "dp": 16,
        "tp": 4,
        "global_m": 4096,
        "local_m": 256,
        "hidden": 2048,
        "intermediate": 6144,
        "local_vocab": 37984,
        "fixed_local_vocab": 38144,
        "endpoint": "tied_embed",
        "max_grad_norm": 1.0,
        "apc": False,
        "candidate": "",
        "split": "",
    },
    "p45": {
        "workload": "frozenlake-dp8-tp8",
        "profile": "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
        "updates": 300,
        "dp": 8,
        "tp": 8,
        "global_m": 2048,
        "local_m": 256,
        "hidden": 4096,
        "intermediate": 12288,
        "local_vocab": 18992,
        "fixed_local_vocab": 19200,
        "endpoint": "untied_lm_head",
        "max_grad_norm": 100.0,
        "apc": False,
        "candidate": "",
        "split": "",
    },
    "m15": {
        "workload": "frozenlake-dp8-tp8",
        "profile": "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
        "updates": 300,
        "dp": 8,
        "tp": 8,
        "global_m": 2048,
        "local_m": 256,
        "hidden": 4096,
        "intermediate": 12288,
        "local_vocab": 18992,
        "fixed_local_vocab": 19200,
        "endpoint": "untied_lm_head",
        "max_grad_norm": 100.0,
        "apc": False,
        "candidate": "m15",
        "split": "main",
    },
}
_PROFILED_STEP = 2
_JAX_CACHE_ENV = {
    "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
    "CANON_GCS_CACHE_BUCKET": (
        "gs://yuxzhang-tunix-models/cache/p33_compilation_cache"
    ),
}
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


def _jax_cache_receipt(
    path: Path,
    *,
    phase: str,
    profile: str,
    reasons: list[str],
) -> dict[str, str]:
  _require(path.is_file(), f"missing_jax_cache_{phase}_receipt", reasons)
  if not path.is_file():
    return {}
  lines = [
      line.strip()
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  _require(len(lines) == 1, f"jax_cache_{phase}_receipt_lines", reasons)
  if len(lines) != 1 or not lines[0].startswith("[JAX_CACHE_SYNC] "):
    _require(False, f"jax_cache_{phase}_receipt_format", reasons)
    return {}
  fields = dict(_FIELD_RE.findall(lines[0]))
  expected_profile = Path(profile).stem
  expected_bucket = (
      f"{_JAX_CACHE_ENV['CANON_GCS_CACHE_BUCKET']}/{expected_profile}"
  )
  allowed_status = (
      {"hit", "empty", "error", "no-tool"}
      if phase == "restore"
      else {"saved", "empty", "error", "no-tool"}
  )
  _require(fields.get("phase") == phase, f"jax_cache_{phase}.phase", reasons)
  _require(
      fields.get("status") in allowed_status,
      f"jax_cache_{phase}.status",
      reasons,
  )
  _require(
      fields.get("profile") == expected_profile,
      f"jax_cache_{phase}.profile",
      reasons,
  )
  _require(
      fields.get("bucket") == expected_bucket,
      f"jax_cache_{phase}.bucket",
      reasons,
  )
  _require(
      fields.get("local") == _JAX_CACHE_ENV["JAX_COMPILATION_CACHE_DIR"],
      f"jax_cache_{phase}.local",
      reasons,
  )
  _require(
      str(fields.get("rc", "")).isdigit(),
      f"jax_cache_{phase}.rc",
      reasons,
  )
  _require(
      str(fields.get("entries", "")).isdigit(),
      f"jax_cache_{phase}.entries",
      reasons,
  )
  if str(fields.get("rc", "")).isdigit() and str(
      fields.get("entries", "")
  ).isdigit():
    rc = int(fields["rc"])
    entries = int(fields["entries"])
    status = fields.get("status")
    tool = fields.get("tool")
    if status in {"hit", "saved"}:
      coherent = rc == 0 and entries > 0 and tool in {"gcloud", "gsutil"}
    elif status == "empty":
      coherent = rc == 0 and entries == 0 and tool in {"gcloud", "gsutil"}
    elif status == "error":
      coherent = rc not in {0, 127} and tool in {"gcloud", "gsutil"}
    elif status == "no-tool":
      coherent = rc == 127 and tool == "none"
    else:
      coherent = False
    _require(coherent, f"jax_cache_{phase}.status_contract", reasons)
  return fields


def _xprof_restore_receipt(
    path: Path,
    *,
    remote: str,
    local_dir: Path,
    reasons: list[str],
) -> dict[str, str]:
  _require(path.is_file(), "missing_xprof_gcs_restore_receipt", reasons)
  if not path.is_file():
    return {}
  lines = [
      line.strip()
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  _require(len(lines) == 1, "xprof_gcs_restore_receipt_lines", reasons)
  if len(lines) != 1 or not lines[0].startswith("[P51.XPROF.GCS] "):
    _require(False, "xprof_gcs_restore_receipt_format", reasons)
    return {}
  fields = dict(_FIELD_RE.findall(lines[0]))
  expected = {
      "phase": "restore",
      "status": "PASS",
      "rc": "0",
      "remote": remote,
      "local": str(local_dir),
  }
  wrong = {
      name: fields.get(name)
      for name, value in expected.items()
      if fields.get(name) != value
  }
  _require(not wrong, f"xprof_gcs_restore={wrong}", reasons)
  _require(
      fields.get("tool") in {"gcloud", "gsutil"},
      "xprof_gcs_restore.tool",
      reasons,
  )
  for name in ("xplanes", "traces"):
    value = fields.get(name, "")
    _require(
        value.isdigit() and int(value) > 0,
        f"xprof_gcs_restore.{name}",
        reasons,
    )
  return fields


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
    xprof_dir: Path | None = None,
    xprof_receipt: Path | None = None,
) -> dict[str, Any]:
  contract = _RECIPES[recipe]
  expected_updates = int(contract["updates"])
  dp_size = int(contract["dp"])
  tp_size = int(contract["tp"])
  local_groups = 256 // dp_size
  expected_alignment_pass = expected_updates * (1 + local_groups)
  reasons: list[str] = []
  if xprof_dir is None:
    xprof_dir = state / "xprof-update"
  if xprof_receipt is None:
    xprof_receipt = state / "xprof_gcs_restore.receipt"

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
      "CANON_PROFILE_FILE": str(contract["profile"]),
      "CANON_V1_HP_FULL": "1",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P63_OVERFLOW_SAFE_CLIP": "1",
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
      **_JAX_CACHE_ENV,
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

  xprof_remote = env.get("CANON_XPROF_DIR", "")
  expected_xprof_prefix = (
      "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/"
      f"{state.name}/attempt-"
  )
  xprof_remote_match = re.fullmatch(
      re.escape(expected_xprof_prefix)
      + r"(?P<attempt>direct|[0-9]+)/xprof-update",
      xprof_remote,
  )
  _require(xprof_remote_match is not None, "resolved_env.CANON_XPROF_DIR", reasons)
  if xprof_remote_match is not None:
    attempt = xprof_remote_match.group("attempt")
    expected_xprof_dir = (
        state / "xprof-update"
        if attempt == "direct"
        else state / f"attempt-{attempt}" / "xprof-update"
    )
    _require(xprof_dir == expected_xprof_dir, "xprof_local_attempt_path", reasons)
    _require(
        xprof_receipt == expected_xprof_dir.parent / "xprof_gcs_restore.receipt",
        "xprof_receipt_attempt_path",
        reasons,
    )

  xprof_restore = _xprof_restore_receipt(
      xprof_receipt,
      remote=xprof_remote,
      local_dir=xprof_dir,
      reasons=reasons,
  )

  cache_receipt_paths = {
      phase: state / f"jax_cache_{phase}.receipt"
      for phase in ("restore", "save")
  }
  cache_receipts = {
      phase: _jax_cache_receipt(
          path,
          phase=phase,
          profile=str(contract["profile"]),
          reasons=reasons,
      )
      for phase, path in cache_receipt_paths.items()
  }

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

  global_m = int(contract["global_m"])
  local_m = int(contract["local_m"])
  local_vocab = int(contract["local_vocab"])
  fixed_local_vocab = int(contract["fixed_local_vocab"])
  hidden = int(contract["hidden"])
  endpoint = str(contract["endpoint"])
  head_prefix = f"[P59.DP{dp_size}] head_cotangent_partition_ready "
  expected_head_receipt = (
      head_prefix
      + f"global_shape=({global_m}, 151936) "
      + f"local_shape=({local_m},{local_vocab}) placement=data,model"
  )
  head_receipts = [
      line.strip()
      for line in text.splitlines()
      if line.strip().startswith(head_prefix)
  ]
  _require(bool(head_receipts), "p59_head_partition_receipt_missing", reasons)
  _require(
      bool(head_receipts)
      and all(line == expected_head_receipt for line in head_receipts),
      "p59_head_partition_shape_or_placement",
      reasons,
  )

  primal_records = [
      dict(_FIELD_RE.findall(line))
      for line in text.splitlines()
      if "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 " in line
      and " p59_local=1 " in line
  ]
  expected_primal = {
      "semantic_M": str(local_m),
      "fixed_M": "256",
      "K": str(hidden),
      "TP": str(tp_size),
      "local_N": str(local_vocab),
      "fixed_N": str(fixed_local_vocab),
      "BM": "128",
      "BN": "256",
      "BK": "256",
      "chunks": "1",
      "endpoint": endpoint,
      "p59_local": "1",
      "global_M": str(global_m),
      "dp": str(dp_size),
  }
  matching_primal = sum(
      all(record.get(name) == value for name, value in expected_primal.items())
      for record in primal_records
  )
  _require(
      matching_primal >= 1 and matching_primal == len(primal_records),
      "p59_fixed_head_primal_global_local_shape_or_chunks",
      reasons,
  )

  vjp_records = [
      dict(_FIELD_RE.findall(line))
      for line in text.splitlines()
      if "[PATHTRACE] CANON_" "P38_FIXED_LM_HEAD_VJP=1 " in line
      and " tp_input_reduction=" in line
  ]
  expected_vjp = {
      "semantic_M": str(global_m),
      "local_M": str(local_m),
      "fixed_M": "256",
      "chunks": "1",
      "accumulation": "lax.scan",
      "order": "ascending",
      "tp_input_reduction": "all_gather_rank_order_f32_barrier",
      "K": str(hidden),
      "TP": str(tp_size),
      "local_N": str(local_vocab),
      "fixed_N": str(fixed_local_vocab),
      "endpoint": endpoint,
  }
  matching_vjp = sum(
      all(record.get(name) == value for name, value in expected_vjp.items())
      for record in vjp_records
  )
  _require(
      matching_vjp >= 1 and matching_vjp == len(vjp_records),
      "p59_fixed_head_vjp_global_local_shape_chunks_or_reduction",
      reasons,
  )

  local_q_heads = 4
  local_kv_heads = 8 // tp_size
  expected_rpa_receipt = (
      "[PATHTRACE] P59_RPA_LOCAL_KV_READY "
      f"tp={tp_size} local_q_heads={local_q_heads} "
      f"local_kv_heads={local_kv_heads} cache_heads={local_kv_heads} "
      "packing=2"
  )
  rpa_receipts = [
      line.strip()
      for line in text.splitlines()
      if line.strip().startswith("[PATHTRACE] P59_RPA_LOCAL_KV_READY ")
  ]
  _require(bool(rpa_receipts), "p59_rpa_local_kv_receipt_missing", reasons)
  _require(
      bool(rpa_receipts)
      and all(line == expected_rpa_receipt for line in rpa_receipts),
      "p59_rpa_local_kv_shape_or_topology",
      reasons,
  )

  fused_linear_records = [
      dict(_FIELD_RE.findall(line))
      for line in text.splitlines()
      if line.strip().startswith(
          "[PATHTRACE] P59_LOCAL_FUSED_LINEAR_READY "
      )
  ]
  expected_fused_linear = {
      "tp": str(tp_size),
      "local_width": str(int(contract["intermediate"]) // tp_size),
      "declared_width": str(contract["intermediate"]),
      "layout_shards": "1",
      "pieces": "1",
  }
  matching_fused_linear = [
      record
      for record in fused_linear_records
      if record.get("site") in ("gate_proj", "up_proj")
      and all(
          record.get(name) == value
          for name, value in expected_fused_linear.items()
      )
  ]
  _require(
      bool(fused_linear_records),
      "p59_local_fused_linear_receipt_missing",
      reasons,
  )
  _require(
      len(matching_fused_linear) == len(fused_linear_records),
      "p59_local_fused_linear_shape_or_topology",
      reasons,
  )
  _require(
      {record.get("site") for record in matching_fused_linear}
      == {"gate_proj", "up_proj"},
      "p59_local_fused_linear_sites",
      reasons,
  )

  updates = _json_lines(update_report)
  _require(len(updates) == expected_updates, f"updates={len(updates)} expected={expected_updates}", reasons)
  p63_fallbacks = 0
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
    commit_evidence = update.get("commit_evidence")
    clip = (
        commit_evidence.get("overflow_safe_clip")
        if isinstance(commit_evidence, dict)
        else None
    )
    if not isinstance(clip, dict):
      reasons.append(f"update[{index}].p63_missing")
      continue
    all_finite = clip.get("all_finite") is True
    naive_finite = clip.get("naive_norm_finite") is True
    fallback = clip.get("fallback_used") is True
    numeric = {}
    for name in (
        "stable_norm", "selected_norm", "clip_factor", "max_norm"
    ):
      value = clip.get(name)
      numeric[name] = (
          float(value)
          if isinstance(value, (int, float)) and not isinstance(value, bool)
          else float("nan")
      )
    valid = (
        clip.get("enabled") is True
        and all_finite
        and fallback == (not naive_finite)
        and math.isfinite(numeric["stable_norm"])
        and numeric["stable_norm"] >= 0.0
        and math.isfinite(numeric["selected_norm"])
        and numeric["selected_norm"] >= 0.0
        and math.isfinite(numeric["clip_factor"])
        and 0.0 < numeric["clip_factor"] <= 1.0
        and numeric["max_norm"] == float(contract["max_grad_norm"])
    )
    naive_norm = clip.get("naive_norm")
    if fallback:
      valid = (
          valid
          and naive_norm == "inf"
          and numeric["selected_norm"] == numeric["stable_norm"]
      )
      p63_fallbacks += 1
    else:
      valid = (
          valid
          and isinstance(naive_norm, (int, float))
          and not isinstance(naive_norm, bool)
          and math.isfinite(float(naive_norm))
          and numeric["selected_norm"] == float(naive_norm)
      )
    if not valid:
      reasons.append(f"update[{index}].p63_invalid={clip}")
  if recipe == "gsm8k":
    _require(p63_fallbacks >= 1, "p63_gsm8k_fallback_not_observed", reasons)

  marker_counts = {
      "continue_decode": text.count("[P57.CONTINUE_DECODE] on-device decode loop enabled"),
      "gathered_logprobs": text.count("[P56.GATHERED_LOGPROBS] installed"),
      "logprob_step_fusion": text.count("[P56.LOGPROB_STEP_FUSION] active"),
      "fixed_ar_gather": text.count("CANON_FIXED_AR=1 gather-ordered-sum"),
      "xprof_labels": text.count("[CANON_XPROF_LABELS] continue-decode stage callables cached"),
      "p59_head_partition": text.count(
          f"[P59.DP{dp_size}] head_cotangent_partition_ready"
      ),
      "p59_rpa_local_kv": len(rpa_receipts),
      "p59_local_fused_linear": len(fused_linear_records),
      "p59_parallel": text.count(f"[P59.DP{dp_size}] gradient_reducer_ready"),
      "p63_configured": text.count(
          "[P63.STABLE_CLIP] configured enabled=1 mode=hybrid "
      ),
      "p63_updates": text.count("[P63.STABLE_CLIP] update="),
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
  _require(
      marker_counts["p59_rpa_local_kv"] >= 1,
      f"marker.p59_rpa_local_kv={marker_counts['p59_rpa_local_kv']}",
      reasons,
  )
  _require(
      marker_counts["p59_local_fused_linear"] >= 2,
      "marker.p59_local_fused_linear="
      f"{marker_counts['p59_local_fused_linear']}",
      reasons,
  )
  _require(marker_counts["p59_parallel"] == expected_updates, f"marker.p59_parallel={marker_counts['p59_parallel']} expected={expected_updates}", reasons)
  _require(
      marker_counts["p63_configured"] == 1,
      f"marker.p63_configured={marker_counts['p63_configured']} expected=1",
      reasons,
  )
  _require(
      marker_counts["p63_updates"] == expected_updates,
      f"marker.p63_updates={marker_counts['p63_updates']} expected={expected_updates}",
      reasons,
  )
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
  if contract["workload"] == "frozenlake-dp8-tp8":
    _require(text.count(apc_marker) == 1, "apc_runtime_marker", reasons)
    opposite_apc_marker = (
        f"[P3_APC_CONFIG] enabled={int(not apc_on)} "
        "workload=frozenlake reader=train_frozenlake_qwen3"
    )
    _require(
        opposite_apc_marker not in text,
        "opposite_apc_runtime_marker",
        reasons,
    )
  else:
    _require("[P3_APC_CONFIG] enabled=1" not in text, "unexpected_apc_on", reasons)
  if apc_on:
    _require(bool(hit_rates) and max(hit_rates) > 0.0, "apc_positive_cache_hit", reasons)

  xplanes = sorted(xprof_dir.rglob("*.xplane.pb"))
  trace_json = sorted(xprof_dir.rglob("*.trace.json.gz"))
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
  for phase, path in cache_receipt_paths.items():
    if path.is_file():
      artifacts[f"jax_cache_{phase}_receipt"] = _artifact(path, state)
  if xprof_receipt.is_file():
    artifacts["xprof_gcs_restore_receipt"] = _artifact(xprof_receipt, state)
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
      "p59_fixed_head_contract": {
          "profile": contract["profile"],
          "global_shape": [global_m, 151936],
          "local_shape": [local_m, local_vocab],
          "matching_head_partition_receipts": len(head_receipts),
          "matching_local_primal_receipts": matching_primal,
          "matching_local_vjp_receipts": matching_vjp,
      },
      "profiled_step": _PROFILED_STEP,
      "profiled_step_excluded_from_steady_mean": True,
      "runtime_markers": marker_counts,
      "apc": {
          "enabled": apc_on,
          "hit_rates_percent": hit_rates,
          "max_hit_rate_percent": max(hit_rates) if hit_rates else None,
      },
      "jax_persistent_cache": {
          "configuration": {
              name: env.get(name) for name in _JAX_CACHE_ENV
          },
          "receipts": cache_receipts,
      },
      "xprof_gcs_restore": xprof_restore,
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
  parser.add_argument("--xprof-dir", type=Path)
  parser.add_argument("--xprof-receipt", type=Path)
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
      xprof_dir=args.xprof_dir,
      xprof_receipt=args.xprof_receipt,
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
