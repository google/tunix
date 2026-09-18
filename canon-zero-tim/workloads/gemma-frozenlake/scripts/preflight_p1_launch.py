#!/usr/bin/env python3
"""Fail-closed launch-preflight trio for Gemma 4 E4B P1 admission."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping


REPO_FROM_SCRIPT = Path(__file__).resolve().parents[4]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


admission = _load(
    "gemma4_e4b_admission_preflight",
    REPO_FROM_SCRIPT / "tunix/rl/gemma4_e4b_admission.py",
)
renderer = _load(
    "gemma4_e4b_renderer_preflight",
    Path(__file__).with_name("render_p1_onehost_stock_admission.py"),
)

EXPECTED_BRANCH = "local/gemma4-e4b-zero-tim-0904"
EXPECTED_ENV = {
    "CANON_PROFILE": "gemma4-e4b-dp1-tp4-frozenlake",
    "CANON_EXPECT_JAX_VERSION": "0.10.2",
    "CANON_EXPECT_PATHWAYS_RELEASE": "20260730-jax_0.10.2",
    "CANON_EXPECT_VISIBLE_DEVICES": "4",
    "CANON_DP_SIZE": "1",
    "CANON_TP_SIZE": "4",
    "FL_ROLLOUT_MESH": "1,4",
    "FL_TRAINER_MESH": "1,4",
    "CANON_ENGINE_MODULE_C": "0",
    "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
    "ROLLOUT_ENGINE": "vllm",
    "WANDB_MODE": "disabled",
    "NEW_MODEL_DESIGN": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


def _git(repo: Path, *args: str) -> str:
  return subprocess.check_output(
      ("git", "-C", str(repo), *args), text=True
  ).strip()


def _resolved_env(
    workload: str, environ: Mapping[str, str]
) -> dict[str, Any]:
  expected = {**EXPECTED_ENV, admission.SELECTOR_ENV: workload}
  actual = {key: environ.get(key) for key in sorted(expected)}
  wrong = {
      key: {"actual": actual[key], "expected": expected[key]}
      for key in expected
      if actual[key] != expected[key]
  }
  try:
    selected = admission.active_workload(environ)
    admission.require_stock_invocation((), environ)
  except (RuntimeError, ValueError) as exc:
    wrong["stock_contract"] = {"error": str(exc)}
    selected = None
  if selected != workload:
    wrong["selected_workload"] = {
        "actual": selected,
        "expected": workload,
    }
  return {
      "schema": "gemma4-e4b-p1-resolved-env-v1",
      "status": "PASS" if not wrong else "FAIL",
      "values": actual,
      "failures": wrong,
  }


def _contract_sweep(workload: str) -> dict[str, Any]:
  checks: list[str] = []

  def accepts(name: str, fn) -> None:
    fn()
    checks.append(name)

  def rejects(name: str, fn, exception=Exception) -> None:
    try:
      fn()
    except exception:
      checks.append(name)
      return
    raise AssertionError(f"negative contract did not fire: {name}")

  base_runtime = dict(
      workload=workload,
      device_count=4,
      local_device_count=4,
      process_count=1,
      platforms=("tpu",),
      device_kinds=("TPU v5",),
      device_ids=(0, 1, 2, 3),
      device_process_indices=(0, 0, 0, 0),
      rollout_mesh_shape=(1, 4),
      trainer_mesh_shape=(1, 4),
      jax_version=admission.EXPECTED_JAX_VERSION,
      checkpoint_root=None,
      prefix_caching=False,
  )
  contract = admission.workload_contract(workload)
  accepts(
      "selector_positive",
      lambda: admission.active_workload({admission.SELECTOR_ENV: workload}),
  )
  rejects(
      "selector_negative",
      lambda: admission.active_workload({admission.SELECTOR_ENV: "wrong"}),
      ValueError,
  )
  accepts("stock_positive", lambda: admission.require_stock_invocation((), {}))
  rejects(
      "canonical_flag_negative",
      lambda: admission.require_stock_invocation(
          (), {"CANON_P59_RANK_PARALLEL_BACKWARD": "1"}
      ),
      ValueError,
  )
  accepts(
      "runtime_positive",
      lambda: admission.require_runtime_contract(**base_runtime),
  )
  rejects(
      "runtime_platform_negative",
      lambda: admission.require_runtime_contract(
          **{**base_runtime, "platforms": ("cpu",)}
      ),
      RuntimeError,
  )
  accepts(
      "shape_positive",
      lambda: admission.shape_ledger(
          workload,
          semantic_prompts=1,
          generations_per_prompt=1,
          mini_batch_size=1,
          train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
          max_num_seqs=4,
          max_num_batched_tokens=4096,
          kv_cache_size=contract["context_hard_cap"] + 256,
      ),
  )
  rejects(
      "shape_capacity_negative",
      lambda: admission.shape_ledger(
          workload,
          semantic_prompts=1,
          generations_per_prompt=1,
          mini_batch_size=1,
          train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
          max_num_seqs=4,
          max_num_batched_tokens=4096,
          kv_cache_size=contract["context_hard_cap"] - 1,
      ),
      RuntimeError,
  )
  identity = admission.identity_receipt(workload)
  if identity["mesh"] != {"dp": 1, "tp": 4}:
    raise AssertionError("identity mesh drifted")
  checks.append("identity_positive")
  return {
      "schema": "gemma4-e4b-p1-contract-sweep-v1",
      "status": "PASS",
      "checks": checks,
      "passed": len(checks),
      "expected": 9,
  }


def _intent_diff(repo: Path, workload: str, manifest: dict) -> dict[str, Any]:
  expected = renderer.render(repo, workload, admission.RUNTIME_IMAGE_DIGEST)
  branch = _git(repo, "branch", "--show-current")
  failures = []
  if branch != EXPECTED_BRANCH:
    failures.append("branch")
  if manifest != expected:
    failures.append("rendered_manifest")
  execution = manifest.get("execution", {})
  required_execution = {
      "host_count": 1,
      "local_tpu_devices": 4,
      "mesh": {"dp": 1, "tp": 4},
      "semantic_prompts": 1,
      "generations_per_prompt": 1,
      "mini_batch_size": 1,
      "train_micro_batch_size": 1,
      "compute_logps_micro_batch_size": 1,
      "rollout_only": True,
      "prefix_caching": False,
      "checkpoint": False,
      "backward": 0,
      "optimizer_commits": 0,
  }
  if any(execution.get(key) != value for key, value in required_execution.items()):
    failures.append("execution")
  reconstruction = manifest.get("source_reconstruction", {})
  if (
      reconstruction.get("base_commit") != manifest.get("source_commit")
      or reconstruction.get("schema")
      != "gemma4-e4b-p1-source-reconstruction-v1"
  ):
    failures.append("source_reconstruction")
  return {
      "schema": "gemma4-e4b-p1-intent-diff-v1",
      "status": "PASS" if not failures else "FAIL",
      "branch": branch,
      "source_commit": manifest.get("source_commit"),
      "source_tree_sha256": manifest.get("source_tree_sha256"),
      "manifest_exact": manifest == expected,
      "failures": failures,
  }


def evaluate(
    repo: Path,
    workload: str,
    manifest: dict,
    environ: Mapping[str, str],
) -> dict[str, Any]:
  failures = []
  resolved_env = _resolved_env(workload, environ)
  try:
    contract_sweep = _contract_sweep(workload)
  except (AssertionError, RuntimeError, ValueError) as exc:
    contract_sweep = {
        "schema": "gemma4-e4b-p1-contract-sweep-v1",
        "status": "FAIL",
        "error": str(exc),
    }
  try:
    intent_diff = _intent_diff(repo, workload, manifest)
  except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
    intent_diff = {
        "schema": "gemma4-e4b-p1-intent-diff-v1",
        "status": "FAIL",
        "error": str(exc),
    }
  for name, receipt in (
      ("resolved_env", resolved_env),
      ("contract_sweep", contract_sweep),
      ("intent_diff", intent_diff),
  ):
    if receipt.get("status") != "PASS":
      failures.append(name)
  return {
      "schema": "gemma4-e4b-p1-launch-preflight-v1",
      "verdict": "PASS" if not failures else "FAIL",
      "workload": workload,
      "failures": failures,
      "resolved_env": resolved_env,
      "contract_sweep": contract_sweep,
      "intent_diff": intent_diff,
  }


def _write(path: Path, value: dict[str, Any]) -> None:
  if path.exists():
    raise FileExistsError(f"refusing to overwrite {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(value, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--workload", choices=tuple(admission.WORKLOADS), required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = evaluate(
      args.repo.resolve(),
      args.workload,
      json.loads(args.manifest.read_text(encoding="utf-8")),
      os.environ,
  )
  _write(args.output, result)
  print(
      "[GEMMA4_E4B_P1_LAUNCH_PREFLIGHT] "
      + json.dumps(result, sort_keys=True),
      flush=True,
  )
  print(
      f"GEMMA4_E4B_P1_LAUNCH_PREFLIGHT_{result['verdict']} "
      f"workload={args.workload}",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
