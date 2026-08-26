#!/usr/bin/env python3
"""Render immutable P45/M15 DP8xTP8 checked-VMA precheck JobSets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

_TASK_DIR = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _TASK_DIR.parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_CLUSTER_DIR = _PACKAGE_ROOT / "cluster"
for path in (_REPO_ROOT, _CLUSTER_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import render_p33_jobsets as p33

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-ab-debug.env"


def _main(document: dict) -> dict:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document: dict) -> dict[str, str]:
  return {item["name"]: item["value"] for item in _main(document)["env"] if "value" in item}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    *, source_commit: str, run_id: str, output_dir: Path, workload: str,
    arm: str, base_path: Path
) -> Path:
  if not _SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be exactly 40 lowercase hex")
  if workload not in ("p45", "m15"):
    raise ValueError("workload must be p45 or m15")
  if arm not in ("p66-off", "serving-scope"):
    raise ValueError("arm must be p66-off or serving-scope")
  if output_dir.exists():
    raise FileExistsError(f"refusing to overwrite output root: {output_dir}")
  output_dir.mkdir(parents=True)
  command = list(p33._frozenlake_command(1, dp_size=8, tp_size=8))
  command[:3] = ("python3", "-u", "-m")
  command.insert(3, "examples.frozenlake.train_frozenlake_qwen3")
  if workload == "m15":
    command = [
        "--max_response_length=8192" if value == "--max_response_length=2048" else
        "--env_max_steps=15" if value == "--env_max_steps=5" else value
        for value in command
    ]
    command.extend(("--p57_workload_candidate=m15", "--p57_data_split=main"))
  command.extend(("--sampler_is=none", "--seed=42", "--eval_every_n_steps=0"))
  spec = p33.JobSpec(
      key=f"v1-fl-tp8-ab-{workload}-{arm}",
      workload="frozenlake",
      stage="backward-no-commit",
      profile=_PROFILE,
      no_commit=True,
      job_prefix=f"canon-v1fl-ab-{workload}",
      command=tuple(command),
      dp_size=8,
      tp_size=8,
      optimizer_resident=True,
      rank_parallel_backward=True,
      fixed_lm_head=True,
      strict_alignment=True,
  )
  document = p33.render_jobset(p33.load_base(base_path), spec, source_commit, run_id)
  values = _env(document)
  state = values["CANON_STATE"]
  p33._set_named_env(
      _main(document)["env"],
      {
          "CANON_V1_FL_TP8_AB_ARM": arm,
          "CANON_P57_WORKLOAD_CANDIDATE": "m15" if workload == "m15" else "",
          "CANON_P57_DATA_SPLIT": "main" if workload == "m15" else "",
          "CANON_P38_PRECHECK_ONLY": "1",
          "CANON_P38_CONTROLLED_EXIT": "1",
          "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
          "CANON_P38_DIAGNOSTIC_ROUND_FILE": f"{state}/p38_diagnostic_round",
          "CANON_P38_MIN_ACTION_KV": "3936" if workload == "m15" else "1686",
          "CANON_V1_HP_FULL": "0",
          "CANON_P59_CHECKED_VMA": "0" if arm == "p66-off" else "1",
          "CANON_P66_P59_CHECK_VMA": "0" if arm == "p66-off" else "1",
          "CANON_P67_P66_VMA_P59_ONLY": "0" if arm == "p66-off" else "1",
          "CANON_V1_HP_FIRST_UPDATE_GATE": "0",
          "CANON_CONTINUE_DECODE": "8",
          "CANON_FIXED_AR_GATHER": "1",
          "CANON_PALLAS_GATHERED_LOGPROBS": "1",
          "CANON_LOGPROB_STEP_FUSION": "1",
          "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
          "CANON_P28_BATCHED_REPORT": "1",
          "CANON_P28_BATCHED_REVERSE": "0",
          "CANON_BATCHED_EVIDENCE": "0",
          "CANON_FUSED_TREE_OPS": "0",
          "CANON_P63_OVERFLOW_SAFE_CLIP": "0",
      },
      remove=(),
  )
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/diagnostic": "frozenlake-tp8-ab",
      "canon.zero-tim/workload": workload,
      "canon.zero-tim/arm": arm,
      "canon.zero-tim/optimizer-commits": "0",
  })
  document["spec"]["failurePolicy"]["maxRestarts"] = 0
  worker_template = document["spec"]["replicatedJobs"][1]["template"]["spec"]["template"]
  worker_template.get("metadata", {}).get("annotations", {}).pop(
      "alpha.jobset.sigs.k8s.io/exclusive-topology", None
  )
  p33.validate_jobset(document, spec, source_commit, run_id)
  values = _env(document)
  required = {
      "CANON_PROFILE_FILE": _PROFILE,
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_V1_FL_TP8_AB_ARM": arm,
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P59_CHECKED_VMA": "0" if arm == "p66-off" else "1",
      "CANON_P66_P59_CHECK_VMA": "0" if arm == "p66-off" else "1",
      "CANON_P67_P66_VMA_P59_ONLY": "0" if arm == "p66-off" else "1",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_CONTINUE_DECODE": "8",
      "CANON_FIXED_AR_GATHER": "1",
      "CANON_PALLAS_GATHERED_LOGPROBS": "1",
      "CANON_LOGPROB_STEP_FUSION": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
  }
  wrong = {name: values.get(name) for name, expected in required.items() if values.get(name) != expected}
  if wrong:
    raise ValueError(f"rendered TP8 A/B contract drifted: {wrong}")
  output = output_dir / f"jobset-v1-fl-tp8-ab-{workload}-{arm}.yaml"
  output.write_text(
      "# Generated by render_fl_tp8_ab_diagnostic.py. Do not edit.\n"
      + yaml.safe_dump(document, sort_keys=False),
      encoding="utf-8",
  )
  receipt = {
      "schema": "v1-fl-tp8-ab-render-v1",
      "source_commit": source_commit,
      "run_id": run_id,
      "workload": workload,
      "arm": arm,
      "jobset": document["metadata"]["name"],
      "path": str(output),
      "sha256": _sha256(output),
      "state": state,
      "run_log": values["CANON_RUN_LOG"],
      "pre_alignment": values["CANON_PRE_ALIGN_REPORT"],
      "classification": f"{state}/v1_fl_tp8_ab.classification.json",
      "backward": 0,
      "optimizer_commits": 0,
  }
  (output_dir / "render-receipt.json").write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"V1_FL_TP8_AB_RENDER_PASS workload={workload} arm={arm} "
      f"jobset={receipt['jobset']} sha256={receipt['sha256']} backward=0 optimizer_commits=0",
      flush=True,
  )
  return output


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--workload", required=True, choices=("p45", "m15"))
  parser.add_argument("--arm", required=True, choices=("p66-off", "serving-scope"))
  parser.add_argument("--base", type=Path, default=_CLUSTER_DIR / "jobset-64chip.yaml")
  args = parser.parse_args()
  render(
      source_commit=args.source_commit,
      run_id=args.run_id,
      output_dir=args.output_dir,
      workload=args.workload,
      arm=args.arm,
      base_path=args.base,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
