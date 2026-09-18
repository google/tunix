#!/usr/bin/env python3
"""Fail closed on one P57 initial/final recovery-evaluation schedule."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
STEPS = (0, 300)
ARMS = {
    "native": ("mismatch", "none", "stock-fast", "0"),
    "is": ("is", "token", "stock-fast", "0"),
    "zero": ("zero", "none", "", "1"),
}
WORKLOADS = {
    "p45": (5, 2048, "", ""),
    "m15": (15, 8192, "m15", "main"),
}


def _container(document: dict) -> dict:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document: dict) -> dict[str, str]:
  return {
      item["name"]: item["value"]
      for item in _container(document)["env"]
      if "value" in item
  }


def _manifest(root: Path, workload: str, step: int, arm: str) -> Path:
  suffix = f"{arm}-eval-{step}"
  if workload == "m15":
    suffix = f"{arm}-m15-main-eval-{step}"
  return root / workload / f"step-{step}" / f"jobset-p57-frozenlake-{suffix}.yaml"


def verify_manifest(
    path: Path,
    *,
    wave: str,
    workload: str,
    source: str,
    campaign_root: str,
    step: int,
) -> None:
  arm, sampler, regime, fixed = ARMS[wave]
  turns, response, candidate, split = WORKLOADS[workload]
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  env = _env(document)
  checkpoint_mode = "new" if step == 0 else "resume"
  expected = {
      "CANON_EXPECT_COMMIT": source,
      "CANON_PROFILE_FILE": PROFILE,
      "CANON_P57_TIM_ARM": arm,
      "CANON_P57_RUN_KIND": "eval",
      "CANON_P57_INFERENCE_REGIME": regime,
      "CANON_P57_EXPECTED_UPDATES": "300",
      "CANON_P57_WORKLOAD_CANDIDATE": candidate,
      "CANON_P57_DATA_SPLIT": split,
      "CANON_P57_EVAL_CHECKPOINT_STEP": str(step),
      "CANON_P38_FIXED_LM_HEAD": fixed,
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_FROZENLAKE_CKPT_MODE": checkpoint_mode,
      "CANON_FROZENLAKE_CKPT_TAG": f"{campaign_root}-{workload}-{arm}",
      "CANON_FROZENLAKE_CKPT_INTERVAL": "300",
      "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
      "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL": "0",
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
      "CANON_P31_ENABLE_EVAL": "0",
  }
  wrong = {
      name: env.get(name)
      for name, value in expected.items()
      if env.get(name) != value
  }
  if wrong:
    raise ValueError(f"{path}: evaluation environment drifted: {wrong}")
  command = env["CANON_RUN_CMD"].split()
  required = {
      "--max_steps=300",
      f"--env_max_steps={turns}",
      "--max_prompt_length=4096",
      f"--max_response_length={response}",
      f"--sampler_is={sampler}",
      "--temperature=0",
      "--num_test_batches=4",
      "--eval_every_n_steps=0",
      "--evaluation_only",
  }
  missing = sorted(required - set(command))
  if missing:
    raise ValueError(f"{path}: evaluation command missing {missing}")
  if command.count(f"--sampler_is={sampler}") != 1:
    raise ValueError(f"{path}: sampler mode is duplicated or conflicting")
  if _container(document)["resources"]["limits"].get("memory") != "350G":
    raise ValueError(f"{path}: P57 memory contract drifted")

  package = Path(__file__).resolve().parents[3]
  with tempfile.TemporaryDirectory(prefix=f"p57-eval-{workload}-{step}-") as tmp:
    result = subprocess.run(
        ["bash", "cluster/steps/00_env.sh"],
        cwd=package,
        env={
            **os.environ,
            **env,
            "CANON_PKG": str(package),
            "CANON_STATE": tmp,
            "INJECTED_HF_TOKEN": "contract-only",
            "INJECTED_WANDB_API_KEY": "contract-only",
        },
        text=True,
        capture_output=True,
        check=False,
    )
  if result.returncode:
    raise ValueError(
        f"{path}: resolved-env preflight failed:\n{result.stdout}\n{result.stderr}"
    )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--wave", choices=tuple(ARMS), required=True)
  parser.add_argument("--source", required=True)
  parser.add_argument("--campaign-root", required=True)
  args = parser.parse_args()
  if len(args.source) != 40 or any(
      char not in "0123456789abcdef" for char in args.source
  ):
    raise ValueError("source must be a full lowercase 40-character commit SHA")
  arm = ARMS[args.wave][0]
  count = 0
  for workload in WORKLOADS:
    for step in STEPS:
      path = _manifest(args.root, workload, step, arm)
      verify_manifest(
          path,
          wave=args.wave,
          workload=workload,
          source=args.source,
          campaign_root=args.campaign_root,
          step=step,
      )
      count += 1
  extras = sorted(
      path for path in args.root.rglob("jobset-*.yaml")
      if path.is_file()
  )
  if len(extras) != count:
    raise ValueError(
        f"evaluation schedule manifest count drifted: found={len(extras)} "
        f"expected={count}"
    )
  print(
      "P57_RECOVERY_EVAL_SCHEDULE_PASS "
      f"wave={args.wave} manifests={count} steps="
      + ",".join(str(step) for step in STEPS),
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
