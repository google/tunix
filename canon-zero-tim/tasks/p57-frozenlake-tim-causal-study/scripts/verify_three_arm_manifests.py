#!/usr/bin/env python3
"""Verify one two-workload P57 treatment wave before cluster launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


BASE_PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
ZERO_HP_PROFILE = (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
)
WANDB_PROJECT = "zero-tim-p57-frozenlake-tim"
ARMS = {
    "native": ("mismatch", "none", "stock-fast", "0", "1"),
    "is": ("is", "token", "stock-fast", "0", "1"),
    "zero": ("zero", "none", "", "1", "0"),
}
WORKLOADS = {
    "p45": (300, 5, 2048, "", ""),
    "m15": (300, 15, 8192, "m15", "main"),
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


def verify(path: Path, *, wave: str, workload: str, source: str) -> None:
  arm, sampler, regime, fixed, warning = ARMS[wave]
  updates, turns, response, candidate, split = WORKLOADS[workload]
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  env = _env(document)
  expected = {
      "CANON_EXPECT_COMMIT": source,
      "CANON_PROFILE_FILE": (
          ZERO_HP_PROFILE if wave == "zero" else BASE_PROFILE
      ),
      "CANON_V1_HP_FULL": "1" if wave == "zero" else "0",
      "CANON_P57_TIM_ARM": arm,
      "CANON_P57_RUN_KIND": "train",
      "CANON_P57_INFERENCE_REGIME": regime,
      "CANON_P57_EXPECTED_UPDATES": str(updates),
      "CANON_P57_STOP_AFTER_STEP": str(updates),
      "CANON_P57_WORKLOAD_CANDIDATE": candidate,
      "CANON_P57_DATA_SPLIT": split,
      "CANON_P38_FIXED_LM_HEAD": fixed,
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": warning,
      "CANON_P33_ENABLE_EVAL": "1",
      "CANON_P33_DISABLE_EVAL": "0",
      "CANON_P31_ENABLE_EVAL": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_FROZENLAKE_CKPT_INTERVAL": "300",
      "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
      "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL": "0",
  }
  if wave == "zero":
    expected["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  wrong = {
      name: env.get(name)
      for name, value in expected.items()
      if env.get(name) != value
  }
  if wrong:
    raise ValueError(f"{path}: environment drifted: {wrong}")
  command = env["CANON_RUN_CMD"].split()
  required = {
      f"--max_steps={updates}",
      "--seed=42",
      f"--env_max_steps={turns}",
      "--max_prompt_length=4096",
      f"--max_response_length={response}",
      f"--sampler_is={sampler}",
      "--num_test_batches=4",
      "--eval_every_n_steps=50",
  }
  missing = sorted(required - set(command))
  sampler_args = [value for value in command if value.startswith("--sampler_is=")]
  if missing or sampler_args != [f"--sampler_is={sampler}"]:
    raise ValueError(
        f"{path}: command drifted: missing={missing} sampler={sampler_args}"
    )
  if "--evaluation_only" in command:
    raise ValueError(f"{path}: full training unexpectedly requests evaluation-only")
  if _container(document)["resources"]["limits"].get("memory") != "350G":
    raise ValueError(f"{path}: P57 memory contract drifted")

  package = Path(__file__).resolve().parents[3]
  with tempfile.TemporaryDirectory(prefix=f"p57-{workload}-{wave}-") as tmp:
    state = Path(tmp)
    result = subprocess.run(
        ["bash", "cluster/steps/00_env.sh"],
        cwd=package,
        env={
            **os.environ,
            **env,
            "CANON_PKG": str(package),
            "CANON_STATE": str(state),
            "INJECTED_HF_TOKEN": "contract-only",
            "INJECTED_WANDB_API_KEY": "contract-only",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
      raise ValueError(
          f"{path}: resolved-env preflight failed:\n"
          f"{result.stdout}\n{result.stderr}"
      )
    snapshot = (state / "env.sh").read_text(encoding="utf-8")
  stock_marker = "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train"
  if wave in ("native", "is") and stock_marker not in result.stdout:
    raise ValueError(f"{path}: native zero-TIM-off receipt is absent")
  if wave == "zero" and stock_marker in result.stdout:
    raise ValueError(f"{path}: zero arm incorrectly selected stock-fast")
  if f"export CANON_WANDB_PROJECT={WANDB_PROJECT}" not in snapshot:
    raise ValueError(f"{path}: P57 W&B project drifted")
  expected_group = f"p57-{arm}"
  if candidate:
    expected_group += f"-{candidate}-{split}"
  if f"export CANON_WANDB_GROUP={expected_group}" not in snapshot:
    raise ValueError(f"{path}: P57 W&B group drifted")
  if wave == "zero":
    for receipt in (
        "export CANON_P59_RANK_PARALLEL_BACKWARD=1",
        "export CANON_P59_CHECKED_VMA=1",
        "export CANON_P66_P59_CHECK_VMA=1",
        "export CANON_P67_P66_VMA_P59_ONLY=1",
        "export CANON_V1_HP_FIRST_UPDATE_GATE=1",
    ):
      if receipt not in snapshot:
        raise ValueError(f"{path}: optimized zero receipt absent: {receipt}")
  print(
      "P57_THREE_ARM_MANIFEST_PASS "
      f"wave={wave} workload={workload} arm={arm} sampler_is={sampler} "
      f"updates={updates} path={path}",
      flush=True,
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--wave", choices=tuple(ARMS), required=True)
  parser.add_argument("--source", required=True)
  parser.add_argument("--p45", type=Path, required=True)
  parser.add_argument("--m15", type=Path, required=True)
  args = parser.parse_args()
  if len(args.source) != 40 or any(c not in "0123456789abcdef" for c in args.source):
    raise ValueError("source must be a full lowercase 40-character commit SHA")
  verify(args.p45, wave=args.wave, workload="p45", source=args.source)
  verify(args.m15, wave=args.wave, workload="m15", source=args.source)
  print(f"P57_THREE_ARM_WAVE_PASS wave={args.wave} manifests=2", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
