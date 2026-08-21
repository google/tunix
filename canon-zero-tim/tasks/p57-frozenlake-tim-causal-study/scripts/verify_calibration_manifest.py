#!/usr/bin/env python3
"""Fail-closed preflight for the single P57 stock-fast calibration JobSet."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import yaml


_PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
_EXPECTED_ENV = {
    "CANON_PROFILE_FILE": _PROFILE,
    "CANON_P57_TIM_ARM": "mismatch",
    "CANON_P57_RUN_KIND": "calibration",
    "CANON_P57_INFERENCE_REGIME": "stock-fast",
    "CANON_P57_CALIBRATION_MODE": "stochastic",
    "CANON_P57_CALIBRATION_RECIPES": "m10,m15,m20",
    "CANON_P38_FIXED_LM_HEAD": "0",
    "CANON_P32_TRAIN_ADMITTED": "0",
    "CANON_P32_DP_REDUCTION_ADMITTED": "0",
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
    "CANON_PRE_ALIGN_GATE": "0",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
}
_EXPECTED_ARGS = {
    "--evaluation_only",
    "--max_steps=1",
    "--num_generations=8",
    "--temperature=0.7",
    "--max_prompt_length=16384",
    "--max_response_length=16384",
    "--env_max_steps=20",
    "--p57_calibration_mode=stochastic",
    "--p57_calibration_recipes=m10,m15,m20",
}


def _container(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  matches = [item for item in pod["containers"] if item["name"] == "jax-tpu"]
  if len(matches) != 1:
    raise ValueError(f"expected one jax-tpu container, found {len(matches)}")
  return matches[0]


def verify(path: Path) -> dict[str, str]:
  documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
  if len(documents) != 1 or not isinstance(documents[0], dict):
    raise ValueError("P57 calibration file must contain exactly one JobSet")
  document = documents[0]
  if document.get("kind") != "JobSet":
    raise ValueError("P57 calibration manifest is not a JobSet")
  labels = document.get("metadata", {}).get("labels", {})
  expected_labels = {
      "canon.zero-tim/tim-study": "p57",
      "canon.zero-tim/tim-arm": "mismatch",
      "canon.zero-tim/run-kind": "calibration",
      "canon.zero-tim/calibration-mode": "stochastic",
  }
  wrong_labels = {
      key: labels.get(key)
      for key, value in expected_labels.items()
      if labels.get(key) != value
  }
  entries = _container(document).get("env", [])
  env = {item["name"]: item.get("value") for item in entries}
  if len(env) != len(entries):
    raise ValueError("P57 calibration manifest contains duplicate env names")
  wrong_env = {
      key: env.get(key)
      for key, value in _EXPECTED_ENV.items()
      if env.get(key) != value
  }
  command = set(shlex.split(env.get("CANON_RUN_CMD", "")))
  missing_args = sorted(_EXPECTED_ARGS - command)
  if wrong_labels or wrong_env or missing_args:
    raise ValueError(
        "P57 calibration manifest drifted: "
        f"labels={wrong_labels} env={wrong_env} args={missing_args}"
    )
  return {
      "name": document["metadata"]["name"],
      "regime": env["CANON_P57_INFERENCE_REGIME"],
      "recipes": env["CANON_P57_CALIBRATION_RECIPES"],
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("manifest", type=Path)
  args = parser.parse_args()
  receipt = verify(args.manifest)
  print(
      "[P57.CALIBRATION.PREFLIGHT] PASS "
      f"name={receipt['name']} regime={receipt['regime']} "
      f"recipes={receipt['recipes']}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
