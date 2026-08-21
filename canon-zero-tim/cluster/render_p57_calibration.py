#!/usr/bin/env python3
"""Render the stock-only stochastic rollout for P57 workload calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml

_CLUSTER_DIR = Path(__file__).resolve().parent
if str(_CLUSTER_DIR) not in sys.path:
  sys.path.insert(0, str(_CLUSTER_DIR))
_REPO_ROOT = _CLUSTER_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import render_p33_jobsets as p33
from examples.frozenlake import p57_workloads


_PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
_CHECKPOINT_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake"
)
_TAG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,50}[a-z0-9])?")
_MEMORY = "350G"
_PHYSICAL_CONTEXT = 16_384
_RECIPES = ("m10", "m15", "m20")
_MODE = "stochastic"
_GENERATIONS = p57_workloads.GENERATIONS_PER_PROMPT
_TEMPERATURE = "0.7"
_SCRIPT_ENTRYPOINT = (
    "python3", "-u", "examples/frozenlake/train_frozenlake_qwen3.py"
)
_MODULE_ENTRYPOINT = (
    "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3"
)


def _container(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document) -> dict[str, str]:
  return {
      item["name"]: item["value"]
      for item in _container(document)["env"]
      if "value" in item
  }


def _replace_env(document, values: dict[str, str]) -> None:
  entries = _container(document)["env"]
  by_name = {item["name"]: item for item in entries}
  for name, value in values.items():
    if name in by_name:
      by_name[name].clear()
      by_name[name].update({"name": name, "value": value})
    else:
      entries.append({"name": name, "value": value})


def _replace_arg(command: list[str], prefix: str, value: str) -> None:
  matches = [index for index, arg in enumerate(command) if arg.startswith(prefix)]
  if len(matches) != 1:
    raise ValueError(f"P57 calibration expected one {prefix!r} argument")
  command[matches[0]] = value


def _use_module_entrypoint(command: list[str]) -> None:
  if tuple(command[:3]) != _SCRIPT_ENTRYPOINT:
    raise ValueError("P57 calibration base entrypoint drifted")
  command[:3] = _MODULE_ENTRYPOINT


def _spec() -> p33.JobSpec:
  command = list(p33._frozenlake_command(1, dp_size=8, tp_size=8))
  _use_module_entrypoint(command)
  replacements = {
      "--num_generations=": f"--num_generations={_GENERATIONS}",
      "--max_prompt_length=": f"--max_prompt_length={_PHYSICAL_CONTEXT}",
      "--max_response_length=": f"--max_response_length={_PHYSICAL_CONTEXT}",
      "--env_max_steps=": "--env_max_steps=20",
      "--temperature=": f"--temperature={_TEMPERATURE}",
  }
  for prefix, value in replacements.items():
    _replace_arg(command, prefix, value)
  command.extend((
      "--num_test_batches=4",
      "--eval_every_n_steps=0",
      "--evaluation_only",
      f"--p57_calibration_mode={_MODE}",
      f"--p57_calibration_recipes={','.join(_RECIPES)}",
  ))
  return p33.JobSpec(
      key="p57-frozenlake-calibration-stochastic",
      workload="frozenlake",
      stage="full",
      profile=_PROFILE,
      no_commit=False,
      job_prefix="canon-p57-cal",
      command=tuple(command),
      enable_evaluation=False,
      dp_size=8,
      tp_size=8,
      optimizer_resident=True,
  )


def render_all(
    *,
    base_path: Path,
    output_dir: Path,
    source_commit: str,
    run_id: str,
    campaign_tag: str,
) -> tuple[Path, ...]:
  if not _TAG_RE.fullmatch(campaign_tag):
    raise ValueError("P57 calibration campaign tag is invalid")
  base = p33.load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  spec = _spec()
  path = output_dir / f"jobset-{spec.key}.yaml"
  if path.exists():
    raise FileExistsError(f"refusing to overwrite P57 calibration: {path}")
  document = p33.render_jobset(base, spec, source_commit, run_id)
  main = _container(document)
  if main.get("resources", {}).get("limits", {}).get("memory") != "200G":
    raise ValueError("P57 calibration base memory contract drifted")
  main["resources"]["limits"]["memory"] = _MEMORY
  job_name = document["metadata"]["name"]
  state = f"/tmp/canon-state/{job_name}"
  _replace_env(document, {
        "CANON_P32_TRAIN_ADMITTED": "0",
        "CANON_P32_DP_REDUCTION_ADMITTED": "0",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
        "CANON_PRE_ALIGN_GATE": "0",
        "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
        "CANON_P38_FIXED_LM_HEAD": "0",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_RUN_KIND": "calibration",
        "CANON_P57_INFERENCE_REGIME": "stock-fast",
        "CANON_P57_EXPECTED_UPDATES": "1",
        "CANON_P57_WORKLOAD_CANDIDATE": "",
        "CANON_P57_DATA_SPLIT": "",
        "CANON_P57_CALIBRATION_MODE": _MODE,
        "CANON_P57_CALIBRATION_RECIPES": ",".join(_RECIPES),
        "CANON_P57_CALIBRATION_OUTPUT": f"{state}/p57_calibration.json",
        "CANON_P57_EVAL_CHECKPOINT_STEP": "",
        "CANON_P57_EVAL_OUTPUT": "",
        "CANON_FROZENLAKE_CKPT_MODE": "new",
        "CANON_FROZENLAKE_CKPT_ROOT": _CHECKPOINT_ROOT,
        "CANON_FROZENLAKE_CKPT_TAG": f"{campaign_tag}-{_MODE}",
        "CANON_FROZENLAKE_CKPT_INTERVAL": "10",
        "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
        "ENABLE_PATHWAYS_PERSISTENCE": "1",
        "CANON_STATE": state,
        "CANON_RUN_LOG": f"{state}/run.log",
        "CANON_PRE_ALIGN_REPORT": f"{state}/pre_alignment.jsonl",
        "CANON_ALIGN_REPORT": f"{state}/alignment.jsonl",
        "CANON_UPDATE_REPORT": f"{state}/updates.jsonl",
        "CANON_WANDB_RUN_NAME": job_name,
  })
  labels = document["metadata"].setdefault("labels", {})
  labels["canon.zero-tim/tim-study"] = "p57"
  labels["canon.zero-tim/tim-arm"] = "mismatch"
  labels["canon.zero-tim/run-kind"] = "calibration"
  labels["canon.zero-tim/calibration-mode"] = _MODE
  p33.validate_jobset(
      document,
      spec,
      source_commit,
      run_id,
      fixed_lm_head=False,
      fixed_lm_head_explicit_off=True,
      alignment_warning_only=False,
      train_admitted=False,
      dp_reduction_admitted=False,
      workload_launch_admitted=False,
      pre_align_gate=False,
  )
  values = _env(document)
  expected = {
        "CANON_PROFILE_FILE": _PROFILE,
        "CANON_P38_FIXED_LM_HEAD": "0",
        "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_RUN_KIND": "calibration",
        "CANON_P57_INFERENCE_REGIME": "stock-fast",
        "CANON_P57_CALIBRATION_MODE": _MODE,
        "CANON_P57_CALIBRATION_RECIPES": ",".join(_RECIPES),
        "CANON_P33_ENABLE_EVAL": "0",
        "CANON_P33_DISABLE_EVAL": "1",
        "CANON_P31_ENABLE_EVAL": "0",
        "CANON_FROZENLAKE_CKPT_MODE": "new",
        "CANON_P32_TRAIN_ADMITTED": "0",
        "CANON_P32_DP_REDUCTION_ADMITTED": "0",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
        "CANON_PRE_ALIGN_GATE": "0",
  }
  wrong = {
      name: values.get(name)
      for name, value in expected.items()
      if values.get(name) != value
  }
  if wrong:
    raise ValueError(f"P57 calibration rendered contract drifted: {wrong}")
  run_command = values["CANON_RUN_CMD"]
  if tuple(run_command.split()[:4]) != _MODULE_ENTRYPOINT:
    raise ValueError("P57 calibration must use the module entrypoint")
  for expected_arg in (
        f"--num_generations={_GENERATIONS}",
        f"--temperature={_TEMPERATURE}",
        f"--max_prompt_length={_PHYSICAL_CONTEXT}",
        f"--max_response_length={_PHYSICAL_CONTEXT}",
        "--env_max_steps=20",
        "--evaluation_only",
  ):
    if expected_arg not in run_command.split():
      raise ValueError(f"P57 calibration command lacks {expected_arg}")
  header = (
      "# Generated by canon-zero-tim/cluster/render_p57_calibration.py.\n"
      "# Do not edit; change the reviewed renderer/profile.\n"
  )
  path.write_text(
      header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
  )
  print(f"[P57.CALIBRATION.JOBSET] RENDERED mode={_MODE} path={path}")
  return (path,)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--campaign-tag", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument(
      "--base", type=Path, default=Path(__file__).with_name("jobset-64chip.yaml")
  )
  args = parser.parse_args()
  outputs = render_all(
      base_path=args.base,
      output_dir=args.output_dir,
      source_commit=args.source_commit,
      run_id=args.run_id,
      campaign_tag=args.campaign_tag,
  )
  print(
      "[P57.CALIBRATION.JOBSET] VERDICT PASS "
      f"count={len(outputs)} recipes={','.join(_RECIPES)}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
