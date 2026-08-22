#!/usr/bin/env python3
"""Render paired P57 FrozenLake training or checkpoint-evaluation arms."""

from __future__ import annotations

import argparse
import dataclasses
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
_ALLOWED_UPDATES = (1, 3, 20, 50, 100, 150, 200, 450)
_BASE_MEMORY = "200G"
_P57_MEMORY = "350G"
_DP_SIZE = 8
_TP_SIZE = 8
_EVAL_GENERATIONS = p57_workloads.GENERATIONS_PER_PROMPT
_SCRIPT_ENTRYPOINT = (
    "python3", "-u", "examples/frozenlake/train_frozenlake_qwen3.py"
)
_MODULE_ENTRYPOINT = (
    "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3"
)
_TRAIN_ARM_ENV_DIFFERENCES = {
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
    "CANON_FROZENLAKE_CKPT_TAG",
    "CANON_P38_FIXED_LM_HEAD",
    "CANON_P57_TIM_ARM",
    "CANON_P57_INFERENCE_REGIME",
    "CANON_STATE",
    "CANON_RUN_LOG",
    "CANON_PRE_ALIGN_REPORT",
    "CANON_ALIGN_REPORT",
    "CANON_UPDATE_REPORT",
    "CANON_WANDB_RUN_NAME",
}
_EVAL_ARM_ENV_DIFFERENCES = (
    _TRAIN_ARM_ENV_DIFFERENCES
    - {"CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"}
    | {"CANON_P57_EVAL_OUTPUT"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class Arm:
  name: str
  fixed_lm_head: bool
  warning_only: bool


_ARMS = (
    Arm("zero", fixed_lm_head=True, warning_only=False),
    Arm("mismatch", fixed_lm_head=False, warning_only=True),
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
  env = _container(document)["env"]
  by_name = {item["name"]: item for item in env}
  for name, value in values.items():
    if name in by_name:
      by_name[name].clear()
      by_name[name].update({"name": name, "value": value})
    else:
      env.append({"name": name, "value": value})


def _replace_command_arg(
    command: list[str], prefix: str, replacement: str
) -> None:
  matches = [index for index, value in enumerate(command) if value.startswith(prefix)]
  if len(matches) != 1:
    raise ValueError(f"P57 expected exactly one {prefix!r} command argument")
  command[matches[0]] = replacement


def _use_module_entrypoint(command: list[str]) -> None:
  if tuple(command[:3]) != _SCRIPT_ENTRYPOINT:
    raise ValueError("P57 base entrypoint drifted")
  command[:3] = _MODULE_ENTRYPOINT


def _validate_purity_command(command: list[str]) -> None:
  if command.count("--sampler_is=none") != 1:
    raise ValueError("P57 recipe must disable sampler/TIS correction exactly once")
  if "--sampler_is=token" in command:
    raise ValueError("P57 recipe must not enable token sampler/TIS correction")


def _spec(
    arm: Arm,
    expected_updates: int,
    *,
    run_kind: str,
    checkpoint_step: int | None,
    workload_candidate: str,
    data_split: str,
) -> p33.JobSpec:
  command = list(
      p33._frozenlake_command(  # pylint: disable=protected-access
          expected_updates, dp_size=_DP_SIZE, tp_size=_TP_SIZE
      )
  )
  _use_module_entrypoint(command)
  command.append("--sampler_is=none")
  if workload_candidate:
    candidate = p57_workloads.candidate(workload_candidate)
    p57_workloads.validate_split(data_split)
    _replace_command_arg(
        command, "--env_max_steps=", f"--env_max_steps={candidate.max_turns}"
    )
    command.extend((
        f"--p57_workload_candidate={workload_candidate}",
        f"--p57_data_split={data_split}",
    ))
    _replace_command_arg(
        command, "--max_prompt_length=", "--max_prompt_length=4096"
    )
    _replace_command_arg(
        command,
        "--max_response_length=",
        f"--max_response_length={candidate.context_hard_cap - 4096}",
    )
  if run_kind == "train":
    command.append("--eval_every_n_steps=0")
    key_suffix = str(expected_updates)
    job_prefix = f"canon-p57-fl-{arm.name[:4]}"
  elif run_kind == "eval":
    if checkpoint_step is None:
      raise ValueError("P57 eval spec requires a checkpoint step")
    if _EVAL_GENERATIONS % _DP_SIZE:
      raise ValueError(
          "P57 eval generations must be divisible by the trainer DP axis: "
          f"generations={_EVAL_GENERATIONS} dp={_DP_SIZE}"
      )
    _replace_command_arg(
        command,
        "--num_generations=",
        f"--num_generations={_EVAL_GENERATIONS}",
    )
    _replace_command_arg(command, "--temperature=", "--temperature=0")
    command.extend((
        "--num_test_batches=4",
        "--eval_every_n_steps=0",
        "--evaluation_only",
    ))
    key_suffix = f"eval-{checkpoint_step}"
    job_prefix = f"canon-p57-fl-ev-{arm.name[:4]}"
  else:
    raise ValueError(f"unsupported P57 run kind: {run_kind!r}")
  _validate_purity_command(command)
  workload_suffix = (
      f"{workload_candidate}-{data_split}-" if workload_candidate else ""
  )
  if workload_candidate:
    job_prefix = f"{job_prefix}-{workload_candidate}"
  return p33.JobSpec(
      key=f"p57-frozenlake-{arm.name}-{workload_suffix}{key_suffix}",
      workload="frozenlake",
      stage="full",
      profile=_PROFILE,
      no_commit=False,
      job_prefix=job_prefix,
      command=tuple(command),
      enable_evaluation=False,
      dp_size=_DP_SIZE,
      tp_size=_TP_SIZE,
      optimizer_resident=True,
  )


def _validate_pair(documents: dict[str, dict], *, run_kind: str) -> None:
  zero = _env(documents["zero"])
  mismatch = _env(documents["mismatch"])
  differing = {
      name
      for name in zero.keys() | mismatch.keys()
      if zero.get(name) != mismatch.get(name)
  }
  expected_differences = (
      _TRAIN_ARM_ENV_DIFFERENCES
      if run_kind == "train"
      else _EVAL_ARM_ENV_DIFFERENCES
  )
  if differing != expected_differences:
    raise ValueError(
        "P57 paired intent diff changed: "
        f"actual={sorted(differing)} expected={sorted(expected_differences)}"
    )
  if zero["CANON_RUN_CMD"] != mismatch["CANON_RUN_CMD"]:
    raise ValueError("P57 arms must run the identical recipe command")
  _validate_purity_command(zero["CANON_RUN_CMD"].split())
  if (zero["CANON_P38_FIXED_LM_HEAD"], mismatch["CANON_P38_FIXED_LM_HEAD"]) != (
      "1",
      "0",
  ):
    raise ValueError("P57 treatment assignment drifted")
  print(
      "[P57.PAIR] INTENT_DIFF_PASS "
      f"run_kind={run_kind} allowed={','.join(sorted(expected_differences))}",
      flush=True,
  )


def render_all(
    *,
    base_path: Path,
    output_dir: Path,
    source_commit: str,
    run_id: str,
    campaign_tag: str,
    checkpoint_mode: str,
    expected_updates: int,
    run_kind: str = "train",
    checkpoint_step: int | None = None,
    workload_candidate: str = "",
    data_split: str = "",
    stock_only: bool = False,
    stop_after_step: int | None = None,
) -> tuple[Path, ...]:
  if expected_updates not in _ALLOWED_UPDATES:
    raise ValueError(
        f"P57 expected updates must be one of {_ALLOWED_UPDATES}, "
        f"got {expected_updates}"
    )
  if checkpoint_mode not in ("new", "resume"):
    raise ValueError("P57 checkpoint mode must be new or resume")
  if run_kind not in ("train", "eval"):
    raise ValueError("P57 run kind must be train or eval")
  if run_kind == "train" and checkpoint_step is not None:
    raise ValueError("P57 training must not name an evaluation checkpoint")
  if bool(workload_candidate) != bool(data_split):
    raise ValueError("P57 workload candidate and data split must be set together")
  if workload_candidate:
    p57_workloads.candidate(workload_candidate)
    p57_workloads.validate_split(data_split)
  if stock_only and (not workload_candidate or data_split != "selection"):
    raise ValueError("P57 stock-only training/eval requires a selection recipe")
  if stock_only and (
      workload_candidate != "m15" or expected_updates != 200
  ):
    raise ValueError("P57 stock curve is frozen to M15 selection for 200 updates")
  if not stock_only and workload_candidate and data_split != "main":
    raise ValueError("P57 paired arms require the frozen main data split")
  if run_kind == "eval":
    expected_mode = "new" if checkpoint_step == 0 else "resume"
    if checkpoint_mode != expected_mode:
      raise ValueError(
          f"P57 checkpoint step {checkpoint_step} requires mode={expected_mode}"
      )
    if (
        checkpoint_step is None
        or checkpoint_step < 0
        or checkpoint_step % 10
        or checkpoint_step > expected_updates
    ):
      raise ValueError(
          "P57 evaluation checkpoint must be zero or a 10-step boundary "
          "within the registered training horizon"
      )
  if not _TAG_RE.fullmatch(campaign_tag):
    raise ValueError("P57 campaign tag is invalid or too long")
  if run_kind == "train":
    stop_after_step = (
        expected_updates if stop_after_step is None else stop_after_step
    )
    if (
        stop_after_step not in (50, 100, 150, 200)
        or stop_after_step > expected_updates
    ):
      raise ValueError("P57 stop-after-step must be a 50-step boundary in horizon")
  elif stop_after_step is not None:
    raise ValueError("P57 evaluation does not accept a training stop boundary")
  base = p33.load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  documents: dict[str, dict] = {}
  outputs = []
  selected_arms = (
      tuple(arm for arm in _ARMS if arm.name == "mismatch")
      if stock_only
      else _ARMS
  )
  for arm in selected_arms:
    spec = _spec(
        arm,
        expected_updates,
        run_kind=run_kind,
        checkpoint_step=checkpoint_step,
        workload_candidate=workload_candidate,
        data_split=data_split,
    )
    path = output_dir / f"jobset-{spec.key}.yaml"
    if path.exists():
      raise FileExistsError(f"refusing to overwrite rendered P57 JobSet: {path}")
    document = p33.render_jobset(base, spec, source_commit, run_id)
    main = _container(document)
    if main.get("resources", {}).get("limits", {}).get("memory") != _BASE_MEMORY:
      raise ValueError("P57 base jax-tpu memory limit drifted")
    main["resources"]["limits"]["memory"] = _P57_MEMORY
    job_name = document["metadata"]["name"]
    state = f"/tmp/canon-state/{job_name}"
    checkpoint_tag = f"{campaign_tag}-{arm.name}"
    _replace_env(
        document,
        {
            "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": (
                "1" if run_kind == "train" and arm.warning_only else "0"
            ),
            "CANON_P38_FIXED_LM_HEAD": "1" if arm.fixed_lm_head else "0",
            "CANON_P57_TIM_ARM": arm.name,
            "CANON_P57_RUN_KIND": run_kind,
            "CANON_P57_INFERENCE_REGIME": (
                "stock-fast" if arm.name == "mismatch" else ""
            ),
            "CANON_P57_EXPECTED_UPDATES": str(expected_updates),
            "CANON_P57_STOP_AFTER_STEP": (
                str(stop_after_step) if run_kind == "train" else ""
            ),
            "CANON_P57_WORKLOAD_CANDIDATE": workload_candidate,
            "CANON_P57_DATA_SPLIT": data_split,
            "CANON_P57_EVAL_CHECKPOINT_STEP": (
                str(checkpoint_step) if checkpoint_step is not None else ""
            ),
            "CANON_P57_EVAL_OUTPUT": (
                f"{state}/p57_evaluation.json" if run_kind == "eval" else ""
            ),
            "CANON_FROZENLAKE_CKPT_MODE": checkpoint_mode,
            "CANON_FROZENLAKE_CKPT_ROOT": _CHECKPOINT_ROOT,
            "CANON_FROZENLAKE_CKPT_TAG": checkpoint_tag,
            "CANON_FROZENLAKE_CKPT_INTERVAL": "10",
            "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
            "ENABLE_PATHWAYS_PERSISTENCE": "1",
            "CANON_STATE": state,
            "CANON_RUN_LOG": f"{state}/run.log",
            "CANON_PRE_ALIGN_REPORT": f"{state}/pre_alignment.jsonl",
            "CANON_ALIGN_REPORT": f"{state}/alignment.jsonl",
            "CANON_UPDATE_REPORT": f"{state}/updates.jsonl",
            "CANON_WANDB_RUN_NAME": job_name,
        },
    )
    labels = document["metadata"].setdefault("labels", {})
    labels["canon.zero-tim/tim-study"] = "p57"
    labels["canon.zero-tim/tim-arm"] = arm.name
    labels["canon.zero-tim/run-kind"] = run_kind
    labels["canon.zero-tim/workload-candidate"] = (
        workload_candidate or "readiness"
    )
    labels["canon.zero-tim/data-split"] = data_split or "readiness"
    p33.validate_jobset(
        document,
        spec,
        source_commit,
        run_id,
        fixed_lm_head=arm.fixed_lm_head,
        fixed_lm_head_explicit_off=not arm.fixed_lm_head,
        alignment_warning_only=(run_kind == "train" and arm.warning_only),
    )
    env = _env(document)
    expected = {
        "CANON_PROFILE_FILE": _PROFILE,
        "CANON_P57_TIM_ARM": arm.name,
        "CANON_P57_RUN_KIND": run_kind,
        "CANON_P57_INFERENCE_REGIME": (
            "stock-fast" if arm.name == "mismatch" else ""
        ),
        "CANON_P57_EXPECTED_UPDATES": str(expected_updates),
        "CANON_P57_WORKLOAD_CANDIDATE": workload_candidate,
        "CANON_P57_DATA_SPLIT": data_split,
        "CANON_P33_ENABLE_EVAL": "0",
        "CANON_P33_DISABLE_EVAL": "1",
        "CANON_P31_ENABLE_EVAL": "0",
        "CANON_FROZENLAKE_CKPT_MODE": checkpoint_mode,
        "CANON_FROZENLAKE_CKPT_ROOT": _CHECKPOINT_ROOT,
        "CANON_FROZENLAKE_CKPT_TAG": checkpoint_tag,
    }
    if run_kind == "eval":
      expected.update({
          "CANON_P57_EVAL_CHECKPOINT_STEP": str(checkpoint_step),
          "CANON_P57_EVAL_OUTPUT": f"{state}/p57_evaluation.json",
          "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      })
    else:
      expected["CANON_P57_STOP_AFTER_STEP"] = str(stop_after_step)
    wrong = {key: env.get(key) for key, value in expected.items() if env.get(key) != value}
    if wrong:
      raise ValueError(f"P57 rendered contract drifted: {wrong}")
    if main["resources"]["limits"].get("memory") != _P57_MEMORY:
      raise ValueError("P57 memory override was not retained")
    documents[arm.name] = document
    outputs.append(path)

  if stock_only:
    stock = _env(documents["mismatch"])
    if (
        stock.get("CANON_P38_FIXED_LM_HEAD") != "0"
        or stock.get("CANON_P57_TIM_ARM") != "mismatch"
        or stock.get("CANON_P57_INFERENCE_REGIME") != "stock-fast"
        or stock.get("CANON_P57_WORKLOAD_CANDIDATE") != workload_candidate
        or stock.get("CANON_P57_DATA_SPLIT") != data_split
    ):
      raise ValueError("P57 stock-only discovery intent drifted")
    print(
        "[P57.STOCK] INTENT_PASS "
        f"run_kind={run_kind} candidate={workload_candidate} "
        f"split={data_split} fixed_lm_head=0",
        flush=True,
    )
  else:
    _validate_pair(documents, run_kind=run_kind)
  for arm, path in zip(selected_arms, outputs, strict=True):
    header = (
        "# Generated by canon-zero-tim/cluster/render_p57_frozenlake_tim.py.\n"
        "# Do not edit; change the reviewed renderer/profile.\n"
    )
    path.write_text(
        header + yaml.safe_dump(documents[arm.name], sort_keys=False),
        encoding="utf-8",
    )
    print(f"[P57.JOBSET] RENDERED arm={arm.name} path={path}")
  return tuple(outputs)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--campaign-tag", required=True)
  parser.add_argument("--checkpoint-mode", choices=("new", "resume"), default="new")
  parser.add_argument("--expected-updates", type=int, required=True)
  parser.add_argument("--run-kind", choices=("train", "eval"), default="train")
  parser.add_argument("--checkpoint-step", type=int)
  parser.add_argument("--workload-candidate", choices=tuple(p57_workloads.CANDIDATES), default="")
  parser.add_argument("--data-split", choices=("calibration", "selection", "main"), default="")
  parser.add_argument("--stock-only", action="store_true")
  parser.add_argument("--stop-after-step", type=int)
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
      checkpoint_mode=args.checkpoint_mode,
      expected_updates=args.expected_updates,
      run_kind=args.run_kind,
      checkpoint_step=args.checkpoint_step,
      workload_candidate=args.workload_candidate,
      data_split=args.data_split,
      stock_only=args.stock_only,
      stop_after_step=args.stop_after_step,
  )
  print(
      "[P57.JOBSET] VERDICT PASS "
      f"count={len(outputs)} updates={args.expected_updates} "
      f"checkpoint_mode={args.checkpoint_mode} run_kind={args.run_kind} "
      f"checkpoint_step={args.checkpoint_step if args.checkpoint_step is not None else 'none'}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
