#!/usr/bin/env python3
"""Render one arm of the paired 128-chip P58 DeepSWE TIM study."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Any, Mapping

import yaml

import render_p34_jobset as p34


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CLEAN_WHITELIST = (
    "canon-zero-tim/clean_data/p46_q4_learnable/"
    "p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl"
)
CLEAN_WHITELIST_SHA256 = (
    "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
)
CLEAN_ROWS = 1012
PROFILE = "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
TOPOLOGY = "4x4x8"
WORKERS = 32
ROLE_DP = 8
ROLE_TP = 8
_STAGE_STEPS = {"three-update": 3, "full": 1000}
_ARMS = ("native", "zero")
_FILTER_STATUSES = (
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
    "TIMEOUT",
    "ENV_TIMEOUT",
    "MODEL_TIMEOUT",
    "REWARD_TIMEOUT",
)


def _service_containers(head: Mapping[str, Any]) -> list[dict[str, Any]]:
  return list(head.get("initContainers", [])) + list(head["containers"])


def _remove_proxy_precision_pin(proxy: dict[str, Any]) -> None:
  env = proxy.get("env", [])
  proxy["env"] = [item for item in env if item.get("name") != p34.PROXY_XLA_ENV]


def _command(stage: str, *, run_root: str, whitelist: str) -> tuple[str, ...]:
  if stage not in _STAGE_STEPS:
    raise ValueError("P58 admits only three-update or full")
  args = list(
      p34._command("three-update", run_root=run_root, whitelist=whitelist)
  )
  replacements = {
      "--model_version=Qwen3-32B": "--model_version=Qwen3-4B-Instruct-2507",
      "--num_generations=8": "--num_generations=16",
      "--episode_timeout_secs=4800": "--episode_timeout_secs=3000",
      "--step_timeout_secs=1800": "--step_timeout_secs=600",
      "--reward_timeout_secs=1800": "--reward_timeout_secs=600",
      "--rollout_batch_timeout_secs=5400": "--rollout_batch_timeout_secs=3600",
      "--rollout_mesh_dp=16": "--rollout_mesh_dp=8",
      "--train_mesh_dp=16": "--train_mesh_dp=8",
      "--rollout_vllm_max_num_seqs=4": "--rollout_vllm_max_num_seqs=16",
      "--max_steps=3": f"--max_steps={_STAGE_STEPS[stage]}",
  }
  for old, new in replacements.items():
    if args.count(old) != 1:
      raise ValueError(f"P34 command no longer contains exactly one {old!r}")
    args[args.index(old)] = new
  args.extend((
      f"--expected_filtered_rows={CLEAN_ROWS}",
      "--loss_scale_factor=16384",
      "--loss_denominator_weighted_accumulation",
      "--overlong_filter",
      "--filter_statuses",
      *_FILTER_STATUSES,
  ))
  return tuple(args)


def render(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    stage: str,
    arm: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str = CLEAN_WHITELIST,
    whitelist_sha256: str = CLEAN_WHITELIST_SHA256,
) -> dict[str, Any]:
  """Returns one immutable P58 native or zero JobSet."""
  if stage not in _STAGE_STEPS:
    raise ValueError("P58 admits only three-update or full")
  if arm not in _ARMS:
    raise ValueError("P58 arm must be native or zero")
  if whitelist != CLEAN_WHITELIST or whitelist_sha256 != CLEAN_WHITELIST_SHA256:
    raise ValueError("P58 requires the reviewed 1012-task clean whitelist")

  # Use the bounded P34 render only as a structural JobSet constructor.  P58
  # replaces its workload contract below and validates the final document.
  document = p34.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage="three-update",
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
      fixed_lm_head=False,
  )

  name = f"canon-p58-ds4b-{arm}-{'three' if stage == 'three-update' else 'full'}-{run_id}"
  if len(name) > 63:
    raise ValueError("rendered P58 JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p58-deepswe-tim",
      "canon.zero-tim/stage": stage,
      "canon.zero-tim/arm": arm,
      "canon.zero-tim/topology": "128",
  })

  head = p34._head(document)
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  manager = p34._container(services, "pathways-rm")
  main = p34._container(head["containers"], "jax-tpu")
  scratch = f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p58/{name}"
  p34._replace_arg(
      proxy["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}"
  )
  p34._replace_arg(
      manager["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}"
  )
  p34._replace_arg(
      manager["args"], "--instance_type=", f"--instance_type=tpuv5:{TOPOLOGY}"
  )
  if arm == "native":
    _remove_proxy_precision_pin(proxy)
  else:
    p34.ensure_proxy_xla_env(proxy)

  p34._set_env(main, {
      "CANON_PROFILE_FILE": PROFILE,
      "CANON_STATE": run_root,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P34_TRAJECTORY_CAPTURE": "0",
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P39_PILOT_ADMITTED": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P43_DEBUG_ADMITTED": "0",
      "CANON_P43_ROLLOUT_ONLY": "0",
      "CANON_P44_DEEPSWE_PARITY": "0",
      "CANON_P44_PARITY_ADMITTED": "0",
      "CANON_P44_TOPOLOGY": "none",
      "CANON_P44_ROLLOUT_ONLY": "0",
      "CANON_P46_DEEPSWE_TRAIN": "0",
      "CANON_P46_EVALUATION": "0",
      "CANON_P46_TOPOLOGY": "none",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_EXPECTED_UPDATES": str(_STAGE_STEPS[stage]),
      "CANON_P58_DEBUG_DIR": f"{run_root}/debug",
      "CANON_P34_CLEAN_ROWS": str(CLEAN_ROWS),
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1" if arm == "native" else "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "3600",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "3000",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "600",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
      "MIN_TOKEN_BUCKET": "2048",
      "CANON_RUN_CMD": shlex.join(
          _command(stage, run_root=run_root, whitelist=whitelist)
      ),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-4b-native-zero",
      "CANON_WANDB_GROUP": f"qwen3-4b-p58-{stage}",
      "CANON_OPTIMIZER_HBM_MIN_FREE_BYTES": str(8 * 1024**3),
  })

  worker = p34._worker(document)
  worker["completions"] = WORKERS
  worker["parallelism"] = WORKERS
  worker_pod = worker["template"]["spec"]
  worker_pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"] = TOPOLOGY
  worker_container = p34._container(worker_pod["containers"], "pathways-worker")
  p34._replace_arg(
      worker_container["args"],
      "--instance_type=",
      f"--instance_type=tpuv5:{TOPOLOGY}",
  )
  address = f"{name}-pathways-head-0-0.{name}"
  p34._replace_arg(
      worker_container["args"],
      "--resource_manager_address=",
      f"--resource_manager_address={address}:29001",
  )
  for item in worker_container.get("env", []):
    if item.get("name") == "PATHWAYS_HEAD" and "value" in item:
      item["value"] = address

  validate(
      document,
      source_commit=source_commit,
      client_image=client_image,
      stage=stage,
      arm=arm,
  )
  return document


def recipe_signature(document: Mapping[str, Any]) -> dict[str, Any]:
  """Returns only the fields that must be equal across the paired arms."""
  env = p34._env(document)
  omitted_prefixes = (
      "--gold_whitelist=",
      "--metric_logger_dir=",
      "--ckpt_dir=",
  )
  command = tuple(
      item for item in shlex.split(env["CANON_RUN_CMD"])
      if not item.startswith(omitted_prefixes)
  )
  return {
      "command": command,
      "stage": env["CANON_P34_RUN_STAGE"],
      "source_commit": env["CANON_EXPECT_COMMIT"],
      "whitelist_sha256": env["CANON_P34_WHITELIST_SHA256"],
      "optimizer_resident": env["CANON_OPT_STATE_RESIDENT"],
      "optimizer_offload": env["CANON_P30_OPT_STATE_OFFLOAD"],
      "workers": p34._worker(document)["completions"],
  }


def treatment_signature(document: Mapping[str, Any]) -> dict[str, Any]:
  """Returns the explicitly registered numerical treatment fields."""
  env = p34._env(document)
  proxy = p34._container(
      _service_containers(p34._head(document)), "pathways-proxy"
  )
  proxy_xla = [
      item.get("value") for item in proxy.get("env", [])
      if item.get("name") == p34.PROXY_XLA_ENV
  ]
  return {
      "arm": env["CANON_P58_TIM_ARM"],
      "alignment_warning_only": env["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"],
      "proxy_xla": proxy_xla,
  }


def validate(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    client_image: str,
    stage: str,
    arm: str,
) -> None:
  if stage not in _STAGE_STEPS or arm not in _ARMS:
    raise ValueError("invalid P58 stage or arm")
  head = p34._head(document)
  worker = p34._worker(document)
  main = p34._container(head["containers"], "jax-tpu")
  env = p34._env(document)
  if document["spec"]["failurePolicy"]["maxRestarts"] != 0:
    raise ValueError("P58 must remain attempt-zero")
  if worker["backoffLimit"] != 0 or worker["completions"] != WORKERS or worker["parallelism"] != WORKERS:
    raise ValueError("P58 worker count does not match 4x4x8")
  if main["image"] != client_image or not p34._DIGEST_IMAGE.fullmatch(main["image"]):
    raise ValueError("P58 client image is not digest-pinned")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_PROFILE_FILE": PROFILE,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_EXPECTED_UPDATES": str(_STAGE_STEPS[stage]),
      "CANON_P34_CLEAN_ROWS": str(CLEAN_ROWS),
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1" if arm == "native" else "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "MIN_TOKEN_BUCKET": "2048",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
  }
  wrong = {
      key: env.get(key) for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P58 rendered environment mismatch: {wrong}")

  args = shlex.split(env["CANON_RUN_CMD"])
  required = (
      "--model_version=Qwen3-4B-Instruct-2507",
      "--batch_size=8",
      "--mini_batch_size=8",
      "--train_micro_batch_size=8",
      "--compute_logps_micro_batch_size=8",
      "--num_generations=16",
      "--max_response_length=16384",
      "--max_turns=50",
      "--temperature=1.0",
      "--top_p=1.0",
      "--top_k=0",
      "--rollout_mesh_dp=8",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=8",
      "--train_mesh_tp=8",
      "--rollout_vllm_max_num_seqs=16",
      "--max_num_batched_tokens=256",
      "--max_concurrency=64",
      "--loss_agg_mode=sequence-mean-token-scale",
      "--loss_scale_factor=16384",
      "--loss_denominator_weighted_accumulation",
      "--use_rollout_logps",
      "--overlong_filter",
      f"--expected_filtered_rows={CLEAN_ROWS}",
      f"--max_steps={_STAGE_STEPS[stage]}",
      "--no-optimizer-offload",
  )
  missing = [item for item in required if item not in args]
  if missing:
    raise ValueError(f"P58 command lost signed fields: {missing}")
  status_index = args.index("--filter_statuses")
  if tuple(args[status_index + 1:status_index + 1 + len(_FILTER_STATUSES)]) != _FILTER_STATUSES:
    raise ValueError("P58 compact-filter status set drifted")
  forbidden = ("--sampler_is", "--group_clip_filter_threshold", "--optimizer-offload")
  if any(item == value or item.startswith(value + "=") for item in args for value in forbidden):
    raise ValueError("P58 command enabled an optional algorithm intervention")
  if not env.get("CANON_P58_DEBUG_DIR", "").endswith("/debug"):
    raise ValueError("P58 trajectory journal path is missing")

  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  proxy_pins = [
      item for item in proxy.get("env", [])
      if item.get("name") == p34.PROXY_XLA_ENV
  ]
  expected_proxy_pins = (
      [] if arm == "native" else [{"name": p34.PROXY_XLA_ENV, "value": p34.PROXY_XLA_FLAG}]
  )
  if proxy_pins != expected_proxy_pins:
    raise ValueError("P58 proxy precision treatment drifted")
  manager = p34._container(services, "pathways-rm")
  if f"--instance_type=tpuv5:{TOPOLOGY}" not in manager["args"]:
    raise ValueError("P58 resource-manager topology drifted")
  worker_pod = worker["template"]["spec"]
  if worker_pod["nodeSelector"].get("cloud.google.com/gke-tpu-topology") != TOPOLOGY:
    raise ValueError("P58 worker topology drifted")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=p34.DEFAULT_SOURCE_BRANCH)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=tuple(_STAGE_STEPS), required=True)
  parser.add_argument("--arm", choices=_ARMS, required=True)
  parser.add_argument("--cpu-nodepool", default="deepswe-cpu-pool")
  parser.add_argument("--worker-nodepool", required=True)
  parser.add_argument("--model-pvc", default="haoyugao-cpu-np-pvc")
  parser.add_argument("--whitelist", default=CLEAN_WHITELIST)
  parser.add_argument("--whitelist-sha256", default=CLEAN_WHITELIST_SHA256)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite JobSet: {args.output}")
  document = render(
      yaml.safe_load(args.base.read_text()),
      source_commit=args.source_commit,
      source_branch=args.source_branch,
      client_image=args.client_image,
      run_id=args.run_id,
      stage=args.stage,
      arm=args.arm,
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      model_pvc=args.model_pvc,
      whitelist=args.whitelist,
      whitelist_sha256=args.whitelist_sha256,
  )
  args.output.write_text(p34.dump_jobset(document))
  print(f"P58_DEEPSWE_TIM_RENDER_PASS arm={args.arm} stage={args.stage} output={args.output}")


if __name__ == "__main__":
  main()
