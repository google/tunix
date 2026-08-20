#!/usr/bin/env python3
"""Renders the P46 DeepSWE workload families on their signed topologies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

import yaml

import render_p34_jobset as p34
import render_p44_deepswe_parity as p44


WORKLOADS = ("q4-debug", "q4-clean-eval", "q32-train")
EVAL_CLEAN_ROWS = p34.P34_CLEAN_ROWS
EVAL_LOGICAL_TASKS = 32
EVAL_PHYSICAL_TASKS = 4
EVALUATION_MODES = ("reward_only", "logprob_observer")
TOPOLOGIES = {
    "64": {"instance": "4x4x4", "workers": 16, "dp": 4, "global_m": 1024},
    "128": {"instance": "4x4x8", "workers": 32, "dp": 8, "global_m": 2048},
    "256": {"instance": "4x8x8", "workers": 64, "dp": 16, "global_m": 4096},
}
WORKLOAD_TOPOLOGIES = {
    "q4-debug": ("64", "128"),
    "q4-clean-eval": ("64", "128"),
    "q32-train": ("64", "256"),
}
_RESUME_TAG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_SHA = re.compile(r"[0-9a-f]{40}")


def _services(document: Mapping[str, Any]) -> list[dict[str, Any]]:
  head = p34._head(document)
  return list(head.get("initContainers", [])) + list(head["containers"])


def _configure_topology(
    document: dict[str, Any],
    *,
    name: str,
    topology: str,
    worker_nodepool: str,
    scratch_phase: str,
) -> None:
  spec = TOPOLOGIES[topology]
  head = p34._head(document)
  proxy = p34._container(_services(document), "pathways-proxy")
  manager = p34._container(_services(document), "pathways-rm")
  scratch = (
      f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/{scratch_phase}/{name}"
  )
  p34._replace_arg(
      proxy["args"],
      "--gcs_scratch_location=",
      f"--gcs_scratch_location={scratch}",
  )
  p34._replace_arg(
      manager["args"],
      "--gcs_scratch_location=",
      f"--gcs_scratch_location={scratch}",
  )
  p34._replace_arg(
      manager["args"],
      "--instance_type=",
      f"--instance_type=tpuv5:{spec['instance']}",
  )
  worker = p34._worker(document)
  worker["backoffLimit"] = 0
  worker["completions"] = spec["workers"]
  worker["parallelism"] = spec["workers"]
  worker_pod = worker["template"]["spec"]
  worker_pod["restartPolicy"] = "Never"
  if worker_nodepool and worker_nodepool not in ("auto", "none", "tpu-v5p-slice", "any"):
    worker_pod["nodeSelector"]["cloud.google.com/gke-nodepool"] = worker_nodepool
  else:
    worker_pod["nodeSelector"].pop("cloud.google.com/gke-nodepool", None)
  worker_pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"] = spec[
      "instance"
  ]
  worker_container = p34._container(
      worker_pod["containers"], "pathways-worker"
  )
  p34._replace_arg(
      worker_container["args"],
      "--instance_type=",
      f"--instance_type=tpuv5:{spec['instance']}",
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


def _q32_command(*, topology: str, run_root: str, whitelist: str) -> tuple[str, ...]:
  args = list(p34._command("full", run_root=run_root, whitelist=whitelist))
  if topology == "64":
    replacements = {
        "--rollout_mesh_dp=16": "--rollout_mesh_dp=4",
        "--train_mesh_dp=16": "--train_mesh_dp=4",
        "--rollout_vllm_max_num_seqs=4": "--rollout_vllm_max_num_seqs=16",
    }
    for old, new in replacements.items():
      if args.count(old) != 1:
        raise ValueError(f"P46 Q32 command lost exactly one {old!r}")
      args[args.index(old)] = new
  return tuple(args)


def _base_render(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
    fixed_lm_head: bool = False,
) -> dict[str, Any]:
  return p34.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage="full",
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
      fixed_lm_head=fixed_lm_head,
  )


def render_q4_debug(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    topology: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
    fixed_lm_head: bool = False,
) -> dict[str, Any]:
  document = p44.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage="three-update",
      topology=topology,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
      fixed_lm_head=fixed_lm_head,
  )
  document["metadata"]["labels"]["canon.zero-tim/profile-family"] = "q4-debug"
  return document


def render_q32_train(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    topology: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
    fixed_lm_head: bool = False,
) -> dict[str, Any]:
  document = _base_render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
      fixed_lm_head=fixed_lm_head,
  )
  name = f"canon-p46-q32-{topology}-{run_id}"
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p46",
      "canon.zero-tim/profile-family": "q32-train",
      "canon.zero-tim/topology": topology,
  })
  _configure_topology(
      document,
      name=name,
      topology=topology,
      worker_nodepool=worker_nodepool,
      scratch_phase="p46-q32",
  )
  main = p34._container(p34._head(document)["containers"], "jax-tpu")
  p34._set_env(main, {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-32b-dp-parity-deepswe-full.env"
      ),
      "CANON_STATE": run_root,
      "CANON_RUN_ID": run_id,
      "CANON_P46_DEEPSWE_TRAIN": "1",
      "CANON_P46_EVALUATION": "0",
      "CANON_P46_TOPOLOGY": topology,
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P39_PILOT_ADMITTED": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P43_DEBUG_ADMITTED": "0",
      "CANON_P44_DEEPSWE_PARITY": "0",
      "CANON_P44_PARITY_ADMITTED": "0",
      "CANON_P44_TOPOLOGY": "none",
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P34_TRAJECTORY_CAPTURE": "1",
      "CANON_P34_CLEAN_ROWS": str(p34.P34_CLEAN_ROWS),
      "CANON_RUN_CMD": shlex.join(
          _q32_command(topology=topology, run_root=run_root, whitelist=whitelist)
      ),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_P34_DEBUG_DIR": f"{run_root}/debug",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-qwen32b-16k",
      "CANON_WANDB_GROUP": f"qwen3-32b-16k-{topology}chip",
      "MIN_TOKEN_BUCKET": str(TOPOLOGIES[topology]["global_m"]),
  })
  validate_q32(
      document,
      source_commit=source_commit,
      client_image=client_image,
      topology=topology,
      fixed_lm_head=fixed_lm_head,
  )
  return document


def render_q4_eval(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    resume_tag: str,
    sampling_source_commit: str,
    legacy_import_id: str | None,
    frozen_v6_import_id: str | None,
    topology: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
    logical_shard_index: int,
    physical_shard_index: int,
    evaluation_mode: str,
    parity_canary: bool,
    full_campaign: bool,
    first_pass_census: bool,
) -> dict[str, Any]:
  if evaluation_mode not in EVALUATION_MODES:
    raise ValueError("unsupported P46 evaluation mode")
  if evaluation_mode == "logprob_observer" and not parity_canary:
    raise ValueError("logprob_observer is restricted to the parity canary")
  if parity_canary and topology != "64":
    raise ValueError("P46 parity canary requires topology 64")
  if full_campaign and parity_canary:
    raise ValueError("P46 full campaign cannot be a parity canary")
  if full_campaign and (logical_shard_index or physical_shard_index):
    raise ValueError("P46 full campaign owns all shard indices")
  if first_pass_census and not full_campaign:
    raise ValueError("P46 first-pass census requires a full campaign")
  if first_pass_census and evaluation_mode != "reward_only":
    raise ValueError("P46 first-pass census requires reward_only evaluation")
  if not _RESUME_TAG.fullmatch(run_id):
    raise ValueError(
        "P46 launch run id must be lowercase and Kubernetes-safe"
    )
  if not _RESUME_TAG.fullmatch(resume_tag):
    raise ValueError(
        "P46 resume tag must be lowercase, Kubernetes-safe, and at most "
        "63 characters"
    )
  if not _SHA.fullmatch(sampling_source_commit):
    raise ValueError("P46 sampling source commit must be a lowercase SHA")
  if legacy_import_id is not None:
    if not full_campaign:
      raise ValueError("P46 legacy import requires a full campaign")
    if not _RESUME_TAG.fullmatch(legacy_import_id):
      raise ValueError(
          "P46 legacy import id must be lowercase and Kubernetes-safe"
      )
  if frozen_v6_import_id is not None:
    if not full_campaign:
      raise ValueError("P46 frozen v6 import requires a full campaign")
    if not _RESUME_TAG.fullmatch(frozen_v6_import_id):
      raise ValueError(
          "P46 frozen v6 import id must be lowercase and Kubernetes-safe"
      )
  if legacy_import_id is not None and frozen_v6_import_id is not None:
    raise ValueError("P46 permits only one frozen resume import")
  document = _base_render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
  )
  logical_tasks = 1 if parity_canary else EVAL_LOGICAL_TASKS
  physical_tasks = 1 if parity_canary else EVAL_PHYSICAL_TASKS
  logical_shards = (EVAL_CLEAN_ROWS + logical_tasks - 1) // logical_tasks
  if not 0 <= logical_shard_index < logical_shards:
    raise ValueError("P46 evaluation logical shard index is out of range")
  tasks_in_logical_shard = min(
      logical_tasks,
      EVAL_CLEAN_ROWS - logical_shard_index * logical_tasks,
  )
  physical_shards = (
      tasks_in_logical_shard + physical_tasks - 1
  ) // physical_tasks
  if not 0 <= physical_shard_index < physical_shards:
    raise ValueError("P46 evaluation shard indices are out of range")
  lane = (
      f"parity-{'obs' if evaluation_mode == 'logprob_observer' else 'reward'}"
      if parity_canary
      else (
          "eval-census"
          if first_pass_census
          else ("eval-camp" if full_campaign else "eval")
      )
  )
  name = f"canon-p46-{lane}-{topology}-{run_id}"
  if not full_campaign:
    name = (
        f"canon-p46-{lane}-{topology}-{logical_shard_index}-"
        f"{physical_shard_index}-{run_id}"
    )
  if len(name) > 36:
    raise ValueError("rendered P46 evaluation JobSet name exceeds 36 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_eval/{resume_tag}"
  if parity_canary:
    run_root = f"{run_root}/parity/{evaluation_mode}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p46",
      "canon.zero-tim/profile-family": "q4-clean-eval",
      "canon.zero-tim/topology": topology,
      "canon.zero-tim/evaluation-mode": evaluation_mode,
      "canon.zero-tim/parity-canary": "1" if parity_canary else "0",
      "canon.zero-tim/full-campaign": "1" if full_campaign else "0",
      "canon.zero-tim/census-first-pass": (
          "1" if first_pass_census else "0"
      ),
      "canon.zero-tim/resume-tag": resume_tag,
  })
  if legacy_import_id is not None:
    document["metadata"]["labels"][
        "canon.zero-tim/legacy-import-id"
    ] = legacy_import_id
  if frozen_v6_import_id is not None:
    document["metadata"]["labels"][
        "canon.zero-tim/frozen-v6-import-id"
    ] = frozen_v6_import_id
  _configure_topology(
      document,
      name=name,
      topology=topology,
      worker_nodepool=worker_nodepool,
      scratch_phase="p46-eval",
  )
  main = p34._container(p34._head(document)["containers"], "jax-tpu")
  p34._set_env(main, {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-4b-dp-parity-deepswe-eval.env"
      ),
      "CANON_MODE": "run",
      "CANON_STATE": (
          f"{run_root}/state-launches/{run_id}"
          if full_campaign
          else f"{run_root}/state-l{logical_shard_index}-p{physical_shard_index}"
      ),
      "CANON_RUN_ID": run_id,
      "CANON_P46_RESUME_TAG": resume_tag,
      "CANON_P46_SAMPLING_SOURCE_COMMIT": sampling_source_commit,
      "CANON_P46_LEGACY_IMPORT_ID": legacy_import_id or "",
      "CANON_P46_FROZEN_V6_IMPORT_ID": frozen_v6_import_id or "",
      "CANON_CLIENT_IMAGE": client_image,
      "CANON_P46_DEEPSWE_TRAIN": "0",
      "CANON_P46_EVALUATION": "1",
      "CANON_P46_EVALUATION_MODE": evaluation_mode,
      "CANON_P46_PARITY_CANARY": "1" if parity_canary else "0",
      "CANON_P46_FULL_CAMPAIGN": "1" if full_campaign else "0",
      "CANON_P46_CENSUS_FIRST_PASS": "1" if first_pass_census else "0",
      "CANON_P46_TOPOLOGY": topology,
      "CANON_P34_TOPOLOGY_ADMITTED": "0",
      "CANON_P34_TP8_ADMITTED": "0",
      "CANON_P34_TRAJECTORY_ADMITTED": "0",
      "CANON_P34_UPDATE_ADMITTED": "0",
      "CANON_P34_TRAJECTORY_CAPTURE": "0",
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P32_DP_REDUCTION_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "0",
      "CANON_PALLAS_LOGSOFTMAX": "0",
      "CANON_ENGINE_MODULE_C": "0",
      "CANON_RPA_VJP2": "0",
      "CANON_ALIGNMENT_GATE": "0",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "0",
      "CANON_PRE_ALIGN_GATE": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P28_SEGMENTED_FORWARD": "0",
      "CANON_P28_SEGMENTED_VJP": "0",
      "CANON_P28_SEGMENTED_TRAIN": "0",
      "CANON_P28_G6_UPDATE": "0",
      "CANON_P29_FULL_TRAIN": "0",
      "CANON_OPT_STATE_RESIDENT": "0",
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "0",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "0",
      "CANON_P30_RELEASE_CAPTURED_STATE": "0",
      "CANON_P30_RESHARD_ACCUMULATOR": "0",
      "CANON_RUN_CMD": "python3 -u examples/deepswe/eval_deepswe.py",
      "CANON_RUN_LOG": (
          f"{run_root}/logs/campaign.log"
          if full_campaign
          else (
              f"{run_root}/logs/l{logical_shard_index}-"
              f"p{physical_shard_index}.log"
          )
      ),
      "CANON_P46_OUTPUT_DIR": f"{run_root}/outputs",
      "CANON_P46_GOLD_JSONL": whitelist,
      "CANON_P46_GOLD_JSONL_SHA256": whitelist_sha256,
      "CANON_P46_MODEL_BASE_DIR": "/mnt/disks/linchai_data/models",
      "CANON_P46_LOGICAL_SHARD_INDEX": str(logical_shard_index),
      "CANON_P46_PHYSICAL_SHARD_INDEX": str(physical_shard_index),
      "NODE_SELECTOR_VAL": cpu_nodepool,
  })
  validate_eval(
      document,
      source_commit=source_commit,
      client_image=client_image,
      topology=topology,
      resume_tag=resume_tag,
      sampling_source_commit=sampling_source_commit,
      legacy_import_id=legacy_import_id,
      frozen_v6_import_id=frozen_v6_import_id,
      evaluation_mode=evaluation_mode,
      parity_canary=parity_canary,
      full_campaign=full_campaign,
      first_pass_census=first_pass_census,
  )
  return document


def _validate_topology(document: Mapping[str, Any], topology: str) -> None:
  spec = TOPOLOGIES[topology]
  worker = p34._worker(document)
  if (
      worker["backoffLimit"] != 0
      or worker["completions"] != spec["workers"]
      or worker["parallelism"] != spec["workers"]
  ):
    raise ValueError("P46 worker cardinality drifted")
  manager = p34._container(_services(document), "pathways-rm")
  if f"--instance_type=tpuv5:{spec['instance']}" not in manager["args"]:
    raise ValueError("P46 resource-manager topology drifted")
  worker_pod = worker["template"]["spec"]
  actual_topology = worker_pod["nodeSelector"].get(
      "cloud.google.com/gke-tpu-topology"
  )
  if actual_topology != spec["instance"]:
    raise ValueError("P46 worker node topology drifted")


def validate_q32(
    document: Mapping[str, Any], *, source_commit: str, client_image: str,
    topology: str, fixed_lm_head: bool = False,
) -> None:
  _validate_topology(document, topology)
  env = p34._env(document)
  main = p34._container(p34._head(document)["containers"], "jax-tpu")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P46_DEEPSWE_TRAIN": "1",
      "CANON_P46_EVALUATION": "0",
      "CANON_P46_TOPOLOGY": topology,
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_TRAJECTORY_CAPTURE": "1",
      "CANON_P34_CLEAN_ROWS": "1851",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "5400",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_P38_FIXED_LM_HEAD": "1" if fixed_lm_head else "0",
      "MIN_TOKEN_BUCKET": str(TOPOLOGIES[topology]["global_m"]),
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P46 Q32 environment mismatch: {wrong}")
  required = (
      "--model_version=Qwen3-32B",
      "--max_response_length=16384",
      "--batch_size=8",
      "--num_generations=8",
      "--max_steps=1000",
      f"--rollout_mesh_dp={TOPOLOGIES[topology]['dp']}",
      f"--train_mesh_dp={TOPOLOGIES[topology]['dp']}",
      "--rollout_mesh_tp=8",
      "--train_mesh_tp=8",
      "--rollout_batch_timeout_secs=5400",
      "--no-optimizer-offload",
  )
  if any(item not in env["CANON_RUN_CMD"] for item in required):
    raise ValueError("P46 Q32 command lost a signed field")
  if main["image"] != client_image:
    raise ValueError("P46 Q32 client image drifted")


def validate_eval(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    client_image: str,
    topology: str,
    resume_tag: str,
    sampling_source_commit: str,
    legacy_import_id: str | None,
    frozen_v6_import_id: str | None,
    evaluation_mode: str,
    parity_canary: bool,
    full_campaign: bool,
    first_pass_census: bool,
) -> None:
  _validate_topology(document, topology)
  env = p34._env(document)
  main = p34._container(p34._head(document)["containers"], "jax-tpu")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_CLIENT_IMAGE": client_image,
      "CANON_P46_EVALUATION": "1",
      "CANON_P46_DEEPSWE_TRAIN": "0",
      "CANON_P46_TOPOLOGY": topology,
      "CANON_P46_RESUME_TAG": resume_tag,
      "CANON_P46_SAMPLING_SOURCE_COMMIT": sampling_source_commit,
      "CANON_P46_LEGACY_IMPORT_ID": legacy_import_id or "",
      "CANON_P46_FROZEN_V6_IMPORT_ID": frozen_v6_import_id or "",
      "CANON_P46_EVALUATION_MODE": evaluation_mode,
      "CANON_P46_PARITY_CANARY": "1" if parity_canary else "0",
      "CANON_P46_FULL_CAMPAIGN": "1" if full_campaign else "0",
      "CANON_P46_CENSUS_FIRST_PASS": "1" if first_pass_census else "0",
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_P34_TRAJECTORY_CAPTURE": "0",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "0",
      "CANON_PALLAS_LOGSOFTMAX": "0",
      "CANON_ENGINE_MODULE_C": "0",
      "CANON_RPA_VJP2": "0",
      "CANON_ALIGNMENT_GATE": "0",
      "CANON_ALIGNMENT_TRAIN": "0",
      "CANON_PRE_ALIGN_GATE": "0",
      "CANON_OPT_STATE_RESIDENT": "0",
      "CANON_RUN_CMD": "python3 -u examples/deepswe/eval_deepswe.py",
      "CANON_P46_GOLD_JSONL_SHA256": p34.P34_CLEAN_WHITELIST_SHA256,
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P46 evaluation environment mismatch: {wrong}")
  if main["image"] != client_image:
    raise ValueError("P46 evaluation client image drifted")


def render(
    base: Mapping[str, Any],
    *,
    workload: str,
    topology: str,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
    resume_tag: str | None = None,
    sampling_source_commit: str | None = None,
    legacy_import_id: str | None = None,
    frozen_v6_import_id: str | None = None,
    logical_shard_index: int = 0,
    physical_shard_index: int = 0,
    evaluation_mode: str = "reward_only",
    parity_canary: bool = False,
    full_campaign: bool = False,
    first_pass_census: bool = False,
    fixed_lm_head: bool = False,
) -> dict[str, Any]:
  if workload not in WORKLOADS:
    raise ValueError(f"unknown P46 workload: {workload}")
  if topology not in WORKLOAD_TOPOLOGIES[workload]:
    allowed = " or ".join(WORKLOAD_TOPOLOGIES[workload])
    raise ValueError(
        f"P46 {workload} topology must be exactly {allowed}"
    )
  if workload != "q4-clean-eval" and (
      evaluation_mode != "reward_only" or parity_canary
      or full_campaign or resume_tag is not None
      or first_pass_census or sampling_source_commit is not None
      or legacy_import_id is not None
      or frozen_v6_import_id is not None
  ):
    raise ValueError("evaluation-only controls cannot modify a training workload")
  if workload == "q4-clean-eval" and full_campaign and resume_tag is None:
    raise ValueError("P46 full campaign requires an explicit resume tag")
  if workload == "q4-clean-eval" and fixed_lm_head:
    raise ValueError("fixed lm-head is restricted to P46 training workloads")
  common = dict(
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      topology=topology,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
  )
  if workload == "q4-debug":
    return render_q4_debug(base, fixed_lm_head=fixed_lm_head, **common)
  if workload == "q32-train":
    return render_q32_train(base, fixed_lm_head=fixed_lm_head, **common)
  return render_q4_eval(
      base,
      resume_tag=resume_tag or run_id,
      sampling_source_commit=sampling_source_commit or source_commit,
      legacy_import_id=legacy_import_id,
      frozen_v6_import_id=frozen_v6_import_id,
      logical_shard_index=logical_shard_index,
      physical_shard_index=physical_shard_index,
      evaluation_mode=evaluation_mode,
      parity_canary=parity_canary,
      full_campaign=full_campaign,
      first_pass_census=first_pass_census,
      **common,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workload", choices=WORKLOADS, required=True)
  parser.add_argument("--topology", choices=tuple(TOPOLOGIES), required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=p34.DEFAULT_SOURCE_BRANCH)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument(
      "--resume-tag",
      help=(
          "stable PVC campaign identity; reuse it with a new --run-id to "
          "continue only missing trajectories"
      ),
  )
  parser.add_argument(
      "--sampling-source-commit",
      help=(
          "sampling lineage SHA; defaults to --source-commit and differs only "
          "for a reviewed legacy-v5 adoption"
      ),
  )
  parser.add_argument(
      "--legacy-import-id",
      help=(
          "frozen snapshot under <resume-root>/imports/<id>; full campaign only"
      ),
  )
  parser.add_argument(
      "--frozen-v6-import-id",
      help=(
          "sealed v6 snapshot under <resume-root>/imports/<id>; migrates "
          "exact sampling evidence into a fresh full-campaign resume tag"
      ),
  )
  parser.add_argument("--cpu-nodepool", required=True)
  parser.add_argument("--worker-nodepool", required=True)
  parser.add_argument("--model-pvc", required=True)
  parser.add_argument("--whitelist", default=p34.P34_CLEAN_WHITELIST)
  parser.add_argument(
      "--whitelist-sha256", default=p34.P34_CLEAN_WHITELIST_SHA256
  )
  parser.add_argument("--logical-shard-index", type=int, default=0)
  parser.add_argument("--physical-shard-index", type=int, default=0)
  parser.add_argument(
      "--evaluation-mode", choices=EVALUATION_MODES, default="reward_only"
  )
  parser.add_argument("--parity-canary", action="store_true")
  parser.add_argument("--full-campaign", action="store_true")
  parser.add_argument(
      "--first-pass-census",
      action="store_true",
      help=(
          "cover each identity once and defer invalid retries; full reward-only "
          "campaign only"
      ),
  )
  parser.add_argument(
      "--fixed-lm-head",
      action="store_true",
      help="enable the default-off fixed-tile Pallas output head",
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite JobSet: {args.output}")
  document = render(
      yaml.safe_load(args.base.read_text(encoding="utf-8")),
      workload=args.workload,
      topology=args.topology,
      source_commit=args.source_commit,
      source_branch=args.source_branch,
      client_image=args.client_image,
      run_id=args.run_id,
      resume_tag=args.resume_tag,
      sampling_source_commit=args.sampling_source_commit,
      legacy_import_id=args.legacy_import_id,
      frozen_v6_import_id=args.frozen_v6_import_id,
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      model_pvc=args.model_pvc,
      whitelist=args.whitelist,
      whitelist_sha256=args.whitelist_sha256,
      logical_shard_index=args.logical_shard_index,
      physical_shard_index=args.physical_shard_index,
      evaluation_mode=args.evaluation_mode,
      parity_canary=args.parity_canary,
      full_campaign=args.full_campaign,
      first_pass_census=args.first_pass_census,
      fixed_lm_head=args.fixed_lm_head,
  )
  args.output.write_text(p34.dump_jobset(document), encoding="utf-8")
  print(
      f"P46_JOBSET_RENDER_PASS workload={args.workload} "
      f"topology={args.topology} output={args.output}"
  )


if __name__ == "__main__":
  main()
