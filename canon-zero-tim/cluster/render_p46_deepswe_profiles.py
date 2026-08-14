#!/usr/bin/env python3
"""Renders the P46 DeepSWE workload families on their signed topologies."""

from __future__ import annotations

import argparse
from pathlib import Path
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
  worker_pod["nodeSelector"]["cloud.google.com/gke-nodepool"] = worker_nodepool
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
  )
  return document


def render_q4_eval(
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
    logical_shard_index: int,
    physical_shard_index: int,
    evaluation_mode: str,
    parity_canary: bool,
    full_campaign: bool,
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
      else ("eval-campaign" if full_campaign else "eval")
  )
  name = f"canon-p46-{lane}-{topology}-{run_id}"
  if not full_campaign:
    name = (
        f"canon-p46-{lane}-{topology}-{logical_shard_index}-"
        f"{physical_shard_index}-{run_id}"
    )
  if len(name) > 63:
    raise ValueError("rendered P46 evaluation JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_eval/{run_id}"
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
  })
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
          f"{run_root}/state-campaign"
          if full_campaign
          else f"{run_root}/state-l{logical_shard_index}-p{physical_shard_index}"
      ),
      "CANON_RUN_ID": run_id,
      "CANON_CLIENT_IMAGE": client_image,
      "CANON_P46_DEEPSWE_TRAIN": "0",
      "CANON_P46_EVALUATION": "1",
      "CANON_P46_EVALUATION_MODE": evaluation_mode,
      "CANON_P46_PARITY_CANARY": "1" if parity_canary else "0",
      "CANON_P46_FULL_CAMPAIGN": "1" if full_campaign else "0",
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
      evaluation_mode=evaluation_mode,
      parity_canary=parity_canary,
      full_campaign=full_campaign,
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
    document: Mapping[str, Any], *, source_commit: str, client_image: str, topology: str
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
    evaluation_mode: str,
    parity_canary: bool,
    full_campaign: bool,
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
      "CANON_P46_EVALUATION_MODE": evaluation_mode,
      "CANON_P46_PARITY_CANARY": "1" if parity_canary else "0",
      "CANON_P46_FULL_CAMPAIGN": "1" if full_campaign else "0",
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
    logical_shard_index: int = 0,
    physical_shard_index: int = 0,
    evaluation_mode: str = "reward_only",
    parity_canary: bool = False,
    full_campaign: bool = False,
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
      or full_campaign
  ):
    raise ValueError("evaluation-only controls cannot modify a training workload")
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
    return render_q4_debug(base, **common)
  if workload == "q32-train":
    return render_q32_train(base, **common)
  return render_q4_eval(
      base,
      logical_shard_index=logical_shard_index,
      physical_shard_index=physical_shard_index,
      evaluation_mode=evaluation_mode,
      parity_canary=parity_canary,
      full_campaign=full_campaign,
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
  )
  args.output.write_text(p34.dump_jobset(document), encoding="utf-8")
  print(
      f"P46_JOBSET_RENDER_PASS workload={args.workload} "
      f"topology={args.topology} output={args.output}"
  )


if __name__ == "__main__":
  main()
