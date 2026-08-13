#!/usr/bin/env python3
"""Renders the dual-topology Qwen3-4B DeepSWE parity-debug ladder."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Any, Mapping

import yaml

import render_p34_jobset as p34


_STAGE_STEPS = {
    "rollout-only": 1,
    "one-update": 1,
    "three-update": 3,
}
_TOPOLOGIES = {
    "64": {
        "dp": 4,
        "role_devices": 32,
        "workers": 16,
        "slice": "4x4x4",
        "global_m": 1024,
        "max_num_seqs": 4,
    },
    "128": {
        "dp": 8,
        "role_devices": 64,
        "workers": 32,
        "slice": "4x4x8",
        "global_m": 2048,
        "max_num_seqs": 2,
    },
}


def _parity_command(
    stage: str, *, topology: str, run_root: str, whitelist: str
) -> tuple[str, ...]:
  """Returns one bounded command from the shared P44 recipe."""
  if stage not in _STAGE_STEPS:
    raise ValueError(f"unknown P44 parity stage: {stage!r}")
  try:
    topology_spec = _TOPOLOGIES[topology]
  except KeyError as exc:
    raise ValueError("P44 topology must be exactly 64 or 128") from exc
  base_stage = "one-update" if stage == "rollout-only" else stage
  args = list(
      p34._command(base_stage, run_root=run_root, whitelist=whitelist)
  )
  replacements = {
      "--model_version=Qwen3-32B": (
          "--model_version=Qwen3-4B-Instruct-2507"
      ),
      "--batch_size=8": "--batch_size=4",
      "--mini_batch_size=8": "--mini_batch_size=4",
      "--train_micro_batch_size=8": "--train_micro_batch_size=4",
      "--compute_logps_micro_batch_size=8": (
          "--compute_logps_micro_batch_size=4"
      ),
      "--num_generations=8": "--num_generations=4",
      "--episode_timeout_secs=4800": "--episode_timeout_secs=3000",
      "--step_timeout_secs=1800": "--step_timeout_secs=600",
      "--reward_timeout_secs=1800": "--reward_timeout_secs=600",
      "--rollout_batch_timeout_secs=5400": (
          "--rollout_batch_timeout_secs=3600"
      ),
      "--rollout_mesh_dp=16": (
          f"--rollout_mesh_dp={topology_spec['dp']}"
      ),
      "--train_mesh_dp=16": f"--train_mesh_dp={topology_spec['dp']}",
      "--rollout_vllm_max_num_seqs=4": (
          "--rollout_vllm_max_num_seqs="
          f"{topology_spec['max_num_seqs']}"
      ),
      "--max_concurrency=64": "--max_concurrency=16",
  }
  for old, new in replacements.items():
    if args.count(old) != 1:
      raise ValueError(f"P34 command no longer contains exactly one {old!r}")
    args[args.index(old)] = new
  expected_steps = f"--max_steps={_STAGE_STEPS[stage]}"
  actual_steps = [item for item in args if item.startswith("--max_steps=")]
  if actual_steps != [expected_steps]:
    raise ValueError(
        "P44 stage step mismatch: "
        f"expected {expected_steps}, got {actual_steps}"
    )
  return tuple(args)


def _service_containers(head: Mapping[str, Any]) -> list[dict[str, Any]]:
  return list(head.get("initContainers", [])) + list(head["containers"])


def render(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    stage: str,
    topology: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
) -> dict[str, Any]:
  """Returns one immutable, attempt-zero P44 parity JobSet."""
  if stage not in _STAGE_STEPS:
    raise ValueError(
        "P44 parity admits only rollout-only, one-update, or three-update"
    )
  try:
    topology_spec = _TOPOLOGIES[topology]
  except KeyError as exc:
    raise ValueError("P44 topology must be exactly 64 or 128") from exc
  if (
      whitelist != p34.P34_CLEAN_WHITELIST
      or whitelist_sha256 != p34.P34_CLEAN_WHITELIST_SHA256
  ):
    raise ValueError(
        "P44 parity requires the reviewed 1851-image clean whitelist path "
        "and SHA-256"
    )
  base_stage = "one-update" if stage == "rollout-only" else stage
  document = p34.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage=base_stage,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
  )

  short_stage = {
      "rollout-only": "rollout",
      "one-update": "one",
      "three-update": "three",
  }[stage]
  name = f"canon-p44-ds4b-t{topology}-{short_stage}-{run_id}"
  if len(name) > 63:
    raise ValueError("rendered P44 JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p44-parity",
      "canon.zero-tim/stage": stage,
      "canon.zero-tim/topology": topology,
  })

  head = p34._head(document)
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  manager = p34._container(services, "pathways-rm")
  main = p34._container(head["containers"], "jax-tpu")
  scratch = f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p44/{name}"
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
      f"--instance_type=tpuv5:{topology_spec['slice']}",
  )

  p34._set_env(main, {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env"
      ),
      "CANON_STATE": run_root,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "1" if stage == "rollout-only" else "0",
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P39_PILOT_ADMITTED": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P43_DEBUG_ADMITTED": "0",
      "CANON_P43_ROLLOUT_ONLY": "0",
      "CANON_P44_DEEPSWE_PARITY": "1",
      "CANON_P44_PARITY_ADMITTED": "1",
      "CANON_P44_TOPOLOGY": topology,
      "CANON_P44_ROLLOUT_ONLY": "1" if stage == "rollout-only" else "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "CANON_P34_CLEAN_ROWS": str(p34.P34_CLEAN_ROWS),
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "3600",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "3000",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "600",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
      "CANON_RUN_CMD": shlex.join(
          _parity_command(
              stage,
              topology=topology,
              run_root=run_root,
              whitelist=whitelist,
          )
      ),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_P44_DEBUG_DIR": f"{run_root}/debug",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-4b-parity",
      "CANON_WANDB_GROUP": f"qwen3-4b-parity-{topology}chip",
      "MIN_TOKEN_BUCKET": str(topology_spec["global_m"]),
      "CANON_OPTIMIZER_HBM_MIN_FREE_BYTES": str(8 * 1024**3),
  })
  command = shlex.split(p34._env(document)["CANON_RUN_CMD"])
  expected_rows = f"--expected_filtered_rows={p34.P34_CLEAN_ROWS}"
  if expected_rows not in command:
    command.append(expected_rows)
    p34._set_env(main, {"CANON_RUN_CMD": shlex.join(command)})

  worker = p34._worker(document)
  worker["completions"] = topology_spec["workers"]
  worker["parallelism"] = topology_spec["workers"]
  worker_pod = worker["template"]["spec"]
  worker_pod["nodeSelector"][
      "cloud.google.com/gke-tpu-topology"
  ] = topology_spec["slice"]
  worker_container = p34._container(
      worker_pod["containers"], "pathways-worker"
  )
  p34._replace_arg(
      worker_container["args"],
      "--instance_type=",
      f"--instance_type=tpuv5:{topology_spec['slice']}",
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
      topology=topology,
  )
  return document


def recipe_signature(document: Mapping[str, Any]) -> dict[str, Any]:
  """Returns the rendered P44 recipe with topology/path fields removed."""
  env = p34._env(document)
  omitted_prefixes = (
      "--rollout_mesh_dp=",
      "--train_mesh_dp=",
      "--rollout_vllm_max_num_seqs=",
      "--gold_whitelist=",
      "--metric_logger_dir=",
      "--ckpt_dir=",
  )
  command = tuple(
      item
      for item in shlex.split(env["CANON_RUN_CMD"])
      if not item.startswith(omitted_prefixes)
  )
  return {
      "command": command,
      "stage": env["CANON_P34_RUN_STAGE"],
      "no_commit": env["CANON_P34_NO_COMMIT"],
      "optimizer_resident": env["CANON_OPT_STATE_RESIDENT"],
      "optimizer_offload": env["CANON_P30_OPT_STATE_OFFLOAD"],
      "alignment_warning_only": env["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"],
      "source_commit": env["CANON_EXPECT_COMMIT"],
      "whitelist_sha256": env["CANON_P34_WHITELIST_SHA256"],
  }


def validate(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    client_image: str,
    stage: str,
    topology: str,
) -> None:
  """Rejects topology, model, batch, artifact, or stage drift."""
  if stage not in _STAGE_STEPS:
    raise ValueError("P44 parity stage is not bounded")
  try:
    topology_spec = _TOPOLOGIES[topology]
  except KeyError as exc:
    raise ValueError("P44 topology must be exactly 64 or 128") from exc
  head = p34._head(document)
  worker = p34._worker(document)
  main = p34._container(head["containers"], "jax-tpu")
  env = p34._env(document)
  if document["spec"]["failurePolicy"]["maxRestarts"] != 0:
    raise ValueError("P44 parity must remain attempt-zero")
  if (
      worker["backoffLimit"] != 0
      or worker["completions"] != topology_spec["workers"]
      or worker["parallelism"] != topology_spec["workers"]
  ):
    raise ValueError("P44 parity worker count does not match topology")
  if main["image"] != client_image or not p34._DIGEST_IMAGE.fullmatch(
      main["image"]
  ):
    raise ValueError("P44 client image is not digest-pinned")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "1" if stage == "rollout-only" else "0",
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P44_DEEPSWE_PARITY": "1",
      "CANON_P44_PARITY_ADMITTED": "1",
      "CANON_P44_TOPOLOGY": topology,
      "CANON_P44_ROLLOUT_ONLY": "1" if stage == "rollout-only" else "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "CANON_P34_CLEAN_ROWS": str(p34.P34_CLEAN_ROWS),
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "3600",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "3000",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "600",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
      "MIN_TOKEN_BUCKET": str(topology_spec["global_m"]),
      "CANON_LOGPROB_M": "256",
  }
  wrong = {
      key: env.get(key)
      for key, expected_value in expected.items()
      if env.get(key) != expected_value
  }
  if wrong:
    raise ValueError(f"P44 rendered environment mismatch: {wrong}")
  required_args = (
      "--model_version=Qwen3-4B-Instruct-2507",
      "--batch_size=4",
      "--mini_batch_size=4",
      "--train_micro_batch_size=4",
      "--compute_logps_micro_batch_size=4",
      "--num_generations=4",
      f"--rollout_mesh_dp={topology_spec['dp']}",
      "--rollout_mesh_tp=8",
      f"--train_mesh_dp={topology_spec['dp']}",
      "--train_mesh_tp=8",
      f"--rollout_vllm_max_num_seqs={topology_spec['max_num_seqs']}",
      "--max_num_batched_tokens=256",
      "--max_response_length=16384",
      "--max_turns=50",
      "--per_turn_timeout_secs=300",
      "--episode_timeout_secs=3000",
      "--step_timeout_secs=600",
      "--reward_timeout_secs=600",
      "--cleanup_timeout_secs=300",
      "--rollout_batch_timeout_secs=3600",
      "--temperature=1.0",
      "--max_concurrency=16",
      "--no-optimizer-offload",
      f"--max_steps={_STAGE_STEPS[stage]}",
      f"--expected_filtered_rows={p34.P34_CLEAN_ROWS}",
  )
  if any(value not in env["CANON_RUN_CMD"] for value in required_args):
    raise ValueError("P44 parity command lost a signed field")
  if env.get("CANON_PROFILE_FILE") != (
      "cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env"
  ):
    raise ValueError("P44 parity profile path drifted")
  if not env.get("CANON_P44_DEBUG_DIR", "").endswith("/debug"):
    raise ValueError("P44 parity artifact path is missing")
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  p34.ensure_proxy_xla_env(proxy)
  manager = p34._container(services, "pathways-rm")
  instance_type = f"--instance_type=tpuv5:{topology_spec['slice']}"
  if instance_type not in manager["args"]:
    raise ValueError("P44 resource manager topology drifted")
  worker_pod = worker["template"]["spec"]
  if worker_pod["nodeSelector"].get(
      "cloud.google.com/gke-tpu-topology"
  ) != topology_spec["slice"]:
    raise ValueError("P44 worker topology drifted")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=p34.DEFAULT_SOURCE_BRANCH)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=tuple(_STAGE_STEPS), required=True)
  parser.add_argument("--topology", choices=tuple(_TOPOLOGIES), required=True)
  parser.add_argument("--cpu-nodepool", required=True)
  parser.add_argument("--worker-nodepool", required=True)
  parser.add_argument("--model-pvc", required=True)
  parser.add_argument("--whitelist", required=True)
  parser.add_argument("--whitelist-sha256", required=True)
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
      topology=args.topology,
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      model_pvc=args.model_pvc,
      whitelist=args.whitelist,
      whitelist_sha256=args.whitelist_sha256,
  )
  args.output.write_text(p34.dump_jobset(document))
  print(
      f"P44_PARITY_JOBSET_RENDER_PASS topology={args.topology} "
      f"output={args.output}"
  )


if __name__ == "__main__":
  main()
