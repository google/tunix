#!/usr/bin/env python3
"""Renders the bounded 64-chip DeepSWE DP4xTP8 resident-state pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Any, Mapping

import yaml

import render_p34_jobset as p34


_STAGES = {"one-update", "three-update"}


def _pilot_command(
    stage: str, *, run_root: str, whitelist: str
) -> tuple[str, ...]:
  """Returns the P34 algorithm with only the bounded pilot geometry changed."""
  args = list(p34._command(stage, run_root=run_root, whitelist=whitelist))
  replacements = {
      "--max_response_length=32768": "--max_response_length=4096",
      "--max_turns=50": "--max_turns=5",
      "--rollout_mesh_dp=16": "--rollout_mesh_dp=4",
      "--train_mesh_dp=16": "--train_mesh_dp=4",
      "--rollout_vllm_max_num_seqs=4": "--rollout_vllm_max_num_seqs=16",
      "--optimizer_offload=True": "--optimizer_offload=False",
  }
  for old, new in replacements.items():
    if args.count(old) != 1:
      raise ValueError(f"P34 command no longer contains exactly one {old!r}")
    args[args.index(old)] = new
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
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    whitelist: str,
    whitelist_sha256: str,
) -> dict[str, Any]:
  """Returns one strict attempt-zero pilot derived from the reviewed base."""
  if stage not in _STAGES:
    raise ValueError("P39 pilot admits only one-update or three-update")
  document = p34.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage=stage,
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
  )

  old_name = document["metadata"]["name"]
  name = f"canon-p39-ds64-{stage.replace('-update', '')}-{run_id}"
  if len(name) > 63:
    raise ValueError("rendered P39 JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p39-pilot",
      "canon.zero-tim/stage": stage,
  })

  head = p34._head(document)
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  manager = p34._container(services, "pathways-rm")
  main = p34._container(head["containers"], "jax-tpu")
  scratch = f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p39/{name}"
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
      manager["args"], "--instance_type=", "--instance_type=tpuv5:4x4x4"
  )

  p34._set_env(main, {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-32b-dp4-tp8-deepswe-pilot.env"
      ),
      "CANON_STATE": run_root,
      "CANON_P39_64CHIP_PILOT": "1",
      "CANON_P39_PILOT_ADMITTED": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "CANON_RUN_CMD": shlex.join(
          _pilot_command(stage, run_root=run_root, whitelist=whitelist)
      ),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-dp4-tp8-pilot",
      "CANON_WANDB_GROUP": "qwen3-32b-dp4-tp8-resident",
      "MIN_TOKEN_BUCKET": "1024",
      "CANON_EXPECTED_SLICE_DEVICES": "64",
      "CANON_OPTIMIZER_HBM_MIN_FREE_BYTES": str(8 * 1024**3),
  })

  worker = p34._worker(document)
  worker["completions"] = 16
  worker["parallelism"] = 16
  worker_pod = worker["template"]["spec"]
  worker_pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"] = "4x4x4"
  worker_container = p34._container(
      worker_pod["containers"], "pathways-worker"
  )
  p34._replace_arg(
      worker_container["args"],
      "--instance_type=",
      "--instance_type=tpuv5:4x4x4",
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

  command = main["command"][-1]
  if old_name in command:
    raise ValueError("P39 client command retained the temporary P34 name")
  validate(
      document,
      source_commit=source_commit,
      client_image=client_image,
      stage=stage,
  )
  return document


def validate(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    client_image: str,
    stage: str,
) -> None:
  """Rejects topology, optimizer-placement, or boundedness drift."""
  if stage not in _STAGES:
    raise ValueError("P39 pilot stage is not bounded")
  head = p34._head(document)
  worker = p34._worker(document)
  main = p34._container(head["containers"], "jax-tpu")
  env = p34._env(document)
  if document["spec"]["failurePolicy"]["maxRestarts"] != 0:
    raise ValueError("P39 pilot must remain attempt-zero")
  if (
      worker["backoffLimit"] != 0
      or worker["completions"] != 16
      or worker["parallelism"] != 16
  ):
    raise ValueError("P39 pilot requires sixteen single-attempt workers")
  if main["image"] != client_image or not p34._DIGEST_IMAGE.fullmatch(
      main["image"]
  ):
    raise ValueError("P39 client image is not digest-pinned")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P39_64CHIP_PILOT": "1",
      "CANON_P39_PILOT_ADMITTED": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "MIN_TOKEN_BUCKET": "1024",
      "CANON_LOGPROB_M": "256",
      "CANON_EXPECTED_SLICE_DEVICES": "64",
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P39 rendered environment mismatch: {wrong}")
  required_args = (
      "--rollout_mesh_dp=4",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=4",
      "--train_mesh_tp=8",
      "--rollout_vllm_max_num_seqs=16",
      "--max_num_batched_tokens=256",
      "--max_response_length=4096",
      "--max_turns=5",
      "--optimizer_offload=False",
  )
  if any(value not in env["CANON_RUN_CMD"] for value in required_args):
    raise ValueError("P39 pilot command lost a signed field")
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  p34.ensure_proxy_xla_env(proxy)
  manager = p34._container(services, "pathways-rm")
  if "--instance_type=tpuv5:4x4x4" not in manager["args"]:
    raise ValueError("P39 resource manager topology drifted")
  worker_pod = worker["template"]["spec"]
  if worker_pod["nodeSelector"].get(
      "cloud.google.com/gke-tpu-topology"
  ) != "4x4x4":
    raise ValueError("P39 worker topology drifted")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=p34.DEFAULT_SOURCE_BRANCH)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=tuple(sorted(_STAGES)), required=True)
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
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      model_pvc=args.model_pvc,
      whitelist=args.whitelist,
      whitelist_sha256=args.whitelist_sha256,
  )
  args.output.write_text(p34.dump_jobset(document))
  print(f"P39_PILOT_JOBSET_RENDER_PASS output={args.output}")


if __name__ == "__main__":
  main()
