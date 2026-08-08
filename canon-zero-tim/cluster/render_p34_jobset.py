#!/usr/bin/env python3
"""Renders one strict P34 DeepSWE JobSet from the reviewed 4x8x8 base."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import shlex
from typing import Any, Iterable, Mapping

import yaml


_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,14}[a-z0-9])?")
_STAGE_STEPS = {
    "backward-no-commit": 1,
    "one-update": 1,
    "three-update": 3,
    "full": 1000,
}


def _head(document: Mapping[str, Any]) -> dict[str, Any]:
  jobs = document["spec"]["replicatedJobs"]
  if [job["name"] for job in jobs] != ["pathways-head", "pathways-worker"]:
    raise ValueError("P34 base JobSet replicated-job layout changed")
  return jobs[0]["template"]["spec"]["template"]["spec"]


def _worker(document: Mapping[str, Any]) -> dict[str, Any]:
  return document["spec"]["replicatedJobs"][1]["template"]["spec"]


def _container(items: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
  matches = [item for item in items if item.get("name") == name]
  if len(matches) != 1:
    raise ValueError(f"expected exactly one container named {name!r}")
  return matches[0]


def _set_env(container: dict[str, Any], values: Mapping[str, str]) -> None:
  env = container.setdefault("env", [])
  by_name = {item["name"]: item for item in env}
  if len(by_name) != len(env):
    raise ValueError("base JobSet contains duplicate jax-tpu environment keys")
  for name, value in values.items():
    item = by_name.get(name)
    if item is None:
      item = {"name": name}
      env.append(item)
      by_name[name] = item
    item.clear()
    item.update({"name": name, "value": value})


def _replace_arg(args: list[str], prefix: str, value: str) -> None:
  matches = [index for index, item in enumerate(args) if item.startswith(prefix)]
  if len(matches) != 1:
    raise ValueError(f"expected one argument with prefix {prefix!r}")
  args[matches[0]] = value


def _command(stage: str, *, run_root: str, whitelist: str) -> tuple[str, ...]:
  steps = _STAGE_STEPS[stage]
  return (
      "python3",
      "-u",
      "examples/deepswe/canonical_entrypoint.py",
      "--model_version=Qwen3-32B",
      "--models_base_dir=/mnt/disks/linchai_data/models",
      "--batch_size=8",
      "--mini_batch_size=8",
      "--train_micro_batch_size=8",
      "--compute_logps_micro_batch_size=8",
      "--rollout_micro_batch_size=1",
      "--num_generations=8",
      "--max_prompt_length=4096",
      "--max_response_length=32768",
      "--max_turns=50",
      f"--max_steps={steps}",
      "--temperature=0.7",
      "--top_k=0",
      "--top_p=1.0",
      "--use_rollout_logps",
      "--rollout_mesh_dp=16",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=16",
      "--train_mesh_tp=8",
      "--rollout_split_fraction=0.5",
      "--rollout_vllm_max_num_seqs=64",
      "--max_num_batched_tokens=8192",
      "--max_concurrency=64",
      "--vllm_utilization=0.6",
      "--optimizer_offload=True",
      f"--gold_whitelist={whitelist}",
      f"--metric_logger_dir={run_root}/metrics",
      f"--ckpt_dir={run_root}/checkpoints",
      "--save_interval_steps=8",
      "--max_to_keep=8",
  )


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
  """Returns a fail-closed P34 JobSet without mutating the base mapping."""
  if not _SHA.fullmatch(source_commit):
    raise ValueError("source_commit must be a lowercase 40-character SHA")
  if not _DIGEST_IMAGE.fullmatch(client_image):
    raise ValueError("client image must be pinned by sha256 digest")
  if not _SHA256.fullmatch(whitelist_sha256):
    raise ValueError("whitelist_sha256 must be a lowercase SHA-256 digest")
  if not _RUN_ID.fullmatch(run_id):
    raise ValueError("run_id must be a 1-16 character lowercase DNS component")
  if stage not in _STAGE_STEPS:
    raise ValueError(f"unknown P34 stage: {stage!r}")
  if not source_branch or any(ch.isspace() for ch in source_branch):
    raise ValueError("source_branch must be a nonempty ref without whitespace")
  for label, value in (
      ("cpu_nodepool", cpu_nodepool),
      ("worker_nodepool", worker_nodepool),
      ("model_pvc", model_pvc),
      ("whitelist", whitelist),
  ):
    if not value:
      raise ValueError(f"{label} must be nonempty")

  document = copy.deepcopy(base)
  name = f"canon-p34-{stage.replace('-update', '').replace('-no-commit', '')}-{run_id}"
  if len(name) > 63:
    raise ValueError("rendered JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/phase": "p34",
      "canon.zero-tim/stage": stage,
      "canon.zero-tim/source": source_commit[:8],
  })
  document["spec"]["failurePolicy"]["maxRestarts"] = 0
  document["spec"]["failurePolicy"]["restartStrategy"] = "Recreate"

  head = _head(document)
  head["nodeSelector"] = {"cloud.google.com/gke-nodepool": cpu_nodepool}
  head_job = document["spec"]["replicatedJobs"][0]["template"]["spec"]
  head_job["backoffLimit"] = 0
  proxy = _container(head["containers"], "pathways-proxy")
  manager = _container(head["containers"], "pathways-rm")
  main = _container(head["containers"], "jax-tpu")
  main["image"] = client_image
  scratch = f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p34/{name}"
  _replace_arg(proxy["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}")
  _replace_arg(manager["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}")
  manager_args = manager["args"]
  _replace_arg(manager_args, "--instance_count=", "--instance_count=1")
  _replace_arg(manager_args, "--instance_type=", "--instance_type=tpuv5:4x8x8")
  main["command"] = ["bash", "-c", """
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
git init -q
git remote set-url origin https://github.com/google/tunix.git 2>/dev/null || git remote add origin https://github.com/google/tunix.git
git fetch -q origin "$CANON_SOURCE_BRANCH"
git reset -q --hard FETCH_HEAD
actual="$(git rev-parse HEAD)"
[ "$actual" = "$CANON_EXPECT_COMMIT" ] || { echo "source commit mismatch: $actual" >&2; exit 1; }
exec bash canon-zero-tim/cluster/entrypoint.sh
""".strip()]

  no_commit = "1" if stage == "backward-no-commit" else "0"
  _set_env(main, {
      "CANON_MODE": "run",
      "CANON_PROFILE_FILE": "cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env",
      "CANON_STATE": run_root,
      "CANON_RUN_ID": run_id,
      "CANON_SOURCE_BRANCH": source_branch,
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P34_TOPOLOGY_ADMITTED": "1",
      "CANON_P34_TP8_ADMITTED": "1",
      "CANON_P34_TRAJECTORY_ADMITTED": "1",
      "CANON_P34_UPDATE_ADMITTED": "1",
      "CANON_P32_TRAIN_ADMITTED": "1",
      "CANON_P32_DP_REDUCTION_ADMITTED": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": no_commit,
      "CANON_P34_WHITELIST": whitelist,
      "CANON_P34_WHITELIST_SHA256": whitelist_sha256,
      "CANON_RUN_CMD": shlex.join(_command(stage, run_root=run_root, whitelist=whitelist)),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-dp16-tp8",
      "CANON_WANDB_GROUP": "qwen3-32b-dp16-tp8",
      "MIN_TOKEN_BUCKET": "4096",
      "CANON_LOGPROB_M": "256",
      "CANON_VJP2_MAX_SEQS": "1",
      "NODE_SELECTOR_VAL": cpu_nodepool,
      "R2E_ACTIVE_DEADLINE_SECONDS": "10800",
      "R2E_POD_START_TIMEOUT_SECONDS": "1200",
  })

  volumes = head.setdefault("volumes", [])
  volumes[:] = [item for item in volumes if item["name"] != "p34-data"]
  volumes.append({
      "name": "p34-data",
      "persistentVolumeClaim": {"claimName": model_pvc},
  })
  mounts = main.setdefault("volumeMounts", [])
  if not any(item.get("name") == "p34-data" for item in mounts):
    mounts.append({"name": "p34-data", "mountPath": "/mnt/disks/linchai_data"})

  worker_job = _worker(document)
  worker_job["backoffLimit"] = 0
  worker_job["completions"] = 64
  worker_job["parallelism"] = 64
  worker_pod = worker_job["template"]["spec"]
  worker_pod["restartPolicy"] = "Never"
  worker_pod["nodeSelector"]["cloud.google.com/gke-nodepool"] = worker_nodepool
  worker_container = _container(worker_pod["containers"], "pathways-worker")
  address = f"{name}-pathways-head-0-0.{name}"
  _replace_arg(
      worker_container["args"],
      "--resource_manager_address=",
      f"--resource_manager_address={address}:29001",
  )
  validate(document, source_commit=source_commit, client_image=client_image, stage=stage)
  return document


def _env(document: Mapping[str, Any]) -> dict[str, str]:
  main = _container(_head(document)["containers"], "jax-tpu")
  return {item["name"]: item["value"] for item in main["env"] if "value" in item}


def validate(document: Mapping[str, Any], *, source_commit: str, client_image: str, stage: str) -> None:
  """Rejects any rendered object that weakens the P34 attempt-zero contract."""
  head = _head(document)
  worker = _worker(document)
  main = _container(head["containers"], "jax-tpu")
  env = _env(document)
  if document["spec"]["failurePolicy"]["maxRestarts"] != 0:
    raise ValueError("P34 JobSet must disable restarts")
  if worker["backoffLimit"] != 0 or worker["completions"] != 64 or worker["parallelism"] != 64:
    raise ValueError("P34 requires 64 single-attempt four-chip worker pods")
  if worker["template"]["spec"]["restartPolicy"] != "Never":
    raise ValueError("P34 workers must not retry in place")
  if main["image"] != client_image or not _DIGEST_IMAGE.fullmatch(main["image"]):
    raise ValueError("P34 client image is not digest-pinned")
  expected = {
      "CANON_MODE": "run",
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_TOPOLOGY_ADMITTED": "1",
      "CANON_P34_TP8_ADMITTED": "1",
      "CANON_P34_TRAJECTORY_ADMITTED": "1",
      "CANON_P34_UPDATE_ADMITTED": "1",
      "MIN_TOKEN_BUCKET": "4096",
      "CANON_LOGPROB_M": "256",
      "CANON_VJP2_MAX_SEQS": "1",
  }
  wrong = {key: env.get(key) for key, value in expected.items() if env.get(key) != value}
  if wrong:
    raise ValueError(f"P34 rendered environment mismatch: {wrong}")
  command = env["CANON_RUN_CMD"]
  required = (
      "--rollout_mesh_dp=16",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=16",
      "--train_mesh_tp=8",
      "--max_num_batched_tokens=8192",
      "--rollout_vllm_max_num_seqs=64",
      "--use_rollout_logps",
  )
  if any(item not in command for item in required):
    raise ValueError("P34 command lost a signed CLI field")
  if "fsdp" in command or "--sampler_is" in command:
    raise ValueError("P34 command introduced FSDP or importance correction")
  if env.get("CANON_P34_WHITELIST") not in command:
    raise ValueError("P34 command does not consume the pinned whitelist path")
  if not _SHA256.fullmatch(env.get("CANON_P34_WHITELIST_SHA256", "")):
    raise ValueError("P34 whitelist digest is missing or malformed")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default="yuxzhang/p34-deepswe-zero-tim")
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=tuple(_STAGE_STEPS), required=True)
  parser.add_argument("--cpu-nodepool", default="deepswe-cpu-pool")
  parser.add_argument("--worker-nodepool", default="mlperf-v5p-256-np-0")
  parser.add_argument("--model-pvc", default="haoyugao-cpu-np-pvc")
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
  args.output.write_text(yaml.safe_dump(document, sort_keys=False))
  print(f"P34_JOBSET_RENDER_PASS output={args.output}")


if __name__ == "__main__":
  main()
