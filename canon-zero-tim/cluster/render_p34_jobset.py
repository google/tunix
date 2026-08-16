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
DEFAULT_SOURCE_BRANCH = "yuxzhang/canon-zero-tim"
_PRIORITY_CLASS = "very-high"
P34_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
P34_DATASET_REVISION = "2e8108ff942f24fcb5686badfaf7f9a8808566d5"
P34_DATASET_SPLIT = "train"
P34_DATASET_ROWS = 4578
P34_CLEAN_WHITELIST = (
    "canon-zero-tim/clean_data/final_filter_result/"
    "task_report_good_qwen3_128_retry_20260713_090141.jsonl"
)
P34_CLEAN_WHITELIST_SHA256 = (
    "2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7"
)
P34_CLEAN_ROWS = 1851
_STAGE_STEPS = {
    "backward-no-commit": 1,
    "one-update": 1,
    "three-update": 3,
    "full": 1000,
}


class _QuotedString(str):
  """A scalar that must remain a YAML string through Kubernetes parsing."""


class _P34Dumper(yaml.SafeDumper):
  """Safe dumper with an explicit representation for ambiguous strings."""


def _represent_quoted_string(
    dumper: yaml.SafeDumper, value: _QuotedString
) -> yaml.ScalarNode:
  return dumper.represent_scalar(
      "tag:yaml.org,2002:str", str(value), style='"'
  )


_P34Dumper.add_representer(_QuotedString, _represent_quoted_string)


def dump_jobset(document: Mapping[str, Any]) -> str:
  """Serializes a rendered JobSet without ambiguous SHA-like scalars."""
  return yaml.dump(document, Dumper=_P34Dumper, sort_keys=False)


def _head(document: Mapping[str, Any]) -> dict[str, Any]:
  jobs = document["spec"]["replicatedJobs"]
  if [job["name"] for job in jobs] != ["pathways-head", "pathways-worker"]:
    raise ValueError("P34 base JobSet replicated-job layout changed")
  return jobs[0]["template"]["spec"]["template"]["spec"]


def _worker(document: Mapping[str, Any]) -> dict[str, Any]:
  return document["spec"]["replicatedJobs"][1]["template"]["spec"]



PROXY_XLA_ENV = "XLA_FLAGS"
PROXY_XLA_FLAG = "--xla_allow_excess_precision=false"
_PROXY_XLA_PREFIX = "--xla_allow_excess_precision="


def ensure_proxy_xla_env(proxy: dict[str, Any]) -> None:
  """Deliver the excess-precision flag through the proxy environment.

  Pathways compiles on the server side, so a client-container XLA_FLAGS value
  never reaches the TPU compiler, and the pinned proxy rejects the flag as a
  command-line argument (P36 flagon1: unknown command line flag).  The verified
  channel is the proxy container environment (P36 envon1: replicated arm
  0/262144 across widths 2/4/8 at depth 8).  Exactly one XLA_FLAGS entry with
  exactly this value is admitted; a raw argv flag or a conflicting entry is a
  contract violation, not something to repair silently.
  """
  raw = [a for a in proxy.get("args", []) if a.startswith(_PROXY_XLA_PREFIX)]
  if raw:
    raise ValueError(
        "Pathways proxy args carry a raw excess-precision flag; the pinned "
        "proxy rejects it as an unknown command line flag"
    )
  env = proxy.setdefault("env", [])
  matches = [e for e in env if e.get("name") == PROXY_XLA_ENV]
  if not matches:
    env.append({"name": PROXY_XLA_ENV, "value": PROXY_XLA_FLAG})
    return
  if matches != [{"name": PROXY_XLA_ENV, "value": PROXY_XLA_FLAG}]:
    raise ValueError(
        "Pathways proxy has a conflicting or duplicate XLA_FLAGS entry"
    )


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
  command = (
      "python3",
      "-u",
      "examples/deepswe/canonical_entrypoint.py",
      "--model_version=Qwen3-32B",
      "--models_base_dir=/mnt/disks/linchai_data/models",
      "--batch_size=8",
      "--mini_batch_size=8",
      "--train_fraction=1.0",
      "--num_epochs=1",
      "--enable_remat=True",
      "--remat_policy=decoder",
      "--train_micro_batch_size=8",
      "--compute_logps_micro_batch_size=8",
      "--rollout_micro_batch_size=1",
      "--num_generations=8",
      "--num_iterations=1",
      "--beta=0.0",
      "--epsilon=0.2",
      "--epsilon_high=0.28",
      "--off_policy_steps=0",
      "--max_prompt_length=4096",
      "--max_response_length=16384",
      "--max_turns=50",
      "--per_turn_timeout_secs=300",
      "--episode_timeout_secs=4800",
      "--step_timeout_secs=1800",
      "--reward_timeout_secs=1800",
      "--cleanup_timeout_secs=300",
      "--rollout_batch_timeout_secs=5400",
      f"--max_steps={steps}",
      "--temperature=1.0",
      "--top_k=0",
      "--top_p=1.0",
      "--loss_agg_mode=sequence-mean-token-scale",
      "--advantage_estimator=rloo",
      "--use_rollout_logps",
      "--learning_rate=1e-6",
      "--b1=0.9",
      "--b2=0.99",
      "--weight_decay=0.01",
      "--max_grad_norm=1.0",
      "--eval_every_n_steps=10",
      "--rollout_mesh_dp=16",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=16",
      "--train_mesh_tp=8",
      "--rollout_split_fraction=0.5",
      "--rollout_vllm_max_num_seqs=4",
      "--max_num_batched_tokens=256",
      "--max_concurrency=64",
      "--vllm_utilization=0.6",
      "--no-optimizer-offload",
      f"--dataset_name={P34_DATASET_NAME}",
      f"--dataset_revision={P34_DATASET_REVISION}",
      f"--dataset_split={P34_DATASET_SPLIT}",
      f"--expected_source_rows={P34_DATASET_ROWS}",
      f"--gold_whitelist={whitelist}",
      f"--metric_logger_dir={run_root}/metrics",
      f"--ckpt_dir={run_root}/checkpoints",
      "--save_interval_steps=8",
      "--max_to_keep=8",
  )
  if stage == "full":
    command += (f"--expected_filtered_rows={P34_CLEAN_ROWS}",)
  return command


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
  if stage == "full" and (
      whitelist != P34_CLEAN_WHITELIST
      or whitelist_sha256 != P34_CLEAN_WHITELIST_SHA256
  ):
    raise ValueError(
        "P34 full training requires the reviewed 1851-image clean whitelist "
        "path and SHA-256"
    )
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
      # A prefix such as 022893e2 is accepted as a number by some YAML 1.1
      # consumers.  Force an explicit string tag in the serialized manifest.
      "canon.zero-tim/source": _QuotedString(source_commit[:8]),
  })
  document["spec"]["failurePolicy"]["maxRestarts"] = 0
  document["spec"]["failurePolicy"]["restartStrategy"] = "Recreate"

  head = _head(document)
  head["nodeSelector"] = {"cloud.google.com/gke-nodepool": cpu_nodepool}
  head_job = document["spec"]["replicatedJobs"][0]["template"]["spec"]
  head_job["backoffLimit"] = 0
  service_containers = head.get("initContainers", []) + head["containers"]
  proxy = _container(service_containers, "pathways-proxy")
  ensure_proxy_xla_env(proxy)
  manager = _container(service_containers, "pathways-rm")
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
# The branch is allowed to advance after a campaign starts. Resume must still
# check out the immutable SHA recorded in the original manifest, while proving
# that SHA belongs to the fetched branch history.
git cat-file -e "$CANON_EXPECT_COMMIT^{commit}"
git merge-base --is-ancestor "$CANON_EXPECT_COMMIT" FETCH_HEAD || {
  echo "expected source commit is not an ancestor of fetched branch" >&2
  exit 1
}
git reset -q --hard "$CANON_EXPECT_COMMIT"
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
      "CANON_P34_TRAJECTORY_CAPTURE": "1" if stage == "full" else "0",
      "CANON_P34_DEBUG_DIR": f"{run_root}/debug",
      "CANON_P34_WHITELIST": whitelist,
      "CANON_P34_WHITELIST_SHA256": whitelist_sha256,
      "CANON_P34_DATASET_NAME": P34_DATASET_NAME,
      "CANON_P34_DATASET_REVISION": P34_DATASET_REVISION,
      "CANON_P34_DATASET_SPLIT": P34_DATASET_SPLIT,
      "CANON_P34_DATASET_ROWS": str(P34_DATASET_ROWS),
      "CANON_P34_CLEAN_ROWS": (
          str(P34_CLEAN_ROWS) if stage == "full" else "0"
      ),
      # Never inherit `_canonical_engine.env`'s four-device mesh order into a
      # 128-device Pathways rollout role.  Physical topology and role placement
      # remain fail-closed in split_4x8x8_role_devices.
      "CANON_EXPECT_MODEL_MESH_IDS": "",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "5400",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "4800",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "1800",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "1800",
      "CANON_RUN_CMD": shlex.join(_command(stage, run_root=run_root, whitelist=whitelist)),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-dp16-tp8",
      "CANON_WANDB_GROUP": "qwen3-32b-dp16-tp8",
      "MIN_TOKEN_BUCKET": "4096",
      "CANON_LOGPROB_M": "256",
      "CANON_VJP2_MAX_SEQS": "1",
      "NODE_SELECTOR_VAL": cpu_nodepool,
      "R2E_ACTIVE_DEADLINE_SECONDS": "5100",
      "R2E_POD_START_TIMEOUT_SECONDS": "1200",
      "R2E_POD_DELETE_TIMEOUT_SECONDS": "300",
      "R2E_K8S_CPU": "2",
      "R2E_K8S_MEM": "4Gi",
      "R2E_K8S_CPU_LIMIT": "4",
      "R2E_K8S_MEM_LIMIT": "8Gi",
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
  priorities = {
      "pathways-head": head.get("priorityClassName"),
      "pathways-worker": worker["template"]["spec"].get(
          "priorityClassName"
      ),
  }
  if any(value != _PRIORITY_CLASS for value in priorities.values()):
    raise ValueError(
        "P34 Pathways Pod priority class drifted: "
        f"expected {_PRIORITY_CLASS!r}, got {priorities}"
    )
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
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P34_TRAJECTORY_CAPTURE": "1" if stage == "full" else "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
      "CANON_P34_DATASET_NAME": P34_DATASET_NAME,
      "CANON_P34_DATASET_REVISION": P34_DATASET_REVISION,
      "CANON_P34_DATASET_SPLIT": P34_DATASET_SPLIT,
      "CANON_P34_DATASET_ROWS": str(P34_DATASET_ROWS),
      "CANON_P34_CLEAN_ROWS": str(P34_CLEAN_ROWS) if stage == "full" else "0",
      "CANON_EXPECT_MODEL_MESH_IDS": "",
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "5400",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "4800",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "1800",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "1800",
      "R2E_ACTIVE_DEADLINE_SECONDS": "5100",
      "R2E_POD_DELETE_TIMEOUT_SECONDS": "300",
      "R2E_K8S_CPU": "2",
      "R2E_K8S_MEM": "4Gi",
      "R2E_K8S_CPU_LIMIT": "4",
      "R2E_K8S_MEM_LIMIT": "8Gi",
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
      "--max_num_batched_tokens=256",
      "--rollout_vllm_max_num_seqs=4",
      "--use_rollout_logps",
      "--train_fraction=1.0",
      "--num_epochs=1",
      "--enable_remat=True",
      "--remat_policy=decoder",
      "--per_turn_timeout_secs=300",
      "--episode_timeout_secs=4800",
      "--step_timeout_secs=1800",
      "--reward_timeout_secs=1800",
      "--cleanup_timeout_secs=300",
      "--rollout_batch_timeout_secs=5400",
      "--temperature=1.0",
      "--num_iterations=1",
      "--beta=0.0",
      "--epsilon=0.2",
      "--epsilon_high=0.28",
      "--off_policy_steps=0",
      "--loss_agg_mode=sequence-mean-token-scale",
      "--advantage_estimator=rloo",
      "--learning_rate=1e-6",
      "--b1=0.9",
      "--b2=0.99",
      "--weight_decay=0.01",
      "--max_grad_norm=1.0",
      "--no-optimizer-offload",
      f"--dataset_name={P34_DATASET_NAME}",
      f"--dataset_revision={P34_DATASET_REVISION}",
      f"--dataset_split={P34_DATASET_SPLIT}",
      f"--expected_source_rows={P34_DATASET_ROWS}",
  )
  if any(item not in command for item in required):
    raise ValueError("P34 command lost a signed CLI field")
  if "fsdp" in command or "--sampler_is" in command:
    raise ValueError("P34 command introduced FSDP or importance correction")
  if env.get("CANON_P34_WHITELIST") not in command:
    raise ValueError("P34 command does not consume the pinned whitelist path")
  if not env.get("CANON_P34_WEIGHT_REPORT", "").endswith(
      "/weight_attestation.jsonl"
  ):
    raise ValueError("P34 weight attestation report path is missing")
  if not _SHA256.fullmatch(env.get("CANON_P34_WHITELIST_SHA256", "")):
    raise ValueError("P34 whitelist digest is missing or malformed")
  if "--optimizer-offload" in command or "--optimizer_offload" in command:
    raise ValueError("P34 command enabled optimizer host offload")
  if stage == "full":
    if f"--expected_filtered_rows={P34_CLEAN_ROWS}" not in command:
      raise ValueError("P34 full command lost the clean-data row contract")
    if not env.get("CANON_P34_DEBUG_DIR", "").endswith("/debug"):
      raise ValueError("P34 full trajectory artifact directory is missing")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=DEFAULT_SOURCE_BRANCH)
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
  args.output.write_text(dump_jobset(document))
  print(f"P34_JOBSET_RENDER_PASS output={args.output}")


if __name__ == "__main__":
  main()
