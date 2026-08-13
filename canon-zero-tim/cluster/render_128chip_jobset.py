#!/usr/bin/env python3
"""Render strict, independent 128-chip P38 JobSets for 4x4x8 TPU v5p slice."""

from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

import yaml


_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,14}[a-z0-9])?")
_BRANCH = "yuxzhang/canon-zero-tim"
_SCRATCH_ROOT = "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p38_128"


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
  if re.match(r"^[0-9]+[eE][0-9]+$", data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
  return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)


@dataclasses.dataclass(frozen=True, slots=True)
class JobSpec128:
  """Defines one 128-chip queue entry."""

  key: str
  workload: str
  stage: str
  profile: str
  no_commit: bool
  job_prefix: str
  command: tuple[str, ...]

  @property
  def filename(self) -> str:
    return f"jobset-128chip-{self.key}.yaml"


def _common_args_128(*, max_steps: int, prompt: int, response: int) -> tuple[str, ...]:
  return (
      "--mesh_dp=32",
      "--mesh_tp=4",
      "--batch_size=64",
      "--mini_batch_size=64",
      "--train_trajectory_micro_batch_size=16",
      f"--max_steps={max_steps}",
      "--num_generations=8",
      f"--max_prompt_length={prompt}",
      f"--max_response_length={response}",
      "--max_concurrency=512",
  )


def _frozenlake_command_128(
    max_steps: int, *, short_alignment: bool = False
) -> tuple[str, ...]:
  return (
      "python3",
      "-u",
      "examples/frozenlake/train_frozenlake_qwen3.py",
      *_common_args_128(
          max_steps=max_steps,
          prompt=4096,
          response=512 if short_alignment else 2048,
      ),
      "--vllm_max_num_seqs=16",
      "--vllm_max_num_batched_tokens=256",
      f"--env_max_steps={2 if short_alignment else 5}",
      "--num_batches=150",
      "--learning_rate=1e-6",
      "--b1=0.9",
      "--b2=0.95",
      "--weight_decay=0",
      "--beta=0",
      "--epsilon=0.003",
      "--epsilon_high=0.005",
      "--loss_algo=gspo-token",
      "--advantage_estimator=rloo",
      "--temperature=0.7",
      "--top_k=0",
      "--top_p=1.0",
  )


def _gsm8k_command_128(max_steps: int) -> tuple[str, ...]:
  return (
      "python3",
      "-u",
      "examples/math_gsm8k/qwen3_grpo_demo.py",
      *_common_args_128(max_steps=max_steps, prompt=1024, response=1024),
      "--train_micro_batch_size=32",
      "--compute_logps_micro_batch_size=32",
      "--rollout_vllm_hbm_utilization=0.20",
      "--rollout_vllm_max_num_seqs=16",
      "--rollout_vllm_max_num_batched_tokens=256",
      "--wandb_project=zero-tim-gsm8k-dp32-tp4",
  )


_SPECS_128 = (
    JobSpec128(
        key="frozenlake-backward-no-commit-128",
        workload="frozenlake",
        stage="backward-no-commit",
        profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
        no_commit=True,
        job_prefix="canon-p38-fl-bwd-128",
        command=_frozenlake_command_128(1),
    ),
    JobSpec128(
        key="gsm8k-full-128",
        workload="gsm8k",
        stage="full",
        profile="cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env",
        no_commit=False,
        job_prefix="canon-p38-gsm8k-full-128",
        command=_gsm8k_command_128(200),
    ),
)


def _job_name(spec: JobSpec128, source_commit: str, run_id: str) -> str:
  name = f"{spec.job_prefix}-{run_id}-{source_commit[:8]}"
  if len(name) > 63:
    raise ValueError(f"jobset name exceeds 63 characters: {name}")
  return name


def render_128chip_manifest(
    spec: JobSpec128,
    source_commit: str,
    run_id: str,
    base_manifest_path: Path,
) -> dict[str, Any]:
  """Renders a 128-chip (4x4x8) JobSet manifest from the base template."""
  document = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
  name = _job_name(spec, source_commit, run_id)

  document["metadata"]["name"] = name
  document["metadata"].setdefault("labels", {})["jobset.sigs.k8s.io/jobset-name"] = name

  # Scale Worker Replicas to 32 (32 nodes * 4 chips = 128 chips)
  document["spec"]["replicatedJobs"][1]["replicas"] = 32

  # Update nodeSelector and topology to 4x4x8
  worker_template = document["spec"]["replicatedJobs"][1]["template"]["spec"]["template"]["spec"]
  worker_template.setdefault("nodeSelector", {})["cloud.google.com/gke-tpu-topology"] = "4x4x8"

  # Update instance type in pathways-rm and pathways-worker to tpuv5:4x4x8
  head_init = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["initContainers"]
  for c in head_init:
    if c["name"] == "pathways-rm":
      c["args"] = [
          "--server_port=29001",
          f"--gcs_scratch_location={_SCRATCH_ROOT}/{name}",
          "--node_type=resource_manager",
          "--instance_count=1",
          "--instance_type=tpuv5:4x4x8",
      ]
    elif c["name"] == "pathways-proxy":
      c["args"] = [
          "--server_port=29000",
          "--resource_manager_address=localhost:29001",
          f"--gcs_scratch_location={_SCRATCH_ROOT}/{name}",
      ]

  worker_containers = worker_template["containers"]
  address = f"{name}-pathways-head-0-0.{name}"
  for c in worker_containers:
    if c["name"] == "pathways-worker":
      c["args"] = [
          "--server_port=29000",
          f"--resource_manager_address={address}:29001",
          "--node_type=worker",
          "--instance_type=tpuv5:4x4x8",
      ]
      for env_entry in c["env"]:
        if env_entry["name"] == "PATHWAYS_HEAD":
          env_entry["value"] = address

  # Update command in jax-tpu container
  head_containers = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"]
  for c in head_containers:
    if c["name"] == "jax-tpu":
      env_map = {entry["name"]: entry for entry in c["env"] if "name" in entry}
      env_map["CANON_EXPECT_COMMIT"]["value"] = source_commit
      env_map["CANON_P33_RUN_STAGE"]["value"] = spec.stage
      env_map["CANON_P33_NO_COMMIT"]["value"] = "1" if spec.no_commit else "0"
      env_map["CANON_P33_SHARED_MESH"]["value"] = "32,4"
      if "CANON_GSM8K_AB_REPORT_ONLY" in env_map:
        env_map["CANON_GSM8K_AB_REPORT_ONLY"]["value"] = "1" if (spec.workload == "gsm8k" and spec.stage == "full") else "0"
      env_map["CANON_RUN_CMD"]["value"] = shlex.join(spec.command)

  return document


def main() -> None:
  parser = argparse.ArgumentParser(description="Render 128-chip P38 JobSets")
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--base", type=Path, default=Path("canon-zero-tim/cluster/jobset-64chip.yaml"))
  args = parser.parse_args()

  if not _SHA_RE.fullmatch(args.source_commit):
    raise ValueError(f"invalid source commit SHA: {args.source_commit}")
  if not _RUN_ID_RE.fullmatch(args.run_id):
    raise ValueError(f"invalid run id: {args.run_id}")

  args.output_dir.mkdir(parents=True, exist_ok=True)

  for spec in _SPECS_128:
    doc = render_128chip_manifest(spec, args.source_commit, args.run_id, args.base)
    out_file = args.output_dir / spec.filename
    out_file.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"[P38.128] RENDERED key={spec.key} path={out_file}")


if __name__ == "__main__":
  main()
