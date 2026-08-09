# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen DP16xTP4 workload contracts for canonical RL training."""

from __future__ import annotations

import dataclasses
import importlib
import os
from typing import Any, Mapping, Sequence

import jax
from jax.experimental import mesh_utils

from tunix.rl import dp_training


_RUN_STAGE_STEPS = {
    "alignment-short": 1,
    "backward-no-commit": 1,
    "one-update": 1,
    "three-update": 3,
}


@dataclasses.dataclass(frozen=True, slots=True)
class DPWorkloadSpec:
  """Describes one release workload without admitting its execution."""

  name: str
  model_id: str
  model_dir_name: str
  global_prompts: int
  num_generations: int
  local_trajectories: int
  max_prompt_length: int
  max_response_length: int
  max_steps: int
  learning_rate: float
  beta: float
  optimizer_b1: float
  optimizer_b2: float
  weight_decay: float
  temperature: float
  wandb_project: str
  periodic_evaluation: bool = True
  dp_size: int = 16
  tp_size: int = 4
  local_m: int = 256

  @property
  def total_devices(self) -> int:
    return self.dp_size * self.tp_size

  @property
  def global_trajectories(self) -> int:
    return self.global_prompts * self.num_generations

  @property
  def local_prompts(self) -> int:
    return self.global_prompts // self.dp_size

  @property
  def global_m(self) -> int:
    return self.dp_size * self.local_m

  @property
  def gradient_groups(self) -> int:
    return self.local_trajectories

  def training_contract(self) -> dp_training.DPTrainingContract:
    contract = dp_training.DPTrainingContract(
        dp_size=self.dp_size,
        tp_size=self.tp_size,
        global_prompts=self.global_prompts,
        num_generations=self.num_generations,
        local_trajectories=self.local_trajectories,
    )
    contract.validate()
    return contract

  def validate(self) -> None:
    """Rejects a workload that no longer matches the P32 release geometry."""
    self.training_contract()
    if (self.dp_size, self.tp_size, self.total_devices) != (16, 4, 64):
      raise ValueError(
          "canonical workloads require exactly DP16xTP4 on 64 devices"
      )
    if (self.global_prompts, self.num_generations) != (32, 8):
      raise ValueError(
          "canonical workloads require 32 prompts and 8 generations"
      )
    if (self.local_prompts, self.local_trajectories) != (2, 16):
      raise ValueError(
          "canonical workloads require 2 prompts and 16 trajectories per rank"
      )
    if self.local_m != 256 or self.global_m != 4096:
      raise ValueError(
          "canonical workloads require local M256 and global M4096"
      )
    if self.gradient_groups != 16:
      raise ValueError(
          "canonical workloads require 16 rank-major gradient groups"
      )

  def command(self, *, run_stage: str = "full") -> tuple[str, ...]:
    """Returns the frozen recipe command for review and launch wrappers."""
    self.validate()
    if run_stage == "full":
      max_steps = self.max_steps
    else:
      try:
        max_steps = _RUN_STAGE_STEPS[run_stage]
      except KeyError as exc:
        raise ValueError(f"unknown P33 run stage: {run_stage!r}") from exc
    short_alignment = run_stage == "alignment-short"
    if short_alignment and self.name != "frozenlake":
      raise ValueError("alignment-short is only defined for FrozenLake")
    max_response_length = 512 if short_alignment else self.max_response_length
    common = (
        "--mesh_dp=16",
        "--mesh_tp=4",
        "--batch_size=32",
        "--mini_batch_size=32",
        "--train_trajectory_micro_batch_size=16",
        f"--max_steps={max_steps}",
        "--num_generations=8",
        f"--max_prompt_length={self.max_prompt_length}",
        f"--max_response_length={max_response_length}",
        "--max_concurrency=256",
    )
    if self.name == "gsm8k":
      return (
          "python3",
          "-u",
          "examples/math_gsm8k/qwen3_grpo_demo.py",
          *common,
          "--train_micro_batch_size=32",
          "--compute_logps_micro_batch_size=32",
          "--rollout_vllm_hbm_utilization=0.20",
          f"--rollout_vllm_max_num_seqs={self.local_trajectories}",
          f"--rollout_vllm_max_num_batched_tokens={self.local_m}",
          f"--wandb_project={self.wandb_project}",
      )
    if self.name == "frozenlake":
      return (
          "python3",
          "-u",
          "examples/frozenlake/train_frozenlake_qwen3.py",
          *common,
          f"--vllm_max_num_seqs={self.local_trajectories}",
          f"--vllm_max_num_batched_tokens={self.local_m}",
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
    raise ValueError(f"unknown canonical workload: {self.name}")


_WORKLOADS = {
    "gsm8k": DPWorkloadSpec(
        name="gsm8k",
        model_id="Qwen/Qwen3-1.7B",
        model_dir_name="qwen1p7b",
        global_prompts=32,
        num_generations=8,
        local_trajectories=16,
        max_prompt_length=1024,
        max_response_length=1024,
        max_steps=200,
        learning_rate=2.0e-7,
        beta=0.04,
        optimizer_b1=0.9,
        optimizer_b2=0.999,
        weight_decay=0.01,
        temperature=1.0,
        wandb_project="zero-tim-gsm8k-dp16-tp4",
    ),
    "frozenlake": DPWorkloadSpec(
        name="frozenlake",
        model_id="Qwen/Qwen3-8B",
        model_dir_name="qwen8b",
        global_prompts=32,
        num_generations=8,
        local_trajectories=16,
        max_prompt_length=4096,
        max_response_length=2048,
        max_steps=450,
        learning_rate=1.0e-6,
        beta=0.0,
        optimizer_b1=0.9,
        optimizer_b2=0.95,
        weight_decay=0.0,
        temperature=0.7,
        wandb_project="zero-tim-frozenlake-dp16-tp4",
        periodic_evaluation=False,
    ),
}


def get_workload(name: str) -> DPWorkloadSpec:
  """Returns one immutable workload or rejects an unknown name."""
  try:
    workload = _WORKLOADS[name]
  except KeyError as exc:
    raise ValueError(
        f"unknown canonical workload {name!r}; expected {sorted(_WORKLOADS)}"
    ) from exc
  workload.validate()
  return workload


def active_workload(
    environ: Mapping[str, str] | None = None,
) -> DPWorkloadSpec | None:
  """Returns the selected default-off workload, if any."""
  values = os.environ if environ is None else environ
  name = values.get("CANON_P32_WORKLOAD", "")
  return None if not name else get_workload(name)


def requires_alignment_train_mode(
    environ: Mapping[str, str] | None = None,
) -> bool:
  """Returns whether the canonical recipe must use alignment train mode."""
  values = os.environ if environ is None else environ
  return (
      values.get("CANON_P31_CONVERGENCE", "") == "1"
      or active_workload(values) is not None
  )


def configure_replicated_parameter_sharding(
    config: Any, *, data_axis: str = "dp"
) -> None:
  """Uses TP-sharded parameters and DP-sharded activations on a DP/TP mesh."""
  sharding_type = type(config.shd_config)
  factory = getattr(sharding_type, "get_data_parallel_sharding", None)
  if factory is None:
    raise TypeError(
        "model sharding config does not support replicated-parameter data "
        "parallelism"
    )
  config.shd_config = factory(data_axis)


def requested_max_steps(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> int:
  """Returns the fail-closed step budget selected by the P33 run stage."""
  values = os.environ if environ is None else environ
  stage = values.get("CANON_P33_RUN_STAGE", "")
  if stage == "full":
    steps = workload.max_steps
  else:
    try:
      steps = _RUN_STAGE_STEPS[stage]
    except KeyError as exc:
      raise ValueError(f"unknown P33 run stage: {stage!r}") from exc
  no_commit = values.get("CANON_P33_NO_COMMIT", "0")
  expected_no_commit = (
      "1" if stage in ("alignment-short", "backward-no-commit") else "0"
  )
  if no_commit != expected_no_commit:
    raise ValueError(
        "P33 run stage/no-commit mismatch: "
        f"stage={stage!r} expected CANON_P33_NO_COMMIT={expected_no_commit}"
    )
  return steps


def validate_environment(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
    *,
    require_reduction_admission: bool,
) -> None:
  """Validates topology, numerical switches, and reduction promotion."""
  workload.validate()
  values = os.environ if environ is None else environ
  expected = {
      "CANON_P32_WORKLOAD": workload.name,
      "CANON_DP_SIZE": "16",
      "CANON_TP_SIZE": "4",
      "CANON_TOTAL_DEVICES": "64",
      "CANON_GLOBAL_PROMPTS": "32",
      "CANON_LOCAL_PROMPTS": "2",
      "CANON_NUM_GENERATIONS": "8",
      "CANON_LOCAL_TRAJECTORIES": "16",
      "CANON_GLOBAL_TRAJECTORIES": "256",
      "CANON_LOGPROB_M": "256",
      "MIN_TOKEN_BUCKET": "4096",
      "CANON_FIXED_AR": "1",
      "CANON_FIXED_AR_EMBED": "1",
      "CANON_RPA_VJP2": "1",
      "CANON_VJP2_MAX_SEQS": "1",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
      "CANON_PALLAS_LOGSOFTMAX": "1",
      "CANON_P32_DP16_SEGMENTED": "1",
      "CANON_P28_SEGMENTED_FORWARD": "1",
      "CANON_P28_SEGMENTED_TRAIN": "1",
      "CANON_P28_G6_UPDATE": "1",
      "CANON_P29_FULL_TRAIN": "1",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "1",
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
      "CANON_P30_RELEASE_CAPTURED_STATE": "1",
      "CANON_P30_RESHARD_ACCUMULATOR": "1",
  }
  expected["FL_SHARED_MESH"] = (
      "16,4" if require_reduction_admission else "1,4"
  )
  expected["CANON_P33_SHORT_ALIGNMENT"] = (
      "1"
      if values.get("CANON_P33_RUN_STAGE", "") == "alignment-short"
      else "0"
  )
  if workload.name == "frozenlake":
    expected["CANON_P33_DISABLE_EVAL"] = "1"
  if workload.name == "gsm8k":
    expected["CANON_GSM8K_GRAD_PROBE"] = "0"
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  if wrong:
    raise ValueError(f"canonical DP workload environment mismatch: {wrong}")
  expected_training_admission = "1" if require_reduction_admission else "0"
  if values.get("CANON_P32_TRAIN_ADMITTED") != expected_training_admission:
    raise ValueError(
        "canonical DP workload training admission mismatch: "
        f"expected {expected_training_admission!r}, found "
        f"{values.get('CANON_P32_TRAIN_ADMITTED')!r}"
    )
  expected_workload_admission = "1" if require_reduction_admission else "0"
  if (
      values.get("CANON_P33_WORKLOAD_LAUNCH_ADMITTED")
      != expected_workload_admission
  ):
    raise ValueError(
        "canonical workload launch admission mismatch: "
        f"expected {expected_workload_admission!r}, found "
        f"{values.get('CANON_P33_WORKLOAD_LAUNCH_ADMITTED')!r}"
    )
  xla_flags = values.get("XLA_FLAGS", "")
  if "--xla_allow_excess_precision=false" not in xla_flags.split():
    raise ValueError(
        "canonical DP workload requires --xla_allow_excess_precision=false"
    )
  reduction = values.get("CANON_P32_DP_REDUCTION_ADMITTED", "0")
  if reduction not in ("0", "1"):
    raise ValueError("CANON_P32_DP_REDUCTION_ADMITTED must be exactly 0 or 1")
  if require_reduction_admission and reduction != "1":
    raise ValueError(
        "production workload launch is refused until the rank-local DP16 "
        "reduction gate is admitted"
    )
  if not require_reduction_admission and reduction != "0":
    raise ValueError(
        "contract-only validation requires an unadmitted DP16 reduction"
    )
  no_commit = values.get("CANON_P33_NO_COMMIT", "0")
  if no_commit not in ("0", "1"):
    raise ValueError("CANON_P33_NO_COMMIT must be exactly 0 or 1")
  if not require_reduction_admission and no_commit != "0":
    raise ValueError(
        "contract-only validation cannot request backward no-commit"
    )
  if require_reduction_admission:
    requested_max_steps(workload, values)
    wandb_expected = {
        "CANON_WANDB_ONLINE_REQUIRED": "1",
        "CANON_P31_MONOTONIC_METRICS": "1",
        "CANON_WANDB_PROJECT": workload.wandb_project,
        "WANDB_MODE": "online",
    }
    wandb_wrong = {
        key: values.get(key)
        for key, expected_value in wandb_expected.items()
        if values.get(key) != expected_value
    }
    if wandb_wrong:
      raise ValueError(
          "canonical workload requires online W&B telemetry: "
          f"{wandb_wrong}"
      )
    for key in ("CANON_WANDB_GROUP", "CANON_WANDB_RUN_NAME", "WANDB_API_KEY"):
      if not values.get(key):
        raise ValueError(
            f"canonical workload requires non-empty online W&B field {key}"
        )


def _wandb_module() -> Any:
  """Imports W&B only when a production workload checks its live run."""
  return importlib.import_module("wandb")


def require_online_wandb_run(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
  """Fails before training unless the process owns the expected online run."""
  values = os.environ if environ is None else environ
  if jax.process_index() != 0:
    return {"status": "non-primary"}
  if values.get("CANON_WANDB_ONLINE_REQUIRED") != "1":
    raise RuntimeError("online W&B run admission is not enabled")
  if values.get("WANDB_MODE") != "online":
    raise RuntimeError("canonical workload requires WANDB_MODE=online")
  if not values.get("WANDB_API_KEY"):
    raise RuntimeError("canonical workload requires WANDB_API_KEY")

  wandb = _wandb_module()
  run = getattr(wandb, "run", None)
  if run is None:
    raise RuntimeError(
        "W&B backend did not initialize an online run before training"
    )
  actual_project = str(getattr(run, "project", "") or "")
  actual_name = str(getattr(run, "name", "") or "")
  actual_group = str(getattr(run, "group", "") or "")
  expected_project = values.get("CANON_WANDB_PROJECT", "")
  expected_name = values.get("CANON_WANDB_RUN_NAME", "")
  expected_group = values.get("CANON_WANDB_GROUP", "")
  if (
      actual_project != expected_project
      or actual_name != expected_name
      or actual_group != expected_group
  ):
    raise RuntimeError(
        "W&B run identity mismatch: "
        "expected project/group/name="
        f"{expected_project!r}/{expected_group!r}/{expected_name!r}, "
        f"found {actual_project!r}/{actual_group!r}/{actual_name!r}"
    )
  settings = getattr(run, "settings", None)
  if settings is None:
    settings = getattr(run, "_settings", None)
  actual_mode = str(getattr(settings, "mode", "") or "")
  if actual_mode != "online":
    raise RuntimeError(
        f"W&B run is not online: settings.mode={actual_mode!r}"
    )
  return {
      "status": "online",
      "project": actual_project,
      "group": actual_group,
      "name": actual_name,
  }


def create_mesh(
    devices: Sequence[jax.Device], workload: DPWorkloadSpec
) -> jax.sharding.Mesh:
  """Builds the topology-aware full-slice DP16xTP4 training mesh."""
  workload.validate()
  devices = tuple(devices)
  if len(devices) != workload.total_devices:
    raise ValueError(
        "workload mesh requires exactly 64 visible devices: "
        f"got {len(devices)}"
    )
  arranged = mesh_utils.create_device_mesh(
      (workload.dp_size, workload.tp_size),
      devices,
      allow_split_physical_axes=True,
  )
  mesh = jax.sharding.Mesh(arranged, ("dp", "tp"))
  visible_ids = {int(device.id) for device in devices}
  mesh_ids = {int(device.id) for device in mesh.devices.flat}
  if mesh.devices.shape != (16, 4) or mesh_ids != visible_ids:
    raise RuntimeError(
        "topology-aware workload mesh does not cover the visible full slice"
    )
  return mesh
