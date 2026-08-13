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

"""Frozen 64-device DP/TP workload contracts for canonical RL training."""

from __future__ import annotations

import dataclasses
import importlib
import os
from typing import Any, Mapping, Sequence

import jax
from jax.experimental import mesh_utils

from tunix.rl import dp_training


_RUN_STAGE_STEPS = {
    "envelope-short": 1,
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
    if (self.dp_size, self.tp_size) not in ((16, 4), (8, 8)):
      raise ValueError(
          "canonical workloads require DP16xTP4 or DP8xTP8"
      )
    if self.total_devices != 64:
      raise ValueError("canonical workloads require exactly 64 devices")
    if (self.global_prompts, self.num_generations) != (32, 8):
      raise ValueError(
          "canonical workloads require 32 prompts and 8 generations"
      )
    expected_local_prompts = self.global_prompts // self.dp_size
    expected_local_trajectories = self.global_trajectories // self.dp_size
    if (
        self.local_prompts,
        self.local_trajectories,
    ) != (expected_local_prompts, expected_local_trajectories):
      raise ValueError(
          "canonical workload local geometry changed: expected "
          f"{expected_local_prompts} prompts and "
          f"{expected_local_trajectories} trajectories per rank"
      )
    if self.local_m != 256 or self.global_m != self.dp_size * 256:
      raise ValueError(
          "canonical workloads require local M256 and global M=DP*256"
      )
    if self.gradient_groups != expected_local_trajectories:
      raise ValueError(
          "canonical workload rank-major gradient group count changed"
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
    envelope_short = run_stage == "envelope-short"
    short_alignment = run_stage == "alignment-short"
    if envelope_short and self.name != "gsm8k":
      raise ValueError("envelope-short is only defined for GSM8K")
    max_response_length = (
        256
        if envelope_short
        else 512
        if short_alignment and self.name == "frozenlake"
        else self.max_response_length
    )
    common = (
        f"--mesh_dp={self.dp_size}",
        f"--mesh_tp={self.tp_size}",
        "--batch_size=32",
        "--mini_batch_size=32",
        f"--train_trajectory_micro_batch_size={self.dp_size}",
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
    if self.name.startswith("frozenlake"):
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
    "frozenlake-dp8-tp8": DPWorkloadSpec(
        name="frozenlake-dp8-tp8",
        model_id="Qwen/Qwen3-8B",
        model_dir_name="qwen8b",
        global_prompts=32,
        num_generations=8,
        local_trajectories=32,
        max_prompt_length=4096,
        max_response_length=2048,
        max_steps=450,
        learning_rate=1.0e-6,
        beta=0.0,
        optimizer_b1=0.9,
        optimizer_b2=0.95,
        weight_decay=0.0,
        temperature=0.7,
        wandb_project="zero-tim-frozenlake-dp8-tp8-resident",
        periodic_evaluation=True,
        dp_size=8,
        tp_size=8,
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
      "1"
      if stage in ("envelope-short", "alignment-short", "backward-no-commit")
      else "0"
  )
  if no_commit != expected_no_commit:
    raise ValueError(
        "P33 run stage/no-commit mismatch: "
        f"stage={stage!r} expected CANON_P33_NO_COMMIT={expected_no_commit}"
    )
  return steps


def validate_frozenlake_max_concurrency(
    workload: DPWorkloadSpec,
    max_concurrency: int,
    environ: Mapping[str, str] | None = None,
) -> None:
  """Admits concurrency 32 only for the bounded stock P38 diagnostic.

  Production FrozenLake, evaluation, alternate topologies, and all other
  diagnostics retain the signed concurrency-256 geometry.  The narrow P38
  exception is intentionally tied to the complete capture/no-commit envelope
  so a stray command-line override cannot silently change a training run.
  """
  if max_concurrency == 256:
    return
  values = os.environ if environ is None else environ
  required = {
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
      "CANON_KV_UNIFIED": "0",
  }
  wrong = {
      name: values.get(name)
      for name, expected in required.items()
      if values.get(name) != expected
  }
  for name in (
      "CANON_P38_SERVING_CAPTURE_DIR",
      "CANON_P38_REQUEST_JOURNAL",
  ):
    if not values.get(name):
      wrong[name] = values.get(name)
  if workload.name != "frozenlake":
    wrong["workload"] = workload.name
  if max_concurrency != 32:
    wrong["max_concurrency"] = max_concurrency
  if wrong:
    raise ValueError(
        "FrozenLake max_concurrency must be 256 except for the bounded stock "
        f"P38 serving-capture arm: {wrong}"
    )


def canonical_optimizer_placement(
    environ: Mapping[str, str] | None = None,
    *,
    require_explicit: bool = False,
) -> str:
  """Returns the attested optimizer placement for a canonical workload."""
  values = os.environ if environ is None else environ
  resident = values.get("CANON_OPT_STATE_RESIDENT")
  offload = values.get("CANON_P30_OPT_STATE_OFFLOAD")
  if resident is None and not require_explicit:
    resident = "0"
  if offload is None and not require_explicit:
    offload = "0"
  if resident not in ("0", "1"):
    raise ValueError("CANON_OPT_STATE_RESIDENT must be exactly 0 or 1")
  if offload not in ("0", "1"):
    raise ValueError("CANON_P30_OPT_STATE_OFFLOAD must be exactly 0 or 1")
  if resident == "1" and offload == "1":
    raise ValueError("optimizer resident and offload modes are mutually exclusive")
  if require_explicit and resident == "0" and offload == "0":
    raise ValueError(
        "canonical optimizer placement must explicitly select resident or offload"
    )
  if resident == "1":
    return "device-resident"
  if offload == "1":
    return "pinned-host-offload"
  return "device-unattested"


def frozenlake_evaluation_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    require_full_training: bool = True,
) -> bool:
  """Returns the explicit canonical FrozenLake evaluation selection."""
  values = os.environ if environ is None else environ
  enabled = values.get("CANON_P33_ENABLE_EVAL")
  disabled = values.get("CANON_P33_DISABLE_EVAL")
  learner_enabled = values.get("CANON_P31_ENABLE_EVAL")
  for name, value in (
      ("CANON_P33_ENABLE_EVAL", enabled),
      ("CANON_P33_DISABLE_EVAL", disabled),
      ("CANON_P31_ENABLE_EVAL", learner_enabled),
  ):
    if value not in ("0", "1"):
      raise ValueError(f"{name} must be exactly 0 or 1")
  if (enabled, disabled) not in (("0", "1"), ("1", "0")):
    raise ValueError(
        "canonical FrozenLake requires exactly one evaluation selection"
    )
  if learner_enabled != enabled:
    raise ValueError(
        "CANON_P31_ENABLE_EVAL must match CANON_P33_ENABLE_EVAL"
    )
  if enabled == "1" and require_full_training:
    if (
        not values.get("CANON_P32_WORKLOAD", "").startswith("frozenlake")
        or values.get("CANON_P33_RUN_STAGE") != "full"
        or values.get("CANON_P33_NO_COMMIT") != "0"
    ):
      raise ValueError(
          "canonical FrozenLake evaluation is admitted only for committed "
          "full training"
      )
  return enabled == "1"


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
      "CANON_DP_SIZE": str(workload.dp_size),
      "CANON_TP_SIZE": str(workload.tp_size),
      "CANON_TOTAL_DEVICES": str(workload.total_devices),
      "CANON_ENGINE_DP_SIZE": str(workload.dp_size),
      "CANON_QWEN3_TP_SIZE": str(workload.tp_size),
      "CANON_GLOBAL_PROMPTS": str(workload.global_prompts),
      "CANON_LOCAL_PROMPTS": str(workload.local_prompts),
      "CANON_NUM_GENERATIONS": str(workload.num_generations),
      "CANON_LOCAL_TRAJECTORIES": str(workload.local_trajectories),
      "CANON_GLOBAL_TRAJECTORIES": str(workload.global_trajectories),
      "CANON_LOGPROB_M": str(workload.local_m),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
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
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
      "CANON_P30_RELEASE_CAPTURED_STATE": "1",
      "CANON_P30_RESHARD_ACCUMULATOR": "1",
  }
  optimizer_placement = canonical_optimizer_placement(
      values, require_explicit=True
  )
  expected["CANON_OPT_STATE_RESIDENT"] = (
      "1" if optimizer_placement == "device-resident" else "0"
  )
  expected["CANON_P30_OPT_STATE_OFFLOAD"] = (
      "1" if optimizer_placement == "pinned-host-offload" else "0"
  )
  expected["FL_SHARED_MESH"] = (
      f"{workload.dp_size},{workload.tp_size}"
      if require_reduction_admission
      else f"1,{workload.tp_size}"
  )
  expected["CANON_P33_SHORT_ALIGNMENT"] = (
      "1"
      if values.get("CANON_P33_RUN_STAGE", "") == "alignment-short"
      else "0"
  )
  if workload.name.startswith("frozenlake"):
    evaluation_enabled = frozenlake_evaluation_enabled(values)
    expected["CANON_P33_ENABLE_EVAL"] = "1" if evaluation_enabled else "0"
    expected["CANON_P33_DISABLE_EVAL"] = "0" if evaluation_enabled else "1"
    expected["CANON_P31_ENABLE_EVAL"] = "1" if evaluation_enabled else "0"
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
  """Builds the topology-aware full-slice DP/TP training mesh."""
  workload.validate()
  devices = tuple(devices)
  if len(devices) != workload.total_devices:
    raise ValueError(
        f"workload mesh requires exactly {workload.total_devices} visible devices: "
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
  if mesh.devices.shape != (workload.dp_size, workload.tp_size) or (
      mesh_ids != visible_ids
  ):
    raise RuntimeError(
        "topology-aware workload mesh does not cover the visible full slice"
    )
  return mesh
