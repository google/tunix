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

"""Frozen DP/TP workload contracts for canonical RL training."""

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
    # Committed six-update horizon for the one-host GSM8K XProf carrier:
    # a three-update run cannot exercise CANON_DP_DISTINCT_SCHEDULE=
    # first-group-warmup (its warmup covers updates 0..2) and cannot tell a
    # recurring dark-time stall from a one-off.  Same admission properties
    # as three-update -- committed, no diagnostic no-commit exemption.
    "six-update": 6,
    "p59-eight-update": 8,
}


# P57.1 deliberately measures the untreated serving stack.  These switches
# are the complete numerical zero-TIM bundle inherited by the P45 carrier.
# Some shims interpret presence as admission, so the contract distinguishes
# truly absent switches from boolean gates that must be the literal string 0.
P57_STOCK_FAST_ABSENT_SWITCHES = (
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_RPA_D",
    "CANON_RPA_P",
    "CANON_RPA_M",
    "CANON_LOGPROB_M",
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
)
P57_STOCK_FAST_ZERO_SWITCHES = (
    "CANON_RPA_VJP2",
    "CANON_VJP2_MAX_SEQS",
    "CANON_PROMPT_PROCESSED_LOGPROBS",
    "CANON_PALLAS_LOGSOFTMAX",
    "CANON_ENGINE_MODULE_C",
    "CANON_KV_UNIFIED",
    "CANON_P32_TRAIN_ADMITTED",
    "CANON_P32_DP_REDUCTION_ADMITTED",
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED",
    "CANON_P32_DP16_SEGMENTED",
    "CANON_FROZENLAKE_L3",
    "CANON_FROZENLAKE_P27",
    "CANON_P28_SEGMENTED_FORWARD",
    "CANON_P28_SEGMENTED_VJP",
    "CANON_P28_SEGMENTED_TRAIN",
    "CANON_P28_G6_UPDATE",
    "CANON_P28_BATCHED_REPORT",
    "CANON_P29_FULL_TRAIN",
    "CANON_ALIGNMENT_GATE",
    "CANON_ALIGNMENT_GATE_ONLY",
    "CANON_ALIGNMENT_UPDATE_CANARY",
    "CANON_ALIGNMENT_TRAIN",
    "CANON_PRE_ALIGN_GATE",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
    "CANON_P38_FIXED_LM_HEAD",
)

# P57.1 stock training keeps only launch, checkpoint, telemetry, and the
# observer that measures the treatment dose.  These switches are the
# numerical/compiled-program part of the canonical bundle and must remain
# literal zero.  Admission and observer switches are checked separately
# because a training arm cannot set them to zero like rollout-only calibration.
P57_STOCK_TRAIN_ZERO_SWITCHES = (
    "CANON_RPA_VJP2",
    "CANON_VJP2_MAX_SEQS",
    "CANON_PALLAS_LOGSOFTMAX",
    "CANON_ENGINE_MODULE_C",
    "CANON_KV_UNIFIED",
    "CANON_P32_DP_ADMISSION",
    "CANON_P32_DP_REDUCTION_ADMITTED",
    "CANON_P32_DP16_SEGMENTED",
    "CANON_FROZENLAKE_L3",
    "CANON_FROZENLAKE_P27",
    "CANON_P28_SEGMENTED_FORWARD",
    "CANON_P28_SEGMENTED_VJP",
    "CANON_P28_SEGMENTED_TRAIN",
    "CANON_P28_G6_UPDATE",
    "CANON_P28_BATCHED_REPORT",
    "CANON_P28_BATCHED_REVERSE",
    "CANON_BATCHED_EVIDENCE",
    "CANON_P29_FULL_TRAIN",
    "CANON_P30_SPARSE_GRAD_ASSEMBLY",
    "CANON_P30_FUSED_PAIR_ACCUMULATION",
    "CANON_P30_REUSE_SEGMENTED_ENGINE",
    "CANON_P30_RELEASE_CAPTURED_STATE",
    "CANON_P30_RESHARD_ACCUMULATOR",
    "CANON_ALIGNMENT_GATE_ONLY",
    "CANON_ALIGNMENT_UPDATE_CANARY",
    "CANON_P38_FIXED_LM_HEAD",
)
P57_STOCK_TRAIN_ONE_SWITCHES = (
    # Observer-only: sampling does not request prompt logprobs, while the
    # post-rollout S_prefill measurement must apply the same temperature/top-k/
    # top-p transform as S_decode.  This value does not select the old-logprob,
    # loss, backward, or optimizer path.
    "CANON_PROMPT_PROCESSED_LOGPROBS",
    "CANON_P32_TRAIN_ADMITTED",
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED",
    "CANON_ALIGNMENT_GATE",
    "CANON_ALIGNMENT_TRAIN",
    "CANON_PRE_ALIGN_GATE",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
    "CANON_OPT_STATE_RESIDENT",
    "CANON_P45_HOST_MEMORY_TELEMETRY",
    "CANON_P45_HOST_GC_INTERVAL",
    "ENABLE_PATHWAYS_PERSISTENCE",
)
P57_STOCK_EVAL_ZERO_SWITCHES = (
    *P57_STOCK_TRAIN_ZERO_SWITCHES,
    "CANON_PROMPT_PROCESSED_LOGPROBS",
    "CANON_P32_TRAIN_ADMITTED",
    "CANON_ALIGNMENT_GATE",
    "CANON_ALIGNMENT_TRAIN",
    "CANON_PRE_ALIGN_GATE",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
)
P57_STOCK_EVAL_ONE_SWITCHES = (
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED",
    "CANON_OPT_STATE_RESIDENT",
    "CANON_P45_HOST_MEMORY_TELEMETRY",
    "CANON_P45_HOST_GC_INTERVAL",
    "ENABLE_PATHWAYS_PERSISTENCE",
)

# Registered stock-runtime treatment/workload tuples.  The M15 selection row
# preserves the discovery campaign; the other four rows are the causal study's
# P45/M15 x no-IS/token-IS matrix.  Keep this closed rather than accepting
# arbitrary values from the environment.
_P57_STOCK_RUNTIME_VARIANTS = {
    ("mismatch", "m15", "selection"): "m15-selection-mismatch",
    ("mismatch", "", ""): "p45-mismatch",
    ("is", "", ""): "p45-is",
    ("mismatch", "m15", "main"): "m15-main-mismatch",
    ("is", "m15", "main"): "m15-main-is",
}
_P57_STOCK_RUNTIME_UPDATES = {
    "m15-selection-mismatch": "200",
    "p45-mismatch": "300",
    "p45-is": "300",
    "m15-main-mismatch": "300",
    "m15-main-is": "300",
}

# P57 materializes M15 with a wider physical response buffer than the original
# P45 carrier.  Keep the admitted pairs closed here: the renderer and training
# entrypoint already sign the same candidate/split tuple, while the P32 adapter
# consumes this table to reject any unregistered token width before tracing.
_P57_DP8_TP8_TOKEN_WIDTHS = {
    ("", ""): (4096, 2048),
    ("m15", "selection"): (4096, 8192),
    ("m15", "main"): (4096, 8192),
}


def _p57_stock_runtime_variant(
    values: Mapping[str, str], *, stage: str
) -> tuple[str, str, str, str]:
  key = (
      values.get("CANON_P57_TIM_ARM", ""),
      values.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
      values.get("CANON_P57_DATA_SPLIT", ""),
  )
  variant = _P57_STOCK_RUNTIME_VARIANTS.get(key)
  if variant is None:
    raise ValueError(
        f"P57 stock-{stage} environment mismatch: "
        f"unregistered_variant={key!r}"
    )
  return (*key, variant)


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
  four_chip_proxy: bool = False
  four_chip_2x2_proxy: bool = False
  unit_data_proxy: bool = False

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
    proxies = (
        self.four_chip_proxy,
        self.four_chip_2x2_proxy,
        self.unit_data_proxy,
    )
    if sum(proxies) > 1:
      raise ValueError("one workload cannot select two one-host proxies")
    if self.four_chip_proxy:
      expected = {
          "name": "gsm8k-p59-dp4-tp1",
          "model_id": "Qwen/Qwen3-1.7B",
          "dp_size": 4,
          "tp_size": 1,
          "global_prompts": 8,
          "num_generations": 8,
          "local_trajectories": 16,
          "local_m": 256,
          "periodic_evaluation": False,
      }
      actual = {name: getattr(self, name) for name in expected}
      wrong = {
          name: actual[name]
          for name, expected_value in expected.items()
          if actual[name] != expected_value
      }
      if wrong:
        raise ValueError(f"P59 four-chip proxy geometry changed: {wrong}")
      if self.total_devices != 4 or self.global_m != 1024:
        raise ValueError(
            "P59 four-chip proxy requires four devices and global M1024"
        )
      return
    if self.four_chip_2x2_proxy:
      # Same four chips and the same global work as the DP4xTP1 carrier
      # (prompts 8, generations 8, trajectories 64), re-cut as data=2 x
      # model=2 so real TP collectives are present in rollout and training.
      expected = {
          "name": "gsm8k-p59-dp2-tp2",
          "model_id": "Qwen/Qwen3-1.7B",
          "dp_size": 2,
          "tp_size": 2,
          "global_prompts": 8,
          "num_generations": 8,
          "local_trajectories": 32,
          "local_m": 256,
          "periodic_evaluation": False,
      }
      actual = {name: getattr(self, name) for name in expected}
      wrong = {
          name: actual[name]
          for name, expected_value in expected.items()
          if actual[name] != expected_value
      }
      if wrong:
        raise ValueError(f"P59 2x2 one-host proxy geometry changed: {wrong}")
      if self.total_devices != 4 or self.global_m != 512:
        raise ValueError(
            "P59 2x2 one-host proxy requires four devices and global M512"
        )
      return
    if self.unit_data_proxy:
      expected = {
          "name": "gsm8k-p66-dp1-tp4",
          "model_id": "Qwen/Qwen3-1.7B",
          "dp_size": 1,
          "tp_size": 4,
          "global_prompts": 2,
          "num_generations": 8,
          "local_trajectories": 16,
          "local_m": 256,
          "periodic_evaluation": False,
      }
      actual = {name: getattr(self, name) for name in expected}
      wrong = {
          name: actual[name]
          for name, expected_value in expected.items()
          if actual[name] != expected_value
      }
      if wrong:
        raise ValueError(f"P66 unit-data TP4 proxy geometry changed: {wrong}")
      if self.total_devices != 4 or self.global_m != 256:
        raise ValueError(
            "P66 unit-data TP4 proxy requires four devices and global M256"
        )
      return
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
    if run_stage == "p59-eight-update" and self.name != "gsm8k-p59-dp4-tp1":
      raise ValueError(
          "p59-eight-update is only defined for gsm8k-p59-dp4-tp1"
      )
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
        f"--batch_size={self.global_prompts}",
        f"--mini_batch_size={self.global_prompts}",
        f"--train_trajectory_micro_batch_size={self.dp_size}",
        f"--max_steps={max_steps}",
        f"--num_generations={self.num_generations}",
        f"--max_prompt_length={self.max_prompt_length}",
        f"--max_response_length={max_response_length}",
        f"--max_concurrency={self.global_trajectories}",
    )
    if self.name.startswith("gsm8k"):
      return (
          "python3",
          "-u",
          "examples/math_gsm8k/qwen3_grpo_demo.py",
          *common,
          f"--train_micro_batch_size={self.global_prompts}",
          f"--compute_logps_micro_batch_size={self.global_prompts}",
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
    "gsm8k-p59-dp4-tp1": DPWorkloadSpec(
        name="gsm8k-p59-dp4-tp1",
        model_id="Qwen/Qwen3-1.7B",
        model_dir_name="qwen1p7b",
        global_prompts=8,
        num_generations=8,
        local_trajectories=16,
        max_prompt_length=1024,
        max_response_length=1024,
        max_steps=3,
        learning_rate=2.0e-7,
        beta=0.04,
        optimizer_b1=0.9,
        optimizer_b2=0.999,
        weight_decay=0.01,
        temperature=1.0,
        wandb_project="zero-tim-gsm8k-p59-dp4-tp1",
        periodic_evaluation=False,
        dp_size=4,
        tp_size=1,
        four_chip_proxy=True,
    ),
    "gsm8k-p59-dp2-tp2": DPWorkloadSpec(
        name="gsm8k-p59-dp2-tp2",
        model_id="Qwen/Qwen3-1.7B",
        model_dir_name="qwen1p7b",
        global_prompts=8,
        num_generations=8,
        local_trajectories=32,
        max_prompt_length=1024,
        max_response_length=1024,
        max_steps=3,
        learning_rate=2.0e-7,
        beta=0.04,
        optimizer_b1=0.9,
        optimizer_b2=0.999,
        weight_decay=0.01,
        temperature=1.0,
        wandb_project="zero-tim-gsm8k-p59-dp2-tp2",
        periodic_evaluation=False,
        dp_size=2,
        tp_size=2,
        four_chip_2x2_proxy=True,
    ),
    "gsm8k-p66-dp1-tp4": DPWorkloadSpec(
        name="gsm8k-p66-dp1-tp4",
        model_id="Qwen/Qwen3-1.7B",
        model_dir_name="qwen1p7b",
        global_prompts=2,
        num_generations=8,
        local_trajectories=16,
        max_prompt_length=1024,
        max_response_length=256,
        max_steps=1,
        learning_rate=2.0e-7,
        beta=0.04,
        optimizer_b1=0.9,
        optimizer_b2=0.999,
        weight_decay=0.01,
        temperature=1.0,
        wandb_project="zero-tim-gsm8k-p66-dp1-tp4",
        periodic_evaluation=False,
        dp_size=1,
        tp_size=4,
        unit_data_proxy=True,
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
      or values.get("CANON_GSM8K_TRAIN", "") == "1"
      or active_workload(values) is not None
  )


def configure_replicated_parameter_sharding(
    config: Any, *, data_axis: str = "dp", tp_axis: str = "tp"
) -> None:
  """Uses TP-sharded parameters and DP-sharded activations on a DP/TP mesh."""
  sharding_type = type(config.shd_config)
  factory = getattr(sharding_type, "get_data_parallel_sharding", None)
  if factory is None:
    raise TypeError(
        "model sharding config does not support replicated-parameter data "
        "parallelism"
    )
  config.shd_config = factory(data_axis=data_axis, tp_axis=tp_axis)


def configure_model_sharding_for_mesh(
    config: Any, mesh_axis_names: Sequence[str]
) -> None:
  """Makes model PartitionSpecs match one registered training mesh exactly."""
  axes = tuple(mesh_axis_names)
  if axes == ("fsdp", "tp"):
    return
  if axes in (("dp", "tp"), ("data", "model")):
    configure_replicated_parameter_sharding(
        config, data_axis=axes[0], tp_axis=axes[1]
    )
    return
  raise ValueError(
      "unsupported training mesh axes for model sharding: "
      f"{axes!r}; expected ('fsdp', 'tp'), ('dp', 'tp'), or "
      "('data', 'model')"
  )


def data_sharding_axis_for_mesh(
    mesh_axis_names: Sequence[str],
) -> tuple[str]:
  """Returns the registered data axis for the actual training mesh."""
  axes = tuple(mesh_axis_names)
  if axes not in (("fsdp", "tp"), ("dp", "tp"), ("data", "model")):
    raise ValueError(
        "unsupported training mesh axes for data sharding: "
        f"{axes!r}"
    )
  return (axes[0],)


def requested_max_steps(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> int:
  """Returns the fail-closed step budget selected by the P33 run stage."""
  values = os.environ if environ is None else environ
  stage = values.get("CANON_P33_RUN_STAGE", "")
  tail8 = values.get("CANON_P59_DP4_TAIL8", "0")
  if tail8 not in ("0", "1"):
    raise ValueError("CANON_P59_DP4_TAIL8 must be exactly 0 or 1")
  if stage == "p59-eight-update":
    if workload.name != "gsm8k-p59-dp4-tp1" or tail8 != "1":
      raise ValueError(
          "p59-eight-update requires gsm8k-p59-dp4-tp1 and "
          "CANON_P59_DP4_TAIL8=1"
      )
  elif tail8 != "0":
    raise ValueError(
        "CANON_P59_DP4_TAIL8=1 requires CANON_P33_RUN_STAGE="
        "p59-eight-update"
    )
  deterministic_ab = values.get("CANON_P60_DETERMINISTIC_AB", "0")
  if deterministic_ab not in ("0", "1"):
    raise ValueError("CANON_P60_DETERMINISTIC_AB must be exactly 0 or 1")
  if deterministic_ab == "1" and workload.name not in (
      "gsm8k-p59-dp4-tp1",
      "gsm8k-p59-dp2-tp2",
      "gsm8k-p60-dp2-tp2",
      "gsm8k-p66-dp1-tp4",
  ):
    raise ValueError(
        "CANON_P60_DETERMINISTIC_AB requires an exact P60 one-host "
        "zero-TIM workload"
    )
  p61_capture_dir = values.get("CANON_P61_BACKWARD_NUMERICAL_DIR", "")
  if p61_capture_dir and (
      not os.path.isabs(p61_capture_dir)
      or workload.name != "gsm8k-p59-dp4-tp1"
      or workload.dp_size != 4
      or workload.tp_size != 1
      or stage != "one-update"
      or tail8 != "0"
      or deterministic_ab != "1"
  ):
    raise ValueError(
        "CANON_P61_BACKWARD_NUMERICAL_DIR requires an absolute path and "
        "exact gsm8k-p59-dp4-tp1 one-update deterministic geometry"
    )
  p62_numeric_debug = values.get(
      "CANON_P62_BACKWARD_NUMERIC_DEBUG", "0"
  )
  if p62_numeric_debug not in ("0", "1"):
    raise ValueError(
        "CANON_P62_BACKWARD_NUMERIC_DEBUG must be exactly 0 or 1"
    )
  if p62_numeric_debug == "1" and (
      workload.name != "gsm8k"
      or (workload.dp_size, workload.tp_size) != (16, 4)
      or stage != "backward-no-commit"
      or values.get("CANON_P33_NO_COMMIT") != "1"
      or values.get("CANON_P59_RANK_PARALLEL_BACKWARD") != "1"
      or values.get("CANON_P38_FIXED_LM_HEAD") != "1"
      or values.get("CANON_V1_HP_FULL", "0") != "0"
  ):
    raise ValueError(
        "CANON_P62_BACKWARD_NUMERIC_DEBUG requires exact GSM8K "
        "DP16xTP4 P59 fixed-head backward-no-commit geometry"
    )
  p64_numeric_debug = values.get("CANON_P64_P45_NUMERIC_DEBUG", "0")
  if p64_numeric_debug not in ("0", "1"):
    raise ValueError(
        "CANON_P64_P45_NUMERIC_DEBUG must be exactly 0 or 1"
    )
  if p62_numeric_debug == "1" and p64_numeric_debug == "1":
    raise ValueError("P62 and P64 numerical observers conflict")
  if p64_numeric_debug == "1" and (
      workload.name != "frozenlake-dp8-tp8"
      or (workload.dp_size, workload.tp_size) != (8, 8)
      or stage != "backward-no-commit"
      or values.get("CANON_P33_NO_COMMIT") != "1"
      or values.get("CANON_P59_RANK_PARALLEL_BACKWARD") != "1"
      or values.get("CANON_P38_FIXED_LM_HEAD") != "1"
      or values.get("CANON_V1_HP_FULL", "0") != "0"
      or values.get("CANON_P64_TRAINING_CAPSULE_MODE")
      not in ("capture", "replay")
  ):
    raise ValueError(
        "CANON_P64_P45_NUMERIC_DEBUG requires exact P45 DP8xTP8 P59 "
        "fixed-head backward-no-commit geometry"
    )
  if stage == "full":
    if values.get("CANON_PROFILE_FILE", "") in (
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env",
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
    ):
      try:
        steps = int(values.get("CANON_P57_EXPECTED_UPDATES", ""))
      except ValueError as exc:
        raise ValueError(
            "P57 expected-update horizon must be a positive integer"
        ) from exc
      if steps <= 0:
        raise ValueError(
            "P57 expected-update horizon must be a positive integer"
        )
    else:
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


def expected_token_widths(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
  """Returns exact prompt/completion widths for the admitted carrier.

  Production workloads retain their frozen widths.  P60 hash-only A/B runs
  use the proven 1024/256 GSM8K envelope so single-request rollout remains
  bounded while the two backward arms see exactly the same trainer shape.
  """
  values = os.environ if environ is None else environ
  deterministic_ab = values.get("CANON_P60_DETERMINISTIC_AB", "0")
  if deterministic_ab not in ("0", "1"):
    raise ValueError("CANON_P60_DETERMINISTIC_AB must be exactly 0 or 1")
  if deterministic_ab == "1":
    if workload.name not in (
        "gsm8k-p59-dp4-tp1",
        "gsm8k-p59-dp2-tp2",
        "gsm8k-p60-dp2-tp2",
        "gsm8k-p66-dp1-tp4",
    ):
      raise ValueError(
          "CANON_P60_DETERMINISTIC_AB requires an exact P60 one-host "
          "zero-TIM workload"
      )
    return (1024, 256)
  p57_key = (
      values.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
      values.get("CANON_P57_DATA_SPLIT", ""),
  )
  if workload.name == "frozenlake-dp8-tp8":
    try:
      return _P57_DP8_TP8_TOKEN_WIDTHS[p57_key]
    except KeyError as exc:
      raise ValueError(
          "frozenlake-dp8-tp8 token widths require an admitted P57 "
          f"candidate/split pair, got {p57_key!r}"
      ) from exc
  if any(p57_key):
    raise ValueError(
        "P57 candidate token widths require frozenlake-dp8-tp8, got "
        f"workload={workload.name!r} candidate_split={p57_key!r}"
    )
  return (workload.max_prompt_length, workload.max_response_length)


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
      "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
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
      "CANON_P38_INCIDENT_LEDGER",
      "CANON_P38_DIAGNOSTIC_ROUND_FILE",
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
    v1_hp_full = values.get("CANON_V1_HP_FULL", "0") == "1"
    expected.update({
        "CANON_DP_COMPARE_MODE": (
            "fingerprint-hybrid" if v1_hp_full else None
        ),
        "CANON_DP_DISTINCT_SCHEDULE": (
            "first-group-warmup" if v1_hp_full else None
        ),
        "CANON_DP_FINITE_FETCH": (
            "batched-commit" if v1_hp_full else None
        ),
        "CANON_P71_SCAN": "fwd" if v1_hp_full else None,
        # P69 remains unadmitted for DP8 and target execution.
        "CANON_DP_COLLECTIVE_REDUCE": None,
    })
  if workload.name.startswith("gsm8k"):
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
    wandb_project = workload.wandb_project
    p57_profile = values.get("CANON_PROFILE_FILE", "") in (
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env",
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
    )
    if (
        p57_profile
        and workload.name == "frozenlake-dp8-tp8"
        and values.get("CANON_P57_RUN_KIND") == "train"
        and values.get("CANON_P57_TIM_ARM") == "zero"
    ):
      wandb_project = "zero-tim-p57-frozenlake-tim"
    wandb_expected = {
        "CANON_WANDB_ONLINE_REQUIRED": "1",
        "CANON_P31_MONOTONIC_METRICS": "1",
        "CANON_WANDB_PROJECT": wandb_project,
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


def validate_p57_stock_fast_environment(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
  """Fails closed unless P57 calibration uses untreated serving numerics."""
  workload.validate()
  values = os.environ if environ is None else environ
  expected = {
      "CANON_P57_RUN_KIND": "calibration",
      "CANON_P57_TIM_ARM": "mismatch",
      "CANON_P57_INFERENCE_REGIME": "stock-fast",
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
      "CANON_TARGET_M": str(workload.local_m),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
  }
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  present = [
      key for key in P57_STOCK_FAST_ABSENT_SWITCHES if key in values
  ]
  nonzero = {
      key: values.get(key)
      for key in P57_STOCK_FAST_ZERO_SWITCHES
      if values.get(key) != "0"
  }
  xla_flags = values.get("XLA_FLAGS", "")
  if "--xla_allow_excess_precision=false" in xla_flags.split():
    wrong["XLA_FLAGS"] = xla_flags
  if present or nonzero or wrong:
    raise ValueError(
        "P57 stock-fast environment mismatch: "
        f"wrong={wrong} present={present} nonzero={nonzero}"
    )
  return {
      "regime": "stock-fast",
      "absent_switches": list(P57_STOCK_FAST_ABSENT_SWITCHES),
      "zero_switches": list(P57_STOCK_FAST_ZERO_SWITCHES),
      "canonical_excess_precision_pin": False,
  }


def validate_p57_stock_train_environment(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
  """Fails closed unless P57 runs the untreated stock training program."""
  workload.validate()
  values = os.environ if environ is None else environ
  arm, candidate, split, variant = _p57_stock_runtime_variant(
      values, stage="train"
  )
  expected = {
      "CANON_P57_RUN_KIND": "train",
      "CANON_P57_INFERENCE_REGIME": "stock-fast",
      "CANON_P57_EXPECTED_UPDATES": _P57_STOCK_RUNTIME_UPDATES[variant],
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
      "CANON_TARGET_M": str(workload.local_m),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
  }
  evaluation_enabled = variant != "m15-selection-mismatch"
  expected.update({
      "CANON_P33_ENABLE_EVAL": "1" if evaluation_enabled else "0",
      "CANON_P33_DISABLE_EVAL": "0" if evaluation_enabled else "1",
      "CANON_P31_ENABLE_EVAL": "1" if evaluation_enabled else "0",
  })
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  present = [
      key for key in P57_STOCK_FAST_ABSENT_SWITCHES if key in values
  ]
  nonzero = {
      key: values.get(key)
      for key in P57_STOCK_TRAIN_ZERO_SWITCHES
      if values.get(key) != "0"
  }
  not_one = {
      key: values.get(key)
      for key in P57_STOCK_TRAIN_ONE_SWITCHES
      if values.get(key) != "1"
  }
  xla_flags = values.get("XLA_FLAGS", "")
  if "--xla_allow_excess_precision=false" in xla_flags.split():
    wrong["XLA_FLAGS"] = xla_flags
  if present or nonzero or not_one or wrong:
    raise ValueError(
        "P57 stock-train environment mismatch: "
        f"wrong={wrong} present={present} nonzero={nonzero} not_one={not_one}"
    )
  return {
      "regime": "stock-fast",
      "arm": arm,
      "workload_candidate": candidate,
      "data_split": split,
      "variant": variant,
      "absent_switches": list(P57_STOCK_FAST_ABSENT_SWITCHES),
      "zero_switches": list(P57_STOCK_TRAIN_ZERO_SWITCHES),
      "one_switches": list(P57_STOCK_TRAIN_ONE_SWITCHES),
      "canonical_excess_precision_pin": False,
  }


def validate_p57_stock_eval_environment(
    workload: DPWorkloadSpec,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
  """Fails closed unless P57 evaluates the untreated stock program."""
  workload.validate()
  values = os.environ if environ is None else environ
  arm, candidate, split, variant = _p57_stock_runtime_variant(
      values, stage="eval"
  )
  expected = {
      "CANON_P57_RUN_KIND": "eval",
      "CANON_P57_INFERENCE_REGIME": "stock-fast",
      "CANON_P57_EXPECTED_UPDATES": _P57_STOCK_RUNTIME_UPDATES[variant],
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
      "CANON_TARGET_M": str(workload.local_m),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
  }
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  present = [
      key for key in P57_STOCK_FAST_ABSENT_SWITCHES if key in values
  ]
  nonzero = {
      key: values.get(key)
      for key in P57_STOCK_EVAL_ZERO_SWITCHES
      if values.get(key) != "0"
  }
  not_one = {
      key: values.get(key)
      for key in P57_STOCK_EVAL_ONE_SWITCHES
      if values.get(key) != "1"
  }
  xla_flags = values.get("XLA_FLAGS", "")
  if "--xla_allow_excess_precision=false" in xla_flags.split():
    wrong["XLA_FLAGS"] = xla_flags
  if present or nonzero or not_one or wrong:
    raise ValueError(
        "P57 stock-eval environment mismatch: "
        f"wrong={wrong} present={present} nonzero={nonzero} not_one={not_one}"
    )
  return {
      "regime": "stock-fast",
      "arm": arm,
      "workload_candidate": candidate,
      "data_split": split,
      "variant": variant,
      "absent_switches": list(P57_STOCK_FAST_ABSENT_SWITCHES),
      "zero_switches": list(P57_STOCK_EVAL_ZERO_SWITCHES),
      "one_switches": list(P57_STOCK_EVAL_ONE_SWITCHES),
      "canonical_excess_precision_pin": False,
  }


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
