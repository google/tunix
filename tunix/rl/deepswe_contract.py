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

"""Fail-closed contracts for canonical DeepSWE training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


WEIGHT_ATTESTATION_SCHEMA = "canon.p34.deepswe.weight-attestation.v1"


@dataclasses.dataclass(frozen=True, slots=True)
class DeepSWEWorkload:
  """The signed first-campaign DeepSWE geometry and algorithm."""

  model_id: str = "Qwen/Qwen3-32B"
  global_prompts: int = 8
  generations: int = 8
  max_prompt_length: int = 4096
  max_response_length: int = 32768
  max_turns: int = 50
  max_steps: int = 1000
  temperature: float = 0.7
  per_turn_timeout_secs: int = 300
  episode_timeout_secs: int = 5400
  step_timeout_secs: int = 1800
  reward_timeout_secs: int = 1800
  num_iterations: int = 1
  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: float = 0.28
  off_policy_steps: int = 0
  learning_rate: float = 1e-6
  b1: float = 0.9
  b2: float = 0.99
  weight_decay: float = 0.01
  max_grad_norm: float = 1.0
  loss_agg_mode: str = "sequence-mean-token-scale"
  advantage_estimator: str = "rloo"
  eval_every_n_steps: int = 10
  train_fraction: float = 1.0
  num_epochs: int = 1
  remat_policy: str = "decoder"
  dp_size: int = 16
  tp_size: int = 8
  devices_per_role: int = 128
  local_m: int = 256
  global_m: int = 4096
  max_num_seqs_per_dp: int = 4
  max_num_batched_tokens_per_dp: int = 256

  @property
  def global_trajectories(self) -> int:
    return self.global_prompts * self.generations

  @property
  def local_trajectories(self) -> int:
    return self.global_trajectories // self.dp_size

  def validate(self) -> None:
    """Rejects any silent change to the first DeepSWE campaign."""
    if self.model_id != "Qwen/Qwen3-32B":
      raise ValueError("P34 requires Qwen/Qwen3-32B")
    if (self.global_prompts, self.generations) != (8, 8):
      raise ValueError("P34 requires 8 prompts and 8 generations")
    if self.global_trajectories != 64 or self.local_trajectories != 4:
      raise ValueError("P34 requires 64 global and 4 local trajectories")
    if (self.dp_size, self.tp_size, self.devices_per_role) != (16, 8, 128):
      raise ValueError("P34 primary topology is DP16xTP8 per 128-device role")
    if self.dp_size * self.tp_size != self.devices_per_role:
      raise ValueError("P34 role topology arithmetic changed")
    if (self.local_m, self.global_m) != (256, 4096):
      raise ValueError("P34 requires local M256 and global M4096")
    if self.dp_size * self.local_m != self.global_m:
      raise ValueError("P34 global M must equal dp_size * local M")
    if (self.max_prompt_length, self.max_response_length, self.max_turns) != (
        4096,
        32768,
        50,
    ):
      raise ValueError("P34 signed context/response/turn limits changed")
    if self.max_steps != 1000 or self.temperature != 0.7:
      raise ValueError("P34 signed optimization campaign changed")
    if (
        self.per_turn_timeout_secs,
        self.episode_timeout_secs,
        self.step_timeout_secs,
        self.reward_timeout_secs,
    ) != (300, 5400, 1800, 1800):
      raise ValueError("P34 signed environment timeouts changed")
    if (
        self.num_iterations,
        self.beta,
        self.epsilon,
        self.epsilon_high,
        self.off_policy_steps,
    ) != (1, 0.0, 0.2, 0.28, 0):
      raise ValueError("P34 signed GRPO algorithm changed")
    if (
        self.learning_rate,
        self.b1,
        self.b2,
        self.weight_decay,
        self.max_grad_norm,
    ) != (1e-6, 0.9, 0.99, 0.01, 1.0):
      raise ValueError("P34 signed optimizer algorithm changed")
    if (
        self.loss_agg_mode,
        self.advantage_estimator,
        self.eval_every_n_steps,
    ) != ("sequence-mean-token-scale", "rloo", 10):
      raise ValueError("P34 signed loss or evaluation schedule changed")
    if (
        self.train_fraction,
        self.num_epochs,
        self.remat_policy,
    ) != (1.0, 1, "decoder"):
      raise ValueError("P34 signed data epoch or remat policy changed")
    if (
        self.max_num_seqs_per_dp,
        self.max_num_batched_tokens_per_dp,
    ) != (4, 256):
      raise ValueError("P34 per-DP rollout scheduler capacity changed")
    if self.max_num_seqs_per_dp * self.dp_size != self.global_trajectories:
      raise ValueError("P34 global scheduler request capacity changed")
    if self.max_num_batched_tokens_per_dp * self.dp_size != self.global_m:
      raise ValueError("P34 global scheduler token capacity changed")

  def rank_major_rows(self) -> tuple[tuple[int, ...], ...]:
    """Returns four groups containing one trajectory from every DP rank."""
    self.validate()
    groups = tuple(
        tuple(group * self.dp_size + rank for rank in range(self.dp_size))
        for group in range(self.local_trajectories)
    )
    flat = tuple(row for group in groups for row in group)
    if flat != tuple(range(self.global_trajectories)):
      raise AssertionError("P34 rank-major grouping lost or duplicated rows")
    return groups


P34_WORKLOAD = DeepSWEWorkload()


@dataclasses.dataclass(frozen=True, slots=True)
class DevicePlacement:
  """Portable Pathways placement evidence for one TPU device."""

  device_id: int
  coords: tuple[int, ...]
  process_index: int


def _as_device_placement(device: Any) -> DevicePlacement:
  coords = tuple(int(value) for value in getattr(device, "coords", ()))
  if len(coords) not in (3, 4):
    raise ValueError(f"P34 device lacks 3D topology coordinates: {coords}")
  process_index = getattr(device, "process_index", None)
  if callable(process_index):
    process_index = process_index()
  if process_index is None:
    process_index = getattr(device, "host_id", None)
  if process_index is None:
    raise ValueError("P34 device lacks a process/host index")
  device_id = getattr(device, "id", None)
  if device_id is None:
    device_id = getattr(device, "device_id", None)
  if device_id is None:
    raise ValueError("P34 device lacks an id")
  return DevicePlacement(int(device_id), coords, int(process_index))


def split_4x8x8_role_devices(
    devices: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
  """Splits one 4x8x8 slice into two host-complete 2x8x8 role halves."""
  if len(devices) != 256:
    raise ValueError(f"P34 requires 256 devices, got {len(devices)}")
  placements = tuple(_as_device_placement(device) for device in devices)
  ids = tuple(item.device_id for item in placements)
  coords = tuple(item.coords for item in placements)
  if len(set(ids)) != 256 or len(set(coords)) != 256:
    raise ValueError("P34 devices must have unique ids and coordinates")
  extents = tuple(len({coord[axis] for coord in coords}) for axis in range(3))
  if extents != (4, 8, 8):
    raise ValueError(f"P34 expected a 4x8x8 slice, got extents={extents}")

  by_id = {placement.device_id: device for placement, device in zip(placements, devices)}
  rollout_placements = tuple(item for item in placements if item.coords[0] < 2)
  trainer_placements = tuple(item for item in placements if item.coords[0] >= 2)
  if len(rollout_placements) != 128 or len(trainer_placements) != 128:
    raise ValueError("P34 role halves must each contain 128 devices")
  role_by_process: dict[int, set[str]] = {}
  for name, role in (("rollout", rollout_placements), ("trainer", trainer_placements)):
    for item in role:
      role_by_process.setdefault(item.process_index, set()).add(name)
  split_processes = sorted(
      process for process, names in role_by_process.items() if len(names) != 1
  )
  if split_processes:
    raise ValueError(
        "P34 physical half split crosses host boundaries: "
        f"processes={split_processes[:8]}"
    )

  def order_key(item):
    core = item.coords[3] if len(item.coords) == 4 else 0
    return (item.coords[1], item.coords[2], item.coords[0], core)
  rollout_placements = tuple(sorted(rollout_placements, key=order_key))
  trainer_placements = tuple(sorted(trainer_placements, key=order_key))
  rollout = tuple(by_id[item.device_id] for item in rollout_placements)
  trainer = tuple(by_id[item.device_id] for item in trainer_placements)
  report = {
      "devices": 256,
      "slice_extents": extents,
      "rollout_devices": 128,
      "trainer_devices": 128,
      "rollout_processes": len({item.process_index for item in rollout_placements}),
      "trainer_processes": len({item.process_index for item in trainer_placements}),
      "disjoint": not bool(set(item.device_id for item in rollout_placements) & set(item.device_id for item in trainer_placements)),
      "exhaustive": len(set(ids)) == len(rollout) + len(trainer),
      "host_complete": True,
      "rollout_ids": tuple(item.device_id for item in rollout_placements),
      "trainer_ids": tuple(item.device_id for item in trainer_placements),
  }
  if not report["disjoint"] or not report["exhaustive"]:
    raise AssertionError("P34 role halves are not disjoint and exhaustive")
  return rollout, trainer, report


def validate_environment(values: Mapping[str, str]) -> None:
  """Validates the exact P34 profile without accepting implicit defaults."""
  expected = {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_TOPOLOGY_ADMITTED": "1",
      "CANON_P34_TP8_ADMITTED": "1",
      "CANON_P34_TRAJECTORY_ADMITTED": "1",
      "CANON_P34_UPDATE_ADMITTED": "1",
      "CANON_FIXED_AR": "1",
      "CANON_FIXED_AR_EMBED": "1",
      "CANON_RPA_VJP2": "1",
      # Each segmented model_fn call activates one request per DP shard.  The
      # scheduler may reserve four request slots per shard, but unrolling four
      # VJPs would differentiate inactive capacity rather than the call.
      "CANON_VJP2_MAX_SEQS": "1",
      "CANON_LOGPROB_M": "256",
      "MIN_TOKEN_BUCKET": "4096",
      "ABCPROD": "256",
      "CANON_QWEN3_TP_SIZE": "8",
      "CANON_P34_PREFIX_CACHE": "0",
      "CANON_P34_MAX_NUM_SEQS": "4",
      "CANON_P34_MAX_BATCHED_TOKENS": "256",
      "CANON_P34_STRICT_CLI": "1",
      "CANON_P34_DISABLE_SAMPLER_IS": "1",
      "CANON_P34_DISABLE_TIS": "1",
      "CANON_PRE_ALIGN_GATE": "1",
      "WANDB_MODE": "online",
  }
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  if wrong:
    raise ValueError(f"P34 environment mismatch: {wrong}")
  flags = values.get("XLA_FLAGS", "")
  if "--xla_allow_excess_precision=false" not in flags:
    raise ValueError("P34 XLA_FLAGS lost the excess-precision contract")
  weight_report = values.get("CANON_P34_WEIGHT_REPORT", "")
  if not weight_report or not os.path.isabs(weight_report):
    raise ValueError("P34 weight attestation report path is missing")
  P34_WORKLOAD.validate()
  requested_max_steps(values)


def requested_max_steps(values: Mapping[str, str]) -> int:
  """Returns the fail-closed P34 promotion-stage update budget."""
  stage = values.get("CANON_P34_RUN_STAGE", "")
  no_commit = values.get("CANON_P34_NO_COMMIT", "0")
  if stage == "backward-no-commit":
    expected_no_commit, steps = "1", 1
  elif stage == "one-update":
    expected_no_commit, steps = "0", 1
  elif stage == "three-update":
    expected_no_commit, steps = "0", 3
  elif stage == "full":
    expected_no_commit, steps = "0", P34_WORKLOAD.max_steps
  else:
    raise ValueError(f"unknown P34 run stage: {stage!r}")
  if no_commit != expected_no_commit:
    raise ValueError(
        "P34 stage/no-commit mismatch: "
        f"stage={stage!r} expected CANON_P34_NO_COMMIT={expected_no_commit}"
    )
  return steps


def validate_multiturn_masks(
    completion_valid_mask: Any, action_mask: Any
) -> dict[str, int]:
  """Separates causal-context validity from policy-loss participation."""
  valid = np.asarray(completion_valid_mask, dtype=np.bool_)
  action = np.asarray(action_mask, dtype=np.bool_)
  if valid.shape != action.shape or valid.ndim != 2:
    raise ValueError("P34 completion and action masks must be equal-rank matrices")
  if np.any(action & ~valid):
    raise ValueError("P34 action mask is not a subset of completion validity")
  context_only = valid & ~action
  if not np.any(context_only):
    raise ValueError("P34 multi-turn sample lacks environment/parser context")
  if not np.any(action):
    raise ValueError("P34 multi-turn sample has no policy action tokens")
  return {
      "valid_tokens": int(valid.sum()),
      "action_tokens": int(action.sum()),
      "context_only_tokens": int(context_only.sum()),
  }


def array_sha256(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  return hashlib.sha256(array.view(np.uint8)).hexdigest()


def validate_four_boundaries(
    s_decode: Any,
    s_prefill: Any,
    t_old: Any,
    t_current: Any,
    action_mask: Any,
) -> dict[str, Any]:
  """Requires full-array equality and exact unit ratios on valid actions."""
  values = tuple(np.asarray(value) for value in (s_decode, s_prefill, t_old, t_current))
  if len({(value.shape, value.dtype.str) for value in values}) != 1:
    raise ValueError("P34 four-boundary array shape or dtype changed")
  if not all(np.array_equal(values[0], value) for value in values[1:]):
    raise ValueError("P34 four-boundary arrays are not exact")
  mask = np.asarray(action_mask, dtype=np.bool_)
  if mask.shape != values[0].shape[:-1] and mask.shape != values[0].shape:
    raise ValueError("P34 action mask shape does not match boundary arrays")
  old = values[2] if mask.shape == values[2].shape else values[2][..., 0]
  current = values[3] if mask.shape == values[3].shape else values[3][..., 0]
  ratios = np.exp(current[mask].astype(np.float64) - old[mask].astype(np.float64))
  if ratios.size == 0 or not np.array_equal(ratios, np.ones_like(ratios)):
    raise ValueError("P34 action-token ratio is not exactly one")
  digest = array_sha256(values[0])
  return {
      "action_tokens": int(mask.sum()),
      "boundary_sha256": digest,
      "all_boundaries_exact": True,
      "ratio_exact": True,
      "clip_hits": 0,
      "tis_hits": 0,
  }


def require_weight_sync(trainer_fingerprint: str, rollout_fingerprint: str) -> None:
  """Rejects rollout when its installed policy differs from the trainer."""
  if not trainer_fingerprint or trainer_fingerprint != rollout_fingerprint:
    raise ValueError("P34 rollout/trainer weight fingerprints differ")


def validate_weight_attestation(
    attestation: Mapping[str, Any], *, step: int
) -> dict[str, Any]:
  """Validates an exact trainer-anchor versus live-engine comparison."""
  mapped_leaves = attestation.get("mapped_leaves")
  live_leaves = attestation.get("live_leaves")
  total_elements = attestation.get("total_elements")
  mismatches = tuple(int(value) for value in attestation.get("mismatch_indices", ()))
  mesh_shape = {
      str(name): int(size)
      for name, size in attestation.get("mesh_shape", ())
  }
  mesh_device_ids = tuple(
      int(value) for value in attestation.get("mesh_device_ids", ())
  )
  checks = {
      "step_nonnegative": isinstance(step, int) and step >= 0,
      "equal": attestation.get("equal") is True,
      "leaf_counts": (
          isinstance(mapped_leaves, int)
          and mapped_leaves > 0
          and live_leaves == mapped_leaves
      ),
      "total_elements": isinstance(total_elements, int) and total_elements > 0,
      "mismatch_indices": not mismatches,
      "mesh_shape": mesh_shape == {"dp": 16, "tp": 8},
      "mesh_device_ids": (
          len(mesh_device_ids) == 128
          and len(set(mesh_device_ids)) == len(mesh_device_ids)
      ),
  }
  failed = sorted(name for name, passed in checks.items() if not passed)
  if failed:
    raise ValueError(
        "P34 rollout/trainer exact weight attestation failed: "
        + ", ".join(failed)
    )
  return {
      "schema": WEIGHT_ATTESTATION_SCHEMA,
      "step": step,
      "verdict": "PASS",
      "equal": True,
      "mapped_leaves": mapped_leaves,
      "live_leaves": live_leaves,
      "total_elements": total_elements,
      "mismatch_indices": list(mismatches),
      "mesh_shape": mesh_shape,
      "mesh_device_ids": list(mesh_device_ids),
      "normalized_memory_leaves": int(
          attestation.get("normalized_memory_leaves", 0)
      ),
  }


def persist_weight_attestation(
    attestation: Mapping[str, Any], *, step: int, report_path: str
) -> dict[str, Any]:
  """Persists and prints one admitted cross-role weight record."""
  if not report_path or not os.path.isabs(report_path):
    raise ValueError("P34 weight attestation requires an absolute report path")
  record = validate_weight_attestation(attestation, step=step)
  path = Path(report_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
  with path.open("a", encoding="utf-8") as report_file:
    report_file.write(payload + "\n")
    report_file.flush()
    os.fsync(report_file.fileno())
  print(f"[P34.WEIGHTS_JSON] {payload}", flush=True)
  print(
      "[P34.WEIGHTS] EXACT "
      f"step={step} leaves={record['mapped_leaves']} "
      f"elements={record['total_elements']} devices=128",
      flush=True,
  )
  return record
