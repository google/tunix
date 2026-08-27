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

import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


WEIGHT_ATTESTATION_SCHEMA = "canon.p34.deepswe.weight-attestation.v1"
P44_TOPOLOGY_FIELDS = frozenset({
    "contract_name",
    "dp_size",
    "devices_per_role",
    "global_m",
    "max_num_seqs_per_dp",
})


@dataclasses.dataclass(frozen=True, slots=True)
class DeepSWEWorkload:
  """The signed first-campaign DeepSWE geometry and algorithm."""

  contract_name: str = "p34-production"
  model_id: str = "Qwen/Qwen3-32B"
  global_prompts: int = 8
  generations: int = 8
  max_prompt_length: int = 4096
  max_response_length: int = 16384
  max_turns: int = 50
  max_steps: int = 1000
  temperature: float = 1.0
  per_turn_timeout_secs: int = 300
  episode_timeout_secs: int = 4800
  step_timeout_secs: int = 1800
  reward_timeout_secs: int = 1800
  cleanup_timeout_secs: int = 300
  rollout_batch_timeout_secs: int = 5400
  sandbox_active_deadline_secs: int = 5100
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
    expected_topology = {
        "p34-production": (
            "Qwen/Qwen3-32B", 8, 8, 16, 8, 128, 4, 4096, 4, 16384, 50,
            1000,
        ),
        "p39-64chip-pilot": (
            "Qwen/Qwen3-32B", 8, 8, 4, 8, 32, 16, 1024, 16, 4096, 5, 3,
        ),
        "p43-64chip-debug": (
            "Qwen/Qwen3-8B", 4, 4, 4, 8, 32, 4, 1024, 4, 4096, 5, 3,
        ),
        "p44-qwen4b-parity-64": (
            "Qwen/Qwen3-4B-Instruct-2507", 4, 4, 4, 8, 32, 4, 1024, 4,
            16384, 50, 3,
        ),
        "p44-qwen4b-parity-128": (
            "Qwen/Qwen3-4B-Instruct-2507", 4, 4, 8, 8, 64, 2, 2048, 2,
            16384, 50, 3,
        ),
        "p46-qwen32b-train-64": (
            "Qwen/Qwen3-32B", 8, 8, 4, 8, 32, 16, 1024, 16, 16384, 50,
            1000,
        ),
        "p46-qwen32b-train-256": (
            "Qwen/Qwen3-32B", 8, 8, 16, 8, 128, 4, 4096, 4, 16384, 50,
            1000,
        ),
        "p58-qwen4b-tim-128": (
            "Qwen/Qwen3-4B-Instruct-2507", 8, 16, 8, 8, 64, 16, 2048, 16,
            16384, 50, 1000,
        ),
    }
    try:
      (
          expected_model,
          expected_prompts,
          expected_generations,
          expected_dp,
          expected_tp,
          expected_devices,
          expected_local_trajectories,
          expected_global_m,
          expected_max_num_seqs,
          expected_response,
          expected_turns,
          expected_steps,
      ) = expected_topology[self.contract_name]
    except KeyError as exc:
      raise ValueError(
          f"unknown DeepSWE contract {self.contract_name!r}"
      ) from exc
    if self.model_id != expected_model:
      raise ValueError(
          f"{self.contract_name} requires {expected_model}"
      )
    if (self.global_prompts, self.generations) != (
        expected_prompts,
        expected_generations,
    ):
      raise ValueError(
          f"{self.contract_name} requires {expected_prompts} prompts and "
          f"{expected_generations} generations"
      )
    expected_global_trajectories = expected_prompts * expected_generations
    if (
        self.global_trajectories != expected_global_trajectories
        or self.local_trajectories != expected_local_trajectories
    ):
      raise ValueError(
          f"{self.contract_name} requires {expected_global_trajectories} "
          "global and "
          f"{expected_local_trajectories} local trajectories"
      )
    if (self.dp_size, self.tp_size, self.devices_per_role) != (
        expected_dp,
        expected_tp,
        expected_devices,
    ):
      raise ValueError(
          f"{self.contract_name} requires DP{expected_dp}xTP{expected_tp} "
          f"per {expected_devices}-device role"
      )
    if self.dp_size * self.tp_size != self.devices_per_role:
      raise ValueError("P34 role topology arithmetic changed")
    if (self.local_m, self.global_m) != (256, expected_global_m):
      raise ValueError(
          f"{self.contract_name} requires local M256 and global "
          f"M{expected_global_m}"
      )
    if self.dp_size * self.local_m != self.global_m:
      raise ValueError("P34 global M must equal dp_size * local M")
    if (self.max_prompt_length, self.max_response_length, self.max_turns) != (
        4096,
        expected_response,
        expected_turns,
    ):
      raise ValueError(
          f"{self.contract_name} signed context/response/turn limits changed"
      )
    if self.max_steps != expected_steps or self.temperature != 1.0:
      raise ValueError(
          f"{self.contract_name} signed optimization campaign changed"
      )
    parity = self.contract_name.startswith("p44-qwen4b-parity-")
    bounded_q4 = parity or self.contract_name == "p58-qwen4b-tim-128"
    expected_timeouts = (
        (300, 3000, 600, 600, 300, 3600, 3300)
        if bounded_q4
        else (300, 4800, 1800, 1800, 300, 5400, 5100)
    )
    if (
        self.per_turn_timeout_secs,
        self.episode_timeout_secs,
        self.step_timeout_secs,
        self.reward_timeout_secs,
        self.cleanup_timeout_secs,
        self.rollout_batch_timeout_secs,
        self.sandbox_active_deadline_secs,
    ) != expected_timeouts:
      raise ValueError("P34 signed environment timeouts changed")
    if (
        self.episode_timeout_secs + self.cleanup_timeout_secs
        != self.sandbox_active_deadline_secs
        or self.sandbox_active_deadline_secs
        >= self.rollout_batch_timeout_secs
    ):
      raise ValueError(
          "DeepSWE timeout nesting must reserve a positive batch-abort margin"
      )
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
    ) != (expected_max_num_seqs, 256):
      raise ValueError(
          f"{self.contract_name} per-DP rollout scheduler capacity changed"
      )
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
P39_PILOT_WORKLOAD = DeepSWEWorkload(
    contract_name="p39-64chip-pilot",
    max_response_length=4096,
    max_turns=5,
    max_steps=3,
    dp_size=4,
    devices_per_role=32,
    global_m=1024,
    max_num_seqs_per_dp=16,
)
P43_DEBUG_WORKLOAD = DeepSWEWorkload(
    contract_name="p43-64chip-debug",
    model_id="Qwen/Qwen3-8B",
    global_prompts=4,
    generations=4,
    max_response_length=4096,
    max_turns=5,
    max_steps=3,
    dp_size=4,
    devices_per_role=32,
    global_m=1024,
    max_num_seqs_per_dp=4,
)
P44_PARITY_64_WORKLOAD = DeepSWEWorkload(
    contract_name="p44-qwen4b-parity-64",
    model_id="Qwen/Qwen3-4B-Instruct-2507",
    global_prompts=4,
    generations=4,
    max_response_length=16384,
    max_turns=50,
    max_steps=3,
    episode_timeout_secs=3000,
    step_timeout_secs=600,
    reward_timeout_secs=600,
    rollout_batch_timeout_secs=3600,
    sandbox_active_deadline_secs=3300,
    dp_size=4,
    devices_per_role=32,
    global_m=1024,
    max_num_seqs_per_dp=4,
)
P44_PARITY_128_WORKLOAD = dataclasses.replace(
    P44_PARITY_64_WORKLOAD,
    contract_name="p44-qwen4b-parity-128",
    dp_size=8,
    devices_per_role=64,
    global_m=2048,
    max_num_seqs_per_dp=2,
)
P46_Q32_64_WORKLOAD = dataclasses.replace(
    P34_WORKLOAD,
    contract_name="p46-qwen32b-train-64",
    dp_size=4,
    devices_per_role=32,
    global_m=1024,
    max_num_seqs_per_dp=16,
)
P46_Q32_256_WORKLOAD = dataclasses.replace(
    P34_WORKLOAD,
    contract_name="p46-qwen32b-train-256",
)
P58_Q4_TIM_128_WORKLOAD = dataclasses.replace(
    P44_PARITY_128_WORKLOAD,
    contract_name="p58-qwen4b-tim-128",
    global_prompts=8,
    generations=16,
    max_steps=1000,
    max_num_seqs_per_dp=16,
)


def p44_workload(topology: str) -> DeepSWEWorkload:
  """Returns one of the two exact Qwen3-4B parity topologies."""
  if topology == "64":
    return P44_PARITY_64_WORKLOAD
  if topology == "128":
    return P44_PARITY_128_WORKLOAD
  raise ValueError("CANON_P44_TOPOLOGY must be exactly 64 or 128")


def p44_recipe_signature(workload: DeepSWEWorkload) -> dict[str, Any]:
  """Normalizes a P44 workload by the preregistered topology allowlist."""
  if workload.contract_name not in (
      "p44-qwen4b-parity-64",
      "p44-qwen4b-parity-128",
  ):
    raise ValueError("P44 recipe signature requires a P44 workload")
  workload.validate()
  return {
      key: value
      for key, value in dataclasses.asdict(workload).items()
      if key not in P44_TOPOLOGY_FIELDS
  }


def p46_q32_workload(topology: str) -> DeepSWEWorkload:
  """Returns the exact 64/256 Qwen3-32B full-training topology."""
  if topology == "64":
    return P46_Q32_64_WORKLOAD
  if topology == "256":
    return P46_Q32_256_WORKLOAD
  raise ValueError("CANON_P46_TOPOLOGY must be exactly 64 or 256")


def active_workload(
    values: Mapping[str, str] | None = None,
) -> DeepSWEWorkload:
  """Returns the exact production, pilot, or debug DeepSWE contract."""
  environ = os.environ if values is None else values
  pilot_raw = environ.get("CANON_P39_64CHIP_PILOT", "0")
  debug_raw = environ.get("CANON_P43_DEEPSWE_DEBUG", "0")
  parity_raw = environ.get("CANON_P44_DEEPSWE_PARITY", "0")
  p46_raw = environ.get("CANON_P46_DEEPSWE_TRAIN", "0")
  p58_raw = environ.get("CANON_P58_DEEPSWE_TIM", "0")
  if pilot_raw not in ("0", "1"):
    raise ValueError("CANON_P39_64CHIP_PILOT must be exactly 0 or 1")
  if debug_raw not in ("0", "1"):
    raise ValueError("CANON_P43_DEEPSWE_DEBUG must be exactly 0 or 1")
  if parity_raw not in ("0", "1"):
    raise ValueError("CANON_P44_DEEPSWE_PARITY must be exactly 0 or 1")
  if p46_raw not in ("0", "1"):
    raise ValueError("CANON_P46_DEEPSWE_TRAIN must be exactly 0 or 1")
  if p58_raw not in ("0", "1"):
    raise ValueError("CANON_P58_DEEPSWE_TIM must be exactly 0 or 1")
  selected = sum(
      raw == "1"
      for raw in (pilot_raw, debug_raw, parity_raw, p46_raw, p58_raw)
  )
  if selected > 1:
    raise ValueError(
        "P39, P43, P44, P46, and P58 DeepSWE modes are mutually exclusive"
    )
  if p58_raw == "1":
    workload = P58_Q4_TIM_128_WORKLOAD
  elif p46_raw == "1":
    workload = p46_q32_workload(environ.get("CANON_P46_TOPOLOGY", ""))
  elif parity_raw == "1":
    workload = p44_workload(environ.get("CANON_P44_TOPOLOGY", ""))
  elif debug_raw == "1":
    workload = P43_DEBUG_WORKLOAD
  elif pilot_raw == "1":
    workload = P39_PILOT_WORKLOAD
  else:
    workload = P34_WORKLOAD
  workload.validate()
  return workload


@dataclasses.dataclass(frozen=True, slots=True)
class DevicePlacement:
  """Portable Pathways placement evidence for one TPU device."""

  device_id: int
  coords: tuple[int, ...]
  host_key: tuple[Any, ...]
  host_source: str


def _device_repr_attr(device: Any, attr_name: str) -> Any:
  """Parses a Pathways-only device attribute without importing JAX/Tunix."""
  match = re.search(
      rf"(?:^|[,(]){re.escape(attr_name)}=(\[[^\]]*\]|[^,)]+)",
      repr(device),
  )
  if match is None:
    return None
  raw_value = match.group(1).strip()
  try:
    return ast.literal_eval(raw_value)
  except (SyntaxError, ValueError):
    return raw_value


def _runtime_device_host_key(
    device: Any,
) -> tuple[tuple[Any, ...] | None, str | None]:
  """Returns Pathways logical-task identity or direct-JAX process identity."""
  logical_task = _device_repr_attr(device, "logical_task")
  if logical_task is not None:
    task_id = logical_task
    source = "logical_task"
  else:
    task_id = getattr(device, "process_index", None)
    if callable(task_id):
      task_id = task_id()
    if task_id is None:
      task_id = getattr(device, "host_id", None)
    source = "process_index" if task_id is not None else None
  if task_id is None:
    return None, None
  slice_id = getattr(device, "slice_index", None)
  if callable(slice_id):
    slice_id = slice_id()
  return (slice_id, task_id), source


def _as_device_placement(device: Any) -> DevicePlacement:
  coords = tuple(int(value) for value in getattr(device, "coords", ()))
  if len(coords) not in (3, 4):
    raise ValueError(f"P34 device lacks 3D topology coordinates: {coords}")
  host_key, host_source = _runtime_device_host_key(device)
  if host_key is None or host_source is None:
    raise ValueError("P34 device lacks a trustworthy runtime host identity")
  device_id = getattr(device, "id", None)
  if device_id is None:
    device_id = getattr(device, "device_id", None)
  if device_id is None:
    raise ValueError("P34 device lacks an id")
  return DevicePlacement(int(device_id), coords, host_key, host_source)


def _validate_host_complete_roles(
    rollout_placements: Sequence[DevicePlacement],
    trainer_placements: Sequence[DevicePlacement],
    *,
    expected_hosts: int,
    expected_role_hosts: int,
    contract_name: str,
) -> dict[str, Any]:
  """Validates exact four-device hosts and a host-complete role partition."""
  role_by_host: dict[tuple[Any, ...], set[str]] = {}
  all_placements = tuple(rollout_placements) + tuple(trainer_placements)
  for name, role in (
      ("rollout", rollout_placements),
      ("trainer", trainer_placements),
  ):
    for item in role:
      role_by_host.setdefault(item.host_key, set()).add(name)
  split_hosts = sorted(
      (host for host, names in role_by_host.items() if len(names) != 1),
      key=str,
  )
  if split_hosts:
    raise ValueError(
        f"{contract_name} physical half split crosses host boundaries: "
        f"hosts={split_hosts[:8]}"
    )

  host_device_counts = {
      host: sum(item.host_key == host for item in all_placements)
      for host in role_by_host
  }
  invalid_host_sizes = sorted(
      (
          (host, count)
          for host, count in host_device_counts.items()
          if count != 4
      ),
      key=lambda item: str(item[0]),
  )
  if len(host_device_counts) != expected_hosts or invalid_host_sizes:
    raise ValueError(
        f"{contract_name} host inventory mismatch: "
        f"hosts={len(host_device_counts)} expected={expected_hosts} "
        f"invalid_sizes={invalid_host_sizes[:8]}"
    )

  host_sources = {item.host_source for item in all_placements}
  if len(host_sources) != 1:
    raise ValueError(
        f"{contract_name} mixed runtime host identity sources: "
        f"{sorted(host_sources, key=str)}"
    )

  rollout_hosts = {item.host_key for item in rollout_placements}
  trainer_hosts = {item.host_key for item in trainer_placements}
  if (
      len(rollout_hosts) != expected_role_hosts
      or len(trainer_hosts) != expected_role_hosts
  ):
    raise ValueError(
        f"{contract_name} role host inventory mismatch: "
        f"rollout={len(rollout_hosts)} trainer={len(trainer_hosts)} "
        f"expected={expected_role_hosts}"
    )
  return {
      "host_source": next(iter(host_sources)),
      "hosts": len(host_device_counts),
      "devices_per_host": 4,
      "rollout_hosts": len(rollout_hosts),
      "trainer_hosts": len(trainer_hosts),
  }


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
  host_report = _validate_host_complete_roles(
      rollout_placements,
      trainer_placements,
      expected_hosts=64,
      expected_role_hosts=32,
      contract_name="P34",
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
      "rollout_processes": host_report["rollout_hosts"],
      "trainer_processes": host_report["trainer_hosts"],
      "disjoint": not bool(set(item.device_id for item in rollout_placements) & set(item.device_id for item in trainer_placements)),
      "exhaustive": len(set(ids)) == len(rollout) + len(trainer),
      "host_complete": True,
      "rollout_ids": tuple(item.device_id for item in rollout_placements),
      "trainer_ids": tuple(item.device_id for item in trainer_placements),
      **host_report,
  }
  if not report["disjoint"] or not report["exhaustive"]:
    raise AssertionError("P34 role halves are not disjoint and exhaustive")
  return rollout, trainer, report


def split_4x4x4_role_devices(
    devices: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
  """Splits one 4x4x4 slice into two host-complete 2x4x4 role halves."""
  if len(devices) != 64:
    raise ValueError(f"P39 pilot requires 64 devices, got {len(devices)}")
  placements = tuple(_as_device_placement(device) for device in devices)
  ids = tuple(item.device_id for item in placements)
  coords = tuple(item.coords for item in placements)
  if len(set(ids)) != 64 or len(set(coords)) != 64:
    raise ValueError("P39 pilot devices must have unique ids and coordinates")
  extents = tuple(len({coord[axis] for coord in coords}) for axis in range(3))
  if extents != (4, 4, 4):
    raise ValueError(
        f"P39 pilot expected a 4x4x4 slice, got extents={extents}"
    )
  by_id = {
      placement.device_id: device
      for placement, device in zip(placements, devices)
  }
  rollout_placements = tuple(item for item in placements if item.coords[0] < 2)
  trainer_placements = tuple(item for item in placements if item.coords[0] >= 2)
  if len(rollout_placements) != 32 or len(trainer_placements) != 32:
    raise ValueError("P39 pilot role halves must each contain 32 devices")
  host_report = _validate_host_complete_roles(
      rollout_placements,
      trainer_placements,
      expected_hosts=16,
      expected_role_hosts=8,
      contract_name="P39 pilot",
  )

  def order_key(item):
    core = item.coords[3] if len(item.coords) == 4 else 0
    return (item.coords[1], item.coords[2], item.coords[0], core)

  rollout_placements = tuple(sorted(rollout_placements, key=order_key))
  trainer_placements = tuple(sorted(trainer_placements, key=order_key))
  rollout = tuple(by_id[item.device_id] for item in rollout_placements)
  trainer = tuple(by_id[item.device_id] for item in trainer_placements)
  report = {
      "devices": 64,
      "slice_extents": extents,
      "rollout_devices": 32,
      "trainer_devices": 32,
      "rollout_processes": host_report["rollout_hosts"],
      "trainer_processes": host_report["trainer_hosts"],
      "disjoint": not bool(
          set(item.device_id for item in rollout_placements)
          & set(item.device_id for item in trainer_placements)
      ),
      "exhaustive": len(set(ids)) == len(rollout) + len(trainer),
      "host_complete": True,
      "rollout_ids": tuple(item.device_id for item in rollout_placements),
      "trainer_ids": tuple(item.device_id for item in trainer_placements),
      **host_report,
  }
  if not report["disjoint"] or not report["exhaustive"]:
    raise AssertionError("P39 pilot role halves are not disjoint and exhaustive")
  return rollout, trainer, report


def split_4x4x8_role_devices(
    devices: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
  """Splits one 4x4x8 slice into two host-complete 2x4x8 role halves."""
  if len(devices) != 128:
    raise ValueError(
        f"P44 128-chip parity requires 128 devices, got {len(devices)}"
    )
  placements = tuple(_as_device_placement(device) for device in devices)
  ids = tuple(item.device_id for item in placements)
  coords = tuple(item.coords for item in placements)
  if len(set(ids)) != 128 or len(set(coords)) != 128:
    raise ValueError(
        "P44 128-chip devices must have unique ids and coordinates"
    )
  extents = tuple(len({coord[axis] for coord in coords}) for axis in range(3))
  if extents != (4, 4, 8):
    raise ValueError(
        f"P44 128-chip parity expected a 4x4x8 slice, got extents={extents}"
    )
  by_id = {
      placement.device_id: device
      for placement, device in zip(placements, devices)
  }
  rollout_placements = tuple(item for item in placements if item.coords[0] < 2)
  trainer_placements = tuple(item for item in placements if item.coords[0] >= 2)
  if len(rollout_placements) != 64 or len(trainer_placements) != 64:
    raise ValueError("P44 128-chip role halves must each contain 64 devices")
  host_report = _validate_host_complete_roles(
      rollout_placements,
      trainer_placements,
      expected_hosts=32,
      expected_role_hosts=16,
      contract_name="P44 128-chip parity",
  )

  def order_key(item):
    core = item.coords[3] if len(item.coords) == 4 else 0
    return (item.coords[1], item.coords[2], item.coords[0], core)

  rollout_placements = tuple(sorted(rollout_placements, key=order_key))
  trainer_placements = tuple(sorted(trainer_placements, key=order_key))
  rollout = tuple(by_id[item.device_id] for item in rollout_placements)
  trainer = tuple(by_id[item.device_id] for item in trainer_placements)
  report = {
      "devices": 128,
      "slice_extents": extents,
      "rollout_devices": 64,
      "trainer_devices": 64,
      "rollout_processes": host_report["rollout_hosts"],
      "trainer_processes": host_report["trainer_hosts"],
      "disjoint": not bool(
          set(item.device_id for item in rollout_placements)
          & set(item.device_id for item in trainer_placements)
      ),
      "exhaustive": len(set(ids)) == len(rollout) + len(trainer),
      "host_complete": True,
      "rollout_ids": tuple(item.device_id for item in rollout_placements),
      "trainer_ids": tuple(item.device_id for item in trainer_placements),
      **host_report,
  }
  if not report["disjoint"] or not report["exhaustive"]:
    raise AssertionError(
        "P44 128-chip role halves are not disjoint and exhaustive"
    )
  return rollout, trainer, report


def p58_sampler_recipe(values: Mapping[str, str]) -> str:
  """Returns the signed P58 sampler recipe or rejects a mixed tuple."""
  if values.get("CANON_P58_DEEPSWE_TIM") != "1":
    raise ValueError("P58 sampler recipe requested outside P58")
  arm = values.get("CANON_P58_TIM_ARM", "")
  sampler_tuple = (
      values.get("CANON_P34_DISABLE_SAMPLER_IS"),
      values.get("CANON_P34_DISABLE_TIS"),
  )
  if arm == "native" and sampler_tuple == ("1", "1"):
    return "native-raw"
  if arm == "native" and sampler_tuple == ("0", "0"):
    return "native-is"
  if arm == "zero" and sampler_tuple == ("1", "1"):
    return "zero"
  raise ValueError(
      "P58 sampler recipe must be native raw 1/1, native IS 0/0, or "
      "zero 1/1"
  )


def validate_environment(values: Mapping[str, str]) -> None:
  """Validates the exact P34 profile without accepting implicit defaults."""
  workload = active_workload(values)
  pilot = workload.contract_name == "p39-64chip-pilot"
  debug = workload.contract_name == "p43-64chip-debug"
  parity = workload.contract_name in (
      "p44-qwen4b-parity-64",
      "p44-qwen4b-parity-128",
  )
  p46_train = workload.contract_name in (
      "p46-qwen32b-train-64",
      "p46-qwen32b-train-256",
  )
  p58_tim = workload.contract_name == "p58-qwen4b-tim-128"
  p58_arm = values.get("CANON_P58_TIM_ARM", "")
  if p58_tim and p58_arm not in ("native", "zero"):
    raise ValueError("CANON_P58_TIM_ARM must be native or zero")
  p58_recipe = p58_sampler_recipe(values) if p58_tim else ""
  p58_hp = values.get("CANON_V1_HP_FULL", "0") == "1"
  p58_vma_diagnostic = values.get(
      "CANON_P58_CHECKED_VMA_DIAGNOSTIC", ""
  )
  if p58_vma_diagnostic not in ("", "off"):
    raise ValueError(
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC must be absent or off"
    )
  if p58_vma_diagnostic and (
      not p58_hp
      or not p58_tim
      or p58_arm != "zero"
      or values.get("CANON_P34_RUN_STAGE", "") != "full"
  ):
    raise ValueError(
        "P58 checked-VMA diagnostic requires the exact Zero/full HP carrier"
    )
  if p58_hp and (
      not p58_tim
      or p58_arm != "zero"
      or values.get("CANON_P34_RUN_STAGE", "") != "full"
  ):
    raise ValueError("P58 v1-hp is admitted only for strict Zero full")
  parity_topology = str(workload.devices_per_role * 2) if parity else "none"
  production_capture = bool(
      not pilot
      and not debug
      and not parity
      and not p58_tim
      and values.get("CANON_P34_RUN_STAGE") == "full"
  )
  numerical_bundle = not p58_tim or p58_arm == "zero"
  expected = {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_TOPOLOGY_ADMITTED": "1",
      "CANON_P34_TP8_ADMITTED": "1",
      "CANON_P34_TRAJECTORY_ADMITTED": "1",
      "CANON_P34_UPDATE_ADMITTED": "1",
      "CANON_FIXED_AR": "1" if numerical_bundle else None,
      "CANON_FIXED_AR_EMBED": "1" if numerical_bundle else None,
      "CANON_RPA_VJP2": "1" if numerical_bundle else "0",
      # Each segmented model_fn call activates one request per DP shard.  The
      # scheduler may reserve four request slots per shard, but unrolling four
      # VJPs would differentiate inactive capacity rather than the call.
      "CANON_VJP2_MAX_SEQS": "1" if numerical_bundle else "0",
      "CANON_LOGPROB_M": "256" if numerical_bundle else None,
      "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": (
          "1" if p58_tim and p58_arm == "native" else (
              "0" if p58_tim else None
          )
      ),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
      "CANON_P34_ABCPROD": "256",
      "CANON_QWEN3_TP_SIZE": "8",
      "CANON_P34_PREFIX_CACHE": "0",
      "CANON_P34_MAX_NUM_SEQS": str(workload.max_num_seqs_per_dp),
      "CANON_P34_MAX_BATCHED_TOKENS": "256",
      "CANON_P34_STRICT_CLI": "1",
      "CANON_P34_DISABLE_SAMPLER_IS": (
          "0" if p58_recipe == "native-is" else "1"
      ),
      "CANON_P34_DISABLE_TIS": (
          "0" if p58_recipe == "native-is" else "1"
      ),
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P34_TRAJECTORY_CAPTURE": "1" if production_capture else "0",
      "CANON_P34_DATASET_NAME": "R2E-Gym/R2E-Gym-Subset",
      "CANON_P34_DATASET_REVISION": (
          "2e8108ff942f24fcb5686badfaf7f9a8808566d5"
      ),
      "CANON_P34_DATASET_SPLIT": "train",
      "CANON_P34_DATASET_ROWS": "4578",
      "CANON_P34_CLEAN_ROWS": (
          "1012" if p58_tim else (
              "1851" if production_capture or parity else "0"
          )
      ),
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": str(
          workload.cleanup_timeout_secs
      ),
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": str(
          workload.rollout_batch_timeout_secs
      ),
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": str(
          workload.per_turn_timeout_secs
      ),
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": str(
          workload.episode_timeout_secs
      ),
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": str(
          workload.step_timeout_secs
      ),
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": str(
          workload.reward_timeout_secs
      ),
      "R2E_ACTIVE_DEADLINE_SECONDS": str(
          workload.sandbox_active_deadline_secs
      ),
      "R2E_POD_DELETE_TIMEOUT_SECONDS": "300",
      "R2E_K8S_CPU": "2",
      "R2E_K8S_MEM": "4Gi",
      "R2E_K8S_CPU_LIMIT": "4",
      "R2E_K8S_MEM_LIMIT": "8Gi",
      "R2E_K8S_QUEUE_NAME": "multislice-queue" if p58_tim else None,
      "WANDB_MODE": "online",
      "CANON_P39_64CHIP_PILOT": "1" if pilot else "0",
      "CANON_P39_PILOT_ADMITTED": "1" if pilot else "0",
      "CANON_P43_DEEPSWE_DEBUG": "1" if debug else "0",
      "CANON_P43_DEBUG_ADMITTED": "1" if debug else "0",
      "CANON_P43_ROLLOUT_ONLY": (
          "1"
          if debug and values.get("CANON_P34_RUN_STAGE") == "rollout-only"
          else "0"
      ),
      "CANON_P44_DEEPSWE_PARITY": "1" if parity else "0",
      "CANON_P44_PARITY_ADMITTED": "1" if parity else "0",
      "CANON_P44_TOPOLOGY": parity_topology,
      "CANON_P44_ROLLOUT_ONLY": (
          "1"
          if parity and values.get("CANON_P34_RUN_STAGE") == "rollout-only"
          else "0"
      ),
      "CANON_P46_DEEPSWE_TRAIN": "1" if p46_train else "0",
      "CANON_P46_TOPOLOGY": (
          str(workload.devices_per_role * 2) if p46_train else "none"
      ),
      "CANON_P58_DEEPSWE_TIM": "1" if p58_tim else "0",
      "CANON_P58_TIM_ADMITTED": "1" if p58_tim else "0",
      "CANON_P58_TIM_ARM": p58_arm if p58_tim else "none",
      "CANON_P58_EXPECTED_UPDATES": (
          str(requested_max_steps(values)) if p58_tim else "0"
      ),
      "CANON_DP_SIZE": str(workload.dp_size),
      "CANON_TP_SIZE": str(workload.tp_size),
      "CANON_TOTAL_DEVICES": str(workload.devices_per_role),
      "CANON_ENGINE_DP_SIZE": str(workload.dp_size),
      "CANON_GLOBAL_PROMPTS": str(workload.global_prompts),
      "CANON_NUM_GENERATIONS": str(workload.generations),
      "CANON_LOCAL_TRAJECTORIES": str(workload.local_trajectories),
      "CANON_GLOBAL_TRAJECTORIES": str(workload.global_trajectories),
      "FL_SHARED_MESH": f"{workload.dp_size},{workload.tp_size}",
  }
  if p58_tim:
    expected.update({
        "CANON_PROMPT_PROCESSED_LOGPROBS": (
            "0" if p58_arm == "native" else "1"
        ),
        "CANON_ENGINE_MODULE_C": "0" if p58_arm == "native" else "1",
        "CANON_V1_HP_FULL": "1" if p58_hp else "0",
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC": (
            "off" if p58_vma_diagnostic else None
        ),
        "CANON_P67_P66_VMA_P59_ONLY": (
            "0" if p58_vma_diagnostic else "1" if p58_hp else None
        ),
    })
    if p58_hp:
      expected.update({
          "CANON_P38_FIXED_LM_HEAD": "1",
          "CANON_CONTINUE_DECODE": "8",
          "CANON_FIXED_AR_GATHER": "1",
          "CANON_PALLAS_GATHERED_LOGPROBS": "1",
          "CANON_LOGPROB_STEP_FUSION": "1",
          "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
          "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
          "CANON_P59_CHECKED_VMA": (
              "0" if p58_vma_diagnostic else "1"
          ),
          "CANON_P66_P59_CHECK_VMA": (
              "0" if p58_vma_diagnostic else "1"
          ),
          "CANON_V1_HP_FIRST_UPDATE_GATE": (
              "0" if p58_vma_diagnostic else "1"
          ),
          "CANON_P63_OVERFLOW_SAFE_CLIP": (
              "0" if p58_vma_diagnostic else "1"
          ),
          "CANON_P28_BATCHED_REPORT": "1",
          "CANON_P28_BATCHED_REVERSE": "0",
          "CANON_BATCHED_EVIDENCE": "0",
          "CANON_FUSED_TREE_OPS": "0",
          "CANON_PALLAS_NORM_MATMUL": "0",
          "CANON_PALLAS_INPUT_FUSION": "0",
          "CANON_SAMPLE_SPLIT_FUSION": "0",
          "CANON_ENGINE_LOGPROB_READBACK": "0",
          "CANON_ANCHOR_OVERLAP": "0",
          "CANON_XPROF_PHASE": "update",
          "CANON_XPROF_SKIP_STEPS": "2",
          "CANON_XPROF_STEPS": "1",
          "CANON_XPROF_PYTHON_TRACER": "0",
          "CANON_XPROF_HOST_TRACER": "1",
          "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
          "CANON_XPROF_LABELS": "1",
          "CANON_PERF_TRACE_EXPORT_STEP": "2",
      })
      if p58_vma_diagnostic:
        expected.update({
            "CANON_P38_PRECHECK_ONLY": "1",
            "CANON_P38_CONTROLLED_EXIT": "1",
            "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
        })
        round_file = values.get("CANON_P38_DIAGNOSTIC_ROUND_FILE", "")
        if not round_file or not os.path.isabs(round_file):
          raise ValueError(
              "P58 checked-VMA diagnostic round file must be absolute"
          )
      else:
        expected.update({
            "CANON_P38_PRECHECK_ONLY": None,
            "CANON_P38_CONTROLLED_EXIT": None,
            "CANON_P38_DIAGNOSTIC_ROUNDS": None,
            "CANON_P38_DIAGNOSTIC_ROUND_FILE": None,
        })
  if pilot or debug or parity or p58_tim:
    expected.update({
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": (
          ("1" if p58_arm == "native" else "0") if p58_tim else (
              "1" if parity or production_capture else "0"
          )
      ),
    })
  else:
    # Production full training is convergence-first: finite alignment drift is
    # durable warning telemetry, while invalid shapes and nonfinite values
    # remain blocking in alignment.py.
    expected.update({
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
        "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1",
    })
  wrong = {
      key: values.get(key)
      for key, expected_value in expected.items()
      if values.get(key) != expected_value
  }
  if wrong:
    raise ValueError(f"P34 environment mismatch: {wrong}")
  flags = values.get("XLA_FLAGS", "")
  has_precision_pin = "--xla_allow_excess_precision=false" in flags
  if numerical_bundle and not has_precision_pin:
    raise ValueError("P34 XLA_FLAGS lost the excess-precision contract")
  if not numerical_bundle and has_precision_pin:
    raise ValueError("P58 native XLA_FLAGS leaked the zero-TIM precision pin")
  weight_report = values.get("CANON_P34_WEIGHT_REPORT", "")
  if not weight_report or not os.path.isabs(weight_report):
    raise ValueError("P34 weight attestation report path is missing")
  if debug:
    debug_dir = values.get("CANON_P43_DEBUG_DIR", "")
    if not debug_dir or not os.path.isabs(debug_dir):
      raise ValueError("P43 debug artifact directory is missing")
  if parity:
    debug_dir = values.get("CANON_P44_DEBUG_DIR", "")
    if not debug_dir or not os.path.isabs(debug_dir):
      raise ValueError("P44 parity artifact directory is missing")
  if p58_tim:
    debug_dir = values.get("CANON_P58_DEBUG_DIR", "")
    if not debug_dir or not os.path.isabs(debug_dir):
      raise ValueError("P58 trajectory artifact directory is missing")
  if production_capture:
    debug_dir = values.get("CANON_P34_DEBUG_DIR", "")
    if not debug_dir or not os.path.isabs(debug_dir):
      raise ValueError("P34 production trajectory artifact directory is missing")
  workload.validate()
  requested_max_steps(values)


def requested_max_steps(values: Mapping[str, str]) -> int:
  """Returns the fail-closed P34 promotion-stage update budget."""
  workload = active_workload(values)
  stage = values.get("CANON_P34_RUN_STAGE", "")
  no_commit = values.get("CANON_P34_NO_COMMIT", "0")
  if stage == "rollout-only":
    if workload.contract_name not in (
        "p43-64chip-debug",
        "p44-qwen4b-parity-64",
        "p44-qwen4b-parity-128",
    ):
      raise ValueError("rollout-only is admitted only for P43/P44 debug")
    expected_no_commit, steps = "1", 1
  elif stage == "backward-no-commit":
    expected_no_commit, steps = "1", 1
  elif stage == "one-update":
    expected_no_commit, steps = "0", 1
  elif stage == "three-update":
    expected_no_commit, steps = "0", 3
  elif stage == "full":
    expected_no_commit, steps = "0", workload.max_steps
  else:
    raise ValueError(f"unknown P34 run stage: {stage!r}")
  if no_commit != expected_no_commit:
    raise ValueError(
        "P34 stage/no-commit mismatch: "
        f"stage={stage!r} expected CANON_P34_NO_COMMIT={expected_no_commit}"
      )
  if workload.contract_name == "p39-64chip-pilot" and stage not in (
      "one-update",
      "three-update",
  ):
    raise ValueError(
        "P39 64-chip pilot admits only one-update or three-update"
    )
  if workload.contract_name == "p43-64chip-debug" and stage not in (
      "rollout-only",
      "one-update",
      "three-update",
  ):
    raise ValueError(
        "P43 64-chip debug admits only rollout-only, one-update, or "
        "three-update"
    )
  if workload.contract_name in (
      "p44-qwen4b-parity-64",
      "p44-qwen4b-parity-128",
  ) and stage not in (
      "rollout-only",
      "one-update",
      "three-update",
  ):
    raise ValueError(
        "P44 Qwen3-4B parity admits only rollout-only, one-update, or "
      "three-update"
    )
  if workload.contract_name == "p58-qwen4b-tim-128" and stage not in (
      "three-update",
      "full",
  ):
    raise ValueError("P58 admits only three-update or full")
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
  workload = active_workload(os.environ)
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
      "mesh_shape": mesh_shape
      == {"dp": workload.dp_size, "tp": workload.tp_size},
      "mesh_device_ids": (
          len(mesh_device_ids) == workload.devices_per_role
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
      f"elements={record['total_elements']} "
      f"devices={len(record['mesh_device_ids'])}",
      flush=True,
  )
  return record
