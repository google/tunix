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

"""Fail-closed contracts for replicated data-parallel RL training."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class DPTrainingContract:
  """Describes one fixed-placement DP training transaction."""

  dp_size: int
  tp_size: int
  global_prompts: int
  num_generations: int
  local_trajectories: int

  @property
  def total_devices(self) -> int:
    return self.dp_size * self.tp_size

  @property
  def global_trajectories(self) -> int:
    return self.global_prompts * self.num_generations

  @property
  def local_prompts(self) -> int:
    return self.global_prompts // self.dp_size

  def validate(self) -> None:
    """Rejects an inconsistent topology or trajectory transaction."""
    positive = {
        "dp_size": self.dp_size,
        "tp_size": self.tp_size,
        "global_prompts": self.global_prompts,
        "num_generations": self.num_generations,
        "local_trajectories": self.local_trajectories,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
      raise ValueError(f"DP training values must be positive: {invalid}")
    if self.global_prompts % self.dp_size:
      raise ValueError(
          "global_prompts must be divisible by dp_size: "
          f"{self.global_prompts} % {self.dp_size}"
      )
    expected_local = self.local_prompts * self.num_generations
    if self.local_trajectories != expected_local:
      raise ValueError(
          "local trajectory count does not match complete prompt groups: "
          f"{self.local_trajectories} != {expected_local}"
      )
    if self.local_trajectories * self.dp_size != self.global_trajectories:
      raise ValueError(
          "local/global trajectory arithmetic does not close: "
          f"{self.local_trajectories} * {self.dp_size} != "
          f"{self.global_trajectories}"
      )

  def trajectory_ranks(self) -> np.ndarray:
    """Returns the frozen rank assignment for prompt-major trajectories."""
    self.validate()
    prompt_ranks = np.repeat(
        np.arange(self.dp_size, dtype=np.int32), self.local_prompts
    )
    return np.repeat(prompt_ranks, self.num_generations)

  def rank_indices(self) -> tuple[np.ndarray, ...]:
    """Returns trajectory indices owned by each DP rank."""
    ranks = self.trajectory_ranks()
    return tuple(np.flatnonzero(ranks == rank) for rank in range(self.dp_size))

  def rank_major_reverse_groups(self) -> tuple[tuple[int, ...], ...]:
    """Pairs the same local trajectory ordinal across replicated DP ranks."""
    self.validate()
    rank_indices = self.rank_indices()
    local_counts = {indices.size for indices in rank_indices}
    if local_counts != {self.local_trajectories}:
      raise ValueError(
          "rank-major reverse groups require equal local trajectory counts: "
          f"{sorted(local_counts)}"
      )
    return tuple(
        tuple(int(rank_indices[rank][local_index]) for rank in range(self.dp_size))
        for local_index in range(self.local_trajectories)
    )

  def validate_prompt_groups(self, group_ids: Sequence[int]) -> None:
    """Checks that every generation group stays on exactly one DP rank."""
    self.validate()
    groups = np.asarray(group_ids)
    if groups.shape != (self.global_trajectories,):
      raise ValueError(
          "prompt-group id shape changed: "
          f"{groups.shape} != {(self.global_trajectories,)}"
      )
    ranks = self.trajectory_ranks()
    unique_groups = np.unique(groups)
    if unique_groups.size != self.global_prompts:
      raise ValueError(
          "prompt-group count changed: "
          f"{unique_groups.size} != {self.global_prompts}"
      )
    for group in unique_groups:
      rows = np.flatnonzero(groups == group)
      if rows.size != self.num_generations:
        raise ValueError(
            f"prompt group {group} has {rows.size} trajectories; "
            f"expected {self.num_generations}"
        )
      owners = np.unique(ranks[rows])
      if owners.size != 1:
        raise ValueError(
            f"prompt group {group} is split across DP ranks {owners.tolist()}"
        )


def _axis_names(spec: Any) -> tuple[str, ...]:
  """Flattens mesh-axis names from a PartitionSpec-like value."""
  names = []

  def visit(value):
    if isinstance(value, str):
      names.append(value)
    elif isinstance(value, (tuple, list)):
      for child in value:
        visit(child)

  visit(tuple(spec) if spec is not None else ())
  return tuple(names)


def validate_dp_replicated_partition_specs(
    specs: Any, *, label: str, dp_axis: str = "dp"
) -> dict[str, int]:
  """Rejects any state leaf partitioned over the DP axis."""
  leaves = jax.tree.leaves(specs)
  violations = [index for index, spec in enumerate(leaves)
                if dp_axis in _axis_names(spec)]
  if not leaves:
    raise ValueError(f"{label} partition inventory is empty")
  if violations:
    raise ValueError(
        f"{label} is not replicated over {dp_axis!r}; "
        f"violating leaf indices={violations[:8]} total={len(violations)}"
    )
  return {"leaves": len(leaves), "dp_partitioned_leaves": 0}


def detach_jax_vllm_cleanup_finalizer(rollout: Any) -> dict[str, bool]:
  """Detaches the torch-only vLLM cleanup hook from a verified JAX model."""
  sampler = getattr(rollout, "_sampler", None)
  llm = getattr(sampler, "llm", None)
  driver = getattr(sampler, "_driver", None)
  engine = getattr(llm, "llm_engine", None)
  if engine is None:
    engine = getattr(driver, "llm_engine", None)
  finalizer = getattr(engine, "_finalizer", None)
  cleanup_model = (
      engine._get_driver_model_for_cleanup()
      if engine is not None
      and hasattr(engine, "_get_driver_model_for_cleanup")
      else None
  )
  if (
      finalizer is None
      or not finalizer.alive
      or cleanup_model is None
      or hasattr(cleanup_model, "modules")
  ):
    raise ValueError(
        "expected the known JAX vLLM cleanup contract: "
        f"finalizer={finalizer!r} model={type(cleanup_model)!r}"
    )
  if finalizer.detach() is None:
    raise ValueError("could not detach the invalid torch cleanup hook")
  return {"jax_vllm_finalizer_detached": True}


def fixed_dp2_sum(left: Any, right: Any) -> Any:
  """Adds two explicit rank contributions in the registered rank order."""
  return jax.tree.map(
      lambda rank0, rank1: (
          jax.lax.optimization_barrier(rank0) + rank1
      ),
      left,
      right,
  )


def fixed_dp2_collective(local_value: Any, axis_name: str = "dp") -> Any:
  """Returns rank0+rank1 identically on both ranks inside a mapped DP axis."""
  rank = jax.lax.axis_index(axis_name)
  peer_value = jax.lax.ppermute(
      local_value, axis_name=axis_name, perm=((0, 1), (1, 0))
  )
  rank0 = jnp.where(rank == 0, local_value, peer_value)
  rank1 = jnp.where(rank == 0, peer_value, local_value)
  return jax.lax.optimization_barrier(rank0) + rank1


def isolate_dp_rank_cotangent(
    value: Any, *, rank: int, dp_size: int = 2
) -> Any:
  """Keeps one row of a DP-group cotangent and zeros every peer row."""
  if dp_size != 2:
    raise ValueError(f"the registered reducer admits dp_size=2, got {dp_size}")
  if rank not in range(dp_size):
    raise ValueError(f"rank {rank} is outside [0, {dp_size})")
  if getattr(value, "ndim", 0) < 1 or value.shape[0] != dp_size:
    raise ValueError(
        "DP-group cotangent must have a leading rank axis: "
        f"shape={getattr(value, 'shape', None)} dp_size={dp_size}"
    )
  mask = jnp.arange(dp_size, dtype=jnp.int32) == rank
  mask = mask.reshape((dp_size,) + (1,) * (value.ndim - 1))
  return jnp.where(mask, value, jnp.zeros_like(value))
