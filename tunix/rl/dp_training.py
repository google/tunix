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
import hashlib
import math
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
    values = {
        'dp_size': self.dp_size,
        'tp_size': self.tp_size,
        'global_prompts': self.global_prompts,
        'num_generations': self.num_generations,
        'local_trajectories': self.local_trajectories,
    }
    invalid = {name: value for name, value in values.items() if value <= 0}
    if invalid:
      raise ValueError(f'DP training values must be positive: {invalid}')
    if self.global_prompts % self.dp_size:
      raise ValueError(
          'global_prompts must be divisible by dp_size: '
          f'{self.global_prompts} % {self.dp_size}'
      )
    expected_local = self.local_prompts * self.num_generations
    if self.local_trajectories != expected_local:
      raise ValueError(
          'local trajectory count does not match complete prompt groups: '
          f'{self.local_trajectories} != {expected_local}'
      )
    if self.local_trajectories * self.dp_size != self.global_trajectories:
      raise ValueError(
          'local/global trajectory arithmetic does not close: '
          f'{self.local_trajectories} * {self.dp_size} != '
          f'{self.global_trajectories}'
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
    return tuple(
        np.flatnonzero(ranks == rank) for rank in range(self.dp_size)
    )

  def rank_major_reverse_groups(self) -> tuple[tuple[int, ...], ...]:
    """Groups the same local trajectory ordinal across all DP ranks."""
    self.validate()
    rank_indices = self.rank_indices()
    local_counts = {indices.size for indices in rank_indices}
    if local_counts != {self.local_trajectories}:
      raise ValueError(
          'rank-major reverse groups require equal local trajectory counts: '
          f'{sorted(local_counts)}'
      )
    return tuple(
        tuple(
            int(rank_indices[rank][local_index])
            for rank in range(self.dp_size)
        )
        for local_index in range(self.local_trajectories)
    )

  def validate_prompt_groups(self, group_ids: Sequence[int]) -> None:
    """Checks that every generation group stays on one DP rank."""
    self.validate()
    groups = np.asarray(group_ids)
    if groups.shape != (self.global_trajectories,):
      raise ValueError(
          'prompt-group id shape changed: '
          f'{groups.shape} != {(self.global_trajectories,)}'
      )
    ranks = self.trajectory_ranks()
    unique_groups = np.unique(groups)
    if unique_groups.size != self.global_prompts:
      raise ValueError(
          'prompt-group count changed: '
          f'{unique_groups.size} != {self.global_prompts}'
      )
    for group in unique_groups:
      rows = np.flatnonzero(groups == group)
      if rows.size != self.num_generations:
        raise ValueError(
            f'prompt group {group} has {rows.size} trajectories; '
            f'expected {self.num_generations}'
        )
      owners = np.unique(ranks[rows])
      if owners.size != 1:
        raise ValueError(
            f'prompt group {group} is split across DP ranks {owners.tolist()}'
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
    specs: Any, *, label: str, dp_axis: str = 'dp'
) -> dict[str, int]:
  """Rejects any state leaf partitioned over the DP axis."""
  leaves = jax.tree.leaves(specs)
  violations = [
      index
      for index, spec in enumerate(leaves)
      if dp_axis in _axis_names(spec)
  ]
  if not leaves:
    raise ValueError(f'{label} partition inventory is empty')
  if violations:
    raise ValueError(
        f'{label} is not replicated over {dp_axis!r}; '
        f'violating leaf indices={violations[:8]} total={len(violations)}'
    )
  return {'leaves': len(leaves), 'dp_partitioned_leaves': 0}


def inspect_dp_replicated_state(
    state: Any,
    *,
    label: str,
    dp_axis: str = 'dp',
    tp_axis: str = 'tp',
    require_tp_partition: bool = True,
) -> dict[str, Any]:
  """Returns a fail-closed sharding inventory for one initialized state."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  specs = nnx.get_partition_spec(state)
  summary = validate_dp_replicated_partition_specs(
      specs, label=label, dp_axis=dp_axis
  )
  spec_leaves = jax.tree.leaves(specs)
  tp_partitioned = sum(
      tp_axis in _axis_names(spec) for spec in spec_leaves
  )
  if require_tp_partition and tp_partitioned == 0:
    raise ValueError(
        f'{label} has no leaves partitioned over the {tp_axis!r} axis'
    )

  arrays = [
      value for value in jax.tree.leaves(state) if isinstance(value, jax.Array)
  ]
  if not arrays:
    raise ValueError(f'{label} initialized-state inventory has no JAX arrays')
  actual_dp_violations = [
      index
      for index, value in enumerate(arrays)
      if dp_axis in _axis_names(getattr(value.sharding, 'spec', ()))
  ]
  if actual_dp_violations:
    raise ValueError(
        f'{label} arrays are sharded over {dp_axis!r}; '
        f'violating indices={actual_dp_violations[:8]} '
        f'total={len(actual_dp_violations)}'
    )
  memory_kinds = sorted({
      str(value.sharding.memory_kind) for value in arrays
  })
  return {
      **summary,
      'arrays': len(arrays),
      'logical_bytes': sum(
          int(value.size * value.dtype.itemsize) for value in arrays
      ),
      'tp_partitioned_leaves': tp_partitioned,
      'memory_kinds': tuple(memory_kinds),
  }


def inspect_training_state_inventories(
    *, model: Any, optimizer: Any, accumulator: Any
) -> dict[str, dict[str, Any]]:
  """Inventories the three DP-replicated state classes before training."""
  return {
      'model': inspect_dp_replicated_state(model, label='model'),
      'optimizer': inspect_dp_replicated_state(optimizer, label='optimizer'),
      'accumulator': inspect_dp_replicated_state(
          accumulator, label='accumulator'
      ),
  }


def inspect_abstract_dp_replicated_state(
    state: Any,
    *,
    label: str,
    dp_axis: str = 'dp',
    tp_axis: str = 'tp',
    require_tp_partition: bool = True,
) -> dict[str, Any]:
  """Inventories ShapeDtypeStruct leaves without materializing model state."""
  leaves = [
      value
      for value in jax.tree.leaves(state)
      if hasattr(value, 'shape') and hasattr(value, 'dtype')
  ]
  if not leaves:
    raise ValueError(f'{label} abstract-state inventory is empty')
  specs = [
      getattr(getattr(value, 'sharding', None), 'spec', ()) for value in leaves
  ]
  dp_violations = [
      index for index, spec in enumerate(specs) if dp_axis in _axis_names(spec)
  ]
  if dp_violations:
    raise ValueError(
        f'{label} abstract state is sharded over {dp_axis!r}; '
        f'violating indices={dp_violations[:8]} total={len(dp_violations)}'
    )
  tp_partitioned = sum(tp_axis in _axis_names(spec) for spec in specs)
  if require_tp_partition and tp_partitioned == 0:
    raise ValueError(
        f'{label} abstract state has no {tp_axis!r}-partitioned leaves'
    )
  return {
      'leaves': len(leaves),
      'logical_bytes': sum(
          int(np.prod(value.shape, dtype=np.int64) * value.dtype.itemsize)
          for value in leaves
      ),
      'dp_partitioned_leaves': 0,
      'tp_partitioned_leaves': tp_partitioned,
      'unsharded_leaves': sum(not _axis_names(spec) for spec in specs),
  }


def inspect_abstract_training_state_inventories(
    *, model: Any, optimizer: Any, accumulator: Any
) -> dict[str, dict[str, Any]]:
  """Inventories abstract Qwen, optimizer, and accumulator state trees."""
  return {
      'model': inspect_abstract_dp_replicated_state(model, label='model'),
      'optimizer': inspect_abstract_dp_replicated_state(
          optimizer, label='optimizer'
      ),
      'accumulator': inspect_abstract_dp_replicated_state(
          accumulator, label='accumulator'
      ),
  }


def attach_adam_state_shardings(
    optimizer_state: Any, *, params: Any, mesh: jax.sharding.Mesh
) -> Any:
  """Attaches parameter shardings to abstract Adam moments fail-closed."""
  if not isinstance(optimizer_state, tuple):
    raise ValueError('Adam optimizer state must be a tuple of transforms')
  candidates = [
      index
      for index, value in enumerate(optimizer_state)
      if all(hasattr(value, name) for name in ('count', 'mu', 'nu'))
  ]
  if len(candidates) != 1:
    raise ValueError(
        'expected exactly one Adam moment state, got '
        f'indices={candidates}'
    )
  index = candidates[0]
  adam_state = optimizer_state[index]
  param_structure = jax.tree.structure(params)
  if (
      jax.tree.structure(adam_state.mu) != param_structure
      or jax.tree.structure(adam_state.nu) != param_structure
  ):
    raise ValueError('Adam moment trees do not match the parameter tree')

  def attach(moment, param):
    if tuple(moment.shape) != tuple(param.shape):
      raise ValueError(
          f'Adam moment shape changed: {moment.shape} != {param.shape}'
      )
    sharding = getattr(param, 'sharding', None)
    if sharding is None:
      raise ValueError('parameter leaf is missing an admitted sharding')
    return jax.ShapeDtypeStruct(
        moment.shape, moment.dtype, sharding=sharding
    )

  mu = jax.tree.map(attach, adam_state.mu, params)
  nu = jax.tree.map(attach, adam_state.nu, params)
  count = jax.ShapeDtypeStruct(
      adam_state.count.shape,
      adam_state.count.dtype,
      sharding=jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec()
      ),
  )
  result = list(optimizer_state)
  result[index] = adam_state._replace(count=count, mu=mu, nu=nu)
  return tuple(result)


def _validate_tree_size(dp_size: int) -> None:
  if dp_size < 2 or dp_size & (dp_size - 1):
    raise ValueError(
        f'fixed DP tree requires a power-of-two dp_size >= 2, got {dp_size}'
    )


def fixed_dp_tree_permutations(
    dp_size: int,
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[tuple[int, int], ...], ...],
]:
  """Returns fixed reduce and broadcast collective-permute schedules."""
  _validate_tree_size(dp_size)
  strides = tuple(1 << index for index in range(int(math.log2(dp_size))))
  reduce_rounds = tuple(
      tuple(
          (base + stride, base)
          for base in range(0, dp_size, 2 * stride)
      )
      for stride in strides
  )
  broadcast_rounds = tuple(
      tuple(
          (base, base + stride)
          for base in range(0, dp_size, 2 * stride)
      )
      for stride in reversed(strides)
  )
  return reduce_rounds, broadcast_rounds


def fixed_dp_collective_count(dp_size: int) -> int:
  """Returns the registered collective-permute count for one DP reduction."""
  reduce_rounds, broadcast_rounds = fixed_dp_tree_permutations(dp_size)
  return len(reduce_rounds) + len(broadcast_rounds)


def fixed_dp_sum(contributions: Sequence[Any]) -> Any:
  """Sums rank contributions with the same binary tree as the collective."""
  values = list(contributions)
  _validate_tree_size(len(values))
  while len(values) > 1:
    values = [
        jax.tree.map(
            lambda left, right: (
                jax.lax.optimization_barrier(left)
                + jax.lax.optimization_barrier(right)
            ),
            values[index],
            values[index + 1],
        )
        for index in range(0, len(values), 2)
    ]
  return values[0]


def fixed_dp2_sum(left: Any, right: Any) -> Any:
  """Preserves the registered rank-zero-then-rank-one DP2 order."""
  return fixed_dp_sum((left, right))


def _select_tree(receiver: Any, received: Any, retained: Any) -> Any:
  return jax.tree.map(
      lambda incoming, current: jnp.where(receiver, incoming, current),
      received,
      retained,
  )


def fixed_dp_collective(
    local_value: Any, *, dp_size: int, axis_name: str = 'dp'
) -> Any:
  """Returns a fixed-order DP sum identically on every mapped DP rank."""
  reduce_rounds, broadcast_rounds = fixed_dp_tree_permutations(dp_size)
  rank = jax.lax.axis_index(axis_name)
  value = local_value
  for round_index, permutation in enumerate(reduce_rounds):
    stride = 1 << round_index
    peer = jax.tree.map(
        lambda leaf: jax.lax.ppermute(
            leaf, axis_name=axis_name, perm=permutation
        ),
        value,
    )
    combined = jax.tree.map(
        lambda left, right: (
            jax.lax.optimization_barrier(left)
            + jax.lax.optimization_barrier(right)
        ),
        value,
        peer,
    )
    receiver = jnp.mod(rank, 2 * stride) == 0
    value = _select_tree(receiver, combined, value)
  for reverse_index, permutation in enumerate(broadcast_rounds):
    stride = 1 << (len(broadcast_rounds) - reverse_index - 1)
    peer = jax.tree.map(
        lambda leaf: jax.lax.ppermute(
            leaf, axis_name=axis_name, perm=permutation
        ),
        value,
    )
    receiver = jnp.mod(rank, 2 * stride) == stride
    value = _select_tree(receiver, peer, value)
  return value


def fixed_dp2_collective(local_value: Any, axis_name: str = 'dp') -> Any:
  """Compatibility wrapper for the previously registered DP2 reducer."""
  return fixed_dp_collective(local_value, dp_size=2, axis_name=axis_name)


def _contains_axis(spec: jax.sharding.PartitionSpec, axis_name: str) -> bool:
  for entry in tuple(spec):
    if entry == axis_name:
      return True
    if isinstance(entry, tuple) and axis_name in entry:
      return True
  return False


def _gradient_signature(tree: Any) -> jax.Array:
  """Returns a compact deterministic signature without copying full leaves."""
  total = jnp.asarray(0.0, jnp.float32)
  absolute = jnp.asarray(0.0, jnp.float32)
  squared = jnp.asarray(0.0, jnp.float32)
  weighted = jnp.asarray(0.0, jnp.float32)
  nonzero = jnp.asarray(0.0, jnp.float32)
  for index, leaf in enumerate(jax.tree.leaves(tree), start=1):
    value = leaf.astype(jnp.float32)
    total = total + jnp.sum(value)
    absolute = absolute + jnp.sum(jnp.abs(value))
    squared = squared + jnp.sum(jnp.square(value))
    weighted = weighted + jnp.asarray(index, jnp.float32) * jnp.sum(value)
    nonzero = nonzero + jnp.count_nonzero(value).astype(jnp.float32)
  return jnp.stack((total, absolute, squared, weighted, nonzero))


def _signature_sha256(signature: Any) -> str:
  value = np.ascontiguousarray(jax.device_get(signature))
  return hashlib.sha256(value.view(np.uint8)).hexdigest()


class FixedDPRankGradientReducer:
  """Stages one contribution per DP rank and reduces it with a fixed tree.

  The leading staging axis is physically partitioned over ``dp``. Therefore a
  logical ``[dp, ...parameter_shape]`` table stores only one rank contribution
  per DP replica. Finalization executes one reduce-and-broadcast transaction
  with the registered collective-permute schedule and returns the original
  DP-replicated, TP-sharded gradient tree.
  """

  def __init__(
      self,
      template: Any,
      *,
      dp_size: int,
      dp_axis: str = 'dp',
      require_distinct_fingerprints: bool = True,
  ):
    _validate_tree_size(dp_size)
    leaves = jax.tree.leaves(template)
    if not leaves or any(not isinstance(leaf, jax.Array) for leaf in leaves):
      raise ValueError('DP gradient reducer requires a nonempty JAX array tree')
    shardings = [leaf.sharding for leaf in leaves]
    if any(
        not isinstance(sharding, jax.sharding.NamedSharding)
        for sharding in shardings
    ):
      raise ValueError('DP gradient reducer requires NamedSharding leaves')
    mesh = shardings[0].mesh
    if dp_axis not in mesh.axis_names or int(mesh.shape[dp_axis]) != dp_size:
      raise ValueError(
          'DP gradient reducer mesh mismatch: '
          f'axes={mesh.axis_names} shape={dict(mesh.shape)} '
          f'expected {dp_axis}={dp_size}'
      )
    for sharding in shardings[1:]:
      if sharding.mesh != mesh:
        raise ValueError('DP gradient reducer leaves use different meshes')

    base_specs = jax.tree.map(lambda leaf: leaf.sharding.spec, template)
    for spec in jax.tree.leaves(base_specs):
      if _contains_axis(spec, dp_axis):
        raise ValueError(
            'DP gradient inputs must be replicated over the DP axis: '
            f'{spec}'
        )
    staged_specs = jax.tree.map(
        lambda spec: jax.sharding.PartitionSpec(dp_axis, *tuple(spec)),
        base_specs,
    )
    staged_shardings = jax.tree.map(
        lambda spec: jax.sharding.NamedSharding(mesh, spec), staged_specs
    )

    def initialize():
      return jax.tree.map(
          lambda leaf: jnp.zeros((dp_size,) + leaf.shape, leaf.dtype),
          template,
      )

    def write(staged, contribution, rank):
      return jax.tree.map(
          lambda table, value: jax.lax.dynamic_update_index_in_dim(
              table, jnp.expand_dims(value, 0), rank, axis=0
          ),
          staged,
          contribution,
      )

    def reduce_local(local_staged):
      local_value = jax.tree.map(
          lambda value: jnp.squeeze(value, axis=0), local_staged
      )
      return fixed_dp_collective(
          local_value, dp_size=dp_size, axis_name=dp_axis
      )

    shard_map_kwargs = {
        'mesh': mesh,
        'in_specs': (staged_specs,),
        'out_specs': base_specs,
    }
    try:
      reduce_mapped = jax.shard_map(
          reduce_local, check_vma=False, **shard_map_kwargs
      )
    except TypeError:
      reduce_mapped = jax.shard_map(
          reduce_local, check_rep=False, **shard_map_kwargs
      )

    permutation = tuple(
        (rank, (rank + 1) % dp_size) for rank in range(dp_size)
    )

    def compare_local(local_tree):
      peer_tree = jax.tree.map(
          lambda leaf: jax.lax.ppermute(
              leaf, axis_name=dp_axis, perm=permutation
          ),
          local_tree,
      )
      exact = jnp.asarray(True)
      for local_leaf, peer_leaf in zip(
          jax.tree.leaves(local_tree),
          jax.tree.leaves(peer_tree),
          strict=True,
      ):
        exact = jnp.logical_and(exact, jnp.array_equal(local_leaf, peer_leaf))
      return jnp.reshape(exact, (1,))

    compare_kwargs = {
        'mesh': mesh,
        'in_specs': (base_specs,),
        'out_specs': jax.sharding.PartitionSpec(dp_axis),
    }
    try:
      compare_mapped = jax.shard_map(
          compare_local, check_vma=False, **compare_kwargs
      )
    except TypeError:
      compare_mapped = jax.shard_map(
          compare_local, check_rep=False, **compare_kwargs
      )

    self._dp_size = dp_size
    self._require_distinct = require_distinct_fingerprints
    self._initialize = jax.jit(initialize, out_shardings=staged_shardings)
    self._write = jax.jit(write, donate_argnums=(0,))
    self._reduce = jax.jit(reduce_mapped, donate_argnums=(0,))
    self._compare = jax.jit(compare_mapped)
    self._signature = jax.jit(_gradient_signature)
    self._staged = None
    self._next_rank = 0
    self._fingerprints = []

  def begin(self) -> None:
    """Starts one reduction transaction with an empty rank table."""
    if self._staged is not None:
      raise ValueError('DP gradient reduction transaction is already active')
    self._staged = self._initialize()
    self._next_rank = 0
    self._fingerprints = []

  def add(self, rank: int, contribution: Any) -> str:
    """Stages exactly one rank contribution in monotonically increasing order."""
    if self._staged is None:
      raise ValueError('DP gradient reduction transaction is not active')
    rank = int(rank)
    if rank != self._next_rank:
      raise ValueError(
          'DP gradient contribution cadence changed: '
          f'expected rank {self._next_rank}, got {rank}'
      )
    fingerprint = _signature_sha256(self._signature(contribution))
    self._staged = self._write(
        self._staged, contribution, jnp.asarray(rank, jnp.int32)
    )
    jax.block_until_ready(self._staged)
    self._fingerprints.append(fingerprint)
    self._next_rank += 1
    return fingerprint

  def finalize(self) -> tuple[Any, dict[str, Any]]:
    """Executes one fixed reduction and proves every resulting replica equal."""
    if self._staged is None:
      raise ValueError('DP gradient reduction transaction is not active')
    if self._next_rank != self._dp_size:
      raise ValueError(
          'DP gradient reduction is missing rank contributions: '
          f'{self._next_rank} != {self._dp_size}'
      )
    if self._require_distinct and len(set(self._fingerprints)) != self._dp_size:
      raise ValueError('DP rank-local gradient fingerprints are not distinct')
    reduced = self._reduce(self._staged)
    flags = np.asarray(jax.device_get(self._compare(reduced)), dtype=np.bool_)
    if flags.size != self._dp_size or not bool(np.all(flags)):
      raise ValueError(
          'fixed DP gradient reduction produced unequal replicas: '
          f'flags={flags.astype(np.int32).tolist()}'
      )
    report = {
        'dp_size': self._dp_size,
        'rank_contributions': self._next_rank,
        'rank_local_fingerprints': tuple(self._fingerprints),
        'rank_local_fingerprints_distinct': (
            len(set(self._fingerprints)) == self._dp_size
        ),
        'reduction_transactions': 1,
        'reduction_rounds': fixed_dp_collective_count(self._dp_size),
        'replica_check_flags': int(flags.size),
        'post_reduction_replicas_exact': True,
      }
    self._staged = None
    self._next_rank = 0
    self._fingerprints = []
    return reduced, report


def assert_dp_replicas_equal(
    value: Any, *, dp_size: int, label: str
) -> dict[str, int]:
  """Fails if a host-visible leading DP axis contains unequal replicas."""
  rows = np.asarray(jax.device_get(value))
  if rows.ndim < 1 or rows.shape[0] != dp_size:
    raise ValueError(
        f'{label} must expose a leading DP axis of size {dp_size}: '
        f'shape={rows.shape}'
    )
  mismatches = [
      rank
      for rank in range(1, dp_size)
      if not np.array_equal(rows[0], rows[rank])
  ]
  if mismatches:
    raise ValueError(
        f'{label} differs from rank 0 on DP ranks {mismatches[:8]}; '
        f'total={len(mismatches)}'
    )
  return {'dp_replicas': dp_size, 'mismatched_replicas': 0}


def isolate_dp_rank_cotangent(
    value: Any, *, rank: int, dp_size: int
) -> Any:
  """Keeps one row of a DP-group cotangent and zeros every peer row."""
  if dp_size <= 1:
    raise ValueError(f'dp_size must be greater than one, got {dp_size}')
  if rank not in range(dp_size):
    raise ValueError(f'rank {rank} is outside [0, {dp_size})')
  if getattr(value, 'ndim', 0) < 1 or value.shape[0] != dp_size:
    raise ValueError(
        'DP-group cotangent must have a leading rank axis: '
        f'shape={getattr(value, "shape", None)} dp_size={dp_size}'
    )
  mask = jnp.arange(dp_size, dtype=jnp.int32) == rank
  mask = mask.reshape((dp_size,) + (1,) * (value.ndim - 1))
  return jnp.where(mask, value, jnp.zeros_like(value))
