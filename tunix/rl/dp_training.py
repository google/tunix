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

import collections
import dataclasses
import hashlib
import math
import os
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


def dp_collective_reduce_mode() -> str:
  """Returns the validated CANON_DP_COLLECTIVE_REDUCE selector value."""
  value = os.environ.get('CANON_DP_COLLECTIVE_REDUCE', '')
  if value not in ('', '0', '1', 'tree'):
    raise ValueError(
        f'CANON_DP_COLLECTIVE_REDUCE must be unset/0/1/tree, got {value!r}'
    )
  return '' if value == '0' else value


def psum_dp_collective(
    local_value: Any, *, dp_size: int, axis_name: str = 'dp'
) -> Any:
  """Sums DP rank contributions with one native psum per leaf.

  Floating leaves below 32-bit precision accumulate in float32 and cast back
  to their original dtype; float32 and wider leaves reduce in their own
  dtype. ``dp_size`` is accepted for reducer-signature parity only; the
  native collective derives the participant set from the mapped axis.
  """
  del dp_size

  def reduce_leaf(leaf):
    original_dtype = leaf.dtype
    if (
        jnp.issubdtype(original_dtype, jnp.floating)
        and jnp.dtype(original_dtype).itemsize < 4
    ):
      leaf = leaf.astype(jnp.float32)
    return jax.lax.psum(leaf, axis_name).astype(original_dtype)

  return jax.tree.map(reduce_leaf, local_value)


def gathered_tree_dp_collective(
    local_value: Any, *, dp_size: int, axis_name: str = 'dp'
) -> Any:
  """Sums DP rank contributions by all-gather then the registered fixed tree.

  Every rank gathers all ``dp_size`` contributions and adds them locally with
  ``fixed_dp_sum``, whose explicit rank pairing and operand optimization
  barriers pin the same binary association order as the registered ppermute
  tree, so every replica computes one identical fixed-order sum.
  """
  gathered = jax.tree.map(
      lambda leaf: jax.lax.all_gather(leaf, axis_name, axis=0), local_value
  )
  rank_values = tuple(
      jax.tree.map(
          lambda table, rank=rank: jax.lax.index_in_dim(
              table, rank, axis=0, keepdims=False
          ),
          gathered,
      )
      for rank in range(dp_size)
  )
  return fixed_dp_sum(rank_values)


def select_dp_collective(mode: str) -> Any:
  """Returns the DP-sum callable registered for one validated reduce mode."""
  if mode == '1':
    return psum_dp_collective
  if mode == 'tree':
    return gathered_tree_dp_collective
  if mode not in ('', '0'):
    raise ValueError(f'unknown DP collective reduce mode: {mode!r}')
  return fixed_dp_collective


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


def _gradient_finite_flags(tree: Any) -> jax.Array:
  """Returns one finite bit per leaf without copying gradient payloads."""
  return jnp.stack(
      tuple(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree))
  )


def _gradient_diagnostics(tree: Any) -> tuple[jax.Array, jax.Array]:
  """Computes the existing signature and finite bits in one dispatch."""
  return _gradient_signature(tree), _gradient_finite_flags(tree)


def _signature_sha256(signature: Any) -> str:
  value = np.ascontiguousarray(jax.device_get(signature))
  return hashlib.sha256(value.view(np.uint8)).hexdigest()


# P70.4 receipt-lightening schedule constants. The hybrid compare keeps the
# legacy full elementwise replica compare on the first
# ``HYBRID_FULL_COMPARE_GROUPS`` groups of every reducer lifetime (one
# reducer serves exactly one optimizer update in the production adapter);
# the distinct-fingerprint schedule keeps the full per-rank signature
# computation on the first group of every update and on every group of the
# first ``DISTINCT_FINGERPRINT_WARMUP_UPDATES`` updates of the process.
HYBRID_FULL_COMPARE_GROUPS = 2
DISTINCT_FINGERPRINT_WARMUP_UPDATES = 3
_SKIPPED_FINGERPRINT = 'skipped:receipt-schedule'
_CHECKSUM_SALT_STRIDE = 0x9E3779B9


def dp_compare_mode() -> str:
  """Returns the validated CANON_DP_COMPARE_MODE selector value."""
  value = os.environ.get('CANON_DP_COMPARE_MODE', '')
  if value not in ('', '0', 'full', 'fingerprint-hybrid'):
    raise ValueError(
        'CANON_DP_COMPARE_MODE must be unset/0/full/fingerprint-hybrid, '
        f'got {value!r}'
    )
  return 'fingerprint-hybrid' if value == 'fingerprint-hybrid' else 'full'


def dp_distinct_schedule_mode() -> str:
  """Returns the validated CANON_DP_DISTINCT_SCHEDULE selector value."""
  value = os.environ.get('CANON_DP_DISTINCT_SCHEDULE', '')
  if value not in ('', '0', 'every-group', 'first-group-warmup'):
    raise ValueError(
        'CANON_DP_DISTINCT_SCHEDULE must be unset/0/every-group/'
        f'first-group-warmup, got {value!r}'
    )
  return 'first-group-warmup' if value == 'first-group-warmup' else 'every-group'


def dp_finite_fetch_mode() -> str:
  """Returns the validated CANON_DP_FINITE_FETCH selector value."""
  value = os.environ.get('CANON_DP_FINITE_FETCH', '')
  if value not in ('', '0', 'sync', 'batched-commit'):
    raise ValueError(
        'CANON_DP_FINITE_FETCH must be unset/0/sync/batched-commit, '
        f'got {value!r}'
    )
  return 'batched-commit' if value == 'batched-commit' else 'sync'


_receipt_schedule_update_counter = [0]


def _next_receipt_schedule_update_index() -> int:
  value = _receipt_schedule_update_counter[0]
  _receipt_schedule_update_counter[0] = value + 1
  return value


def reset_receipt_schedule_update_counter_for_tests() -> None:
  """Rewinds the process-level update counter; test isolation only."""
  _receipt_schedule_update_counter[0] = 0


def _leaf_checksum_words(leaf: jax.Array) -> jax.Array:
  """Reinterprets one leaf's exact payload bits as uint32 lanes.

  Sub-32-bit dtypes zero-extend after the bitcast, so the mapping from leaf
  bytes to lanes stays a bijection: any single changed payload bit changes
  exactly one lane. No floating-point value math is performed anywhere.
  """
  flat = jnp.ravel(leaf)
  itemsize = jnp.dtype(leaf.dtype).itemsize
  if itemsize == 8:
    return jnp.ravel(jax.lax.bitcast_convert_type(flat, jnp.uint32))
  if itemsize == 4:
    return jax.lax.bitcast_convert_type(flat, jnp.uint32)
  if itemsize == 2:
    return jax.lax.bitcast_convert_type(flat, jnp.uint16).astype(jnp.uint32)
  if itemsize == 1:
    return jax.lax.bitcast_convert_type(flat, jnp.uint8).astype(jnp.uint32)
  raise ValueError(f'unsupported DP checksum leaf dtype: {leaf.dtype}')


def _leaf_dual_checksum(leaf: jax.Array) -> jax.Array:
  """Two structurally independent uint32 checksums over one leaf's bits.

  Mixer A (rot-add): every lane is XORed with a Weyl position salt
  (``index * 0x9E3779B9 mod 2**32``), rotated left by a position-derived
  amount in [1, 31], then summed with uint32 wraparound. Carry propagation
  makes the sum mix bits across lane positions; the position salt and
  rotation make it order-sensitive, unlike a naive lane sum.

  Mixer B (rot-xor fold): every lane ADDS the same Weyl salt (wraparound),
  is rotated by a different position-derived schedule, and the lanes are
  folded with carry-free XOR. No hash-state multiplication is used
  (product-free); the mixer lives in GF(2), algebraically independent from
  mixer A's modular-addition group, so a crafted compensating perturbation
  that preserves A must simultaneously solve an unrelated XOR system to
  also preserve B.

  Both mixers are deterministic, associative-reduction safe (wraparound add
  and XOR are exactly associative and commutative), and read every payload
  bit exactly once. An all-zero and an empty leaf checksum to fixed values.
  """
  words = _leaf_checksum_words(leaf)
  if words.shape[0] == 0:
    return jnp.zeros((2,), jnp.uint32)
  index = jnp.arange(words.shape[0], dtype=jnp.uint32)
  salt = index * jnp.uint32(_CHECKSUM_SALT_STRIDE)
  rotation_a = (index % jnp.uint32(31)) + jnp.uint32(1)
  mixed_a = words ^ salt
  rotated_a = (mixed_a << rotation_a) | (
      mixed_a >> (jnp.uint32(32) - rotation_a)
  )
  checksum_a = jnp.sum(rotated_a, dtype=jnp.uint32)
  rotation_b = (
      (index * jnp.uint32(7) + jnp.uint32(3)) % jnp.uint32(31)
  ) + jnp.uint32(1)
  mixed_b = words + salt
  rotated_b = (mixed_b << rotation_b) | (
      mixed_b >> (jnp.uint32(32) - rotation_b)
  )
  checksum_b = jax.lax.reduce(
      rotated_b,
      jnp.uint32(0),
      lambda accumulator, lane: jax.lax.bitwise_xor(accumulator, lane),
      (0,),
  )
  return jnp.stack((checksum_a, checksum_b))


def _tree_dual_checksums(tree: Any) -> jax.Array:
  """Stacks the per-leaf dual checksums into one ``(n_leaf, 2)`` vector."""
  return jnp.stack(
      tuple(_leaf_dual_checksum(leaf) for leaf in jax.tree.leaves(tree))
  )


# P70.5 reducer program cache. The production adapter constructs one
# FixedDPRankGradientReducer per optimizer update, and every construction
# used to build fresh ``jax.jit`` wrappers, so the host re-traced every
# reducer-scope program on every update (~2.2 s/update for the legacy
# diagnostics/reduce/compare closures plus ~1.8 s/update for the P70.4
# fingerprint program; P70.B attribution, cycle g00). Every one of those
# programs depends only on static construction identity — template
# structure, per-leaf shape/dtype/sharding, mesh, DP geometry, and the
# validated mode selectors — so constructions with an identical identity
# share one program bundle. Bundle closures never capture device arrays
# (the template is reduced to ``jax.ShapeDtypeStruct`` leaves before
# tracing) and cache keys never include array values, so a cache hit
# returns byte-for-byte the same traced programs a fresh build would
# produce: zero numerical change, host re-trace cost removed.
_REDUCER_PROGRAM_CACHE_LIMIT = 4
_reducer_program_cache: collections.OrderedDict = collections.OrderedDict()
_reducer_program_cache_stats = {'hits': 0, 'misses': 0, 'uncacheable': 0}


def reset_reducer_program_cache_for_tests() -> None:
  """Clears the process-level reducer program cache; test isolation only."""
  _reducer_program_cache.clear()
  for key in _reducer_program_cache_stats:
    _reducer_program_cache_stats[key] = 0


@dataclasses.dataclass(frozen=True)
class _ReducerPrograms:
  """One reducer construction's jitted programs and staged shardings.

  A bundle may outlive the reducer that built it (process-level cache), so
  every field must be static configuration or a jitted callable whose
  closure captures only static configuration — never a device array.
  """

  staged_shardings: Any
  initialize: Any
  write: Any
  reduce: Any
  compare: Any
  signature: Any
  finite_flags: Any
  batched_diagnostics: Any
  compare_fingerprint: Any
  batched_finite: Any


def _reducer_program_cache_key(
    template: Any,
    *,
    dp_size: int,
    dp_axis: str,
    reduce_mode: str,
    compare_mode: str,
    distinct_schedule: str,
    finite_fetch: str,
    require_distinct_fingerprints: bool,
) -> Any:
  """Builds the full static identity of one reducer program bundle.

  The key covers everything a bundle program can depend on: tree structure,
  per-leaf shape/dtype/weak-type/partition-spec/memory-kind, mesh identity
  (axis names, axis sizes, device platform+id order, axis types), DP
  geometry, and every validated mode selector read at construction time
  (``finite_fetch`` and ``require_distinct_fingerprints`` shape no traced
  program today; they stay in the key so a future program that reads them
  can never be shared across differing values). Raises when any component
  is not hashable; the caller treats that as uncacheable and builds a
  fresh bundle — programs are never shared on an ambiguous identity.
  """
  leaves, treedef = jax.tree_util.tree_flatten(template)
  mesh = leaves[0].sharding.mesh
  mesh_key = (
      tuple(mesh.axis_names),
      tuple((str(name), int(size)) for name, size in mesh.shape.items()),
      tuple(
          (str(device.platform), int(device.id))
          for device in mesh.devices.flat
      ),
      getattr(mesh, 'axis_types', None),
  )
  leaf_key = tuple(
      (
          tuple(int(dim) for dim in leaf.shape),
          jnp.dtype(leaf.dtype),
          bool(getattr(leaf, 'weak_type', False)),
          leaf.sharding.spec,
          getattr(leaf.sharding, 'memory_kind', None),
      )
      for leaf in leaves
  )
  key = (
      treedef,
      leaf_key,
      mesh_key,
      int(dp_size),
      str(dp_axis),
      str(reduce_mode),
      str(compare_mode),
      str(distinct_schedule),
      str(finite_fetch),
      bool(require_distinct_fingerprints),
  )
  hash(key)  # every component must hash, or the bundle is uncacheable
  return key


def _build_reducer_programs(
    spec_template: Any,
    *,
    dp_size: int,
    dp_axis: str,
    reduce_mode: str,
    compare_mode: str,
    distinct_schedule: str,
) -> _ReducerPrograms:
  """Traces one program bundle from a metadata-only template.

  ``spec_template`` must hold ``jax.ShapeDtypeStruct`` leaves carrying the
  original ``NamedSharding``s: the bundle can outlive the constructing
  reducer inside the process-level cache, so its closures capture shapes,
  dtypes, and shardings — never device buffers.
  """
  mesh = jax.tree.leaves(spec_template)[0].sharding.mesh
  base_specs = jax.tree.map(
      lambda leaf: leaf.sharding.spec, spec_template
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
        spec_template,
    )

  def write(staged, contribution, rank):
    return jax.tree.map(
        lambda table, value: jax.lax.dynamic_update_index_in_dim(
            table, jnp.expand_dims(value, 0), rank, axis=0
        ),
        staged,
        contribution,
    )

  reduce_collective = select_dp_collective(reduce_mode)

  def reduce_local(local_staged):
    local_value = jax.tree.map(
        lambda value: jnp.squeeze(value, axis=0), local_staged
    )
    return reduce_collective(
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

  signature_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(dp_axis, None)
  )
  compare_fingerprint = None
  if compare_mode == 'fingerprint-hybrid':

    def compare_fingerprint_local(local_tree):
      fingerprints = _tree_dual_checksums(local_tree)
      peer_fingerprints = jax.lax.ppermute(
          fingerprints, axis_name=dp_axis, perm=permutation
      )
      matches = jnp.all(fingerprints == peer_fingerprints, axis=1)
      return jnp.reshape(matches, (1, matches.shape[0]))

    compare_fingerprint_kwargs = {
        'mesh': mesh,
        'in_specs': (base_specs,),
        'out_specs': jax.sharding.PartitionSpec(dp_axis, None),
    }
    try:
      compare_fingerprint_mapped = jax.shard_map(
          compare_fingerprint_local,
          check_vma=False,
          **compare_fingerprint_kwargs,
      )
    except TypeError:
      compare_fingerprint_mapped = jax.shard_map(
          compare_fingerprint_local,
          check_rep=False,
          **compare_fingerprint_kwargs,
      )
    compare_fingerprint = jax.jit(compare_fingerprint_mapped)
  batched_finite = None
  if distinct_schedule == 'first-group-warmup':
    batched_finite = jax.jit(
        jax.vmap(_gradient_finite_flags),
        out_shardings=signature_sharding,
    )
  return _ReducerPrograms(
      staged_shardings=staged_shardings,
      initialize=jax.jit(initialize, out_shardings=staged_shardings),
      write=jax.jit(write, donate_argnums=(0,)),
      reduce=jax.jit(reduce_mapped, donate_argnums=(0,)),
      compare=jax.jit(compare_mapped),
      signature=jax.jit(_gradient_signature),
      finite_flags=jax.jit(_gradient_finite_flags),
      batched_diagnostics=jax.jit(
          jax.vmap(_gradient_diagnostics),
          out_shardings=(signature_sharding, signature_sharding),
      ),
      compare_fingerprint=compare_fingerprint,
      batched_finite=batched_finite,
  )


def _reducer_programs_for(
    template: Any,
    *,
    dp_size: int,
    dp_axis: str,
    reduce_mode: str,
    compare_mode: str,
    distinct_schedule: str,
    finite_fetch: str,
    require_distinct_fingerprints: bool,
) -> _ReducerPrograms:
  """Returns the cached program bundle for one identity, building on miss."""
  try:
    key = _reducer_program_cache_key(
        template,
        dp_size=dp_size,
        dp_axis=dp_axis,
        reduce_mode=reduce_mode,
        compare_mode=compare_mode,
        distinct_schedule=distinct_schedule,
        finite_fetch=finite_fetch,
        require_distinct_fingerprints=require_distinct_fingerprints,
    )
  except (TypeError, ValueError, AttributeError):
    # Fail closed: an identity this key cannot express is never shared.
    key = None
    _reducer_program_cache_stats['uncacheable'] += 1
  if key is not None:
    cached = _reducer_program_cache.get(key)
    if cached is not None:
      _reducer_program_cache.move_to_end(key)
      _reducer_program_cache_stats['hits'] += 1
      return cached
  spec_template = jax.tree.map(
      lambda leaf: jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype, sharding=leaf.sharding
      ),
      template,
  )
  programs = _build_reducer_programs(
      spec_template,
      dp_size=dp_size,
      dp_axis=dp_axis,
      reduce_mode=reduce_mode,
      compare_mode=compare_mode,
      distinct_schedule=distinct_schedule,
  )
  if key is not None:
    _reducer_program_cache_stats['misses'] += 1
    _reducer_program_cache[key] = programs
    while len(_reducer_program_cache) > _REDUCER_PROGRAM_CACHE_LIMIT:
      _reducer_program_cache.popitem(last=False)
  return programs


class FixedDPRankGradientReducer:
  """Stages one contribution per DP rank and reduces it with a fixed tree.

  The leading staging axis is physically partitioned over the explicitly named
  DP mesh axis. Therefore a logical ``[dp, ...parameter_shape]`` table stores
  only one rank contribution per DP replica. Finalization executes one
  reduce-and-broadcast transaction with the registered collective-permute
  schedule and returns the original DP-replicated, TP-sharded gradient tree.
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
    admitted_leaf_types = (jax.Array, jax.ShapeDtypeStruct)
    if not leaves or any(
        not isinstance(leaf, admitted_leaf_types) for leaf in leaves
    ):
      raise ValueError(
          'DP gradient reducer requires a nonempty JAX array/spec tree'
      )
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

    self._dp_size = dp_size
    self._dp_axis = dp_axis
    self._require_distinct = require_distinct_fingerprints
    reduce_mode = dp_collective_reduce_mode()
    # P70.4 receipt-lightening wiring. Every mode defaults to the legacy
    # behavior; with all three flags unset the bundle below only records
    # the legacy mode names and never builds a new program.
    self._compare_mode = dp_compare_mode()
    self._distinct_schedule = dp_distinct_schedule_mode()
    self._finite_fetch = dp_finite_fetch_mode()
    # P70.5: one construction per optimizer update used to re-trace every
    # reducer-scope jax.jit program; constructions with an identical static
    # identity now share one cached bundle (see _reducer_programs_for).
    programs = _reducer_programs_for(
        template,
        dp_size=dp_size,
        dp_axis=dp_axis,
        reduce_mode=reduce_mode,
        compare_mode=self._compare_mode,
        distinct_schedule=self._distinct_schedule,
        finite_fetch=self._finite_fetch,
        require_distinct_fingerprints=require_distinct_fingerprints,
    )
    self._programs = programs
    self._initialize = programs.initialize
    self._write = programs.write
    self._reduce = programs.reduce
    self._compare = programs.compare
    self._signature = programs.signature
    self._finite_flags = programs.finite_flags
    self._batched_diagnostics = programs.batched_diagnostics
    self._compare_fingerprint = programs.compare_fingerprint
    self._batched_finite = programs.batched_finite
    self._template_structure = jax.tree.structure(template)
    self._leaf_paths = tuple(
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(template)[0]
    )
    self._staged_metadata = tuple(
        (
            (dp_size,) + tuple(leaf.shape),
            leaf.dtype,
            sharding,
        )
        for leaf, sharding in zip(
            leaves, jax.tree.leaves(programs.staged_shardings), strict=True
        )
    )
    self._staged = None
    self._next_rank = 0
    self._fingerprints = []
    self._group_index = 0
    self._pending_finite_receipts = []
    self._update_index = (
        _next_receipt_schedule_update_index()
        if self._distinct_schedule == 'first-group-warmup'
        else None
    )

  def _distinct_fingerprint_scheduled(self) -> bool:
    """True when this group must compute real per-rank fingerprints."""
    if self._distinct_schedule != 'first-group-warmup':
      return True
    return (
        self._group_index == 0
        or self._update_index < DISTINCT_FINGERPRINT_WARMUP_UPDATES
    )

  @property
  def pending_finite_receipt_count(self) -> int:
    """Deferred finite receipts that a commit-gate drain must validate."""
    return len(self._pending_finite_receipts)

  def drain_deferred_finite_receipts(self) -> dict[str, Any]:
    """Validates every deferred isfinite receipt in one batched fetch.

    With ``CANON_DP_FINITE_FETCH=batched-commit`` the per-group synchronous
    host reads are replaced by device-resident flag vectors staged on this
    reducer. The optimizer commit MUST NOT consume any gradient this
    reducer produced until this method returns: a non-finite receipt raises
    here, before the commit, naming the group, stage, rank, and leaf path —
    the same fail-closed verdict the legacy per-group check raised, moved
    to the commit gate. The fetch concatenates every pending flag vector
    into one int32 vector and issues a single ``jax.device_get`` (the P68
    batched-evidence receipt channel pattern).
    """
    pending = self._pending_finite_receipts
    self._pending_finite_receipts = []
    if not pending:
      return {
          'deferred_finite_receipt_groups': 0,
          'deferred_finite_receipts': 0,
          'deferred_finite_flags_checked': 0,
          'all_finite': True,
      }
    vector = jnp.concatenate(
        tuple(
            jnp.ravel(flags).astype(jnp.int32) for _, _, flags in pending
        )
    )
    fetched = np.asarray(jax.device_get(vector), dtype=np.int32)
    offset = 0
    groups = set()
    for group, stage, flags in pending:
      size = math.prod(flags.shape)
      values = fetched[offset:offset + size].reshape(flags.shape)
      offset += size
      groups.add(group)
      if bool(np.all(values != 0)):
        continue
      if values.ndim == 2:
        bad = np.argwhere(values == 0)
        examples = [
            {
                'rank': int(rank),
                'leaf': int(leaf),
                'path': self._leaf_paths[int(leaf)],
            }
            for rank, leaf in bad[:16]
        ]
      else:
        bad = np.flatnonzero(values == 0)
        examples = [
            {'leaf': int(leaf), 'path': self._leaf_paths[int(leaf)]}
            for leaf in bad[:16].tolist()
        ]
      raise ValueError(
          'deferred DP gradient finite receipts failed before the '
          f'optimizer commit: group={group} stage={stage} '
          f'examples={examples} total={len(bad)}'
      )
    if offset != int(fetched.size):
      raise ValueError(
          'deferred DP gradient finite receipts lost coverage: '
          f'{offset} != {int(fetched.size)}'
      )
    return {
        'deferred_finite_receipt_groups': len(groups),
        'deferred_finite_receipts': len(pending),
        'deferred_finite_flags_checked': int(fetched.size),
        'all_finite': True,
    }

  def _check_replicas_elementwise(self, reduced: Any) -> int:
    """Runs the legacy full elementwise replica compare; returns flag count."""
    flags = np.asarray(jax.device_get(self._compare(reduced)), dtype=np.bool_)
    if flags.size != self._dp_size or not bool(np.all(flags)):
      raise ValueError(
          'fixed DP gradient reduction produced unequal replicas: '
          f'flags={flags.astype(np.int32).tolist()}'
      )
    return int(flags.size)

  def _fingerprint_replica_matches(self, reduced: Any) -> np.ndarray:
    """Fetches the per-rank, per-leaf dual-checksum match matrix."""
    matches = np.asarray(
        jax.device_get(self._compare_fingerprint(reduced)), dtype=np.bool_
    )
    expected_shape = (self._dp_size, len(self._staged_metadata))
    if matches.shape != expected_shape:
      raise ValueError(
          'DP fingerprint replica compare changed shape: '
          f'{matches.shape} != {expected_shape}'
      )
    return matches

  def _assert_fingerprint_replicas_equal(self, reduced: Any) -> None:
    """Raises with rank/leaf/path evidence when fingerprints mismatch."""
    matches = self._fingerprint_replica_matches(reduced)
    if bool(np.all(matches)):
      return
    bad = np.argwhere(~matches)
    examples = [
        {
            'rank': int(rank),
            'leaf': int(leaf),
            'path': self._leaf_paths[int(leaf)],
        }
        for rank, leaf in bad[:16]
    ]
    raise ValueError(
        'fixed DP gradient reduction produced unequal replicas '
        f'(dual-checksum fingerprint): examples={examples} '
        f'total={len(bad)}'
    )

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
    if self._distinct_fingerprint_scheduled():
      fingerprint = _signature_sha256(self._signature(contribution))
    else:
      fingerprint = _SKIPPED_FINGERPRINT
    self._staged = self._write(
        self._staged, contribution, jnp.asarray(rank, jnp.int32)
    )
    jax.block_until_ready(self._staged)
    self._fingerprints.append(fingerprint)
    self._next_rank += 1
    return fingerprint

  def _finalize_staged(
      self,
      staged: Any,
      fingerprints: Sequence[str],
      *,
      staging_mode: str,
      fingerprints_computed: bool = True,
  ) -> tuple[Any, dict[str, Any]]:
    """Reduces one validated rank table and emits common evidence."""
    unique_fingerprints = len(set(fingerprints))
    if (
        self._require_distinct
        and fingerprints_computed
        and unique_fingerprints != self._dp_size
    ):
      raise ValueError('DP rank-local gradient fingerprints are not distinct')
    reduced = self._reduce(staged)
    if self._finite_fetch == 'batched-commit':
      # P70.4 knife 3: keep the finite bits on device; the commit-gate
      # drain validates them in one batched fetch before any optimizer
      # commit may consume this gradient.
      self._pending_finite_receipts.append(
          (self._group_index, 'reduced', self._finite_flags(reduced))
      )
      finite_flag_count = len(self._staged_metadata)
      post_reduction_all_finite = 'deferred-commit'
    else:
      finite_flags = np.asarray(
          jax.device_get(self._finite_flags(reduced)), dtype=np.bool_
      )
      if finite_flags.shape != (len(self._staged_metadata),):
        raise ValueError(
            'reduced DP gradient finite flags changed shape: '
            f'{finite_flags.shape} != {(len(self._staged_metadata),)}'
        )
      if not bool(np.all(finite_flags)):
        bad_leaves = np.flatnonzero(~finite_flags).astype(np.int32).tolist()
        examples = [
            {'leaf': index, 'path': self._leaf_paths[index]}
            for index in bad_leaves[:16]
        ]
        raise ValueError(
            'fixed DP gradient reduction produced non-finite values: '
            f'examples={examples} total={len(bad_leaves)}'
        )
      finite_flag_count = int(finite_flags.size)
      post_reduction_all_finite = True
    if self._compare_mode == 'fingerprint-hybrid':
      if self._group_index < HYBRID_FULL_COMPARE_GROUPS:
        # Full-compare group: the exact elementwise compare stays the
        # verdict; the fingerprint program runs alongside it as a per-update
        # self-check of the lightened instrument against ground truth.
        fingerprint_matches = self._fingerprint_replica_matches(reduced)
        replica_flag_count = self._check_replicas_elementwise(reduced)
        if not bool(np.all(fingerprint_matches)):
          diverged = np.flatnonzero(
              ~np.all(fingerprint_matches, axis=0)
          ).tolist()
          raise ValueError(
              'DP fingerprint replica compare diverged from the exact '
              f'elementwise compare on leaves {diverged[:16]} '
              f'total={len(diverged)}'
          )
        replica_check_mode = 'full+fingerprint-selfcheck'
      else:
        self._assert_fingerprint_replicas_equal(reduced)
        replica_flag_count = self._dp_size
        replica_check_mode = 'fingerprint'
    else:
      replica_flag_count = self._check_replicas_elementwise(reduced)
      replica_check_mode = 'full'
    report = {
        'dp_size': self._dp_size,
        'dp_axis': self._dp_axis,
        'rank_contributions': self._dp_size,
        'rank_local_fingerprints': tuple(fingerprints),
        'rank_local_fingerprint_unique_count': unique_fingerprints,
        'rank_local_fingerprint_duplicate_count': (
            self._dp_size - unique_fingerprints
        ),
        'rank_local_fingerprints_distinct': unique_fingerprints == self._dp_size,
        'rank_gradient_staging_mode': staging_mode,
        'reduction_transactions': 1,
        'reduction_rounds': fixed_dp_collective_count(self._dp_size),
        'replica_check_flags': replica_flag_count,
        'finite_leaf_flags': finite_flag_count,
        'post_reduction_all_finite': post_reduction_all_finite,
        'post_reduction_replicas_exact': True,
      }
    if (
        self._compare_mode != 'full'
        or self._distinct_schedule != 'every-group'
        or self._finite_fetch != 'sync'
    ):
      # Additive receipt keys, emitted only when a P70.4 mode is active so
      # the default-mode report stays byte-identical to the legacy schema.
      report['replica_check_mode'] = replica_check_mode
      report['rank_local_fingerprint_mode'] = (
          'computed' if fingerprints_computed else 'skipped'
      )
      report['finite_check_mode'] = (
          'deferred-commit'
          if self._finite_fetch == 'batched-commit'
          else 'sync'
      )
      report['pending_finite_receipts'] = len(self._pending_finite_receipts)
    self._group_index += 1
    return reduced, report

  def finalize(self) -> tuple[Any, dict[str, Any]]:
    """Executes one fixed reduction and proves every resulting replica equal."""
    if self._staged is None:
      raise ValueError('DP gradient reduction transaction is not active')
    if self._next_rank != self._dp_size:
      raise ValueError(
          'DP gradient reduction is missing rank contributions: '
          f'{self._next_rank} != {self._dp_size}'
      )
    reduced, report = self._finalize_staged(
        self._staged,
        self._fingerprints,
        staging_mode='serial_add',
        fingerprints_computed=_SKIPPED_FINGERPRINT not in self._fingerprints,
    )
    self._staged = None
    self._next_rank = 0
    self._fingerprints = []
    return reduced, report

  def finalize_staged(self, staged: Any) -> tuple[Any, dict[str, Any]]:
    """Consumes an already DP-sharded table of rank-local gradients."""
    if self._staged is not None:
      raise ValueError(
          'cannot consume a staged DP gradient table during an active '
          'serial transaction'
      )
    if jax.tree.structure(staged) != self._template_structure:
      raise ValueError('staged DP gradient tree does not match the template')
    leaves = jax.tree.leaves(staged)
    for index, (leaf, metadata) in enumerate(
        zip(leaves, self._staged_metadata, strict=True)
    ):
      expected_shape, expected_dtype, expected_sharding = metadata
      if not isinstance(leaf, jax.Array):
        raise ValueError(
            f'staged DP gradient leaf {index} is not a JAX array'
        )
      if tuple(leaf.shape) != expected_shape or leaf.dtype != expected_dtype:
        raise ValueError(
            f'staged DP gradient leaf {index} shape/dtype changed: '
            f'{leaf.shape}/{leaf.dtype} != '
            f'{expected_shape}/{expected_dtype}'
        )
      if leaf.sharding != expected_sharding:
        raise ValueError(
            f'staged DP gradient leaf {index} sharding changed: '
            f'{leaf.sharding} != {expected_sharding}'
        )
    distinct_scheduled = self._distinct_fingerprint_scheduled()
    deferred_finite = self._finite_fetch == 'batched-commit'
    signatures = None
    staged_finite = None
    if distinct_scheduled:
      signatures_device, staged_finite_device = self._batched_diagnostics(
          staged
      )
      if deferred_finite:
        self._pending_finite_receipts.append(
            (self._group_index, 'staged', staged_finite_device)
        )
        signatures = jax.device_get(signatures_device)
      else:
        signatures, staged_finite = jax.device_get(
            (signatures_device, staged_finite_device)
        )
      signatures = np.asarray(signatures, dtype=np.float32)
      if signatures.shape != (self._dp_size, 5):
        raise ValueError(
            'staged DP gradient signatures changed shape: '
            f'{signatures.shape} != {(self._dp_size, 5)}'
        )
    else:
      # P70.4 knife 2: an unscheduled group computes no per-rank signature;
      # only the finite bits are produced, on the same vmapped layout.
      staged_finite_device = self._batched_finite(staged)
      if deferred_finite:
        self._pending_finite_receipts.append(
            (self._group_index, 'staged', staged_finite_device)
        )
      else:
        staged_finite = jax.device_get(staged_finite_device)
    if staged_finite is not None:
      staged_finite = np.asarray(staged_finite, dtype=np.bool_)
      expected_finite_shape = (self._dp_size, len(self._staged_metadata))
      if staged_finite.shape != expected_finite_shape:
        raise ValueError(
            'staged DP gradient finite flags changed shape: '
            f'{staged_finite.shape} != {expected_finite_shape}'
        )
      if not bool(np.all(staged_finite)):
        bad = np.argwhere(~staged_finite)
        examples = [
            {
                'rank': int(rank),
                'leaf': int(leaf),
                'path': self._leaf_paths[int(leaf)],
            }
            for rank, leaf in bad[:16]
        ]
        raise ValueError(
            'staged DP gradient contains non-finite values: '
            f'examples={examples} total={len(bad)}'
        )
    if signatures is not None:
      fingerprints = tuple(
          _signature_sha256(signatures[rank])
          for rank in range(self._dp_size)
      )
    else:
      fingerprints = (_SKIPPED_FINGERPRINT,) * self._dp_size
    return self._finalize_staged(
        staged,
        fingerprints,
        staging_mode='parallel_table',
        fingerprints_computed=signatures is not None,
    )


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
