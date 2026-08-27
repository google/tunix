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

"""P70.4 regression suite: receipt lightening behind three approved selectors.

CL 6dbf8557 "Lighten the DP gradient receipts behind three approved
selectors" added three independent default-off enum flags, each an
approved R1 weakening with a mandatory kill-test:

  knife 1  CANON_DP_COMPARE_MODE=fingerprint-hybrid
           full elementwise replica compare on the first
           HYBRID_FULL_COMPARE_GROUPS groups of every reducer lifetime,
           per-leaf DUAL independent uint32 checksums (rot-add + rot-xor)
           ppermuted instead of the full tree on the rest.
  knife 2  CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
           per-rank distinct-fingerprint signatures computed on the first
           group of each update and on all groups of the first
           DISTINCT_FINGERPRINT_WARMUP_UPDATES updates; skipped otherwise.
  knife 3  CANON_DP_FINITE_FETCH=batched-commit
           isfinite bits stay on device per group; one batched int32
           fetch validates ALL of them at the commit gate, before any
           optimizer commit (fail-closed semantics unchanged).

Kill-test coverage (numerics-admission R1): (a) a single-bit flip is
flagged AND attributed to its leaf, and a crafted compensating swap that
fools a naive lane sum is caught by both mixers; (b) injected non-finites
are rejected BEFORE the simulated optimizer commit; (c) the hybrid compare
and distinct warm-up schedules are asserted with call counting; (d) with
every flag unset the legacy dispatch is byte-identical (report schema,
values, and — against the unpatched reference — jaxpr equality) and the
adapter refuses the selectors under deterministic_repeat.

Fixtures and test bodies are the surviving P70.4 standalone,
tasks/three_lane_system/scripts/p70_4_killtests.py, ported in-repo with
the module under test pinned to the in-tree ``tunix.rl.dp_training``
(the standalone keeps its draft-tree/bench-registry CLI roles).

Reference-parity cases compare against the UNPATCHED dp_training. The
extraction is not checked in; produce it with

  git show 367abfca:tunix/rl/dp_training.py > /tmp/reference_dp_training.py

and point ``P70_REFERENCE_DP_TRAINING`` at the file. Without it those
cases SKIP; everything else runs self-contained.
"""

import ast
import contextlib
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if '--xla_force_host_platform_device_count' not in os.environ.get(
    'XLA_FLAGS', ''
):
  os.environ['XLA_FLAGS'] = (
      os.environ.get('XLA_FLAGS', '')
      + ' --xla_force_host_platform_device_count=16'
  ).strip()

import importlib.util
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tunix.rl import canonical_qwen3_adapter
from tunix.rl import dp_training

# Pre-P70.4 worktree state (last dp_training-touching commit before
# 6dbf8557); the reference extraction one-liner in the module docstring
# uses this SHA.
REFERENCE_DP_TRAINING_SHA = '367abfca'


_P70_4_FLAGS = (
    'CANON_DP_COMPARE_MODE',
    'CANON_DP_DISTINCT_SCHEDULE',
    'CANON_DP_FINITE_FETCH',
)


def _dpt():
  """The module under test is the in-tree landed module."""
  return dp_training


_REF = None


def _ref():
  """Loads the unpatched reference module, or SKIPs the calling test."""
  global _REF
  if _REF is not None:
    return _REF
  path = os.environ.get('P70_REFERENCE_DP_TRAINING', '')
  if not path or not os.path.exists(path):
    pytest.skip(
        'unpatched dp_training extraction absent; produce it with '
        f'`git show {REFERENCE_DP_TRAINING_SHA}:tunix/rl/dp_training.py '
        '> <file>` and set P70_REFERENCE_DP_TRAINING=<file>'
    )
  spec = importlib.util.spec_from_file_location(
      'p70_4_dp_training_reference', path
  )
  module = importlib.util.module_from_spec(spec)
  # dataclass(slots=True) resolves its defining module via sys.modules.
  sys.modules['p70_4_dp_training_reference'] = module
  spec.loader.exec_module(module)
  _REF = module
  return _REF


@contextlib.contextmanager
def _flags(**values):
  saved = {key: os.environ.get(key) for key in _P70_4_FLAGS}
  try:
    for key in _P70_4_FLAGS:
      os.environ.pop(key, None)
    for key, value in values.items():
      os.environ[key] = value
    yield
  finally:
    for key, value in saved.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


def _mesh(dp, tp=1):
  devices = jax.devices()
  if len(devices) < dp * tp:
    pytest.skip(f'requires at least {dp * tp} forced CPU devices')
  return Mesh(np.asarray(devices[: dp * tp]).reshape(dp, tp), ('dp', 'tp'))


def _template_arrays():
  return {
      'wa': (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0) / 7.0,
      'wb': np.linspace(-2.0, 2.0, 5).astype(ml_dtypes.bfloat16),
      'wc': np.asarray([[1.5, -0.5], [3.25, 0.125]], dtype=np.float32),
  }


# jax dict trees flatten in sorted key order.
_LEAF_ORDER = ('wa', 'wb', 'wc')


def _replicated_tree(mesh, arrays):
  sharding = NamedSharding(mesh, P())
  return {
      key: jax.device_put(jnp.asarray(value), sharding)
      for key, value in arrays.items()
  }


def _rank_contributions(dp):
  base = _template_arrays()
  contributions = []
  for rank in range(dp):
    contributions.append({
        'wa': base['wa'] + np.float32(rank) * np.float32(0.25),
        'wb': (
            np.asarray(base['wb'], np.float32) + np.float32(rank)
        ).astype(ml_dtypes.bfloat16),
        'wc': base['wc'] * np.float32(rank + 1),
    })
  return contributions


def _staged_table(mesh, contributions, mutate=None):
  staged = {}
  for key in contributions[0]:
    stacked = np.stack([np.array(c[key], copy=True) for c in contributions])
    if mutate is not None:
      stacked = mutate(key, stacked)
    staged[key] = jax.device_put(
        jnp.asarray(stacked), NamedSharding(mesh, P('dp'))
    )
  return staged


def _make_reducer(module, mesh, dp, require_distinct=False):
  template = _replicated_tree(mesh, _template_arrays())
  return module.FixedDPRankGradientReducer(
      template,
      dp_size=dp,
      dp_axis='dp',
      require_distinct_fingerprints=require_distinct,
  )


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _divergent_replicated(mesh, base_arrays, victim_key, victim_rank, mutate):
  """Builds a physically divergent tree under a replicated sharding.

  Every leaf claims full replication; the victim leaf's per-device buffers
  differ on the victim DP rank. This is exactly what a replica-divergence
  fault looks like on hardware, expressed with
  jax.make_array_from_single_device_arrays (SYNTHETIC-MUTATION fixture).
  """
  tp = mesh.devices.shape[1]
  sharding = NamedSharding(mesh, P())
  tree = {}
  for key, value in base_arrays.items():
    if key != victim_key:
      tree[key] = jax.device_put(jnp.asarray(value), sharding)
      continue
    buffers = []
    for flat_index, device in enumerate(mesh.devices.flatten()):
      leaf = np.array(value, copy=True)
      if flat_index // tp == victim_rank:
        leaf = mutate(leaf)
      buffers.append(jax.device_put(jnp.asarray(leaf), device))
    tree[key] = jax.make_array_from_single_device_arrays(
        value.shape, sharding, buffers
    )
  return tree


def _clean_reduced(module, mesh, dp):
  reducer = _make_reducer(module, mesh, dp)
  reduced, report = reducer.finalize_staged(
      _staged_table(mesh, _rank_contributions(dp))
  )
  return reducer, reduced, report


def _flip_bit_f32(flat_index, bit):
  def mutate(leaf):
    view = leaf.view(np.uint32).reshape(-1)
    view[flat_index] ^= np.uint32(1 << bit)
    return leaf

  return mutate


def _flip_bit_bf16(flat_index, bit):
  def mutate(leaf):
    view = leaf.view(np.uint16).reshape(-1)
    view[flat_index] ^= np.uint16(1 << bit)
    return leaf

  return mutate


def _swap_elements(index_a, index_b):
  def mutate(leaf):
    flat = leaf.reshape(-1)
    flat[index_a], flat[index_b] = (
        np.copy(flat[index_b]),
        np.copy(flat[index_a]),
    )
    return leaf

  return mutate


def _lane_pair_compensation(index_a, index_b):
  def mutate(leaf):
    view = leaf.view(np.uint32).reshape(-1)
    view[index_a] += np.uint32(1)
    view[index_b] -= np.uint32(1)
    return leaf

  return mutate


def _naive_lane_sum(array):
  lanes = np.ascontiguousarray(array)
  if lanes.dtype.itemsize == 2:
    words = lanes.view(np.uint16).astype(np.uint64)
  else:
    words = lanes.view(np.uint32).astype(np.uint64)
  return int(np.sum(words)) % (2**32)


def _dual_checksum_host(module, array):
  return np.asarray(
      jax.jit(module._leaf_dual_checksum)(jnp.asarray(array))
  )


# ---------------------------------------------------------------------------
# Flag readers
# ---------------------------------------------------------------------------


def test_flag_readers_default_to_legacy():
  module = _dpt()
  for value in (None, '', '0'):
    with _flags(**({} if value is None else {
        'CANON_DP_COMPARE_MODE': value,
        'CANON_DP_DISTINCT_SCHEDULE': value,
        'CANON_DP_FINITE_FETCH': value,
    })):
      assert module.dp_compare_mode() == 'full'
      assert module.dp_distinct_schedule_mode() == 'every-group'
      assert module.dp_finite_fetch_mode() == 'sync'
  with _flags(
      CANON_DP_COMPARE_MODE='full',
      CANON_DP_DISTINCT_SCHEDULE='every-group',
      CANON_DP_FINITE_FETCH='sync',
  ):
    assert module.dp_compare_mode() == 'full'
    assert module.dp_distinct_schedule_mode() == 'every-group'
    assert module.dp_finite_fetch_mode() == 'sync'


def test_flag_readers_accept_treatment_values():
  module = _dpt()
  with _flags(
      CANON_DP_COMPARE_MODE='fingerprint-hybrid',
      CANON_DP_DISTINCT_SCHEDULE='first-group-warmup',
      CANON_DP_FINITE_FETCH='batched-commit',
  ):
    assert module.dp_compare_mode() == 'fingerprint-hybrid'
    assert module.dp_distinct_schedule_mode() == 'first-group-warmup'
    assert module.dp_finite_fetch_mode() == 'batched-commit'


def test_flag_readers_reject_unknown_values():
  module = _dpt()
  with _flags(CANON_DP_COMPARE_MODE='hybrid'):
    with pytest.raises(ValueError, match='CANON_DP_COMPARE_MODE'):
      module.dp_compare_mode()
  with _flags(CANON_DP_DISTINCT_SCHEDULE='sometimes'):
    with pytest.raises(ValueError, match='CANON_DP_DISTINCT_SCHEDULE'):
      module.dp_distinct_schedule_mode()
  with _flags(CANON_DP_FINITE_FETCH='async'):
    with pytest.raises(ValueError, match='CANON_DP_FINITE_FETCH'):
      module.dp_finite_fetch_mode()


def test_update_counter_reset_helper():
  module = _dpt()
  mesh = _mesh(4)
  module.reset_receipt_schedule_update_counter_for_tests()
  with _flags(CANON_DP_DISTINCT_SCHEDULE='first-group-warmup'):
    first = _make_reducer(module, mesh, 4)
    second = _make_reducer(module, mesh, 4)
    assert (first._update_index, second._update_index) == (0, 1)
    module.reset_receipt_schedule_update_counter_for_tests()
    third = _make_reducer(module, mesh, 4)
    assert third._update_index == 0
  with _flags():
    legacy = _make_reducer(module, mesh, 4)
    assert legacy._update_index is None


# ---------------------------------------------------------------------------
# Kill-test (d): flag-off byte-identity against the unpatched module
# ---------------------------------------------------------------------------


def test_default_mode_matches_reference_parallel_table():
  module, reference = _dpt(), _ref()
  mesh = _mesh(4)
  with _flags():
    _, reduced, report = _clean_reduced(module, mesh, 4)
    _, reference_reduced, reference_report = _clean_reduced(
        reference, mesh, 4
    )
  assert _tree_bytes(reduced) == _tree_bytes(reference_reduced)
  assert report == reference_report


def test_default_mode_matches_reference_serial():
  module, reference = _dpt(), _ref()
  mesh = _mesh(4)
  contributions = _rank_contributions(4)
  results = []
  with _flags():
    for candidate in (module, reference):
      reducer = _make_reducer(candidate, mesh, 4)
      reducer.begin()
      for rank, contribution in enumerate(contributions):
        reducer.add(rank, _replicated_tree(mesh, contribution))
      results.append(reducer.finalize())
  (reduced, report), (reference_reduced, reference_report) = results
  assert _tree_bytes(reduced) == _tree_bytes(reference_reduced)
  assert report == reference_report


def test_default_mode_frozen_compare_program():
  module, reference = _dpt(), _ref()
  mesh = _mesh(4)
  with _flags():
    reducer = _make_reducer(module, mesh, 4)
    reference_reducer = _make_reducer(reference, mesh, 4)
  template = _replicated_tree(mesh, _template_arrays())
  jaxpr = str(jax.make_jaxpr(reducer._compare)(template))
  reference_jaxpr = str(
      jax.make_jaxpr(reference_reducer._compare)(template)
  )
  assert jaxpr == reference_jaxpr
  reduce_jaxpr = str(
      jax.make_jaxpr(reducer._reduce)(
          _staged_table(mesh, _rank_contributions(4))
      )
  )
  reference_reduce_jaxpr = str(
      jax.make_jaxpr(reference_reducer._reduce)(
          _staged_table(mesh, _rank_contributions(4))
      )
  )
  assert reduce_jaxpr == reference_reduce_jaxpr


def test_default_mode_builds_no_new_programs():
  module = _dpt()
  mesh = _mesh(4)
  with _flags():
    reducer = _make_reducer(module, mesh, 4)
  assert reducer._compare_fingerprint is None
  assert reducer._batched_finite is None
  assert reducer.pending_finite_receipt_count == 0


# ---------------------------------------------------------------------------
# Dual-checksum design properties
# ---------------------------------------------------------------------------


def test_checksum_words_are_bit_lossless():
  module = _dpt()
  words = jax.jit(module._leaf_checksum_words)
  f32 = np.asarray([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
  lanes = np.asarray(words(jnp.asarray(f32)))
  assert lanes.dtype == np.uint32
  assert lanes.tolist() == f32.view(np.uint32).tolist()
  # +0.0 and -0.0 carry different payload bits and must map to different
  # lanes: the fingerprint compare is bitwise, strictly stronger than the
  # legacy == compare on signed zeros.
  assert lanes[0] != lanes[1]
  bf16 = np.asarray([1.5, -2.25, 0.0], dtype=ml_dtypes.bfloat16)
  bf16_lanes = np.asarray(words(jnp.asarray(bf16)))
  assert bf16_lanes.dtype == np.uint32
  assert bf16_lanes.tolist() == (
      bf16.view(np.uint16).astype(np.uint32).tolist()
  )


def test_checksum_single_bit_sensitivity_exhaustive():
  module = _dpt()
  base_f32 = (np.arange(8, dtype=np.float32) - 3.0) / 7.0
  clean = _dual_checksum_host(module, base_f32)
  for flat_index in range(base_f32.size):
    for bit in range(32):
      corrupt = np.array(base_f32, copy=True)
      view = corrupt.view(np.uint32)
      view[flat_index] ^= np.uint32(1 << bit)
      dirty = _dual_checksum_host(module, corrupt)
      assert not np.array_equal(clean, dirty), (flat_index, bit)
  base_bf16 = np.linspace(-1.0, 1.0, 6).astype(ml_dtypes.bfloat16)
  clean_bf16 = _dual_checksum_host(module, base_bf16)
  for flat_index in range(base_bf16.size):
    for bit in range(16):
      corrupt = np.array(base_bf16, copy=True)
      view = corrupt.view(np.uint16)
      view[flat_index] ^= np.uint16(1 << bit)
      dirty = _dual_checksum_host(module, corrupt)
      assert not np.array_equal(clean_bf16, dirty), (flat_index, bit)


def test_checksum_catches_compensating_swap_that_fools_naive_sum():
  module = _dpt()
  base = np.asarray(
      [0.5, 1.25, -2.0, 3.5, -0.125, 7.0, 0.75, -4.5], dtype=np.float32
  )
  swapped = _swap_elements(1, 5)(np.array(base, copy=True))
  assert _naive_lane_sum(base) == _naive_lane_sum(swapped)
  assert not np.array_equal(base, swapped)
  clean = _dual_checksum_host(module, base)
  dirty = _dual_checksum_host(module, swapped)
  assert clean[0] != dirty[0]  # position-salted rot-add mixer fires
  assert clean[1] != dirty[1]  # independent rot-xor mixer fires too


def test_checksum_catches_compensating_lane_pair():
  module = _dpt()
  base = np.asarray(
      [0.5, 1.25, -2.0, 3.5, -0.125, 7.0, 0.75, -4.5], dtype=np.float32
  )
  paired = _lane_pair_compensation(0, 5)(np.array(base, copy=True))
  assert _naive_lane_sum(base) == _naive_lane_sum(paired)
  clean = _dual_checksum_host(module, base)
  dirty = _dual_checksum_host(module, paired)
  assert not np.array_equal(clean, dirty)


# ---------------------------------------------------------------------------
# Knife 1: fingerprint-hybrid compare (kill-test (a) + schedule)
# ---------------------------------------------------------------------------


def test_fingerprint_compare_clean_groups_pass():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer = _make_reducer(module, mesh, 4)
    modes = []
    for _ in range(4):
      _, report = reducer.finalize_staged(
          _staged_table(mesh, _rank_contributions(4))
      )
      assert report['post_reduction_replicas_exact'] is True
      modes.append(report['replica_check_mode'])
  assert modes == [
      'full+fingerprint-selfcheck',
      'full+fingerprint-selfcheck',
      'fingerprint',
      'fingerprint',
  ]


def test_killtest_a_single_bit_flip_flags_and_identifies_leaf():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer, reduced, _ = _clean_reduced(module, mesh, 4)
  base = {key: np.asarray(value) for key, value in reduced.items()}
  divergent = _divergent_replicated(
      mesh, base, 'wc', 1, _flip_bit_f32(1, 7)
  )
  matches = reducer._fingerprint_replica_matches(divergent)
  bad_leaves = np.flatnonzero(~np.all(matches, axis=0)).tolist()
  bad_ranks = np.flatnonzero(~np.all(matches, axis=1)).tolist()
  assert bad_leaves == [_LEAF_ORDER.index('wc')]
  assert bad_ranks == [1, 2]  # victim rank and its ppermute receiver
  with pytest.raises(ValueError) as excinfo:
    reducer._assert_fingerprint_replicas_equal(divergent)
  message = str(excinfo.value)
  assert 'dual-checksum fingerprint' in message
  assert "'leaf': 2" in message
  assert "['wc']" in message


def test_killtest_a_bf16_single_bit_flip_detected():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer, reduced, _ = _clean_reduced(module, mesh, 4)
  base = {key: np.asarray(value) for key, value in reduced.items()}
  divergent = _divergent_replicated(
      mesh, base, 'wb', 3, _flip_bit_bf16(2, 3)
  )
  with pytest.raises(ValueError, match='dual-checksum fingerprint'):
    reducer._assert_fingerprint_replicas_equal(divergent)


def test_killtest_a_compensating_swap_detected_through_compare():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer, reduced, _ = _clean_reduced(module, mesh, 4)
  base = {key: np.asarray(value) for key, value in reduced.items()}
  assert _naive_lane_sum(base['wa']) == _naive_lane_sum(
      _swap_elements(1, 5)(np.array(base['wa'], copy=True))
  )
  divergent = _divergent_replicated(
      mesh, base, 'wa', 2, _swap_elements(1, 5)
  )
  with pytest.raises(ValueError) as excinfo:
    reducer._assert_fingerprint_replicas_equal(divergent)
  assert "['wa']" in str(excinfo.value)


def test_full_compare_detection_retained_on_first_groups():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer, reduced, _ = _clean_reduced(module, mesh, 4)
  base = {key: np.asarray(value) for key, value in reduced.items()}
  divergent = _divergent_replicated(
      mesh, base, 'wc', 1, _flip_bit_f32(0, 3)
  )
  with pytest.raises(ValueError, match='produced unequal replicas'):
    reducer._check_replicas_elementwise(divergent)


class _CountingWrapper:

  def __init__(self, wrapped):
    self.wrapped = wrapped
    self.calls = 0

  def __call__(self, *args, **kwargs):
    self.calls += 1
    return self.wrapped(*args, **kwargs)


def test_killtest_c_compare_schedule_counts():
  module = _dpt()
  mesh = _mesh(4)
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer = _make_reducer(module, mesh, 4)
    full_spy = _CountingWrapper(reducer._compare)
    fingerprint_spy = _CountingWrapper(reducer._compare_fingerprint)
    reducer._compare = full_spy
    reducer._compare_fingerprint = fingerprint_spy
    for _ in range(4):
      reducer.finalize_staged(_staged_table(mesh, _rank_contributions(4)))
  assert full_spy.calls == module.HYBRID_FULL_COMPARE_GROUPS == 2
  assert fingerprint_spy.calls == 4  # self-check on full groups + verdict


def test_killtest_c_default_mode_never_calls_fingerprint():
  module = _dpt()
  mesh = _mesh(4)
  with _flags():
    reducer = _make_reducer(module, mesh, 4)
    full_spy = _CountingWrapper(reducer._compare)
    reducer._compare = full_spy
    for _ in range(3):
      reducer.finalize_staged(_staged_table(mesh, _rank_contributions(4)))
  assert full_spy.calls == 3
  assert reducer._compare_fingerprint is None


# ---------------------------------------------------------------------------
# Knife 2: distinct-fingerprint schedule (kill-test (c))
# ---------------------------------------------------------------------------


def _fingerprint_kinds(report):
  fingerprints = report['rank_local_fingerprints']
  if all(len(value) == 64 for value in fingerprints):
    return 'computed'
  if all(value == 'skipped:receipt-schedule' for value in fingerprints):
    return 'skipped'
  return 'mixed'


def test_killtest_c_distinct_schedule_warmup_then_first_group_only():
  module = _dpt()
  mesh = _mesh(4)
  module.reset_receipt_schedule_update_counter_for_tests()
  with _flags(CANON_DP_DISTINCT_SCHEDULE='first-group-warmup'):
    observed = []
    for _ in range(4):  # four simulated updates = four reducer lifetimes
      reducer = _make_reducer(module, mesh, 4)
      update_kinds = []
      for _ in range(3):
        _, report = reducer.finalize_staged(
            _staged_table(mesh, _rank_contributions(4))
        )
        update_kinds.append(
            (_fingerprint_kinds(report),
             report['rank_local_fingerprint_mode'])
        )
      observed.append(update_kinds)
  computed = ('computed', 'computed')
  skipped = ('skipped', 'skipped')
  assert observed[0] == [computed, computed, computed]
  assert observed[1] == [computed, computed, computed]
  assert observed[2] == [computed, computed, computed]
  assert observed[3] == [computed, skipped, skipped]


def test_killtest_c_distinct_schedule_serial_add_uses_placeholders():
  module = _dpt()
  mesh = _mesh(2)
  module.reset_receipt_schedule_update_counter_for_tests()
  contributions = _rank_contributions(2)
  with _flags(CANON_DP_DISTINCT_SCHEDULE='first-group-warmup'):
    for _ in range(module.DISTINCT_FINGERPRINT_WARMUP_UPDATES):
      _make_reducer(module, mesh, 2)  # consume the warm-up updates
    reducer = _make_reducer(module, mesh, 2)
    reports = []
    for _ in range(2):
      reducer.begin()
      for rank, contribution in enumerate(contributions):
        reducer.add(rank, _replicated_tree(mesh, contribution))
      _, report = reducer.finalize()
      reports.append(report)
  assert _fingerprint_kinds(reports[0]) == 'computed'
  assert _fingerprint_kinds(reports[1]) == 'skipped'


def test_distinct_check_enforced_on_scheduled_groups_only():
  module = _dpt()
  mesh = _mesh(4)
  module.reset_receipt_schedule_update_counter_for_tests()
  duplicate = [_rank_contributions(4)[0]] * 4  # identical rank payloads
  with _flags(CANON_DP_DISTINCT_SCHEDULE='first-group-warmup'):
    for _ in range(module.DISTINCT_FINGERPRINT_WARMUP_UPDATES):
      _make_reducer(module, mesh, 4, require_distinct=True)
    reducer = _make_reducer(module, mesh, 4, require_distinct=True)
    with pytest.raises(ValueError, match='not distinct'):
      reducer.finalize_staged(_staged_table(mesh, duplicate))
    # Group 0 consumed the scheduled slot even though it raised; rebuild.
    module.reset_receipt_schedule_update_counter_for_tests()
    for _ in range(module.DISTINCT_FINGERPRINT_WARMUP_UPDATES):
      _make_reducer(module, mesh, 4, require_distinct=True)
    reducer = _make_reducer(module, mesh, 4, require_distinct=True)
    reducer.finalize_staged(_staged_table(mesh, _rank_contributions(4)))
    _, report = reducer.finalize_staged(_staged_table(mesh, duplicate))
  assert report['rank_local_fingerprint_mode'] == 'skipped'
  assert report['post_reduction_replicas_exact'] is True


# ---------------------------------------------------------------------------
# Knife 3: batched-commit finite fetch (kill-test (b))
# ---------------------------------------------------------------------------


def test_killtest_b_nonfinite_rejected_at_commit_gate():
  module = _dpt()
  mesh = _mesh(4)

  def inject_inf(key, stacked):
    if key == 'wb':
      corrupted = np.asarray(stacked, np.float32)
      corrupted[2, 1] = np.inf
      return corrupted.astype(ml_dtypes.bfloat16)
    return stacked

  commit_log = []
  with _flags(CANON_DP_FINITE_FETCH='batched-commit'):
    reducer = _make_reducer(module, mesh, 4)
    # The corrupted group must NOT raise at group time (that is the
    # approved weakening) ...
    _, report = reducer.finalize_staged(
        _staged_table(mesh, _rank_contributions(4), mutate=inject_inf)
    )
    assert report['post_reduction_all_finite'] == 'deferred-commit'
    assert report['finite_check_mode'] == 'deferred-commit'
    assert reducer.pending_finite_receipt_count == 2  # staged + reduced
    # ... and the commit gate MUST reject before the optimizer commit.
    with pytest.raises(ValueError) as excinfo:
      reducer.drain_deferred_finite_receipts()
      commit_log.append('optimizer-commit')  # must never run
  message = str(excinfo.value)
  assert 'before the optimizer commit' in message
  assert 'stage=staged' in message
  assert "'rank': 2" in message
  assert "['wb']" in message
  assert not commit_log
  assert reducer.pending_finite_receipt_count == 0


def test_killtest_b_reduced_stage_inf_detected_via_serial_path():
  # +inf compares equal across replicas, so the group-time replica compare
  # stays green and the ONLY guard is the deferred finite receipt — the
  # exact protection knife 3 must not lose.
  module = _dpt()
  mesh = _mesh(2)
  contributions = _rank_contributions(2)
  for contribution in contributions:
    contribution['wc'] = np.array(contribution['wc'], copy=True)
    contribution['wc'][0, 0] = np.inf
  commit_log = []
  with _flags(CANON_DP_FINITE_FETCH='batched-commit'):
    reducer = _make_reducer(module, mesh, 2)
    reducer.begin()
    for rank, contribution in enumerate(contributions):
      reducer.add(rank, _replicated_tree(mesh, contribution))
    _, report = reducer.finalize()
    assert report['post_reduction_all_finite'] == 'deferred-commit'
    with pytest.raises(ValueError) as excinfo:
      reducer.drain_deferred_finite_receipts()
      commit_log.append('optimizer-commit')
  message = str(excinfo.value)
  assert 'stage=reduced' in message
  assert "['wc']" in message
  assert not commit_log


def test_nan_with_deferred_finite_still_fails_closed_at_group():
  # Identical-bit NaN replicas make the legacy elementwise compare report
  # unequal replicas (NaN != NaN). With knife 3 alone the group therefore
  # still aborts BEFORE the commit — earlier than the drain, with the
  # compare's verdict label. No protection gap; label hazard documented.
  module = _dpt()
  mesh = _mesh(4)

  def inject_nan(key, stacked):
    if key == 'wc':
      stacked = np.array(stacked, copy=True)
      stacked[1, 0, 0] = np.nan
    return stacked

  with _flags(CANON_DP_FINITE_FETCH='batched-commit'):
    reducer = _make_reducer(module, mesh, 4)
    with pytest.raises(ValueError, match='unequal replicas'):
      reducer.finalize_staged(
          _staged_table(mesh, _rank_contributions(4), mutate=inject_nan)
      )
    # The deferred receipts of the aborted group are still pending; a
    # commit-gate drain (which every non-raising update runs) names the
    # non-finite leaf.
    with pytest.raises(ValueError, match='before the optimizer commit'):
      reducer.drain_deferred_finite_receipts()


def test_nan_composition_hybrid_compare_caught_at_drain():
  # knives 1+3 together: on a fingerprint group, identical-bit NaN
  # replicas pass the bitwise fingerprint compare (by design), and the
  # commit-gate drain is the guard that fires.
  module = _dpt()
  mesh = _mesh(4)

  def inject_nan(key, stacked):
    if key == 'wc':
      stacked = np.array(stacked, copy=True)
      stacked[1, 0, 0] = np.nan
    return stacked

  commit_log = []
  with _flags(
      CANON_DP_COMPARE_MODE='fingerprint-hybrid',
      CANON_DP_FINITE_FETCH='batched-commit',
  ):
    reducer = _make_reducer(module, mesh, 4)
    for _ in range(module.HYBRID_FULL_COMPARE_GROUPS):
      reducer.finalize_staged(_staged_table(mesh, _rank_contributions(4)))
    _, report = reducer.finalize_staged(
        _staged_table(mesh, _rank_contributions(4), mutate=inject_nan)
    )
    assert report['replica_check_mode'] == 'fingerprint'
    assert report['post_reduction_all_finite'] == 'deferred-commit'
    with pytest.raises(ValueError) as excinfo:
      reducer.drain_deferred_finite_receipts()
      commit_log.append('optimizer-commit')
  message = str(excinfo.value)
  assert 'before the optimizer commit' in message
  assert 'stage=staged' in message
  assert "['wc']" in message
  assert not commit_log


def test_killtest_b_sync_mode_still_raises_per_group():
  module = _dpt()
  mesh = _mesh(4)

  def inject_inf(key, stacked):
    if key == 'wb':
      corrupted = np.asarray(stacked, np.float32)
      corrupted[2, 1] = np.inf
      return corrupted.astype(ml_dtypes.bfloat16)
    return stacked

  with _flags():
    reducer = _make_reducer(module, mesh, 4)
    with pytest.raises(ValueError, match='non-finite'):
      reducer.finalize_staged(
          _staged_table(mesh, _rank_contributions(4), mutate=inject_inf)
      )
    assert reducer.pending_finite_receipt_count == 0


def test_drain_clean_counts_and_empty_idempotence():
  module = _dpt()
  mesh = _mesh(4)
  leaf_count = len(_LEAF_ORDER)
  with _flags(CANON_DP_FINITE_FETCH='batched-commit'):
    reducer = _make_reducer(module, mesh, 4)
    for _ in range(3):
      reducer.finalize_staged(_staged_table(mesh, _rank_contributions(4)))
    assert reducer.pending_finite_receipt_count == 6
    receipt = reducer.drain_deferred_finite_receipts()
  assert receipt == {
      'deferred_finite_receipt_groups': 3,
      'deferred_finite_receipts': 6,
      'deferred_finite_flags_checked': 3 * (4 * leaf_count + leaf_count),
      'all_finite': True,
  }
  empty = reducer.drain_deferred_finite_receipts()
  assert empty['deferred_finite_receipts'] == 0
  assert empty['all_finite'] is True


# ---------------------------------------------------------------------------
# Composition and value invariance
# ---------------------------------------------------------------------------


def test_all_knives_together_never_change_gradient_values():
  module = _dpt()
  mesh = _mesh(4)
  module.reset_receipt_schedule_update_counter_for_tests()
  with _flags():
    _, default_reduced, _ = _clean_reduced(module, mesh, 4)
  with _flags(
      CANON_DP_COMPARE_MODE='fingerprint-hybrid',
      CANON_DP_DISTINCT_SCHEDULE='first-group-warmup',
      CANON_DP_FINITE_FETCH='batched-commit',
  ):
    reducer = _make_reducer(module, mesh, 4)
    lightened_reduced, report = reducer.finalize_staged(
        _staged_table(mesh, _rank_contributions(4))
    )
    drain = reducer.drain_deferred_finite_receipts()
  assert _tree_bytes(default_reduced) == _tree_bytes(lightened_reduced)
  assert report['replica_check_mode'] == 'full+fingerprint-selfcheck'
  assert report['pending_finite_receipts'] == 2
  assert drain['all_finite'] is True


def test_hybrid_compare_on_tp_sharded_mesh():
  module = _dpt()
  mesh = _mesh(2, tp=2)
  sharding = NamedSharding(mesh, P())
  tp_sharding = NamedSharding(mesh, P('tp'))
  template = {
      'wa': jax.device_put(
          jnp.asarray(_template_arrays()['wa']), sharding
      ),
      'wd': jax.device_put(
          jnp.asarray(np.arange(8, dtype=np.float32)), tp_sharding
      ),
  }
  with _flags(CANON_DP_COMPARE_MODE='fingerprint-hybrid'):
    reducer = module.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis='dp',
        require_distinct_fingerprints=False,
    )
  matches = reducer._fingerprint_replica_matches(template)
  assert matches.shape == (2, 2)
  assert bool(np.all(matches))


# ---------------------------------------------------------------------------
# Adapter wiring (commit gate + config guard + receipt honesty)
# ---------------------------------------------------------------------------


def _adapter_function_segment(name):
  path = canonical_qwen3_adapter.__file__
  with open(path, 'r') as handle:
    source = handle.read()
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == name:
      return ast.get_source_segment(source, node)
  raise AssertionError(f'{name} not found in {path}')


def test_adapter_drains_receipts_before_gradients_escape():
  segment = _adapter_function_segment('segmented_dp_grpo_value_and_grad')
  drain_index = segment.index('drain_deferred_finite_receipts()')
  # The anchor string also appears in an earlier numeric-debug receipt;
  # the RETURN dict is the last occurrence inside this function.
  return_index = segment.rindex('"loss_output": loss_output')
  assert drain_index < return_index
  assert '"deferred_finite_receipts": deferred_finite_receipts' in segment
  assert 'pending_finite_receipt_count' in segment


def test_adapter_guards_deterministic_repeat_against_lightening():
  segment = _adapter_function_segment('segmented_dp_grpo_value_and_grad')
  guard_index = segment.index(
      'P70.4 receipt-lightening flags are incompatible with'
  )
  assert 'dp_training.dp_compare_mode()' in segment
  assert 'dp_training.dp_distinct_schedule_mode()' in segment
  assert 'dp_training.dp_finite_fetch_mode()' in segment
  assert guard_index < segment.index('def reverse_reduce_group')


def test_adapter_propagates_deferred_finite_receipt_string():
  segment = _adapter_function_segment('segmented_dp_grpo_value_and_grad')
  assert 'isinstance(reduction_finite, str)' in segment
  rank_parallel_arm = segment[: segment.index('rank_counts = []')]
  assert '"gradient_finite": bool(reduction_finite)' not in rank_parallel_arm
