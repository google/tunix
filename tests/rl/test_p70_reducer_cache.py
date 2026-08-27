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

"""P70.5 regression suite: the process-level reducer program cache.

CL de48b9b4 "Cache the reducer's jitted programs across constructions"
moved the ``FixedDPRankGradientReducer`` closure bodies into a
module-level builder (``_build_reducer_programs``) fronted by a
process-level LRU (``_reducer_program_cache``, 4 entries) keyed on the
full static identity (template treedef, per-leaf
shape/dtype/weak-type/sharding, mesh content, DP geometry, every mode
selector). Bundles trace from ``jax.ShapeDtypeStruct`` templates so no
cached closure can capture a device array (the pre-CL legacy initialize
program closed over the caller's template arrays — a real hazard found
and fixed by this CL).

This suite pins the CL's claims: an identical-config second construction
performs ZERO fresh builds and zero fresh ``jax.jit`` wrapper
constructions, a differing identity (or an unhashable one) builds fresh
and is never shared, fresh and cached bundles are jaxpr-identical,
gradients/receipts are bitwise across fresh/hit reducers, the P70.4
kill-tests still fire through cache-hit reducers, and a transitive walk
over every cached closure reaches no ``jax.Array``.

Some fixtures are ported from the surviving P70.4 standalone,
tasks/three_lane_system/scripts/p70_4_killtests.py.

Reference-parity cases compare against the pre-P70.5 dp_training. The
extraction is not checked in; produce it with

  git show 5017c279:tunix/rl/dp_training.py > /tmp/reference_dp_training_pre_p70_5.py

and point ``P70_5_REFERENCE_DP_TRAINING`` at the file. Without it those
cases SKIP; everything else runs self-contained.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16"
  ).strip()

import contextlib
import dataclasses
import functools
import importlib.util
import sys
import types
from unittest import mock

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tunix.rl import dp_training

# Pre-P70.5 worktree state (parent commit of de48b9b4); the reference
# extraction one-liner in the module docstring uses this SHA.
REFERENCE_DP_TRAINING_SHA = "5017c279"

_P70_4_FLAGS = (
    "CANON_DP_COMPARE_MODE",
    "CANON_DP_DISTINCT_SCHEDULE",
    "CANON_DP_FINITE_FETCH",
)


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


def _fresh_state():
  dp_training.reset_reducer_program_cache_for_tests()
  dp_training.reset_receipt_schedule_update_counter_for_tests()


def _mesh(dp, tp=1):
  devices = jax.devices()
  if len(devices) < dp * tp:
    pytest.skip(f"requires at least {dp * tp} forced CPU devices")
  return Mesh(np.asarray(devices[: dp * tp]).reshape(dp, tp), ("dp", "tp"))


def _template_arrays(rows=3):
  return {
      "wa": (
          np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) - 5.0
      ) / 7.0,
      "wb": np.linspace(-2.0, 2.0, 5).astype(ml_dtypes.bfloat16),
      "wc": np.asarray([[1.5, -0.5], [3.25, 0.125]], dtype=np.float32),
  }


# jax dict trees flatten in sorted key order.
_LEAF_ORDER = ("wa", "wb", "wc")


def _replicated_tree(mesh, arrays):
  sharding = NamedSharding(mesh, P())
  return {
      key: jax.device_put(jnp.asarray(value), sharding)
      for key, value in arrays.items()
  }


def _rank_contributions(dp, rows=3):
  base = _template_arrays(rows)
  contributions = []
  for rank in range(dp):
    contributions.append({
        "wa": base["wa"] + np.float32(rank) * np.float32(0.25),
        "wb": (
            np.asarray(base["wb"], np.float32) + np.float32(rank)
        ).astype(ml_dtypes.bfloat16),
        "wc": base["wc"] * np.float32(rank + 1),
    })
  return contributions


def _staged_table(mesh, contributions):
  staged = {}
  for key in contributions[0]:
    stacked = np.stack([np.array(c[key], copy=True) for c in contributions])
    staged[key] = jax.device_put(
        jnp.asarray(stacked), NamedSharding(mesh, P("dp"))
    )
  return staged


def _make_reducer(module, mesh, dp, rows=3):
  template = _replicated_tree(mesh, _template_arrays(rows))
  return module.FixedDPRankGradientReducer(
      template,
      dp_size=dp,
      dp_axis="dp",
      require_distinct_fingerprints=False,
  )


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _divergent_replicated(mesh, base_arrays, victim_key, victim_rank, mutate):
  """SYNTHETIC-MUTATION replica-divergence fixture (ported from the P70.4
  standalone): every leaf claims replication, the victim leaf's buffers
  differ on the victim DP rank."""
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


_BUNDLE_FIELDS = tuple(
    field.name for field in dataclasses.fields(dp_training._ReducerPrograms)  # pylint: disable=protected-access
)


def _reachable_jax_arrays(root, max_objects=500000):
  """Transitively walks captured state and returns every reachable jax.Array.

  Walks closures (cells), defaults, containers, dataclasses, partials and
  jit wrappers (``__wrapped__``/``_fun``/``func``). Module globals are
  deliberately NOT walked: the audited hazard is device buffers captured
  by a bundle's closures, not module-level state.
  """
  seen, stack, found = set(), [root], []
  while stack:
    if len(seen) > max_objects:
      raise AssertionError("array-reachability walk exploded")
    obj = stack.pop()
    if id(obj) in seen:
      continue
    seen.add(id(obj))
    if isinstance(obj, jax.Array):
      found.append(obj)
      continue
    if isinstance(
        obj,
        (str, bytes, int, float, bool, complex, type(None), np.dtype,
         np.ndarray, jax.ShapeDtypeStruct),
    ):
      continue
    if isinstance(obj, types.ModuleType):
      continue
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
      stack.extend(
          getattr(obj, field.name) for field in dataclasses.fields(obj)
      )
      continue
    if isinstance(obj, dict):
      stack.extend(obj.keys())
      stack.extend(obj.values())
      continue
    if isinstance(obj, (list, tuple, set, frozenset)):
      stack.extend(obj)
      continue
    if isinstance(obj, functools.partial):
      stack.append(obj.func)
      stack.extend(obj.args)
      stack.extend(obj.keywords.values())
      continue
    if isinstance(obj, types.CellType):
      try:
        stack.append(obj.cell_contents)
      except ValueError:
        pass
      continue
    if isinstance(obj, (types.FunctionType, types.MethodType)):
      function = obj.__func__ if isinstance(obj, types.MethodType) else obj
      if function.__closure__:
        stack.extend(function.__closure__)
      if function.__defaults__:
        stack.extend(function.__defaults__)
      if function.__kwdefaults__:
        stack.extend(function.__kwdefaults__.values())
      continue
    for attribute in ("__wrapped__", "_fun", "func"):
      inner = getattr(obj, attribute, None)
      if inner is not None:
        stack.append(inner)
  return found


# ---------------------------------------------------------------------------
# Identical identity: zero fresh builds, zero fresh jit constructions
# ---------------------------------------------------------------------------


def test_identical_second_construction_builds_nothing_fresh():
  _fresh_state()
  mesh = _mesh(4)
  build_calls = []
  original_build = dp_training._build_reducer_programs  # pylint: disable=protected-access

  def counting_build(*args, **kwargs):
    build_calls.append(1)
    return original_build(*args, **kwargs)

  with mock.patch.object(
      dp_training, "_build_reducer_programs", counting_build
  ):
    first = _make_reducer(dp_training, mesh, 4)
    assert len(build_calls) == 1
    jit_calls = []
    original_jit = jax.jit

    def counting_jit(*args, **kwargs):
      jit_calls.append(1)
      return original_jit(*args, **kwargs)

    # The re-trace regression pin: an identical second construction must
    # perform zero fresh builds AND zero fresh jax.jit constructions.
    with mock.patch.object(jax, "jit", counting_jit):
      second = _make_reducer(dp_training, mesh, 4)
    assert len(build_calls) == 1
    assert not jit_calls
  assert second._programs is first._programs  # pylint: disable=protected-access
  for name in _BUNDLE_FIELDS:
    assert getattr(second._programs, name) is getattr(  # pylint: disable=protected-access
        first._programs, name  # pylint: disable=protected-access
    )
  stats = dp_training._reducer_program_cache_stats  # pylint: disable=protected-access
  assert stats["hits"] == 1 and stats["misses"] == 1
  assert stats["uncacheable"] == 0


def test_different_identity_builds_fresh():
  _fresh_state()
  mesh = _mesh(4)
  base = _make_reducer(dp_training, mesh, 4)
  # Different DP geometry.
  smaller = _make_reducer(dp_training, _mesh(2), 2)
  assert smaller._programs is not base._programs  # pylint: disable=protected-access
  # Different leaf shapes.
  wider = _make_reducer(dp_training, mesh, 4, rows=5)
  assert wider._programs is not base._programs  # pylint: disable=protected-access
  # Different mode selector.
  with _flags(CANON_DP_COMPARE_MODE="fingerprint-hybrid"):
    hybrid = _make_reducer(dp_training, mesh, 4)
  assert hybrid._programs is not base._programs  # pylint: disable=protected-access
  assert hybrid._compare_fingerprint is not None  # pylint: disable=protected-access
  assert base._compare_fingerprint is None  # pylint: disable=protected-access
  stats = dp_training._reducer_program_cache_stats  # pylint: disable=protected-access
  assert stats["misses"] == 4 and stats["hits"] == 0
  # Each identity is itself cached: rebuilding any of them is a hit.
  with _flags(CANON_DP_COMPARE_MODE="fingerprint-hybrid"):
    hybrid_again = _make_reducer(dp_training, mesh, 4)
  assert hybrid_again._programs is hybrid._programs  # pylint: disable=protected-access
  assert dp_training._reducer_program_cache_stats["hits"] == 1  # pylint: disable=protected-access


def test_unhashable_identity_is_uncacheable_and_never_shared():
  _fresh_state()
  mesh = _mesh(4)
  with mock.patch.object(
      dp_training,
      "_reducer_program_cache_key",
      side_effect=TypeError("synthetic unhashable identity"),
  ):
    first = _make_reducer(dp_training, mesh, 4)
    second = _make_reducer(dp_training, mesh, 4)
  # Fail closed: never shared on ambiguity, never cached.
  assert first._programs is not second._programs  # pylint: disable=protected-access
  stats = dp_training._reducer_program_cache_stats  # pylint: disable=protected-access
  assert stats["uncacheable"] == 2
  assert not dp_training._reducer_program_cache  # pylint: disable=protected-access


def test_cache_is_lru_bounded():
  _fresh_state()
  mesh = _mesh(4)
  limit = dp_training._REDUCER_PROGRAM_CACHE_LIMIT  # pylint: disable=protected-access
  assert limit == 4
  for rows in range(3, 3 + limit + 1):  # one identity more than the limit
    _make_reducer(dp_training, mesh, 4, rows=rows)
    assert len(dp_training._reducer_program_cache) <= limit  # pylint: disable=protected-access
  stats = dp_training._reducer_program_cache_stats  # pylint: disable=protected-access
  assert stats["misses"] == limit + 1
  # The oldest identity was evicted: rebuilding it is a fresh miss.
  _make_reducer(dp_training, mesh, 4, rows=3)
  assert dp_training._reducer_program_cache_stats["misses"] == limit + 2  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Fresh and cached bundles are program-identical
# ---------------------------------------------------------------------------


def _bundle_jaxprs(reducer, mesh, dp):
  template = _replicated_tree(mesh, _template_arrays())
  staged = _staged_table(mesh, _rank_contributions(dp))
  jaxprs = {
      "initialize": str(jax.make_jaxpr(reducer._initialize)()),  # pylint: disable=protected-access
      "reduce": str(jax.make_jaxpr(reducer._reduce)(staged)),  # pylint: disable=protected-access
      "compare": str(jax.make_jaxpr(reducer._compare)(template)),  # pylint: disable=protected-access
      "diagnostics": str(
          jax.make_jaxpr(reducer._batched_diagnostics)(staged)  # pylint: disable=protected-access
      ),
  }
  if reducer._compare_fingerprint is not None:  # pylint: disable=protected-access
    jaxprs["fingerprint"] = str(
        jax.make_jaxpr(reducer._compare_fingerprint)(template)  # pylint: disable=protected-access
    )
  return jaxprs


@pytest.mark.parametrize("hybrid", [False, True])
def test_jaxpr_equality_fresh_vs_cached(hybrid):
  _fresh_state()
  mesh = _mesh(4)
  flag_values = (
      {"CANON_DP_COMPARE_MODE": "fingerprint-hybrid"} if hybrid else {}
  )
  with _flags(**flag_values):
    fresh_a = _make_reducer(dp_training, mesh, 4)
    hit = _make_reducer(dp_training, mesh, 4)
    dp_training.reset_reducer_program_cache_for_tests()
    fresh_b = _make_reducer(dp_training, mesh, 4)
  jaxprs_a = _bundle_jaxprs(fresh_a, mesh, 4)
  assert _bundle_jaxprs(hit, mesh, 4) == jaxprs_a  # cache hit
  assert _bundle_jaxprs(fresh_b, mesh, 4) == jaxprs_a  # deterministic build
  if hybrid:
    assert "fingerprint" in jaxprs_a


# ---------------------------------------------------------------------------
# Bitwise gradients and receipts across fresh/hit reducers
# ---------------------------------------------------------------------------


def test_gradients_and_receipts_bitwise_across_fresh_and_hit_reducers():
  _fresh_state()
  mesh = _mesh(4)
  contributions = _rank_contributions(4)
  fresh = _make_reducer(dp_training, mesh, 4)
  hit = _make_reducer(dp_training, mesh, 4)
  assert hit._programs is fresh._programs  # pylint: disable=protected-access
  fresh_reduced, fresh_report = fresh.finalize_staged(
      _staged_table(mesh, contributions)
  )
  hit_reduced, hit_report = hit.finalize_staged(
      _staged_table(mesh, contributions)
  )
  assert _tree_bytes(fresh_reduced) == _tree_bytes(hit_reduced)
  assert fresh_report == hit_report
  # Serial transaction path through the shared bundle, byte-identical too.
  serial = _make_reducer(dp_training, mesh, 4)
  serial.begin()
  for rank, contribution in enumerate(contributions):
    serial.add(rank, _replicated_tree(mesh, contribution))
  serial_reduced, _ = serial.finalize()
  assert _tree_bytes(serial_reduced) == _tree_bytes(fresh_reduced)


def test_killtests_fire_through_cache_hit_reducers():
  _fresh_state()
  mesh = _mesh(4)
  with _flags(
      CANON_DP_COMPARE_MODE="fingerprint-hybrid",
      CANON_DP_FINITE_FETCH="batched-commit",
  ):
    _make_reducer(dp_training, mesh, 4)  # populate the cache
    reducer = _make_reducer(dp_training, mesh, 4)  # cache hit
    assert dp_training._reducer_program_cache_stats["hits"] >= 1  # pylint: disable=protected-access
    reduced, _ = reducer.finalize_staged(
        _staged_table(mesh, _rank_contributions(4))
    )
    base = {key: np.asarray(value) for key, value in reduced.items()}

    def flip_bit(leaf):
      view = leaf.view(np.uint32).reshape(-1)
      view[1] ^= np.uint32(1 << 7)
      return leaf

    divergent = _divergent_replicated(mesh, base, "wc", 1, flip_bit)
    with pytest.raises(ValueError, match="dual-checksum fingerprint"):
      reducer._assert_fingerprint_replicas_equal(divergent)  # pylint: disable=protected-access

    # Non-finite injection still drains fail-closed through the hit.
    contributions = _rank_contributions(4)
    contributions[2]["wc"] = np.array(contributions[2]["wc"], copy=True)
    contributions[2]["wc"][0, 0] = np.inf
    reducer.finalize_staged(_staged_table(mesh, contributions))
    with pytest.raises(ValueError, match="before the optimizer commit"):
      reducer.drain_deferred_finite_receipts()


# ---------------------------------------------------------------------------
# No jax.Array reachable from cached closures
# ---------------------------------------------------------------------------


def test_array_walker_detects_a_planted_capture():
  # Negative control for the audit instrument itself: a closure that
  # captures a device array MUST be found.
  planted = jnp.ones(3)

  def capturing():
    return planted

  assert len(_reachable_jax_arrays(capturing)) == 1


@pytest.mark.parametrize("hybrid", [False, True])
def test_no_jax_array_reachable_from_cached_closures(hybrid):
  _fresh_state()
  mesh = _mesh(4)
  flag_values = (
      {
          "CANON_DP_COMPARE_MODE": "fingerprint-hybrid",
          "CANON_DP_DISTINCT_SCHEDULE": "first-group-warmup",
          "CANON_DP_FINITE_FETCH": "batched-commit",
      }
      if hybrid
      else {}
  )
  with _flags(**flag_values):
    reducer = _make_reducer(dp_training, mesh, 4)
  bundle = reducer._programs  # pylint: disable=protected-access
  assert not _reachable_jax_arrays(bundle)
  for name in _BUNDLE_FIELDS:
    assert not _reachable_jax_arrays(getattr(bundle, name)), name
  # The cache itself holds no device buffers either.
  assert not _reachable_jax_arrays(
      dict(dp_training._reducer_program_cache)  # pylint: disable=protected-access
  )


def test_shape_dtype_struct_template_is_admitted():
  # P70.5 admits metadata-only templates: the bundle traces from
  # ShapeDtypeStruct leaves, so a buffer-free construction must work and
  # share programs with an array-template construction of the same
  # identity.
  _fresh_state()
  mesh = _mesh(4)
  array_reducer = _make_reducer(dp_training, mesh, 4)
  sharding = NamedSharding(mesh, P())
  spec_template = {
      key: jax.ShapeDtypeStruct(
          np.shape(value), jnp.asarray(value).dtype, sharding=sharding
      )
      for key, value in _template_arrays().items()
  }
  spec_reducer = dp_training.FixedDPRankGradientReducer(
      spec_template,
      dp_size=4,
      dp_axis="dp",
      require_distinct_fingerprints=False,
  )
  assert spec_reducer._programs is array_reducer._programs  # pylint: disable=protected-access
  spec_reducer.begin()
  staged = spec_reducer._staged  # pylint: disable=protected-access
  assert jax.tree.leaves(staged)[0].shape[0] == 4


# ---------------------------------------------------------------------------
# Reference parity against the pre-P70.5 module (optional extraction)
# ---------------------------------------------------------------------------


def _reference_module_or_skip():
  path = os.environ.get("P70_5_REFERENCE_DP_TRAINING", "")
  if not path or not os.path.exists(path):
    pytest.skip(
        "pre-P70.5 dp_training extraction absent; produce it with "
        f"`git show {REFERENCE_DP_TRAINING_SHA}:tunix/rl/dp_training.py "
        "> <file>` and set P70_5_REFERENCE_DP_TRAINING=<file>"
    )
  name = "p70_5_dp_training_reference"
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


@pytest.mark.parametrize("hybrid", [False, True])
def test_programs_match_pre_p70_5_reference_jaxprs(hybrid):
  reference = _reference_module_or_skip()
  _fresh_state()
  mesh = _mesh(4)
  flag_values = (
      {"CANON_DP_COMPARE_MODE": "fingerprint-hybrid"} if hybrid else {}
  )
  with _flags(**flag_values):
    landed = _make_reducer(dp_training, mesh, 4)
    unpatched = _make_reducer(reference, mesh, 4)
  landed_jaxprs = _bundle_jaxprs(landed, mesh, 4)
  reference_jaxprs = _bundle_jaxprs(unpatched, mesh, 4)
  assert landed_jaxprs == reference_jaxprs
  # Values bitwise too.
  landed_reduced, _ = landed.finalize_staged(
      _staged_table(mesh, _rank_contributions(4))
  )
  reference_reduced, _ = unpatched.finalize_staged(
      _staged_table(mesh, _rank_contributions(4))
  )
  assert _tree_bytes(landed_reduced) == _tree_bytes(reference_reduced)


def test_reference_initialize_capture_hazard_is_detected_and_fixed():
  # The CL's one real capture hazard: the pre-P70.5 legacy initialize
  # closed over the caller's template ARRAYS. The audit walker must find
  # those captures in the reference module and none in the landed one —
  # proving both the fix and the detector.
  reference = _reference_module_or_skip()
  _fresh_state()
  mesh = _mesh(4)
  with _flags():
    landed = _make_reducer(dp_training, mesh, 4)
    unpatched = _make_reducer(reference, mesh, 4)
  assert _reachable_jax_arrays(unpatched._initialize)  # pylint: disable=protected-access
  assert not _reachable_jax_arrays(landed._initialize)  # pylint: disable=protected-access
