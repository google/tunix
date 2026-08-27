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

"""P70.3 regression suite: the group-hoisted reverse dispatch preparation.

CL 2655471c "Hoist the per-layer reverse dispatch preparation to once per
group" moved the host python repeated by every per-layer backward dispatch
(state canonicalization, leaf-count validation, per-layer leaf gather,
host-boundary tracer walk) to one ``prepare_block_pullback_group`` call
per group plus one ``check_pullback_group_boundary`` per chunk; each
per-layer body shrank to fetch-precomputed-tuple then jitted dispatch
(``run_block_pullback_prepared`` / ``..._tape_prepared`` /
``..._rank_parallel_prepared``).

This suite pins the CL's claims: the prepared tuples hold exactly the
objects the per-call path gathers for every layer, every hoisted
validation keeps its detection power (malformed structure and outer
tracers still raise at the hoisted site, before any dispatch), and the
per-layer dispatch sequence and results are identical to the unprepared
entry points that remain in-tree for the P66 oracle and diagnostics.

Rebuilt 2026-08-27 after the original scratch suite was lost; assertion
inventory reconstructed from tasks/v1_hp_zero_tim/phases/v1-p70-tail-fusion.md
and tasks/v1_hp_zero_tim/p70a_acceptance_20260827.md.
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

import importlib.util
import sys
from pathlib import Path
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter

FunctionalMappingError = canonical_qwen3_adapter.FunctionalMappingError

_SEGMENTED_ENV = {
    "CANON_P28_SEGMENTED_FORWARD": "1",
    "CANON_P28_SEGMENTED_TRAIN": "1",
    "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
}

_FIXTURES_NAME = "p70_adapter_test_fixtures"


def _load_module_by_path(path, name):
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _fixtures():
  path = (
      Path(canonical_qwen3_adapter.__file__).resolve().parents[2]
      / "tests"
      / "rl"
      / "canonical_qwen3_adapter_test.py"
  )
  if not path.exists():
    path = Path(__file__).resolve().with_name(
        "canonical_qwen3_adapter_test.py"
    )
  return _load_module_by_path(path, _FIXTURES_NAME)


def _build_engine(runner=None):
  fixtures = _fixtures()
  runner = runner or fixtures._SegmentedRunner()
  runner.mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")
  )
  with mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False):
    return (
        canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner),
        runner,
    )


def _layer_operands(hidden_shape=(2, 3)):
  hidden = jnp.arange(
      int(np.prod(hidden_shape)), dtype=jnp.float32
  ).reshape(hidden_shape) / 7.0
  cache = jnp.asarray(0.25, jnp.float32)
  metadata = jnp.asarray(0.125, jnp.float32)
  dnext_cache = jnp.asarray(1.5, jnp.float32)
  dnext_hidden = jnp.ones(hidden_shape, jnp.float32) / 3.0
  return cache, hidden, metadata, dnext_cache, dnext_hidden


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


# ---------------------------------------------------------------------------
# Prepared tuples equal the per-call construction, for every layer
# ---------------------------------------------------------------------------


def test_prepared_tuples_equal_per_call_construction_for_every_layer():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  prepared = engine.prepare_block_pullback_group(leaves)
  layer_indices = engine._local_layer_full_indices  # pylint: disable=protected-access
  assert len(prepared) == len(layer_indices) > 1
  for layer_index, indices in enumerate(layer_indices):
    # Exactly the per-call gather of run_block_pullback, object-identical
    # leaves in the same order.
    per_call = tuple(leaves[index] for index in indices)
    assert prepared[layer_index] == per_call
    for prepared_leaf, gathered_leaf in zip(
        prepared[layer_index], per_call, strict=True
    ):
      assert prepared_leaf is gathered_leaf


def test_prepare_without_state_returns_captured_leaves_and_release_raises():
  engine, _ = _build_engine()
  prepared = engine.prepare_block_pullback_group(None)
  assert prepared is engine._local_layer_leaves  # pylint: disable=protected-access
  engine.release_captured_state()
  with pytest.raises(
      FunctionalMappingError, match="requires explicit current state"
  ):
    engine.prepare_block_pullback_group(None)
  # Explicit state still prepares after the release.
  _, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  assert len(engine.prepare_block_pullback_group(leaves)) == len(
      engine._local_layer_full_indices  # pylint: disable=protected-access
  )


# ---------------------------------------------------------------------------
# Malformed structure still raises at the hoisted site
# ---------------------------------------------------------------------------


def test_prepare_validates_leaf_count_at_hoisted_site():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  with pytest.raises(
      FunctionalMappingError, match="P28 pullback state leaf count changed"
  ):
    engine.prepare_block_pullback_group(leaves[:-1])
  with pytest.raises(
      FunctionalMappingError, match="P59 pullback state leaf count changed"
  ):
    engine.prepare_block_pullback_group(leaves + leaves[:1], label="P59")


def test_boundary_check_detects_outer_tracer_before_any_dispatch():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  prepared = engine.prepare_block_pullback_group(leaves)
  recorded = []
  original = engine._local_layer_pullback_fns  # pylint: disable=protected-access
  engine._local_layer_pullback_fns = tuple(  # pylint: disable=protected-access
      (lambda *args, _k=k: recorded.append(_k))
      for k in range(len(original))
  )
  try:

    def traced(value):
      # An outer jit/grad makes every enclosing-frame value a Tracer; the
      # once-per-chunk boundary walk must fire exactly then.
      engine.check_pullback_group_boundary((value,), prepared)
      return value

    with pytest.raises(FunctionalMappingError, match="host boundary"):
      jax.make_jaxpr(traced)(jnp.ones(2))
    assert not recorded  # detection happened before any dispatch
  finally:
    engine._local_layer_pullback_fns = original  # pylint: disable=protected-access
  # Concrete trees pass the same walk.
  engine.check_pullback_group_boundary((jnp.ones(2),), prepared)


def test_prepared_dispatch_guards_raise_before_dispatch():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  prepared = engine.prepare_block_pullback_group(leaves)
  operands = _layer_operands()
  recorded = []
  original = engine._local_layer_pullback_fns  # pylint: disable=protected-access
  original_tape = engine._local_layer_pullback_tape_fns  # pylint: disable=protected-access
  stub = lambda *args: recorded.append(args)
  engine._local_layer_pullback_fns = tuple(  # pylint: disable=protected-access
      stub for _ in original
  )
  engine._local_layer_pullback_tape_fns = tuple(  # pylint: disable=protected-access
      stub for _ in original_tape
  )
  try:
    with pytest.raises(
        FunctionalMappingError, match="pullback layer index out of range"
    ):
      engine.run_block_pullback_prepared(prepared, len(prepared), *operands)
    with pytest.raises(
        FunctionalMappingError,
        match="prepared pullback group has wrong layer count",
    ):
      engine.run_block_pullback_prepared(prepared[:-1], 0, *operands)
    with pytest.raises(
        FunctionalMappingError, match="tape pullback layer index out of range"
    ):
      engine.run_block_pullback_tape_prepared(prepared, -1, *operands)
    with pytest.raises(
        FunctionalMappingError,
        match="prepared pullback group has wrong layer count",
    ):
      engine.run_block_pullback_tape_prepared(
          prepared + prepared[:1], 0, *operands
      )
    assert not recorded  # every guard fired before any dispatch
  finally:
    engine._local_layer_pullback_fns = original  # pylint: disable=protected-access
    engine._local_layer_pullback_tape_fns = original_tape  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Dispatch-sequence parity: prepared path vs the per-call path
# ---------------------------------------------------------------------------


def test_prepared_dispatch_bitwise_matches_per_call_for_every_layer():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  prepared = engine.prepare_block_pullback_group(leaves)
  operands = _layer_operands()
  for layer_index in range(len(prepared)):
    per_call = engine.run_block_pullback(
        layer_index, *operands, state_leaves=leaves
    )
    hoisted = engine.run_block_pullback_prepared(
        prepared, layer_index, *operands
    )
    assert _tree_bytes(hoisted) == _tree_bytes(per_call)


def test_dispatch_sequence_parity_with_recording_stubs():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  operands = _layer_operands()
  original = engine._local_layer_pullback_fns  # pylint: disable=protected-access

  def make_stub(layer_index, log):
    def stub(local_leaves, cache, hidden, metadata, dnext_cache, dnext_hidden):
      log.append((
          layer_index,
          tuple(id(leaf) for leaf in local_leaves),
          id(cache),
          id(hidden),
          id(metadata),
          id(dnext_cache),
          id(dnext_hidden),
      ))
      return (
          jnp.asarray(float(layer_index)),
          jnp.zeros_like(cache),
          jnp.zeros_like(hidden),
      )

    return stub

  per_call_log, prepared_log = [], []
  try:
    engine._local_layer_pullback_fns = tuple(  # pylint: disable=protected-access
        make_stub(k, per_call_log) for k in range(len(original))
    )
    for layer_index in reversed(range(len(original))):
      engine.run_block_pullback(
          layer_index, *operands, state_leaves=leaves
      )
    engine._local_layer_pullback_fns = tuple(  # pylint: disable=protected-access
        make_stub(k, prepared_log) for k in range(len(original))
    )
    prepared = engine.prepare_block_pullback_group(leaves)
    engine.check_pullback_group_boundary(operands)
    for layer_index in reversed(range(len(original))):
      engine.run_block_pullback_prepared(prepared, layer_index, *operands)
  finally:
    engine._local_layer_pullback_fns = original  # pylint: disable=protected-access
  # Same layers in the same order, with the SAME operand objects (leaf
  # tuples included) — the compiled programs cannot tell the paths apart.
  assert prepared_log == per_call_log


def test_prepared_dispatch_updates_issue_anatomy_accounting():
  engine, runner = _build_engine()
  leaves = tuple(runner.state_leaves)
  prepared = engine.prepare_block_pullback_group(leaves)
  anatomy = canonical_qwen3_adapter._ISSUE_ANATOMY  # pylint: disable=protected-access
  before = anatomy["n"]
  engine.run_block_pullback_prepared(prepared, 0, *_layer_operands())
  # The [PERF] vag_reverse split keeps counting prep vs call per dispatch.
  assert anatomy["n"] == before + 1


# ---------------------------------------------------------------------------
# The migrated P59 rank-parallel caller
# ---------------------------------------------------------------------------


def test_rank_parallel_prepared_matches_unprepared_dp2_tp2():
  if len(jax.devices()) < 4:
    pytest.skip("requires four forced CPU devices")
  fixtures = _fixtures()
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model")
  )
  runner = fixtures._CompleteSegmentedRunner()
  graphdef, state = nnx.split(runner.model)
  state = jax.tree.map(
      lambda value: jax.device_put(
          value,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
      ),
      state,
  )
  runner.model = nnx.merge(graphdef, state)
  _, runner.state = nnx.split(runner.model)
  runner.state_leaves = tuple(jax.tree.leaves(runner.state))
  runner.mesh = mesh
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data")
  )

  def rank_local_layer(module, cache, layer_hidden, metadata):
    output = layer_hidden * module.scale[...] + metadata + cache * 0.1
    return cache + output, output

  with (
      mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False),
      mock.patch.object(
          fixtures._SegmentedLayer, "__call__", rank_local_layer
      ),
  ):
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
    leaves = tuple(runner.state_leaves)
    hidden = jax.device_put(
        jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1) / 7.0, sharding
    )
    cache = jax.device_put(jnp.ones_like(hidden) / 11.0, sharding)
    dnext_cache = jax.device_put(jnp.ones_like(cache) / 5.0, sharding)
    dnext_hidden = jax.device_put(jnp.ones_like(hidden) / 3.0, sharding)
    metadata = jnp.asarray(0.125, jnp.float32)
    prepared = engine.prepare_block_pullback_group(leaves, label="P59")
    unprepared = engine.run_block_pullback_rank_parallel(
        0,
        cache,
        hidden,
        metadata,
        dnext_cache,
        dnext_hidden,
        state_leaves=leaves,
    )
    hoisted = engine.run_block_pullback_rank_parallel_prepared(
        prepared, 0, cache, hidden, metadata, dnext_cache, dnext_hidden
    )
    assert _tree_bytes(hoisted) == _tree_bytes(unprepared)
    with pytest.raises(
        FunctionalMappingError, match="P59 pullback layer index out of range"
    ):
      engine.run_block_pullback_rank_parallel_prepared(
          prepared, 99, cache, hidden, metadata, dnext_cache, dnext_hidden
      )
