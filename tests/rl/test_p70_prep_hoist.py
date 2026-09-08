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

import dataclasses
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


class _ProgramKeyLayer(nnx.Module):

  def __init__(self, prefix, activation, p22xh_site="input_layernorm"):
    self.prefix = prefix
    self._p22xh_prefix = f"{prefix}.{p22xh_site}"
    self.kernel_init = jax.nn.initializers.uniform()
    self.activation = activation
    self.weight = nnx.Param(jnp.asarray(1.0, jnp.float32))


def test_layer_program_key_normalizes_only_non_execution_metadata():
  left_graphdef, _ = nnx.split(_ProgramKeyLayer("model.layers.0.x", "silu"))
  right_graphdef, _ = nnx.split(_ProgramKeyLayer("model.layers.1.x", "silu"))
  changed_graphdef, _ = nnx.split(
      _ProgramKeyLayer("model.layers.1.x", "gelu")
  )
  changed_site_graphdef, _ = nnx.split(
      _ProgramKeyLayer(
          "model.layers.1.x", "silu", "post_attention_layernorm"
      )
  )
  left_key, left_prefixes, left_initializers = (
      canonical_qwen3_adapter._p59_layer_graph_program_key(  # pylint: disable=protected-access
          left_graphdef, 0
      )
  )
  right_key, right_prefixes, right_initializers = (
      canonical_qwen3_adapter._p59_layer_graph_program_key(  # pylint: disable=protected-access
          right_graphdef, 1
      )
  )
  changed_key, _, _ = (
      canonical_qwen3_adapter._p59_layer_graph_program_key(  # pylint: disable=protected-access
          changed_graphdef, 1
      )
  )
  changed_site_key, _, _ = (
      canonical_qwen3_adapter._p59_layer_graph_program_key(  # pylint: disable=protected-access
          changed_site_graphdef, 1
      )
  )
  assert left_key == right_key
  assert (left_prefixes, left_initializers) == (2, 1)
  assert (right_prefixes, right_initializers) == (2, 1)
  assert changed_key != left_key
  assert changed_site_key != left_key


def test_layer_program_key_component_observer_is_fail_closed():
  @dataclasses.dataclass(frozen=True)
  class GraphKey:
    semantic: str
    outer_index: int | None

    def with_no_outer_index(self):
      return dataclasses.replace(self, outer_index=None)

  treedef = jax.tree.structure((jnp.asarray(0, jnp.float32),))
  abstract = ((4, 8), "bfloat16", False)
  physical_a = ("physical", "device", (("tpu", 0, 0),))
  physical_b = ("physical", "device", (("tpu", 0, 1),))

  def key(graph, physical):
    return (graph, 7, 7, (treedef, (abstract + (physical,),)))

  component_counts = (
      canonical_qwen3_adapter._p59_layer_program_key_component_counts
  )
  equal = component_counts(
      (
          key(GraphKey("graph-a", 0), physical_a),
          key(GraphKey("graph-a", 0), physical_a),
      )
  )
  assert equal == {
      "graph_keys": 1,
      "graph_no_outer_keys": 1,
      "normalization_keys": 1,
      "state_treedefs": 1,
      "state_abstracts": 1,
      "state_physical_layouts": 1,
      "product_keys": 1,
  }

  graph_changed = (
      component_counts(
          (
              key(GraphKey("graph-a", 0), physical_a),
              key(GraphKey("graph-b", 0), physical_a),
          )
      )
  )
  assert graph_changed == {
      **equal,
      "graph_keys": 2,
      "graph_no_outer_keys": 2,
      "product_keys": 2,
  }

  outer_changed = component_counts(
      (
          key(GraphKey("graph-a", 0), physical_a),
          key(GraphKey("graph-a", 1), physical_a),
      )
  )
  assert outer_changed == {
      **equal,
      "graph_keys": 2,
      "product_keys": 2,
  }

  physical_changed = (
      component_counts(
          (
              key(GraphKey("graph-a", 0), physical_a),
              key(GraphKey("graph-a", 0), physical_b),
          )
      )
  )
  assert physical_changed == {
      **equal,
      "state_physical_layouts": 2,
      "product_keys": 2,
  }

  with pytest.raises(
      FunctionalMappingError, match="layer program keys must be nonempty"
  ):
    component_counts(())


def test_layer_graph_difference_observer_is_bounded_and_value_safe():
  @dataclasses.dataclass(frozen=True)
  class Nested:
    semantic: str
    outer_index: int
    callback: object
    value: object

  def callback_factory():
    return lambda x: x

  left = Nested("silu", 0, callback_factory(), np.asarray([1], np.int32))
  right = Nested("gelu", 1, callback_factory(), np.asarray([1], np.int32))
  difference_paths = canonical_qwen3_adapter._p59_static_difference_paths  # pylint: disable=protected-access
  differences, truncated = difference_paths(left, right)
  assert not truncated
  assert differences == (
      ("graph.semantic", "scalar:str"),
      ("graph.outer_index", "scalar:int"),
      (
          "graph.callback",
          "callable-identity:"
          "test_p70_prep_hoist."
          "test_layer_graph_difference_observer_is_bounded_and_value_safe."
          "<locals>.callback_factory.<locals>.<lambda>",
      ),
      ("graph.value", "array-identity:ndarray"),
  )
  limited, was_truncated = difference_paths(left, right, limit=2)
  assert limited == differences[:2]
  assert was_truncated
  assert difference_paths(left, left) == ((), False)
  device_left = jnp.asarray([1.0], jnp.float32)
  device_right = jnp.asarray([1.0], jnp.float32)
  with jax.transfer_guard("disallow"):
    device_differences = difference_paths(device_left, device_right)
  assert device_differences == (
      (("graph", "array-identity:ArrayImpl"),),
      False,
  )
  with pytest.raises(
      FunctionalMappingError, match="difference limit must be positive"
  ):
    difference_paths(left, right, limit=0)


def test_layer_graph_difference_profiles_group_matching_layers():
  @dataclasses.dataclass(frozen=True)
  class GraphKey:
    activation: str

    def with_no_outer_index(self):
      return self

  treedef = jax.tree.structure((jnp.asarray(0, jnp.float32),))
  signature = (treedef, (((1,), "float32", False, None),))
  keys = (
      (GraphKey("silu"), 0, 0, signature),
      (GraphKey("gelu"), 0, 0, signature),
      (GraphKey("gelu"), 0, 0, signature),
  )
  profiles = (
      canonical_qwen3_adapter._p59_layer_graph_difference_profiles(keys)  # pylint: disable=protected-access
  )
  assert profiles == ({
      "layers": (1, 2),
      "differences": (("graph.activation", "scalar:str"),),
      "truncated": False,
  },)


def test_layer_graph_difference_names_attributes_and_compares_static_objects():
  @dataclasses.dataclass(frozen=True)
  class Attribute:
    value: object

  @dataclasses.dataclass(frozen=True)
  class Graph:
    attributes: tuple

  class Config:

    def __init__(self, output_size, fuse_matmuls):
      self.output_size = output_size
      self.fuse_matmuls = fuse_matmuls

  class Method:

    def __init__(self, output_size, fuse_matmuls):
      self.linear_config = Config(output_size, fuse_matmuls)

  left = Graph((
      ("method", Attribute(Method(8, True))),
      ("method", Attribute(Method(16, True))),
  ))
  equal = Graph((
      ("method", Attribute(Method(8, True))),
      ("method", Attribute(Method(16, True))),
  ))
  changed = Graph((
      ("method", Attribute(Method(8, True))),
      ("method", Attribute(Method(16, False))),
  ))
  difference_paths = canonical_qwen3_adapter._p59_static_difference_paths  # pylint: disable=protected-access
  assert difference_paths(left, equal) == ((), False)
  assert difference_paths(left, changed) == ((
      (
          "graph.attributes[1:'method'].value.linear_config.fuse_matmuls",
          "scalar:bool",
      ),
  ), False)


def test_layer_graph_difference_handles_named_pair_drift_and_cycles():
  class Cyclic:

    def __init__(self, mode):
      self.mode = mode
      self.child = self

  difference_paths = canonical_qwen3_adapter._p59_static_difference_paths  # pylint: disable=protected-access
  left = (("method", Cyclic("silu")),)
  right = (("method", Cyclic("gelu")),)
  assert difference_paths(left, right) == ((
      ("graph[0:'method'].mode", "scalar:str"),
  ), False)
  renamed = (("quant_method", Cyclic("silu")),)
  assert difference_paths(left, renamed) == ((
      ("graph[0]", "named-pair-key:'method'->'quant_method'"),
  ), False)


def test_layer_program_key_component_receipt_is_rank_parallel_only(capsys):
  fixtures = _fixtures()
  with mock.patch.dict(
      os.environ,
      {**_SEGMENTED_ENV, "CANON_P59_RANK_PARALLEL_BACKWARD": "1"},
      clear=False,
  ):
    _build_rank_parallel_engine(fixtures, 4, 1)
  receipts = capsys.readouterr().out
  assert receipts.count("[P59.LAYER_PROGRAM_KEY_COMPONENTS]") == 1
  assert (
      "[P59.LAYER_PROGRAM_KEY_COMPONENTS] layers=2 graph_keys=1 "
      "graph_no_outer_keys=1 normalization_keys=1 state_treedefs=1 "
      "state_abstracts=1 "
      "state_physical_layouts=1 product_keys=1 array_values_read=0 "
      "host_transfers=0"
  ) in receipts

  with mock.patch.dict(
      os.environ,
      {**_SEGMENTED_ENV, "CANON_P59_RANK_PARALLEL_BACKWARD": "0"},
      clear=False,
  ):
    _build_rank_parallel_engine(fixtures, 4, 1)
  assert "[P59.LAYER_PROGRAM_KEY_COMPONENTS]" not in capsys.readouterr().out


def test_layer_graph_difference_receipt_reaches_engine_construction(capsys):
  fixtures = _fixtures()
  with mock.patch.dict(
      os.environ,
      {**_SEGMENTED_ENV, "CANON_P59_RANK_PARALLEL_BACKWARD": "1"},
      clear=False,
  ):
    _build_rank_parallel_engine(fixtures, 4, 1, graph_difference=True)
  receipts = capsys.readouterr().out
  marker = "[P59.LAYER_GRAPH_DIFF] reference_layer=0 payload="
  assert receipts.count(marker) == 1
  payload = receipts.split(marker, maxsplit=1)[1].split(
      " array_values_read=0 host_transfers=0", maxsplit=1
  )[0]
  assert '"layers":[1]' in payload
  assert '"scalar:str"' in payload


def _build_rank_parallel_engine(
    fixtures, dp, tp, *, distinct_keys=False, graph_difference=False
):
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
      ("data", "model"),
  )
  runner = fixtures._CompleteSegmentedRunner()
  for layer_index, layer in enumerate(runner.model.model.layers):
    layer.scale = nnx.Param(
        jnp.full((tp,), 1.5 + 0.5 * layer_index, jnp.float32)
    )
    layer._p22xh_prefix = (
        f"model.layers.{layer_index}.input_layernorm"
    )
    if graph_difference:
      layer.p59_test_activation = "gelu" if layer_index else "silu"
  graphdef, state = nnx.split(runner.model)
  replicated = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  model_sharded = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("model")
  )
  state = jax.tree.map(
      lambda value: jax.device_put(
          value,
          model_sharded
          if value.ndim == 1 and value.shape == (tp,)
          else replicated,
      ),
      state,
  )
  runner.model = nnx.merge(graphdef, state)
  _, runner.state = nnx.split(runner.model)
  runner.state_leaves = tuple(jax.tree.leaves(runner.state))
  runner.mesh = mesh
  with mock.patch.dict(
      os.environ,
      {**_SEGMENTED_ENV, "CANON_P66_P59_CHECK_VMA": "1" if tp > 1 else "0"},
      clear=False,
  ):
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
  if distinct_keys:
    engine._p59_layer_pullback_program_keys = tuple(  # pylint: disable=protected-access
        (key, layer_index)
        for layer_index, key in enumerate(
            engine._p59_layer_pullback_program_keys  # pylint: disable=protected-access
        )
    )
  rank_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data", None, "model")
  )
  model_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("model")
  )
  hidden = jax.device_put(
      jnp.arange(dp * 3 * tp, dtype=jnp.float32).reshape(dp, 3, tp)
      / 7.0,
      rank_sharding,
  )
  cache = jax.device_put(jnp.ones_like(hidden) / 11.0, rank_sharding)
  operands = (
      cache,
      hidden,
      jax.device_put(jnp.full((tp,), 0.125, jnp.float32), model_sharding),
      jax.device_put(jnp.ones_like(cache) / 5.0, rank_sharding),
      jax.device_put(jnp.ones_like(hidden) / 3.0, rank_sharding),
  )
  return engine, runner, operands


def _run_rank_parallel_chain(engine, runner, operands):
  prepared = engine.prepare_block_pullback_group(
      tuple(runner.state_leaves), label="P59"
  )
  dcache, dhidden = operands[-2:]
  outputs = []
  for layer_index in reversed(range(len(prepared))):
    result = engine.run_block_pullback_rank_parallel_prepared(
        prepared,
        layer_index,
        *operands[:3],
        dcache,
        dhidden,
    )
    result = jax.block_until_ready(result)
    outputs.append(result)
    dcache, dhidden = result[1:]
  return outputs


@pytest.mark.parametrize(("dp", "tp"), ((4, 1), (2, 2), (2, 4)))
def test_homogeneous_layers_share_one_rank_parallel_program(
    dp, tp, capsys
):
  if len(jax.devices()) < dp * tp:
    pytest.skip(f"requires {dp * tp} forced CPU devices")
  if tp > 1 and not hasattr(
      jax.core.ShapedArray((), np.dtype("float32")), "mat"
  ):
    pytest.skip("checked-VMA requires the pinned JAX manual-axis API")
  fixtures = _fixtures()

  def rank_local_layer(module, cache, layer_hidden, metadata):
    output = layer_hidden * module.scale[...] + metadata + cache * 0.1
    return cache + output, output

  with (
      mock.patch.object(
          fixtures._SegmentedLayer, "__call__", rank_local_layer
      ),
      mock.patch.dict(
          os.environ,
          {"CANON_P66_P59_CHECK_VMA": "1" if tp > 1 else "0"},
          clear=False,
      ),
  ):
    baseline, baseline_runner, baseline_operands = (
        _build_rank_parallel_engine(
            fixtures, dp, tp, distinct_keys=True
        )
    )
    shared, shared_runner, shared_operands = _build_rank_parallel_engine(
        fixtures, dp, tp
    )
    assert (
        shared._p59_layer_pullback_program_keys[0]  # pylint: disable=protected-access
        == shared._p59_layer_pullback_program_keys[1]  # pylint: disable=protected-access
    )
    shared_prepared = shared.prepare_block_pullback_group(
        tuple(shared_runner.state_leaves), label="P59 signature"
    )
    assert (
        canonical_qwen3_adapter._p59_program_tree_signature(  # pylint: disable=protected-access
            (shared_prepared[0], *shared_operands)
        )
        == canonical_qwen3_adapter._p59_program_tree_signature(  # pylint: disable=protected-access
            (shared_prepared[1], *shared_operands)
        )
    )
    with jax.transfer_guard("disallow"):
      baseline_outputs = _run_rank_parallel_chain(
          baseline, baseline_runner, baseline_operands
      )
    capsys.readouterr()
    with jax.transfer_guard("disallow"):
      shared_outputs = _run_rank_parallel_chain(
          shared, shared_runner, shared_operands
      )
    receipts = capsys.readouterr().out

  assert _tree_bytes(shared_outputs) == _tree_bytes(baseline_outputs)
  assert len(shared._p59_layer_pullback_programs) == 1  # pylint: disable=protected-access
  assert len({
      id(function)
      for function in shared._p59_layer_pullback_fns  # pylint: disable=protected-access
  }) == 1
  assert (
      "[P59.LAYER_PROGRAM_REUSE] enabled=1 layers=2 static_keys=1 "
      "mapped_programs=1 logical_calls_per_layer=1 "
      f"checked_vma={int(tp > 1)} host_transfers=0"
  ) in receipts
  assert receipts.count("[P66.VMA] outer_check_enabled") == (
      2 if tp > 1 else 0
  )


def test_nonhomogeneous_layer_key_keeps_distinct_programs(capsys):
  fixtures = _fixtures()

  def rank_local_layer(module, cache, layer_hidden, metadata):
    output = layer_hidden * module.scale[...] + metadata + cache * 0.1
    return cache + output, output

  with (
      mock.patch.object(
          fixtures._SegmentedLayer, "__call__", rank_local_layer
      ),
      mock.patch.dict(
          os.environ, {"CANON_P66_P59_CHECK_VMA": "0"}, clear=False
      ),
  ):
    engine, runner, operands = _build_rank_parallel_engine(
        fixtures, 4, 1, distinct_keys=True
    )
    with jax.transfer_guard("disallow"):
      _run_rank_parallel_chain(engine, runner, operands)
  receipts = capsys.readouterr().out
  assert len(engine._p59_layer_pullback_programs) == 2  # pylint: disable=protected-access
  assert (
      "[P59.LAYER_PROGRAM_REUSE] enabled=0 layers=2 static_keys=2 "
      "mapped_programs=2 logical_calls_per_layer=1 checked_vma=0 "
      "host_transfers=0"
  ) in receipts
