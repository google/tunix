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

"""P70.1 regression suite: the three fused gradient programs.

CL 2d418187 "Collapse the eager gradient assembly and tree strips into
three jitted programs" replaced three eager per-leaf dispatch strips of the
grouped reverse pass with one cached jitted program each:

  * ``_P28SegmentedEngineForward._p70_sparse_assembly``
    (module ``zt_tr_grad_assembly``) — the sparse full-state assembly,
  * ``Qwen3EngineForwardAdapter._p70_grad_tree_start``
    (module ``zt_tr_grad_tree_start``) — the first-chunk ``0 + x`` start,
  * ``Qwen3EngineForwardAdapter._p70_grad_tree_add``
    (module ``zt_tr_grad_tree_add``) — the cross-chunk ``a + b``
    accumulate, which donates its consumed accumulator.

This suite pins the CL's claims: bitwise parity of each jitted program
against the eager per-leaf expressions it replaced (bf16 + fp32, signed
zeros, subnormals, the tied embed/head merge), the runtime-zero design that
defeats XLA's ``add(const 0, x) -> x`` constant folding (a folded variant
would return -0.0 where the landed path returns +0.0 — pinned red by a
dedicated test), single-cached-program dispatch, and the fail-closed
structure guards.

Reference-parity cases additionally compare against the UNPATCHED adapter.
The extraction is not checked in; produce it with

  git show 019d7a7e:tunix/rl/canonical_qwen3_adapter.py \
      > /tmp/reference_adapter_pre_p70_1.py

and point ``P70_REFERENCE_ADAPTER`` at the file. Without it those cases
SKIP; everything else runs self-contained.

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
import ml_dtypes
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter

# Pre-P70.1 worktree state (parent commit of 2d418187); the reference
# extraction one-liner in the module docstring uses this SHA.
REFERENCE_ADAPTER_SHA = "019d7a7e"

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
  # dataclass(slots=True)/absltest resolve their defining module through
  # sys.modules, so register before exec.
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _fixtures():
  """Loads the tiny segmented-model fixtures from the adapter test module."""
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


def _unit_mesh():
  return jax.sharding.Mesh(
      np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")
  )


def _build_engine(runner):
  runner.mesh = _unit_mesh()
  with mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False):
    return canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)


def _tied_engine():
  return _build_engine(_fixtures()._TiedSegmentedRunner())


def _untied_engine():
  return _build_engine(_fixtures()._SegmentedRunner())


class _Bf16TiedEmbed(nnx.Module):
  """Tied embed whose shared leaf is bf16, so the tied merge rounds."""

  def __init__(self):
    self.weight = nnx.Param(
        jnp.asarray(
            [[0.5], [1.0], [1.5], [2.0], [2.5]], ml_dtypes.bfloat16
        )
    )

  def __call__(self, token_ids):
    return self.weight[token_ids]

  def decode(self, hidden):
    return jnp.dot(hidden, self.weight[...].T)


def _bf16_tied_engine():
  fixtures = _fixtures()
  runner = fixtures._TiedSegmentedRunner()
  runner.model.model.embed_tokens = _Bf16TiedEmbed()
  _, runner.state = nnx.split(runner.model)
  runner.state_leaves = tuple(jax.tree.leaves(runner.state))
  return _build_engine(runner)


def _weird_values(leaves, *, seed, rank_axis_size=None, dtype=None):
  """Cotangent fixtures seeded with -0.0 and f32/bf16 subnormals."""
  values = []
  for index, leaf in enumerate(leaves):
    shape = (
        leaf.shape
        if rank_axis_size is None
        else (rank_axis_size,) + leaf.shape
    )
    size = max(int(np.prod(shape, dtype=np.int64)), 1)
    flat = (
        np.linspace(-2.0, 2.0, size, dtype=np.float32) * (index + seed)
    ).astype(np.float32)
    flat[0] = -0.0
    if size >= 2:
      flat[1] = np.float32(1e-42)  # f32 subnormal (bf16 flushes on cast)
    if size >= 3:
      flat[2] = np.float32(-1e-42)
    target_dtype = dtype if dtype is not None else leaf.dtype
    values.append(jnp.asarray(flat[:size].reshape(shape).astype(target_dtype)))
  return tuple(values)


def _assembly_inputs(engine, *, rank_axis_size=None, dtype=None):
  embed = _weird_values(
      engine._embed_local_leaves,  # pylint: disable=protected-access
      seed=3,
      rank_axis_size=rank_axis_size,
      dtype=dtype,
  )
  layers = tuple(
      _weird_values(
          leaves, seed=7 + index, rank_axis_size=rank_axis_size, dtype=dtype
      )
      for index, leaves in enumerate(
          engine._local_layer_leaves  # pylint: disable=protected-access
      )
  )
  norm = _weird_values(
      engine._norm_local_leaves,  # pylint: disable=protected-access
      seed=11,
      rank_axis_size=rank_axis_size,
      dtype=dtype,
  )
  head = _weird_values(
      engine._head_local_leaves,  # pylint: disable=protected-access
      seed=13,
      rank_axis_size=rank_axis_size,
      dtype=dtype,
  )
  return embed, layers, norm, head


def _eager_sparse_assembly(
    engine, *, embed, layers, norm, head, rank_axis_size=None
):
  """The exact eager per-leaf expressions P70.1 replaced.

  This is a verbatim restatement of the legacy sparse branch that remains
  in ``assemble_full_state_gradient`` as the reference implementation:
  per leaf ``cast = value.astype(target.dtype)`` then
  ``jnp.asarray(0, target.dtype) + cast`` (each an eager per-leaf
  dispatch), with tied leaves first merged as
  ``embed.astype(target.dtype) + head.astype(target.dtype)``.
  """
  del rank_axis_size  # shapes already carry the staged prefix
  targets = engine._full_state_leaves  # pylint: disable=protected-access
  full = [None] * len(targets)

  def add(indices, values):
    for index, value in zip(
        indices, tuple(jax.tree_util.tree_leaves(values)), strict=True
    ):
      assert full[index] is None
      cast = value.astype(targets[index].dtype)
      full[index] = jnp.asarray(0, targets[index].dtype) + cast

  if engine._tied_word_embeddings:  # pylint: disable=protected-access
    embed_values = tuple(jax.tree_util.tree_leaves(embed))
    head_values = tuple(jax.tree_util.tree_leaves(head))
    tied_values = tuple(
        embed_value.astype(targets[index].dtype)
        + head_value.astype(targets[index].dtype)
        for index, embed_value, head_value in zip(
            engine._embed_full_indices,  # pylint: disable=protected-access
            embed_values,
            head_values,
            strict=True,
        )
    )
    add(engine._embed_full_indices, tied_values)  # pylint: disable=protected-access
  else:
    add(engine._embed_full_indices, embed)  # pylint: disable=protected-access
  for indices, values in zip(
      engine._local_layer_full_indices,  # pylint: disable=protected-access
      layers,
      strict=True,
  ):
    add(indices, values)
  add(engine._norm_full_indices, norm)  # pylint: disable=protected-access
  if not engine._tied_word_embeddings:  # pylint: disable=protected-access
    add(engine._head_full_indices, head)  # pylint: disable=protected-access
  assert all(value is not None for value in full)
  return tuple(full)


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _assert_bitwise_equal(actual_tree, expected_tree):
  actual = jax.tree.leaves(actual_tree)
  expected = jax.tree.leaves(expected_tree)
  assert len(actual) == len(expected)
  for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
    assert a.dtype == b.dtype, index
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), (
        f"leaf {index} differs bitwise"
    )


def _bits(value):
  array = np.asarray(value)
  if array.dtype.itemsize == 2:
    return array.view(np.uint16).reshape(-1)
  return array.view(np.uint32).reshape(-1)


class _CountingWrapper:

  def __init__(self, wrapped):
    self.wrapped = wrapped
    self.calls = 0

  def __call__(self, *args, **kwargs):
    self.calls += 1
    return self.wrapped(*args, **kwargs)


def _bare_adapter():
  """Adapter shell for the instance-cached tree-op programs."""
  return object.__new__(canonical_qwen3_adapter.Qwen3EngineForwardAdapter)


def _mixed_pack():
  """A nested per-chunk gradient pack with f32/bf16, -0.0, subnormals."""
  return (
      jnp.asarray([-0.0, 1.5, 1e-42, -3.25], jnp.float32),
      (
          jnp.asarray([[2.0, -0.0], [-1.5, 0.125]], ml_dtypes.bfloat16),
          jnp.asarray([7.0, -2.5], jnp.float32),
      ),
      jnp.asarray(-0.0, ml_dtypes.bfloat16),
  )


def _eager_tree_start(pack):
  # Verbatim: the removed legacy `tree_start` per-leaf expression.
  return jax.tree.map(lambda value: jnp.asarray(0, value.dtype) + value, pack)


def _eager_tree_add(left, right):
  # Verbatim: the removed legacy `tree_add` per-leaf expression.
  return jax.tree.map(lambda a, b: a + b, left, right)


# ---------------------------------------------------------------------------
# Assembly parity: jitted program vs eager per-leaf expressions
# ---------------------------------------------------------------------------


def test_untied_assembly_matches_eager_expression_bitwise():
  engine = _untied_engine()
  assert not engine._tied_word_embeddings  # pylint: disable=protected-access
  embed, layers, norm, head = _assembly_inputs(engine)
  jitted = engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  eager = _eager_sparse_assembly(
      engine, embed=embed, layers=layers, norm=norm, head=head
  )
  _assert_bitwise_equal(jitted, eager)
  # The signed-zero canonicalization survived the fusion: every -0.0
  # cotangent leaf slot is +0.0 after `0 + cast`, in both paths.
  for leaf in jitted:
    assert _bits(leaf)[0] == 0


def test_tied_assembly_matches_eager_expression_bitwise():
  engine = _tied_engine()
  assert engine._tied_word_embeddings  # pylint: disable=protected-access
  assert (
      engine._embed_full_indices  # pylint: disable=protected-access
      == engine._head_full_indices  # pylint: disable=protected-access
  )
  embed, layers, norm, head = _assembly_inputs(engine)
  jitted = engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  eager = _eager_sparse_assembly(
      engine, embed=embed, layers=layers, norm=norm, head=head
  )
  _assert_bitwise_equal(jitted, eager)
  # The single-slot tied contract: exactly one merged embed+head leaf.
  # Oracle via eager XLA ops (CPU XLA flushes subnormals; numpy does not).
  shared = engine._embed_full_indices[0]  # pylint: disable=protected-access
  merged = jnp.asarray(0, embed[0].dtype) + (embed[0] + head[0])
  assert (
      np.asarray(jitted[shared]).tobytes() == np.asarray(merged).tobytes()
  )


def test_tied_bf16_assembly_merge_rounding_matches_eager():
  engine = _bf16_tied_engine()
  shared = engine._embed_full_indices[0]  # pylint: disable=protected-access
  assert engine._full_state_leaves[shared].dtype == ml_dtypes.bfloat16  # pylint: disable=protected-access
  embed, layers, norm, head = _assembly_inputs(engine)
  # Force a merge whose bf16 sum rounds: 1.0 + 2**-9 is not
  # bf16-representable, so `bf16(e) + bf16(h)` must round — the exact
  # single bf16 rounding the P70.0 conviction pinned as NOT_REDUNDANT.
  embed_leaf = np.asarray(embed[0], np.float32)
  head_leaf = np.asarray(head[0], np.float32)
  embed_leaf[0, 0] = 1.0
  head_leaf[0, 0] = float(2.0**-9)
  embed_leaf[1, 0] = -0.0
  head_leaf[1, 0] = -0.0
  embed = (jnp.asarray(embed_leaf.astype(ml_dtypes.bfloat16)),)
  head = (jnp.asarray(head_leaf.astype(ml_dtypes.bfloat16)),)
  jitted = engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  eager = _eager_sparse_assembly(
      engine, embed=embed, layers=layers, norm=norm, head=head
  )
  _assert_bitwise_equal(jitted, eager)
  merged = np.asarray(jitted[shared])
  # bf16(1.0 + 2**-9) rounds back to 1.0 under round-to-nearest-even.
  assert float(merged[0, 0]) == 1.0
  # (-0.0) + (-0.0) = -0.0, then the `0 + x` canonicalization makes +0.0.
  assert _bits(merged[1, 0])[0] == 0


def test_rank_staged_assembly_matches_eager_expression_bitwise():
  engine = _untied_engine()
  rank = 4
  embed, layers, norm, head = _assembly_inputs(engine, rank_axis_size=rank)
  jitted = engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head, rank_axis_size=rank
  )
  eager = _eager_sparse_assembly(
      engine,
      embed=embed,
      layers=layers,
      norm=norm,
      head=head,
      rank_axis_size=rank,
  )
  _assert_bitwise_equal(jitted, eager)
  for index, leaf in enumerate(jitted):
    assert leaf.shape == (rank,) + tuple(
        engine._full_state_leaves[index].shape  # pylint: disable=protected-access
    )


# ---------------------------------------------------------------------------
# The -0.0 runtime-zero const-folding trap (dedicated test)
# ---------------------------------------------------------------------------


def test_const_folded_zero_variant_returns_minus_zero_landed_returns_plus():
  """Pins the XLA add(const 0, x) folding trap the CL threads zeros around.

  Inside jit a TRACE-TIME constant zero lets XLA fold `0 + x -> x`, which
  silently preserves -0.0; the landed programs therefore take the scalar
  zeros as RUNTIME operands. A const-folded variant of each strip must
  return -0.0 exactly where the landed path and the eager reference return
  +0.0 — if this test ever goes green on the folded variant, the
  canonicalization contract has been broken.
  """
  for dtype, view in (
      (jnp.float32, np.uint32),
      (ml_dtypes.bfloat16, np.uint16),
  ):
    negative_zero = jnp.asarray([-0.0], dtype)

    # The const-folded variant somebody could "simplify" the code into.
    @jax.jit
    def const_folded(value):
      return jnp.asarray(0, value.dtype) + value

    folded = np.asarray(const_folded(negative_zero)).view(view)
    assert folded[0] == np.array([-0.0], dtype).view(view)[0], (
        "XLA no longer folds add(const 0, x); the trap premise changed"
    )

    eager = np.asarray(
        jnp.asarray(0, dtype) + negative_zero
    ).view(view)
    assert eager[0] == 0

    adapter = _bare_adapter()
    landed = np.asarray(
        jax.tree.leaves(adapter._p70_grad_tree_start((negative_zero,)))[0]  # pylint: disable=protected-access
    ).view(view)
    assert landed[0] == 0, "landed tree_start dropped the +0 canonicalization"

  # Same trap pinned through the assembly program on a -0.0 cotangent.
  engine = _tied_engine()
  embed, layers, norm, head = _assembly_inputs(engine)
  jitted = engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  norm_index = engine._norm_full_indices[0]  # pylint: disable=protected-access
  assert _bits(jitted[norm_index])[0] == 0  # -0.0 norm cotangent -> +0.0


# ---------------------------------------------------------------------------
# Single-program dispatch and instance caching
# ---------------------------------------------------------------------------


def test_assembly_is_one_cached_program():
  engine = _tied_engine()
  embed, layers, norm, head = _assembly_inputs(engine)
  assert getattr(engine, "_p70_assembly_fn", None) is None
  engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  built = engine._p70_assembly_fn  # pylint: disable=protected-access
  assert built is not None
  spy = _CountingWrapper(built)
  engine._p70_assembly_fn = spy  # pylint: disable=protected-access
  engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  # ONE jitted dispatch covers the whole strip (312 adds + 310 converts
  # per group before the CL).
  assert spy.calls == 1
  engine._p70_assembly_fn = built  # pylint: disable=protected-access
  engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  assert engine._p70_assembly_fn is built  # pylint: disable=protected-access
  assert built._cache_size() == 1  # one compiled specialization


def test_tree_ops_single_program_and_instance_cached():
  adapter = _bare_adapter()
  pack = _mixed_pack()
  started = adapter._p70_grad_tree_start(pack)  # pylint: disable=protected-access
  start_fn = adapter._p70_tree_start_fn  # pylint: disable=protected-access
  spy = _CountingWrapper(start_fn)
  adapter._p70_tree_start_fn = spy  # pylint: disable=protected-access
  adapter._p70_grad_tree_start(pack)  # pylint: disable=protected-access
  assert spy.calls == 1
  adapter._p70_tree_start_fn = start_fn  # pylint: disable=protected-access

  adapter._p70_grad_tree_add(started, pack)  # pylint: disable=protected-access
  add_fn = adapter._p70_tree_add_fn  # pylint: disable=protected-access
  add_spy = _CountingWrapper(add_fn)
  adapter._p70_tree_add_fn = add_spy  # pylint: disable=protected-access
  adapter._p70_grad_tree_add(adapter._p70_grad_tree_start(pack), pack)  # pylint: disable=protected-access
  assert add_spy.calls == 1
  adapter._p70_tree_add_fn = add_fn  # pylint: disable=protected-access
  assert start_fn._cache_size() == 1
  assert add_fn._cache_size() == 1


# ---------------------------------------------------------------------------
# Fail-closed structure guards
# ---------------------------------------------------------------------------


def test_assembly_malformed_structure_raises():
  engine = _tied_engine()
  embed, layers, norm, head = _assembly_inputs(engine)

  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="layer-gradient count changed",
  ):
    engine.assemble_full_state_gradient(
        embed=embed, layers=layers[:-1], norm=norm, head=head
    )

  bad_embed = tuple(value[None] for value in embed)
  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="cotangent shape changed",
  ):
    engine.assemble_full_state_gradient(
        embed=bad_embed, layers=layers, norm=norm, head=head
    )

  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="tied embed/head cotangent count changed",
  ):
    engine.assemble_full_state_gradient(
        embed=embed + embed, layers=layers, norm=norm, head=head
    )

  # Same shapes, different dtype: host validation passes, the built
  # program's signature guard must fail closed.
  engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  bf16_embed = tuple(value.astype(ml_dtypes.bfloat16) for value in embed)
  bf16_head = tuple(value.astype(ml_dtypes.bfloat16) for value in head)
  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="P70 sparse-assembly cotangent signature changed",
  ):
    engine.assemble_full_state_gradient(
        embed=bf16_embed, layers=layers, norm=norm, head=bf16_head
    )


def test_tree_ops_signature_guards_raise():
  adapter = _bare_adapter()
  pack = _mixed_pack()
  started = adapter._p70_grad_tree_start(pack)  # pylint: disable=protected-access
  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="P70 tree-start gradient pack signature changed",
  ):
    adapter._p70_grad_tree_start((pack[0],))  # pylint: disable=protected-access
  adapter._p70_grad_tree_add(started, pack)  # pylint: disable=protected-access
  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError,
      match="P70 tree-add gradient pack signature changed",
  ):
    adapter._p70_grad_tree_add(pack, (pack[0],))  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Tree-strip parity and donation semantics
# ---------------------------------------------------------------------------


def test_tree_start_and_tree_add_match_eager_expressions_bitwise():
  adapter = _bare_adapter()
  pack = _mixed_pack()
  started = adapter._p70_grad_tree_start(pack)  # pylint: disable=protected-access
  eager_started = _eager_tree_start(pack)
  _assert_bitwise_equal(started, eager_started)
  # -0.0 canonicalizes to +0.0 through the fused start, f32 and bf16.
  assert _bits(jax.tree.leaves(started)[0])[0] == 0
  assert _bits(jax.tree.leaves(started)[3])[0] == 0

  second = jax.tree.map(lambda value: value * 2 - 1, pack)
  added = adapter._p70_grad_tree_add(started, second)  # pylint: disable=protected-access
  eager_added = _eager_tree_add(eager_started, second)
  _assert_bitwise_equal(added, eager_added)


def test_tree_add_donates_accumulator_and_start_does_not():
  adapter = _bare_adapter()
  pack = _mixed_pack()
  started = adapter._p70_grad_tree_start(pack)  # pylint: disable=protected-access
  # tree_start does NOT donate: its operand is oracle-retained upstream.
  assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(pack))
  second = jax.tree.map(jnp.ones_like, pack)
  adapter._p70_grad_tree_add(started, second)  # pylint: disable=protected-access
  # tree_add donates exactly its consumed accumulator (argnum 0) — the
  # only call site rebinds the accumulator to the result immediately.
  assert all(leaf.is_deleted() for leaf in jax.tree.leaves(started))
  assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(second))


# ---------------------------------------------------------------------------
# Reference parity against the unpatched adapter (optional extraction)
# ---------------------------------------------------------------------------


def _reference_adapter_or_skip():
  path = os.environ.get("P70_REFERENCE_ADAPTER", "")
  if not path or not os.path.exists(path):
    pytest.skip(
        "unpatched-adapter extraction absent; produce it with "
        f"`git show {REFERENCE_ADAPTER_SHA}:tunix/rl/"
        "canonical_qwen3_adapter.py > <file>` and set "
        "P70_REFERENCE_ADAPTER=<file>"
    )
  return _load_module_by_path(path, "p70_reference_adapter_pre_p70_1")


@pytest.mark.parametrize("tied", [True, False])
def test_assembly_matches_pre_p70_1_reference_module(tied):
  reference = _reference_adapter_or_skip()
  fixtures = _fixtures()
  make_runner = (
      fixtures._TiedSegmentedRunner if tied else fixtures._SegmentedRunner
  )
  landed_engine = _build_engine(make_runner())
  runner = make_runner()
  runner.mesh = _unit_mesh()
  with mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False):
    reference_engine = reference.build_p28_segmented_engine_forward(runner)
  embed, layers, norm, head = _assembly_inputs(landed_engine)
  landed = landed_engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  expected = reference_engine.assemble_full_state_gradient(
      embed=embed, layers=layers, norm=norm, head=head
  )
  assert _tree_bytes(landed) == _tree_bytes(expected)
