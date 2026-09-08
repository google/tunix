"""One program per reverse chunk (tasks/v2_dispatch Phase 15, flagless; the Phase 7 class redone on the certified lane).

Under the certified configuration (rank-parallel pullbacks consuming the
kept tape, no diagnostic arm) each chunk's reverse body -- chunk inputs,
entry-cache rebuild, norm/head recompute, rows pullback, head-cotangent
partition, then the head, norm, per-layer and embed mapped pullbacks --
runs as ONE program that calls the already-built mapped pullback
programs.  The per-program loop stays the bootstrap (it builds the
mapped programs and records the output shardings) and the test oracle
(``chunk_program=False``).  Bitwise on the paged fixture; one bwd_chunk
launch per chunk and none of the per-chunk programs.

Run this file in its own process.  XLA:CPU contracts ``a*b + c`` into an
FMA depending on the surrounding module: the standalone per-layer
pullback program rounds the layer's ``dh*scale + dcache*0.5`` once (FMA)
while the same HLO inside the chunk program rounds twice, a 1-ulp
difference on the fixture that no XLA pass controls (the optimization
barrier the program places on every stage boundary is expanded before
fusion on CPU).  ``--xla_cpu_max_isa=AVX`` removes the FMA instructions
from CPU code generation, so both compile to the same two roundings and
the comparison is exact; the TPU has no such contraction and honours the
barrier as a fusion boundary, which the one-host certification run
verifies with the anchors.  The frozen digests of other test files were
pinned under the default ISA and must not share a process with this one.
"""

import importlib.util
import os
import pathlib
import types
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ISA_CAP = "--xla_cpu_max_isa=AVX"
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16 "
      + _ISA_CAP
  ).strip()

import launch_budget  # noqa: E402  (patches JAX at import; before the adapter)
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter


def _load(name):
  path = pathlib.Path(__file__).with_name(name)
  spec = importlib.util.spec_from_file_location(name[:-3], path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _fixture():
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = t71._two_chunk_spec(adapter)  # pylint: disable=protected-access
  # The fixture's chunk inputs are eager slices with the chunk start as
  # the layer "metadata"; give the chunk program the same traced form
  # (as tests/rl/test_p32_chunk_batch.py does for the forward program).
  cache_sharding = jax.sharding.NamedSharding(
      runner.mesh, jax.sharding.PartitionSpec("data")
  )

  def chunk_inputs_traced(self, group_spec, start):
    bucket = self._sequence_bucket
    ids = jax.lax.dynamic_slice_in_dim(
        group_spec["packed_ids"], start, bucket, axis=1
    )
    targets = jax.lax.dynamic_slice_in_dim(
        group_spec["next_ids"], start, bucket, axis=1
    )
    return (
        jax.lax.with_sharding_constraint(ids.reshape(-1), cache_sharding),
        jax.lax.with_sharding_constraint(targets.reshape(-1), cache_sharding),
        start,
    )

  adapter._p32_chunk_inputs_traced = types.MethodType(  # pylint: disable=protected-access
      chunk_inputs_traced, adapter
  )
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  return adapter, runner, spec, env


def _forward(adapter, engine, leaves, spec):
  return adapter._p32_forward_group(  # pylint: disable=protected-access
      engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
  )


def _reverse(adapter, engine, leaves, spec, forward, chunk_program):
  reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
      engine, leaves, spec, jnp.ones_like(forward["logps"]),
      jnp.zeros_like(forward["entropy"]), replay=forward,
      chunk_program=chunk_program,
  )
  jax.block_until_ready(reverse["engine_gradients"])
  return reverse


def _same(a, b):
  for key in ("engine_gradients", "initial_cache_cotangents"):
    assert _tree_bytes(a[key]) == _tree_bytes(b[key]), key
  assert a["counts"] == b["counts"]


def _require_fma_free_cpu():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  if "--xla_cpu_max_isa=" not in os.environ.get("XLA_FLAGS", ""):
    pytest.skip(
        "bitwise comparison needs an FMA-free CPU ISA cap; run this file "
        f"in its own process (XLA_FLAGS gets {_ISA_CAP})"
    )


def test_reverse_chunk_program_is_bitwise_and_one_program_per_chunk():
  _require_fma_free_cpu()
  adapter, runner, spec, env = _fixture()
  with mock.patch.dict(os.environ, env, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P32_CHUNK_BATCH", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    forward = _forward(adapter, engine, leaves, spec)
    # The oracle: every chunk through the per-program loop.  It builds the
    # mapped pullbacks but records nothing for the chunk program.
    reference = _reverse(adapter, engine, leaves, spec, forward, False)
    assert engine.__dict__.get("_p32_reverse_chunk_out_shardings") is None
    # First admitted pass: the last chunk (reversed order) is the
    # bootstrap and records the output shardings; the other chunk already
    # runs as one program.
    first = _reverse(adapter, engine, leaves, spec, forward, True)
    assert engine.__dict__.get("_p32_reverse_chunk_out_shardings") is not None
    steady = _reverse(adapter, engine, leaves, spec, forward, True)
    with launch_budget.counting() as records:
      counted = _reverse(adapter, engine, leaves, spec, forward, True)
  _same(first, reference)
  _same(steady, reference)
  _same(counted, reference)
  programs = launch_budget.by_program(records)
  assert programs.get("bwd_chunk", 0) == spec["num_chunks"] == 2, programs
  # Nothing of the per-chunk reverse body launches on its own any more.
  for name in ("local_pullback", "bwd_logprob", "rebuild"):
    assert programs.get(name, 0) == 0, (name, programs)
  # Two compiled variants: the group's first reversed chunk (zero cache
  # cotangents made in-graph) and every later chunk.
  assert sorted(engine.__dict__["_p32_reverse_chunk_programs"]) == [
      (False,), (True,)
  ]


def test_reverse_chunk_program_follows_fresh_engine_leaves():
  """The program must read the engine state handed to each call, not the
  leaves it was built with (the state changes after every commit)."""
  _require_fma_free_cpu()
  adapter, runner, spec, env = _fixture()
  with mock.patch.dict(os.environ, env, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P32_CHUNK_BATCH", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    scaled = tuple(
        leaf * 1.5 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
        for leaf in leaves
    )
    forward = _forward(adapter, engine, leaves, spec)
    _reverse(adapter, engine, leaves, spec, forward, True)  # bootstrap
    _reverse(adapter, engine, leaves, spec, forward, True)  # builds the program
    forward_scaled = _forward(adapter, engine, scaled, spec)
    wrapped = _reverse(adapter, engine, scaled, spec, forward_scaled, True)
    oracle = _reverse(adapter, engine, scaled, spec, forward_scaled, False)
    stale = _reverse(adapter, engine, leaves, spec, forward, True)
  _same(wrapped, oracle)
  assert _tree_bytes(wrapped["engine_gradients"]) != _tree_bytes(
      stale["engine_gradients"]
  )


def test_reverse_chunk_program_is_not_admitted_without_the_kept_tape():
  """Without the kept tape the reverse rebuilds its own tape; that path
  keeps the per-program loop and never records or builds the program."""
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  adapter, runner, spec, env = _fixture()
  with mock.patch.dict(os.environ, env, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P32_CHUNK_BATCH", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    forward = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=True
    )
    reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
        engine, leaves, spec, jnp.ones_like(forward["logps"]),
        jnp.zeros_like(forward["entropy"]), replay=forward,
    )
    jax.block_until_ready(reverse["engine_gradients"])
  assert engine.__dict__.get("_p32_reverse_chunk_out_shardings") is None
  assert not engine.__dict__.get("_p32_reverse_chunk_programs")
