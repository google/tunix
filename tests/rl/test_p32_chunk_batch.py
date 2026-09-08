"""k chunks of a group's forward run as one program (CANON_P32_CHUNK_BATCH=k).

Every chunk keeps its 256-wide programs; the batch program is a plain
Python loop over k chunk bodies traced into one straight-line graph, the
caches and the glue buffers carried in-graph, the per-chunk tape returned
as outputs.  Bitwise against the per-chunk loop on the paged fixture and
one forward program per k chunks.
"""

import importlib.util
import os
import pathlib
import types
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16"
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


def _run(batch, forward_block=False):
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = t71._two_chunk_spec(adapter)  # pylint: disable=protected-access
  # The fixture's chunk inputs are eager slices with the chunk start as
  # the layer "metadata"; give the batched program the same traced form.
  cache_sharding = jax.sharding.NamedSharding(
      runner.mesh, jax.sharding.PartitionSpec("data")
  )

  def chunk_inputs_traced(self, group_spec, start):
    bucket = self._sequence_bucket
    ids = jax.lax.dynamic_slice_in_dim(group_spec["packed_ids"], start, bucket, axis=1)
    targets = jax.lax.dynamic_slice_in_dim(group_spec["next_ids"], start, bucket, axis=1)
    return (
        jax.lax.with_sharding_constraint(ids.reshape(-1), cache_sharding),
        jax.lax.with_sharding_constraint(targets.reshape(-1), cache_sharding),
        start,
    )

  adapter._p32_chunk_inputs_traced = types.MethodType(chunk_inputs_traced, adapter)  # pylint: disable=protected-access
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  env["CANON_P32_CHUNK_BATCH"] = str(batch)
  if forward_block:
    env["CANON_P71_SCAN"] = "fwd_block"
  with mock.patch.dict(os.environ, env, clear=False):
    if not forward_block:
      os.environ["CANON_P71_SCAN"] = "off"  # per-layer forward (default is fwd_block)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    run = lambda: adapter._p32_forward_group(  # pylint: disable=protected-access,g-long-lambda
        engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
    )
    run()
    run()
    with launch_budget.counting() as records:
      forward = run()
    jax.block_until_ready(forward["logps"])
    reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
        engine, leaves, spec, jnp.ones_like(forward["logps"]),
        jnp.zeros_like(forward["entropy"]), replay=forward,
    )
  return spec["num_chunks"], forward, reverse, list(records)


@pytest.mark.parametrize("forward_block", [False, True], ids=["per-layer", "fwd_block"])
def test_two_chunk_batch_is_bitwise_and_one_forward_program(forward_block):
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  chunks, single, reverse_a, _ = _run(1, forward_block)
  chunks_b, batched, reverse_b, records = _run(2, forward_block)
  assert chunks == chunks_b == 2
  for key in ("logps", "entropy", "final_caches", "hidden_inputs", "final_hiddens"):
    assert _tree_bytes(batched[key]) == _tree_bytes(single[key]), key
  assert batched["counts"] == single["counts"]
  assert _tree_bytes(reverse_b["engine_gradients"]) == _tree_bytes(
      reverse_a["engine_gradients"]
  )
  programs = launch_budget.by_program(records)
  assert programs.get("forward_chunks", 0) == 1, programs
  for name in ("fwd_layer", "fwd_layers_block", "fwd_embed", "fwd_norm", "fwd_lm_head", "fwd_logprob"):
    assert programs.get(name, 0) == 0, (name, programs)


def test_chunk_batch_selector_parses_fail_closed():
  parse = canonical_qwen3_adapter._p32_chunk_batch  # pylint: disable=protected-access
  # tasks/v2_dispatch Phase 16: empty selects the certified two-chunk batch
  # unless the layer-scan rung is selected; '0' and '1' are the per-chunk loop.
  with mock.patch.dict(os.environ, {"CANON_P32_CHUNK_BATCH": ""}, clear=False):
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    assert parse() == 2
    with mock.patch.dict(os.environ, {"CANON_P28_LAYER_SCAN": "1"}, clear=False):
      assert parse() == 1
  for value in ("0", "1"):
    with mock.patch.dict(os.environ, {"CANON_P32_CHUNK_BATCH": value}, clear=False):
      assert parse() == 1
  with mock.patch.dict(os.environ, {"CANON_P32_CHUNK_BATCH": "8"}, clear=False):
    assert parse() == 8
  for value in ("2.5", "-1", "k", "65"):
    with mock.patch.dict(os.environ, {"CANON_P32_CHUNK_BATCH": value}, clear=False):
      with pytest.raises(canonical_qwen3_adapter.FunctionalMappingError):
        parse()


def test_batched_forward_follows_fresh_engine_leaves():
  """The batched program must read the engine state handed to each call,
  not the leaves it was built with (the state changes after every commit)."""
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = t71._two_chunk_spec(adapter)  # pylint: disable=protected-access
  cache_sharding = jax.sharding.NamedSharding(runner.mesh, jax.sharding.PartitionSpec("data"))

  def chunk_inputs_traced(self, group_spec, start):
    bucket = self._sequence_bucket
    ids = jax.lax.dynamic_slice_in_dim(group_spec["packed_ids"], start, bucket, axis=1)
    targets = jax.lax.dynamic_slice_in_dim(group_spec["next_ids"], start, bucket, axis=1)
    return (jax.lax.with_sharding_constraint(ids.reshape(-1), cache_sharding),
            jax.lax.with_sharding_constraint(targets.reshape(-1), cache_sharding), start)

  adapter._p32_chunk_inputs_traced = types.MethodType(chunk_inputs_traced, adapter)  # pylint: disable=protected-access
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  with mock.patch.dict(os.environ, env, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    scaled = tuple(leaf * 1.5 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf for leaf in leaves)

    def forward(batch, state):
      with mock.patch.dict(os.environ, {"CANON_P32_CHUNK_BATCH": str(batch)}, clear=False):
        return adapter._p32_forward_group(  # pylint: disable=protected-access
            engine, state, spec, keep_cache_inputs=True, keep_tape=True
        )

    forward(2, leaves)  # build the batched program with the ORIGINAL leaves
    batched = forward(2, scaled)
    single = forward(1, scaled)
    stale = forward(1, leaves)
  assert _tree_bytes(batched["logps"]) == _tree_bytes(single["logps"])
  assert _tree_bytes(batched["logps"]) != _tree_bytes(stale["logps"])
