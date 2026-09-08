"""Pins the number of programs one grouped chunk pass launches (CPU).

Measured on the one-host xplanes, a chunk pass launched ~363 programs of
which only ~66 were the adapter's named programs; the rest were eager
glue (chunk inputs, glue writes, cotangent slices, scalar transfers).  The
budget here is for the toy fixture (two layers), so the named programs
are few and every eager launch stands out.
"""

import importlib.util
import os
import pathlib
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


def _spec(adapter, completion_tokens):
  """A sixteen-row group with three prompt + ``completion_tokens`` tokens."""
  row = jnp.arange(16, dtype=jnp.int32)[:, None]
  prompt = jnp.concatenate((1 + row % 2, 2 + row % 2, 1 + row % 3), axis=1)
  completion = jnp.concatenate(
      [1 + (row + k) % 3 for k in range(completion_tokens)], axis=1
  )
  return adapter._p32_group_spec(  # pylint: disable=protected-access
      prompt,
      completion,
      jnp.ones_like(prompt, dtype=bool),
      jnp.ones_like(completion, dtype=bool),
      1.0,
  )


def _launches(completion_tokens):
  """Eager launches attributed to the chunk-loop functions for one pass."""
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = _spec(adapter, completion_tokens)
  # The fixture stubs the chunk inputs with eager slices; the production
  # builder is one program and is pinned by test_p32_chunk_inputs.py.
  stub = adapter._p32_group_chunk_inputs  # pylint: disable=protected-access

  def quiet_stub(group_spec, chunk_index):
    with launch_budget.paused():
      return stub(group_spec, chunk_index)

  adapter._p32_group_chunk_inputs = quiet_stub  # pylint: disable=protected-access
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  with mock.patch.dict(os.environ, env, clear=False):
    # The per-chunk loops are the subject: select them explicitly (the
    # defaults are the forward block and the two-chunk batch).
    os.environ["CANON_P71_SCAN"] = "off"
    os.environ["CANON_P32_CHUNK_BATCH"] = "1"
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)

    def run():
      forward = adapter._p32_forward_group(  # pylint: disable=protected-access
          engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
      )
      dlogps = jnp.ones_like(forward["logps"])
      dentropy = jnp.zeros_like(forward["entropy"])
      reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
          engine, leaves, spec, dlogps, dentropy, replay=forward
      )
      jax.block_until_ready(reverse["engine_gradients"])
      return forward, reverse

    run()
    run()  # every program compiled and cached
    with launch_budget.counting() as records:
      run()
  loops = launch_budget.in_function(records, "_p32_forward_group") + (
      launch_budget.in_function(records, "_p32_reverse_group")
  )
  named = set(launch_budget.named_programs(loops))
  eager = [entry for entry in loops if entry not in named]
  return spec["num_chunks"], eager


def test_chunk_loops_issue_few_eager_programs_per_chunk_and_per_group():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  one_chunk, eager_one = _launches(completion_tokens=1)
  two_chunks, eager_two = _launches(completion_tokens=3)
  assert (one_chunk, two_chunks) == (1, 2)
  per_chunk = len(eager_two) - len(eager_one)
  per_group = len(eager_one) - per_chunk
  # Per chunk: one chunk_start scalar transfer per direction and the
  # gradient-pack accumulate; the rows glue, cotangent slices and glue
  # writes are inside the rows programs now.
  assert per_chunk <= 3, (per_chunk, launch_budget.by_site(eager_two).most_common())
  # Per group: the fixture's own fresh-cache stub (4), the glue zero
  # buffers, the cotangent scatter and the final gather/where pair.
  assert per_group <= 16, (per_group, launch_budget.by_site(eager_one).most_common())


def test_glue_buffers_follow_the_engine_input_sharding_not_the_trainer_axis_name():
  """The one-host engine mesh is named ('dp', 'tp'); the trainer axis is 'data'."""
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  import numpy as np
  t71 = _load("test_p71_fwd_scan.py")
  adapter, _ = t71._group_adapter(True)  # pylint: disable=protected-access
  engine_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:16]).reshape(8, 2), ("dp", "tp")
  )
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      engine_mesh, jax.sharding.PartitionSpec("dp")
  )
  assert adapter._dp_axis == "data"  # pylint: disable=protected-access
  like = jax.device_put(jnp.zeros((16, 4)), adapter._input_sharding)  # pylint: disable=protected-access
  logps, entropies = adapter._p32_glue_zeros(like)  # pylint: disable=protected-access
  for buffer in (logps, entropies):
    assert buffer.sharding.mesh == engine_mesh
    assert buffer.sharding.spec == jax.sharding.PartitionSpec("dp")
  dlogps = jnp.ones((16, 6), jnp.float32)
  valid = jnp.ones((16, 6), bool)
  rows = jnp.tile(jnp.arange(6, dtype=jnp.int32)[None, :], (16, 1))
  flat_dlogps, _ = adapter._p32_flat_cotangents_fn(dlogps)(dlogps, dlogps, valid, rows)  # pylint: disable=protected-access
  assert flat_dlogps.sharding.mesh == engine_mesh
  # A replicated engine input sharding (no data axis) is honoured too.
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      engine_mesh, jax.sharding.PartitionSpec(None)
  )
  adapter.__dict__.pop("_p32_glue_zero_programs", None)
  logps, _ = adapter._p32_glue_zeros(like)  # pylint: disable=protected-access
  assert logps.sharding.spec == jax.sharding.PartitionSpec(None)


def test_signature_difference_names_the_tree_leaf_and_sharding():
  from tunix.rl.canonical_qwen3_adapter import _p59_signature_difference
  treedef = jax.tree.structure(((1, 2), (3,), (4,)))
  leaves = (((1,), "f32"), ((2,), "f32"), ((3,), "bf16"), ((4,), "f32"))
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]).reshape(2, 1), ("dp", "tp"))
  a = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  b = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  parts = (("trainer_state", 2), ("engine_gradients", 1), ("staged_accumulator", 1))
  same = ((treedef, leaves), (a, a, a, a), "plan")
  moved = ((treedef, leaves), (a, a, b, a), "plan")
  assert _p59_signature_difference(same, moved, parts).startswith(
      "engine_gradients[0] sharding"
  )
  reshaped = ((treedef, leaves[:3] + (((5,), "f32"),)), (a, a, a, a), "plan")
  assert _p59_signature_difference(same, reshaped, parts).startswith(
      "staged_accumulator[0] shape/dtype"
  )
  assert _p59_signature_difference(same, same, parts).startswith("no visible")


def test_replicated_sharding_spellings_share_one_signature_and_respell_without_copy():
  from tunix.rl.canonical_qwen3_adapter import (
      _p59_canonical_sharding,
      _p59_respell_replicated,
  )
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
  spelled = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  canonical = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  assert spelled != canonical  # the raw comparison the guard used to make
  assert _p59_canonical_sharding(spelled) == canonical
  assert _p59_canonical_sharding(canonical) == canonical
  sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
  assert _p59_canonical_sharding(sharded) is sharded
  padded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp", None))
  short = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp"))
  assert padded != short and _p59_canonical_sharding(padded) == short
  assert _p59_canonical_sharding(short) is short
  leaf = jax.device_put(jnp.arange(8.0), spelled)
  tree = _p59_respell_replicated({"a": leaf, "b": 3, "c": jax.device_put(jnp.ones((4,)), sharded)})
  assert tree["a"].sharding == canonical
  assert np.asarray(tree["a"]).tobytes() == np.asarray(leaf).tobytes()
  assert tree["b"] == 3 and tree["c"].sharding == sharded
