"""The grouped forward runs all layers of a chunk as one unrolled program.

``CANON_P71_SCAN=fwd_block`` traces the per-layer forward programs of a
chunk into one straight-line program (no scan, no parameter or cache
stacking): every layer's parameters and cache stay separate operands, the
outputs are the per-layer caches and layer inputs the reverse already
consumes.  Bitwise against the per-layer path on the paged fixture, and
one program per chunk instead of one per layer.
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


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _forward(mode):
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = t71._two_chunk_spec(adapter)  # pylint: disable=protected-access
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  env["CANON_P32_CHUNK_BATCH"] = "1"  # one chunk per program (default is 2)
  if mode:
    env["CANON_P71_SCAN"] = mode
  with mock.patch.dict(os.environ, env, clear=False):
    if not mode:
      os.environ["CANON_P71_SCAN"] = "off"  # the per-layer oracle (default is fwd_block)
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


def test_forward_block_is_bitwise_and_one_program_per_chunk():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  chunks, per_layer, reverse_a, _ = _forward("")
  chunks_b, block, reverse_b, records = _forward("fwd_block")
  assert chunks == chunks_b
  for key in ("logps", "entropy", "final_caches", "hidden_inputs", "final_hiddens"):
    assert _tree_bytes(block[key]) == _tree_bytes(per_layer[key]), key
  assert block["counts"]["layer_forward"] == per_layer["counts"]["layer_forward"]
  # The reverse consumes the block's tape exactly as the per-layer one.
  assert _tree_bytes(reverse_b["engine_gradients"]) == _tree_bytes(
      reverse_a["engine_gradients"]
  )
  programs = launch_budget.by_program(records)
  assert programs.get("fwd_layer", 0) == 0, programs
  assert programs.get("fwd_layers_block", 0) == chunks, programs


def test_forward_block_selector_leaves_the_reverse_ladder_alone():
  # tasks/v2_dispatch Phase 16: the forward block is the default rung;
  # 'off' keeps the per-layer forward; neither touches the reverse ladder.
  with mock.patch.dict(os.environ, {}, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    assert canonical_qwen3_adapter._p71_scan_mode() == "fwd_block"  # pylint: disable=protected-access
  with mock.patch.dict(os.environ, {"CANON_P71_SCAN": "off"}, clear=False):
    assert canonical_qwen3_adapter._p71_scan_mode() == ""  # pylint: disable=protected-access
  with mock.patch.dict(os.environ, {"CANON_P71_SCAN": "fwd_block"}, clear=False):
    assert canonical_qwen3_adapter._p71_scan_mode() == "fwd_block"  # pylint: disable=protected-access
