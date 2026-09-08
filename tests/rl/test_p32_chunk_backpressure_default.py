"""Phase 14 (tasks/v2_dispatch) / Phase C (tasks/v2_integrate): the
rank-parallel reverse bounds the host's dispatch lead without a flag.

Without CANON_P77_CHUNK_BACKPRESSURE one readiness wait follows every
_P77_CHUNK_LEAD_DEPTH-th chunk's accumulation and the group's last chunk,
on every carrier (FrozenLake included), whenever the reverse is
rank-parallel and the P76 device ticket is off; the waits read no value and
change no arithmetic, so the gradients are bitwise the ones of a loop that
drains only at the group's end.
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


def _one_pass(workload, wait_calls):
  budget = _load("test_p32_launch_budget.py")
  t71 = _load("test_p71_fwd_scan.py")
  adapter, runner = t71._group_adapter(True)  # pylint: disable=protected-access
  spec = budget._spec(adapter, 5)  # pylint: disable=protected-access
  env = dict(t71._SEGMENTED_ENV)  # pylint: disable=protected-access
  env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  env["CANON_P32_KEEP_TAPE"] = "stream"
  env.pop("CANON_P77_CHUNK_BACKPRESSURE", None)
  if workload is None:
    env.pop("CANON_P32_WORKLOAD", None)
  else:
    env["CANON_P32_WORKLOAD"] = workload
  original = canonical_qwen3_adapter._p77_wait_for_chunk_completion  # pylint: disable=protected-access

  def counting(completed):
    wait_calls.append(completed)
    return original(completed)

  with mock.patch.dict(os.environ, env, clear=False):
    for key in ("CANON_P71_SCAN", "CANON_P77_CHUNK_BACKPRESSURE"):
      os.environ.pop(key, None)
    if workload is None:
      os.environ.pop("CANON_P32_WORKLOAD", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    forward = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
    )
    dlogps = jnp.ones_like(forward["logps"])
    dentropy = jnp.zeros_like(forward["entropy"])
    with mock.patch.object(
        canonical_qwen3_adapter, "_p77_wait_for_chunk_completion", counting
    ):
      reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
          engine, leaves, spec, dlogps, dentropy, replay=forward
      )
    jax.block_until_ready(reverse["engine_gradients"])
  return int(spec["num_chunks"]), reverse["engine_gradients"]


def _bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def _expected_waits(num_chunks, depth):
  return sum(
      1
      for finished in range(1, num_chunks + 1)
      if finished % depth == 0 or finished == num_chunks
  )


def test_rank_parallel_reverse_waits_twice_per_chunk_on_every_carrier():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  # Budget 0: every carrier keeps the Phase 14 two waits per chunk.
  waits = []
  num_chunks, gradients = _one_pass(None, waits)
  assert len(waits) == 2 * num_chunks, (len(waits), num_chunks)
  # A FrozenLake carrier is bounded the same way (Phase 14 excluded it).
  frozen_waits = []
  _, frozen_gradients = _one_pass("frozenlake-p45-onehost-dp2-tp2", frozen_waits)
  assert len(frozen_waits) == 2 * num_chunks
  assert _bytes(gradients) == _bytes(frozen_gradients)


def test_lead_depth_two_drains_every_second_chunk_and_is_bitwise():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  depth = canonical_qwen3_adapter._P77_CHUNK_LEAD_DEPTH  # pylint: disable=protected-access
  assert depth == 2
  waits = []
  num_chunks, gradients = _one_pass(None, waits)
  lead_waits = []
  with mock.patch.object(canonical_qwen3_adapter, "_P77_LEAD_PACK_BUDGET_GIB", 1e9):
    _, lead_gradients = _one_pass(None, lead_waits)
  assert len(lead_waits) == _expected_waits(num_chunks, depth), (len(lead_waits), num_chunks)
  assert len(lead_waits) < len(waits)
  assert _bytes(gradients) == _bytes(lead_gradients)
  # A depth larger than the group drains only after its last chunk.
  loose_waits = []
  with mock.patch.object(canonical_qwen3_adapter, "_P77_LEAD_PACK_BUDGET_GIB", 1e9), \
       mock.patch.object(canonical_qwen3_adapter, "_P77_CHUNK_LEAD_DEPTH", 10**6):
    _, loose_gradients = _one_pass(None, loose_waits)
  assert len(loose_waits) == 1
  assert _bytes(gradients) == _bytes(loose_gradients)


def test_flag_print_stays_flag_only():
  import inspect
  source = inspect.getsource(
      canonical_qwen3_adapter.Qwen3EngineForwardAdapter._p32_reverse_group  # pylint: disable=protected-access
  )
  assert "if p77_flag_backpressure:" in source
  assert "chunk_lead_depth = _p77_chunk_lead_depth(" in source
  assert "chunk_index == 0" in source


def test_lead_depth_follows_the_pack_budget():
  depth = canonical_qwen3_adapter._p77_chunk_lead_depth  # pylint: disable=protected-access
  # Budget 0 (the certified first version): every pack keeps the two waits.
  assert canonical_qwen3_adapter._P77_LEAD_PACK_BUDGET_GIB == 0.0  # pylint: disable=protected-access
  assert depth(1.7) == 0 and depth(3.4) == 0 and depth(4.1) == 0 and depth(16.4) == 0
  with mock.patch.object(canonical_qwen3_adapter, "_P77_LEAD_PACK_BUDGET_GIB", 6.0):
    # 1.7B TP2 / TP4 and 8B TP8 packs would lead by two chunks ...
    assert depth(3.4) == 2 and depth(1.7) == 2 and depth(4.1) == 2 and depth(6.0) == 2
    # ... the 8B one-host packs keep the two waits at every boundary.
    assert depth(8.2) == 0 and depth(16.4) == 0 and depth(32.8) == 0
  gib = canonical_qwen3_adapter._p77_pack_gib  # pylint: disable=protected-access
  leaves = (jnp.zeros((1024, 1024), jnp.bfloat16), jnp.zeros((1024,), jnp.float32))
  assert gib(leaves, 2) == (1024 * 1024 + 1024) * 4 / 2 / 2**30


def test_big_pack_keeps_two_waits_per_chunk_and_is_bitwise():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  waits = []
  num_chunks, gradients = _one_pass(None, waits)
  bounded_waits = []
  with mock.patch.object(canonical_qwen3_adapter, "_P77_LEAD_PACK_BUDGET_GIB", 0.0):
    _, bounded_gradients = _one_pass(None, bounded_waits)
  assert len(bounded_waits) == 2 * num_chunks
  assert _bytes(gradients) == _bytes(bounded_gradients)
