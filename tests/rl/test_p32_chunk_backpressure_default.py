"""Phase 14 (tasks/v2_dispatch): the GSM8K carrier's rank-parallel reverse
bounds the host's dispatch lead at every chunk boundary without a flag.

The two P77 readiness waits (before the whole-tree start/add and after it)
run on every chunk of every group whenever the reverse is rank-parallel,
the P76 device ticket is off and the workload is not a FrozenLake carrier;
they read no value and change no arithmetic, so the gradients are bitwise
the ones of the unbounded loop.
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


def test_gsm8k_rank_parallel_reverse_waits_twice_per_chunk_and_is_bitwise():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  waits = []
  num_chunks, gradients = _one_pass(None, waits)
  assert len(waits) == 2 * num_chunks, (len(waits), num_chunks)
  # A FrozenLake workload keeps the flag-only P77 behaviour: no waits.
  frozen_waits = []
  _, frozen_gradients = _one_pass("frozenlake-p45-onehost-dp2-tp2", frozen_waits)
  assert frozen_waits == []
  assert _bytes(gradients) == _bytes(frozen_gradients)


def test_flag_print_stays_flag_only():
  import inspect
  source = inspect.getsource(
      canonical_qwen3_adapter.Qwen3EngineForwardAdapter._p32_reverse_group  # pylint: disable=protected-access
  )
  assert "if p77_flag_backpressure:" in source
  assert "_p32_workload_is_frozenlake()" in source
