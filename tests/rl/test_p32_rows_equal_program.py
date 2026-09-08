"""Phase 9 K9: the per-group replay equality check as one program.

The reverse loop compared each group's replayed logprobs with the
forward's by an eager array_equal (an equal and an all per group); the
flag now comes from one program per group, produced while both arrays
are alive, and is read at the same point as before.
"""

import importlib.util
import os
import pathlib

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


def _adapter():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  path = pathlib.Path(__file__).with_name("canonical_qwen3_adapter_test.py")
  spec = importlib.util.spec_from_file_location("rows_equal_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  case = module.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, _ = case._make_p32_group_adapter()  # pylint: disable=protected-access
  return adapter


def test_pair_equal_matches_array_equal_in_one_launch():
  adapter = _adapter()
  rng = np.random.default_rng(3)
  left = jnp.asarray(rng.standard_normal((16, 6)).astype(np.float32))
  same = jnp.asarray(np.asarray(left))
  moved = left.at[5, 1].add(1e-3)
  wider = jnp.zeros((16, 7), jnp.float32)
  adapter._p32_pair_equal(left, same)  # pylint: disable=protected-access  # compile
  with launch_budget.counting() as records:
    flag_same = adapter._p32_pair_equal(left, same)  # pylint: disable=protected-access
    flag_moved = adapter._p32_pair_equal(left, moved)  # pylint: disable=protected-access
    jax.block_until_ready((flag_same, flag_moved))
  assert len(records) == 2, launch_budget.by_program(records).most_common()
  assert bool(flag_same) is True and bool(flag_moved) is False
  assert bool(flag_same) == bool(jnp.array_equal(left, same))
  assert bool(flag_moved) == bool(jnp.array_equal(left, moved))
  assert bool(adapter._p32_pair_equal(left, wider)) is False  # pylint: disable=protected-access
  assert bool(jnp.array_equal(left, wider)) is False
