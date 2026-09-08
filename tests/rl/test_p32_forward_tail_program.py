"""Phase 9 K8/K9: the forward tail as one program, cached chunk-start
scalars, and one equality program for the replay check.

Gathers, selects, equality and a cached integer scalar are exact; the
gates are byte identity against the eager ops and the launch counts.
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
  spec = importlib.util.spec_from_file_location("forward_tail_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  case = module.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, _ = case._make_p32_group_adapter()  # pylint: disable=protected-access
  return adapter


def test_group_rows_program_matches_the_eager_gather_and_mask_bitwise():
  adapter = _adapter()
  rng = np.random.default_rng(21)
  flat_logps = jnp.asarray(rng.standard_normal((16, 64)).astype(np.float32))
  flat_entropies = jnp.asarray(rng.standard_normal((16, 64)).astype(np.float32))
  source_rows = jnp.asarray(rng.integers(0, 64, (16, 6), dtype=np.int32))
  completion_valid = jnp.asarray(rng.integers(0, 2, (16, 6)).astype(bool))
  want_logps = np.where(
      np.asarray(completion_valid),
      np.take_along_axis(np.asarray(flat_logps), np.asarray(source_rows), axis=1),
      np.zeros((16, 6), np.float32),
  )
  want_entropy = np.where(
      np.asarray(completion_valid),
      np.take_along_axis(np.asarray(flat_entropies), np.asarray(source_rows), axis=1),
      np.zeros((16, 6), np.float32),
  )
  fn = adapter._p32_group_rows_fn()  # pylint: disable=protected-access
  fn(flat_logps, flat_entropies, completion_valid, source_rows)  # compile
  with launch_budget.counting() as records:
    logps, entropy = fn(flat_logps, flat_entropies, completion_valid, source_rows)
    jax.block_until_ready((logps, entropy))
  assert len(records) == 1, launch_budget.by_program(records).most_common()
  assert np.asarray(logps).tobytes() == want_logps.tobytes()
  assert np.asarray(entropy).tobytes() == want_entropy.tobytes()
  assert logps.dtype == entropy.dtype == jnp.float32


def test_chunk_start_scalars_are_cached_by_value():
  adapter = _adapter()
  first = adapter._p32_start_scalar(8)  # pylint: disable=protected-access
  with launch_budget.counting() as records:
    again = adapter._p32_start_scalar(8)  # pylint: disable=protected-access
    other = adapter._p32_start_scalar(np.int64(12))  # pylint: disable=protected-access
  assert again is first
  assert int(first) == 8 and first.dtype == jnp.int32
  assert int(other) == 12 and other is not first
  # The cached value costs nothing; a new value costs its one creation.
  reuse = [name for _, name in records]
  assert len(reuse) <= 1, reuse
  with launch_budget.counting() as records:
    adapter._p32_start_scalar(12)  # pylint: disable=protected-access
  assert not records
