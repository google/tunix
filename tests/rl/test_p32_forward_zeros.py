"""One program for a group's fresh caches and zero glue buffers (Phase 11).

``_p32_forward_zeros`` replaces the fresh-caches program plus the
glue-zeros program at the start of every group's forward with one
program whose outputs keep exactly the shardings, shapes and dtypes the
two producers pinned; without the cache geometry (the CPU fixtures
replace ``_fresh_caches``) it falls back to the two producers.
"""

import os
import types

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


def _adapter(with_geometry=True):
  if len(jax.devices()) < 4:
    pytest.skip("requires four forced CPU devices")
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:4]).reshape(2, 2), ("dp", "tp")
  )
  adapter = object.__new__(canonical_qwen3_adapter.Qwen3EngineForwardAdapter)
  adapter._data_size = 2  # pylint: disable=protected-access
  adapter._sequence_bucket = 4  # pylint: disable=protected-access
  adapter._max_model_len = 10  # pylint: disable=protected-access
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      mesh, jax.sharding.PartitionSpec("dp", None)
  )
  adapter._runner = types.SimpleNamespace(kv_caches=[None, None, None])  # pylint: disable=protected-access
  adapter._cache_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      mesh, jax.sharding.PartitionSpec("dp", None, "tp")
  )
  adapter._cache_dtype = jnp.bfloat16  # pylint: disable=protected-access
  if with_geometry:
    adapter._cache_shape = (8, 4, 2)  # pylint: disable=protected-access
  return adapter


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def test_forward_zeros_is_one_program_with_the_producers_shardings():
  adapter = _adapter()
  reference_caches = tuple(adapter._fresh_caches())  # pylint: disable=protected-access
  reference_glue = adapter._p32_glue_zeros(None)  # pylint: disable=protected-access
  adapter._p32_forward_zeros()  # pylint: disable=protected-access  # build
  with launch_budget.counting() as records:
    caches, logps, entropy = adapter._p32_forward_zeros()  # pylint: disable=protected-access
  programs = launch_budget.by_program(records)
  assert programs == {"forward_zeros": 1}, programs
  assert len(caches) == 3
  for got, want in zip(caches, reference_caches):
    assert got.shape == want.shape and got.dtype == want.dtype
    assert got.sharding == want.sharding
  for got, want in zip((logps, entropy), reference_glue):
    assert got.shape == want.shape == (2, 12) and got.dtype == jnp.float32
    assert got.sharding == want.sharding
  assert _tree_bytes((caches, logps, entropy)) == _tree_bytes(
      (reference_caches, *reference_glue)
  )
  assert not any(np.asarray(leaf).any() for leaf in jax.tree.leaves(caches))


def test_forward_zeros_falls_back_to_the_two_producers_without_geometry():
  adapter = _adapter(with_geometry=False)
  adapter._fresh_caches = types.MethodType(  # pylint: disable=protected-access
      lambda self: [jnp.zeros((4, 1), jnp.float32) for _ in range(2)], adapter
  )
  caches, logps, entropy = adapter._p32_forward_zeros()  # pylint: disable=protected-access
  assert len(caches) == 2 and caches[0].shape == (4, 1)
  assert logps.shape == entropy.shape == (2, 12)
  assert not adapter.__dict__.get("_p32_forward_zero_programs")
