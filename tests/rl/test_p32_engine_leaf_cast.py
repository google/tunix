"""The trainer-to-engine dtype casts run as ONE program per update.

The mapping cast one leaf at a time (one eager convert per leaf, 310
launches per update on the one-host census).  A convert is a single
elementwise op, so one program casting every leaf changes no value; each
output keeps its input's sharding as the eager cast did.
"""

import importlib.util
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import launch_budget  # noqa: E402  (patches JAX at import; before the adapter)
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import canonical_qwen3_adapter


def _fixture_module():
  path = pathlib.Path(__file__).with_name("canonical_qwen3_adapter_test.py")
  spec = importlib.util.spec_from_file_location("leaf_cast_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _mapping_inputs(module):
  source = module._state({  # pylint: disable=protected-access
      "trainer": {
          "embedding": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 7,
          "head": jnp.asarray([5.0 / 3], jnp.float32),
          "layers": jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 9,
          "norm": jnp.asarray([3.0 / 7], jnp.float32),
      }
  })
  target = module._state({  # pylint: disable=protected-access
      "engine": {
          "embedding": jnp.zeros((3, 2), jnp.bfloat16),
          "head": jnp.zeros((1,), jnp.bfloat16),
          "layers": {
              "0": {"weight": jnp.zeros((4,), jnp.bfloat16)},
              "1": {"weight": jnp.zeros((4,), jnp.bfloat16)},
          },
          "norm": jnp.zeros((1,), jnp.bfloat16),
      }
  })
  mappings = {
      "trainer.embedding": ("engine.embedding", (None, None)),
      "trainer.head": ("engine.head", (None,)),
      "trainer.layers": ("engine.layers.*.weight", ("layer", None)),
      "trainer.norm": ("engine.norm", (None,)),
  }
  return source, target, mappings


def _map(source, target, mappings):
  return canonical_qwen3_adapter.map_trainer_state_to_engine_leaves(
      trainer_state=source,
      engine_state_contract=target,
      key_mappings=mappings,
      transpose_keys={"embedding": (1, 0)},
  )


def test_engine_leaf_casts_are_one_program_and_bitwise():
  module = _fixture_module()
  source, target, mappings = _mapping_inputs(module)
  # Reference: the per-leaf eager cast the mapping used to do.
  batched = canonical_qwen3_adapter._p32_cast_leaves  # pylint: disable=protected-access
  try:
    canonical_qwen3_adapter._p32_cast_leaves = lambda values, dtypes: tuple(  # pylint: disable=protected-access,g-long-lambda
        value.astype(dtype) for value, dtype in zip(values, dtypes)
    )
    reference = _map(source, target, mappings)
  finally:
    canonical_qwen3_adapter._p32_cast_leaves = batched  # pylint: disable=protected-access
  _map(source, target, mappings)  # compile the cast program
  with launch_budget.counting() as records:
    mapped = _map(source, target, mappings)
    jax.block_until_ready(mapped.leaves)
  programs = launch_budget.by_program(records)
  assert programs.get("cast_engine_leaves", 0) == 1, programs.most_common()
  assert programs.get("convert_element_type", 0) == 0, programs.most_common()
  assert mapped.paths == reference.paths
  for path, got, want in zip(mapped.paths, mapped.leaves, reference.leaves):
    assert got.dtype == want.dtype == jnp.bfloat16, path
    assert got.shape == want.shape, path
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), path
    assert got.sharding == want.sharding, path


def test_dtype_equal_and_host_leaves_keep_the_per_leaf_path():
  module = _fixture_module()
  source, target, mappings = _mapping_inputs(module)
  # A target that already has the source dtype casts nothing.
  same = module._state({  # pylint: disable=protected-access
      "engine": {
          "embedding": jnp.zeros((3, 2), jnp.float32),
          "head": jnp.zeros((1,), jnp.float32),
          "layers": {
              "0": {"weight": jnp.zeros((4,), jnp.float32)},
              "1": {"weight": jnp.zeros((4,), jnp.float32)},
          },
          "norm": jnp.zeros((1,), jnp.float32),
      }
  })
  _map(source, same, mappings)
  with launch_budget.counting() as records:
    mapped = _map(source, same, mappings)
    jax.block_until_ready(mapped.leaves)
  assert launch_budget.by_program(records).get("cast_engine_leaves", 0) == 0
  assert all(leaf.dtype == jnp.float32 for leaf in mapped.leaves)
