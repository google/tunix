"""CPU smoke tests for the P32 model-init materializer."""

from __future__ import annotations

import pathlib
import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from tunix.models.qwen3 import model as model_lib


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import probe_qwen8b_init  # pylint: disable=g-import-not-at-top


def _tiny_config() -> model_lib.ModelConfig:
  return model_lib.ModelConfig(
      num_layers=1,
      vocab_size=32,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=1,
      rope_theta=10_000,
      norm_eps=1.0e-6,
      dtype=jnp.bfloat16,
      param_dtype=jnp.float32,
      shd_config=model_lib.ShardingConfig.get_data_parallel_sharding(),
  )


class ModelInitProbeTest(unittest.TestCase):

  def test_tiny_model_optimizer_and_accumulator_materialize(self):
    if len(jax.devices()) != 1:
      self.skipTest("tiny materialization smoke requires one CPU device")
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ("dp", "tp"))
    config, model, optimizer, accumulator, inventory = (
        probe_qwen8b_init.materialize_training_state(
            mesh, optimizer_memory_kind="device", config=_tiny_config()
        )
    )
    self.assertEqual(config.num_layers, 1)
    self.assertEqual(len(jax.tree.leaves(model)), 14)
    self.assertEqual(len(jax.tree.leaves(optimizer)), 29)
    self.assertEqual(len(jax.tree.leaves(accumulator)), 14)
    self.assertEqual(set(inventory), {"model", "optimizer", "accumulator"})
    for summary in inventory.values():
      self.assertEqual(summary["dp_partitioned_leaves"], 0)
      self.assertEqual(summary["memory_kinds"], ("device",))

  def test_index_shape_resolves_open_slices(self):
    self.assertEqual(
        probe_qwen8b_init._index_shape(
            (slice(None), slice(2, 8, 2)), (4, 10)
        ),
        (4, 3),
    )

  def test_topology_rejects_wrong_device_count(self):
    with self.assertRaisesRegex(RuntimeError, "requires exactly"):
      probe_qwen8b_init._topology_mesh(list(jax.devices()), 2, 1)


if __name__ == "__main__":
  unittest.main()
