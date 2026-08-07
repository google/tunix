#!/usr/bin/env python3
"""Small-device tests for the DP checkpoint-forward program."""

from __future__ import annotations

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from tunix.models.qwen3 import model as model_lib

from probe_qwen8b_rc import _make_inputs
from probe_qwen8b_rc import build_forward_program


class ProbeRCTest(unittest.TestCase):

  def test_forward_is_real_repeatable_and_non_mutating(self):
    if len(jax.devices()) != 4:
      self.skipTest("the explicit DP2xTP2 smoke requires four devices")
    mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(2, 2), ("dp", "tp"))
    config = model_lib.ModelConfig(
        num_layers=2,
        vocab_size=64,
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        head_dim=8,
        num_kv_heads=2,
        rope_theta=10_000,
        norm_eps=1.0e-6,
        use_tied_embedding=True,
        shd_config=model_lib.ShardingConfig.get_data_parallel_sharding(),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_flash_attention=False,
    )
    with jax.set_mesh(mesh):
      model = model_lib.Qwen3(config, rngs=nnx.Rngs(params=7))
    graphdef, state = nnx.split(model, nnx.Param)
    state = jax.tree.map(
        jax.device_put, state, nnx.get_named_sharding(state, mesh)
    )
    inputs = _make_inputs(mesh, global_batch=4, seq_len=4, vocab_size=64)
    forward = build_forward_program(graphdef)
    first = forward(state, *inputs)
    second = forward(state, *inputs)
    jax.block_until_ready((first, second))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    self.assertEqual(first.shape, (4, 64))


if __name__ == "__main__":
  unittest.main()
