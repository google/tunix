#!/usr/bin/env python3
"""Small-device tests for the explicit DP release-candidate programs."""

from __future__ import annotations

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax

from tunix.models.qwen3 import model as model_lib

from probe_qwen8b_rc import _build_optimizer_state
from probe_qwen8b_rc import _commit_program
from probe_qwen8b_rc import _make_inputs
from probe_qwen8b_rc import _put_memory_kind
from probe_qwen8b_rc import _release_candidate_model_config
from probe_qwen8b_rc import _replica_samples_exact
from probe_qwen8b_rc import _sample_tree_sha256
from probe_qwen8b_rc import _state_memory_kinds
from probe_qwen8b_rc import _stream_fixed_rank_gradient
from probe_qwen8b_rc import _tree_exact
from probe_qwen8b_rc import _tree_health
from probe_qwen8b_rc import build_dp_programs


class ReleaseCandidateConfigTest(unittest.TestCase):

  def test_bounded_contract_uses_dense_reference_attention(self):
    config = _release_candidate_model_config()
    self.assertFalse(config.use_flash_attention)
    self.assertEqual(
        config.shd_config,
        model_lib.ShardingConfig.get_data_parallel_sharding(),
    )


class ProbeRCTest(unittest.TestCase):

  def setUp(self):
    if len(jax.devices()) != 4:
      self.skipTest("the explicit DP2xTP2 smoke requires four devices")
    self.mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(2, 2), ("dp", "tp"))
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
    with jax.set_mesh(self.mesh):
      model = model_lib.Qwen3(config, rngs=nnx.Rngs(params=7))
    graphdef, state = nnx.split(model, nnx.Param)
    shardings = nnx.get_named_sharding(state, self.mesh)
    state = jax.tree.map(jax.device_put, state, shardings)
    self.graphdef = graphdef
    self.state = state
    self.config = config

  def test_forward_and_fixed_gradient_are_real_and_exact(self):
    plain, rank_value_and_grad = build_dp_programs(
        graphdef=self.graphdef,
        mesh=self.mesh,
        global_batch=4,
        vocab_size=self.config.vocab_size,
    )
    inputs = _make_inputs(
        self.mesh, global_batch=4, seq_len=4, vocab_size=self.config.vocab_size
    )
    plain_rows = plain(self.state, *inputs)
    loss, grad_rows, gradients, fingerprints = _stream_fixed_rank_gradient(
        rank_value_and_grad, self.state, inputs, dp_size=2
    )
    jax.block_until_ready(
        (plain_rows, loss, grad_rows, gradients)
    )
    self.assertEqual(np.asarray(plain_rows).shape, np.asarray(grad_rows).shape)
    health = _tree_health(gradients)
    self.assertTrue(health["finite"])
    self.assertGreater(health["nonzero"], 0)
    self.assertGreater(health["norm"], 0.0)
    self.assertEqual(len(fingerprints), 2)
    self.assertNotEqual(fingerprints[0], fingerprints[1])
    self.assertTrue(_replica_samples_exact(gradients))

    def global_objective(candidate):
      model = nnx.merge(self.graphdef, candidate)
      logits, _ = model(inputs[0], inputs[1], None, inputs[2])
      rows = logits[:, -1, :].astype(jnp.float32)
      return jnp.sum(jnp.square(rows)) / jnp.asarray(
          4 * self.config.vocab_size, jnp.float32
      )

    stock = jax.jit(jax.grad(global_objective))(self.state)
    jax.block_until_ready(stock)
    for fixed_leaf, stock_leaf in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(stock), strict=True
    ):
      np.testing.assert_allclose(
          np.asarray(fixed_leaf), np.asarray(stock_leaf), rtol=2.0e-5, atol=2.0e-6
      )

  def test_three_updates_move_optimizer_state_and_parameters(self):
    _, rank_value_and_grad = build_dp_programs(
        graphdef=self.graphdef,
        mesh=self.mesh,
        global_batch=4,
        vocab_size=self.config.vocab_size,
    )
    inputs = _make_inputs(
        self.mesh, global_batch=4, seq_len=4, vocab_size=self.config.vocab_size
    )
    tx = optax.adamw(
        learning_rate=1.0e-3, b1=0.9, b2=0.95, weight_decay=0.0
    )
    optimizer_state = _build_optimizer_state(self.state, self.mesh, tx)
    self.assertEqual(_state_memory_kinds(optimizer_state), ("pinned_host",))
    commit = _commit_program(tx)
    params = self.state
    parameter_fingerprints = []
    for _ in range(3):
      _, _, gradients, _ = _stream_fixed_rank_gradient(
          rank_value_and_grad, params, inputs, dp_size=2
      )
      optimizer_state = _put_memory_kind(optimizer_state, "device")
      self.assertEqual(_state_memory_kinds(optimizer_state), ("device",))
      previous_params = params
      params, optimizer_state = commit(params, optimizer_state, gradients)
      jax.block_until_ready((params, optimizer_state))
      self.assertFalse(_tree_exact(previous_params, params))
      optimizer_state = _put_memory_kind(optimizer_state, "pinned_host")
      self.assertEqual(
          _state_memory_kinds(optimizer_state), ("pinned_host",)
      )
      parameter_fingerprints.append(_sample_tree_sha256(params))
    self.assertEqual(len(set(parameter_fingerprints)), 3)


if __name__ == "__main__":
  unittest.main()
