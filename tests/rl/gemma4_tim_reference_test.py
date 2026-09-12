"""Full 35-layer real-model construction; reduced width is not target proof."""

import dataclasses
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from examples.frozenlake.gemma4_tim import reference
from tunix.models.gemma4 import model as gemma


def tiny_e2b():
  sharding = gemma.ShardingConfig.get_default_sharding()
  sharding = dataclasses.replace(sharding, **{
      f.name: (None,) * len(getattr(sharding, f.name))
      for f in dataclasses.fields(sharding) if isinstance(getattr(sharding, f.name), tuple)
  })
  config = gemma.ModelConfig.gemma4_e2b_it(sharding)
  config.num_embed, config.embed_dim, config.hidden_dim = 19, 16, 24
  config.override_kv_shared_ffw_hidden = 48  # Preserve shared FFN doubling.
  config.num_heads, config.num_kv_heads = 2, 1
  config.head_dim, config.global_key_size = 8, 16
  config.per_layer_input_dim, config.sliding_window_size = 8, 2
  config.vision_encoder, config.audio_encoder = None, None
  model = gemma.Gemma4(config, rngs=nnx.Rngs(17))
  rng = np.random.default_rng(31)
  state = jax.tree.map(lambda x: jnp.asarray(rng.normal(0, .08, x.shape), x.dtype)
                       if x.ndim > 1 else x, nnx.state(model, nnx.Param))
  nnx.update(model, state)
  return model


class ReferenceTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = tiny_e2b()

  def test_actual_35_layer_e2b_graph_and_doubled_ffn(self):
    cfg = gemma.ModelConfig.gemma4_e2b_it()
    actual = gemma.create_kv_cache_sharing_patterns(
        35, cfg.frac_shared_layers, True, True, cfg.attention_pattern * 7)
    self.assertEqual(tuple(actual), reference.E2B_KV_PRODUCERS)
    self.assertEqual(tuple(actual[:15]), tuple(range(15)))
    self.assertEqual(set(actual[15:]), {13, 14})
    self.assertEqual(len(self.model.layers), 35)
    self.assertEqual(cfg.override_kv_shared_ffw_hidden, cfg.hidden_dim * 2)

  def test_invalid_graph_and_chunk_continuation_refused(self):
    for types, shared in ((('local', 'global'), 1), (('bad',), 0), (('local',), 1)):
      with self.assertRaises(ValueError):
        reference.kv_producers(types, shared)
    tokens = jnp.array([[1, 2]], jnp.int32)
    mask = jnp.ones_like(tokens, dtype=jnp.bool_)
    for kwargs in ({"cache": {}}, {"chunk_offset": 1}):
      with self.assertRaisesRegex(ValueError, "full recompute"):
        reference.action_logps(self.model, tokens, mask, mask, **kwargs)

  def test_full_depth_gradient_producers_consumers_dead_row_and_sign(self):
    tokens = jnp.array([[2, 4, 7, 0], [3, 6, 8, 10]], jnp.int32)
    valid = jnp.ones_like(tokens, dtype=jnp.bool_)
    actions = jnp.array([[False, True, True, True], [False] * 4])
    def loss(model, sign=1.):
      return -sign * reference.action_logps(model, tokens, valid, actions).sum() / 3
    value, grads = nnx.value_and_grad(loss)(self.model)
    self.assertTrue(np.isfinite(value))
    leaves = jax.tree.leaves(grads)
    self.assertTrue(all(np.isfinite(x).all() for x in leaves))
    self.assertGreater(float(jnp.linalg.norm(grads.embedder.input_embedding.value)), 0)
    for producer in (13, 14):
      self.assertGreater(float(jnp.linalg.norm(grads.layers[producer].attn.kv_einsum.w.value)), 0)
    for consumer in range(15, 35):
      np.testing.assert_array_equal(grads.layers[consumer].attn.kv_einsum.w.value, 0)
    def one_loss(model):
      return -reference.action_logps(model, tokens[:1], valid[:1], actions[:1]).sum() / 3
    single = nnx.grad(one_loss)(self.model)
    for a, b in zip(leaves, jax.tree.leaves(single), strict=True):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    opposite = nnx.grad(loss)(self.model, -1.)
    for a, b in zip(leaves, jax.tree.leaves(opposite), strict=True):
      np.testing.assert_array_equal(a, -b)

  def test_eos_equal_pad_is_scored_and_empty_action_gradient_zero(self):
    # ID0 is a valid action here. Do not infer padding from token ID.
    tokens = jnp.array([[2, 4, 0]], jnp.int32)
    valid = jnp.ones_like(tokens, dtype=jnp.bool_)
    actions = jnp.array([[False, False, True]])
    scored = reference.action_logps(self.model, tokens, valid, actions)
    self.assertNotEqual(float(scored[0, 1]), 0)
    self.assertEqual(float(scored[0, 0]), 0)
    def empty(model):
      return reference.action_logps(model, tokens, valid, jnp.zeros_like(actions)).sum()
    value, grads = nnx.value_and_grad(empty)(self.model)
    self.assertEqual(float(value), 0)
    for leaf in jax.tree.leaves(grads):
      np.testing.assert_array_equal(leaf, 0)

  def test_tied_head_and_softcap_once(self):
    hidden = jnp.ones((1, 1, 16)) * 100
    self.assertFalse(hasattr(self.model, "lm_head"))
    raw = self.model.embedder.decode(hidden)
    np.testing.assert_array_equal(self.model.compute_final_logits(hidden), jnp.tanh(raw / 30) * 30)

  def test_real_shared_attention_detach_fault(self):
    rng = np.random.default_rng(44)
    x = jnp.asarray(rng.normal(size=(1, 4, 16)), jnp.float32)
    kv = {k: jnp.asarray(rng.normal(size=(1, 4, 1, 8)), jnp.float32) for k in ("k", "v")}
    positions = jnp.arange(4)[None]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=bool))
    def forward(values, detach=False):
      if detach:
        values = jax.tree.map(jax.lax.stop_gradient, values)
      return self.model.layers[15].attn(x, positions, None, mask, kv_shared_cache=values)[1]
    np.testing.assert_array_equal(forward(kv), forward(kv, True))
    healthy = jax.grad(lambda v: forward(v).sum())(kv)
    broken = jax.grad(lambda v: forward(v, True).sum())(kv)
    for key in ("k", "v"):
      self.assertGreater(float(jnp.linalg.norm(healthy[key])), 0)
      np.testing.assert_array_equal(broken[key], 0)


if __name__ == "__main__":
  unittest.main()
