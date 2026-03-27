from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from tunix.models.qwen2 import model as qwen2_model


class Qwen2KvCacheDtypeTest(absltest.TestCase):

  def test_attention_cache_update_casts_to_cache_dtype(self):
    cfg = qwen2_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=8,
        hidden_dim=16,
        num_heads=2,
        head_dim=4,
        num_kv_heads=1,
        rope_theta=10000,
        norm_eps=1e-6,
        shd_config=qwen2_model.ShardingConfig.get_default_sharding(),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_flash_attention=False,
    )

    attn = qwen2_model.Attention(cfg, rngs=nnx.Rngs(0))
    x = jnp.zeros((1, 1, cfg.embed_dim), dtype=jnp.float32)
    sin = jnp.zeros((1, 1, cfg.head_dim // 2), dtype=jnp.float32)
    cos = jnp.ones((1, 1, cfg.head_dim // 2), dtype=jnp.float32)
    cache = {
        'end_index': jnp.array([0], dtype=jnp.int32),
        'v': jnp.zeros((1, 4, cfg.num_kv_heads, cfg.head_dim), dtype=jnp.bfloat16),
        'k': jnp.zeros((1, 4, cfg.num_kv_heads, cfg.head_dim), dtype=jnp.bfloat16),
    }

    new_cache, _ = attn.block(x, cache, None, sin, cos)

    self.assertIsNotNone(new_cache)
    self.assertEqual(new_cache['v'].dtype, cache['v'].dtype)
    self.assertEqual(new_cache['k'].dtype, cache['k'].dtype)
    self.assertEqual(new_cache['v'].shape, cache['v'].shape)
    self.assertEqual(new_cache['k'].shape, cache['k'].shape)
    self.assertEqual(int(new_cache['end_index'][0]), 1)


if __name__ == '__main__':
  absltest.main()
