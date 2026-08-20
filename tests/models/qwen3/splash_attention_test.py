# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numeric correctness tests for Qwen3 splash-attention implementations.

Uses Qwen3-0.6B model configuration as the base configuration for testing
the Attention module structure against Qwen3's own native reference implementation:
1. Qwen3 Base Reference: Qwen3's own native attention implementation
   (`use_flash_attention=False`, pure einsum + softmax without any JAX or Tokamax splash attention).
2. Tokamax Splash Attention: `use_flash_attention=True`, `splash_attention_impl=TOKAMAX`.
3. JAX Splash Attention: `use_flash_attention=True`, `splash_attention_impl=JAX`.

Tests both MQA (num_kv_heads=1) and GQA (num_kv_heads=8) configurations.

Compares both splash implementations against Qwen3's own base attention output
and reports error metrics.

Splash attention is a TPU-only Pallas kernel, so these tests are skipped when
no TPU device is available.
"""

import copy

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.qwen3 import model as model_lib


def _has_tpu() -> bool:
  try:
    return any(dev.platform == "tpu" for dev in jax.devices())
  except RuntimeError:
    return False


def _setup_mesh() -> jax.sharding.Mesh:
  num_devices = len(jax.devices())
  mesh_config = [(num_devices, 1), ("fsdp", "tp")]
  return jax.make_mesh(
      *mesh_config,
      axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]),
  )


class SplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not _has_tpu():
      self.skipTest("Splash attention requires a TPU device.")

  @parameterized.named_parameters(
      dict(
          testcase_name="mqa",
          num_kv_heads=1,
      ),
      dict(
          testcase_name="gqa",
          num_kv_heads=8,
      ),
  )
  def test_splash_attention_vs_qwen3_base(self, num_kv_heads):
    # 1. Base reference: Qwen3's own native attention implementation
    # (use_flash_attention=False, pure einsum + softmax without splash attention).
    config_base = model_lib.ModelConfig.qwen3_0p6b()
    config_base.num_kv_heads = num_kv_heads
    config_base.use_flash_attention = False

    # 2. Tokamax splash attention implementation.
    config_tokamax = copy.deepcopy(config_base)
    config_tokamax.use_flash_attention = True
    config_tokamax.splash_attention_impl = model_lib.SplashAttentionImpl.TOKAMAX
    config_tokamax.flash_attention_block_size = 512

    # 3. JAX splash attention implementation.
    config_jax = copy.deepcopy(config_base)
    config_jax.use_flash_attention = True
    config_jax.splash_attention_impl = model_lib.SplashAttentionImpl.JAX
    config_jax.flash_attention_block_size = 512

    # Instantiate the three Attention structures and synchronize weights.
    attn_base = model_lib.Attention(config_base, rngs=nnx.Rngs(0))
    attn_tokamax = model_lib.Attention(config_tokamax, rngs=nnx.Rngs(0))
    attn_jax = model_lib.Attention(config_jax, rngs=nnx.Rngs(0))

    _, base_state = nnx.split(attn_base)
    nnx.update(attn_tokamax, base_state)
    nnx.update(attn_jax, base_state)

    b, t, d = 2, 1024, config_base.embed_dim
    x = jax.random.normal(jax.random.PRNGKey(42), (b, t, d), dtype=config_base.dtype)
    attn_mask = jnp.tril(jnp.ones((b, t, t), dtype=jnp.bool_))
    positions = jnp.tile(jnp.arange(t)[None, :], (b, 1))

    mesh = _setup_mesh()
    with jax.set_mesh(mesh):
      _, out_base = attn_base(x, positions, None, attn_mask)
      _, out_tokamax = attn_tokamax(x, positions, None, attn_mask)
      _, out_jax = attn_jax(x, positions, None, attn_mask)

    self.assertEqual(out_tokamax.shape, out_base.shape)
    self.assertEqual(out_jax.shape, out_base.shape)

    tokamax_max_err = float(jnp.max(jnp.abs(out_tokamax - out_base)))
    tokamax_mean_err = float(jnp.mean(jnp.abs(out_tokamax - out_base)))

    jax_max_err = float(jnp.max(jnp.abs(out_jax - out_base)))
    jax_mean_err = float(jnp.mean(jnp.abs(out_jax - out_base)))

    closer_max = "tokamax" if tokamax_max_err < jax_max_err else "jax"
    closer_mean = "tokamax" if tokamax_mean_err < jax_mean_err else "jax"

    logging.info(
        "[Qwen3 (num_kv_heads=%d)] Error vs Qwen3 native base attention:\n"
        "  tokamax splash: max_abs=%.4e, mean_abs=%.4e\n"
        "  jax splash:     max_abs=%.4e, mean_abs=%.4e\n"
        "  -> %s is closer to Qwen3 native base by max error\n"
        "  -> %s is closer to Qwen3 native base by mean error",
        num_kv_heads,
        tokamax_max_err,
        tokamax_mean_err,
        jax_max_err,
        jax_mean_err,
        closer_max,
        closer_mean,
    )

    np.testing.assert_allclose(
        out_tokamax,
        out_base,
        atol=7e-3,
        rtol=7e-3,
        err_msg=(
            f"tokamax splash attention diverged from Qwen3 native base"
            f" attention for num_kv_heads={num_kv_heads}"
        ),
    )
    np.testing.assert_allclose(
        out_jax,
        out_base,
        atol=7e-3,
        rtol=7e-3,
        err_msg=(
            f"jax splash attention diverged from Qwen3 native base"
            f" attention for num_kv_heads={num_kv_heads}"
        ),
    )


if __name__ == "__main__":
  absltest.main()
