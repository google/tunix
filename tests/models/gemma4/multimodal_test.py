# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 4 tests: Gemma4Multimodal wiring (merge + forward).

JAX-only (no torch/checkpoint). Validates that vision soft tokens are scattered
into image-token positions, that text positions are untouched, and that the
image actually influences the language-model output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import multimodal as mm_lib
from tunix.models.gemma4 import vision_real

_IMAGE_TOKEN_ID = 5
_NUM_SOFT = 4  # 4x4 patches, pool 2x2


def _build():
  text_cfg = model_lib.ModelConfig.gemma4_e2b()  # text_only -> no SigLIP
  text_cfg.num_layers = 1
  text_cfg.embed_dim = 64
  text_cfg.hidden_dim = 128
  text_cfg.num_heads = 4
  text_cfg.head_dim = 16
  text_cfg.num_kv_heads = 1
  text_cfg.frac_shared_layers = 0.0
  text_model = model_lib.Gemma4(text_cfg, rngs=nnx.Rngs(0))

  vcfg = vision_real.Gemma4VisionConfig(
      hidden_size=32, intermediate_size=64, num_hidden_layers=1,
      num_attention_heads=4, num_key_value_heads=4, head_dim=8, patch_size=4,
      position_embedding_size=64, pooling_kernel_size=2,
  )
  vision = vision_real.Gemma4VisionStack(
      vcfg, text_hidden_size=text_cfg.embed_dim, rngs=nnx.Rngs(1)
  )
  model = mm_lib.Gemma4Multimodal(
      text_model, vision, image_token_id=_IMAGE_TOKEN_ID
  )
  return model, vcfg


def _square_positions(side: int) -> jnp.ndarray:
  xs, ys = jnp.meshgrid(jnp.arange(side), jnp.arange(side), indexing="xy")
  return jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)[None]


def _image_inputs(vcfg, seed):
  side = 4
  pos = _square_positions(side)
  px = jax.random.uniform(
      jax.random.PRNGKey(seed), (1, side * side, 3 * vcfg.patch_size**2)
  )
  return px, pos.astype(jnp.int32)


class Gemma4MultimodalTest(absltest.TestCase):

  def test_merge_places_soft_tokens_at_image_positions(self):
    model, vcfg = _build()
    px, pos = _image_inputs(vcfg, seed=2)
    # 4 image placeholders at positions 1..4; text elsewhere.
    tokens = jnp.array([[2, 5, 5, 5, 5, 10, 11, 12]], dtype=jnp.int32)

    merged = model.encode_multimodal_inputs(tokens, px, pos)
    text_only = model.text_model.embedder.encode(tokens)
    soft, _ = model.vision(px, pos)

    self.assertEqual(merged.shape, text_only.shape)
    # Image positions must hold the projected soft tokens...
    np.testing.assert_allclose(
        np.asarray(merged[0, 1:5]), np.asarray(soft[0, :4]), rtol=1e-5, atol=1e-5
    )
    # ...and text positions must be unchanged.
    np.testing.assert_allclose(
        np.asarray(merged[0, 5:]), np.asarray(text_only[0, 5:]), rtol=1e-5, atol=1e-5
    )

  def test_forward_shape_and_image_affects_logits(self):
    model, vcfg = _build()
    tokens = jnp.array([[2, 5, 5, 5, 5, 10, 11, 12]], dtype=jnp.int32)
    px_a, pos = _image_inputs(vcfg, seed=2)
    px_b, _ = _image_inputs(vcfg, seed=99)

    logits_a, _ = model(tokens, px_a, pos)
    logits_b, _ = model(tokens, px_b, pos)

    self.assertEqual(logits_a.shape, (1, 8, model.text_model.config.num_embed))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits_a))))
    # A text position AFTER the image must react to the image content (causal
    # attention from the soft tokens), so different images => different logits.
    self.assertFalse(
        np.allclose(np.asarray(logits_a[0, -1]), np.asarray(logits_b[0, -1]),
                    rtol=1e-4, atol=1e-4)
    )

  def test_attention_mask_bidirectional_over_image_span(self):
    model, _ = _build()
    tokens = jnp.array([[2, 5, 5, 5, 5, 10]], dtype=jnp.int32)
    mask = model.get_attention_mask(tokens)[0]
    # First image soft-token (pos 1) should attend forward to the rest of the
    # image span (pos 2..4) -> bidirectional within the image.
    self.assertTrue(bool(mask[1, 4]))
    # A text token after the image stays causal: pos 5 cannot see nothing beyond.
    self.assertTrue(bool(mask[5, 4]))


if __name__ == "__main__":
  absltest.main()
