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

"""Stage-1 tests for the real Gemma 4 vision tower port.

These verify WIRING and SHAPES and that the nnx module tree maps 1:1 onto the
real `google/gemma-4-e2b-it` checkpoint key names. They do NOT verify numeric
parity with HuggingFace — that requires the real weights + torch and is a
separate stage (see docs/gemma4_vision_port.md).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx
from tunix.models.gemma4 import vision_real as vr


def _state_keys(module) -> set[str]:
  """Flatten an nnx module's params to loader-style dotted keys."""
  _, state = nnx.split(module)
  keys = set()
  for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]:
    parts = []
    for k in path:
      name = getattr(k, "name", None)
      if name is None:
        idx = getattr(k, "idx", None)
        name = idx if idx is not None else getattr(k, "key", k)
      if name == "value":  # nnx.Param leaf suffix
        continue
      parts.append(str(name))
    keys.add(".".join(parts))
  return keys


def _square_positions(side: int) -> jnp.ndarray:
  xs, ys = jnp.meshgrid(jnp.arange(side), jnp.arange(side), indexing="xy")
  return jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)[None]


class VisionRealTest(absltest.TestCase):

  def test_forward_shapes_and_pooling(self):
    cfg = vr.Gemma4VisionConfig(num_hidden_layers=2)
    model = vr.Gemma4VisionModel(cfg, rngs=nnx.Rngs(0))

    side = 12  # 144 patches -> pool by 3x3 -> 16 soft tokens.
    p = side * side
    pos = _square_positions(side)
    px = jax.random.uniform(
        jax.random.PRNGKey(1), (1, p, 3 * cfg.patch_size**2)
    )

    soft, mask = model(px, pos)
    self.assertEqual(soft.shape, (1, 16, cfg.hidden_size))
    self.assertEqual(mask.shape, (1, 16))
    self.assertTrue(bool(jnp.all(mask)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(soft))))

  def test_projector_maps_to_text_dim(self):
    proj = vr.Gemma4MultimodalEmbedder(
        vision_hidden_size=768, text_hidden_size=1536, eps=1e-6, rngs=nnx.Rngs(0)
    )
    out = proj(jnp.ones((1, 16, 768)))
    self.assertEqual(out.shape, (1, 16, 1536))

  def test_module_tree_matches_checkpoint_keys(self):
    """Every param path must correspond to a real checkpoint key name.

    The real keys (verified from google/gemma-4-e2b-it) are, e.g.:
      model.vision_tower.patch_embedder.input_proj.weight
      model.vision_tower.patch_embedder.position_embedding_table
      model.vision_tower.encoder.layers.N.self_attn.q_proj.linear.weight
      model.vision_tower.encoder.layers.N.self_attn.q_norm.weight
      model.vision_tower.encoder.layers.N.mlp.gate_proj.linear.weight
      model.vision_tower.encoder.layers.N.input_layernorm.weight
    A loader maps `...weight`->`...kernel` (linears, transposed) and
    `...weight`->`...scale` (norms). This test pins the nnx side of that map.
    """
    cfg = vr.Gemma4VisionConfig(num_hidden_layers=1)
    keys = _state_keys(vr.Gemma4VisionModel(cfg, rngs=nnx.Rngs(0)))
    for expected in [
        "patch_embedder.input_proj.kernel",
        "patch_embedder.position_embedding_table",
        "encoder.layers.0.self_attn.q_proj.linear.kernel",
        "encoder.layers.0.self_attn.k_proj.linear.kernel",
        "encoder.layers.0.self_attn.v_proj.linear.kernel",
        "encoder.layers.0.self_attn.o_proj.linear.kernel",
        "encoder.layers.0.self_attn.q_norm.scale",
        "encoder.layers.0.self_attn.k_norm.scale",
        "encoder.layers.0.mlp.gate_proj.linear.kernel",
        "encoder.layers.0.mlp.up_proj.linear.kernel",
        "encoder.layers.0.mlp.down_proj.linear.kernel",
        "encoder.layers.0.input_layernorm.scale",
        "encoder.layers.0.post_attention_layernorm.scale",
        "encoder.layers.0.pre_feedforward_layernorm.scale",
        "encoder.layers.0.post_feedforward_layernorm.scale",
    ]:
      self.assertIn(expected, keys, f"missing param path: {expected}")

  def test_v_norm_has_no_scale(self):
    # HF v_norm uses with_scale=False -> no learnable param.
    cfg = vr.Gemma4VisionConfig(num_hidden_layers=1)
    keys = _state_keys(vr.Gemma4VisionModel(cfg, rngs=nnx.Rngs(0)))
    self.assertNotIn("encoder.layers.0.self_attn.v_norm.scale", keys)


if __name__ == "__main__":
  absltest.main()
