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

"""Key-coverage tests for the Gemma 4 vision safetensors mapping.

Verifies WITHOUT real weights that every real checkpoint vision key maps to an
existing model param, that audio keys and clip buffers are skipped, and that
every loadable model param is covered by exactly one checkpoint key. This is the
check that catches the bugs that broke the SigLIP loader (wrong prefix, the
`.linear.` nesting, missing transposes, uninitialised params).
"""

from __future__ import annotations

import jax
from absl.testing import absltest
from flax import nnx
from tunix.models.gemma4 import vision_params_safetensors as vp
from tunix.models.gemma4 import vision_real
from tunix.utils import torch_utils

_NUM_LAYERS = 2  # structurally identical to the real 16-layer tower


def _model_param_keys() -> set[str]:
  cfg = vision_real.Gemma4VisionConfig(num_hidden_layers=_NUM_LAYERS)
  model = vision_real.Gemma4VisionStack(cfg, text_hidden_size=1536, rngs=nnx.Rngs(0))
  _, state = nnx.split(model)
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


def _real_checkpoint_keys():
  """Generate the real `google/gemma-4-*-it` vision/audio/clip key names."""
  vision, skip = [], []
  vision.append("model.vision_tower.patch_embedder.input_proj.weight")
  vision.append("model.vision_tower.patch_embedder.position_embedding_table")
  vision.append("model.embed_vision.embedding_projection.weight")
  for n in range(_NUM_LAYERS):
    base = f"model.vision_tower.encoder.layers.{n}"
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
      vision.append(f"{base}.self_attn.{proj}.linear.weight")
      # e2b clip buffers riding alongside each linear -> must be skipped.
      for buf in ("input_min", "input_max", "output_min", "output_max"):
        skip.append(f"{base}.self_attn.{proj}.{buf}")
    for norm in ("q_norm", "k_norm"):
      vision.append(f"{base}.self_attn.{norm}.weight")
    for proj in ("gate_proj", "up_proj", "down_proj"):
      vision.append(f"{base}.mlp.{proj}.linear.weight")
      for buf in ("input_min", "input_max", "output_min", "output_max"):
        skip.append(f"{base}.mlp.{proj}.{buf}")
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
      vision.append(f"{base}.{norm}.weight")
  # Out-of-scope towers that must be skipped, not errored.
  skip.append("model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight")
  skip.append("model.embed_audio.embedding_projection.weight")
  return vision, skip


class VisionMappingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = vision_real.Gemma4VisionConfig(num_hidden_layers=_NUM_LAYERS)
    self.mapping = vp.vision_key_mapping(self.cfg)
    self.param_keys = _model_param_keys()
    self.vision_keys, self.skip_keys = _real_checkpoint_keys()

  def test_every_vision_key_maps_to_an_existing_param(self):
    for k in self.vision_keys:
      jax_key, _ = torch_utils.torch_key_to_jax_key(self.mapping, k)
      self.assertIn(
          jax_key,
          self.param_keys,
          f"checkpoint key {k!r} -> {jax_key!r} which is not a model param",
      )

  def test_linears_transpose_norms_do_not(self):
    jk, t = torch_utils.torch_key_to_jax_key(
        self.mapping, "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"
    )
    self.assertEqual(jk, "vision_tower.encoder.layers.0.mlp.gate_proj.linear.kernel")
    self.assertEqual(t, ((1, 0), None))  # transpose
    jk, t = torch_utils.torch_key_to_jax_key(
        self.mapping, "model.vision_tower.encoder.layers.0.input_layernorm.weight"
    )
    self.assertEqual(jk, "vision_tower.encoder.layers.0.input_layernorm.scale")
    self.assertIsNone(t)  # no transform

  def test_audio_and_clip_buffers_are_skipped(self):
    # The generic loader skips keys that match no pattern (ValueError -> skip).
    for k in self.skip_keys:
      with self.assertRaises(ValueError):
        torch_utils.torch_key_to_jax_key(self.mapping, k)

  def test_no_double_match(self):
    # Each real vision key must match exactly one pattern (loader requirement).
    for k in self.vision_keys:
      torch_utils.torch_key_to_jax_key(self.mapping, k)  # raises if != 1 match

  def test_every_loadable_param_is_covered(self):
    # No uninitialised params: every model param (except the derived rope buffer)
    # must be the target of exactly one checkpoint key.
    mapped = set()
    for k in self.vision_keys:
      jax_key, _ = torch_utils.torch_key_to_jax_key(self.mapping, k)
      mapped.add(jax_key)
    uncovered = self.param_keys - mapped
    # rotary inv_freq is computed at call time, not stored -> shouldn't appear.
    uncovered = {k for k in uncovered if "inv_freq" not in k}
    self.assertEqual(uncovered, set(), f"params with no checkpoint key: {uncovered}")


if __name__ == "__main__":
  absltest.main()
