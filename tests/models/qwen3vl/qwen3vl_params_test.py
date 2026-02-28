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

"""Tests for Qwen3-VL model parameters and vision weight loading."""

import os
import shutil
import tempfile
import unittest

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models import dummy_model_creator
from tunix.models.qwen3vl import model as qwen3vl_model
from tunix.models.qwen3vl import params as qwen3vl_params
from tunix.models.qwen3vl.vision import VisionModelConfig
from tunix.tests import lora_params_test_base
from tunix.tests import test_common


def _make_small_vision_config() -> VisionModelConfig:
  """Small VisionModelConfig suitable for unit tests."""
  return VisionModelConfig(
      hidden_size=16,
      out_hidden_size=16,
      depth=2,
      num_heads=2,
      intermediate_size=32,
      patch_size=4,
      temporal_patch_size=2,
      spatial_merge_size=2,
      window_size=8,
      in_channels=3,
      num_position_embeddings=16,  # 4×4 grid
      deepstack_visual_indexes=(0,),
      mrope_section=(2, 1, 1),
      image_pad_id=0,
  )


def _make_small_config() -> qwen3vl_model.ModelConfig:
  """Small ModelConfig with vision, suitable for unit tests."""
  return qwen3vl_model.ModelConfig(
      num_layers=2,
      vocab_size=64,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=2,
      rope_theta=10000,
      norm_eps=1e-6,
      use_tied_embedding=True,
      vision_config=_make_small_vision_config(),
  )


def _extract_state(
    model: qwen3vl_model.Qwen3VL,
    cfg: qwen3vl_model.ModelConfig,
) -> dict[str, np.ndarray]:
  """Return model weights serialised in HuggingFace safetensors key format.

  This is the inverse of the transform mapping in qwen3vl/params.py:
  each NNX tensor is reshaped / transposed back to the original PT shape so
  that params.create_model_from_safe_tensors can reload it correctly.
  """
  v_cfg = cfg.vision_config
  state: dict[str, np.ndarray] = {}

  # ── vision: patch embed ────────────────────────────────────────────────────
  # NNX kernel (pixel_volume, hidden_size)
  # → HF weight (hidden_size, temporal, channels, patch_h, patch_w)
  kernel = np.array(model.visual.patch_embed.proj.kernel.value)
  kernel = kernel.reshape(
      v_cfg.temporal_patch_size,
      v_cfg.in_channels,
      v_cfg.patch_size,
      v_cfg.patch_size,
      v_cfg.hidden_size,
  )
  state['model.visual.patch_embed.proj.weight'] = kernel.transpose(
      4, 0, 1, 2, 3
  )
  state['model.visual.patch_embed.proj.bias'] = np.array(
      model.visual.patch_embed.proj.bias.value
  )

  # ── vision: positional embedding ───────────────────────────────────────────
  state['model.visual.pos_embed.weight'] = np.array(
      model.visual.pos_embed.embedding.value
  )

  # ── vision: transformer blocks ─────────────────────────────────────────────
  for i, block in enumerate(model.visual.blocks):
    pfx = f'model.visual.blocks.{i}'
    # Attention: NNX kernel (in, out) → HF weight (out, in)
    state[f'{pfx}.attn.qkv.weight'] = np.array(
        block.attn.qkv_proj.kernel.value
    ).T
    state[f'{pfx}.attn.qkv.bias'] = np.array(block.attn.qkv_proj.bias.value)
    state[f'{pfx}.attn.proj.weight'] = np.array(
        block.attn.out_proj.kernel.value
    ).T
    state[f'{pfx}.attn.proj.bias'] = np.array(block.attn.out_proj.bias.value)
    # MLP: same transpose convention
    state[f'{pfx}.mlp.linear_fc1.weight'] = np.array(
        block.mlp.linear1.kernel.value
    ).T
    state[f'{pfx}.mlp.linear_fc1.bias'] = np.array(block.mlp.linear1.bias.value)
    state[f'{pfx}.mlp.linear_fc2.weight'] = np.array(
        block.mlp.linear2.kernel.value
    ).T
    state[f'{pfx}.mlp.linear_fc2.bias'] = np.array(block.mlp.linear2.bias.value)
    # LayerNorm: scale → weight, bias → bias (no reshape)
    state[f'{pfx}.norm1.weight'] = np.array(block.norm1.scale.value)
    state[f'{pfx}.norm1.bias'] = np.array(block.norm1.bias.value)
    state[f'{pfx}.norm2.weight'] = np.array(block.norm2.scale.value)
    state[f'{pfx}.norm2.bias'] = np.array(block.norm2.bias.value)

  # ── vision: deepstack mergers ───────────────────────────────────────────────
  for i, merger in enumerate(model.visual.deepstack_mergers):
    pfx = f'model.visual.deepstack_merger_list.{i}'
    state[f'{pfx}.linear_fc1.weight'] = np.array(
        merger.linear_fc1.kernel.value
    ).T
    state[f'{pfx}.linear_fc1.bias'] = np.array(merger.linear_fc1.bias.value)
    state[f'{pfx}.linear_fc2.weight'] = np.array(
        merger.linear_fc2.kernel.value
    ).T
    state[f'{pfx}.linear_fc2.bias'] = np.array(merger.linear_fc2.bias.value)
    state[f'{pfx}.norm.weight'] = np.array(merger.norm.scale.value)
    state[f'{pfx}.norm.bias'] = np.array(merger.norm.bias.value)

  # ── vision: final merger ────────────────────────────────────────────────────
  state['model.visual.merger.linear_fc1.weight'] = np.array(
      model.visual.merger.linear_fc1.kernel.value
  ).T
  state['model.visual.merger.linear_fc1.bias'] = np.array(
      model.visual.merger.linear_fc1.bias.value
  )
  state['model.visual.merger.linear_fc2.weight'] = np.array(
      model.visual.merger.linear_fc2.kernel.value
  ).T
  state['model.visual.merger.linear_fc2.bias'] = np.array(
      model.visual.merger.linear_fc2.bias.value
  )
  state['model.visual.merger.norm.weight'] = np.array(
      model.visual.merger.norm.scale.value
  )
  state['model.visual.merger.norm.bias'] = np.array(
      model.visual.merger.norm.bias.value
  )

  # ── text: token embedding ───────────────────────────────────────────────────
  state['model.embed_tokens.weight'] = np.array(
      model.embedder.input_embedding.value
  )

  # ── text: final RMSNorm ─────────────────────────────────────────────────────
  state['model.norm.weight'] = np.array(model.final_norm.w.value)

  # ── text: decoder layers ────────────────────────────────────────────────────
  for i, layer in enumerate(model.layers):
    pfx = f'model.layers.{i}'
    state[f'{pfx}.input_layernorm.weight'] = np.array(
        layer.input_layernorm.w.value
    )
    state[f'{pfx}.post_attention_layernorm.weight'] = np.array(
        layer.post_attention_layernorm.w.value
    )
    state[f'{pfx}.self_attn.q_norm.weight'] = np.array(
        layer.attn.q_norm.w.value
    )
    state[f'{pfx}.self_attn.k_norm.weight'] = np.array(
        layer.attn.k_norm.w.value
    )

    # Einsum w (embed_dim, num_heads, head_dim)
    # → HF weight (num_heads*head_dim, embed_dim)
    w = np.array(layer.attn.q_proj.w.value)
    state[f'{pfx}.self_attn.q_proj.weight'] = w.reshape(cfg.embed_dim, -1).T
    w = np.array(layer.attn.k_proj.w.value)
    state[f'{pfx}.self_attn.k_proj.weight'] = w.reshape(cfg.embed_dim, -1).T
    w = np.array(layer.attn.v_proj.w.value)
    state[f'{pfx}.self_attn.v_proj.weight'] = w.reshape(cfg.embed_dim, -1).T

    # Einsum w (num_heads, head_dim, embed_dim)
    # → HF weight (embed_dim, num_heads*head_dim)
    w = np.array(layer.attn.o_proj.w.value)
    state[f'{pfx}.self_attn.o_proj.weight'] = w.reshape(-1, cfg.embed_dim).T

    # nnx.Linear kernel (in, out) → HF weight (out, in)
    state[f'{pfx}.mlp.gate_proj.weight'] = np.array(
        layer.mlp.gate_proj.kernel.value
    ).T
    state[f'{pfx}.mlp.up_proj.weight'] = np.array(
        layer.mlp.up_proj.kernel.value
    ).T
    state[f'{pfx}.mlp.down_proj.weight'] = np.array(
        layer.mlp.down_proj.kernel.value
    ).T

  # ── text: lm_head (only when not using tied embeddings) ────────────────────
  if not cfg.use_tied_embedding:
    # Einsum w (embed_dim, vocab_size) → HF weight (vocab_size, embed_dim)
    state['lm_head.weight'] = np.array(model.lm_head.w.value).T

  return state


# ──────────────────────────────────────────────────────────────────────────────
# LoRA / parameter merging tests
# ──────────────────────────────────────────────────────────────────────────────


class Qwen3VLParamsTest(lora_params_test_base.LoraParamsTestBase):
  """Tests for Qwen3-VL text-decoder parameters and LoRA weight merging.

  Mirrors tests/models/qwen3/qwen_params_test.py but for the multimodal
  variant.  LoRA is applied to the text decoder only; vision weights are
  included in the checkpoint so that the merged model can be reloaded via
  params.create_model_from_safe_tensors.
  """

  def create_config(self):
    return _make_small_config()

  def get_model_class(self):
    return qwen3vl_model.Qwen3VL

  def get_lora_module_path(self) -> str:
    return (
        '.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj'
    )

  def get_projection_keys(self, layer_idx: int) -> list[str]:
    pfx = f'model.layers.{layer_idx}'
    return [
        f'{pfx}.self_attn.q_proj.weight',
        f'{pfx}.self_attn.k_proj.weight',
        f'{pfx}.self_attn.v_proj.weight',
        f'{pfx}.self_attn.o_proj.weight',
        f'{pfx}.mlp.gate_proj.weight',
        f'{pfx}.mlp.up_proj.weight',
        f'{pfx}.mlp.down_proj.weight',
    ]

  def save_merged_model(self, lora_model):
    qwen3vl_params.save_lora_merged_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        lora_model=lora_model,
        rank=self.rank,
        alpha=self.alpha,
    )

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    return qwen3vl_params.create_model_from_safe_tensors(
        file_dir=checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def create_checkpoint(self, model) -> str:
    os.makedirs(self.base_checkpoint_dir, exist_ok=True)
    safe_np.save_file(
        _extract_state(model, self.config),
        os.path.join(self.base_checkpoint_dir, 'model.safetensors'),
    )
    with open(os.path.join(self.base_checkpoint_dir, 'config.json'), 'w') as f:
      f.write('{"model_type": "qwen3_vl"}')
    return self.base_checkpoint_dir

  # ── Qwen3-VL has a 3D M-RoPE position tensor and extra vision args ──────────

  def _create_test_inputs(self):
    batch_size = 2
    seq_len = 4
    input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.ones((3, batch_size, seq_len), dtype=jnp.int32)
    return input_tokens, positions, None  # attention_mask=None

  def _run_forward_pass(self, model, input_tokens, positions, attention_mask):
    return model(
        input_tokens=input_tokens,
        positions=positions,
        pixel_values=None,
        vision_precomputed=None,
        cache=None,
        attention_mask=attention_mask,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Vision parameter round-trip tests
# ──────────────────────────────────────────────────────────────────────────────


class Qwen3VLVisionParamsTest(absltest.TestCase):
  """Tests that vision encoder weights survive a save→load round trip.

  Each test creates a randomly initialised model, serialises it to the
  HuggingFace safetensors format via _extract_state, reloads it with
  params.create_model_from_safe_tensors, and checks that the NNX tensors
  are numerically identical to the originals.
  """

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoint')
    os.makedirs(self.checkpoint_dir)
    self.config = _make_small_config()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)
    super().tearDown()

  def _make_model(self) -> qwen3vl_model.Qwen3VL:
    return dummy_model_creator.create_dummy_model(
        model_class=qwen3vl_model.Qwen3VL,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
        random_seed=42,
        scale=0.01,
    )

  def _save_and_reload(
      self, model: qwen3vl_model.Qwen3VL
  ) -> qwen3vl_model.Qwen3VL:
    safe_np.save_file(
        _extract_state(model, self.config),
        os.path.join(self.checkpoint_dir, 'model.safetensors'),
    )
    with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
      f.write('{"model_type": "qwen3_vl"}')
    return qwen3vl_params.create_model_from_safe_tensors(
        file_dir=self.checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def test_patch_embed_round_trip(self):
    """Vision patch-embed Conv3d→Linear weight mapping survives a round trip."""
    model = self._make_model()
    reloaded = self._save_and_reload(model)

    np.testing.assert_array_almost_equal(
        np.array(model.visual.patch_embed.proj.kernel.value),
        np.array(reloaded.visual.patch_embed.proj.kernel.value),
        decimal=6,
        err_msg='patch_embed kernel mismatch',
    )
    np.testing.assert_array_almost_equal(
        np.array(model.visual.patch_embed.proj.bias.value),
        np.array(reloaded.visual.patch_embed.proj.bias.value),
        decimal=6,
        err_msg='patch_embed bias mismatch',
    )

  def test_pos_embed_round_trip(self):
    """Vision positional embedding survives a round trip."""
    model = self._make_model()
    reloaded = self._save_and_reload(model)

    np.testing.assert_array_almost_equal(
        np.array(model.visual.pos_embed.embedding.value),
        np.array(reloaded.visual.pos_embed.embedding.value),
        decimal=6,
        err_msg='pos_embed mismatch',
    )

  def test_vision_blocks_round_trip(self):
    """Vision block attention, MLP and LayerNorm weights survive a round trip."""
    model = self._make_model()
    reloaded = self._save_and_reload(model)

    for i in range(len(model.visual.blocks)):
      orig = model.visual.blocks[i]
      loaded = reloaded.visual.blocks[i]

      for name, o_arr, l_arr in [
          (
              'attn.qkv_proj.kernel',
              orig.attn.qkv_proj.kernel,
              loaded.attn.qkv_proj.kernel,
          ),
          (
              'attn.qkv_proj.bias',
              orig.attn.qkv_proj.bias,
              loaded.attn.qkv_proj.bias,
          ),
          (
              'attn.out_proj.kernel',
              orig.attn.out_proj.kernel,
              loaded.attn.out_proj.kernel,
          ),
          (
              'attn.out_proj.bias',
              orig.attn.out_proj.bias,
              loaded.attn.out_proj.bias,
          ),
          (
              'mlp.linear1.kernel',
              orig.mlp.linear1.kernel,
              loaded.mlp.linear1.kernel,
          ),
          ('mlp.linear1.bias', orig.mlp.linear1.bias, loaded.mlp.linear1.bias),
          (
              'mlp.linear2.kernel',
              orig.mlp.linear2.kernel,
              loaded.mlp.linear2.kernel,
          ),
          ('mlp.linear2.bias', orig.mlp.linear2.bias, loaded.mlp.linear2.bias),
          ('norm1.scale', orig.norm1.scale, loaded.norm1.scale),
          ('norm1.bias', orig.norm1.bias, loaded.norm1.bias),
          ('norm2.scale', orig.norm2.scale, loaded.norm2.scale),
          ('norm2.bias', orig.norm2.bias, loaded.norm2.bias),
      ]:
        np.testing.assert_array_almost_equal(
            np.array(o_arr.value),
            np.array(l_arr.value),
            decimal=6,
            err_msg=f'block[{i}].{name} mismatch',
        )

  def test_vision_merger_round_trip(self):
    """Vision final merger weights survive a round trip."""
    model = self._make_model()
    reloaded = self._save_and_reload(model)

    for attr in ('linear_fc1', 'linear_fc2'):
      for sub in ('kernel', 'bias'):
        o = getattr(getattr(model.visual.merger, attr), sub)
        l = getattr(getattr(reloaded.visual.merger, attr), sub)
        np.testing.assert_array_almost_equal(
            np.array(o.value),
            np.array(l.value),
            decimal=6,
            err_msg=f'merger.{attr}.{sub} mismatch',
        )
    np.testing.assert_array_almost_equal(
        np.array(model.visual.merger.norm.scale.value),
        np.array(reloaded.visual.merger.norm.scale.value),
        decimal=6,
        err_msg='merger.norm.scale mismatch',
    )
    np.testing.assert_array_almost_equal(
        np.array(model.visual.merger.norm.bias.value),
        np.array(reloaded.visual.merger.norm.bias.value),
        decimal=6,
        err_msg='merger.norm.bias mismatch',
    )

  def test_deepstack_mergers_round_trip(self):
    """Vision deepstack merger weights survive a round trip."""
    model = self._make_model()
    reloaded = self._save_and_reload(model)

    for i in range(len(model.visual.deepstack_mergers)):
      orig = model.visual.deepstack_mergers[i]
      loaded = reloaded.visual.deepstack_mergers[i]
      for attr in ('linear_fc1', 'linear_fc2'):
        for sub in ('kernel', 'bias'):
          o = getattr(getattr(orig, attr), sub)
          l = getattr(getattr(loaded, attr), sub)
          np.testing.assert_array_almost_equal(
              np.array(o.value),
              np.array(l.value),
              decimal=6,
              err_msg=f'deepstack_mergers[{i}].{attr}.{sub} mismatch',
          )
      np.testing.assert_array_almost_equal(
          np.array(orig.norm.scale.value),
          np.array(loaded.norm.scale.value),
          decimal=6,
          err_msg=f'deepstack_mergers[{i}].norm.scale mismatch',
      )


if __name__ == '__main__':
  if test_common.is_running_in_colab():
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    unittest.TextTestRunner(verbosity=2).run(suite)
  else:
    absltest.main()
