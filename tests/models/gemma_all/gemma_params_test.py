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

# BEGIN-GOOGLE-INTERNAL
# Tests for Llama3 model parameter loading from safetensors files.
# WARNING: This test is intended for external environments, such as GCE.
# It should not be run as part of a standard internal codebase or Blaze build.

# Setup:
# 1. Run `huggingface-cli login` to authenticate with Hugging Face
# 2. Ensure you have the corresponding model access.

# Usage:
# Script: python params_test.py
# Jupyter: %run params_test.py

# Each test validates model loading, device placement, and display
# functionality.
# Tests are skipped if model paths are not configured.
# END-GOOGLE-INTERNAL

import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from flax.traverse_util import flatten_dict
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.tests import lora_params_test_base
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import params_safetensors as gemma3_params_safetensors
from tunix.tests import test_common

class GemmaParamsTest(parameterized.TestCase):

  @parameterized.named_parameters(
    dict(
        testcase_name="gemma3",
        model_type="gemma3",
    ),
    dict(
        testcase_name="gemma2",
        model_type="gemma2",
    ),
  )
  def test_map_from_upstream_checkpoint(self, model_type):
    # Tiny shapes to demonstrate logic only
    embed         = np.arange(5*3, dtype=np.float32).reshape(5, 3)        # (vocab=5, dim=3)
    final_scale   = np.arange(3, dtype=np.float32)                        # (3,)
    gate_up       = np.arange(2*6*3, dtype=np.float32).reshape(2, 6, 3)   # -> two (3,6) after .T
    down          = np.arange(6*3, dtype=np.float32).reshape(6, 3)        # stays (6,3)
    q_w           = np.arange(4*3*2, dtype=np.float32).reshape(4, 3, 2)   # (4,3,2)
    kv_w          = np.arange(2*1*3*2, dtype=np.float32).reshape(2, 1, 3, 2)  # (2,1,3,2)
    o_w           = np.arange(4*2*3, dtype=np.float32).reshape(4, 2, 3)   # (4,2,3)
    pre_attn      = np.arange(3, dtype=np.float32)
    post_attn     = np.arange(3, dtype=np.float32)
    pre_ffw       = np.arange(3, dtype=np.float32)
    post_ffw      = np.arange(3, dtype=np.float32)
    siglip_dummy  = np.array([1.0], dtype=np.float32)
    mm_dummy      = np.array([2.0], dtype=np.float32)

    upstream = {
      "transformer/embedder": {"input_embedding": embed},
      "transformer/final_norm": {"scale": final_scale},

      # Should be skipped:
      "transformer/siglip_encoder": {"whatever": siglip_dummy},
      "transformer/embedder/mm_patch": {"kernel": mm_dummy},

      # Layer 0 (tiny shapes)
      "transformer/layer_0/attn/_key_norm":     {"scale": np.arange(2, dtype=np.float32)},
      "transformer/layer_0/attn/_query_norm":   {"scale": np.arange(2, dtype=np.float32)},
      "transformer/layer_0/attn/attn_vec_einsum": {"w": o_w},
      "transformer/layer_0/attn/kv_einsum":       {"w": kv_w},
      "transformer/layer_0/attn/q_einsum":        {"w": q_w},
      "transformer/layer_0/mlp/gating_einsum":    {"w": gate_up},
      "transformer/layer_0/mlp/linear":           {"w": down},
      "transformer/layer_0/post_attention_norm":  {"scale": post_attn},
      "transformer/layer_0/post_ffw_norm":        {"scale": post_ffw},
      "transformer/layer_0/pre_attention_norm":   {"scale": pre_attn},
      "transformer/layer_0/pre_ffw_norm":         {"scale": pre_ffw},
    }

    mapped = gemma3_params.map_from_upstream_checkpoint(upstream, model_type)
    flat_m = flatten_dict(mapped)  # tuple keys

    # --- Keys & shapes we expect after mapping (tiny) ---
    expected = {
      ('embedder', 'input_embedding'):              (5, 3),
      ('final_norm', 'scale'):                      (3,),

      ('layers', 0, 'attn', '_key_norm', 'scale'):  (2,),
      ('layers', 0, 'attn', '_query_norm', 'scale'):(2,),
      ('layers', 0, 'attn', 'attn_vec_einsum', 'w'):(4, 2, 3),
      ('layers', 0, 'attn', 'kv_einsum', 'w'):      (2, 1, 3, 2),
      ('layers', 0, 'attn', 'q_einsum', 'w'):       (4, 3, 2),

      ('layers', 0, 'mlp', 'down_proj', 'kernel'):  (6, 3),
      ('layers', 0, 'mlp', 'gate_proj', 'kernel'):  (3, 6),  # from gate_up[0].T
      ('layers', 0, 'mlp', 'up_proj', 'kernel'):    (3, 6),  # from gate_up[1].T

      ('layers', 0, 'post_attn_norm' if model_type=="gemma2" else 'post_attention_norm', 'scale'):     (3,),
      ('layers', 0, 'post_ffw_norm', 'scale'):      (3,),
      ('layers', 0, 'pre_attention_norm', 'scale'): (3,),
      ('layers', 0, 'pre_ffw_norm', 'scale'):       (3,),
    }

    # 1) keys and shapes
    for k, shp in expected.items():
      assert k in flat_m, f"Missing key {k}"
      assert flat_m[k].shape == shp, f"Shape mismatch for {k}: got {flat_m[k].shape}, want {shp}"

    # 2) value checks for transforms & pass-through
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'gate_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/gating_einsum"]["w"][0].T,
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'up_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/gating_einsum"]["w"][1].T,
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'down_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/linear"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'attn_vec_einsum', 'w')],
      upstream["transformer/layer_0/attn/attn_vec_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'kv_einsum', 'w')],
      upstream["transformer/layer_0/attn/kv_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'q_einsum', 'w')],
      upstream["transformer/layer_0/attn/q_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'post_attn_norm', 'scale') if model_type=="gemma2" else
      ('layers', 0, 'post_attention_norm', 'scale')],
      upstream["transformer/layer_0/post_attention_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'post_ffw_norm', 'scale')],
      upstream["transformer/layer_0/post_ffw_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'pre_attention_norm', 'scale')],
      upstream["transformer/layer_0/pre_attention_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'pre_ffw_norm', 'scale')],
      upstream["transformer/layer_0/pre_ffw_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('embedder', 'input_embedding')],
      upstream["transformer/embedder"]["input_embedding"],
    )
    np.testing.assert_array_equal(
      flat_m[('final_norm', 'scale')],
      upstream["transformer/final_norm"]["scale"],
    )

    # 3) ensure skipped subtrees absent
    assert not any(k[0] == 'siglip_encoder' for k in flat_m.keys())
    assert ('embedder', 'mm_patch') not in mapped.get('embedder', {})


class Gemma3LoraParamsTest(lora_params_test_base.LoraParamsTestBase):
  """Tests for Gemma3 LoRA merged model saving and loading."""

  def create_config(self):
    """Create Gemma3 model config for testing."""
    return gemma3_model.ModelConfig(
        num_layers=2,
        num_embed=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        head_dim=16,
        num_kv_heads=1,
        sliding_window_size=128,  # Required for LOCAL_SLIDING attention
    )

  def get_model_class(self):
    """Get Gemma3 model class."""
    return gemma3_model.Gemma3

  def get_lora_module_path(self) -> str:
    """Get LoRA target modules for Gemma3."""
    return '.*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*up_proj|.*down_proj'

  def get_projection_keys(self, layer_idx: int) -> list[str]:
    """Get projection keys for Gemma3."""
    prefix = f'model.layers.{layer_idx}'
    return [
        f'{prefix}.self_attn.q_proj.weight',
        f'{prefix}.self_attn.k_proj.weight',
        f'{prefix}.self_attn.v_proj.weight',
        f'{prefix}.self_attn.o_proj.weight',
        f'{prefix}.mlp.gate_proj.weight',
        f'{prefix}.mlp.up_proj.weight',
        f'{prefix}.mlp.down_proj.weight',
    ]

  def save_merged_model(self, lora_model):
    """Save Gemma3 LoRA merged model."""
    gemma3_params.save_lora_merged_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        lora_model=lora_model,
        rank=self.rank,
        alpha=self.alpha,
    )

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    """Load Gemma3 model from checkpoint."""
    return gemma3_params_safetensors.create_model_from_safe_tensors(
        file_dir=checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def _create_test_inputs(self):
    """Create test inputs for Gemma3 forward pass."""
    batch_size = 2
    seq_len = 10

    input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    # Gemma3 requires attention mask - create causal mask
    attention_mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len)))

    return input_tokens, positions, attention_mask

  def _run_forward_pass(self, model, input_tokens, positions, attention_mask):
    """Run forward pass through Gemma3 model."""
    # Gemma3 uses `last_tokens` parameter name
    return model(
        last_tokens=input_tokens,
        positions=positions,
        cache=None,
        attention_mask=attention_mask,
    )

  def create_checkpoint(self, model) -> str:
    """Extract model weights and save in safetensors format.

    Uses the model's actual weights and applies inverse transformations
    to create a valid safetensors file compatible with Gemma3 loader.

    Key difference from Qwen3: kv_einsum must be decomposed into k_proj and
    v_proj.

    Args:
      model: Base model to extract weights from.

    Returns:
      Path to the created checkpoint directory.
    """
    os.makedirs(self.base_checkpoint_dir, exist_ok=True)

    base_state = {}

    # Embedder (no transformation needed)
    base_state['model.embed_tokens.weight'] = np.array(
        model.embedder.input_embedding.value
    )

    # Final norm (no transformation needed)
    base_state['model.norm.weight'] = np.array(model.final_norm.scale.value)

    # Extract and transform weights for all layers
    for layer_idx, layer in enumerate(model.layers):
      prefix = f'model.layers.{layer_idx}'

      # Layer norms (no transformation needed)
      base_state[f'{prefix}.input_layernorm.weight'] = np.array(
          layer.pre_attention_norm.scale.value
      )
      base_state[f'{prefix}.post_attention_layernorm.weight'] = np.array(
          layer.post_attention_norm.scale.value
      )
      base_state[f'{prefix}.pre_feedforward_layernorm.weight'] = np.array(
          layer.pre_ffw_norm.scale.value
      )
      base_state[f'{prefix}.post_feedforward_layernorm.weight'] = np.array(
          layer.post_ffw_norm.scale.value
      )

      # Query/Key norms (no transformation needed)
      base_state[f'{prefix}.self_attn.q_norm.weight'] = np.array(
          layer.attn._query_norm.scale.value
      )
      base_state[f'{prefix}.self_attn.k_norm.weight'] = np.array(
          layer.attn._key_norm.scale.value
      )

      # Attention projections
      # q_einsum: nnx (num_heads, embed_dim, head_dim) → safetensors (num_heads*head_dim, embed_dim)
      if hasattr(layer.attn, 'q_einsum'):
        w = np.array(
            layer.attn.q_einsum.w.value
        )  # (num_heads, embed_dim, head_dim)
        w = w.transpose(0, 2, 1)  # (num_heads, head_dim, embed_dim)
        w = w.reshape(
            -1, self.config.embed_dim
        )  # (num_heads*head_dim, embed_dim)
        base_state[f'{prefix}.self_attn.q_proj.weight'] = w

      # kv_einsum: nnx (2, num_kv_heads, embed_dim, head_dim) →
      # safetensors k_proj (num_kv_heads*head_dim, embed_dim) and v_proj (num_kv_heads*head_dim, embed_dim)
      if hasattr(layer.attn, 'kv_einsum'):
        kv_w = np.array(
            layer.attn.kv_einsum.w.value
        )  # (2, num_kv_heads, embed_dim, head_dim)

        # Split into k and v
        k_w = kv_w[0]  # (num_kv_heads, embed_dim, head_dim)
        v_w = kv_w[1]  # (num_kv_heads, embed_dim, head_dim)

        # Transform k
        k_w = k_w.transpose(0, 2, 1)  # (num_kv_heads, head_dim, embed_dim)
        k_w = k_w.reshape(
            -1, self.config.embed_dim
        )  # (num_kv_heads*head_dim, embed_dim)
        base_state[f'{prefix}.self_attn.k_proj.weight'] = k_w

        # Transform v
        v_w = v_w.transpose(0, 2, 1)  # (num_kv_heads, head_dim, embed_dim)
        v_w = v_w.reshape(
            -1, self.config.embed_dim
        )  # (num_kv_heads*head_dim, embed_dim)
        base_state[f'{prefix}.self_attn.v_proj.weight'] = v_w

      # o_proj (attn_vec_einsum): nnx (num_heads, head_dim, embed_dim) → safetensors (embed_dim, num_heads*head_dim)
      if hasattr(layer.attn, 'attn_vec_einsum'):
        w = np.array(
            layer.attn.attn_vec_einsum.w.value
        )  # (num_heads, head_dim, embed_dim)
        w = w.reshape(
            -1, self.config.embed_dim
        )  # (num_heads*head_dim, embed_dim)
        base_state[f'{prefix}.self_attn.o_proj.weight'] = (
            w.T
        )  # (embed_dim, num_heads*head_dim)

      # MLP projections
      # nnx: (in_features, out_features) → safetensors: (out_features, in_features)
      if hasattr(layer.mlp, 'gate_proj'):
        base_state[f'{prefix}.mlp.gate_proj.weight'] = np.array(
            layer.mlp.gate_proj.kernel.value
        ).T

      if hasattr(layer.mlp, 'up_proj'):
        base_state[f'{prefix}.mlp.up_proj.weight'] = np.array(
            layer.mlp.up_proj.kernel.value
        ).T

      if hasattr(layer.mlp, 'down_proj'):
        base_state[f'{prefix}.mlp.down_proj.weight'] = np.array(
            layer.mlp.down_proj.kernel.value
        ).T

    # Save to disk
    safe_np.save_file(
        base_state, os.path.join(self.base_checkpoint_dir, 'model.safetensors')
    )

    # Minimal config for file copying test
    with open(os.path.join(self.base_checkpoint_dir, 'config.json'), 'w') as f:
      f.write('{"model_type": "gemma3"}')

    return self.base_checkpoint_dir

  def test_kv_einsum_decomposition(self):
    """Test that kv_einsum is properly decomposed into k_proj and v_proj."""
    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA
    lora_model = self._apply_lora_to_model(base_model)

    # Save merged model
    self.save_merged_model(lora_model)

    # Load the merged state
    merged_state = safe_np.load_file(
        os.path.join(self.merged_output_dir, 'model.safetensors')
    )

    # Verify k_proj and v_proj exist (decomposed from kv_einsum)
    for layer_idx in range(self.config.num_layers):
      k_proj_key = f'model.layers.{layer_idx}.self_attn.k_proj.weight'
      v_proj_key = f'model.layers.{layer_idx}.self_attn.v_proj.weight'

      self.assertIn(
          k_proj_key, merged_state, f'Missing k_proj for layer {layer_idx}'
      )
      self.assertIn(
          v_proj_key, merged_state, f'Missing v_proj for layer {layer_idx}'
      )

      # Verify shapes
      expected_shape = (
          self.config.num_kv_heads * self.config.head_dim,
          self.config.embed_dim,
      )
      self.assertEqual(
          merged_state[k_proj_key].shape,
          expected_shape,
          f'Wrong shape for k_proj in layer {layer_idx}',
      )
      self.assertEqual(
          merged_state[v_proj_key].shape,
          expected_shape,
          f'Wrong shape for v_proj in layer {layer_idx}',
      )


if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  if test_common.is_running_in_colab():
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(Gemma3LoraParamsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  else:
    # Running as a script - use absltest.main()
    absltest.main()
