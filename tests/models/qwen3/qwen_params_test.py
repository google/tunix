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

"""Tests for Qwen3 model parameters and LoRA merged model saving."""

import os
import unittest

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.tests import lora_params_test_base
from tunix.tests import test_common


class Qwen3ParamsTest(lora_params_test_base.LoraParamsTestBase):
  """Tests for Qwen3 model parameters and LoRA merging."""

  def create_config(self):
    """Create Qwen3 model config for testing."""
    return qwen3_model.ModelConfig(
        num_layers=2,
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        num_experts=None,
        num_experts_per_tok=None,
    )

  def get_model_class(self):
    """Get Qwen3 model class."""
    return qwen3_model.Qwen3

  def get_lora_module_path(self) -> str:
    """Get LoRA target modules for Qwen3."""
    return (
        ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    )

  def get_projection_keys(self, layer_idx: int) -> list[str]:
    """Get projection keys for Qwen3."""
    prefix = f"model.layers.{layer_idx}"
    return [
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]

  def save_merged_model(self, lora_model):
    """Save Qwen3 LoRA merged model."""
    qwen3_params.save_lora_merged_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        lora_model=lora_model,
        rank=self.rank,
        alpha=self.alpha,
    )

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    """Load Qwen3 model from checkpoint."""
    return qwen3_params.create_model_from_safe_tensors(
        file_dir=checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def create_checkpoint(self, model) -> str:
    """Extract model weights and save in safetensors format.

    Uses the model's actual weights and applies inverse transformations
    (from _get_key_and_transform_mapping) to create a valid safetensors file.

    Args:
      model: Base model to extract weights from.

    Returns:
      Path to the created checkpoint directory.
    """
    os.makedirs(self.base_checkpoint_dir, exist_ok=True)

    base_state = {}
    base_state["model.embed_tokens.weight"] = np.array(
        model.embedder.input_embedding.value
    )
    base_state["model.norm.weight"] = np.array(model.final_norm.w.value)
    base_state["lm_head.weight"] = np.array(model.lm_head.w.value).T

    # Extract and transform weights for all layers
    # Based on inverse of _get_key_and_transform_mapping in qwen3/params.py
    for layer_idx, layer in enumerate(model.layers):
      prefix = f"model.layers.{layer_idx}"

      base_state[f"{prefix}.input_layernorm.weight"] = np.array(
          layer.input_layernorm.w.value
      )
      base_state[f"{prefix}.post_attention_layernorm.weight"] = np.array(
          layer.post_attention_layernorm.w.value
      )

      base_state[f"{prefix}.self_attn.q_norm.weight"] = np.array(
          layer.attn.q_norm.w.value
      )
      base_state[f"{prefix}.self_attn.k_norm.weight"] = np.array(
          layer.attn.k_norm.w.value
      )

      # Attention projections
      # nnx: (num_heads, embed_dim, head_dim) → safetensors: (num_heads*head_dim, embed_dim)
      # Transform: transpose(1,0,2) then reshape then transpose
      if hasattr(layer.attn, "q_proj"):
        w = np.array(
            layer.attn.q_proj.w.value
        )  # (num_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_heads*head_dim)
        base_state[f"{prefix}.self_attn.q_proj.weight"] = w.T

      if hasattr(layer.attn, "k_proj"):
        w = np.array(
            layer.attn.k_proj.w.value
        )  # (num_kv_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_kv_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_kv_heads*head_dim)
        base_state[f"{prefix}.self_attn.k_proj.weight"] = w.T

      if hasattr(layer.attn, "v_proj"):
        w = np.array(
            layer.attn.v_proj.w.value
        )  # (num_kv_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_kv_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_kv_heads*head_dim)
        base_state[f"{prefix}.self_attn.v_proj.weight"] = w.T

      if hasattr(layer.attn, "o_proj"):
        w = np.array(
            layer.attn.o_proj.w.value
        )  # (num_heads, head_dim, embed_dim)
        w = w.transpose(2, 0, 1)  # (embed_dim, num_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_heads*head_dim)
        base_state[f"{prefix}.self_attn.o_proj.weight"] = w.T

      # MLP projections
      # nnx: (in_features, out_features) → safetensors: (out_features, in_features)
      # Transform: just transpose
      if hasattr(layer.mlp, "gate_proj"):
        base_state[f"{prefix}.mlp.gate_proj.weight"] = np.array(
            layer.mlp.gate_proj.kernel.value
        ).T

      if hasattr(layer.mlp, "up_proj"):
        base_state[f"{prefix}.mlp.up_proj.weight"] = np.array(
            layer.mlp.up_proj.kernel.value
        ).T

      if hasattr(layer.mlp, "down_proj"):
        base_state[f"{prefix}.mlp.down_proj.weight"] = np.array(
            layer.mlp.down_proj.kernel.value
        ).T

    safe_np.save_file(
        base_state, os.path.join(self.base_checkpoint_dir, "model.safetensors")
    )

    # Minimal config for file copying test
    with open(os.path.join(self.base_checkpoint_dir, "config.json"), "w") as f:
      f.write('{"model_type": "qwen3"}')

    return self.base_checkpoint_dir


if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  if test_common.is_running_in_colab():
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(Qwen3ParamsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  else:
    # Running as a script - use absltest.main()
    absltest.main()
