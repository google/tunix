# Copyright 2026 Google LLC
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

"""Unit test for LoRA update_params in SGLangJax sampler.

This test verifies that update_params correctly transfers LoRA weights
from the trainer model to the SGLangJax rollout engine.
"""

import os
import tempfile
from absl.testing import absltest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import re
import transformers

from tunix.generate import mappings
from tunix.generate import sglang_jax_sampler
from tunix.models.llama3 import model as llama_lib
from tunix.tests.test_common import download_from_huggingface
from tunix.models.dummy_model_creator import create_dummy_model


class SglangJaxLoRATest(absltest.TestCase):
  """Test LoRA parameter updates in SGLangJax sampler."""

  @classmethod
  def setUpClass(cls):
    """Set up test fixtures."""
    super().setUpClass()

    # Configuration
    cls.rank = 64
    cls.alpha = 64.0
    cls.num_devices = min(jax.device_count(), 4)  # Use up to 4 devices

    # Create mesh
    cls.mesh = jax.make_mesh(
        (1, cls.num_devices),
        ("fsdp", "tp"),
        devices=jax.devices()[: cls.num_devices],
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

    model_version = "meta-llama/Llama-3.2-1B-Instruct"

    # Model configuration (smaller for faster testing)
    cls.model_config = llama_lib.ModelConfig.llama3p2_1b()

    # Create temporary directory for model files
    cls.model_path = model_version

    # Download tokenizer
    cls.tokenizer = transformers.AutoTokenizer.from_pretrained(model_version)

  def test_lora_update_params(self):
    """Test that update_params correctly transfers LoRA weights."""

    # Create base model with dummy weights
    with self.mesh:
      base_model = create_dummy_model(
          model_class=llama_lib.Llama3,
          config=self.model_config,
          mesh=self.mesh,
          dtype=jnp.bfloat16,
          random_seed=3,
      )

      # Apply LoRA to gate_proj (simplified - just one module for testing)
      lora_provider = qwix.LoraProvider(
          module_path=".*gate_proj",
          rank=self.rank,
          alpha=self.alpha,
      )

      model_input = base_model.get_model_input()
      lora_model = qwix.apply_lora_to_model(
          base_model, lora_provider, **model_input
      )

      # Get mapping config
      mapping_config = mappings.MappingConfig.build(
          model=base_model, backend="sglang_jax"
      )

      # Create sampler
      sampler_config = sglang_jax_sampler.SglangJaxConfig(
          mesh=self.mesh,
          mapping_config=mapping_config,
          model_version=self.model_path,
          context_length=512,
          mem_fraction_static=0.2,
          init_with_random_weights=True,
          disable_radix_cache=True,
          enable_static_lora=True,
          lora_target_modules=["gate_proj"],
          max_lora_rank=self.rank,
          lora_scaling=self.alpha / self.rank,
          precompile_bs_paddings=[2],
          precompile_token_paddings=[512],
          load_format="dummy",
      )

      sampler = sglang_jax_sampler.SglangJaxSampler(
          tokenizer=self.tokenizer,
          config=sampler_config,
      )

      # Get the LoRA state from trainer model and modify it
      _, trainer_state = nnx.split(lora_model)

      # Modify a specific LoRA parameter to test transfer
      # Find the gate_proj LoRA parameters
      flatten_trainer_state = trainer_state.flat_state()

      test_param_src = None
      test_param_value = None
      src_path = None

      for keys, param in flatten_trainer_state:
        path = ".".join(str(key) for key in keys)
        # Find first gate_proj lora_a parameter
        if "gate_proj" in path and "kernel_lora_a" in path:
          test_param_src = param
          src_path = path
          # Create a unique random test value
          print(f"============original value: {param.value=}")
          test_param_value = (
              jnp.ones_like(param.value, dtype=param.value.dtype) * 42
          )
          print(f"============test_param_value: {test_param_value=}")
          param.value = test_param_value

          try:
            np.testing.assert_equal(
                jax.device_get(param.value),
                jax.device_get(test_param_value),
                err_msg="Parameter update did not reflect in trainer_state",
            )
            raise ValueError(
                f"new_param: {test_param_value} is expected to not be equal to"
                f" original_param: {param.value}"
            )
          except Exception as e:
            print(
                f"new_param and original param are not equal and this is"
                f" required!"
            )
          break

      self.assertIsNotNone(test_param_src, "Failed to find test parameter")

      # Call update_params to transfer weights to sampler
      sampler.update_params(trainer_state)

      # Verify the transfer by checking the sampler's model state
      sampler_state = sampler.transformer_state
      flatten_sampler_state = sampler_state.flat_state()

      # Find the corresponding target parameter
      # Based on mapping: layers.*.mlp.gate_proj.kernel_lora_a -> model.layers.*.mlp.gate_proj.A_buffer
      tgt_path = self._get_target_path(src_path, mapping_config)

      test_param_tgt = None
      for keys, param in flatten_sampler_state:
        path = ".".join(str(key) for key in keys)
        if path == tgt_path:
          test_param_tgt = param
          break

      self.assertIsNotNone(
          test_param_tgt, f"Failed to find target parameter {tgt_path}"
      )

      # Verify the values match (after transpose)
      # LoRA A needs transpose: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
      expected_value = jnp.transpose(test_param_value[None, :, :], (0, 2, 1))

      actual_value = (
          test_param_tgt.value
          if hasattr(test_param_tgt, "value")
          else test_param_tgt
      )

      np.testing.assert_array_equal(
          jax.device_get(actual_value),
          jax.device_get(expected_value),
          err_msg=(
              f"LoRA parameter {src_path} -> {tgt_path} was not correctly"
              " transferred"
          ),
      )

  def _get_target_path(
      self, src_path: str, mapping_config: mappings.MappingConfig
  ) -> str:
    """Convert source path to target path using mappings."""
    # Determine if this is a LoRA parameter
    is_lora_param = "lora_a" in src_path or "lora_b" in src_path

    # Use only the appropriate mapping
    if is_lora_param and mapping_config.lora_to_hf_mappings:
      mappings_to_use = mapping_config.lora_to_hf_mappings
    elif mapping_config.to_hf_mappings:
      mappings_to_use = mapping_config.to_hf_mappings
    else:
      raise ValueError(f"No mappings available for {src_path}")

    # Try to find matching pattern
    for pattern, (target_pattern, _) in mappings_to_use.items():
      if "*" in pattern:
        # Convert glob pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"(\d+)")
        match = re.match(regex_pattern, src_path)
        if match:
          # Replace * in target pattern with matched numbers
          target_path = target_pattern
          for group in match.groups():
            target_path = target_path.replace("*", group, 1)
          return target_path
      elif pattern == src_path:
        return target_pattern

    raise ValueError(f"No mapping found for {src_path}")


if __name__ == "__main__":
  absltest.main()
