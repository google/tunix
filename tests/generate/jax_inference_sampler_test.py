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

import os
import unittest
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import numpy as np
import transformers

from tunix.generate import jax_inference_sampler
from tunix.generate import mappings
from tunix.generate import sampler as vanilla_sampler
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.tests import test_common as tc


@unittest.skipIf(
    jax_inference_sampler.JaxInferenceEngine is None,
    "jax-inference is required for JaxInferenceSampler tests.",
)
class JaxInferenceSamplerTest(absltest.TestCase):
  model_path: str | None = None
  repo_id: str = "Qwen/Qwen3-1.7B-base"

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    if jax_inference_sampler.JaxInferenceEngine is None:
      raise unittest.SkipTest(
          "jax-inference is required for JaxInferenceSampler tests."
      )

    # Resolve model path from HuggingFace cache if available
    repo_cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{cls.repo_id.replace('/', '--')}/snapshots"
    )
    if os.path.exists(repo_cache) and os.listdir(repo_cache):
      cls.model_path = os.path.join(repo_cache, os.listdir(repo_cache)[0])
    elif os.path.isdir(cls.repo_id):
      cls.model_path = cls.repo_id
    else:
      cls.model_path = None

    if not cls.model_path or not os.path.isdir(cls.model_path):
      raise unittest.SkipTest(
          f"Model checkpoint for {cls.repo_id} not found locally."
      )

    mesh_shape = (1, len(jax.devices()))
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

  def test_jax_inference_sampler_e2e(self):
    """Tests end-to-end generation and weight update with JaxInferenceSampler."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path, trust_remote_code=True
    )

    # 1. Initialize JaxInferenceSampler
    config = jax_inference_sampler.JaxInferenceConfig(
        model_name=self.model_path,
        mesh=self.mesh,
        tensor_parallel_size=len(jax.devices()),
        num_blocks=128,
        block_size=256,
        kv_cache_dtype="bf16",
        unroll_steps=16,
    )

    sampler = jax_inference_sampler.JaxInferenceSampler(
        tokenizer=tokenizer,
        config=config,
    )

    self.assertIsNotNone(sampler.transformer)
    self.assertIsNotNone(sampler.transformer_state)
    self.assertIsNotNone(sampler.mesh)

    prompts = [
        "The capital of France is",
        "2 + 2 =",
    ]

    # 2. Run generation
    output = sampler(
        input_strings=prompts,
        max_generation_steps=16,
        temperature=0.0,
    )

    self.assertLen(output.text, 2)
    self.assertLen(output.tokens, 2)
    self.assertEqual(output.padded_prompt_tokens.shape[0], 2)
    self.assertTrue(all(isinstance(t, str) and len(t) > 0 for t in output.text))
    print("JaxInference generated texts:", output.text)

  def test_update_params_with_mappings(self):
    """Tests weight synchronization using mapping config into JaxInferenceSampler."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path, trust_remote_code=True
    )

    # Create small 1-layer Tunix Qwen3 model to test parameter transfer
    model_config = qwen3_model.ModelConfig.qwen3_1p7b_base()
    model_config.num_layers = 1

    tunix_model = qwen3_params.create_model_from_safe_tensors(
        self.model_path, model_config, self.mesh
    )

    mapping_config = mappings.MappingConfig(**mapping_vllm_jax.VLLM_JAX_MAPPING)

    config = jax_inference_sampler.JaxInferenceConfig(
        model_name=self.model_path,
        mesh=self.mesh,
        tensor_parallel_size=len(jax.devices()),
        num_blocks=64,
        block_size=256,
        kv_cache_dtype="bf16",
        mapping_config=mapping_config,
    )

    sampler = jax_inference_sampler.JaxInferenceSampler(
        tokenizer=tokenizer,
        config=config,
    )

    # Test weight syncing
    state = nnx.state(tunix_model)
    sampler.update_params(state)

  def test_jax_inference_sampler_temperature(self):
    """Tests JaxInferenceSampler with temperature > 0, top_k, top_p, PRNGKey seed, and multi_sampling."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path, trust_remote_code=True
    )

    config = jax_inference_sampler.JaxInferenceConfig(
        model_name=self.model_path,
        mesh=self.mesh,
        tensor_parallel_size=len(jax.devices()),
        num_blocks=128,
        block_size=256,
        kv_cache_dtype="bf16",
        unroll_steps=16,
    )

    sampler = jax_inference_sampler.JaxInferenceSampler(
        tokenizer=tokenizer,
        config=config,
    )

    prompts = ["What is the color of the sky?"]

    # 1. Test with temperature > 0, top_k, top_p and JAX PRNGKey
    key = jax.random.PRNGKey(42)
    output1 = sampler(
        input_strings=prompts,
        max_generation_steps=16,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        seed=key,
    )
    self.assertLen(output1.text, 1)
    self.assertTrue(len(output1.text[0]) > 0)

    # 2. Test multi_sampling > 1 with temperature > 0
    output_multi = sampler(
        input_strings=prompts,
        max_generation_steps=16,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        multi_sampling=3,
    )
    self.assertLen(output_multi.text, 3)
    self.assertEqual(output_multi.padded_prompt_tokens.shape[0], 3)
    self.assertLen(output_multi.tokens, 3)
    print("Multi-sample results with temperature=0.7:", output_multi.text)


if __name__ == "__main__":
  absltest.main()

