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

import os
import re
import tempfile

from absl.testing import absltest
from flax import nnx
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import transformers
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import sglang_jax_sampler, mappings
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params
from tunix.sft import utils as base_utils
from tunix.tests import test_common as tc


class SglangJaxSamplerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())

    cls.repo_id = (  ## use smaller models to prevent OOM in v5e
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)
    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

  def load_llama3_model(self, model_version: str):
    model_config = {
        "meta-llama/Llama-3.2-3B-Instruct": llama_lib.ModelConfig.llama3_2_3b,
        "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3_1_8b,
    }
    assert (
        model_version in model_config
    ), f"Invalid model version: {model_version}"
    model_config = model_config[model_version]()

    llama3 = llama_params.create_model_from_safe_tensors(
        self.model_path, model_config, self.mesh
    )
    # nnx.display(llama3)
    return llama3

  def test_sglang_jax_sampler(self):
    tunix_model = self.load_llama3_model(self.repo_id)

    args = {}
    args["model_path"] = self.model_path

    base_utils.show_hbm_usage("After loading tunix model")

    # Sampler setup
    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is Tom.",
        "The capital of France is",
        "why is sky blue?",
    ]

    inputs = tc.batch_templatize(prompts, tokenizer=model_tokenizer)

    vn_sampler = vanilla_sampler.Sampler(
        transformer=tunix_model,
        tokenizer=model_tokenizer,
        cache_config=vanilla_sampler.CacheConfig(
            cache_size=512, num_layers=32, num_kv_heads=8, head_dim=128
        ),
    )
    vanilla_output = vn_sampler(
        input_strings=inputs,
        max_generation_steps=128,  # Changed from 768 to 128 for sglang-jax
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    mapping_config = mappings.MappingConfig.build(
        model=tunix_model, backend="sglang_jax"
    )

    sglang_jax_config = sglang_jax_sampler.SglangJaxConfig(
        model_version=self.model_path,
        context_length=512,
        mesh=self.mesh,
        mem_fraction_static=0.2,
        init_with_random_weights=True,
        disable_radix_cache=True,
        enable_deterministic_sampling=False,
        mapping_config=mapping_config,
    )

    vl_sampler = sglang_jax_sampler.SglangJaxSampler(
        tokenizer=model_tokenizer,
        config=sglang_jax_config,
    )
    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    base_utils.show_hbm_usage("After loading sglang jax sampler")

    sglang_jax_output = vl_sampler(
        input_strings=inputs,
        max_generation_steps=128,
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    expected_output_pattern = [
        (prompts[0], ["Tom", "help"]),
        (prompts[1], ["Paris"]),
        (prompts[2], ["Rayleigh", "scattering"]),
    ]

    print("-" * 50)
    print(f"Vanilla Generated text: {vanilla_output.text}")

    tc.validate_llm_outputs(expected_output_pattern, vanilla_output.text)

    print("-" * 50)
    print(f"sglang jax Generated text: {sglang_jax_output.text}")
    tc.validate_llm_outputs(expected_output_pattern, sglang_jax_output.text)

    _, tunix_state = nnx.split(tunix_model)
    _, sglangjax_state = nnx.split(vl_sampler._model_runner.model)
    self.assertTrue(
        np.allclose(
            tunix_state["embedder"]["input_embedding"].value,
            sglangjax_state["model"]["embed_tokens"]["embedding"].value,
        )
    )


if __name__ == "__main__":
  absltest.main()
