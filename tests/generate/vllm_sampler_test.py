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
import tempfile
from absl.testing import absltest
from flax import nnx
import huggingface_hub
import jax
import numpy as np
from qwix import lora
import transformers
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import vllm_sampler
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params


os.environ["TPU_BACKEND_TYPE"] = "jax"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# vLLM Jax backend suggest to use old model desing for now.
# os.environ["NEW_MODEL_DESIGN"]="True"

# TODO(b/433750353): Enable random weights after vLLM supports it properly
# os.environ["JAX_RANDOM_WEIGHTS"] = "True"


class VllmSamplerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
    axis_names = ("fsdp", "tp")  #
    self.mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())

    self.repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    temp_dir = tempfile.gettempdir()
    self.model_path = os.path.join(temp_dir, "models", self.repo_id)
    all_files = huggingface_hub.list_repo_files(self.repo_id)
    filtered_files = [f for f in all_files if not f.startswith("original/")]

    for filename in filtered_files:
      huggingface_hub.hf_hub_download(
          repo_id=self.repo_id, filename=filename, local_dir=self.model_path
      )
    print(f"Downloaded {filtered_files} to: {self.model_path}")

    # TODO(b/432096319): Enable after LoRA support in vLLM
    self.enable_lora = False

  def get_lora_model(self, base_model):
    lora_provider = lora.LoraProvider(
        module_path=(
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
        ),
        rank=64,
        alpha=64.0,
    )

    model_input = base_model.get_model_input()
    lora_model = lora.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

    return lora_model

  def load_llama3_model(
      self, model_version: str = "llama3-1b", enable_lora: bool = False
  ):
    model_config = {
        "llama3-1b": llama_lib.ModelConfig.llama3_1b,
        "llama3-8b": llama_lib.ModelConfig.llama3_8b,
    }
    assert (
        model_version in model_config
    ), f"Invalid model version: {model_version}"
    model_config = model_config[model_version]()

    llama3 = llama_params.create_model_from_safe_tensors(
        self.model_path, model_config, self.mesh
    )
    if enable_lora:
      llama3 = self.get_lora_model(llama3)
      print(f"Loaded LoRA model: {model_version} with LoRA enabled")
    # nnx.display(llama3)
    return llama3

  def print_mem_stats(self, label: str):
    print(f"\nMemstats: {label}:")
    try:
      for d in jax.local_devices():
        stats = d.memory_stats()
        used = round(stats["bytes_in_use"] / 2**30, 2)
        limit = round(stats["bytes_limit"] / 2**30, 2)
        print(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
    except (RuntimeError, KeyError, TypeError) as ex:
      print(f"\tMemstats unavailable, error: {ex}")

  def templatize(self, prompts, tokenizer=None):
    out = []
    for p in prompts:
      out.append(
          tokenizer.apply_chat_template(
              [
                  {"role": "user", "content": p},
              ],
              tokenize=False,
              add_generation_prompt=True,
          )
      )
    return out

  def test_vllm_sampler(self):
    if "8B" in self.repo_id:
      tunix_model_type = "llama3-8b"
    elif "1B" in self.repo_id:
      tunix_model_type = "llama3-1b"
    else:
      raise ValueError(f"Unsupported model type: {self.repo_id}")

    tunix_model = self.load_llama3_model(
        tunix_model_type, enable_lora=self.enable_lora
    )

    args = {}
    args["model"] = self.model_path
    args["additional_config"] = {}
    args["additional_config"]["lora_config"] = None
    if self.enable_lora:
      args["additional_config"]["lora_config"] = {
          "rank": 64,
          "alpha": 64.0,
          "module_path": (
              ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
          ),
          # "dropout": 0.0,
          # "bias": "none",
      }

    self.print_mem_stats("After loading tunix model")

    # Sampler setup
    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    inputs = self.templatize(prompts, tokenizer=model_tokenizer)

    vn_sampler = vanilla_sampler.Sampler(
        transformer=tunix_model,
        tokenizer=model_tokenizer,
        cache_config=vanilla_sampler.CacheConfig(
            cache_size=512, num_layers=32, num_kv_heads=8, head_dim=128
        ),
    )
    vanilla_output = vn_sampler(
        input_strings=inputs,
        total_generation_steps=128,  # Changed from 768 to 128 for vLLM
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    vl_sampler = vllm_sampler.vLLMSampler(
        tokenizer=model_tokenizer,
        mesh=self.mesh,
        max_model_len=512,  # Set to 1024 for vLLM
        model_version=self.model_path,
        mapping_config=vllm_sampler.MappingConfig(
            to_hf_mappings=tunix_model.to_hf_mappings(),
            to_hf_transpose_keys=tunix_model.to_hf_transpose_keys(),
            lora_to_hf_mappings=tunix_model.lora_to_hf_mappings(),
            lora_config=args["additional_config"]["lora_config"],
        ),
    )
    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    self.print_mem_stats("After loading vLLM sampler")

    vllm_output = vl_sampler(
        input_strings=inputs,
        total_generation_steps=128,  # Changed from 768 to 128 for vLLM
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )
    # Print the outputs.
    print("-" * 50)
    print(f"Vanilla Generated text: {vanilla_output.text}")
    self.assertEqual(
        vanilla_output.text, ["Nice to meet you. What's your name?", "Paris."]
    )

    print("-" * 50)
    print(f"vLLM Generated text: {vllm_output.text}")
    self.assertEqual(
        vllm_output.text,
        [
            (
                "It's nice to meet you. Is there something I can help you with"
                " or would you like to chat?"
            ),
            "The capital of France is Paris.",
        ],
    )

    _, tunix_state = nnx.split(tunix_model)
    vllm_state = vl_sampler._model_runner.state
    if os.environ.get("NEW_MODEL_DESIGN") == "True":
      self.assertTrue(
          np.allclose(
              tunix_state["lm_head"]["w"].value,
              vllm_state["lm_head"]["input_embedding_table_DV"].value,
          )
      )
    else:
      self.assertTrue(
          np.allclose(
              tunix_state["lm_head"]["w"].value, vllm_state["lm_head"].value
          )
      )


if __name__ == "__main__":
  absltest.main()
