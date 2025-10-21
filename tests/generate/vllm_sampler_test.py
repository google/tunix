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

import asyncio
import concurrent.futures
import functools
import os
import re
import socket
import tempfile
import threading
import time
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import numpy as np
import qwix
import transformers
from tunix.generate import mappings
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import vllm_sampler
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params
from tunix.sft import utils as base_utils
from tunix.tests import test_common as tc
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
import asyncio

# vLLM Jax backend suggest to use old model desing for now.
# os.environ["NEW_MODEL_DESIGN"]="True"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"


class VllmSamplerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.repo_id = "meta-llama/Llama-3.2-1B-Instruct"
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)

    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

    # TODO(b/432096319): Enable after LoRA support in vLLM
    cls.enable_lora = False

    mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())

  def load_llama3_model(self, model_version: str, enable_lora: bool = False):
    model_config = {
        "meta-llama/Llama-3.2-1B-Instruct": llama_lib.ModelConfig.llama3_2_1b,
        "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3_1_8b,
    }
    assert (
        model_version in model_config
    ), f"Invalid model version: {model_version}"
    model_config = model_config[model_version]()

    llama3 = llama_params.create_model_from_safe_tensors(
        self.model_path, model_config, self.mesh
    )
    if enable_lora:
      llama3 = tc.get_lora_model(
          llama3,
          model_path=".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
          rank=64,
          alpha=64.0,
          mesh=self.mesh,
      )
      print(f"Loaded LoRA model: {model_version} with LoRA enabled")
    # nnx.display(llama3)
    return llama3, model_config

  # Parametized test always fails on vLLM HBM usage exceeding limit, no matter how much HBM we allocated to it, and no matter how we clear the Jax cache (delete all the live arrays, gc collect, clear cache, clear test cache). vLLM will allocate all the assigned HBM to weights + KV cache. The conclusion is parametized test doesn't reset Jax properly, therefore the 2nd test adds on top of the previous HBM usage. This is the workaround for that.
  def test_vllm_sampler_batch_mode(self):
    self._run_vllm_sampler(server_mode=False)

  def test_vllm_sampler_server_mode(self):
    self._run_vllm_sampler(server_mode=True)

  def _run_vllm_sampler(self, server_mode):
    tunix_model, model_config = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
    )

    base_utils.show_hbm_usage("After loading tunix model")

    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    args = {}
    args["model"] = self.model_path
    args["additional_config"] = {}
    args["additional_config"]["lora_config"] = None
    if self.enable_lora:
      args["additional_config"]["lora_config"] = {
          "rank": 64,
          "alpha": 64.0,
          "module_path": ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
          # "dropout": 0.0,
          # "bias": "none",
      }

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

    inputs = tc.batch_templatize(prompts, model_tokenizer)

    vn_sampler = vanilla_sampler.Sampler(
        transformer=tunix_model,
        tokenizer=model_tokenizer,
        cache_config=vanilla_sampler.CacheConfig(
            cache_size=512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    vanilla_output = vn_sampler(
        input_strings=inputs,
        max_generation_steps=128,  # Changed from 768 to 128 for vLLM
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    mapping_config = mappings.MappingConfig.build(tunix_model)

    vllm_config = vllm_sampler.VllmConfig(
        model_version=self.model_path,
        max_model_len=512,
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        lora_config=args["additional_config"]["lora_config"],
        server_mode=server_mode,
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=model_tokenizer,
        config=vllm_config,
    )
    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    base_utils.show_hbm_usage("After loading vLLM sampler")

    vllm_output = vl_sampler(
        input_strings=inputs,
        max_generation_steps=128,  # Changed from 768 to 128 for vLLM
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
    print(f"vLLM Generated text: {vllm_output.text}")

    tc.validate_llm_outputs(expected_output_pattern, vllm_output.text)

    _, tunix_state = nnx.split(tunix_model)
    vllm_state = vl_sampler._model_runner.state

    if os.environ.get("NEW_MODEL_DESIGN") == "True":
      self.assertTrue(
          np.allclose(
              tunix_state["embedder"]["input_embedding"].value,
              vllm_state["embedder"]["input_embedding_table_VD"].value,
          )
      )
    else:
      self.assertTrue(
          np.allclose(
              tunix_state["embedder"]["input_embedding"].value,
              vllm_state["model"]["embed"]["embedding"].value,
          )
      )
    if vllm_config.server_mode:
      vl_sampler.stop()

  def test_vllm_sampler_run_in_executor_concurrency(self):
    tunix_model, _ = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    mapping_config = mappings.MappingConfig.build(tunix_model)
    vllm_config = vllm_sampler.VllmConfig(
        model_version=self.model_path,
        max_model_len=512,
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=True,
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_config,
    )
    self.addCleanup(vl_sampler.stop)

    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    base_prompts = [
        "Hello, my name is Tom.",
        "The capital of France is",
        "why is sky blue?",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "Write a haiku about winter.",
        "Summarize the plot of Romeo and Juliet.",
        "Give me a recipe for pancakes.",
        "What is the boiling point of water at sea level?",
        "Share a motivational quote about perseverance.",
    ]
    prompts = list(base_prompts)
    templated_prompts = tc.batch_templatize(prompts, tokenizer)

    expected_keywords = {
        base_prompts[0]: ["Tom", "help"],
        base_prompts[1]: ["Paris"],
        base_prompts[2]: ["Rayleigh", "scattering"],
        base_prompts[3]: ["relativity", "einstein"],
        base_prompts[4]: ["health", "energy"],
        base_prompts[5]: ["winter"],
        base_prompts[6]: ["romeo", "juliet"],
        base_prompts[7]: ["pancake"],
        base_prompts[8]: ["100", "celsius"],
        base_prompts[9]: ["seven", "eight"],
    }
    prompt_expectations = [
        (prompt, expected_keywords.get(prompt, [])) for prompt in prompts
    ]

    delays = [0.05 * (len(prompts) - idx) for idx in range(len(prompts))]

    def _call_sampler(templated_prompt: str, delay: float):
      time.sleep(delay)
      return vl_sampler(
          input_strings=[templated_prompt],
          max_generation_steps=128,
          max_prompt_length=None,
          temperature=0.0,
          top_k=1,
          seed=0,
          echo=False,
          pad_output=True,
      )

    async def __call_sampler_async(index: int, templated_prompt: str, delay: float):
      loop = asyncio.get_running_loop()
      result = await loop.run_in_executor(
          None,
          _call_sampler,
          templated_prompt,
          delay,
      )
      return index, result

    async def dispatch_requests():
      loop = asyncio.get_running_loop()
      tasks = []
      for idx, templated_prompt in enumerate(templated_prompts):
        task = loop.create_task(
            __call_sampler_async(
                idx, templated_prompt, delays[idx]
            )
        )

        tasks.append(task)

      completion_order = []
      results_by_idx = {}
      for task in asyncio.as_completed(tasks):
        idx, result = await task
        completion_order.append(idx)
        results_by_idx[idx] = result

      ordered_results = [results_by_idx[i] for i in range(len(tasks))]
      return ordered_results, completion_order

    results, completion_order = asyncio.run(dispatch_requests())

    self.assertLen(results, len(prompts))

    for (prompt, expectations), sampler_output in zip(
        prompt_expectations, results
    ):
      tc.validate_llm_outputs(
          [(prompt, expectations)], sampler_output.text
      )

    expected_order = list(range(len(prompts)))
    self.assertCountEqual(completion_order, expected_order)
    self.assertNotEqual(
        completion_order,
        expected_order,
        msg=(
            "Responses returned strictly in submission order; "
            "expected out-of-order completions."
        ),
    )

if __name__ == "__main__":
  absltest.main()
