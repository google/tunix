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

import concurrent.futures
import os
import re
import socket
import tempfile
import threading
import time
from unittest import mock
from absl.testing import absltest
from flax import nnx
import huggingface_hub
import jax
import numpy as np
import qwix
import transformers
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import vllm_sampler
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams


# vLLM Jax backend suggest to use old model desing for now.
# os.environ["NEW_MODEL_DESIGN"]="True"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"


class VllmSamplerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
    axis_names = ("fsdp", "tp")  #
    self.mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())

    self.repo_id = "meta-llama/Llama-3.2-1B-Instruct"
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
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
        ),
        rank=64,
        alpha=64.0,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
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
      llama3 = self.get_lora_model(llama3)
      print(f"Loaded LoRA model: {model_version} with LoRA enabled")
    # nnx.display(llama3)
    return llama3, model_config

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

  def _pick_free_port(self) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind(("127.0.0.1", 0))
      return sock.getsockname()[1]

  def test_vllm_sampler(self):
    tunix_model, model_config = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
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
        "Hello, my name is Tom.",
        "The capital of France is",
        "why is sky blue?",
    ]

    inputs = self.templatize(prompts, tokenizer=model_tokenizer)

    vn_sampler = vanilla_sampler.Sampler(
        transformer=tunix_model,
        tokenizer=model_tokenizer,
        cache_config=vanilla_sampler.CacheConfig(
            cache_size=512, num_layers=model_config.num_layers, num_kv_heads=model_config.num_kv_heads, head_dim=model_config.head_dim
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

    vllm_config = vllm_sampler.VllmConfig(
        model_version=self.model_path,
        max_model_len=512,
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=vllm_sampler.MappingConfig(
            to_hf_mappings=tunix_model.to_hf_mappings(),
            to_hf_transpose_keys=tunix_model.to_hf_transpose_keys(),
            lora_to_hf_mappings=tunix_model.lora_to_hf_mappings(),
            lora_config=args["additional_config"]["lora_config"],
            to_hf_hook_fns=None,
        ),
        async_mode=True,
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=model_tokenizer,
        config=vllm_config,
    )
    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    self.print_mem_stats("After loading vLLM sampler")

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

    def _validate_outputs(serving_outputs):
      for (prompt, expectations), generated in zip(expected_output_pattern, serving_outputs):
        normalized = generated.strip().lower()
        for keyword in expectations:
          self.assertIn(
              keyword.lower(),
              normalized,
              msg=(
                  f"Response '{generated}' for prompt '{prompt}' does not contain"
                  f" expected keyword '{keyword}'."
              ),
          )

    _validate_outputs(vanilla_output.text)

    print("-" * 50)
    print(f"vLLM Generated text: {vllm_output.text}")

    _validate_outputs(vllm_output.text)

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
    if vllm_config.async_mode:
      vl_sampler.stop()

  def test_async_mode_e2e_real_model_out_of_order(self):
    tunix_model, _ = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    prompts = [
        "Please summarize the following text concisely:\n"
        + " ".join(["alpha"] * length)
        for length in [320, 80, 400, 40, 360, 20, 300, 120, 340, 60]
    ]

    mapping_config = vllm_sampler.MappingConfig(
        to_hf_mappings=tunix_model.to_hf_mappings(),
        to_hf_transpose_keys=tunix_model.to_hf_transpose_keys(),
        lora_to_hf_mappings=tunix_model.lora_to_hf_mappings(),
        lora_config=None,
        to_hf_hook_fns=None,
    )
    vllm_config = vllm_sampler.VllmConfig(
        model_version=self.model_path,
        max_model_len=512,
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type=None,
        mapping_config=mapping_config,
        async_mode=True,
        port=self._pick_free_port(),
        served_model_name=None,
        api_key=None,
        extra_args=None,
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_config,
    )
    self.addCleanup(vl_sampler.stop)

    state = nnx.state(tunix_model)
    vl_sampler.load_checkpoint(state)

    driver = vl_sampler._driver
    self.assertIsNotNone(driver)

    base_params = SamplingParams()
    base_params.detokenize = True
    base_params.max_tokens = 64
    base_params.n = 1
    base_params.temperature = 0.0
    base_params.top_k = 1
    base_params.stop_token_ids = [tokenizer.eos_token_id]
    base_params.skip_special_tokens = True

    submitted_order = [str(i) for i in range(len(prompts))]
    futures = []
    for idx, prompt in enumerate(prompts):
      token_ids = vl_sampler.tokenize(prompt)
      prompt_obj = TokensPrompt(prompt_token_ids=token_ids)
      params = base_params.clone()
      future = driver.submit_request(
          request_id=str(idx),
          prompt=prompt_obj,
          params=params,
      )
      futures.append(future)

    completion_order = []
    results_by_request: dict[str, object] = {}
    for future in concurrent.futures.as_completed(futures, timeout=600):
      output = future.result()
      completion_order.append(output.request_id)
      results_by_request[output.request_id] = output

    self.assertCountEqual(completion_order, submitted_order)
    self.assertNotEqual(
        completion_order,
        submitted_order,
        msg="Completions arrived strictly in submission order; "
        "continuous batching did not reorder responses.",
    )

    for request_id in submitted_order:
      output = results_by_request[request_id]
      print(f"\nYY {output=}")
      self.assertTrue(output.finished)
      self.assertGreater(len(output.outputs), 0)
      generated_text = output.outputs[0].text
      self.assertIsInstance(generated_text, str)
      self.assertNotEqual(generated_text.strip(), "")

    if vllm_config.async_mode:
      vl_sampler.stop()

if __name__ == "__main__":
  absltest.main()
