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
import types
from unittest import mock

from absl.testing import absltest
from flax import nnx
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import transformers
from tunix.generate import mappings
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import sglang_jax_sampler
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
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

    cls.repo_id = (  ## use smaller models to prevent OOM in v5e
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)
    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

  def load_llama3_model(self, model_version: str):
    model_config = {
        "meta-llama/Llama-3.2-3B-Instruct": llama_lib.ModelConfig.llama3p2_3b,
        "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3p1_8b,
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

    sgl_sampler = sglang_jax_sampler.SglangJaxSampler(
        tokenizer=model_tokenizer,
        config=sglang_jax_config,
        # Test kwargs forwarding while precompiling only the single extend
        # (512 tokens) and decode (bs=4) bucket needed by the 3 test prompts.
        disable_precompile=False,
        chunked_prefill_size=512,
        precompile_token_paddings=[512],
        max_running_requests=4,
        precompile_bs_paddings=[4],
    )
    self.assertNotEqual(sgl_sampler.mesh, self.mesh)
    state = nnx.state(tunix_model)
    sgl_sampler.load_checkpoint(state)

    base_utils.show_hbm_usage("After loading sglang jax sampler")

    sgl_output = sgl_sampler(
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
    print(f"sglang jax Generated text: {sgl_output.text}")
    tc.validate_llm_outputs(expected_output_pattern, sgl_output.text)

    _, tunix_state = nnx.split(tunix_model)
    _, sglangjax_state = nnx.split(sgl_sampler._model_runner.model)
    self.assertTrue(
        np.allclose(
            tunix_state["embedder"]["input_embedding"].value,
            sglangjax_state["model"]["embed_tokens"]["embedding"].value,
        )
    )


class SglangJaxSamplerTokenInputTest(absltest.TestCase):

  def _make_sampler(self, max_model_len: int | None = 8):
    sampler = object.__new__(sglang_jax_sampler.SglangJaxSampler)
    sampler.args = {"context_length": max_model_len}
    sampler.tokenizer = types.SimpleNamespace(
        pad_id=lambda: 0,
        eos_id=lambda: 2,
        bos_id=lambda: 1,
        encode=lambda text: [ord(ch) for ch in text],
        dedup_bos_ids=lambda ids: ids,
    )
    params = types.SimpleNamespace(
        max_new_tokens=0,
        n=1,
        temperature=0.0,
        stop_token_ids=[],
        skip_special_tokens=True,
        top_p=None,
        top_k=None,
        truncate_prompt_tokens=None,
    )
    params.convert_to_dict = lambda: dict(params.__dict__)
    sampler.engine = types.SimpleNamespace(
        get_default_sampling_params=lambda: params
    )
    sampler.tokenize = mock.Mock(
        side_effect=AssertionError("tokenize must not be called")
    )
    return sampler

  def test_prompt_token_ids_bypasses_tokenize_and_returns_exact_padded_ids(
      self,
  ):
    sampler = self._make_sampler(max_model_len=8)
    captured = {}

    def fake_generate(*, input_ids, sampling_params):
      captured["input_ids"] = input_ids
      captured["sampling_params"] = sampling_params
      return [
          {
              "text": "a",
              "output_ids": [9, 2],
              "meta_info": {"id": "0", "prompt_tokens": len(input_ids[0])},
          },
          {
              "text": "b",
              "output_ids": [8],
              "meta_info": {"id": "1", "prompt_tokens": len(input_ids[1])},
          },
      ]

    sampler._generate_with_loop_guard = fake_generate
    out = sampler(
        prompt_token_ids=[[0, 3], [4, 0, 5]],
        max_generation_steps=2,
        max_prompt_length=4,
    )

    self.assertEqual(captured["input_ids"], [[0, 3], [4, 0, 5]])
    np.testing.assert_array_equal(
        out.padded_prompt_tokens,
        np.array([[0, 0, 0, 3], [0, 4, 0, 5]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        out.prompt_lengths, np.array([2, 3], dtype=np.int32)
    )
    self.assertEqual(out.text, ["a", "b"])
    np.testing.assert_array_equal(
        out.tokens[0], np.array([9, 2], dtype=np.int32)
    )
    np.testing.assert_array_equal(out.tokens[1], np.array([8], dtype=np.int32))

  def test_input_strings_populates_prompt_lengths(self):
    sampler = self._make_sampler(max_model_len=8)
    sampler.tokenize = lambda text: [1] + [ord(ch) for ch in text]
    sampler._generate_with_loop_guard = lambda *, input_ids, sampling_params: [
        {
            "text": "x",
            "output_ids": [7],
            "meta_info": {"id": str(i), "prompt_tokens": len(ids)},
        }
        for i, ids in enumerate(input_ids)
    ]

    out = sampler(
        input_strings=["a", "bc"],
        max_generation_steps=2,
        max_prompt_length=4,
    )

    np.testing.assert_array_equal(
        out.prompt_lengths, np.array([2, 3], dtype=np.int32)
    )

  def test_rejects_invalid_inputs_before_calling_engine(self):
    sampler = self._make_sampler(max_model_len=5)
    sampler._generate_with_loop_guard = lambda **_: (_ for _ in ()).throw(
        AssertionError("engine must not be called")
    )

    with self.assertRaisesRegex(ValueError, "exactly one"):
      sampler(
          input_strings=["hi"], prompt_token_ids=[[1]], max_generation_steps=1
      )
    with self.assertRaisesRegex(ValueError, "exactly one"):
      sampler(max_generation_steps=1)
    with self.assertRaisesRegex(ValueError, "one output per row"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1, multi_sampling=2)
    with self.assertRaisesRegex(ValueError, "one output per row"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1, beam_size=2)
    with self.assertRaisesRegex(ValueError, "one output per row"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1, n=2)
    with self.assertRaisesRegex(ValueError, "truncate_prompt_tokens"):
      sampler(
          prompt_token_ids=[[1]],
          max_generation_steps=1,
          truncate_prompt_tokens=1,
      )
    with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
      sampler(prompt_token_ids=[[1, 2, 3, 4]], max_generation_steps=2)
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      sampler(input_strings=[], max_generation_steps=1)
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      sampler(prompt_token_ids=[], max_generation_steps=1)
    with self.assertRaisesRegex(ValueError, "1-D"):
      sampler(
          prompt_token_ids=[np.zeros((1, 2), dtype=np.int32)],
          max_generation_steps=1,
      )

  def test_rejects_mismatched_or_duplicate_engine_outputs(self):
    sampler = self._make_sampler(max_model_len=8)

    sampler._generate_with_loop_guard = lambda **_: [{
        "text": "a",
        "output_ids": [9],
        "prompt_token_ids": [1, 999],
        "meta_info": {"id": "0", "prompt_tokens": 2},
    }]
    with self.assertRaisesRegex(
        ValueError, "prompt_token_ids differed from input"
    ):
      sampler(prompt_token_ids=[[1, 2]], max_generation_steps=1)

    sampler._generate_with_loop_guard = lambda **_: [{
        "text": "a",
        "output_ids": [9],
        "meta_info": {"id": "0", "prompt_tokens": 3},
    }]
    with self.assertRaisesRegex(
        ValueError, "prompt_token_ids differed from input"
    ):
      sampler(prompt_token_ids=[[1, 2]], max_generation_steps=1)

    sampler._generate_with_loop_guard = lambda **_: [
        {
            "text": "a",
            "output_ids": [9],
            "meta_info": {"id": "dup", "prompt_tokens": 1},
        },
        {
            "text": "b",
            "output_ids": [8],
            "meta_info": {"id": "dup", "prompt_tokens": 1},
        },
    ]
    with self.assertRaisesRegex(ValueError, "Duplicate request_id"):
      sampler(prompt_token_ids=[[1], [2]], max_generation_steps=1)

    sampler._generate_with_loop_guard = lambda **_: [
        {"text": "a", "output_ids": [9], "prompt_token_ids": [1]},
    ]
    with self.assertRaisesRegex(ValueError, "missing request_id"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1)

    sampler._generate_with_loop_guard = lambda **_: [
        {"text": "a", "output_ids": [9], "meta_info": {"id": "0"}},
    ]
    with self.assertRaisesRegex(ValueError, "missing prompt echo"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1)

    sampler._generate_with_loop_guard = lambda **_: []
    with self.assertRaisesRegex(ValueError, "Expected 1 outputs, got 0"):
      sampler(prompt_token_ids=[[1]], max_generation_steps=1)


if __name__ == "__main__":
  absltest.main()
