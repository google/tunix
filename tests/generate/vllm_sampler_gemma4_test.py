# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end Tunix -> vLLM (tpu-inference JAX backend) weight sync for Gemma4.

Syncs a real Gemma4 E2B checkpoint into a randomly initialised vLLM engine and
checks (a) every Tunix param is mapped onto the tpu-inference pytree and
(b) greedy decoding from the synced engine produces the expected answer.
"""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import transformers
from tunix.generate import mappings
from tunix.generate import utils as generate_utils
from tunix.generate import vllm_sampler
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors as gemma4_params
from tunix.tests import test_common as tc


class VllmSamplerGemma4Test(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.repo_id = "google/gemma-4-E2B-it"
    cls.model_path = os.environ.get("GEMMA4_MODEL_PATH")
    if not cls.model_path:
      cls.model_path = os.path.join(
          tempfile.gettempdir(), "models", cls.repo_id
      )
      tc.download_from_huggingface(
          repo_id=cls.repo_id, model_path=cls.model_path
      )

    mesh_shape = (1, len(jax.devices()))
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

  def test_gemma4_e2b_weight_sync_and_generate(self):
    config = gemma4_model.ModelConfig.gemma4_e2b()
    tunix_model = gemma4_params.create_model_from_safe_tensors(
        self.model_path, config, self.mesh, text_only=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    mapping_config = mappings.MappingConfig.build(tunix_model)
    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.3,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=False,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 256,
            # The Tunix model is text-only and the mapping targets
            # `language_model.*`; load vLLM's text-only Gemma4 too so no
            # vision/audio tower is instantiated (its head count also does not
            # divide by TP=8).
            "hf_overrides": {"architectures": ["Gemma4ForCausalLM"]},
        },
    )
    sampler = vllm_sampler.VllmSampler(tokenizer=tokenizer, config=vllm_config)

    # Record mapping diagnostics emitted by transfer_state_with_mappings.
    errors, warnings = [], []
    real_error = generate_utils.logging.error
    real_warning = generate_utils.logging.warning

    def _error(msg, *args, **kw):
      errors.append(msg % args if args else msg)
      real_error(msg, *args, **kw)

    def _warning(msg, *args, **kw):
      warnings.append(msg % args if args else msg)
      real_warning(msg, *args, **kw)

    try:
      with mock.patch.object(sampler.llm, "reset_prefix_cache"), \
           mock.patch.object(sampler.llm, "collective_rpc"), \
           mock.patch.object(generate_utils.logging, "error", _error), \
           mock.patch.object(generate_utils.logging, "warning", _warning):
        sampler.load_checkpoint(nnx.state(tunix_model, nnx.Param))

      # Every Tunix param must land somewhere in the vLLM pytree ...
      self.assertFalse(
          [e for e in errors if "No mapping for source key" in e],
          f"Unmapped Tunix params: {errors}",
      )
      # ... and every attention projection in the vLLM pytree must be fed.
      unmapped_attn = [
          w
          for w in warnings
          if "No mapping for flat states" in w and "self_attn" in w
      ]
      self.assertFalse(
          unmapped_attn, f"vLLM attention params left unsynced: {unmapped_attn}"
      )

      prompts = ["The capital of France is", "why is sky blue?"]
      inputs = tc.batch_templatize(prompts, tokenizer)
      output = sampler(
          input_strings=inputs,
          max_generation_steps=64,
          temperature=0.0,
          top_k=1,
          echo=False,
          pad_output=True,
      )
      print(f"vLLM Generated text: {output.text}")
      tc.validate_llm_outputs(
          [
              (prompts[0], ["Paris"]),
              (prompts[1], ["Rayleigh", "scattering"]),
          ],
          output.text,
      )
    finally:
      if hasattr(sampler, "stop"):
        sampler.stop()


if __name__ == "__main__":
  absltest.main()
