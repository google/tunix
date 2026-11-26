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

"""Unit tests for generic HF -> vLLM-JAX manifest dispatch and metadata."""

from __future__ import annotations

import dataclasses

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from tunix.generate import vllm_hf_manifest
from tunix.models import hf_mapping


@dataclasses.dataclass
class _FakeHfConfig:
  model_type: str
  vocab_size: int = 1024
  hidden_size: int = 128
  intermediate_size: int = 256
  num_hidden_layers: int = 1
  num_attention_heads: int = 4
  num_key_value_heads: int | None = 2
  head_dim: int | None = 64
  tie_word_embeddings: bool = True
  attention_bias: bool = False


class VllmHfManifestTest(absltest.TestCase):

  def test_build_from_hf_manifest_dispatches_on_model_type(self):
    cfg = _FakeHfConfig(model_type="qwen3")
    self.assertIsInstance(cfg, vllm_hf_manifest.HfModelConfig)
    manifest = vllm_hf_manifest.build_from_hf_manifest(cfg, tp_size=2)
    self.assertEqual(manifest["model_type"], "qwen3")

  def test_build_from_hf_manifest_rejects_unsupported_or_invalid_model_type(
      self,
  ):
    with self.assertRaisesRegex(NotImplementedError, "model_type='llama'"):
      vllm_hf_manifest.build_from_hf_manifest(_FakeHfConfig(model_type="llama"))
    with self.assertRaisesRegex(ValueError, "non-empty str"):
      vllm_hf_manifest.build_from_hf_manifest(_FakeHfConfig(model_type=""))
    with self.assertRaises(AttributeError):
      vllm_hf_manifest.build_from_hf_manifest(object())  # type: ignore[arg-type]

  def test_derive_hf_variables_from_manifest(self):
    template_rules = [
        {
            "id": "embed",
            "source_pattern": ["model.embed.weight"],
            "target_pattern": ["model.embed.weight"],
            "ops": [{"pad": {"axis": 0, "sizes": [0, 3]}}],
        },
        {
            "id": "proj",
            "source_pattern": ["model.layers.{i}.proj.weight"],
            "target_pattern": ["model.layers.{i}.proj.weight"],
            "ops": [{"transpose": {"axes": [1, 0]}}],
        },
        {
            "id": "lm_head",
            "source_pattern": ["lm_head.weight"],
            "target_pattern": ["lm_head.weight"],
            "ops": [{"transpose": {"axes": [1, 0]}}],
        },
    ]
    spec_tied = hf_mapping.generate_manifest(
        model_type="synthetic",
        template_rules=template_rules,
        axes={"i": 1},
        sharding_rules={
            "embed": (None, None),
            "proj": (None, None),
            "lm_head": (None, None),
        },
        dropped_keys=["lm_head.weight"],
    )
    nnx_state_tied = nnx.State({
        "rng": nnx.Param(jnp.zeros((2,), dtype=jnp.float32)),
        "model": {
            "embed": {
                "weight": nnx.Param(jnp.zeros((11, 8), dtype=jnp.float32))
            },
            "layers": {
                "0": {
                    "proj": {
                        "weight": nnx.Param(
                            jnp.zeros((8, 12), dtype=jnp.bfloat16)
                        )
                    }
                }
            },
        },
    })

    vars_tied = vllm_hf_manifest.derive_hf_variables_from_manifest(
        spec_tied, nnx_state_tied
    )
    names_tied = {v.name: v for v in vars_tied}
    self.assertNotIn("rng", names_tied)
    self.assertNotIn("lm_head.weight", names_tied)
    self.assertEqual(names_tied["model.embed.weight"].shape, (8, 8))
    self.assertEqual(names_tied["model.embed.weight"].item_size, 4)
    self.assertEqual(
        names_tied["model.layers.0.proj.weight"].shape, (12, 8)
    )
    self.assertEqual(names_tied["model.layers.0.proj.weight"].item_size, 2)

    spec_untied = hf_mapping.generate_manifest(
        model_type="synthetic",
        template_rules=template_rules,
        axes={"i": 1},
        sharding_rules={
            "embed": (None, None),
            "proj": (None, None),
            "lm_head": (None, None),
        },
        dropped_keys=[],
    )
    flat_untied = {
        "model.embed.weight": jnp.zeros((11, 8), dtype=jnp.float32),
        "model.layers.0.proj.weight": jnp.zeros((8, 12), dtype=jnp.float32),
        "lm_head.weight": jnp.zeros((8, 11), dtype=jnp.float32),
    }
    vars_untied = vllm_hf_manifest.derive_hf_variables_from_manifest(
        spec_untied, flat_untied
    )
    names_untied = {v.name: v for v in vars_untied}
    self.assertIn("lm_head.weight", names_untied)
    self.assertEqual(names_untied["lm_head.weight"].shape, (11, 8))

    incomplete = dict(flat_untied)
    incomplete.pop("lm_head.weight")
    with self.assertRaisesRegex(KeyError, "lm_head.weight"):
      vllm_hf_manifest.derive_hf_variables_from_manifest(
          spec_untied, incomplete
      )

    with self.assertRaisesRegex(TypeError, "nnx.State or Mapping"):
      vllm_hf_manifest.derive_hf_variables_from_manifest(
          spec_untied, "not_a_state"  # type: ignore[arg-type]
      )


if __name__ == "__main__":
  absltest.main()
