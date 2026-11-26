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

"""Unit tests for model-family vLLM rollout configuration helpers."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.common import datatypes as common_datatypes
from tunix.generate import mappings as mappings_lib
from tunix.generate import vllm_model_config
from tunix.models.qwen3p5 import mapping_vllm_jax as qwen3p5_mapping
from tunix.utils import maxtext_utils


class VllmModelConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      ("gemma4-e2b", True),
      ("gemma4_e2b", True),
      ("gemma_4", True),
      ("google/gemma-4-E2B-it", True),
      ("/ckpt/qwen3.5-baselines/gemma-4-E2B-it", True),
      ("gemma4-e2b/llama_cmp", True),
      ("gemma-3-4b", False),
      ("gemma4b", False),
      ("/ckpt/gemma-4-baselines/Qwen3.5-2B", False),
  )
  def test_is_gemma4_model(self, model_name: str, expected: bool):
    self.assertEqual(vllm_model_config.is_gemma4_model(model_name), expected)

  @parameterized.parameters(
      ("Qwen/Qwen3.5-0.8B", True, False),
      ("qwen3p5_0p8b", True, False),
      ("qwen3_5-2b", True, False),
      ("Qwen/Qwen3.5-2B-amoeba", True, False),
      ("/ckpt/Qwen3.5-2B/phi_sweep/0", True, False),
      ("/gcs/moe_experiments/Qwen3.5-2B", True, False),
      ("/gcs/Qwen3.5-2B/moe_eval", True, False),
      ("/ckpt/gemma-4-baselines/Qwen3.5-2B", True, False),
      ("Qwen/Qwen3.5-35B-A3B", True, True),
      ("Qwen/Qwen3.5-397B-A17B", True, True),
      ("qwen3p5_35b_a3b_it", True, True),
      ("Qwen3.5-35B-A3B_FP8", True, True),
      ("Qwen/Qwen3.5-MoE-35B", True, True),
      ("/ckpt/qwen3p5/Qwen3.5-35B-A3B/0/items", True, True),
      ("Qwen/Qwen3-8B", False, False),
      ("Qwen/Qwen3-0.6B", False, False),
      ("Qwen/Qwen3-5B", False, False),
      ("qwen3_1p7b", False, False),
      ("Qwen/Qwen3-30B-A3B", False, False),
      ("/ckpt/myqwen3/items", False, False),
      ("/ckpt/qwen3.5-baselines/Qwen3-8B/0/items", False, False),
  )
  def test_qwen3p5_and_moe_detection(
      self, model_name: str, expected_qwen3p5: bool, expected_moe: bool
  ):
    self.assertEqual(
        vllm_model_config.is_qwen3p5_model(model_name), expected_qwen3p5
    )
    self.assertEqual(
        vllm_model_config.is_qwen3p5_moe_model(model_name), expected_moe
    )

  def test_active_tunix_weight_sync_mappings_and_hf_overrides(self):
    gemma4_cfg = vllm_model_config.vllm_mapping_config_for_model(
        "google/gemma-4-E2B-it"
    )
    self.assertIsNotNone(gemma4_cfg.preprocess_src_state)
    self.assertIsNotNone(gemma4_cfg.to_hf_mappings)
    assert gemma4_cfg.to_hf_mappings is not None
    self.assertIn("layers.*.mlp.gate_up_proj.kernel", gemma4_cfg.to_hf_mappings)
    self.assertIn("layers.*.attn.q_einsum.w", gemma4_cfg.to_hf_mappings)

    qwen3p5_cfg = vllm_model_config.vllm_mapping_config_for_model(
        "Qwen/Qwen3.5-0.8B"
    )
    self.assertIs(
        qwen3p5_cfg.preprocess_src_state, qwen3p5_mapping.preprocess_src_state
    )
    self.assertEqual(qwen3p5_cfg.to_hf_mappings, qwen3p5_mapping.TO_HF_MAPPINGS)
    self.assertEqual(
        vllm_model_config.vllm_hf_overrides_for_model("Qwen/Qwen3.5-0.8B"),
        {"architectures": ["Qwen3_5ForCausalLM"]},
    )
    self.assertEmpty(
        vllm_model_config.vllm_hf_overrides_for_model(
            "Qwen/Qwen3.5-0.8B", sampler_type="vllm"
        )
    )
    self.assertEmpty(
        vllm_model_config.vllm_hf_overrides_for_model(
            "Qwen/Qwen3.5-0.8B",
            weight_sync_mode=common_datatypes.WeightSyncMode.RAIDEN,
            init_with_random_weights=False,
        )
    )

    qwen3_cfg = vllm_model_config.vllm_mapping_config_for_model(
        "/ckpt/myqwen3/items"
    )
    self.assertTrue(qwen3_cfg.to_hf_mappings)
    self.assertIsNone(qwen3_cfg.preprocess_src_state)
    self.assertEmpty(
        vllm_model_config.vllm_hf_overrides_for_model("/ckpt/myqwen3/items")
    )

    with self.assertRaisesRegex(NotImplementedError, "Qwen3.5 MoE"):
      vllm_model_config.vllm_mapping_config_for_model(
          "Qwen/Qwen3.5-35B-A3B",
          weight_sync_mode=common_datatypes.WeightSyncMode.RAIDEN,
      )
    self.assertEmpty(
        vllm_model_config.vllm_hf_overrides_for_model("Qwen/Qwen3.5-35B-A3B")
    )

  def test_basic_weight_sync_selects_hf_mapping_for_qwen3_only(self):
    qwen3_basic_cfg = vllm_model_config.vllm_mapping_config_for_model(
        "Qwen/Qwen3-1.7B",
        weight_sync_mode="basic",
    )
    self.assertEqual(qwen3_basic_cfg, mappings_lib.MappingConfig())

  @parameterized.parameters(
      ("google/gemma-4-E2B-it",),
      ("Qwen/Qwen3.5-0.8B",),
      ("Qwen/Qwen3.5-35B-A3B",),
  )
  def test_basic_weight_sync_raises_not_implemented_for_unsupported_families(
      self, model_name: str
  ):
    with self.assertRaisesRegex(
        NotImplementedError, "Hugging Face-to-vLLM JAX mapping"
    ):
      vllm_model_config.vllm_mapping_config_for_model(
          model_name,
          weight_sync_mode="basic",
      )

  @parameterized.parameters(
      ("Qwen/Qwen3.5-0.8B", "", common_datatypes.WeightSyncMode.NONE),
      ("Qwen/Qwen3.5-35B-A3B", "", common_datatypes.WeightSyncMode.NOOP),
      ("Qwen/Qwen3-1.7B", "", common_datatypes.WeightSyncMode.NONE),
      ("google/gemma-4-E2B-it", "", common_datatypes.WeightSyncMode.NOOP),
      ("meta-llama/Llama-3.1-8B", "", common_datatypes.WeightSyncMode.NONE),
      ("Qwen/Qwen3.5-35B-A3B", "qwen3.5-35b-a3b", "raiden"),
      ("meta-llama/Llama-3.1-8B", "llama3.1-8b", "raiden"),
  )
  def test_non_tunix_weight_sync_returns_empty_mapping_for_all_families(
      self,
      model_name: str,
      maxtext_model_name: str,
      weight_sync_mode: common_datatypes.WeightSyncMode | str,
  ):
    self.assertEqual(
        vllm_model_config.vllm_mapping_config_for_model(
            model_name,
            maxtext_model_name=maxtext_model_name,
            weight_sync_mode=weight_sync_mode,
        ),
        mappings_lib.MappingConfig(),
    )
    overrides = vllm_model_config.vllm_hf_overrides_for_model(
        model_name,
        maxtext_model_name=maxtext_model_name,
        weight_sync_mode=weight_sync_mode,
    )
    if maxtext_model_name:
      self.assertEqual(overrides, dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES))
    elif not vllm_model_config.is_gemma4_model(model_name):
      self.assertEmpty(overrides)

  @parameterized.parameters(
      ("meta-llama/Llama-3.1-8B",),
      ("/ckpt/qwen3.5-baselines/Llama-3.1-8B/0/items",),
      ("/ckpt/gemma-4-baselines/mistral-nemo/0/items",),
  )
  def test_unsupported_family_under_parent_dir_raises_value_error(
      self, model_name: str
  ):
    self.assertFalse(vllm_model_config.is_qwen3p5_model(model_name))
    self.assertFalse(vllm_model_config.is_gemma4_model(model_name))
    self.assertEmpty(vllm_model_config.vllm_hf_overrides_for_model(model_name))
    with self.assertRaisesRegex(ValueError, "supports Qwen3, Qwen3.5"):
      vllm_model_config.vllm_mapping_config_for_model(model_name)

  @parameterized.parameters(
      (
          "Qwen/Qwen3.5-0.8B",
          "vllm",
          1,
          common_datatypes.WeightSyncMode.RAIDEN,
          False,
          True,
          ValueError,
          "use inprocess_vllm instead",
      ),
      (
          "google/gemma-4-E2B-it",
          "vllm",
          1,
          "raiden",
          False,
          True,
          ValueError,
          "use inprocess_vllm instead",
      ),
      (
          "Qwen/Qwen3.5-35B-A3B",
          "inprocess_vllm",
          1,
          "raiden",
          False,
          True,
          NotImplementedError,
          "Qwen3.5 MoE",
      ),
      (
          "Qwen/Qwen3.5-2B",
          "inprocess_vllm",
          2,
          "raiden",
          False,
          True,
          NotImplementedError,
          "tensor_parallel_size=1 only",
      ),
      (
          "Qwen/Qwen3.5-0.8B",
          "inprocess_vllm",
          1,
          "raiden",
          True,
          True,
          NotImplementedError,
          "Qwen3.5 LoRA",
      ),
      (
          "Qwen/Qwen3.5-0.8B",
          "inprocess_vllm",
          1,
          "raiden",
          False,
          False,
          ValueError,
          "requires init_with_random_weights=True",
      ),
      (
          "Qwen/Qwen3.5-0.8B",
          "inprocess_vllm",
          1,
          "invalid_mode",
          False,
          True,
          ValueError,
          "not a valid WeightSyncMode",
      ),
      (
          "Qwen/Qwen3-1.7B",
          "vllm",
          1,
          "basic",
          False,
          True,
          ValueError,
          "use inprocess_vllm instead",
      ),
      (
          "Qwen/Qwen3-1.7B",
          "inprocess_vllm",
          1,
          "basic",
          True,
          True,
          NotImplementedError,
          "LoRA rollout weight sync",
      ),
      (
          "Qwen/Qwen3.5-0.8B",
          "inprocess_vllm",
          1,
          "basic",
          False,
          True,
          NotImplementedError,
          "Hugging Face-to-vLLM JAX mapping",
      ),
      (
          "google/gemma-4-E2B-it",
          "inprocess_vllm",
          1,
          "basic",
          False,
          True,
          NotImplementedError,
          "Hugging Face-to-vLLM JAX mapping",
      ),
      (
          "meta-llama/Llama-3.1-8B",
          "inprocess_vllm",
          1,
          "basic",
          False,
          True,
          ValueError,
          "supports Qwen3",
      ),
  )
  def test_validate_rejects_invalid_or_unsupported_configs(
      self,
      model_name: str,
      sampler_type: str,
      tp_size: int,
      weight_sync_mode: common_datatypes.WeightSyncMode | str,
      use_lora: bool,
      init_with_random_weights: bool,
      exc_type: type[Exception],
      expected_regex: str,
  ):
    with self.assertRaisesRegex(exc_type, expected_regex):
      vllm_model_config.validate_vllm_rollout_config(
          model_name,
          sampler_type=sampler_type,
          tp_size=tp_size,
          weight_sync_mode=weight_sync_mode,
          use_lora=use_lora,
          init_with_random_weights=init_with_random_weights,
      )

  @parameterized.parameters(
      (
          "Qwen/Qwen3.5-0.8B",
          "inprocess_vllm",
          1,
          common_datatypes.WeightSyncMode.RAIDEN,
          "",
          False,
          True,
      ),
      ("Qwen/Qwen3.5-0.8B", "vanilla", 2, "raiden", "", False, True),
      (
          "Qwen/Qwen3.5-2B",
          "vllm",
          2,
          common_datatypes.WeightSyncMode.NONE,
          "",
          True,
          False,
      ),
      (
          "Qwen/Qwen3.5-35B-A3B",
          "inprocess_vllm",
          2,
          common_datatypes.WeightSyncMode.NOOP,
          "",
          False,
          False,
      ),
      (
          "Qwen/Qwen3.5-35B-A3B",
          "inprocess_vllm",
          4,
          "raiden",
          "qwen3.5-35b-a3b",
          True,
          True,
      ),
      (
          "Qwen/Qwen3-1.7B",
          "inprocess_vllm",
          2,
          "basic",
          "",
          False,
          True,
      ),
  )
  def test_validate_allows_supported_and_no_sync_or_maxtext_configs(
      self,
      model_name: str,
      sampler_type: str,
      tp_size: int,
      weight_sync_mode: common_datatypes.WeightSyncMode | str,
      maxtext_model_name: str,
      use_lora: bool,
      init_with_random_weights: bool,
  ):
    vllm_model_config.validate_vllm_rollout_config(
        model_name,
        sampler_type=sampler_type,
        tp_size=tp_size,
        weight_sync_mode=weight_sync_mode,
        maxtext_model_name=maxtext_model_name,
        use_lora=use_lora,
        init_with_random_weights=init_with_random_weights,
    )

  def test_basic_weight_sync_rejects_maxtext_rollout(self):
    with self.assertRaisesRegex(
        NotImplementedError, "MaxText rollout with weight_sync_mode='basic'"
    ):
      vllm_model_config.validate_vllm_rollout_config(
          "Qwen/Qwen3-1.7B",
          sampler_type="inprocess_vllm",
          tp_size=4,
          weight_sync_mode="basic",
          maxtext_model_name="qwen3-1.7b",
      )
    with self.assertRaisesRegex(
        NotImplementedError, "MaxText rollout with weight_sync_mode='basic'"
    ):
      vllm_model_config.vllm_mapping_config_for_model(
          "Qwen/Qwen3-1.7B",
          maxtext_model_name="qwen3-1.7b",
          weight_sync_mode="basic",
      )

  @parameterized.parameters(
      ("Qwen/Qwen3-30B-A3B", True),
      ("Qwen/Qwen3-235B-A22B", True),
      ("qwen3_30b_a3b", True),
      ("qwen3-moe", True),
      ("/ckpt/myqwen3-30b-a3b/items", True),
      ("/gcs/moe_experiments/Qwen3-8B", False),
      ("/gcs/moe_experiments/myqwen3/items", False),
      ("Qwen/Qwen3-1.7B", False),
      ("Qwen/Qwen3.5-35B-A3B", False),
  )
  def test_is_qwen3_moe_model(self, model_name: str, expected: bool):
    self.assertEqual(
        vllm_model_config.is_qwen3_moe_model(model_name), expected
    )
    self.assertFalse(vllm_model_config._is_qwen3_model("Qwen/Qwen3.5-0.8B"))

  @parameterized.parameters(
      ("Qwen/Qwen3-30B-A3B",),
      ("Qwen/Qwen3-235B-A22B",),
      ("qwen3_30b_a3b",),
      ("qwen3-moe",),
      ("/ckpt/myqwen3-30b-a3b/items",),
  )
  def test_basic_weight_sync_rejects_qwen3_moe_models(self, model_name: str):
    with self.assertRaisesRegex(NotImplementedError, "Qwen3 MoE"):
      vllm_model_config.validate_vllm_rollout_config(
          model_name,
          sampler_type="inprocess_vllm",
          tp_size=1,
          weight_sync_mode="basic",
      )
    with self.assertRaisesRegex(NotImplementedError, "Qwen3 MoE"):
      vllm_model_config.vllm_mapping_config_for_model(
          model_name,
          weight_sync_mode="basic",
      )

  def test_basic_weight_sync_non_qwen3_error_mentions_qwen3_only(self):
    with self.assertRaisesRegex(ValueError, r"supports Qwen3 with") as ctx:
      vllm_model_config.vllm_mapping_config_for_model(
          "meta-llama/Llama-3.1-8B",
          weight_sync_mode="basic",
      )
    self.assertNotIn("Qwen3.5", str(ctx.exception))
    self.assertNotIn("Gemma4", str(ctx.exception))

  @parameterized.parameters(
      ("/gcs/moe_experiments/Qwen3-8B",),
      ("Qwen/Qwen3-1.7B",),
  )
  def test_basic_weight_sync_allows_dense_qwen3_models(self, model_name: str):
    vllm_model_config.validate_vllm_rollout_config(
        model_name,
        sampler_type="inprocess_vllm",
        tp_size=1,
        weight_sync_mode="basic",
    )
    self.assertEqual(
        vllm_model_config.vllm_mapping_config_for_model(
            model_name,
            weight_sync_mode="basic",
        ),
        mappings_lib.MappingConfig(),
    )


if __name__ == "__main__":
  absltest.main()
