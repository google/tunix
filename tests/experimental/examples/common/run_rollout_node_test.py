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

"""Tests for distributed rollout model-family configuration."""

from unittest import mock

from absl.testing import absltest
from tunix.experimental.examples.common import run_rollout_node
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.utils import maxtext_utils


class RunRolloutNodeTest(absltest.TestCase):

  def test_gemma4_uses_gemma4_chat_parser(self):
    parser_factory = mock.Mock(return_value=mock.sentinel.chat_parser)
    with mock.patch.dict(
        run_rollout_node.CHAT_PARSERS,
        {"gemma-4": parser_factory},
        clear=True,
    ):
      result = run_rollout_node._chat_parser_for(
          "google/gemma-4-E2B-it",
          mock.sentinel.tokenizer,
          enable_thinking=False,
      )

    self.assertIs(result, mock.sentinel.chat_parser)
    parser_factory.assert_called_once_with(
        mock.sentinel.tokenizer, enable_thinking=False
    )

  def test_gemma4_mapping_includes_preprocessing_and_fused_projections(self):
    config = run_rollout_node._mapping_config_for("google/gemma-4-E2B-it")

    # Gemma4 fuses gate/up on the source state before transfer and maps
    # separate q/k/v projections.
    self.assertIsNotNone(config.preprocess_src_state)
    self.assertIn("layers.*.mlp.gate_up_proj.kernel", config.to_hf_mappings)
    self.assertIn("layers.*.attn.q_einsum.w", config.to_hf_mappings)
    self.assertIn("layers.*.attn.k_einsum.w", config.to_hf_mappings)
    self.assertIn("layers.*.attn.v_einsum.w", config.to_hf_mappings)

  def test_qwen3_mapping_is_selected_for_qwen3(self):
    config = run_rollout_node._mapping_config_for("Qwen/Qwen3-8B")

    self.assertTrue(config.to_hf_mappings)
    self.assertIsNone(config.preprocess_src_state)

  def test_unsupported_family_raises(self):
    with self.assertRaisesRegex(ValueError, "supports Qwen3 and Gemma4"):
      run_rollout_node._mapping_config_for("meta-llama/Llama-3-8B")

  def test_gemma4_vllm_overrides_match_text_only_reference(self):
    args = run_rollout_node._parse_args(["--model_id=google/gemma-4-E2B-it"])

    overrides = run_rollout_node._vllm_hf_overrides(args)

    self.assertEqual(overrides["architectures"], ["Gemma4ForCausalLM"])
    self.assertEqual(overrides["final_logit_softcapping"], 30.0)
    self.assertEqual(
        overrides["text_config"]["final_logit_softcapping"], 30.0
    )

  def test_qwen_does_not_get_gemma4_overrides(self):
    args = run_rollout_node._parse_args(["--model_id=Qwen/Qwen3-8B"])

    self.assertEmpty(run_rollout_node._vllm_hf_overrides(args))

  def test_maxtext_overrides_take_precedence_over_gemma4(self):
    args = run_rollout_node._parse_args([
        "--model_id=google/gemma-4-E2B-it",
        "--maxtext_model_name=gemma3-4b",
    ])

    overrides = run_rollout_node._vllm_hf_overrides(args)

    self.assertEqual(overrides, dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES))

  def test_registered_parser_is_gemma4_specific(self):
    self.assertIs(
        run_rollout_node.CHAT_PARSERS["gemma-4"],
        parser.Gemma4ChatTemplateParser,
    )

  def test_inprocess_vllm_sampler_sets_processed_logprobs_mode(self):
    args = run_rollout_node._parse_args(["--model_id=Qwen/Qwen3-8B"])
    mock_vllm_sampler = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.eos_token_id = 1
    with (
        mock.patch.object(
            run_rollout_node,
            "_import_vllm_sampler",
            return_value=mock_vllm_sampler,
        ),
        mock.patch.object(
            run_rollout_node,
            "_create_rollout_mesh",
            return_value=mock.sentinel.mesh,
        ),
        mock.patch.object(
            run_rollout_node,
            "_get_tensor_parallel_size",
            return_value=1,
        ),
        mock.patch.object(
            run_rollout_node,
            "_eos_token_ids",
            return_value=[1],
        ),
        mock.patch.object(
            run_rollout_node,
            "_rollout_config_kwargs",
            return_value={},
        ),
        mock.patch(
            "tunix.experimental.rollout.inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter"
        ),
    ):
      run_rollout_node._create_inprocess_vllm_sampler(args, mock_tokenizer)

    mock_vllm_sampler.VllmConfig.assert_called_once()
    engine_kwargs = mock_vllm_sampler.VllmConfig.call_args.kwargs[
        "engine_kwargs"
    ]
    self.assertEqual(engine_kwargs.get("logprobs_mode"), "processed_logprobs")


if __name__ == "__main__":
  absltest.main()
