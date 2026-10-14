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

import contextlib
import io
import sys
import types
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import gcs_cache
from tunix.experimental.examples.common import run_rollout_node
from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import vllm_sampler_adapter
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.utils import maxtext_utils

_ROUTED_EXPERTS_ARGV = (
    "--maxtext_model_name=qwen3-30b-a3b",
    "--return_routed_experts",
)


def _vllm_engine_kwargs(argv: list[str]) -> dict[str, object]:
  """Returns the AsyncEngineArgs kwargs `_create_vllm_sampler` builds."""
  arg_utils = types.ModuleType("vllm.engine.arg_utils")
  arg_utils.AsyncEngineArgs = mock.Mock()
  with mock.patch.dict(sys.modules, {"vllm.engine.arg_utils": arg_utils}):
    with mock.patch.object(vllm_sampler_adapter, "VllmSamplerAdapter"):
      run_rollout_node._create_vllm_sampler(
          run_rollout_node._parse_args(argv), tokenizer=None
      )
  return arg_utils.AsyncEngineArgs.call_args.kwargs


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

  def test_in_flight_weight_updates_flag_sets_partial_rollout(self):
    args_default = run_rollout_node._parse_args(["--model_id=Qwen/Qwen3-8B"])
    self.assertFalse(args_default.in_flight_weight_updates)
    kwargs_default = run_rollout_node._rollout_config_kwargs(args_default)
    self.assertFalse(kwargs_default["partial_rollout"])

    args_enabled = run_rollout_node._parse_args([
        "--model_id=Qwen/Qwen3-8B",
        "--in_flight_weight_updates",
    ])
    self.assertTrue(args_enabled.in_flight_weight_updates)
    kwargs_enabled = run_rollout_node._rollout_config_kwargs(args_enabled)
    self.assertTrue(kwargs_enabled["partial_rollout"])

  def test_routed_experts_flags_reach_rollout_config(self):
    defaults = run_rollout_node._rollout_config_kwargs(
        run_rollout_node._parse_args([])
    )
    self.assertFalse(defaults["return_routed_experts"])
    self.assertEqual(defaults["routed_experts_encode_mode"], "auto")
    self.assertIsNone(defaults["routed_experts_num_experts"])

    enabled = run_rollout_node._rollout_config_kwargs(
        run_rollout_node._parse_args([
            "--return_routed_experts",
            "--routed_experts_encode_mode=uint10",
            "--routed_experts_num_experts=256",
        ])
    )
    self.assertTrue(enabled["return_routed_experts"])
    self.assertEqual(enabled["routed_experts_encode_mode"], "uint10")
    self.assertEqual(enabled["routed_experts_num_experts"], 256)

  def test_unknown_routed_experts_encode_mode_is_rejected(self):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(["--routed_experts_encode_mode=int8"])
    self.assertIn("invalid choice: 'int8'", stderr.getvalue())

  def test_vanilla_sampler_rejects_return_routed_experts(self):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(
          ["--sampler=vanilla", "--return_routed_experts"]
      )
    self.assertIn("needs a vLLM sampler", stderr.getvalue())

  def test_vllm_sampler_captures_routed_experts_only_when_asked(self):
    default = _vllm_engine_kwargs(["--maxtext_model_name=qwen3-30b-a3b"])
    self.assertNotIn("enable_return_routed_experts", default)
    self.assertNotIn(
        "return_routed_experts", default["additional_config"]["maxtext_config"]
    )

    enabled = _vllm_engine_kwargs(list(_ROUTED_EXPERTS_ARGV))
    self.assertIs(enabled["enable_return_routed_experts"], True)
    self.assertIs(
        enabled["additional_config"]["maxtext_config"]["return_routed_experts"],
        True,
    )

  def test_inprocess_vllm_sampler_captures_routed_experts(self):
    vllm_sampler = mock.Mock()
    with (
        mock.patch.object(
            run_rollout_node, "_import_vllm_sampler", return_value=vllm_sampler
        ),
        mock.patch.object(run_rollout_node, "_create_rollout_mesh"),
        mock.patch.object(
            inprocess_vllm_sampler_adapter, "InprocessVllmSamplerAdapter"
        ),
    ):
      run_rollout_node._create_inprocess_vllm_sampler(
          run_rollout_node._parse_args(list(_ROUTED_EXPERTS_ARGV)),
          tokenizer=None,
      )

    config_kwargs = vllm_sampler.VllmConfig.call_args.kwargs
    self.assertIs(config_kwargs["return_routed_experts"], True)
    self.assertIs(
        config_kwargs["additional_config"]["maxtext_config"][
            "return_routed_experts"
        ],
        True,
    )

  def test_main_restores_jax_cache_before_jax_init(self):
    events: list[str] = []
    context = mock.Mock()

    def _fake_jax_init() -> None:
      events.append("jax_init")
      raise RuntimeError("stop at jax init")

    context.jax.initialize.side_effect = _fake_jax_init

    with (
        mock.patch.object(run_rollout_node.importlib, "import_module"),
        mock.patch.object(
            gcs_cache,
            "restore_jax_cache",
            side_effect=lambda params=None: (
                events.append(f"restore:{params.sampler}") or True
            ),
        ) as mock_restore,
    ):
      with self.assertRaisesRegex(RuntimeError, "stop at jax init"):
        run_rollout_node.main(["--sampler=vanilla"], context=context)

    mock_restore.assert_called_once()
    self.assertEqual(mock_restore.call_args.kwargs["params"].sampler, "vanilla")
    self.assertEqual(events, ["restore:vanilla", "jax_init"])


if __name__ == "__main__":
  absltest.main()
