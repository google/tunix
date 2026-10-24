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
import json
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.examples.common import run_rollout_node
from tunix.dist_rl.rollout import inprocess_vllm_sampler
from tunix.dist_rl.rollout import vllm_sampler
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
    with mock.patch.object(vllm_sampler, "VllmSampler"):
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

  def test_detokenize_flag_defaults_off(self):
    self.assertFalse(run_rollout_node._parse_args([]).detokenize)
    self.assertTrue(
        run_rollout_node._parse_args(["--detokenize=true"]).detokenize
    )
    self.assertTrue(run_rollout_node._parse_args(["--detokenize"]).detokenize)

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
            inprocess_vllm_sampler, "InprocessVllmSampler"
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

  def test_vllm_samplers_plumb_float32_logit_flags_when_set(self):
    default_kwargs = _vllm_engine_kwargs(["--maxtext_model_name=qwen3-30b-a3b"])
    default_mt_cfg = default_kwargs["additional_config"]["maxtext_config"]
    self.assertNotIn("float32_gate_logits", default_mt_cfg)
    self.assertNotIn("float32_logits", default_mt_cfg)

    argv = [
        "--maxtext_model_name=qwen3-30b-a3b",
        "--maxtext_float32_gate_logits=true",
        "--maxtext_float32_logits=false",
    ]
    vllm_kwargs = _vllm_engine_kwargs(argv)
    vllm_mt_cfg = vllm_kwargs["additional_config"]["maxtext_config"]
    self.assertIs(vllm_mt_cfg["float32_gate_logits"], True)
    self.assertIs(vllm_mt_cfg["float32_logits"], False)

    vllm_sampler = mock.Mock()
    with (
        mock.patch.object(
            run_rollout_node, "_import_vllm_sampler", return_value=vllm_sampler
        ),
        mock.patch.object(run_rollout_node, "_create_rollout_mesh"),
        mock.patch.object(
            inprocess_vllm_sampler, "InprocessVllmSampler"
        ),
    ):
      run_rollout_node._create_inprocess_vllm_sampler(
          run_rollout_node._parse_args(argv),
          tokenizer=None,
      )
    inproc_mt_cfg = vllm_sampler.VllmConfig.call_args.kwargs[
        "additional_config"
    ]["maxtext_config"]
    self.assertIs(inproc_mt_cfg["float32_gate_logits"], True)
    self.assertIs(inproc_mt_cfg["float32_logits"], False)

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

  def test_episode_timeout_derives_drain_timeout_in_rollout_config(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("EPISODE_TIMEOUT_SECS", None)
      defaults = run_rollout_node._rollout_config_kwargs(
          run_rollout_node._parse_args([])
      )
    self.assertEqual(defaults["drain_timeout_s"], 660.0)

    deepswe_cfg = run_rollout_node._rollout_config_kwargs(
        run_rollout_node._parse_args(["--episode_timeout_secs=1800"])
    )
    self.assertEqual(deepswe_cfg["drain_timeout_s"], 1860.0)

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(["--episode_timeout_secs=0"])
    self.assertIn(
        "episode_timeout_secs must be a positive finite number",
        stderr.getvalue(),
    )

  def test_vllm_config_json_merges_35b_recipe_config(self):
    recipe_vllm_config = {
        "enable_expert_parallel": True,
        "prefix_cache_retention_interval": 0,
        "mamba_cache_mode": "align",
        "kv_cache_dtype": "bfloat16",
        "block_size": 256,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "reasoning_parser": "qwen3",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "additional_config": {
            "sharding": {
                "sharding_strategy": {
                    "expert_parallelism": 4,
                    "tensor_parallelism": 1,
                    "enable_dp_attention": True,
                }
            },
            "custom_mamba_cache_multiplier": 16,
            "maxtext_config": {
                "use_multimodal": False,
                "per_device_batch_size": 0.0,
            },
        },
    }
    kwargs = _vllm_engine_kwargs([
        "--sampler=vllm",
        "--maxtext_model_name=qwen3.5-35b-a3b",
        "--maxtext_attention=vllm_rpa",
        "--prefuse_moe_weights=true",
        "--return_routed_experts=true",
        "--maxtext_float32_gate_logits=true",
        "--maxtext_float32_logits=true",
        "--vllm_max_model_len=65536",
        "--vllm_max_num_seqs=16",
        "--vllm_max_num_batched_tokens=2048",
        "--vllm_async_scheduling",
        "--vllm_hbm_utilization=0.9",
        "--enable_prefix_caching=true",
        f"--vllm_config_json={json.dumps(recipe_vllm_config)}",
    ])
    self.assertEqual(kwargs["max_model_len"], 65536)
    self.assertEqual(kwargs["max_num_seqs"], 16)
    self.assertEqual(kwargs["max_num_batched_tokens"], 2048)
    self.assertTrue(kwargs["async_scheduling"])
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.9)
    self.assertTrue(kwargs["enable_prefix_caching"])
    self.assertTrue(kwargs["enable_return_routed_experts"])
    self.assertTrue(kwargs["enable_expert_parallel"])
    self.assertEqual(kwargs["prefix_cache_retention_interval"], 0)
    self.assertEqual(kwargs["mamba_cache_mode"], "align")
    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
    self.assertEqual(kwargs["block_size"], 256)
    self.assertTrue(kwargs["enable_chunked_prefill"])
    self.assertTrue(kwargs["language_model_only"])
    self.assertEqual(kwargs["reasoning_parser"], "qwen3")
    self.assertEqual(kwargs["limit_mm_per_prompt"], {"image": 0, "video": 0})
    additional = kwargs["additional_config"]
    self.assertIsInstance(additional, dict)
    self.assertEqual(additional["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(
        additional["sharding"],
        {
            "sharding_strategy": {
                "expert_parallelism": 4,
                "tensor_parallelism": 1,
                "enable_dp_attention": True,
            }
        },
    )
    mt_cfg = additional["maxtext_config"]
    self.assertFalse(mt_cfg["scan_layers"])
    self.assertTrue(mt_cfg["allow_split_physical_axes"])
    self.assertEqual(mt_cfg["attention"], "vllm_rpa")
    self.assertTrue(mt_cfg["prefuse_moe_weights"])
    self.assertTrue(mt_cfg["return_routed_experts"])
    self.assertTrue(mt_cfg["float32_gate_logits"])
    self.assertTrue(mt_cfg["float32_logits"])
    self.assertFalse(mt_cfg["use_multimodal"])
    self.assertEqual(mt_cfg["per_device_batch_size"], 0.0)

  def test_vllm_config_json_rejected_for_non_vllm_sampler(self):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(
          ["--sampler=inprocess_vllm", "--vllm_config_json={}"]
      )
    self.assertIn(
        "--vllm_config_json is only supported when --sampler=vllm",
        stderr.getvalue(),
    )

  def test_vllm_config_json_rejects_forbidden_keys(self):
    with self.assertRaisesRegex(ValueError, "must decode to a JSON object"):
      _vllm_engine_kwargs(["--sampler=vllm", "--vllm_config_json=[1, 2]"])

    for key in run_rollout_node._FORBIDDEN_VLLM_CONFIG_JSON_KEYS:
      with self.subTest(top_level_key=key):
        with self.assertRaisesRegex(
            ValueError, "managed by dedicated CLI flags"
        ):
          _vllm_engine_kwargs([
              "--sampler=vllm",
              f"--vllm_config_json={json.dumps({key: 1})}",
          ])

    for mt_key in run_rollout_node._FORBIDDEN_MAXTEXT_CONFIG_JSON_KEYS:
      with self.subTest(maxtext_key=mt_key):
        payload = {"additional_config": {"maxtext_config": {mt_key: 1}}}
        with self.assertRaisesRegex(
            ValueError, "managed by dedicated CLI flags"
        ):
          _vllm_engine_kwargs([
              "--sampler=vllm",
              f"--vllm_config_json={json.dumps(payload)}",
          ])


if __name__ == "__main__":
  absltest.main()
