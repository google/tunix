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
from unittest import mock

from absl.testing import absltest
import transformers
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.examples.common import run_rollout_node
from tunix.dist_rl.rollout import inprocess_vllm_sampler
from tunix.dist_rl.rollout import vllm_sampler
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import rollout_worker
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.utils import maxtext_utils

_ROUTED_EXPERTS_ARGV = (
    "--maxtext_model_name=qwen3-30b-a3b",
    "--return_routed_experts",
)


def _vllm_rollout_config(argv: list[str]) -> rollout_worker.RolloutConfig:
  """Returns the RolloutConfig `_create_vllm_sampler` passes to VllmSampler."""
  with mock.patch.object(
      vllm_sampler, "VllmSampler", autospec=True
  ) as sampler_cls:
    run_rollout_node._create_vllm_sampler(
        run_rollout_node._parse_args(argv), tokenizer=None
    )
  return sampler_cls.call_args.kwargs["config"]


def _vllm_engine_kwargs(argv: list[str]) -> dict[str, object]:
  """Returns the AsyncEngineArgs kwargs of the `--sampler=vllm` worker."""
  sampler = run_rollout_node._create_vllm_sampler(
      run_rollout_node._parse_args(argv), tokenizer=None
  )
  return sampler._engine_kwargs


def _inprocess_vllm_rollout_config(
    argv: list[str],
) -> rollout_worker.RolloutConfig:
  """Returns the RolloutConfig `_create_inprocess_vllm_sampler` builds."""
  with (
      mock.patch.object(run_rollout_node, "_import_vllm_sampler"),
      mock.patch.object(run_rollout_node, "_create_rollout_mesh"),
      mock.patch.object(
          inprocess_vllm_sampler, "InprocessVllmSampler", autospec=True
      ) as sampler_cls,
  ):
    run_rollout_node._create_inprocess_vllm_sampler(
        run_rollout_node._parse_args(argv), tokenizer=None
    )
  return sampler_cls.call_args.kwargs["config"]


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
    config_default = run_rollout_node._rollout_config(args_default)
    self.assertFalse(config_default.partial_rollout)

    args_enabled = run_rollout_node._parse_args([
        "--model_id=Qwen/Qwen3-8B",
        "--in_flight_weight_updates",
    ])
    self.assertTrue(args_enabled.in_flight_weight_updates)
    config_enabled = run_rollout_node._rollout_config(args_enabled)
    self.assertTrue(config_enabled.partial_rollout)

  def test_detokenize_flag_defaults_on_and_reaches_vllm_sampler_config(self):
    self.assertTrue(run_rollout_node._parse_args([]).detokenize)
    self.assertTrue(
        run_rollout_node._parse_args(["--detokenize=true"]).detokenize
    )
    self.assertTrue(run_rollout_node._parse_args(["--detokenize"]).detokenize)
    self.assertFalse(
        run_rollout_node._parse_args(["--detokenize=false"]).detokenize
    )
    self.assertIs(
        _vllm_rollout_config([]).rollout_vllm_sampling_kwargs["detokenize"],
        True,
    )
    self.assertIs(
        _vllm_rollout_config(
            ["--detokenize=false"]
        ).rollout_vllm_sampling_kwargs["detokenize"],
        False,
    )

  def test_routed_experts_flags_reach_rollout_config(self):
    defaults = run_rollout_node._rollout_config(
        run_rollout_node._parse_args([])
    )
    self.assertFalse(defaults.return_routed_experts)
    self.assertEqual(defaults.routed_experts_encode_mode, "auto")
    self.assertIsNone(defaults.routed_experts_num_experts)

    enabled = run_rollout_node._rollout_config(
        run_rollout_node._parse_args([
            "--return_routed_experts",
            "--routed_experts_encode_mode=uint10",
            "--routed_experts_num_experts=256",
        ])
    )
    self.assertTrue(enabled.return_routed_experts)
    self.assertEqual(enabled.routed_experts_encode_mode, "uint10")
    self.assertEqual(enabled.routed_experts_num_experts, 256)

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
    self.assertIs(default["enable_return_routed_experts"], False)
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
    config = _inprocess_vllm_rollout_config(list(_ROUTED_EXPERTS_ARGV))
    self.assertIs(config.return_routed_experts, True)
    self.assertIs(
        config.rollout_vllm_additional_config["maxtext_config"][
            "return_routed_experts"
        ],
        True,
    )

  def test_vllm_samplers_plumb_float32_logit_flags_when_set(self):
    default_mt_cfg = _vllm_engine_kwargs(
        ["--maxtext_model_name=qwen3-30b-a3b"]
    )["additional_config"]["maxtext_config"]
    self.assertNotIn("float32_gate_logits", default_mt_cfg)
    self.assertNotIn("float32_logits", default_mt_cfg)

    argv = [
        "--maxtext_model_name=qwen3-30b-a3b",
        "--maxtext_float32_gate_logits=true",
        "--maxtext_float32_logits=false",
    ]
    vllm_mt_cfg = _vllm_engine_kwargs(argv)["additional_config"][
        "maxtext_config"
    ]
    self.assertIs(vllm_mt_cfg["float32_gate_logits"], True)
    self.assertIs(vllm_mt_cfg["float32_logits"], False)

    inproc_mt_cfg = _inprocess_vllm_rollout_config(
        argv
    ).rollout_vllm_additional_config["maxtext_config"]
    self.assertIs(inproc_mt_cfg["float32_gate_logits"], True)
    self.assertIs(inproc_mt_cfg["float32_logits"], False)

  def test_vllm_engine_length_defaults_to_prompt_plus_response(self):
    argv = [
        "--model_id=Qwen/Qwen3-8B",
        "--max_prompt_length=300",
        "--max_response_length=700",
    ]
    for sampler, config in (
        ("vllm", _vllm_rollout_config(argv)),
        ("inprocess_vllm", _inprocess_vllm_rollout_config(argv)),
    ):
      with self.subTest(sampler=sampler, max_model_len="default"):
        self.assertEqual(config.kv_cache_size, 1000)
    self.assertEqual(_vllm_engine_kwargs(argv)["max_model_len"], 1000)

    argv.append("--vllm_max_model_len=4096")
    for sampler, config in (
        ("vllm", _vllm_rollout_config(argv)),
        ("inprocess_vllm", _inprocess_vllm_rollout_config(argv)),
    ):
      with self.subTest(sampler=sampler, max_model_len=4096):
        self.assertEqual(config.kv_cache_size, 4096)
    self.assertEqual(_vllm_engine_kwargs(argv)["max_model_len"], 4096)

  def test_inprocess_vllm_rollout_config_carries_engine_settings(self):
    config = _inprocess_vllm_rollout_config([
        "--model_id=Qwen/Qwen3-8B",
        "--vllm_hbm_utilization=0.7",
        "--vllm_max_num_seqs=64",
        "--use_lora",
        "--lora_rank=8",
    ])
    self.assertEqual(config.rollout_vllm_model_version, "Qwen/Qwen3-8B")
    self.assertEqual(config.rollout_vllm_hbm_utilization, 0.7)
    self.assertEqual(config.rollout_vllm_max_num_seqs, 64)
    self.assertEqual(config.rollout_vllm_tpu_backend_type, "jax")
    self.assertEqual(
        config.rollout_vllm_lora_config, {"max_lora_rank": 8, "max_loras": 1}
    )
    self.assertIsNotNone(config.rollout_mapping_config)

  def test_main_restores_jax_cache_before_jax_init(self):
    events: list[str] = []
    context = mock.Mock()

    def _fake_jax_init() -> None:
      events.append("jax_init")
      raise RuntimeError("stop at jax init")

    context.jax.initialize.side_effect = _fake_jax_init

    with (
        mock.patch.object(run_rollout_node.importlib, "import_module"),
        mock.patch.object(otel_setup, "setup_metrics") as mock_otel_setup,
        mock.patch.object(
            gcs_cache,
            "restore_jax_cache",
            side_effect=lambda params=None: (
                events.append(f"restore:{params.sampler}") or True
            ),
        ) as mock_restore,
    ):
      with self.assertRaisesRegex(RuntimeError, "stop at jax init"):
        run_rollout_node.main(
            [
                "--sampler=vanilla",
                "--worker_id=rollout-7",
                "--run_id=run_123",
            ],
            context=context,
        )

    mock_otel_setup.assert_called_once_with(
        default_service_name="tunix-rollout",
        resource_attributes={
            "tunix.worker_id": "rollout-7",
            "tunix.role": "rollout",
            "tunix.run_id": "run_123",
        },
    )
    mock_restore.assert_called_once()
    self.assertEqual(mock_restore.call_args.kwargs["params"].sampler, "vanilla")
    self.assertEqual(events, ["restore:vanilla", "jax_init"])

  def test_episode_timeout_derives_drain_timeout_in_rollout_config(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("EPISODE_TIMEOUT_SECS", None)
      defaults = run_rollout_node._rollout_config(
          run_rollout_node._parse_args([])
      )
    self.assertEqual(defaults.drain_timeout_s, 660.0)

    deepswe_cfg = run_rollout_node._rollout_config(
        run_rollout_node._parse_args(["--episode_timeout_secs=1800"])
    )
    self.assertEqual(deepswe_cfg.drain_timeout_s, 1860.0)

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(["--episode_timeout_secs=0"])
    self.assertIn(
        "episode_timeout_secs must be a positive finite number",
        stderr.getvalue(),
    )

  def test_vllm_hbm_utilization_defaults_per_sampler(self):
    self.assertNotIn(
        "gpu_memory_utilization", _vllm_engine_kwargs(["--sampler=vllm"])
    )
    self.assertEqual(
        _vllm_engine_kwargs(["--sampler=vllm", "--vllm_hbm_utilization=0.8"])[
            "gpu_memory_utilization"
        ],
        0.8,
    )

    self.assertIsNone(
        _inprocess_vllm_rollout_config(
            ["--sampler=inprocess_vllm"]
        ).rollout_vllm_hbm_utilization
    )
    self.assertEqual(
        _inprocess_vllm_rollout_config([
            "--sampler=inprocess_vllm",
            "--vllm_hbm_utilization=0.8",
        ]).rollout_vllm_hbm_utilization,
        0.8,
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
    self.assertIs(kwargs["async_scheduling"], True)
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.9)
    self.assertIs(kwargs["enable_prefix_caching"], True)
    self.assertIs(kwargs["enable_return_routed_experts"], True)
    self.assertIs(kwargs["enable_expert_parallel"], True)
    self.assertEqual(kwargs["prefix_cache_retention_interval"], 0)
    self.assertEqual(kwargs["mamba_cache_mode"], "align")
    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
    self.assertEqual(kwargs["block_size"], 256)
    self.assertIs(kwargs["enable_chunked_prefill"], True)
    self.assertIs(kwargs["language_model_only"], True)
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
    self.assertIs(mt_cfg["scan_layers"], False)
    self.assertIs(mt_cfg["allow_split_physical_axes"], True)
    self.assertEqual(mt_cfg["attention"], "vllm_rpa")
    self.assertIs(mt_cfg["prefuse_moe_weights"], True)
    self.assertIs(mt_cfg["return_routed_experts"], True)
    self.assertIs(mt_cfg["float32_gate_logits"], True)
    self.assertIs(mt_cfg["float32_logits"], True)
    self.assertIs(mt_cfg["use_multimodal"], False)
    self.assertEqual(mt_cfg["per_device_batch_size"], 0.0)

  def test_vllm_config_json_without_maxtext_and_override_precedence(self):
    non_maxtext_json = {
        "enable_expert_parallel": True,
        "additional_config": {"sharding": {"tensor_parallelism": 4}},
    }
    kwargs = _vllm_engine_kwargs([
        "--sampler=vllm",
        f"--vllm_config_json={json.dumps(non_maxtext_json)}",
    ])
    self.assertIs(kwargs["enable_expert_parallel"], True)
    self.assertEqual(
        kwargs["additional_config"], {"sharding": {"tensor_parallelism": 4}}
    )
    self.assertNotIn("hf_overrides", kwargs)

    blank_args = run_rollout_node._parse_args(
        ["--sampler=vllm", "--vllm_config_json=   "]
    )
    self.assertIsNone(blank_args.vllm_config_json)
    self.assertNotIn(
        "additional_config",
        _vllm_engine_kwargs(["--sampler=vllm", "--vllm_config_json="]),
    )

    override_json = {
        "additional_config": {
            "maxtext_config": {"allow_split_physical_axes": False}
        }
    }
    mt_kwargs = _vllm_engine_kwargs([
        "--sampler=vllm",
        "--maxtext_model_name=qwen3.5-35b-a3b",
        f"--vllm_config_json={json.dumps(override_json)}",
    ])
    self.assertIs(
        mt_kwargs["additional_config"]["maxtext_config"][
            "allow_split_physical_axes"
        ],
        False,
    )

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

  def _assert_parse_args_rejects(
      self, argv: list[str], expected_error: str
  ) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
      run_rollout_node._parse_args(argv)
    self.assertIn(expected_error, stderr.getvalue())

  def test_vllm_config_json_rejects_forbidden_keys_and_malformed_inputs(self):
    self._assert_parse_args_rejects(
        ["--sampler=vllm", "--vllm_config_json={bad"],
        "--vllm_config_json must be a valid JSON object.",
    )
    self._assert_parse_args_rejects(
        ["--sampler=vllm", "--vllm_config_json=[1, 2]"],
        "--vllm_config_json must decode to a JSON object.",
    )
    self._assert_parse_args_rejects(
        ["--sampler=vllm", '--vllm_config_json={"additional_config": 1}'],
        "--vllm_config_json['additional_config'] must be a JSON object.",
    )
    self._assert_parse_args_rejects(
        [
            "--sampler=vllm",
            "--maxtext_model_name=qwen3.5-35b-a3b",
            '--vllm_config_json={"additional_config": {"maxtext_config": 1}}',
        ],
        "--vllm_config_json['additional_config']['maxtext_config'] must be a"
        " JSON object.",
    )
    self._assert_parse_args_rejects(
        [
            "--sampler=vllm",
            (
                '--vllm_config_json={"additional_config": {"maxtext_config":'
                ' {"use_multimodal": false}}}'
            ),
        ],
        "requires --maxtext_model_name to be set.",
    )

    for key in run_rollout_node._FORBIDDEN_ROLLOUT_KWARGS:
      with self.subTest(top_level_key=key):
        self._assert_parse_args_rejects(
            [
                "--sampler=vllm",
                f"--vllm_config_json={json.dumps({key: 1})}",
            ],
            "managed by dedicated CLI flags or rollout contracts",
        )

    for mt_key in run_rollout_node._FORBIDDEN_MAXTEXT_CONFIG_KEYS:
      with self.subTest(maxtext_key=mt_key):
        payload = {"additional_config": {"maxtext_config": {mt_key: 1}}}
        self._assert_parse_args_rejects(
            [
                "--sampler=vllm",
                "--maxtext_model_name=qwen3.5-35b-a3b",
                f"--vllm_config_json={json.dumps(payload)}",
            ],
            "managed by dedicated CLI flags or rollout contracts",
        )

    # Drift guard: every `AsyncEngineArgs` key produced by `_create_vllm_sampler`
    # under maximal CLI flags (except `additional_config`, which is deep-merged)
    # must stay covered by `_FORBIDDEN_ROLLOUT_KWARGS`.
    maximal_kwargs = _vllm_engine_kwargs([
        "--sampler=vllm",
        "--maxtext_model_name=qwen3.5-35b-a3b",
        "--maxtext_attention=vllm_rpa",
        "--prefuse_moe_weights=true",
        "--return_routed_experts=true",
        "--maxtext_float32_gate_logits=true",
        "--maxtext_float32_logits=true",
        "--vllm_hbm_utilization=0.9",
        "--vllm_max_num_seqs=16",
        "--vllm_max_num_batched_tokens=2048",
        "--vllm_max_model_len=8192",
        "--vllm_async_scheduling",
        "--enable_prefix_caching=true",
        "--use_lora",
        "--lora_rank=8",
        "--mesh_fsdp=1",
    ])
    self.assertLessEqual(
        set(maximal_kwargs) - {"additional_config"},
        run_rollout_node._FORBIDDEN_ROLLOUT_KWARGS,
    )
    self.assertLessEqual(
        {
            "model_name",
            "model_call_mode",
            "scan_layers",
            "attention",
            "prefuse_moe_weights",
            "return_routed_experts",
            "float32_gate_logits",
            "float32_logits",
        },
        run_rollout_node._FORBIDDEN_MAXTEXT_CONFIG_KEYS,
    )

  def test_main_rpc_stop_callback_drains_worker_cleanly_or_exits_nonzero_on_abort(
      self,
  ):
    context = mock.MagicMock()
    mock_tokenizer = mock.MagicMock(pad_token_id=0, eos_token="<eos>")
    mock_worker_service = mock.MagicMock(spec=rollout_worker.RolloutWorker)
    mock_worker_service.trajectory_store = None
    mock_sampler = mock.MagicMock()
    mock_sampler.stop = mock.AsyncMock()
    mock_worker_service.sampler = mock_sampler
    registered_callbacks = []
    mock_worker_service.set_on_stop.side_effect = registered_callbacks.append

    mock_server = mock.MagicMock(
        spec=remote_execution.GrpcRemoteExecutionServer
    )
    mock_server.start_serving_async = mock.AsyncMock()
    mock_server.stop_serving = mock.AsyncMock()

    with (
        mock.patch.object(run_rollout_node.importlib, "import_module"),
        mock.patch.object(otel_setup, "setup_metrics"),
        mock.patch.object(gcs_cache, "restore_jax_cache", return_value=False),
        mock.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            return_value=mock_tokenizer,
        ),
        mock.patch.object(
            run_rollout_node,
            "_create_vllm_worker",
            return_value=mock_worker_service,
        ),
        mock.patch.object(
            remote_execution,
            "GrpcRemoteExecutionServer",
            return_value=mock_server,
        ),
    ):

      shutdown_order: list[str] = []

      async def _stop_serving(grace: float = 0.5):
        self.assertEqual(grace, 5.0)
        shutdown_order.append("server.stop_serving")

      def _stop_worker():
        shutdown_order.append("worker_service.stop")

      async def _stop_sampler():
        shutdown_order.append("sampler.stop")

      mock_server.stop_serving = mock.AsyncMock(side_effect=_stop_serving)
      mock_worker_service.stop.side_effect = _stop_worker
      mock_sampler.stop = mock.AsyncMock(side_effect=_stop_sampler)

      def _register_and_trigger_clean_stop(*, metadata):
        del metadata
        self.assertTrue(callable(registered_callbacks[0]))
        registered_callbacks[0](False)

      context.ipc.discovery.register.side_effect = (
          _register_and_trigger_clean_stop
      )
      with mock.patch.dict(os.environ):
        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        run_rollout_node.main(
            [
                "--sampler=inprocess_vllm",
                "--port=20001",
                "--worker_id=rollout-0",
            ],
            context=context,
        )
        self.assertEqual(
            os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), "spawn"
        )
      self.assertIsNone(registered_callbacks[-1])
      self.assertEqual(
          shutdown_order,
          ["server.stop_serving", "worker_service.stop", "sampler.stop"],
      )
      mock_server.stop_serving.assert_awaited_once_with(grace=5.0)

      shutdown_order.clear()
      mock_sampler.stop.reset_mock()
      mock_server.stop_serving = mock.AsyncMock(
          side_effect=RuntimeError("gRPC stop_serving failed")
      )

      def _stop_worker_and_fail():
        shutdown_order.append("worker_service.stop")
        raise RuntimeError("worker_service.stop failed")

      mock_worker_service.stop = mock.MagicMock(
          side_effect=_stop_worker_and_fail
      )
      registered_callbacks.clear()

      def _register_and_trigger_abort_stop(*, metadata):
        del metadata
        self.assertTrue(callable(registered_callbacks[0]))
        registered_callbacks[0](True)

      context.ipc.discovery.register.side_effect = (
          _register_and_trigger_abort_stop
      )
      with self.assertRaises(SystemExit) as cm:
        run_rollout_node.main(
            [
                "--sampler=inprocess_vllm",
                "--port=20001",
                "--worker_id=rollout-0",
            ],
            context=context,
        )
      self.assertEqual(cm.exception.code, 1)
      self.assertIsNone(registered_callbacks[-1])
      mock_server.stop_serving.assert_awaited_once_with(grace=5.0)
      self.assertEqual(shutdown_order, ["worker_service.stop", "sampler.stop"])


if __name__ == "__main__":
  absltest.main()
