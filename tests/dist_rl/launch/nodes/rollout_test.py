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

"""Unit tests for the distributed RL rollout worker node."""

from collections.abc import Mapping
import contextlib
import json
import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import rollout
from tunix.dist_rl.rollout import inprocess_vllm_sampler
from tunix.dist_rl.rollout import vanilla_sampler
from tunix.dist_rl.rollout import vllm_sampler
from tunix.dist_rl.weight_sync import weight_sync as weight_sync_lib
from tunix.models import automodel
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.utils import maxtext_utils

_MAXTEXT_ROLLOUT_ARGV = (
    "rollout_model_config.model_source=maxtext",
    "rollout_model_config.model_name=qwen3-30b-a3b",
)
_ROUTED_EXPERTS_ARGV = (
    *_MAXTEXT_ROLLOUT_ARGV,
    "vllm_config.return_routed_experts=true",
)


def _hp(*argv: str) -> resolve_config.DistHyperParameters:
  return resolve_config.initialize(list(argv))


def _create_vllm_sampler_kwargs(
    *argv: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
  """Returns the `AsyncEngineArgs` and `VllmSampler` kwargs it builds."""
  arg_utils = types.ModuleType("vllm.engine.arg_utils")
  async_engine_args = mock.Mock()
  setattr(arg_utils, "AsyncEngineArgs", async_engine_args)
  hp = _hp("rollout_engine=vllm", *argv)
  config = rollout._rollout_config(hp, mock.Mock(), sampler_type="vllm")
  with mock.patch.dict(sys.modules, {"vllm.engine.arg_utils": arg_utils}):
    with mock.patch.object(vllm_sampler, "VllmSampler") as sampler_cls:
      rollout._create_vllm_sampler(hp, "rollout-0", mock.Mock(), config)
  return async_engine_args.call_args.kwargs, sampler_cls.call_args.kwargs


def _vllm_engine_kwargs(*argv: str) -> Mapping[str, Any]:
  """Returns the AsyncEngineArgs kwargs `_create_vllm_sampler` builds."""
  engine_kwargs, _ = _create_vllm_sampler_kwargs(*argv)
  return engine_kwargs


def _inprocess_vllm_config_kwargs(*argv: str) -> Mapping[str, Any]:
  """Returns the VllmConfig kwargs `_create_inprocess_vllm_sampler` builds."""
  vllm_sampler_module = mock.Mock()
  with (
      mock.patch.object(
          rollout, "_import_vllm_sampler", return_value=vllm_sampler_module
      ),
      mock.patch.object(node_utils, "create_mesh"),
      mock.patch.object(inprocess_vllm_sampler, "InprocessVllmSampler"),
  ):
    hp = _hp(*argv)
    config = rollout._rollout_config(
        hp, mock.Mock(), sampler_type="inprocess_vllm"
    )
    rollout._create_inprocess_vllm_sampler(hp, "rollout-0", mock.Mock(), config)
  return vllm_sampler_module.VllmConfig.call_args.kwargs


def _rollout_config(*argv: str):
  return rollout._rollout_config(
      _hp(*argv), mock.Mock(), sampler_type="inprocess_vllm"
  )


class ModelFamilyTest(absltest.TestCase):

  def test_auto_chat_parser_matches_the_most_specific_family(self):
    # Uses the real `CHAT_PARSERS`: "gemma-4"/"gemma4" must win over the
    # "gemma" substring.
    tokenizer = mock.MagicMock(bos_token="<bos>", eos_token="<eos>")
    for model_id, expected in (
        ("google/gemma-4-E2B-it", parser.Gemma4ChatTemplateParser),
        ("gemma4_e2b", parser.Gemma4ChatTemplateParser),
        ("google/gemma-2-2b-it", parser.GemmaChatTemplateParser),
        ("Qwen/Qwen3-1.7B", parser.QwenChatTemplateParser),
    ):
      with self.subTest(model_id=model_id):
        result = rollout._chat_parser_for(
            model_id, tokenizer, "auto", enable_thinking=False
        )
        self.assertIs(type(result), expected)
        if not isinstance(result, parser.BaseChatTemplateParser):
          self.fail(f"Expected a template parser, got {type(result)}.")
        self.assertFalse(result.enable_thinking)

  def test_raw_mode_uses_raw_text_parser(self):
    hp = _hp("chat_parser_config.type=raw")
    self.assertIsInstance(
        rollout._create_chat_parser(hp, mock.Mock()), parser.RawTextParser
    )

  def test_gemma4_mapping_includes_preprocessing_and_fused_projections(self):
    config = rollout._mapping_config_for("google/gemma-4-E2B-it")

    # Gemma4 fuses gate/up on the source state before transfer and maps
    # separate q/k/v projections.
    self.assertIsNotNone(config.preprocess_src_state)
    to_hf_mappings = config.to_hf_mappings
    if to_hf_mappings is None:
      self.fail("Gemma4 mapping has no to_hf_mappings.")
    self.assertIn("layers.*.mlp.gate_up_proj.kernel", to_hf_mappings)
    self.assertIn("layers.*.attn.q_einsum.w", to_hf_mappings)
    self.assertIn("layers.*.attn.k_einsum.w", to_hf_mappings)
    self.assertIn("layers.*.attn.v_einsum.w", to_hf_mappings)

  def test_qwen3_mapping_is_selected_for_qwen3(self):
    config = rollout._mapping_config_for("Qwen/Qwen3-8B")

    self.assertTrue(config.to_hf_mappings)
    self.assertIsNone(config.preprocess_src_state)

  def test_unsupported_family_raises(self):
    with self.assertRaisesRegex(ValueError, "supports Qwen3 and Gemma4"):
      rollout._mapping_config_for("meta-llama/Llama-3-8B")

  def test_gemma4_vllm_overrides_match_text_only_reference(self):
    overrides = rollout._vllm_hf_overrides(
        _hp("model_config.model_id=google/gemma-4-E2B-it")
    )

    self.assertEqual(overrides["architectures"], ["Gemma4ForCausalLM"])
    self.assertEqual(overrides["final_logit_softcapping"], 30.0)
    self.assertEqual(overrides["text_config"]["final_logit_softcapping"], 30.0)

  def test_qwen_does_not_get_gemma4_overrides(self):
    self.assertEmpty(
        rollout._vllm_hf_overrides(_hp("model_config.model_id=Qwen/Qwen3-8B"))
    )

  def test_maxtext_overrides_take_precedence_over_gemma4(self):
    overrides = rollout._vllm_hf_overrides(
        _hp(
            "rollout_model_config.model_id=google/gemma-4-E2B-it",
            "rollout_model_config.model_source=maxtext",
            "rollout_model_config.model_name=gemma3-4b",
        )
    )

    self.assertEqual(overrides, dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES))

  def test_maxtext_model_name_prefers_kwargs_over_model_name(self):
    self.assertEqual(rollout._maxtext_model_name(_hp()), "")
    self.assertEqual(
        rollout._maxtext_model_name(_hp(*_MAXTEXT_ROLLOUT_ARGV)),
        "qwen3-30b-a3b",
    )
    self.assertEqual(
        rollout._maxtext_model_name(
            _hp(
                *_MAXTEXT_ROLLOUT_ARGV,
                "rollout_model_config.kwargs.maxtext_model_name=qwen3-30b",
            )
        ),
        "qwen3-30b",
    )

  def test_registered_parser_is_gemma4_specific(self):
    self.assertIs(
        rollout.CHAT_PARSERS["gemma-4"],
        parser.Gemma4ChatTemplateParser,
    )


class RolloutConfigTest(absltest.TestCase):

  def test_builds_dist_rollout_config_from_config(self):
    config = rollout._rollout_config(
        _hp(
            "rollout_config.max_prompt_length=128",
            "rollout_config.total_generation_steps=64",
            "rollout_config.temperature=0.7",
            "rollout_config.top_p=0.95",
            "weight_sync_config.mode=raiden",
        ),
        mock.Mock(),
        sampler_type="vllm",
    )
    # No local weights in `model_download_path`, so vLLM serves `model_id`.
    self.assertEqual(config.rollout_vllm_model_version, "Qwen/Qwen3-1.7B")
    self.assertEqual(
        config.weight_sync_mode, weight_sync_lib.WeightSyncMode.RAIDEN
    )
    self.assertEqual(config.max_prompt_length, 128)
    self.assertEqual(config.max_tokens_to_generate, 64)
    self.assertEqual(config.temperature, 0.7)
    self.assertEqual(config.top_p, 0.95)
    self.assertTrue(config.return_logprobs)
    self.assertEqual(config.eos_tokens, [151645, 151643])
    self.assertIsNone(config.trajectory_store_config)

  def test_eos_token_strings_resolve_through_tokenizer(self):
    tokenizer = mock.Mock()
    tokenizer.encode.return_value = [7]
    hp = _hp("rollout_config.eos_tokens='<|im_end|>,3'")
    self.assertEqual(rollout._eos_token_ids(hp, tokenizer), [7, 3])
    tokenizer.encode.assert_called_once_with(
        "<|im_end|>", add_special_tokens=False
    )

    tokenizer.encode.return_value = [7, 8]
    with self.assertRaisesRegex(ValueError, "is not a single token"):
      rollout._eos_token_ids(hp, tokenizer)

  def test_empty_eos_tokens_defer_to_tokenizer(self):
    self.assertIsNone(
        rollout._eos_token_ids(_hp("rollout_config.eos_tokens="), mock.Mock())
    )

  def test_trajectory_store_config_reaches_rollout_config(self):
    self.assertIsNone(_rollout_config().trajectory_store_config)
    self.assertEqual(
        _rollout_config(
            "trajectory_store_config.enable=true",
            "trajectory_store_config.dir=/tmp/store",
            "cluster_config.run_id=run_1",
        ).trajectory_store_config,
        {"backend": "file", "root_dir": "/tmp/store", "run_id": "run_1"},
    )

  def test_in_flight_weight_updates_sets_partial_rollout(self):
    self.assertFalse(_rollout_config().partial_rollout)
    self.assertTrue(
        _rollout_config(
            "weight_sync_config.in_flight_weight_updates=true"
        ).partial_rollout
    )

  def test_routed_experts_settings_reach_rollout_config(self):
    defaults = _rollout_config()
    self.assertFalse(defaults.return_routed_experts)
    self.assertEqual(defaults.routed_experts_encode_mode, "auto")
    self.assertIsNone(defaults.routed_experts_num_experts)

    enabled = _rollout_config(
        "vllm_config.return_routed_experts=true",
        "vllm_config.routed_experts_encode_mode=uint10",
        "vllm_config.routed_experts_num_experts=256",
    )
    self.assertTrue(enabled.return_routed_experts)
    self.assertEqual(enabled.routed_experts_encode_mode, "uint10")
    self.assertEqual(enabled.routed_experts_num_experts, 256)

  def test_lora_config_requires_rank(self):
    with self.assertRaisesRegex(ValueError, "missing 'rank'"):
      rollout._validate_rollout_config(
          _hp("rollout_model_config.lora_config={'alpha': 64.0}")
      )
    self.assertEqual(
        rollout._lora_rank(
            _hp("rollout_model_config.lora_config={'rank': 16, 'alpha': 1.0}")
        ),
        16,
    )

  def test_episode_timeout_derives_drain_timeout_in_rollout_config(self):
    self.assertEqual(_rollout_config().drain_timeout_s, 660.0)
    self.assertEqual(
        _rollout_config(
            "agentic_grpo_config.episode_timeout=1800"
        ).drain_timeout_s,
        1860.0,
    )
    with self.assertRaisesRegex(
        ValueError, "episode_timeout_secs must be a positive finite number"
    ):
      rollout._validate_rollout_config(
          _hp("agentic_grpo_config.episode_timeout=0")
      )


class RolloutConfigPlumbingTest(absltest.TestCase):

  def test_top_k_reaches_rollout_config(self):
    self.assertEqual(_rollout_config("rollout_config.top_k=20").top_k, 20)

  def test_defaults_disable_top_k(self):
    config = _rollout_config()
    self.assertEqual(config.top_k, -1)
    self.assertTrue(config.return_logprobs)

  def test_vllm_model_version_only_for_vllm_samplers(self):
    hp = _hp()
    self.assertEqual(
        rollout._rollout_config(
            hp, mock.Mock(), sampler_type="vanilla"
        ).rollout_vllm_model_version,
        "",
    )
    self.assertEqual(
        rollout._rollout_config(
            hp, mock.Mock(), sampler_type="vllm"
        ).rollout_vllm_model_version,
        "Qwen/Qwen3-1.7B",
    )


class ValidateRolloutConfigTest(absltest.TestCase):

  def test_rejects_return_logprobs_false(self):
    with self.assertRaisesRegex(ValueError, "return_logprobs must be true"):
      rollout._validate_rollout_config(
          _hp("rollout_config.return_logprobs=false")
      )

  def test_rejects_lora_with_vanilla_sampler(self):
    with self.assertRaisesRegex(ValueError, "not supported with.*vanilla"):
      rollout._validate_rollout_config(
          _hp(
              "rollout_engine=vanilla",
              "rollout_model_config.lora_config={'rank': 8, 'alpha': 8.0}",
          )
      )

  def test_rejects_server_mode_false_with_inprocess_vllm(self):
    with self.assertRaisesRegex(ValueError, "server_mode: false"):
      rollout._validate_rollout_config(
          _hp("rollout_engine=inprocess_vllm", "vllm_config.server_mode=false")
      )

  def test_server_mode_false_allowed_with_vllm_engine(self):
    rollout._validate_rollout_config(
        _hp("rollout_engine=vllm", "vllm_config.server_mode=false")
    )


class CreateRolloutWorkerTest(absltest.TestCase):
  """`_create_rollout_worker` with only the sampler/model classes mocked."""

  def setUp(self):
    super().setUp()
    self.tokenizer = mock.MagicMock(bos_token="<bos>", eos_token="<eos>")
    self.mock_worker_cls = self.enter_context(
        mock.patch.object(rollout.rollout_worker, "RolloutWorker")
    )

  def _worker_kwargs(self) -> Mapping[str, Any]:
    self.mock_worker_cls.assert_called_once()
    return self.mock_worker_cls.call_args.kwargs

  def test_vanilla_loads_local_weights_and_shares_config(self):
    mesh = mock.MagicMock()
    with (
        mock.patch.object(
            node_utils, "ensure_local_safetensors", return_value="/models/q"
        ) as mock_ensure,
        mock.patch.object(node_utils, "create_mesh", return_value=mesh),
        mock.patch.object(
            automodel.AutoModel,
            "from_pretrained",
            return_value=(mock.sentinel.model, "/models/q"),
        ) as mock_from_pretrained,
        mock.patch.object(vanilla_sampler, "VanillaSampler") as sampler_cls,
    ):
      rollout._create_rollout_worker(
          _hp(
              "rollout_engine=vanilla",
              "rollout_model_config.dtype=bfloat16",
              "rollout_model_config.load_dtype=float32",
              "rollout_config.max_prompt_length=128",
              "rollout_config.total_generation_steps=64",
          ),
          "rollout-0",
          self.tokenizer,
      )

    # Cold start: the weights are fetched into `model_download_path` first.
    self.assertEqual(mock_ensure.call_args.args[1], "Qwen/Qwen3-1.7B")
    load_kwargs = mock_from_pretrained.call_args.kwargs
    self.assertEqual(load_kwargs["model_path"], "/models/q")
    self.assertEqual(load_kwargs["dtype"], "bfloat16")
    self.assertEqual(load_kwargs["load_dtype"], "float32")
    sampler_config = sampler_cls.call_args.kwargs["config"]
    self.assertEqual(sampler_config.kv_cache_size, 192)
    worker_kwargs = self._worker_kwargs()
    self.assertIs(worker_kwargs["config"], sampler_config)
    self.assertIs(worker_kwargs["sampler"], sampler_cls.return_value)
    # The base config's `chat_parser_config.type` is "raw".
    self.assertIsInstance(worker_kwargs["chat_parser"], parser.RawTextParser)

  def test_vllm_shares_config_with_sampler(self):
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    setattr(arg_utils, "AsyncEngineArgs", mock.Mock())
    with (
        mock.patch.dict(sys.modules, {"vllm.engine.arg_utils": arg_utils}),
        mock.patch.object(vllm_sampler, "VllmSampler") as sampler_cls,
    ):
      rollout._create_rollout_worker(
          _hp("rollout_engine=vllm"), "rollout-0", self.tokenizer
      )
    worker_kwargs = self._worker_kwargs()
    self.assertIs(worker_kwargs["sampler"], sampler_cls.return_value)
    self.assertIs(
        worker_kwargs["config"], sampler_cls.call_args.kwargs["config"]
    )
    self.assertEqual(
        worker_kwargs["config"].rollout_vllm_model_version, "Qwen/Qwen3-1.7B"
    )
    # The base config's `env_kwargs` / `agent_kwargs` are empty.
    self.assertEqual(worker_kwargs["env_config"], {})
    self.assertEqual(worker_kwargs["agent_config"], {})

  def test_worker_gets_env_and_agent_settings(self):
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    setattr(arg_utils, "AsyncEngineArgs", mock.Mock())
    with (
        mock.patch.dict(sys.modules, {"vllm.engine.arg_utils": arg_utils}),
        mock.patch.object(vllm_sampler, "VllmSampler"),
    ):
      rollout._create_rollout_worker(
          _hp(
              "rollout_engine=vllm",
              "env_class_path=frozenlake_env",
              "agent_class_path=frozenlake_agent",
              "env_kwargs={'map_size': 4, 'slippery': False}",
              "agent_kwargs.use_multistep_prompt=true",
          ),
          "rollout-0",
          self.tokenizer,
      )
    worker_kwargs = self._worker_kwargs()
    self.assertEqual(worker_kwargs["env_name"], "frozenlake_env")
    self.assertEqual(worker_kwargs["agent_name"], "frozenlake_agent")
    self.assertEqual(
        worker_kwargs["env_config"], {"map_size": 4, "slippery": False}
    )
    self.assertEqual(
        worker_kwargs["agent_config"], {"use_multistep_prompt": True}
    )

  def test_inprocess_vllm_builds_inprocess_sampler(self):
    with (
        mock.patch.object(
            rollout, "_import_vllm_sampler", return_value=mock.Mock()
        ),
        mock.patch.object(node_utils, "create_mesh"),
        mock.patch.object(
            inprocess_vllm_sampler, "InprocessVllmSampler"
        ) as sampler_cls,
    ):
      rollout._create_rollout_worker(
          _hp("rollout_engine=inprocess_vllm"), "rollout-0", self.tokenizer
      )
    worker_kwargs = self._worker_kwargs()
    self.assertIs(worker_kwargs["sampler"], sampler_cls.return_value)


class SamplerTest(absltest.TestCase):

  def test_vllm_sampler_engine_kwargs_follow_config(self):
    kwargs = _vllm_engine_kwargs(
        "rollout_model_config.mesh.shape=(2,4)",
        "vllm_config.max_num_seqs=64",
        "vllm_config.max_model_len=4096",
        "vllm_config.dtype=float32",
    )
    self.assertEqual(kwargs["tensor_parallel_size"], 4)
    self.assertEqual(kwargs["data_parallel_size"], 2)
    self.assertEqual(kwargs["max_num_seqs"], 64)
    self.assertNotIn("max_num_batched_tokens", kwargs)
    self.assertEqual(kwargs["max_model_len"], 4096)
    self.assertEqual(kwargs["dtype"], "float32")
    self.assertFalse(kwargs["enable_lora"])
    self.assertFalse(kwargs["enable_prefix_caching"])
    self.assertEqual(kwargs["model"], "Qwen/Qwen3-1.7B")

  def test_vllm_sampler_detokenize_follows_config(self):
    _, default = _create_vllm_sampler_kwargs()
    self.assertIs(default["detokenize"], False)
    _, enabled = _create_vllm_sampler_kwargs("vllm_config.detokenize=true")
    self.assertIs(enabled["detokenize"], True)

  def test_vllm_sampler_rejects_lora_with_data_parallelism(self):
    with self.assertRaisesRegex(ValueError, "LoRA is not supported"):
      _vllm_engine_kwargs(
          "rollout_model_config.mesh.shape=(2,1)",
          "vllm_config.tensor_parallel_size=1",
          "rollout_model_config.lora_config={'rank': 8, 'alpha': 8.0}",
      )

  def test_vllm_sampler_captures_routed_experts_only_when_asked(self):
    default = _vllm_engine_kwargs(*_MAXTEXT_ROLLOUT_ARGV)
    self.assertNotIn("enable_return_routed_experts", default)
    self.assertNotIn(
        "return_routed_experts", default["additional_config"]["maxtext_config"]
    )

    enabled = _vllm_engine_kwargs(*_ROUTED_EXPERTS_ARGV)
    self.assertIs(enabled["enable_return_routed_experts"], True)
    self.assertIs(
        enabled["additional_config"]["maxtext_config"]["return_routed_experts"],
        True,
    )

  def test_inprocess_vllm_sampler_captures_routed_experts(self):
    config_kwargs = _inprocess_vllm_config_kwargs(*_ROUTED_EXPERTS_ARGV)
    self.assertIs(config_kwargs["return_routed_experts"], True)
    self.assertIs(
        config_kwargs["additional_config"]["maxtext_config"][
            "return_routed_experts"
        ],
        True,
    )

  def test_inprocess_vllm_config_follows_config(self):
    config_kwargs = _inprocess_vllm_config_kwargs(
        "rollout_model_config.mesh.shape=(1,4)",
        "vllm_config.hbm_utilization=0.75",
        "weight_sync_config.in_flight_weight_updates=true",
    )
    self.assertEqual(config_kwargs["tensor_parallel_size"], 4)
    self.assertEqual(config_kwargs["data_parallel_size"], 1)
    self.assertEqual(config_kwargs["hbm_utilization"], 0.75)
    self.assertTrue(config_kwargs["server_mode"])
    self.assertTrue(config_kwargs["partial_rollout"])
    # Partial rollout needs prefix caching to resume frozen trajectories.
    self.assertTrue(config_kwargs["engine_kwargs"]["enable_prefix_caching"])
    self.assertIsNone(config_kwargs["lora_config"])
    self.assertIsNone(config_kwargs["additional_config"])

  def test_inprocess_vllm_server_mode_defaults_on(self):
    self.assertTrue(_inprocess_vllm_config_kwargs()["server_mode"])

  def test_vllm_samplers_plumb_float32_logit_flags_when_set(self):
    default_mt_cfg = _vllm_engine_kwargs(*_MAXTEXT_ROLLOUT_ARGV)[
        "additional_config"
    ]["maxtext_config"]
    self.assertNotIn("float32_gate_logits", default_mt_cfg)
    self.assertNotIn("float32_logits", default_mt_cfg)

    argv = (
        *_MAXTEXT_ROLLOUT_ARGV,
        "rollout_model_config.kwargs.float32_gate_logits=true",
        "rollout_model_config.kwargs.float32_logits=false",
    )
    vllm_mt_cfg = _vllm_engine_kwargs(*argv)["additional_config"][
        "maxtext_config"
    ]
    self.assertIs(vllm_mt_cfg["float32_gate_logits"], True)
    self.assertIs(vllm_mt_cfg["float32_logits"], False)

    inproc_mt_cfg = _inprocess_vllm_config_kwargs(*argv)["additional_config"][
        "maxtext_config"
    ]
    self.assertIs(inproc_mt_cfg["float32_gate_logits"], True)
    self.assertIs(inproc_mt_cfg["float32_logits"], False)

  def test_vllm_hbm_utilization_defaults_per_sampler(self):
    self.assertNotIn(
        "gpu_memory_utilization", _vllm_engine_kwargs("rollout_engine=vllm")
    )
    self.assertEqual(
        _vllm_engine_kwargs(
            "rollout_engine=vllm", "vllm_config.hbm_utilization=0.8"
        )["gpu_memory_utilization"],
        0.8,
    )

    self.assertEqual(
        _inprocess_vllm_config_kwargs("rollout_engine=inprocess_vllm")[
            "hbm_utilization"
        ],
        0.5,
    )
    self.assertEqual(
        _inprocess_vllm_config_kwargs(
            "rollout_engine=inprocess_vllm", "vllm_config.hbm_utilization=0.8"
        )["hbm_utilization"],
        0.8,
    )


class EngineKwargsTest(absltest.TestCase):

  def test_engine_kwargs_merges_maxtext_recipe_config(self):
    recipe_engine_kwargs = {
        "enable_expert_parallel": True,
        "kv_cache_dtype": "bfloat16",
        "block_size": 256,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "additional_config": {
            "sharding": {"sharding_strategy": {"expert_parallelism": 4}},
            "custom_mamba_cache_multiplier": 16,
            "maxtext_config": {
                "use_multimodal": False,
                "allow_split_physical_axes": False,
            },
        },
    }
    kwargs = _vllm_engine_kwargs(
        "rollout_engine=vllm",
        *_ROUTED_EXPERTS_ARGV,
        "rollout_model_config.kwargs.attention=vllm_rpa",
        "vllm_config.max_model_len=65536",
        "vllm_config.max_num_seqs=16",
        "vllm_config.hbm_utilization=0.9",
        f"vllm_config.engine_kwargs={json.dumps(recipe_engine_kwargs)}",
    )
    self.assertEqual(kwargs["max_model_len"], 65536)
    self.assertEqual(kwargs["max_num_seqs"], 16)
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.9)
    self.assertIs(kwargs["enable_return_routed_experts"], True)
    self.assertIs(kwargs["enable_expert_parallel"], True)
    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
    self.assertEqual(kwargs["block_size"], 256)
    self.assertEqual(kwargs["limit_mm_per_prompt"], {"image": 0, "video": 0})
    additional = kwargs["additional_config"]
    self.assertEqual(additional["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(
        additional["sharding"],
        {"sharding_strategy": {"expert_parallelism": 4}},
    )
    mt_cfg = additional["maxtext_config"]
    # Node-built MaxText keys survive the deep merge.
    self.assertEqual(mt_cfg["model_name"], "qwen3-30b-a3b")
    self.assertIs(mt_cfg["scan_layers"], False)
    self.assertEqual(mt_cfg["attention"], "vllm_rpa")
    self.assertIs(mt_cfg["return_routed_experts"], True)
    # User keys are added, and non-forbidden node-built keys can be overridden.
    self.assertIs(mt_cfg["use_multimodal"], False)
    self.assertIs(mt_cfg["allow_split_physical_axes"], False)

  def test_engine_kwargs_without_maxtext_and_override_precedence(self):
    kwargs = _vllm_engine_kwargs(
        "rollout_engine=vllm",
        "vllm_config.engine_kwargs.enable_expert_parallel=true",
        "vllm_config.engine_kwargs.additional_config.sharding"
        ".tensor_parallelism=4",
    )
    self.assertIs(kwargs["enable_expert_parallel"], True)
    self.assertEqual(
        kwargs["additional_config"], {"sharding": {"tensor_parallelism": 4}}
    )
    self.assertNotIn("hf_overrides", kwargs)

    self.assertNotIn(
        "additional_config", _vllm_engine_kwargs("rollout_engine=vllm")
    )
    self.assertNotIn(
        "additional_config",
        _vllm_engine_kwargs(
            "rollout_engine=vllm",
            "vllm_config.engine_kwargs.additional_config={}",
        ),
    )

  def test_engine_kwargs_rejected_for_non_vllm_rollout_engine(self):
    for rollout_engine in ("inprocess_vllm", "vanilla"):
      with self.subTest(rollout_engine=rollout_engine):
        with self.assertRaisesRegex(
            ValueError,
            "vllm_config.engine_kwargs is only supported with rollout_engine:"
            " vllm",
        ):
          rollout._validate_rollout_config(
              _hp(
                  f"rollout_engine={rollout_engine}",
                  "vllm_config.engine_kwargs.enable_expert_parallel=true",
              )
          )

  def test_engine_kwargs_rejects_forbidden_keys_and_malformed_inputs(self):
    with self.assertRaisesRegex(
        ValueError, r"engine_kwargs\.additional_config must be a mapping"
    ):
      rollout._vllm_engine_kwargs(
          _hp(
              "rollout_engine=vllm",
              "vllm_config.engine_kwargs.additional_config=1",
          )
      )
    with self.assertRaisesRegex(
        ValueError,
        r"engine_kwargs\.additional_config\.maxtext_config must be a mapping",
    ):
      rollout._vllm_engine_kwargs(
          _hp(
              "rollout_engine=vllm",
              *_MAXTEXT_ROLLOUT_ARGV,
              "vllm_config.engine_kwargs.additional_config.maxtext_config=1",
          )
      )
    with self.assertRaisesRegex(
        ValueError,
        r"engine_kwargs\.additional_config\.maxtext_config requires a MaxText"
        " rollout model",
    ):
      rollout._vllm_engine_kwargs(
          _hp(
              "rollout_engine=vllm",
              "vllm_config.engine_kwargs.additional_config.maxtext_config"
              ".use_multimodal=false",
          )
      )

    for key in rollout._FORBIDDEN_ENGINE_KWARGS_KEYS:
      with self.subTest(top_level_key=key):
        with self.assertRaisesRegex(
            ValueError,
            r"vllm_config\.engine_kwargs contains keys managed by other"
            rf" config fields or rollout invariants: \['{key}'\]",
        ):
          rollout._vllm_engine_kwargs(
              _hp("rollout_engine=vllm", f"vllm_config.engine_kwargs.{key}=1")
          )

    for mt_key in rollout._FORBIDDEN_MAXTEXT_CONFIG_KEYS:
      with self.subTest(maxtext_key=mt_key):
        with self.assertRaisesRegex(
            ValueError,
            r"maxtext_config contains keys managed by other config fields or"
            rf" rollout invariants: \['{mt_key}'\]",
        ):
          rollout._vllm_engine_kwargs(
              _hp(
                  "rollout_engine=vllm",
                  *_MAXTEXT_ROLLOUT_ARGV,
                  "vllm_config.engine_kwargs.additional_config.maxtext_config"
                  f".{mt_key}=1",
              )
          )

  def test_forbidden_keys_cover_node_built_engine_kwargs(self):
    # Drift guard: every `AsyncEngineArgs` key `_create_vllm_sampler` builds
    # from other config fields (except `additional_config`, which is
    # deep-merged) must stay in `_FORBIDDEN_ENGINE_KWARGS_KEYS`.
    maximal_argv = (
        "rollout_engine=vllm",
        *_ROUTED_EXPERTS_ARGV,
        "rollout_model_config.kwargs.attention=vllm_rpa",
        "rollout_model_config.kwargs.float32_gate_logits=true",
        "rollout_model_config.kwargs.float32_logits=true",
        "weight_sync_config.prefuse_moe_weights=true",
        "rollout_model_config.lora_config={'rank': 8, 'alpha': 8.0}",
        "vllm_config.hbm_utilization=0.9",
        "vllm_config.max_num_seqs=16",
        "vllm_config.max_num_batched_tokens=2048",
        "vllm_config.max_model_len=8192",
    )
    kwargs = _vllm_engine_kwargs(*maximal_argv)
    self.assertLessEqual(
        set(kwargs) - {"additional_config"},
        rollout._FORBIDDEN_ENGINE_KWARGS_KEYS,
    )
    mt_built = rollout._maxtext_additional_config(_hp(*maximal_argv))
    if mt_built is None:
      self.fail("MaxText rollout built no additional_config.")
    self.assertLessEqual(
        set(mt_built["maxtext_config"])
        - {
            "enable_dp_attention",
            "allow_split_physical_axes",
            "log_config",
            "weight_dtype",
        },
        rollout._FORBIDDEN_MAXTEXT_CONFIG_KEYS,
    )


class RolloutMainTest(absltest.TestCase):

  def test_cache_params_follow_config(self):
    params = rollout._build_rollout_cache_params(
        _hp(
            "rollout_engine=vllm",
            "rollout_model_config.mesh.shape=(2,4)",
            "vllm_config.tensor_parallel_size=4",
            "rollout_config.max_prompt_length=256",
            "rollout_config.total_generation_steps=128",
        )
    )
    self.assertEqual(params.sampler, "vllm")
    self.assertEqual(params.model_name, "Qwen3-1.7B")
    self.assertEqual(params.mesh_tp, 4)
    self.assertEqual(params.mesh_fsdp, 2)
    self.assertEqual(params.max_prompt_length, 256)
    self.assertEqual(params.max_response_length, 128)
    self.assertFalse(params.use_lora)

  def test_main_restores_jax_cache_before_jax_init(self):
    events: list[str] = []
    context = mock.Mock()

    def _fake_jax_init() -> None:
      events.append("jax_init")
      raise RuntimeError("stop at jax init")

    context.jax.initialize.side_effect = _fake_jax_init

    def _fake_restore(
        params: gcs_cache.RolloutCacheKeyParams | None = None,
    ) -> bool:
      if params is None:
        raise AssertionError("restore_jax_cache needs cache params.")
      events.append(f"restore:{params.sampler}")
      return True

    with contextlib.ExitStack() as stack:
      mock_import = stack.enter_context(
          mock.patch.object(rollout.importlib, "import_module")
      )
      mock_restore = stack.enter_context(
          mock.patch.object(
              gcs_cache,
              "restore_jax_cache",
              side_effect=_fake_restore,
          )
      )
      with self.assertRaisesRegex(RuntimeError, "stop at jax init"):
        rollout.main(
            [
                "rollout_engine=vanilla",
                "cluster_config.registry_module=my_task.registry",
            ],
            context=context,
        )

    mock_import.assert_called_once_with("my_task.registry")
    mock_restore.assert_called_once()
    self.assertEqual(mock_restore.call_args.kwargs["params"].sampler, "vanilla")
    self.assertEqual(events, ["restore:vanilla", "jax_init"])

  def test_main_raises_without_discovery_context(self):
    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      rollout.main([], context=None)


if __name__ == "__main__":
  absltest.main()
