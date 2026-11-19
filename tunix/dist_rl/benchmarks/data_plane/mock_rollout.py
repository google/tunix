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

"""Mock RolloutWorker builder and node entrypoint for data_plane_benchmark."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import importlib
import logging
from typing import cast

import jax
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.benchmarks.data_plane import data_plane_benchmark
from tunix.dist_rl.common import test_utils
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import rollout_worker
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
from tunix.rl.agentic.parser.chat_template_parser import (
    parser as chat_parser_lib,
)

LOCAL_ROLLOUT_WORKER_ID = "rollout-0"
_ENV_OBS_RESERVE_PER_TURN = 8
_CHAT_PARSERS: dict[str, type[chat_parser_lib.BaseChatTemplateParser]] = {
    "gemma-4": chat_parser_lib.Gemma4ChatTemplateParser,
    "gemma4": chat_parser_lib.Gemma4ChatTemplateParser,
    "qwen": chat_parser_lib.QwenChatTemplateParser,
    "llama": chat_parser_lib.LlamaChatTemplateParser,
    "gemma": chat_parser_lib.GemmaChatTemplateParser,
}


def _resolve_eos_token_ids(
    eos_raw: object, tokenizer: node_utils.HfTokenizer
) -> list[int] | None:
  """Resolves `rollout_config.eos_tokens` into token IDs, or `None`."""
  if eos_raw is None:
    return None
  items = (
      eos_raw if isinstance(eos_raw, (list, tuple)) else str(eos_raw).split(",")
  )
  entries = [str(item).strip() for item in items if str(item).strip()]
  if not entries:
    return None
  resolved: list[int] = []
  for entry in entries:
    if entry.lstrip("-").isdigit():
      resolved.append(int(entry))
      continue
    encoded = tokenizer.encode(entry, add_special_tokens=False)
    if len(encoded) != 1:
      raise ValueError(
          f"rollout_config.eos_tokens entry {entry!r} must map to 1 token,"
          f" got {encoded}."
      )
    resolved.append(encoded[0])
  return resolved


def _build_chat_parser(
    hp: resolve_config.DistHyperParameters, tokenizer: node_utils.HfTokenizer
) -> chat_parser_lib.BaseChatTemplateParser | chat_parser_lib.RawTextParser:
  """Constructs the chat parser without importing `launch/nodes/rollout.py`."""
  chat_cfg = hp.config_mapping("chat_parser_config")
  if str(chat_cfg["type"]) == "raw":
    return chat_parser_lib.RawTextParser(tokenizer)
  rollout_model = hp.config_mapping("rollout_model_config")
  family = str(
      rollout_model["model_name"] or rollout_model["model_id"]
  ).lower()
  parser_cls = next(
      (cls for key, cls in _CHAT_PARSERS.items() if key in family),
      chat_parser_lib.DefaultChatTemplateParser,
  )
  return parser_cls(
      tokenizer, enable_thinking=bool(chat_cfg["enable_thinking"])
  )


def _resolve_response_tokens_per_turn(
    model_kwargs: Mapping[str, object], max_resp: int, *, max_turns: int = 1
) -> int:
  """Resolves per-turn mock generation length.

  Defaults to `max_resp - 1` for single-turn runs (`max_turns == 1`) so
  `env.step()` executes instead of clipping at `MAX_CONTEXT_LIMIT_REACHED`, and
  divides the remaining budget across `max_turns` (reserving space for
  intermediate environment observations) when `max_turns >= 2` and
  `mock_response_tokens_per_turn` is unset.

  Args:
    model_kwargs: `rollout_model_config.kwargs` mapping.
    max_resp: `rollout_config.total_generation_steps` response budget.
    max_turns: `agentic_grpo_config.max_turns` turn limit.

  Returns:
    Number of tokens `MockBaseSamplerImpl` emits per turn.
  """
  if "mock_response_tokens_per_turn" in model_kwargs:
    return data_plane_benchmark.kwarg_int(
        model_kwargs, "mock_response_tokens_per_turn", max_resp
    )
  turns = max(1, int(max_turns))
  reserve = 1 + (turns - 1) * _ENV_OBS_RESERVE_PER_TURN
  per_turn = (max_resp - reserve) // turns
  if per_turn < 1:
    raise ValueError(
        "rollout_config.total_generation_steps must leave >= 1 token/turn"
        f" across max_turns={turns} (need >= {turns + reserve}), got"
        f" {max_resp}."
    )
  return per_turn


def build_mock_rollout_worker(
    hp: resolve_config.DistHyperParameters,
    worker_id_str: str,
    tokenizer: node_utils.HfTokenizer,
) -> rollout_worker.RolloutWorker:
  """Constructs a `RolloutWorker` backed by `MockBaseSamplerImpl`."""
  rollout_cfg = hp.config_mapping("rollout_config")
  if not rollout_cfg["return_logprobs"]:
    raise ValueError("rollout_config.return_logprobs must be true.")
  grpo_cfg = hp.config_mapping("agentic_grpo_config")
  ws_cfg = hp.config_mapping("weight_sync_config")
  vllm_cfg = hp.config_mapping("vllm_config")
  max_turns = max(1, int(grpo_cfg["max_turns"]))
  episode_timeout_s = timeouts.validate_positive_finite(
      grpo_cfg["episode_timeout"], "agentic_grpo_config.episode_timeout"
  )
  model_kwargs = cast(
      Mapping[str, object], hp.config_mapping("rollout_model_config")["kwargs"]
  )
  max_prompt_len = int(rollout_cfg["max_prompt_length"])
  max_gen_steps = int(rollout_cfg["total_generation_steps"])
  raw_num_experts = vllm_cfg["routed_experts_num_experts"]
  worker_rollout_cfg = rollout_worker.RolloutConfig(
      weight_sync_mode=common_datatypes.WeightSyncMode(str(ws_cfg["mode"])),
      max_prompt_length=max_prompt_len,
      max_tokens_to_generate=max_gen_steps,
      kv_cache_size=max_prompt_len + max_gen_steps,
      temperature=float(rollout_cfg["temperature"]),
      top_p=float(rollout_cfg["top_p"]),
      top_k=int(rollout_cfg["top_k"]),
      return_logprobs=True,
      eos_tokens=_resolve_eos_token_ids(rollout_cfg["eos_tokens"], tokenizer),
      partial_rollout=bool(ws_cfg["in_flight_weight_updates"]),
      drain_timeout_s=timeouts.drain_timeout_for(episode_timeout_s),
      trajectory_store_config=node_utils.trajectory_store_config(hp),
      return_routed_experts=bool(vllm_cfg["return_routed_experts"]),
      routed_experts_encode_mode=cast(
          common_datatypes.RoutedExpertsEncodeMode,
          str(vllm_cfg["routed_experts_encode_mode"]),
      ),
      routed_experts_num_experts=(
          None if raw_num_experts is None else int(raw_num_experts)
      ),
  )
  tok_adapter = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  shape_spec, num_experts, _ = data_plane_benchmark.resolve_routed_experts_spec(
      vllm_cfg, model_kwargs
  )
  tokens_per_turn = _resolve_response_tokens_per_turn(
      model_kwargs,
      worker_rollout_cfg.max_tokens_to_generate,
      max_turns=max_turns,
  )
  decode_sleep_s = data_plane_benchmark.kwarg_float(
      model_kwargs, "mock_decode_sleep_s", 0.05
  )
  eos_id = (
      worker_rollout_cfg.eos_tokens[0]
      if worker_rollout_cfg.eos_tokens
      else None
  )

  sampler = test_utils.MockBaseSamplerImpl(
      sampler_name=worker_id_str,
      default_delay=decode_sleep_s,
      server_id=worker_id_str,
      config=worker_rollout_cfg,
      response_tokens_per_turn=tokens_per_turn,
      min_turns=max_turns - 1,
      routed_experts_shape_spec=shape_spec,
      num_experts=num_experts,
      eos_token_id=eos_id,
      tokenizer=tok_adapter,
  )
  env_cfg = {"max_steps": max_turns, **hp.config_mapping("env_kwargs")}
  return rollout_worker.RolloutWorker(
      worker_id=worker_id_str,
      config=worker_rollout_cfg,
      sampler=sampler,
      tokenizer=tok_adapter,
      chat_parser=_build_chat_parser(hp, tokenizer),
      env_name=str(hp.config["env_class_path"]),
      agent_name=str(hp.config["agent_class_path"]),
      env_config=env_cfg,
      agent_config=hp.config_mapping("agent_kwargs"),
      max_concurrency=int(grpo_cfg["max_concurrency"]),
  )


def rollout_main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a mock `RolloutWorker` over gRPC + Bulk TCP until stopped."""
  jax.config.update("jax_platforms", "cpu")
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [MockRolloutNode] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  node_utils.apply_library_env(hp, "rollout")
  worker_id_str = node_utils.worker_id(hp, "rollout", LOCAL_ROLLOUT_WORKER_ID)
  port = node_utils.service_port(hp, "rollout")
  node_utils.setup_otel_metrics(hp, "rollout", worker_id_str)
  node_utils.ensure_repo_root_on_path()
  registry_mod = str(hp.config_mapping("cluster_config")["registry_module"])
  if registry_mod:
    importlib.import_module(registry_mod)
  config_sha256 = node_utils.log_resolved_config(hp, "rollout")
  tokenizer = node_utils.load_hf_tokenizer(hp)

  async def _serve() -> None:
    worker_service = build_mock_rollout_worker(hp, worker_id_str, tokenizer)
    rss_mib, hwm_mib = data_plane_benchmark.read_process_memory_mib()
    logging.info(
        "[DataPlaneBench][Rollout %s] Initialized VmRSS=%.1f MiB"
        " VmHWM=%.1f MiB",
        worker_id_str,
        rss_mib,
        hwm_mib,
    )

    async def _drain() -> None:
      end_rss_mib, end_hwm_mib = data_plane_benchmark.read_process_memory_mib()
      logging.info(
          "[DataPlaneBench][Rollout %s] Draining VmRSS=%.1f MiB VmHWM=%.1f MiB",
          worker_id_str,
          end_rss_mib,
          end_hwm_mib,
      )
      try:
        worker_service.stop()
      finally:
        await worker_service.sampler.stop()

    await node_utils.serve_worker(
        worker_service,
        process_context=process_context,
        role="rollout",
        worker_id_str=worker_id_str,
        port=port,
        config_sha256=config_sha256,
        extra_metadata={
            "trajectory_store_config": (
                worker_service.trajectory_store.to_config()
                if worker_service.trajectory_store is not None
                else None
            ),
        },
        drain_fn=_drain,
    )

  asyncio.run(_serve())
