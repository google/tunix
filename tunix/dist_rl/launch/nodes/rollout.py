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

"""Rollout worker node for distributed RL.

Launched as
`python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=...
--process_main=tunix.dist_rl.launch.nodes.rollout.main <config overrides>`.
The overrides are resolved once by `resolve_config.initialize`, and the sampler,
rollout config and worker are built directly from that validated config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import dataclasses
import importlib
import logging
import os
import sys
import types
from typing import Any, cast

from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.weight_sync import raiden_preload
from tunix.dist_rl.weight_sync import weight_sync as weight_sync_lib
from tunix.dist_rl.worker import rollout_worker
from tunix.generate import mappings as mappings_lib
from tunix.models import automodel
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.utils import maxtext_utils

# Import Raiden before any other libraries to ensure correct JAX compilation.
raiden_preload.import_raiden()

LOCAL_WORKER_ID = "rollout-0"

# This must be set before the first vLLM import. Keep vLLM imports lazy so the
# rollout process can start with non-vLLM samplers in environments where vLLM is
# not installed.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_ROLLOUT_KEY = "rollout_model_config"

CHAT_PARSERS: dict[str, type[chat_parser_lib.BaseChatTemplateParser]] = {
    "gemma-4": chat_parser_lib.Gemma4ChatTemplateParser,
    "gemma4": chat_parser_lib.Gemma4ChatTemplateParser,
    "qwen": chat_parser_lib.QwenChatTemplateParser,
    "llama": chat_parser_lib.LlamaChatTemplateParser,
    "gemma": chat_parser_lib.GemmaChatTemplateParser,
}

ChatParser = (
    chat_parser_lib.BaseChatTemplateParser | chat_parser_lib.RawTextParser
)
Tokenizer = node_utils.HfTokenizer

# `vllm_config.hbm_utilization: null` uses this for `inprocess_vllm`; the
# `vllm` sampler leaves `gpu_memory_utilization` to vLLM's own default.
_DEFAULT_INPROCESS_VLLM_HBM_UTILIZATION = 0.5

# Keep in sync with every `AsyncEngineArgs` key set from other config fields or
# rollout invariants in this file's `_create_vllm_sampler` (`hbm_utilization` is
# an alias guard for `gpu_memory_utilization`).
_FORBIDDEN_ENGINE_KWARGS_KEYS = frozenset({
    "model",
    "tokenizer",
    "tensor_parallel_size",
    "data_parallel_size",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "async_scheduling",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "hbm_utilization",
    "enable_return_routed_experts",
    "hf_overrides",
    "dtype",
    "enable_lora",
    "max_lora_rank",
    "max_loras",
    "max_logprobs",
    "logprobs_mode",
    "trust_remote_code",
})

# Keep in sync with config-owned keys and rollout invariants in
# `maxtext_utils.build_vllm_maxtext_additional_config`.
_FORBIDDEN_MAXTEXT_CONFIG_KEYS = frozenset({
    "model_name",
    "model_call_mode",
    "scan_layers",
    "attention",
    "prefuse_moe_weights",
    "return_routed_experts",
    "float32_gate_logits",
    "float32_logits",
})


def _import_vllm_sampler() -> types.ModuleType:
  logging.info(
      "Importing tunix.generate.vllm_sampler before rollout adapters..."
  )
  vllm_sampler = importlib.import_module("tunix.generate.vllm_sampler")
  logging.info("Finished importing tunix.generate.vllm_sampler.")
  return vllm_sampler


def _is_gemma4_model(model_name: str) -> bool:
  """Returns whether `model_name` names a Gemma4 variant.

  Model construction goes through `automodel`, but the rollout node still needs
  the family up front to pick the vLLM weight mapping and `hf_overrides`, before
  any model or tokenizer exists.

  Args:
    model_name: Model id or demo selector, e.g. "gemma-4-e2b" or "gemma4-e2b".

  Returns:
    True when the name refers to a Gemma4 model.
  """
  normalized = model_name.lower().replace("_", "-")
  return "gemma-4" in normalized or "gemma4" in normalized


def _rollout_model_family_name(hp: resolve_config.DistHyperParameters) -> str:
  """Returns the name used to detect the rollout model family.

  `rollout_model_config.model_id` (e.g. "Qwen/Qwen3-8B"), or `model_name` when
  the id is empty (e.g. MaxText checkpoints addressed by name only).

  Args:
    hp: Validated distributed RL configuration.
  """
  rollout_model = hp.config_mapping(_ROLLOUT_KEY)
  return str(rollout_model["model_id"] or rollout_model["model_name"])


def _maxtext_model_name(hp: resolve_config.DistHyperParameters) -> str:
  """Returns the MaxText model name, or "" when the rollout is not MaxText.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    `rollout_model_config.kwargs.maxtext_model_name`, falling back to
    `rollout_model_config.model_name`, when `model_source == "maxtext"`.
  """
  rollout_model = hp.config_mapping(_ROLLOUT_KEY)
  if rollout_model["model_source"] != "maxtext":
    return ""
  kwargs = rollout_model["kwargs"]
  if "maxtext_model_name" in kwargs:
    return str(kwargs["maxtext_model_name"])
  return str(rollout_model["model_name"])


def _episode_timeout_secs(hp: resolve_config.DistHyperParameters) -> float:
  """Returns `agentic_grpo_config.episode_timeout`, validated positive finite."""
  return timeouts.validate_positive_finite(
      float(hp.config_mapping("agentic_grpo_config")["episode_timeout"]),
      "episode_timeout_secs",
  )


def _tensor_parallel_size(hp: resolve_config.DistHyperParameters) -> int:
  """Returns the rollout TP degree, `rollout_model_config.mesh` tp.

  `vllm_config.tensor_parallel_size`, when set, is validated equal to it by
  `DistHyperParameters`.

  Args:
    hp: Validated distributed RL configuration.
  """
  return hp.mesh_axes(_ROLLOUT_KEY)["tp"]


def _data_parallel_size(hp: resolve_config.DistHyperParameters) -> int:
  """Returns the rollout DP degree, `rollout_model_config.mesh` fsdp."""
  return hp.mesh_axes(_ROLLOUT_KEY)["fsdp"]


def _lora_rank(hp: resolve_config.DistHyperParameters) -> int | None:
  """Returns `rollout_model_config.lora_config.rank`, or None without LoRA.

  Args:
    hp: Validated distributed RL configuration.

  Raises:
    ValueError: If `lora_config` is set without `rank`.
  """
  lora_cfg = hp.config_mapping(_ROLLOUT_KEY)["lora_config"]
  if not lora_cfg:
    return None
  if "rank" not in lora_cfg:
    raise ValueError("rollout_model_config.lora_config is missing 'rank'.")
  return int(lora_cfg["rank"])


def _vllm_engine_kwargs(
    hp: resolve_config.DistHyperParameters,
) -> dict[str, Any]:
  """Returns validated `vllm_config.engine_kwargs` for `AsyncEngineArgs`.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    A copy of the validated overrides (empty when unset).

  Raises:
    ValueError: If the overrides are set without `rollout_engine: vllm`, are not
      a mapping, contain keys owned by other config fields or rollout
      invariants, or carry a malformed `additional_config` /
      `additional_config.maxtext_config` (the latter also needs a MaxText
      rollout model).
  """
  engine_kwargs = hp.config_mapping("vllm_config")["engine_kwargs"]
  if not isinstance(engine_kwargs, Mapping):
    raise ValueError(
        "vllm_config.engine_kwargs must be a mapping, got"
        f" {type(engine_kwargs).__name__}."
    )
  if not engine_kwargs:
    return {}
  rollout_engine = hp.config["rollout_engine"]
  if rollout_engine != "vllm":
    raise ValueError(
        "vllm_config.engine_kwargs is only supported with rollout_engine:"
        f" vllm; got rollout_engine: {rollout_engine!r}."
    )
  forbidden = sorted(set(engine_kwargs) & _FORBIDDEN_ENGINE_KWARGS_KEYS)
  if forbidden:
    raise ValueError(
        "vllm_config.engine_kwargs contains keys managed by other config"
        f" fields or rollout invariants: {forbidden}"
    )
  if "additional_config" in engine_kwargs:
    additional = engine_kwargs["additional_config"]
    if not isinstance(additional, Mapping):
      raise ValueError(
          "vllm_config.engine_kwargs.additional_config must be a mapping."
      )
    if "maxtext_config" in additional:
      maxtext_cfg = additional["maxtext_config"]
      if not isinstance(maxtext_cfg, Mapping):
        raise ValueError(
            "vllm_config.engine_kwargs.additional_config.maxtext_config must"
            " be a mapping."
        )
      if not _maxtext_model_name(hp):
        raise ValueError(
            "vllm_config.engine_kwargs.additional_config.maxtext_config"
            " requires a MaxText rollout model"
            " (rollout_model_config.model_source: maxtext)."
        )
      forbidden_mt = sorted(set(maxtext_cfg) & _FORBIDDEN_MAXTEXT_CONFIG_KEYS)
      if forbidden_mt:
        raise ValueError(
            "vllm_config.engine_kwargs.additional_config.maxtext_config"
            " contains keys managed by other config fields or rollout"
            f" invariants: {forbidden_mt}"
        )
  return dict(engine_kwargs)


def _merge_engine_kwargs(
    engine_kwargs: dict[str, Any], overrides: Mapping[str, Any]
) -> None:
  """Merges validated `vllm_config.engine_kwargs` into `engine_kwargs`.

  Top-level keys are set directly. `additional_config` keys are merged onto the
  node-built `additional_config`, and `additional_config.maxtext_config` keys
  onto the node-built MaxText config.

  Args:
    engine_kwargs: `AsyncEngineArgs` kwargs built by the node; updated in place.
    overrides: Output of `_vllm_engine_kwargs`.
  """
  top_level = dict(overrides)
  if "additional_config" in top_level:
    extra_additional = dict(top_level.pop("additional_config"))
    additional = (
        dict(engine_kwargs["additional_config"])
        if "additional_config" in engine_kwargs
        else {}
    )
    extra_maxtext = (
        extra_additional.pop("maxtext_config")
        if "maxtext_config" in extra_additional
        else None
    )
    additional.update(extra_additional)
    if extra_maxtext is not None:
      # `_vllm_engine_kwargs` only allows `maxtext_config` for MaxText rollout
      # models, whose node-built `additional_config` carries `maxtext_config`.
      additional["maxtext_config"] = {
          **additional["maxtext_config"],
          **extra_maxtext,
      }
    if additional:
      engine_kwargs["additional_config"] = additional
  engine_kwargs.update(top_level)


def _validate_rollout_config(hp: resolve_config.DistHyperParameters) -> None:
  """Validates rollout-only settings before any heavy import or JAX init.

  Args:
    hp: Validated distributed RL configuration.

  Raises:
    ValueError: On a non-positive episode timeout, a `lora_config` without
      `rank`, invalid `vllm_config.engine_kwargs`, or a setting the configured
      `rollout_engine` would silently ignore. The typed `vllm_config`
      routed-experts and detokenize settings are validated by
      `DistHyperParameters`.
  """
  _episode_timeout_secs(hp)
  lora_rank = _lora_rank(hp)
  _vllm_engine_kwargs(hp)
  rollout_engine = str(hp.config["rollout_engine"])
  if not bool(hp.config_mapping("rollout_config")["return_logprobs"]):
    raise ValueError(
        "rollout_config.return_logprobs must be true: the trainer consumes the"
        " rollout logprobs."
    )
  if rollout_engine == "vanilla" and lora_rank is not None:
    raise ValueError(
        "rollout_model_config.lora_config is not supported with"
        " rollout_engine: vanilla (the vanilla sampler loads the base model"
        " only); use a vLLM rollout_engine or drop lora_config."
    )
  if (
      rollout_engine == "inprocess_vllm"
      and hp.config_mapping("vllm_config")["server_mode"] is False
  ):
    raise ValueError(
        "vllm_config.server_mode: false is not supported with rollout_engine:"
        " inprocess_vllm, which always runs the in-process engine driver"
        " (concurrent sample() calls need it); remove it or use"
        " rollout_engine: vllm."
    )


def _chat_parser_for(
    model_id: str,
    tokenizer: Tokenizer,
    mode: str,
    *,
    enable_thinking: bool,
) -> ChatParser:
  """Selects the chat parser: `raw` text, or the model family's template.

  `BasePipeline._create_chat_parser` is not used: the dist schema selects
  `auto`/`raw` (validated by `DistHyperParameters`), and `auto` resolves the
  template from the model family rather than from `chat_parser_config.type`.

  Args:
    model_id: Model name used to pick the family-specific template parser.
    tokenizer: Tokenizer handed to the parser.
    mode: Either `auto` (model family's template) or `raw` (no template).
    enable_thinking: Whether the template parser opens a thinking block.

  Returns:
    The chat parser to use for this rollout worker.
  """
  if mode == "raw":
    return chat_parser_lib.RawTextParser(tokenizer)
  name = model_id.lower()
  for family, parser_cls in CHAT_PARSERS.items():
    if family in name:
      return parser_cls(tokenizer, enable_thinking=enable_thinking)
  return chat_parser_lib.DefaultChatTemplateParser(
      tokenizer, enable_thinking=enable_thinking
  )


def _create_chat_parser(
    hp: resolve_config.DistHyperParameters, tokenizer: Tokenizer
) -> ChatParser:
  chat_cfg = hp.config_mapping("chat_parser_config")
  return _chat_parser_for(
      _rollout_model_family_name(hp),
      tokenizer,
      str(chat_cfg["type"]),
      enable_thinking=bool(chat_cfg["enable_thinking"]),
  )


def _mapping_config_for(model_name: str) -> mappings_lib.MappingConfig:
  """Returns the Tunix-to-vLLM JAX mapping for a supported model family."""
  if _is_gemma4_model(model_name):
    from tunix.models.gemma4 import (  # pylint: disable=g-import-not-at-top
        mapping_vllm_jax,
    )
  elif "qwen3" in model_name.lower().replace("_", "-"):
    from tunix.models.qwen3 import (  # pylint: disable=g-import-not-at-top
        mapping_vllm_jax,
    )
  else:
    raise ValueError(
        "The in-process distributed rollout supports Qwen3 and Gemma4 "
        f"mappings; got model {model_name!r}."
    )

  return mappings_lib.MappingConfig(**mapping_vllm_jax.VLLM_JAX_MAPPING)


def _vllm_hf_overrides(
    hp: resolve_config.DistHyperParameters,
) -> dict[str, Any]:
  """Returns the vLLM `hf_overrides` to apply for the rollout model.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    The overrides dict, empty when the model needs no override.
  """
  if _maxtext_model_name(hp):
    # MaxText serves the model natively, so its architecture override takes the
    # place of any per-model-family one.
    return dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES)
  if _is_gemma4_model(_rollout_model_family_name(hp)):
    # Matches the text-only Gemma4 rollout configuration in FrozenLake.
    return {
        "final_logit_softcapping": 30.0,
        "text_config": {"final_logit_softcapping": 30.0},
        "architectures": ["Gemma4ForCausalLM"],
    }
  return {}


def _maxtext_additional_config(
    hp: resolve_config.DistHyperParameters,
) -> dict[str, Any] | None:
  """Returns the vLLM `additional_config` of a MaxText rollout, else None."""
  maxtext_model_name = _maxtext_model_name(hp)
  if not maxtext_model_name:
    return None
  # The architectures override is applied by `_vllm_hf_overrides`.
  logging.info(
      "Loading MaxText model %r natively via maxtext_vllm_adapter's"
      " MaxTextForCausalLM (architectures override).",
      maxtext_model_name,
  )
  kwargs = hp.config_mapping(_ROLLOUT_KEY)["kwargs"]
  return maxtext_utils.build_vllm_maxtext_additional_config(
      maxtext_model_name,
      attention=str(kwargs["attention"]) if "attention" in kwargs else "",
      prefuse_moe_weights=bool(
          hp.config_mapping("weight_sync_config")["prefuse_moe_weights"]
      ),
      return_routed_experts=bool(
          hp.config_mapping("vllm_config")["return_routed_experts"]
      ),
      float32_gate_logits=node_utils.optional_bool(
          kwargs, "float32_gate_logits"
      ),
      float32_logits=node_utils.optional_bool(kwargs, "float32_logits"),
  )


def _eos_token_ids(
    hp: resolve_config.DistHyperParameters, tokenizer: Tokenizer
) -> list[int] | None:
  """Resolves `rollout_config.eos_tokens` into token ids, or None when empty.

  Entries are either literal token ids or token strings (e.g. `<|im_end|>` for
  Qwen chat models), given as a comma-separated string or a list.

  Args:
    hp: Validated distributed RL configuration.
    tokenizer: Tokenizer used to resolve token strings to ids.

  Returns:
    The resolved token ids, or None when no entry is configured.

  Raises:
    ValueError: If an entry does not map to exactly one token id.
  """
  eos_raw = hp.config_mapping("rollout_config")["eos_tokens"]
  if eos_raw is None:
    raw_entries = []
  elif isinstance(eos_raw, (list, tuple)):
    raw_entries = [str(entry) for entry in eos_raw]
  else:
    raw_entries = str(eos_raw).split(",")
  entries = [entry.strip() for entry in raw_entries if entry.strip()]
  if not entries:
    return None
  token_ids = []
  for entry in entries:
    if entry.lstrip("-").isdigit():
      token_ids.append(int(entry))
      continue
    ids = tokenizer.encode(entry, add_special_tokens=False)
    if len(ids) != 1:
      raise ValueError(
          f"rollout_config.eos_tokens entry {entry!r} is not a single token"
          f" for this tokenizer (encoded to {ids})."
      )
    token_ids.extend(ids)
  logging.info("Resolved eos_tokens %s to ids %s", entries, token_ids)
  return token_ids


def _rollout_config(
    hp: resolve_config.DistHyperParameters,
    tokenizer: Tokenizer,
    *,
    sampler_type: str,
) -> rollout_worker.RolloutConfig:
  """Builds the dist `rollout_worker.RolloutConfig`.

  `BasePipeline.create_rollout_config()` is not used: it builds the base
  `RolloutConfig` (not the dist subclass with sampler, weight-sync, drain, and
  trajectory-store fields), passes `eos_tokens` through as a raw string, adds
  256 to `kv_cache_size`, and for `rollout_engine: vllm` sets in-engine vLLM
  fields (server mode, max_num_seqs, ...) that the dist samplers configure
  themselves.

  Args:
    hp: Validated distributed RL configuration.
    tokenizer: Tokenizer used to resolve `eos_tokens` strings.
    sampler_type: Sampler adapter the worker runs ("vanilla", "inprocess_vllm",
      "vllm"); vLLM samplers also get `rollout_vllm_model_version`.

  Returns:
    The rollout config.
  """
  rollout = hp.config_mapping("rollout_config")
  ws = hp.config_mapping("weight_sync_config")
  vllm = hp.config_mapping("vllm_config")
  num_experts = vllm["routed_experts_num_experts"]
  max_prompt_length = int(rollout["max_prompt_length"])
  max_tokens_to_generate = int(rollout["total_generation_steps"])
  return rollout_worker.RolloutConfig(
      rollout_vllm_model_version=(
          "" if sampler_type == "vanilla" else _vllm_model(hp)
      ),
      weight_sync_mode=weight_sync_lib.WeightSyncMode(str(ws["mode"])),
      max_prompt_length=max_prompt_length,
      max_tokens_to_generate=max_tokens_to_generate,
      # Only the vanilla sampler reads it.
      kv_cache_size=max_prompt_length + max_tokens_to_generate,
      temperature=float(rollout["temperature"]),
      top_p=float(rollout["top_p"]),
      # <= 0 disables top-k filtering in both the vanilla and vLLM samplers.
      top_k=int(rollout["top_k"]),
      # Required by the trainer; `_validate_rollout_config` rejects `false`.
      return_logprobs=True,
      eos_tokens=_eos_token_ids(hp, tokenizer),
      partial_rollout=bool(ws["in_flight_weight_updates"]),
      drain_timeout_s=timeouts.drain_timeout_for(_episode_timeout_secs(hp)),
      trajectory_store_config=node_utils.trajectory_store_config(hp),
      return_routed_experts=bool(vllm["return_routed_experts"]),
      routed_experts_encode_mode=cast(
          common_datatypes.RoutedExpertsEncodeMode,
          str(vllm["routed_experts_encode_mode"]),
      ),
      routed_experts_num_experts=(
          None if num_experts is None else int(num_experts)
      ),
  )


def _max_model_len(hp: resolve_config.DistHyperParameters) -> int:
  """Returns `vllm_config.max_model_len`, or prompt + response length."""
  max_model_len = hp.config_mapping("vllm_config")["max_model_len"]
  if max_model_len is not None:
    return int(max_model_len)
  rollout = hp.config_mapping("rollout_config")
  return int(rollout["max_prompt_length"]) + int(
      rollout["total_generation_steps"]
  )


def _vllm_model(hp: resolve_config.DistHyperParameters) -> str:
  """Returns the local model dir when it holds files, else the model id."""
  rollout_model = hp.config_mapping(_ROLLOUT_KEY)
  model_dir = str(rollout_model["model_download_path"])
  if model_dir and os.path.isdir(model_dir) and os.listdir(model_dir):
    return model_dir
  return str(rollout_model["model_id"])


def _common_vllm_engine_kwargs(
    hp: resolve_config.DistHyperParameters,
    *,
    vllm_model: str,
    max_model_len: int,
    partial_rollout: bool,
) -> dict[str, Any]:
  """Builds the `engine_kwargs` shared by both vLLM rollout samplers."""
  vllm = hp.config_mapping("vllm_config")
  engine_kwargs: dict[str, Any] = {
      "model": vllm_model,
      "max_model_len": max_model_len,
      "enable_prefix_caching": bool(
          vllm["enable_prefix_caching"] or partial_rollout
      ),
      "async_scheduling": bool(vllm["async_scheduling"]),
      "dtype": str(vllm["dtype"]),
      "max_logprobs": 1,
      "logprobs_mode": "processed_logprobs",
  }
  hf_overrides = _vllm_hf_overrides(hp)
  if hf_overrides:
    engine_kwargs["hf_overrides"] = hf_overrides
  if vllm["max_num_seqs"] is not None:
    engine_kwargs["max_num_seqs"] = int(vllm["max_num_seqs"])
  if vllm["max_num_batched_tokens"] is not None:
    engine_kwargs["max_num_batched_tokens"] = int(
        vllm["max_num_batched_tokens"]
    )
  return engine_kwargs


def _create_vanilla_sampler(
    hp: resolve_config.DistHyperParameters,
    worker_id: str,
    tokenizer: Tokenizer,
    config: rollout_worker.RolloutConfig,
) -> sampler_lib.Sampler:
  """Creates the vanilla (native JAX) sampler on the rollout mesh."""
  from tunix.dist_rl.rollout import (  # pylint: disable=g-import-not-at-top
      vanilla_sampler,
  )

  rollout_model = hp.config_mapping(_ROLLOUT_KEY)
  # `model_download_path` is always set by `DistHyperParameters`; make sure it
  # holds the weights (downloading `model_id` on a cold start) before loading.
  model_dir = node_utils.ensure_local_safetensors(
      str(rollout_model["model_download_path"]),
      str(rollout_model["model_id"]),
      model_key=_ROLLOUT_KEY,
  )
  logging.info("Creating native sampler on the rollout mesh...")
  mesh = node_utils.create_mesh(hp, _ROLLOUT_KEY, "Rollout")
  logging.info("Rollout mesh: %s", mesh)
  with mesh:
    model, _ = automodel.AutoModel.from_pretrained(
        model_id=str(rollout_model["model_name"]),
        mesh=mesh,
        model_path=model_dir,
        dtype=str(rollout_model["dtype"]),
        load_dtype=str(rollout_model["load_dtype"]),
    )
  return vanilla_sampler.VanillaSampler(
      server_id=worker_id,
      transformer=model,
      tokenizer=tokenizer,
      config=config,
  )


def _create_inprocess_vllm_sampler(
    hp: resolve_config.DistHyperParameters,
    worker_id: str,
    tokenizer: Tokenizer,
    config: rollout_worker.RolloutConfig,
) -> sampler_lib.Sampler:
  """Creates the in-process vLLM sampler."""
  vllm_sampler = _import_vllm_sampler()
  from tunix.dist_rl.rollout import (  # pylint: disable=g-import-not-at-top
      inprocess_vllm_sampler,
  )

  vllm = hp.config_mapping("vllm_config")
  ws = hp.config_mapping("weight_sync_config")
  logging.info("Creating vLLM mapping config...")
  family_name = _rollout_model_family_name(hp)
  mapping_config = _mapping_config_for(family_name)
  vllm_model = _vllm_model(hp)
  max_model_len = _max_model_len(hp)
  multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "")

  partial_rollout = bool(ws["in_flight_weight_updates"])
  engine_kwargs = _common_vllm_engine_kwargs(
      hp,
      vllm_model=vllm_model,
      max_model_len=max_model_len,
      partial_rollout=partial_rollout,
  )
  is_gemma4 = _is_gemma4_model(family_name)
  if is_gemma4:
    engine_kwargs["kv_cache_metrics"] = True
    engine_kwargs["disable_log_stats"] = False
  if multihost_backend:
    engine_kwargs["distributed_executor_backend"] = multihost_backend
  # `additional_config` must be set on VllmConfig (not engine_kwargs): the
  # sampler overwrites args["additional_config"] from the VllmConfig field.
  maxtext_additional_config = _maxtext_additional_config(hp)

  # Enable VLLMInProcessDriver (`server_mode=True`) on both single-host and
  # multi-host TPUs so concurrent `InprocessVllmSampler.sample()` worker
  # threads submit into a single engine loop for continuous batching without
  # racing on donated JAX KV-cache buffers. `InprocessVllmSampler` forces it
  # on regardless, so `_validate_rollout_config` rejects `server_mode: false`.
  rollout_mesh = (
      None
      if multihost_backend
      else node_utils.create_mesh(hp, _ROLLOUT_KEY, "Rollout")
  )
  tp_size = _tensor_parallel_size(hp)
  dp_size = _data_parallel_size(hp)
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "data_parallel_size=%d max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      tp_size,
      dp_size,
      max_model_len,
  )
  lora_rank = _lora_rank(hp)
  lora_config = (
      {"max_lora_rank": lora_rank, "max_loras": 1}
      if lora_rank is not None
      else None
  )
  return_routed_experts = bool(
      hp.config_mapping("vllm_config")["return_routed_experts"]
  )
  max_concurrency = int(
      hp.config_mapping("agentic_grpo_config")["max_concurrency"]
  )
  vllm_config = vllm_sampler.VllmConfig(
      server_mode=True,
      mesh=rollout_mesh,
      tensor_parallel_size=tp_size,
      data_parallel_size=dp_size,
      return_logprobs=True,
      return_routed_experts=return_routed_experts,
      init_with_random_weights=bool(vllm["init_with_random_weights"]),
      hbm_utilization=(
          float(vllm["hbm_utilization"])
          if vllm["hbm_utilization"] is not None
          else _DEFAULT_INPROCESS_VLLM_HBM_UTILIZATION
      ),
      lora_config=lora_config,
      mapping_config=mapping_config,
      additional_config=maxtext_additional_config,
      sampling_kwargs={"skip_special_tokens": False} if is_gemma4 else {},
      engine_kwargs=engine_kwargs,
      eos_tokens=config.eos_tokens,
      partial_rollout=partial_rollout,
  )
  return inprocess_vllm_sampler.InprocessVllmSampler(
      server_id=worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
      weight_sync_mode=weight_sync_lib.WeightSyncMode(str(ws["mode"])),
      max_concurrency=max_concurrency,
      partial_rollout=partial_rollout,
  )


def _create_vllm_sampler(
    hp: resolve_config.DistHyperParameters,
    worker_id: str,
    tokenizer: Tokenizer,
    config: rollout_worker.RolloutConfig,
) -> sampler_lib.Sampler:
  """Creates the vLLM `AsyncLLM` sampler."""
  del tokenizer  # The engine loads its own tokenizer from `tokenizer_path`.
  from tunix.dist_rl.rollout import (  # pylint: disable=g-import-not-at-top
      vllm_sampler,
  )
  from vllm.engine.arg_utils import (  # pylint: disable=g-import-not-at-top
      AsyncEngineArgs,
  )

  vllm = hp.config_mapping("vllm_config")
  ws = hp.config_mapping("weight_sync_config")
  vllm_model = _vllm_model(hp)
  max_model_len = _max_model_len(hp)
  tp_size = _tensor_parallel_size(hp)
  dp_size = max(1, _data_parallel_size(hp))
  lora_rank = _lora_rank(hp)
  logging.info(
      "Creating vLLM VllmSampler config for model=%s tensor_parallel_size=%d "
      "data_parallel_size=%d (%d chips) max_model_len=%d...",
      vllm_model,
      tp_size,
      dp_size,
      tp_size * dp_size,
      max_model_len,
  )
  if dp_size > 1 and lora_rank is not None:
    raise ValueError(
        "LoRA is not supported with rollout data parallelism:"
        f" rollout_model_config.mesh fsdp={dp_size} with"
        " rollout_model_config.lora_config. Set fsdp=1 or drop LoRA."
    )
  partial_rollout = bool(ws["in_flight_weight_updates"])
  tokenizer_path = str(hp.config_mapping("tokenizer_config")["tokenizer_path"])
  engine_kwargs = _common_vllm_engine_kwargs(
      hp,
      vllm_model=vllm_model,
      max_model_len=max_model_len,
      partial_rollout=partial_rollout,
  )
  engine_kwargs.update(
      tokenizer=tokenizer_path or vllm_model,
      tensor_parallel_size=tp_size,
      data_parallel_size=dp_size,
      trust_remote_code=True,
      enable_lora=lora_rank is not None,
      max_lora_rank=lora_rank,
      max_loras=1 if lora_rank is not None else None,
  )
  if vllm["hbm_utilization"] is not None:
    engine_kwargs["gpu_memory_utilization"] = float(vllm["hbm_utilization"])
  if bool(vllm["return_routed_experts"]):
    engine_kwargs["enable_return_routed_experts"] = True
  maxtext_additional_config = _maxtext_additional_config(hp)
  if maxtext_additional_config is not None:
    engine_kwargs["additional_config"] = maxtext_additional_config
  _merge_engine_kwargs(engine_kwargs, _vllm_engine_kwargs(hp))
  engine_args = AsyncEngineArgs(**engine_kwargs)
  return vllm_sampler.VllmSampler(
      server_id=worker_id,
      engine_args=engine_args,
      model_name=vllm_model,
      config=config,
      weight_sync_mode=weight_sync_lib.WeightSyncMode(str(ws["mode"])),
      partial_rollout=partial_rollout,
      detokenize=bool(vllm["detokenize"]),
  )


SamplerBuilder = Callable[
    [
        resolve_config.DistHyperParameters,
        str,
        Tokenizer,
        rollout_worker.RolloutConfig,
    ],
    sampler_lib.Sampler,
]

# `rollout_engine` -> sampler builder; every builder takes the same arguments
# and the shared `RolloutConfig`.
_SAMPLER_BUILDERS: Mapping[str, SamplerBuilder] = {
    "vanilla": _create_vanilla_sampler,
    "inprocess_vllm": _create_inprocess_vllm_sampler,
    "vllm": _create_vllm_sampler,
}


def _create_rollout_worker(
    hp: resolve_config.DistHyperParameters, worker_id: str, tokenizer: Tokenizer
) -> rollout_worker.RolloutWorker:
  """Builds the `RolloutConfig` and sampler, and wraps them in a worker."""
  from tunix.generate import (  # pylint: disable=g-import-not-at-top
      tokenizer_adapter as tokenizer_adapter_lib,
  )

  rollout_engine = str(hp.config["rollout_engine"])
  config = _rollout_config(hp, tokenizer, sampler_type=rollout_engine)
  sampler = _SAMPLER_BUILDERS[rollout_engine](hp, worker_id, tokenizer, config)
  logging.info("Creating RolloutWorker wrapper...")
  return rollout_worker.RolloutWorker(
      worker_id=worker_id,
      config=config,
      sampler=sampler,
      tokenizer=tokenizer_adapter_lib.TokenizerAdapter(tokenizer),
      chat_parser=_create_chat_parser(hp, tokenizer),
      env_name=str(hp.config["env_class_path"]),
      agent_name=str(hp.config["agent_class_path"]),
      env_config=hp.config_mapping("env_kwargs"),
      agent_config=hp.config_mapping("agent_kwargs"),
      max_concurrency=int(
          hp.config_mapping("agentic_grpo_config")["max_concurrency"]
      ),
  )


def _build_rollout_cache_params(
    hp: resolve_config.DistHyperParameters,
) -> gcs_cache.RolloutCacheKeyParams:
  """Builds `RolloutCacheKeyParams` from the config and environment."""
  env_params = gcs_cache.RolloutCacheKeyParams.from_env()
  rollout_model = hp.config_mapping(_ROLLOUT_KEY)
  rollout = hp.config_mapping("rollout_config")
  vllm = hp.config_mapping("vllm_config")
  lora_rank = _lora_rank(hp)
  max_model_len = vllm["max_model_len"]
  return dataclasses.replace(
      env_params,
      model_name=(
          _maxtext_model_name(hp)
          or str(rollout_model["model_name"])
          or str(rollout_model["model_id"])
      ),
      sampler=str(hp.config["rollout_engine"]),
      max_prompt_length=int(rollout["max_prompt_length"]),
      max_response_length=int(rollout["total_generation_steps"]),
      vllm_max_model_len=(
          int(max_model_len) if max_model_len is not None else None
      ),
      mesh_tp=_tensor_parallel_size(hp),
      mesh_fsdp=_data_parallel_size(hp),
      use_lora=lora_rank is not None,
      # RolloutCacheKeyParams' own "no LoRA" rank.
      lora_rank=lora_rank if lora_rank is not None else env_params.lora_rank,
      enable_prefix_caching=bool(vllm["enable_prefix_caching"]),
      vllm_server_mode=(
          bool(vllm["server_mode"])
          if vllm["server_mode"] is not None
          else env_params.vllm_server_mode
      ),
      vllm_async_scheduling=bool(vllm["async_scheduling"]),
  )


def main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a RolloutWorker over gRPC until SIGINT/SIGTERM.

  Args:
    argv: `override_config_file=...` and `section.key=value` overrides.
    context: Runtime process context providing JAX init and discovery.
  """
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )

  hp = resolve_config.initialize(argv)
  _validate_rollout_config(hp)
  cluster = hp.config_mapping("cluster_config")
  if cluster["debug"]:
    logging.getLogger().setLevel(logging.DEBUG)
    os.environ["TUNIX_LOG_ROLLOUT_TEXT"] = "1"
  logging.info("Resolved config: %s", hp.config)

  node_utils.ensure_repo_root_on_path()

  registry_module = str(cluster["registry_module"])
  logging.info("Importing rollout registry module: %s", registry_module)
  importlib.import_module(registry_module)

  gcs_cache.restore_jax_cache(params=_build_rollout_cache_params(hp))

  sampler_type = str(hp.config["rollout_engine"])
  if sampler_type == "vanilla":
    process_context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if _maxtext_model_name(hp):
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")

  tokenizer = node_utils.load_hf_tokenizer(hp)
  worker_id = node_utils.worker_id(hp, "rollout", LOCAL_WORKER_ID)
  port = node_utils.service_port(hp, "rollout")

  async def grpc_server_main() -> None:
    logging.info("Creating rollout worker service...")
    worker_service = _create_rollout_worker(hp, worker_id, tokenizer)

    async def _drain() -> None:
      # Cancel in-flight trajectory collections, then shut down the engine.
      worker_service.stop()
      await worker_service.sampler.stop()

    await node_utils.serve_worker(
        worker_service,
        process_context=process_context,
        role="rollout",
        worker_id_str=worker_id,
        port=port,
        extra_metadata={
            "trajectory_store_config": (
                worker_service.trajectory_store.to_config()
                if worker_service.trajectory_store is not None
                else None
            ),
        },
        drain_fn=_drain,
    )

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])
