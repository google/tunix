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

"""Model-family vLLM rollout validation, weight mappings, and HF overrides."""

from __future__ import annotations

import re
from typing import Any

from tunix.common import datatypes as common_datatypes
from tunix.generate import mappings as mappings_lib
from tunix.utils import maxtext_utils

_GEMMA4_PATTERN = re.compile(
    r"(?<![a-z0-9])gemma[-_]?4(?![a-z0-9])", re.IGNORECASE
)
_QWEN3P5_PATTERN = re.compile(
    r"(?<![a-z0-9])qwen[-_]?3(?:[._]5|p5)(?![a-z0-9])", re.IGNORECASE
)
_QWEN3P5_MOE_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:moe|a[0-9]+b)(?![a-z0-9])", re.IGNORECASE
)
_MODEL_SEGMENT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:gemma[-_]?[1-9]|qwen[-_]?[1-9]"
    r"|llama[-_]?[1-9]|phi[-_]?[1-9]"
    r"|(?:deepseek|mistral|mixtral)(?![a-z]))",
    re.IGNORECASE,
)


def _target_model_segment(model_name: str) -> str:
  """Returns the last path segment in `model_name` naming a model family."""
  segments = [seg for seg in re.split(r"[/\\]+", model_name) if seg]
  for seg in reversed(segments):
    if _MODEL_SEGMENT_PATTERN.search(seg):
      return seg
  return segments[-1] if segments else model_name


_NO_WEIGHT_TRANSFER_MODES = frozenset({
    common_datatypes.WeightSyncMode.NONE,
    common_datatypes.WeightSyncMode.NOOP,
})


def _normalize_weight_sync_mode(
    weight_sync_mode: common_datatypes.WeightSyncMode | str,
) -> common_datatypes.WeightSyncMode:
  """Normalizes `weight_sync_mode` into a `WeightSyncMode` enum member."""
  if isinstance(weight_sync_mode, common_datatypes.WeightSyncMode):
    return weight_sync_mode
  return common_datatypes.WeightSyncMode(weight_sync_mode.lower())


def _uses_tunix_weight_sync(
    *,
    maxtext_model_name: str,
    weight_sync_mode: common_datatypes.WeightSyncMode | str | None,
) -> bool:
  """Returns whether Tunix-to-vLLM parameter synchronization is active."""
  normalized_mode = (
      _normalize_weight_sync_mode(weight_sync_mode)
      if weight_sync_mode is not None
      else None
  )
  return not maxtext_model_name and (
      normalized_mode is None
      or normalized_mode not in _NO_WEIGHT_TRANSFER_MODES
  )


def is_gemma4_model(model_name: str) -> bool:
  """Returns whether `model_name` names a Gemma4 variant.

  Model construction goes through `automodel`, but the rollout node still needs
  the family up front to pick the vLLM weight mapping and `hf_overrides` before
  any model or tokenizer exists.

  Args:
    model_name: Hugging Face model ID, local path, or Tunix selector (e.g.
      "gemma-4-e2b").
  """
  return bool(_GEMMA4_PATTERN.search(_target_model_segment(model_name)))


def is_qwen3p5_model(model_name: str) -> bool:
  """Returns whether `model_name` names a Qwen3.5 variant.

  Args:
    model_name: Hugging Face model ID, local path, or Tunix selector (e.g.
      "Qwen/Qwen3.5-0.8B" or "qwen3p5_0p8b").
  """
  return bool(_QWEN3P5_PATTERN.search(_target_model_segment(model_name)))


def is_qwen3p5_moe_model(model_name: str) -> bool:
  """Returns whether `model_name` names a Mixture-of-Experts Qwen3.5 variant.

  Matches delimited MoE tokens (`moe` or `a<N>b`, e.g. `A3B`) after the last
  Qwen3.5 family token in the resolved model path segment, so parent directory
  names such as `/ckpt/qwen3p5/Qwen3.5-35B-A3B`,
  `gs://b/qwen3.5-moe-sweeps/Qwen3.5-2B`, or `/gcs/moe_experiments/Qwen3.5-2B`
  do not flip MoE vs. dense classification.

  Args:
    model_name: Hugging Face model ID, local path, or Tunix selector (e.g.
      "Qwen/Qwen3.5-35B-A3B" or "qwen3p5_35b_a3b_base").
  """
  segment = _target_model_segment(model_name)
  matches = list(_QWEN3P5_PATTERN.finditer(segment))
  if not matches:
    return False
  return bool(_QWEN3P5_MOE_PATTERN.search(segment[matches[-1].end() :]))


def _is_qwen3_model(model_name: str) -> bool:
  """Returns whether `model_name` names a Qwen3 (non-3.5) model family."""
  target_segment = _target_model_segment(model_name)
  has_known_family = bool(_MODEL_SEGMENT_PATTERN.search(target_segment))
  return "qwen3" in target_segment.lower() or (
      not has_known_family and "qwen3" in model_name.lower()
  )


def validate_vllm_rollout_config(
    model_name: str,
    *,
    sampler_type: str,
    tp_size: int,
    weight_sync_mode: common_datatypes.WeightSyncMode | str,
    maxtext_model_name: str = "",
    use_lora: bool = False,
    init_with_random_weights: bool = True,
) -> None:
  """Validates model-family constraints for vLLM rollout samplers.

  MaxText rollouts (`maxtext_model_name` non-empty) and non-transferring modes
  (`weight_sync_mode` in `{"none", "noop"}`) bypass Tunix-to-vLLM parameter
  conversion and return without error once `weight_sync_mode` is validated. For
  active Tunix trainer weight sync:
  - Under `WeightSyncMode.TCP`, weights arrive in canonical HuggingFace format
    via `VllmSampler.update_params_from_hf`, requiring `inprocess_vllm`, no
    LoRA adapters, and a supported Qwen3 model family.
  - Out-of-process `sampler_type == "vllm"` (`VllmSampler`) returns `None` from
    `get_target_state()`, causing `PeftTrainerV2` to bind raw Tunix state
    without `MappingConfig` conversion for all non-MaxText families, and
    supports Raiden transport only (no `update_params` fallback). This guard
    scopes rejection to `Qwen3.5` and `Gemma4` so existing `Qwen3` +
    `sampler_type == "vllm"` configs remain unblocked pending broader cleanup.
  - Tunix `qwen3p5` `vllm_jax` mappings currently cover full-parameter dense
    `Qwen3.5` variants at `tensor_parallel_size=1` only (rejecting MoE, LoRA,
    and `tp_size != 1`) and require `init_with_random_weights=True` because
    official `Qwen3.5` checkpoints include `model.visual.*` tensors rejected by
    vLLM's text-only `Qwen3_5ForCausalLM` weight loader.

  Args:
    model_name: Rollout model ID or family selector.
    sampler_type: Configured rollout sampler ("vllm", "inprocess_vllm", or
      "vanilla").
    tp_size: Rollout tensor-parallel degree (`tensor_parallel_size`).
    weight_sync_mode: Weight synchronization mode (`WeightSyncMode` or string
      such as "raiden", "tcp", "in_memory", "fs", "noop", or "none").
    maxtext_model_name: MaxText model name when serving via
      `MaxTextForCausalLM`, or "" for non-MaxText rollouts.
    use_lora: Whether LoRA adapters are enabled on the rollout model.
    init_with_random_weights: Whether `inprocess_vllm` initializes dummy weights
      rather than loading checkpoint safetensors prior to weight sync.

  Raises:
    ValueError: If `weight_sync_mode` is invalid, if `weight_sync_mode` is
      `"tcp"` with a non-`inprocess_vllm` sampler or non-Qwen3 model, if
      `sampler_type == "vllm"` is paired with active non-MaxText weight sync on
      `Qwen3.5` or `Gemma4`, or if active non-MaxText `Qwen3.5`
      `inprocess_vllm` weight sync disables `init_with_random_weights`.
    NotImplementedError: If `weight_sync_mode` is `"tcp"` with LoRA or an
      unsupported Gemma4/Qwen3.5 family, or if active non-MaxText Qwen3.5 vLLM
      weight sync uses an MoE Qwen3.5 variant, LoRA adapters, or `tp_size != 1`.
  """
  mode = _normalize_weight_sync_mode(weight_sync_mode)
  if mode == common_datatypes.WeightSyncMode.TCP:
    if maxtext_model_name:
      raise NotImplementedError(
          "MaxText rollout with weight_sync_mode='tcp' is not yet supported."
      )
    if sampler_type != "inprocess_vllm":
      raise ValueError(
          f"Rollout sampler={sampler_type!r} with weight_sync_mode='tcp' is"
          " not supported; use inprocess_vllm instead."
      )
    if use_lora:
      raise NotImplementedError(
          "LoRA rollout weight sync with weight_sync_mode='tcp' is not yet"
          " supported."
      )
    if is_gemma4_model(model_name) or is_qwen3p5_model(model_name):
      raise NotImplementedError(
          "Hugging Face-to-vLLM JAX mapping is not yet supported for"
          f" {model_name!r}."
      )
    if not _is_qwen3_model(model_name):
      raise ValueError(
          "The in-process distributed rollout supports Qwen3 with"
          f" weight_sync_mode='tcp'; got model {model_name!r}."
      )
    return
  if not _uses_tunix_weight_sync(
      maxtext_model_name=maxtext_model_name, weight_sync_mode=mode
  ):
    return
  if sampler_type == "vllm" and (
      is_qwen3p5_model(model_name) or is_gemma4_model(model_name)
  ):
    raise ValueError(
        "Out-of-process vLLM rollout (sampler='vllm') with"
        f" weight_sync_mode={mode.value!r} is not supported for"
        f" {model_name!r}; use inprocess_vllm instead."
    )
  if sampler_type == "inprocess_vllm" and is_qwen3p5_model(model_name):
    if is_qwen3p5_moe_model(model_name):
      raise NotImplementedError(
          "Qwen3.5 MoE vLLM rollout weight sync without MaxText is not yet"
          f" implemented for {model_name!r}; only dense Qwen3.5 variants are"
          " supported."
      )
    if use_lora:
      raise NotImplementedError(
          "Qwen3.5 LoRA vLLM rollout weight sync without MaxText is not yet"
          f" implemented for {model_name!r}; disable LoRA or use MaxText."
      )
    if tp_size != 1:
      raise NotImplementedError(
          "Qwen3.5 vLLM rollout weight sync currently supports"
          f" tensor_parallel_size=1 only; got tensor_parallel_size={tp_size}."
      )
    if not init_with_random_weights:
      raise ValueError(
          "Qwen3.5 inprocess_vllm rollout with"
          f" weight_sync_mode={mode.value!r} requires"
          " init_with_random_weights=True (use weight_sync_mode='raiden' in"
          " distributed RL without disabling random weight initialization):"
          " official Qwen3.5 checkpoints include vision-tower tensors"
          " ('model.visual.*') rejected by vLLM's text-only"
          " Qwen3_5ForCausalLM loader, while Tunix weight sync targets"
          " Qwen3_5ForCausalLM parameter paths."
      )


def vllm_mapping_config_for_model(
    model_name: str,
    *,
    maxtext_model_name: str = "",
    weight_sync_mode: common_datatypes.WeightSyncMode | str | None = None,
) -> mappings_lib.MappingConfig:
  """Returns the Tunix-to-vLLM JAX `MappingConfig` for a supported family.

  Args:
    model_name: Hugging Face model ID or Tunix selector.
    maxtext_model_name: MaxText model name when serving via
      `MaxTextForCausalLM`, or "" for non-MaxText rollouts.
    weight_sync_mode: Optional rollout weight sync mode. Under MaxText
      (`maxtext_model_name` non-empty), non-transferring modes
      (`weight_sync_mode` in `{"none", "noop"}`), or `WeightSyncMode.TCP` for
      Qwen3 (which uses `VllmSampler.from_hf_manifest()` /
      `update_params_from_hf()` directly), an empty `MappingConfig()` is
      returned.

  Returns:
    A `MappingConfig` populated from the model family's `VLLM_JAX_MAPPING`, or
    an empty `MappingConfig()` when Tunix-NNX weight sync is inactive or `TCP`
    routes directly through `VllmSampler.update_params_from_hf()`.

  Raises:
    NotImplementedError: If `model_name` is a Qwen3.5 MoE variant and Tunix
      weight sync is active (or `weight_sync_mode` is omitted without MaxText),
      or if `weight_sync_mode` is `"tcp"` for Gemma4 or Qwen3.5.
    ValueError: If `weight_sync_mode` is invalid or `model_name` does not belong
      to Qwen3, Qwen3.5, or Gemma4 when Tunix weight sync is active.
  """
  normalized_mode = (
      _normalize_weight_sync_mode(weight_sync_mode)
      if weight_sync_mode is not None
      else None
  )
  if normalized_mode == common_datatypes.WeightSyncMode.TCP:
    if maxtext_model_name:
      raise NotImplementedError(
          "MaxText rollout with weight_sync_mode='tcp' is not yet supported."
      )
    if is_gemma4_model(model_name) or is_qwen3p5_model(model_name):
      raise NotImplementedError(
          "Hugging Face-to-vLLM JAX mapping is not yet supported for"
          f" {model_name!r}."
      )
    if _is_qwen3_model(model_name):
      return mappings_lib.MappingConfig()
    raise ValueError(
        "The in-process distributed rollout supports Qwen3, Qwen3.5, and "
        f"Gemma4 mappings; got model {model_name!r}."
    )
  if not _uses_tunix_weight_sync(
      maxtext_model_name=maxtext_model_name,
      weight_sync_mode=weight_sync_mode,
  ):
    return mappings_lib.MappingConfig()
  if is_gemma4_model(model_name):
    from tunix.models.gemma4 import (  # pylint: disable=g-import-not-at-top
        mapping_vllm_jax as gemma4_mapping,
    )

    return mappings_lib.MappingConfig(**gemma4_mapping.VLLM_JAX_MAPPING)
  if is_qwen3p5_model(model_name):
    if is_qwen3p5_moe_model(model_name):
      raise NotImplementedError(
          "Qwen3.5 MoE vLLM JAX weight mapping is not yet implemented for"
          f" {model_name!r}; only dense Qwen3.5 variants are supported."
      )
    from tunix.models.qwen3p5 import (  # pylint: disable=g-import-not-at-top
        mapping_vllm_jax as qwen3p5_mapping,
    )

    return mappings_lib.MappingConfig(**qwen3p5_mapping.VLLM_JAX_MAPPING)
  if _is_qwen3_model(model_name):
    from tunix.models.qwen3 import (  # pylint: disable=g-import-not-at-top
        mapping_vllm_jax as qwen3_mapping,
    )

    return mappings_lib.MappingConfig(**qwen3_mapping.VLLM_JAX_MAPPING)
  raise ValueError(
      "The in-process distributed rollout supports Qwen3, Qwen3.5, and "
      f"Gemma4 mappings; got model {model_name!r}."
  )


def vllm_hf_overrides_for_model(
    model_name: str,
    *,
    sampler_type: str = "inprocess_vllm",
    maxtext_model_name: str = "",
    weight_sync_mode: common_datatypes.WeightSyncMode | str | None = None,
    init_with_random_weights: bool = True,
) -> dict[str, Any]:
  """Returns the vLLM `hf_overrides` dict for `model_name`.

  Args:
    model_name: Hugging Face model ID or Tunix selector.
    sampler_type: Rollout sampler backend ("inprocess_vllm", "vllm", or
      "vanilla"). Only `inprocess_vllm` selects `Qwen3_5ForCausalLM` when
      dummy-initialized weights are overwritten by Tunix weight sync.
    maxtext_model_name: MaxText model name when serving via
      `MaxTextForCausalLM`, or "" for non-MaxText rollouts.
    weight_sync_mode: Optional rollout weight sync mode. Non-transferring
      Qwen3.5 rollouts (`weight_sync_mode` in `{"none", "noop"}`) keep the
      checkpoint's native `*ForConditionalGeneration` architecture so multimodal
      safetensors load without error.
    init_with_random_weights: Whether vLLM starts from dummy weights rather than
      loading checkpoint safetensors. `Qwen3_5ForCausalLM` is selected only when
      active Tunix weight sync initializes dummy weights into flat
      `vllm_model.model.*` paths.

  Returns:
    Dictionary of `hf_overrides` for vLLM engine initialization, or `{}` when no
    override is needed.
  """
  uses_tunix_weight_sync = _uses_tunix_weight_sync(
      maxtext_model_name=maxtext_model_name,
      weight_sync_mode=weight_sync_mode,
  )
  if maxtext_model_name:
    # MaxText serves the model natively, so its architecture override takes the
    # place of any per-model-family one.
    return dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES)
  if is_gemma4_model(model_name):
    # Matches the text-only Gemma4 rollout configuration in FrozenLake.
    return {
        "final_logit_softcapping": 30.0,
        "text_config": {"final_logit_softcapping": 30.0},
        "architectures": ["Gemma4ForCausalLM"],
    }
  if (
      sampler_type == "inprocess_vllm"
      and is_qwen3p5_model(model_name)
      and uses_tunix_weight_sync
      and init_with_random_weights
      and not is_qwen3p5_moe_model(model_name)
  ):
    # Official Qwen3.5 Hugging Face checkpoints declare multimodal
    # `Qwen3_5ForConditionalGeneration` in `config.json` (which nests LLM
    # weights under `language_model.*`); override to `Qwen3_5ForCausalLM` when
    # dummy-initialized `inprocess_vllm` receives Tunix-synced weights at
    # `vllm_model.model.*`.
    return {"architectures": ["Qwen3_5ForCausalLM"]}
  return {}
