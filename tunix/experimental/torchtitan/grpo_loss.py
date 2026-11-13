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

"""GRPO loss computation, PyTorch policy loss registry, and sequence packing."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import threading
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import torch
from tunix.rl import algorithm_config

AnyLossPluginT = TypeVar("AnyLossPluginT")

# Partition every `GRPOConfig` field between loss-path enforcement and
# orchestrator/rollout/advantage-only control so any newly added upstream field
# fails fast until explicitly audited here.
_LOSS_FIELDS_IMPLEMENTED: frozenset[str] = frozenset({
    "policy_loss_fn",
    "loss_algo",
    "loss_agg_mode",
    "epsilon",
    "epsilon_high",
    "epsilon_c",
    "beta",
    "temperature",
    "use_rollout_logps",
    "sampler_is",
    "sampler_is_threshold",
    "sampler_rs",
    "sampler_rs_min",
    "sampler_rs_max",
    "seq_logprob_error_threshold",
    "kl_loss_mode",
    "kl_clamp_value",
    "force_compute_kl",
    "score_centering",
    "score_centering_top_k",
    "score_centering_eps",
})

_FIELDS_IGNORED_BY_LOSS: frozenset[str] = frozenset({
    "algo_variant",
    "advantage_estimator",
    "reward_mode",
    "num_generations",
    "num_iterations",
    "drop_zero_variance_groups",
    "max_consecutive_filtered_groups",
    "reward_num_workers",
    "reward_worker_timeout_seconds",
    "routed_experts_encode_mode",
    "routed_experts_num_experts",
})


TorchPolicyLoss: TypeAlias = algorithm_config.PolicyLossPlugin[torch.Tensor]


_TORCH_POLICY_LOSS_REGISTRY: dict[str, TorchPolicyLoss] = {}
_REGISTRY_LOCK = threading.Lock()


@overload
def register_torch_policy_loss(
    name: str,
    loss_plugin: TorchPolicyLoss,
    *,
    overwrite: bool = False,
) -> TorchPolicyLoss:
  ...


@overload
def register_torch_policy_loss(
    name: str,
    loss_plugin: None = None,
    *,
    overwrite: bool = False,
) -> Callable[[AnyLossPluginT], AnyLossPluginT]:
  ...


def register_torch_policy_loss(
    name: str,
    loss_plugin: TorchPolicyLoss | None = None,
    *,
    overwrite: bool = False,
) -> Any:
  """Registers a `TorchPolicyLoss` plugin by name or returns a decorator."""
  if not name or not isinstance(name, str):
    raise ValueError(f"Policy loss name must be a non-empty str, got {name!r}.")

  def _store(plugin_or_cls: Any) -> Any:
    plugin = (
        plugin_or_cls() if isinstance(plugin_or_cls, type) else plugin_or_cls
    )
    if not isinstance(plugin, algorithm_config.PolicyLossPlugin):
      raise TypeError(
          "Registered PyTorch policy loss must implement TorchPolicyLoss"
          f" (validate + __call__), got {type(plugin).__name__}."
      )
    with _REGISTRY_LOCK:
      if name in _TORCH_POLICY_LOSS_REGISTRY and not overwrite:
        raise ValueError(
            f"Torch policy loss {name!r} is already registered;"
            " pass overwrite=True to replace it."
        )
      _TORCH_POLICY_LOSS_REGISTRY[name] = plugin
    return plugin_or_cls

  if loss_plugin is not None:
    _store(loss_plugin)
    return loss_plugin
  return _store


def get_torch_policy_loss(name: str) -> TorchPolicyLoss:
  """Returns the registered `TorchPolicyLoss` plugin for `name`."""
  with _REGISTRY_LOCK:
    if name in _TORCH_POLICY_LOSS_REGISTRY:
      return _TORCH_POLICY_LOSS_REGISTRY[name]
    available = sorted(_TORCH_POLICY_LOSS_REGISTRY)
  raise ValueError(
      f"Unknown PyTorch policy loss {name!r}; registered losses: {available}."
  )


def list_torch_policy_losses() -> list[str]:
  """Lists all registered PyTorch policy loss names."""
  with _REGISTRY_LOCK:
    return sorted(_TORCH_POLICY_LOSS_REGISTRY)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GRPOLossOutput:
  """Structured output of GRPO loss computation with pure tensor scalars."""

  unreduced_loss: torch.Tensor
  local_denom: torch.Tensor
  token_denom: torch.Tensor
  metric_sums: dict[str, torch.Tensor]


def compact_left_padded_batch_np(
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    completion_ids: np.ndarray,
    pad_id: int,
    total_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compacts left-padded prompt + right-padded completion into right-padded stream.

  Performs vectorized CPU compaction without Python for loops.

  Args:
    prompt_ids: [B, P] left-padded prompt token IDs.
    prompt_mask: [B, P] prompt mask (1 for valid prompt token, 0 for pad).
    completion_ids: [B, R] completion token IDs.
    pad_id: Pad token identifier.
    total_length: Optional total padded sequence length.

  Returns:
    compact_tokens: [B, L] right-padded concatenated sequence tokens.
    positions: [B, L] causal sequential positions.
    prompt_lens: [B] valid prompt length P_i per row.
  """
  batch_size, prompt_width = prompt_ids.shape
  _, completion_width = completion_ids.shape
  eff_total = (
      total_length
      if total_length is not None
      else (prompt_width + completion_width)
  )

  valid_prompt_mask = prompt_mask > 0
  prompt_lens = valid_prompt_mask.sum(axis=1).astype(np.int64)
  if np.any(prompt_lens <= 0):
    raise ValueError(
        "Every RL prompt must have >= 1 valid token; found row with prompt_len"
        " <= 0."
    )

  compact_tokens = np.full(
      (batch_size, eff_total), pad_id, dtype=prompt_ids.dtype
  )

  row_idx, col_idx = np.nonzero(valid_prompt_mask)
  within_row_pos = np.cumsum(valid_prompt_mask, axis=1)[row_idx, col_idx] - 1
  compact_tokens[row_idx, within_row_pos] = prompt_ids[row_idx, col_idx]

  comp_cols = prompt_lens[:, None] + np.arange(completion_width)[None, :]
  batch_idx = np.arange(batch_size)[:, None]
  compact_tokens[batch_idx, comp_cols] = completion_ids

  positions = np.tile(np.arange(eff_total, dtype=np.int64), batch_size)
  return compact_tokens, positions, prompt_lens


def extract_completion_logits(
    logits: torch.Tensor,
    prompt_lens: torch.Tensor,
    completion_width: int,
) -> torch.Tensor:
  """Extracts the R completion logits from the model output using static-shape gather.

  Args:
    logits: [B, L, V] logits from the model forward.
    prompt_lens: [B] valid prompt lengths P_i for each row.
    completion_width: Number of completion tokens R.

  Returns:
    [B, R, V] extracted logits for completion tokens.
  """
  offsets = torch.arange(
      completion_width, device=logits.device, dtype=torch.int64
  ).unsqueeze(0)
  prompt_lens_dev = prompt_lens.to(
      dtype=torch.int64, device=logits.device
  ).unsqueeze(1)
  gather_idx = (
      (prompt_lens_dev - 1 + offsets)
      .unsqueeze(-1)
      .expand(-1, -1, logits.shape[-1])
  )
  return torch.gather(logits, dim=1, index=gather_idx)


def compute_selective_log_softmax(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
  """Computes selective log probabilities for target tokens in float32.

  Matches Tunix `selective_log_softmax` bit-for-bit:
    logits_f32 = clamp(where(isnan, -1e9, logits), -1e9, 1e9)
    normalizer = logsumexp(logits_f32, dim=-1)
    target_logits - normalizer

  Args:
    logits: [B, R, V] logits.
    target_ids: [B, R] target token IDs.
    temperature: Scaling temperature.

  Returns:
    [B, R] log probabilities.
  """
  logits_f32 = logits.to(torch.float32)
  if temperature != 1.0 and temperature > 0:
    logits_f32 = logits_f32 / temperature

  logits_f32 = torch.where(
      torch.isnan(logits_f32),
      torch.tensor(-1e9, dtype=torch.float32, device=logits_f32.device),
      torch.clamp(logits_f32, min=-1e9, max=1e9),
  )

  target_ids_expanded = target_ids.unsqueeze(-1)
  target_logits = torch.gather(
      logits_f32, dim=-1, index=target_ids_expanded
  ).squeeze(-1)
  normalizer = torch.logsumexp(logits_f32, dim=-1)
  return target_logits - normalizer


def compute_kl_divergence(
    per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    method: str = "kl",
    clamp_value: float | None = None,
) -> torch.Tensor:
  """Computes per-token KL divergence matching Tunix `common.compute_kl_divergence`.

  Args:
    per_token_logps: [B, R] policy logps.
    ref_per_token_logps: [B, R] reference logps.
    method: "kl", "mse_kl", or "low_var_kl".
    clamp_value: Optional clamp on divergence.

  Returns:
    [B, R] per-token KL divergence.
  """
  p = per_token_logps.to(torch.float32)
  r = ref_per_token_logps.to(torch.float32)

  p = torch.where(
      torch.isnan(p),
      torch.tensor(-100.0, dtype=torch.float32, device=p.device),
      torch.clamp(p, min=-100.0, max=100.0),
  )
  r = torch.where(
      torch.isnan(r),
      torch.tensor(-100.0, dtype=torch.float32, device=r.device),
      torch.clamp(r, min=-100.0, max=100.0),
  )

  if method == "kl":
    kl = p - r
  elif method == "mse_kl":
    kl = 0.5 * torch.square(p - r)
  elif method == "low_var_kl":
    diff = torch.clamp(r - p, min=-20.0, max=20.0)
    kl = torch.exp(diff) - diff - 1.0
  else:
    raise ValueError(f"Unsupported KL method: {method!r}")

  if clamp_value is not None:
    kl = torch.clamp(kl, min=-clamp_value, max=clamp_value)
  return kl


def compute_grpo_loss(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_per_token_logps: torch.Tensor | None,
    ref_per_token_logps: torch.Tensor | None,
    algo_config: algorithm_config.GRPOConfig,
) -> GRPOLossOutput:
  """Computes unreduced GRPO loss sum and denominators without scalar unpacks.

  Matches `tunix.rl.algo_core.grpo_loss_fn` and
  `tunix.rl.common.aggregate_loss`.

  Args:
    per_token_logps: [B, R] policy logps.
    completion_mask: [B, R] completion mask.
    advantages: [B] or [B, R] advantages.
    old_per_token_logps: [B, R] or None.
    ref_per_token_logps: [B, R] or None.
    algo_config: Algorithm GRPO configuration.

  Returns:
    GRPOLossOutput containing pure tensor scalars.
  """
  device = per_token_logps.device
  per_token_logps = per_token_logps.to(torch.float32)

  if old_per_token_logps is None:
    old_per_token_logps = per_token_logps.detach()
  else:
    old_per_token_logps = old_per_token_logps.to(torch.float32)

  active_mask = (
      (completion_mask > 0)
      & torch.isfinite(per_token_logps)
      & torch.isfinite(old_per_token_logps)
  )

  safe_per_token_logps = torch.where(
      active_mask, per_token_logps, torch.tensor(0.0, device=device)
  )
  safe_old_per_token_logps = torch.where(
      active_mask, old_per_token_logps, torch.tensor(0.0, device=device)
  )
  seq_importance_ratio = safe_per_token_logps - safe_old_per_token_logps

  token_denom = torch.sum(completion_mask.to(torch.float32))
  unreduced_ppo_kl = torch.sum(
      torch.where(
          active_mask, -seq_importance_ratio, torch.tensor(0.0, device=device)
      )
  )

  seq_importance_ratio = torch.clamp(seq_importance_ratio, min=-20.0, max=20.0)

  if algo_config.loss_algo == "gspo-token":
    seq_mean_ratio = torch.sum(
        seq_importance_ratio, dim=-1, keepdim=True
    ) / torch.clamp(completion_mask.sum(dim=-1, keepdim=True), min=1.0)
    seq_importance_ratio = (
        safe_per_token_logps
        - safe_per_token_logps.detach()
        + seq_mean_ratio.detach()
    )
    seq_importance_ratio = torch.clamp(
        seq_importance_ratio, min=-10.0, max=10.0
    )

  is_ratio = torch.exp(seq_importance_ratio)

  raw_adv = (
      advantages if advantages.ndim == 2 else advantages.unsqueeze(1)
  ).to(torch.float32)
  adv = torch.where(
      active_mask & torch.isfinite(raw_adv),
      raw_adv,
      torch.tensor(0.0, device=device),
  )

  epsilon_high = (
      algo_config.epsilon
      if algo_config.epsilon_high is None
      else algo_config.epsilon_high
  )
  pg_loss_1 = -adv * is_ratio
  pg_loss_2 = -adv * torch.clamp(
      is_ratio,
      min=1.0 - algo_config.epsilon,
      max=1.0 + epsilon_high,
  )

  per_token_loss = torch.maximum(pg_loss_1, pg_loss_2)

  unreduced_clip_frac = torch.sum(
      torch.where(
          active_mask,
          (pg_loss_2 > pg_loss_1).to(torch.float32),
          torch.tensor(0.0, device=device),
      )
  )

  # KL divergence
  unreduced_kl = torch.tensor(0.0, dtype=torch.float32, device=device)
  if ref_per_token_logps is not None and (
      algo_config.beta != 0.0 or algo_config.force_compute_kl
  ):
    kl = compute_kl_divergence(
        per_token_logps,
        ref_per_token_logps,
        method=algo_config.kl_loss_mode,
        clamp_value=algo_config.kl_clamp_value,
    )
    kl = torch.where(active_mask, kl, torch.tensor(0.0, device=device))
    unreduced_kl = torch.sum(kl)
    if algo_config.beta != 0.0:
      per_token_loss = per_token_loss + algo_config.beta * kl

  # Aggregation mode
  if algo_config.loss_agg_mode == "token-mean":
    unreduced_loss = torch.sum(
        torch.where(
            active_mask, per_token_loss, torch.tensor(0.0, device=device)
        )
    )
    local_denom = torch.sum(completion_mask.to(torch.float32))
  elif algo_config.loss_agg_mode == "sequence-mean-token-mean":
    seq_mask = completion_mask.sum(dim=-1).to(torch.float32)
    masked_per_token_loss = torch.where(
        active_mask, per_token_loss, torch.tensor(0.0, device=device)
    )
    seq_loss = masked_per_token_loss.sum(dim=-1) / torch.clamp(
        seq_mask, min=1.0
    )
    unreduced_loss = torch.sum(seq_loss)
    local_denom = torch.sum((seq_mask > 0).to(torch.float32))
  else:
    raise ValueError(
        f"Unsupported loss_agg_mode: {algo_config.loss_agg_mode!r}"
    )

  metric_sums = {
      "pg_clipfrac": unreduced_clip_frac,
      "ppo_kl": unreduced_ppo_kl,
      "kl": unreduced_kl,
  }

  return GRPOLossOutput(
      unreduced_loss=unreduced_loss,
      local_denom=local_denom,
      token_denom=token_denom,
      metric_sums=metric_sums,
  )


class GRPOPolicyLoss:
  """Registered `'grpo'` policy loss plugin delegating to `compute_grpo_loss`."""

  def validate(
      self,
      algo_config: algorithm_config.AlgorithmConfig,
      *,
      fuse_agreement_in_loss: bool = False,
  ) -> None:
    """Validates `GRPOConfig` fields and options supported by TorchTitan."""
    if not isinstance(algo_config, algorithm_config.GRPOConfig):
      raise TypeError(
          "TorchTitan GRPOPolicyLoss requires GRPOConfig, got"
          f" {type(algo_config).__name__}."
      )
    known_fields = _LOSS_FIELDS_IMPLEMENTED | _FIELDS_IGNORED_BY_LOSS
    actual_fields = {f.name for f in dataclasses.fields(algo_config)}
    unknown_fields = sorted(actual_fields - known_fields)
    if unknown_fields:
      raise ValueError(
          "GRPOConfig contains unsupported fields not audited by"
          f" TorchTitanEngine: {unknown_fields}."
      )
    if algo_config.policy_loss_fn != "grpo":
      raise ValueError(
          "TorchTitanEngine only supports policy_loss_fn='grpo', got"
          f" {algo_config.policy_loss_fn!r}."
      )
    if algo_config.loss_algo not in ("grpo", "gspo-token"):
      raise ValueError(
          "TorchTitanEngine only supports loss_algo in ('grpo', 'gspo-token'),"
          f" got {algo_config.loss_algo!r}."
      )
    if algo_config.loss_agg_mode not in (
        "token-mean",
        "sequence-mean-token-mean",
    ):
      raise ValueError(
          "TorchTitanEngine only supports loss_agg_mode in ('token-mean',"
          f" 'sequence-mean-token-mean'), got {algo_config.loss_agg_mode!r}."
      )
    if algo_config.kl_loss_mode not in ("kl", "mse_kl", "low_var_kl"):
      raise ValueError(
          "TorchTitanEngine only supports kl_loss_mode in ('kl', 'mse_kl',"
          f" 'low_var_kl'), got {algo_config.kl_loss_mode!r}."
      )
    if (
        algo_config.kl_clamp_value is not None
        and algo_config.kl_clamp_value <= 0
    ):
      raise ValueError(
          "GRPOConfig.kl_clamp_value must be positive if set, got"
          f" {algo_config.kl_clamp_value!r}."
      )
    temperature = (
        1.0
        if algo_config.temperature is None
        else float(algo_config.temperature)
    )
    if temperature <= 0:
      raise ValueError(
          "GRPOConfig.temperature must be > 0, got"
          f" {algo_config.temperature!r}."
      )
    if algo_config.epsilon_c is not None:
      raise ValueError(
          "TorchTitanEngine does not support epsilon_c constraint clipping."
      )
    if (
        algo_config.sampler_is is not None
        or algo_config.sampler_rs is not None
        or algo_config.sampler_rs_min is not None
        or algo_config.sampler_rs_max is not None
    ):
      raise ValueError(
          "TorchTitanEngine does not yet support sampler_is or sampler_rs."
      )
    if algo_config.seq_logprob_error_threshold is not None:
      raise ValueError(
          "TorchTitanEngine does not yet support seq_logprob_error_threshold."
      )
    if algo_config.score_centering:
      raise ValueError(
          "TorchTitanEngine does not yet support score_centering=True."
      )
    if fuse_agreement_in_loss:
      raise ValueError(
          "TorchTitanEngine does not yet support fuse_agreement_in_loss=True."
      )

  def __call__(
      self,
      inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
      algo_config: algorithm_config.AlgorithmConfig,
  ) -> algorithm_config.PolicyLossOutput[torch.Tensor]:
    if not isinstance(algo_config, algorithm_config.GRPOConfig):
      raise TypeError(
          "GRPOPolicyLoss requires GRPOConfig, got"
          f" {type(algo_config).__name__}."
      )
    if inputs.segment_ids is not None:
      raise ValueError(
          "GRPOPolicyLoss does not support packed payloads (segment_ids must"
          " be None)."
      )
    if inputs.sampler_is_weights is not None:
      raise ValueError(
          "GRPOPolicyLoss does not yet support non-None sampler_is_weights."
      )
    out = compute_grpo_loss(
        per_token_logps=inputs.per_token_logps,
        completion_mask=inputs.completion_mask,
        advantages=inputs.advantages,
        old_per_token_logps=inputs.old_per_token_logps,
        ref_per_token_logps=inputs.ref_per_token_logps,
        algo_config=algo_config,
    )
    return algorithm_config.PolicyLossOutput(
        loss_sum=out.unreduced_loss,
        denominator=out.local_denom,
        metrics=out.metric_sums,
        token_denom=out.token_denom,
    )


register_torch_policy_loss("grpo", GRPOPolicyLoss(), overwrite=True)
