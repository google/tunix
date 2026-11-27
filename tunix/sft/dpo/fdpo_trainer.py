# Copyright 2025 Google LLC
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

"""fDPO trainer."""

from __future__ import annotations

import dataclasses
from typing import Any

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
# TODO(abheesht): We should move TokenizerAdapter outside `generate`.
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.sft import peft_trainer
from typing_extensions import override


@flax.struct.dataclass(frozen=True)
class FDPODataInput:
  """Training data input for fDPO (Fine-grained DPO).

  This can be used when inputs are raw strings with separate description
  and reasoning segments. Tokenization, padding and preprocessing is
  taken care of by `FDPOTrainer`.

  Attributes:
    prompts: A list of prompts.
    chosen_desc: A list of chosen description responses (positive).
    rejected_desc: A list of rejected description responses (negative).
    chosen_reason: A list of chosen reasoning responses (positive).
    rejected_reason: A list of rejected reasoning responses (negative).
  """

  prompts: list[str]
  chosen_desc: list[str]
  rejected_desc: list[str]
  chosen_reason: list[str]
  rejected_reason: list[str]


@flax.struct.dataclass(frozen=True)
class FDPOTrainingInput:
  """Tokenized training input for fDPO (Fine-grained DPO).

  This can be used when inputs are already tokenized, padded and preprocessed,
  with separate description and reasoning segments.

  Attributes:
    prompt_ids: Prompt IDs. Should be left-padded.
    prompt_mask: Prompt mask. Should be left-padded.
    chosen_desc_ids: Chosen description IDs. Should be right-padded.
    chosen_desc_mask: Chosen description mask. Should be right-padded.
    rejected_desc_ids: Rejected description IDs. Should be right-padded.
    rejected_desc_mask: Rejected description mask. Should be right-padded.
    chosen_reason_ids: Chosen reasoning IDs. Should be right-padded.
    chosen_reason_mask: Chosen reasoning mask. Should be right-padded.
    rejected_reason_ids: Rejected reasoning IDs. Should be right-padded.
    rejected_reason_mask: Rejected reasoning mask. Should be right-padded.
  """

  # Prompt IDs should be left padded.
  prompt_ids: jax.Array | np.ndarray
  prompt_mask: jax.Array | np.ndarray
  # Description segment IDs should be right padded.
  chosen_desc_ids: jax.Array | np.ndarray
  chosen_desc_mask: jax.Array | np.ndarray
  rejected_desc_ids: jax.Array | np.ndarray
  rejected_desc_mask: jax.Array | np.ndarray
  # Reasoning segment IDs should be right padded.
  chosen_reason_ids: jax.Array | np.ndarray
  chosen_reason_mask: jax.Array | np.ndarray
  rejected_reason_ids: jax.Array | np.ndarray
  rejected_reason_mask: jax.Array | np.ndarray


@flax.struct.dataclass(frozen=True)
class FDPOTrainExample:
  """Training example for fDPO with separate description and reasoning segments."""
  # Description segment data
  desc_input_ids: jax.Array  # Concatenated [prompt_ids, desc_completion_ids]
  desc_positions: jax.Array
  desc_attention_mask: jax.Array
  ref_chosen_desc_logps: jax.Array | None
  ref_rejected_desc_logps: jax.Array | None
  desc_completion_mask: jax.Array
  desc_logits_to_keep: int = flax.struct.field(pytree_node=False)
  # Reasoning segment data
  reason_input_ids: jax.Array  # Concatenated [prompt_ids, reason_completion_ids]
  reason_positions: jax.Array
  reason_attention_mask: jax.Array
  ref_chosen_reason_logps: jax.Array | None
  ref_rejected_reason_logps: jax.Array | None
  reason_completion_mask: jax.Array
  reason_logits_to_keep: int = flax.struct.field(pytree_node=False)


@dataclasses.dataclass(slots=True, kw_only=True)
class FDPOTrainingConfig(peft_trainer.TrainingConfig):
  """fDPO (Fine-grained DPO) Training Config.

  fDPO introduces segment-level preference optimization with separate
  description and reasoning components, each with its own dynamic beta value.
  """

  algorithm: str = "fdpo"
  beta: float = (
      0.1  # Base 𝛽 for KL penalty https://arxiv.org/pdf/2305.18290
  )
  # fDPO-specific hyperparameters
  fdpo_lambda: float = 1.0  # λ: Controls sensitivity of segment weights
  fdpo_alpha: float = 0.5  # α: Controls maximum scaling amplitude for β
  label_smoothing: float = 0.0

  # Should be specified only if your input has strings instead of tokenized IDs.
  max_prompt_length: int | None = None
  max_response_length: int | None = None
  # Separate max lengths for description and reasoning segments
  max_desc_length: int | None = None
  max_reason_length: int | None = None


@nnx.jit(static_argnums=(4,))
def compute_logps(
    model,
    input_ids,
    positions,
    attention_mask,
    logits_to_keep,
    completion_mask,
):
  """Computes the log probabilities for chosen and rejected tokens."""
  token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  token_logps = (token_logps * completion_mask).sum(axis=-1)

  batch_size = token_logps.shape[0]
  chosen_logps = token_logps[: batch_size // 2]
  rejected_logps = token_logps[batch_size // 2 :]
  return chosen_logps, rejected_logps


@nnx.jit(static_argnums=(4,))
def compute_segment_logps(
    model,
    input_ids,
    positions,
    attention_mask,
    logits_to_keep,
    completion_mask,
):
  """Computes the log probabilities for a single segment (desc or reason)."""
  token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  token_logps = (token_logps * completion_mask).sum(axis=-1)

  batch_size = token_logps.shape[0]
  chosen_logps = token_logps[: batch_size // 2]
  rejected_logps = token_logps[batch_size // 2 :]
  return chosen_logps, rejected_logps


def compute_fdpo_beta_weights(
    delta_desc: jax.Array,
    delta_reason: jax.Array,
    base_beta: float,
    fdpo_lambda: float,
    fdpo_alpha: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes adaptive beta weights for fDPO.

  Following the fDPO formulation:
  - w_s = exp(λ * ΔR_s) / (exp(λ * ΔR_desc) + exp(λ * ΔR_reason))
  - β_s = β * [1 + α * (2*w_s - 1)]

  Args:
    delta_desc: Preference differential for description segment.
    delta_reason: Preference differential for reasoning segment.
    base_beta: Base β hyperparameter.
    fdpo_lambda: λ controlling sensitivity of weights.
    fdpo_alpha: α controlling maximum scaling amplitude.

  Returns:
    Tuple of (beta_desc, beta_reason) for each sample in the batch.
  """
  # Compute segment weights using numerically stable softmax
  # Stack the scaled deltas and use jax.nn.softmax for stability
  scaled_desc = fdpo_lambda * delta_desc
  scaled_reason = fdpo_lambda * delta_reason

  # Use log-sum-exp trick for numerical stability
  # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
  max_val = jnp.maximum(scaled_desc, scaled_reason)
  exp_desc = jnp.exp(scaled_desc - max_val)
  exp_reason = jnp.exp(scaled_reason - max_val)
  normalizer = exp_desc + exp_reason + 1e-10  # Add small epsilon to prevent division by zero

  w_desc = exp_desc / normalizer
  w_reason = exp_reason / normalizer

  # Clamp weights to valid range [0, 1] to prevent any numerical issues
  w_desc = jnp.clip(w_desc, 0.0, 1.0)
  w_reason = jnp.clip(w_reason, 0.0, 1.0)

  # Compute adaptive beta values (Eq. 5)
  # β_s = β * [1 + α * (2*w_s - 1)]
  beta_desc = base_beta * (1 + fdpo_alpha * (2 * w_desc - 1))
  beta_reason = base_beta * (1 + fdpo_alpha * (2 * w_reason - 1))

  return beta_desc, beta_reason


class FDPOTrainer(peft_trainer.PeftTrainer):
  """Fine-grained Direct Preference Optimization (fDPO) trainer.

  fDPO is a fine-grained preference learning algorithm that introduces
  segment-level preference granularity. It separates responses into
  description and reasoning components, applying adaptive beta values
  based on preference differentials for each segment.

  This implementation adapts the fDPO method for LLM training with
  separate description (answer) and reasoning segments.

  References:
  Yifan Shen, Yuanzhe Liu, Jingyuan Zhu, Xu Cao, Xiaofeng Zhang, Yixiao He, Wenming Ye, James Matthew Rehg, and Ismini Lourentzou, 
  Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs, 
  In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS) 2025
  https://plan-lab.github.io/
  """

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module | None,
      optimizer: optax.GradientTransformation,
      training_config: FDPOTrainingConfig,
      tokenizer: Any | None = None,
  ):
    """Initializes the fDPO trainer.

    Args:
      model: The policy model to be trained.
      ref_model: The reference/anchor model which is kept fixed/frozen during
        training. It is used to prevent the policy model from drifting too far
        from its original capabilities. If None, we don't use it in the loss.
      optimizer: The optimizer used for training the policy model.
      training_config: An `FDPOTrainingConfig` object containing fDPO-specific
        hyperparameters like `beta`, `fdpo_lambda`, `fdpo_alpha`, and
        `label_smoothing`.
      tokenizer: An optional tokenizer. If provided, the trainer can accept
        string inputs and tokenize them internally.
    """
    self.model = model
    self.ref_model = ref_model
    self.fdpo_config = training_config
    super().__init__(model, optimizer, training_config)

    self.tokenizer = (
        None
        if tokenizer is None
        else tokenizer_adapter.TokenizerAdapter(tokenizer)
    )

    self.with_loss_fn(fdpo_loss_fn, has_aux=True)
    self.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "beta": self.fdpo_config.beta,
            "fdpo_lambda": self.fdpo_config.fdpo_lambda,
            "fdpo_alpha": self.fdpo_config.fdpo_alpha,
            "label_smoothing": self.fdpo_config.label_smoothing,
        }
    )
    self.gen_model_input_fn = lambda x: {
        "train_example": x,
        "beta": self.fdpo_config.beta,
        "fdpo_lambda": self.fdpo_config.fdpo_lambda,
        "fdpo_alpha": self.fdpo_config.fdpo_alpha,
        "label_smoothing": self.fdpo_config.label_smoothing,
    }

    self._has_aux = True

    # If reference model is not provided, we don't use it in the loss term.
    self._ref_model_exists = ref_model is not None

    self._aux_metrics_to_log = {
        "rewards/chosen": np.mean,
        "rewards/rejected": np.mean,
        "rewards/margin": np.mean,
        "rewards/accuracy": np.mean,
        "log_probs/chosen_desc": np.mean,
        "log_probs/rejected_desc": np.mean,
        "log_probs/chosen_reason": np.mean,
        "log_probs/rejected_reason": np.mean,
        "beta/desc": np.mean,
        "beta/reason": np.mean,
        "F/desc": np.mean,
        "F/reason": np.mean,
    }

  @override
  def _prepare_inputs(
      self,
      training_input: dict[str, Any] | FDPODataInput | FDPOTrainingInput,
  ) -> Any:
    if isinstance(training_input, dict):
      training_input = _preprocess_fdpo_dict(training_input)

    # If the inputs are list of strings, let's tokenise them and pad them.
    if isinstance(training_input, FDPODataInput):
      if self.tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided if training input is not tokenized."
        )

      max_prompt_length = self.fdpo_config.max_prompt_length
      max_desc_length = self.fdpo_config.max_desc_length
      max_reason_length = self.fdpo_config.max_reason_length

      if max_prompt_length is None:
        raise ValueError("max_prompt_length must be provided.")

      # Use max_response_length as fallback if segment lengths not specified
      if max_desc_length is None:
        max_desc_length = self.fdpo_config.max_response_length
      if max_reason_length is None:
        max_reason_length = self.fdpo_config.max_response_length

      if max_desc_length is None or max_reason_length is None:
        raise ValueError(
            "max_desc_length and max_reason_length (or max_response_length) "
            "must be provided if training input is not tokenized."
        )

      training_input = process_fdpo_record(
          record={
              "prompts": training_input.prompts,
              "chosen_desc": training_input.chosen_desc,
              "rejected_desc": training_input.rejected_desc,
              "chosen_reason": training_input.chosen_reason,
              "rejected_reason": training_input.rejected_reason,
          },
          tokenizer=self.tokenizer,
          max_prompt_length=max_prompt_length,
          max_desc_length=max_desc_length,
          max_reason_length=max_reason_length,
      )

    # Process description segment
    desc_prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids], axis=0
    )
    desc_prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask], axis=0
    )
    desc_completion_ids = jnp.concatenate(
        [training_input.chosen_desc_ids, training_input.rejected_desc_ids], axis=0
    )
    desc_completion_mask = jnp.concatenate(
        [training_input.chosen_desc_mask, training_input.rejected_desc_mask], axis=0
    )
    desc_input_ids = jnp.concat([desc_prompt_ids, desc_completion_ids], axis=1)
    desc_mask = jnp.concat([desc_prompt_mask, desc_completion_mask], axis=1)
    desc_attention_mask = common.make_causal_attn_mask(desc_mask)
    desc_logits_to_keep = desc_completion_ids.shape[1]
    desc_positions = common.build_positions_from_mask(desc_mask)

    # Process reasoning segment
    reason_prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids], axis=0
    )
    reason_prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask], axis=0
    )
    reason_completion_ids = jnp.concatenate(
        [training_input.chosen_reason_ids, training_input.rejected_reason_ids], axis=0
    )
    reason_completion_mask = jnp.concatenate(
        [training_input.chosen_reason_mask, training_input.rejected_reason_mask], axis=0
    )
    reason_input_ids = jnp.concat([reason_prompt_ids, reason_completion_ids], axis=1)
    reason_mask = jnp.concat([reason_prompt_mask, reason_completion_mask], axis=1)
    reason_attention_mask = common.make_causal_attn_mask(reason_mask)
    reason_logits_to_keep = reason_completion_ids.shape[1]
    reason_positions = common.build_positions_from_mask(reason_mask)

    # Compute reference model log probabilities if ref model exists
    ref_chosen_desc_logps = None
    ref_rejected_desc_logps = None
    ref_chosen_reason_logps = None
    ref_rejected_reason_logps = None

    if self._ref_model_exists:
      ref_chosen_desc_logps, ref_rejected_desc_logps = compute_segment_logps(
          self.ref_model,
          desc_input_ids,
          desc_positions,
          desc_attention_mask,
          desc_logits_to_keep,
          desc_completion_mask,
      )
      ref_chosen_reason_logps, ref_rejected_reason_logps = compute_segment_logps(
          self.ref_model,
          reason_input_ids,
          reason_positions,
          reason_attention_mask,
          reason_logits_to_keep,
          reason_completion_mask,
      )

    return FDPOTrainExample(
        desc_input_ids=desc_input_ids,
        desc_positions=desc_positions,
        desc_attention_mask=desc_attention_mask,
        ref_chosen_desc_logps=ref_chosen_desc_logps,
        ref_rejected_desc_logps=ref_rejected_desc_logps,
        desc_completion_mask=desc_completion_mask,
        desc_logits_to_keep=desc_logits_to_keep,
        reason_input_ids=reason_input_ids,
        reason_positions=reason_positions,
        reason_attention_mask=reason_attention_mask,
        ref_chosen_reason_logps=ref_chosen_reason_logps,
        ref_rejected_reason_logps=ref_rejected_reason_logps,
        reason_completion_mask=reason_completion_mask,
        reason_logits_to_keep=reason_logits_to_keep,
    )

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    assert self._buffered_train_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )


def fdpo_loss_fn(
    model: nnx.Module,
    train_example: FDPOTrainExample,
    beta: float = 0.1,
    fdpo_lambda: float = 1.0,
    fdpo_alpha: float = 0.5,
    label_smoothing: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """fDPO (Fine-grained DPO) loss function.

  Implements the fDPO loss with segment-level preference optimization:
  L_fDPO = -log σ(β_desc * F_desc + β_reason * F_reason)

  where F_s measures the segment-specific preference margin in log-likelihood
  ratios relative to a reference policy.

  Args:
    model: The model to compute loss for.
    train_example: fDPO training example containing desc and reason segments.
    beta: Base β for KL penalty.
    fdpo_lambda: λ controlling sensitivity of segment weights.
    fdpo_alpha: α controlling maximum scaling amplitude for β.
    label_smoothing: Label smoothing factor.

  Returns:
    A tuple of (loss, auxiliary_metrics_dict).
  """
  # Compute log probabilities for description segment
  chosen_desc_logps, rejected_desc_logps = compute_segment_logps(
      model,
      train_example.desc_input_ids,
      train_example.desc_positions,
      train_example.desc_attention_mask,
      train_example.desc_logits_to_keep,
      train_example.desc_completion_mask,
  )

  # Compute log probabilities for reasoning segment
  chosen_reason_logps, rejected_reason_logps = compute_segment_logps(
      model,
      train_example.reason_input_ids,
      train_example.reason_positions,
      train_example.reason_attention_mask,
      train_example.reason_logits_to_keep,
      train_example.reason_completion_mask,
  )

  # Compute log ratios for description segment (F_desc in the paper)
  chosen_desc_log_ratio = chosen_desc_logps
  if train_example.ref_chosen_desc_logps is not None:
    chosen_desc_log_ratio = chosen_desc_log_ratio - train_example.ref_chosen_desc_logps
  rejected_desc_log_ratio = rejected_desc_logps
  if train_example.ref_rejected_desc_logps is not None:
    rejected_desc_log_ratio = rejected_desc_log_ratio - train_example.ref_rejected_desc_logps

  # F_desc = log(π_θ(R^p_desc|x)/π_ref(R^p_desc|x)) - log(π_θ(R^l_desc|x)/π_ref(R^l_desc|x))
  F_desc = chosen_desc_log_ratio - rejected_desc_log_ratio

  # Compute log ratios for reasoning segment (F_reason in the paper)
  chosen_reason_log_ratio = chosen_reason_logps
  if train_example.ref_chosen_reason_logps is not None:
    chosen_reason_log_ratio = chosen_reason_log_ratio - train_example.ref_chosen_reason_logps
  rejected_reason_log_ratio = rejected_reason_logps
  if train_example.ref_rejected_reason_logps is not None:
    rejected_reason_log_ratio = rejected_reason_log_ratio - train_example.ref_rejected_reason_logps

  # F_reason = log(π_θ(R^p_reason|x)/π_ref(R^p_reason|x)) - log(π_θ(R^l_reason|x)/π_ref(R^l_reason|x))
  F_reason = chosen_reason_log_ratio - rejected_reason_log_ratio

  # Compute preference differentials (ΔR_desc and ΔR_reason)
  # Using log probability differences as the preference signal
  # ΔR_s = score(R^p_s) - score(R^l_s)
  # Normalize by sequence length to prevent scale issues
  desc_mask_sum = train_example.desc_completion_mask.sum(axis=-1)
  reason_mask_sum = train_example.reason_completion_mask.sum(axis=-1)
  batch_size = desc_mask_sum.shape[0] // 2
  
  # Get per-sample sequence lengths for normalization
  chosen_desc_len = jnp.maximum(desc_mask_sum[:batch_size], 1.0)
  rejected_desc_len = jnp.maximum(desc_mask_sum[batch_size:], 1.0)
  chosen_reason_len = jnp.maximum(reason_mask_sum[:batch_size], 1.0)
  rejected_reason_len = jnp.maximum(reason_mask_sum[batch_size:], 1.0)
  
  # Normalize log probabilities by sequence length for stable delta computation
  delta_desc = (chosen_desc_logps / chosen_desc_len) - (rejected_desc_logps / rejected_desc_len)
  delta_reason = (chosen_reason_logps / chosen_reason_len) - (rejected_reason_logps / rejected_reason_len)

  # Compute adaptive beta values
  beta_desc, beta_reason = compute_fdpo_beta_weights(
      delta_desc, delta_reason, beta, fdpo_lambda, fdpo_alpha
  )

  # Compute fDPO loss: L_fDPO = -log σ(β_desc * F_desc + β_reason * F_reason)
  combined_delta = beta_desc * F_desc + beta_reason * F_reason
  
  # Clip combined_delta to prevent numerical overflow in sigmoid
  combined_delta = jnp.clip(combined_delta, -50.0, 50.0)

  losses = -(
      jax.nn.log_sigmoid(combined_delta) * (1 - label_smoothing)
      + jax.nn.log_sigmoid(-combined_delta) * label_smoothing
  )
  
  # Replace any NaN/Inf with a large but finite loss value
  losses = jnp.where(jnp.isfinite(losses), losses, 10.0)

  # Compute rewards for logging
  chosen_desc_rewards = beta_desc * chosen_desc_log_ratio
  rejected_desc_rewards = beta_desc * rejected_desc_log_ratio
  chosen_reason_rewards = beta_reason * chosen_reason_log_ratio
  rejected_reason_rewards = beta_reason * rejected_reason_log_ratio

  # Combined rewards
  chosen_rewards = chosen_desc_rewards + chosen_reason_rewards
  rejected_rewards = rejected_desc_rewards + rejected_reason_rewards

  aux = {
      "rewards/chosen": chosen_rewards.mean(),
      "rewards/rejected": rejected_rewards.mean(),
      "rewards/margin": (chosen_rewards - rejected_rewards).mean(),
      "rewards/accuracy": (chosen_rewards > rejected_rewards).mean(),
      "log_probs/chosen_desc": chosen_desc_logps.mean(),
      "log_probs/rejected_desc": rejected_desc_logps.mean(),
      "log_probs/chosen_reason": chosen_reason_logps.mean(),
      "log_probs/rejected_reason": rejected_reason_logps.mean(),
      "beta/desc": beta_desc.mean(),
      "beta/reason": beta_reason.mean(),
      "F/desc": F_desc.mean(),
      "F/reason": F_reason.mean(),
  }

  return losses.mean(), aux


def _generate_ids_and_masks(
    input_strings: list[str],
    tokenizer: Any,
    max_length: int,
    left_pad: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Generates ids and masks for a list of strings."""
  tokens = [_tokenize(x, tokenizer) for x in input_strings]
  all_input_ids = jnp.array([
      common.pad_to_length(
          x[:max_length],
          target_length=max_length,
          pad_value=tokenizer.pad_id(),
          left=left_pad,
          axis=-1,
      )
      for x in tokens
  ])
  # generate masks
  all_input_mask = (all_input_ids != tokenizer.pad_id()).astype("int32")
  return all_input_ids, all_input_mask


def _tokenize(input_string: str, tokenizer: Any) -> jax.Array:
  """Tokenizes the input string."""
  input_ids = tokenizer.encode(input_string)
  bos_tok = [tokenizer.bos_id()] if tokenizer.bos_id() else []
  input_ids = jnp.array(
    tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=jnp.int32
  )
  return input_ids


def _preprocess_fdpo_dict(
    training_input: dict[str, Any],
) -> FDPODataInput | FDPOTrainingInput:
  """Wraps input dict with either FDPODataInput or FDPOTrainingInput."""

  fdpo_data_input_fields = [
      field.name for field in dataclasses.fields(FDPODataInput)
  ]
  fdpo_tokenized_input_fields = [
      field.name for field in dataclasses.fields(FDPOTrainingInput)
  ]

  # If the dict contains tokenized fields, we should wrap it with
  # FDPOTrainingInput.
  if all(field in training_input for field in fdpo_tokenized_input_fields):
    return FDPOTrainingInput(
        **{field: training_input[field] for field in fdpo_tokenized_input_fields}
    )
  elif all(field in training_input for field in fdpo_data_input_fields):
    return FDPODataInput(
        **{field: training_input[field] for field in fdpo_data_input_fields}
    )
  else:
    raise ValueError(
        "Training input must contain either tokenized fields "
        f"({fdpo_tokenized_input_fields}) or raw string fields "
        f"({fdpo_data_input_fields}). Received: {training_input.keys()}."
    )


def process_fdpo_record(
    record: dict[str, str | list[str]],
    tokenizer: Any,
    max_prompt_length: int,
    max_desc_length: int,
    max_reason_length: int,
) -> FDPOTrainingInput:
  """Processes and tokenizes a single record for fDPO training.

  This function takes a dictionary containing a prompt, chosen/rejected
  description segments, and chosen/rejected reasoning segments. It tokenizes
  each text field and creates the corresponding attention masks.

  Args:
      record: A dictionary, containing "prompts", "chosen_desc", "rejected_desc",
        "chosen_reason", and "rejected_reason" as keys. The values can be a
        single string or a list of strings.
      tokenizer: The tokenizer to use for converting text into token IDs.
      max_prompt_length: The maximum length for the tokenized prompts.
      max_desc_length: The maximum length for the tokenized description segments.
      max_reason_length: The maximum length for the tokenized reasoning segments.

  Returns:
      An `FDPOTrainingInput` object.
  """
  prompts = record["prompts"]
  chosen_desc = record["chosen_desc"]
  rejected_desc = record["rejected_desc"]
  chosen_reason = record["chosen_reason"]
  rejected_reason = record["rejected_reason"]

  unbatched = isinstance(prompts, str)

  if unbatched:
    prompts = [prompts]
  if isinstance(chosen_desc, str):
    chosen_desc = [chosen_desc]
  if isinstance(rejected_desc, str):
    rejected_desc = [rejected_desc]
  if isinstance(chosen_reason, str):
    chosen_reason = [chosen_reason]
  if isinstance(rejected_reason, str):
    rejected_reason = [rejected_reason]

  # Only prompt is left padded, others are right padded.
  prompt_ids, prompt_mask = _generate_ids_and_masks(
      prompts,
      tokenizer,
      max_prompt_length,
      left_pad=True,
  )

  # Tokenize description segments
  chosen_desc_ids, chosen_desc_mask = _generate_ids_and_masks(
      chosen_desc, tokenizer, max_desc_length, left_pad=False
  )
  rejected_desc_ids, rejected_desc_mask = _generate_ids_and_masks(
      rejected_desc, tokenizer, max_desc_length, left_pad=False
  )

  # Tokenize reasoning segments
  chosen_reason_ids, chosen_reason_mask = _generate_ids_and_masks(
      chosen_reason, tokenizer, max_reason_length, left_pad=False
  )
  rejected_reason_ids, rejected_reason_mask = _generate_ids_and_masks(
      rejected_reason, tokenizer, max_reason_length, left_pad=False
  )

  if unbatched:
    prompt_ids = jnp.squeeze(prompt_ids, axis=0)
    prompt_mask = jnp.squeeze(prompt_mask, axis=0)
    chosen_desc_ids = jnp.squeeze(chosen_desc_ids, axis=0)
    chosen_desc_mask = jnp.squeeze(chosen_desc_mask, axis=0)
    rejected_desc_ids = jnp.squeeze(rejected_desc_ids, axis=0)
    rejected_desc_mask = jnp.squeeze(rejected_desc_mask, axis=0)
    chosen_reason_ids = jnp.squeeze(chosen_reason_ids, axis=0)
    chosen_reason_mask = jnp.squeeze(chosen_reason_mask, axis=0)
    rejected_reason_ids = jnp.squeeze(rejected_reason_ids, axis=0)
    rejected_reason_mask = jnp.squeeze(rejected_reason_mask, axis=0)

  return FDPOTrainingInput(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      chosen_desc_ids=chosen_desc_ids,
      chosen_desc_mask=chosen_desc_mask,
      rejected_desc_ids=rejected_desc_ids,
      rejected_desc_mask=rejected_desc_mask,
      chosen_reason_ids=chosen_reason_ids,
      chosen_reason_mask=chosen_reason_mask,
      rejected_reason_ids=rejected_reason_ids,
      rejected_reason_mask=rejected_reason_mask,
  )

# fDPO aliases
FdpoTrainingConfig = FDPOTrainingConfig
FdpoTrainer = FDPOTrainer