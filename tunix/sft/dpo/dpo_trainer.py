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

"""DPO trainer."""

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
class DataInput:
  """Training data input for DPO.

  This can be used when inputs are raw strings. Tokenization, padding and
  preprocessing is taken care of by `DPOTrainer`.

  Attributes:
    prompts: A list of prompts.
    chosen_responses: A list of chosen responses.
    rejected_responses: A list of rejected responses.
  """

  prompts: list[str]
  chosen_responses: list[str]
  rejected_responses: list[str]


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  """Tokenized training input for DPO.

  This can be used when inputs are already tokenized, padded and preprocessed.

  Attributes:
    prompt_ids: Prompt IDs. Should be left-padded.
    prompt_mask: Prompt mask. Should be left-padded.
    chosen_ids: Chosen response IDs. Should be right-padded.
    chosen_mask: Chosen response mask. Should be right-padded.
    rejected_ids: Rejected response IDs. Should be right-padded.
    rejected_mask: Rejected response mask. Should be right-padded.
  """

  # Prompt IDs should be left padded.
  prompt_ids: jax.Array | np.ndarray
  prompt_mask: jax.Array | np.ndarray
  # Chosen IDs should be right padded.
  chosen_ids: jax.Array | np.ndarray
  chosen_mask: jax.Array | np.ndarray
  # Rejected IDs should be right padded.
  rejected_ids: jax.Array | np.ndarray
  rejected_mask: jax.Array | np.ndarray


@flax.struct.dataclass(frozen=True)
class TrainExample:
  input_ids: jax.Array  # Concatenated [prompt_ids, completion_ids]
  positions: jax.Array
  attention_mask: jax.Array
  ref_chosen_logps: jax.Array
  ref_rejected_logps: jax.Array
  completion_mask: jax.Array
  logits_to_keep: int = flax.struct.field(pytree_node=False)


@dataclasses.dataclass(slots=True, kw_only=True)
class DPOTrainingConfig(peft_trainer.TrainingConfig):
  """DPO Training Config."""

  beta: float = 0.1  # 𝛽 for KL penalty https://arxiv.org/pdf/2305.18290
  label_smoothing: float = 0.0

  # Should be specified only if your input has strings instead of tokenized IDs.
  max_prompt_length: int | None = None
  max_response_length: int | None = None


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
  token_logps, _ = common.get_per_token_logps(
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


class DPOTrainer(peft_trainer.PeftTrainer):
  """Direct Preference Optimization (DPO) trainer.

  DPO is a preference tuning method for aligning large language models with
  human or AI preferences. It is a more efficient, performant alternative
  to RLHF.

  DPO is simpler because it eliminates the need for text generation in the
  training loop. Moreover, DPO bypasses the reward modeling step entirely, i.e.,
  we do not need to train a separate reward model. It uses a dataset of
  preferences (pairs of "chosen" and "rejected responses) to directly optimize
  the policy model by using a classification-style loss.

  References:
  - https://arxiv.org/abs/2305.18290
  """

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: DPOTrainingConfig,
      tokenizer: Any | None = None,
  ):
    """Initializes the DPO trainer.

    Args:
      model: The policy model to be trained.
      ref_model: The reference/anchor model which is kept fixed/frozen during
        training. It is used to prevent the policy model from drifting too far
        from its original capabilities.
      optimizer: The optimizer used for training the policy model.
      training_config: A `DPOTrainingConfig` object containing DPO-specific
        hyperparameters like `beta` and `label_smoothing`.
      tokenizer: An optional tokenizer. If provided, the trainer can accept
        string inputs and tokenize them internally.
    """
    self.model = model
    self.ref_model = ref_model
    self.dpo_config = training_config
    super().__init__(model, optimizer, training_config)

    self.tokenizer = (
        None
        if tokenizer is None
        else tokenizer_adapter.TokenizerAdapter(tokenizer)
    )

    self.loss_fn = dpo_loss_fn
    self.gen_model_input_fn = lambda x: {
        "train_example": x,
        "beta": self.dpo_config.beta,
        "label_smoothing": self.dpo_config.label_smoothing,
    }
    self._has_aux = True

  @override
  def _prepare_inputs(
      self,
      training_input: dict[str, Any] | DataInput | TrainingInput,
  ) -> Any:
    if isinstance(training_input, dict):
      training_input = _preprocess_dict(training_input)

    # If the inputs are list of strings, let's tokenise them and pad them.
    if isinstance(training_input, DataInput):
      if self.tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided if training input is not tokenized."
        )

      max_prompt_length = self.dpo_config.max_prompt_length
      max_response_length = self.dpo_config.max_response_length
      if (
          self.dpo_config.max_prompt_length is None
          or self.dpo_config.max_response_length is None
      ):
        raise ValueError(
            "max_prompt_length and max_response_length must be provided if "
            "training input is not tokenized. Received: "
            f"max_prompt_length={max_prompt_length}, "
            f"max_response_length={max_response_length}."
        )

      training_input = process_dpo_record(
          record={
              "prompts": training_input.prompts,
              "chosen_responses": training_input.chosen_responses,
              "rejected_responses": training_input.rejected_responses,
          },
          tokenizer=self.tokenizer,
          max_prompt_length=self.dpo_config.max_prompt_length,
          max_response_length=self.dpo_config.max_response_length,
      )

    # Concatenate chosen and rejected IDs so we can do a forward pass together.
    prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids], axis=0
    )
    prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask], axis=0
    )
    completion_ids = jnp.concatenate(
        [training_input.chosen_ids, training_input.rejected_ids], axis=0
    )
    completion_mask = jnp.concatenate(
        [training_input.chosen_mask, training_input.rejected_mask], axis=0
    )
    input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)

    # Compute positions, attention mask, etc., to be fed to the model.
    mask = jnp.concat([prompt_mask, completion_mask], axis=1)
    attention_mask = common.make_causal_attn_mask(mask)
    logits_to_keep = completion_ids.shape[1]
    positions = common.build_positions_from_mask(mask)

    # Compute the log probabilities for the chosen and rejected tokens.
    ref_chosen_logps, ref_rejected_logps = compute_logps(
        self.ref_model,
        input_ids,
        positions,
        attention_mask,
        logits_to_keep,
        completion_mask,
    )
    return TrainExample(
        input_ids=input_ids,
        positions=positions,
        attention_mask=attention_mask,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        completion_mask=completion_mask,
        logits_to_keep=logits_to_keep,
    )

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    m, s = self._mode, self._train_steps
    self.metrics_logger.log("rewards/chosen", aux["rewards/chosen"], m, s)
    self.metrics_logger.log("rewards/rejected", aux["rewards/rejected"], m, s)
    self.metrics_logger.log("rewards/margin", aux["rewards/margin"], m, s)
    self.metrics_logger.log("rewards/accuracy", aux["rewards/accuracy"], m, s)
    self.metrics_logger.log("log_probs/chosen", aux["log_probs/chosen"], m, s)
    self.metrics_logger.log(
        "log_probs/rejected", aux["log_probs/rejected"], m, s
    )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    m, s = self._mode, self._train_steps
    self.metrics_logger.log("rewards/chosen", aux["rewards/chosen"], m, s)
    self.metrics_logger.log("rewards/rejected", aux["rewards/rejected"], m, s)
    self.metrics_logger.log("rewards/margin", aux["rewards/margin"], m, s)
    self.metrics_logger.log("rewards/accuracy", aux["rewards/accuracy"], m, s)
    self.metrics_logger.log("log_probs/chosen", aux["log_probs/chosen"], m, s)
    self.metrics_logger.log(
        "log_probs/rejected", aux["log_probs/rejected"], m, s
    )


def dpo_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    beta: float,
    label_smoothing: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """DPO loss function."""
  chosen_logps, rejected_logps = compute_logps(
      model,
      train_example.input_ids,
      train_example.positions,
      train_example.attention_mask,
      train_example.logits_to_keep,
      train_example.completion_mask,
  )

  chosen_rewards = chosen_logps - train_example.ref_chosen_logps
  rejected_rewards = rejected_logps - train_example.ref_rejected_logps
  margin = chosen_rewards - rejected_rewards

  losses = (
      -jax.nn.log_sigmoid(beta * margin) * (1 - label_smoothing)
      - jax.nn.log_sigmoid(-beta * margin) * label_smoothing
  )

  aux = {
      "rewards/chosen": chosen_rewards.mean(),
      "rewards/rejected": rejected_rewards.mean(),
      "rewards/margin": margin.mean(),
      "rewards/accuracy": (chosen_rewards > rejected_rewards).mean(),
      "log_probs/chosen": chosen_logps.mean(),
      "log_probs/rejected": rejected_logps.mean(),
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
  input_ids = jnp.array(bos_tok + input_ids, dtype=jnp.int32)
  return input_ids


def _preprocess_dict(
    training_input: dict[str, Any],
) -> DataInput | TrainingInput:
  """Wraps input dict with either DataInput or TrainingInput."""

  training_input_fields = [
      field.name for field in dataclasses.fields(DataInput)
  ]
  tokenized_input_fields = [
      field.name for field in dataclasses.fields(TrainingInput)
  ]

  # If the dict contains tokenized fields, we should wrap it with
  # TrainingInput.
  if all(field in training_input for field in tokenized_input_fields):
    return TrainingInput(
        **{field: training_input[field] for field in tokenized_input_fields}
    )
  elif all(field in training_input for field in training_input_fields):
    return DataInput(
        **{field: training_input[field] for field in training_input_fields}
    )
  else:
    raise ValueError(
        "Training input must contain either tokenized fields "
        f"({training_input_fields}) or raw string fields "
        f"({training_input_fields}). Received: {training_input.keys()}."
    )


def process_dpo_record(
    record: dict[str, str | list[str]],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
) -> TrainingInput:
  """Processes and tokenizes a single record for DPO training.

  This function takes a dictionary containing a prompt, a chosen response,
  and a rejected response. It tokenizes each text field and creates the
  corresponding attention masks.

  Note: We use a dictionary here, to make it easier to use on any Grain dataset
  with `.map`.

  Args:
      record: A dictionary, containing "prompts", "chosen_responses", and
        "rejected_responses" as keys. The values can be a single string or a
        list of strings.
      tokenizer: The tokenizer to use for converting text into token IDs.
      max_prompt_length: The maximum length for the tokenized prompts. Any
        sequence longer than this will be truncated.
      max_response_length: The maximum length for the tokenized responses. Any
        sequence longer than this will be truncated.

  Returns:
      A `TrainingInput` object.
  """

  prompts = record["prompts"]
  chosen_responses = record["chosen_responses"]
  rejected_responses = record["rejected_responses"]

  unbatched = isinstance(prompts, str)

  if unbatched:
    prompts = [prompts]
  if isinstance(chosen_responses, str):
    chosen_responses = [chosen_responses]
  if isinstance(rejected_responses, str):
    rejected_responses = [rejected_responses]

  # Only prompt is left padded, others are right padded.
  prompt_ids, prompt_mask = _generate_ids_and_masks(
      prompts,
      tokenizer,
      max_prompt_length,
      left_pad=True,
  )
  chosen_ids, chosen_mask = _generate_ids_and_masks(
      chosen_responses, tokenizer, max_response_length, left_pad=False
  )
  rejected_ids, rejected_mask = _generate_ids_and_masks(
      rejected_responses, tokenizer, max_response_length, left_pad=False
  )

  if unbatched:
    prompt_ids = jnp.squeeze(prompt_ids, axis=0)
    chosen_ids = jnp.squeeze(chosen_ids, axis=0)
    rejected_ids = jnp.squeeze(rejected_ids, axis=0)
    prompt_mask = jnp.squeeze(prompt_mask, axis=0)
    chosen_mask = jnp.squeeze(chosen_mask, axis=0)
    rejected_mask = jnp.squeeze(rejected_mask, axis=0)

  return TrainingInput(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      chosen_ids=chosen_ids,
      chosen_mask=chosen_mask,
      rejected_ids=rejected_ids,
      rejected_mask=rejected_mask,
  )

DpoTrainingConfig = DPOTrainingConfig
DpoTrainer = DPOTrainer
