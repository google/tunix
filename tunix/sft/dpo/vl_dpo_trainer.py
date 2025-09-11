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
from tunix.rl import common
from tunix.sft import peft_trainer
from typing_extensions import override


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  prompt_ids: jax.Array | np.ndarray  # Prompt ids should be left padded.
  prompt_mask: jax.Array | np.ndarray
  chosen_ids: jax.Array | np.ndarray  # Chosen ids should be right padded.
  chosen_mask: jax.Array | np.ndarray
  rejected_ids: jax.Array | np.ndarray  # Rejected ids should be right padded.
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


def _generate_ids_and_masks(
    input_strings: list[str],
    tokenizer: Any,
    max_length: int,
    left_pad: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Generates ids and masks for a list of strings."""
  tokens = [tokenizer.tokenize(x) for x in input_strings]
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
  return all_input_ids[0], all_input_mask[0]


def process_dpo_record(
    record: dict[str, Any], tokenizer: Any, max_seq_length: int
) -> TrainingInput:
  """Processes and tokenizes a single record for DPO training.

  This function takes a dictionary containing a prompt, a chosen response,
  and a rejected response. It tokenizes each text field into ids and creates
  the corresponding attention masks.

  Args:
      record: A dictionary containing the training data. Expected to have
        'prompt', 'chosen', and 'rejected' keys, each with a string value.
      tokenizer: The tokenizer to use for converting text into token IDs.
      max_seq_length: The maximum length for the tokenized sequences. Any
        sequence longer than this will be truncated.

  Returns:
      A `TrainingInput` object
  """

  # only prompt is left padded, others are right padded.
  prompt_ids, prompt_mask = _generate_ids_and_masks(
      [record["prompt"]], tokenizer, max_seq_length
  )
  chosen_ids, chosen_mask = _generate_ids_and_masks(
      [record["chosen"]], tokenizer, max_seq_length, left_pad=False
  )
  rejected_ids, rejected_mask = _generate_ids_and_masks(
      [record["rejected"]], tokenizer, max_seq_length, left_pad=False
  )

  # Ensure the shapes are correct
  assert prompt_ids.shape == chosen_ids.shape == rejected_ids.shape
  assert prompt_mask.shape == chosen_mask.shape == rejected_mask.shape

  return TrainingInput(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      chosen_ids=chosen_ids,
      chosen_mask=chosen_mask,
      rejected_ids=rejected_ids,
      rejected_mask=rejected_mask,
  )


@dataclasses.dataclass(slots=True, kw_only=True)
class DpoTrainingConfig(peft_trainer.TrainingConfig):
  beta: float = 0.1  # 𝛽 for KL penalty https://arxiv.org/pdf/2305.18290
  label_smoothing: float = 0.0
  padding_value: int = 0  # Padding value from tokenizer, default to 0.


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


class DpoTrainer(peft_trainer.PeftTrainer):
  """DPO trainer."""

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: DpoTrainingConfig,
  ):
    self.model = model
    self.ref_model = ref_model
    super(DpoTrainer, self).__init__(model, optimizer, training_config)
    self.dpo_config = training_config
    self.loss_fn = dpo_loss_fn
    self.gen_model_input_fn = lambda x: {
        "train_example": x,
        "beta": self.dpo_config.beta,
        "label_smoothing": self.dpo_config.label_smoothing,
    }
    self._has_aux = True

  @override
  def _prepare_inputs(self, training_input: TrainingInput) -> Any:
    # Concat chosen and rejected ids so we can compute together.
    prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids]
    )
    prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask]
    )
    max_len = max(
        training_input.chosen_ids.shape[1],
        training_input.rejected_ids.shape[1],
    )
    pad_value = self.dpo_config.padding_value
    completion_ids = jnp.concatenate([
        common.pad_to_length(
            training_input.chosen_ids, max_len, pad_value, axis=-1
        ),
        common.pad_to_length(
            training_input.rejected_ids, max_len, pad_value, axis=-1
        ),
    ])
    completion_mask = jnp.concatenate([
        common.pad_to_length(training_input.chosen_mask, max_len, 0, axis=-1),
        common.pad_to_length(training_input.rejected_mask, max_len, 0, axis=-1),
    ])

    input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)
    attention_mask = common.make_causal_attn_mask(
        jnp.concat([prompt_mask, completion_mask], axis=1)
    )
    logits_to_keep = completion_ids.shape[1]
    positions = common.build_positions_from_mask(
        jnp.concat([prompt_mask, completion_mask], axis=1)
    )

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
    self.metrics_logger.log("chosen_rewards", aux["chosen_rewards"], m, s)
    self.metrics_logger.log("rejected_rewards", aux["rejected_rewards"], m, s)
    self.metrics_logger.log("rewards_margin", aux["rewards_margin"], m, s)
    self.metrics_logger.log("rewards_accuracy", aux["rewards_accuracy"], m, s)

  # TODO(tsbao): override _post_process_eval_step once eval step is properly
  # tracked.


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
      "chosen_rewards": chosen_rewards.mean(),
      "rejected_rewards": rejected_rewards.mean(),
      "rewards_margin": margin.mean(),
      "rewards_accuracy": (chosen_rewards > rejected_rewards).mean(),
  }

  return losses.mean(), aux
