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
from __future__ import annotations

import dataclasses
from typing import Any
import functools

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
class VLMTrainingInput:
    prompt_ids: jax.Array | np.ndarray      # [B, P]
    prompt_mask: jax.Array | np.ndarray     # [B, P]
    chosen_ids: jax.Array | np.ndarray      # [B, A]
    chosen_mask: jax.Array | np.ndarray     # [B, A]
    rejected_ids: jax.Array | np.ndarray    # [B, A]
    rejected_mask: jax.Array | np.ndarray   # [B, A]
    pixel_values: jax.Array | np.ndarray    # [B, H, W, C]


@flax.struct.dataclass(frozen=True)
class VLMTrainExample:
    input_ids: jax.Array           # [2B, P+A]
    positions: jax.Array           # [2B, P+A]
    attention_mask: jax.Array      # [2B, P+A, P+A]
    completion_mask: jax.Array     # [2B, A]
    pixel_values: jax.Array        # [2B, H, W, C]
    # NEW:
    ref_chosen_logps: jax.Array    # [B]
    ref_rejected_logps: jax.Array  # [B]


@dataclasses.dataclass(slots=True, kw_only=True)
class VlmDpoTrainingConfig(peft_trainer.TrainingConfig):
    beta: float = 0.1
    label_smoothing: float = 0.0
    padding_value: int = 0  # Padding value from tokenizer, default to 0.


@nnx.jit
def compute_logps_vlm(
    model,
    input_ids,
    positions,
    attention_mask,
    completion_mask,
    pixel_values,
):
    # derive dynamically
    logits_to_keep = completion_mask.shape[-1]

    token_logps = common.get_per_token_logps(
        model,
        input_tokens=input_ids,
        positions=positions,
        attn_mask=attention_mask,
        logits_to_keep=logits_to_keep,
        pixel_values=pixel_values,
    )
    token_logps = (token_logps * completion_mask).sum(axis=-1)

    b = token_logps.shape[0]
    return token_logps[: b // 2], token_logps[b // 2 :]


def vl_dpo_loss_fn(
    model: nnx.Module,
    train_example: VLMTrainExample,
    beta: float,
    label_smoothing: float,
):
    # policy logps
    pol_ch, pol_rj = compute_logps_vlm(
        model,
        train_example.input_ids,
        train_example.positions,
        train_example.attention_mask,
        train_example.completion_mask,
        train_example.pixel_values,
    )

    # reference logps were precomputed in _prepare_inputs and are just arrays
    ref_ch = train_example.ref_chosen_logps
    ref_rj = train_example.ref_rejected_logps

    chosen_rewards   = pol_ch - ref_ch
    rejected_rewards = pol_rj - ref_rj
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




class VLM_DpoTrainer(peft_trainer.PeftTrainer):
    """VLM DPO Trainer."""

    def __init__(
        self,
        model: nnx.Module,
        ref_model: nnx.Module,
        optimizer: optax.GradientTransformation,
        training_config: VlmDpoTrainingConfig,
    ):
        self.model = model
        self.ref_model = ref_model
        super().__init__(model, optimizer, training_config)
        self.dpo_config = training_config

        # Single loss function that runs both policy & ref in one jitted graph
        self.loss_fn = vl_dpo_loss_fn
        self.gen_model_input_fn = lambda x: {
            "train_example": x,
            "beta": self.dpo_config.beta,
            "label_smoothing": self.dpo_config.label_smoothing,
        }
        self._has_aux = True

    @override
    def _prepare_inputs(self, ti: VLMTrainingInput) -> Any:
        prompt_ids  = jnp.concatenate([ti.prompt_ids,  ti.prompt_ids ])
        prompt_mask = jnp.concatenate([ti.prompt_mask, ti.prompt_mask])

        max_len = max(ti.chosen_ids.shape[1], ti.rejected_ids.shape[1])
        pad_val = self.dpo_config.padding_value
        completion_ids = jnp.concatenate([
            common.pad_to_length(ti.chosen_ids,   max_len, pad_val, axis=-1),
            common.pad_to_length(ti.rejected_ids, max_len, pad_val, axis=-1),
        ])
        completion_mask = jnp.concatenate([
            common.pad_to_length(ti.chosen_mask,   max_len, 0, axis=-1),
            common.pad_to_length(ti.rejected_mask, max_len, 0, axis=-1),
        ])

        input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
        full_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
        attention_mask = common.make_causal_attn_mask(full_mask)
        positions = common.build_positions_from_mask(full_mask)
        pixel_values = jnp.concatenate([ti.pixel_values, ti.pixel_values], axis=0)

        # Compute REFERENCE log-probs *outside* the loss/grad path
        ref_chosen_logps, ref_rejected_logps = compute_logps_vlm(
            self.ref_model,
            input_ids,
            positions,
            attention_mask,
            completion_mask,
            pixel_values,
        )

        return VLMTrainExample(
            input_ids=input_ids,
            positions=positions,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            pixel_values=pixel_values,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

    @override
    def _post_process_train_step(self, aux: Any) -> None:
        m, s = self._mode, self._train_steps
        self.metrics_logger.log("chosen_rewards",   float(aux["chosen_rewards"]),   m, s)
        self.metrics_logger.log("rejected_rewards", float(aux["rejected_rewards"]), m, s)
        self.metrics_logger.log("rewards_margin",   float(aux["rewards_margin"]),   m, s)
        self.metrics_logger.log("rewards_accuracy", float(aux["rewards_accuracy"]), m, s)
        if self._buffered_train_metrics is not None:
            self._buffered_train_metrics.losses = [float(x) for x in self._buffered_train_metrics.losses]


# (Optional) helper if you transform single records
def process_vlm_dpo_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    image_key: str = "pixel_values",
) -> VLMTrainingInput:
    # only prompt is left padded, others are right padded.
    prompt_ids,  prompt_mask  = _generate_ids_and_masks([record["prompt"]],  tokenizer, max_seq_length)
    chosen_ids,  chosen_mask  = _generate_ids_and_masks([record["chosen"]],  tokenizer, max_seq_length, left_pad=False)
    rejected_ids,rejected_mask= _generate_ids_and_masks([record["rejected"]],tokenizer, max_seq_length, left_pad=False)

    pixel_values = record[image_key]
    if pixel_values.ndim == 3:
        pixel_values = pixel_values[None, ...]  # [1,H,W,C]

    return VLMTrainingInput(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        chosen_ids=chosen_ids,
        chosen_mask=chosen_mask,
        rejected_ids=rejected_ids,
        rejected_mask=rejected_mask,
        pixel_values=pixel_values,
    )