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
"""Proximal Policy Optimization trainer."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Iterable, Iterator, List

import flax
from flax import nnx
import gc
import jax
import jax.numpy as jnp
import optax

from tunix.generate import sampler as sampler_lib
from tunix.rl import common
from tunix.rl.grpo import grpo_trainer  # Reuse helper functions
from tunix.sft import peft_trainer
from typing_extensions import override

# Single reward fn taking prompts and completions -> rewards
RewardFn = Callable[[List[str], List[str]], List[float]]


class RepeatTrainingInputIter:
  """Repeat batches for gradient accumulation."""

  def __init__(
      self,
      data: Iterable[dict[str, Any]],
      gradient_accumulation_steps: int = 1,
  ) -> None:
    if gradient_accumulation_steps <= 0:
      raise ValueError(
          f"gradient accumulation steps must be positive: {gradient_accumulation_steps}"
      )
    self._data = data
    self._gas = gradient_accumulation_steps
    self._data_buffer = [None] * self._gas
    self._itr_cnt = 0

  def __iter__(self) -> "RepeatTrainingInputIter":
    self._iterator = iter(self._data)
    return self

  def __next__(self) -> dict[str, Any]:
    if self._itr_cnt % self._gas == 0:
      self._data_buffer[self._itr_cnt % self._gas] = next(self._iterator)
    res = self._data_buffer[self._itr_cnt % self._gas]
    self._itr_cnt += 1
    return res


class EpochBatchIter:
  """Iterates over PPO epochs and mini-batches."""

  def __init__(
      self,
      trainer: "PpoTrainer",
      data: Iterable[dict[str, Any]],
      num_epochs: int,
      num_mini_batches: int,
  ) -> None:
    self._trainer = trainer
    self._data = data
    self._epochs = num_epochs
    self._mbs = num_mini_batches

  def __iter__(self) -> "EpochBatchIter":
    self._iterator = iter(self._data)
    self._buffer: list[TrainExample] = []
    self._index = 0
    return self

  def _prepare_buffer(self):
    training_input = next(self._iterator)
    ex = self._trainer._build_train_example(training_input)
    batch_size = ex.input_ids.shape[0]
    if batch_size % self._mbs != 0:
      raise ValueError("batch size must be multiple of num_mini_batches")
    mb_size = batch_size // self._mbs
    self._buffer = []
    for _ in range(self._epochs):
      perm = jax.random.permutation(jax.random.PRNGKey(self._index), batch_size)
      perm = jnp.array(perm)
      for i in range(self._mbs):
        idx = perm[i * mb_size : (i + 1) * mb_size]
        mb_ex = jax.tree.map(lambda x: x[idx], ex)
        self._buffer.append(mb_ex)
    self._index = 0

  def __next__(self) -> TrainExample:
    if self._index >= len(self._buffer):
      self._prepare_buffer()
    ex = self._buffer[self._index]
    self._index += 1
    return ex


@flax.struct.dataclass(frozen=True)
class TrainExample:
  input_ids: jax.Array
  positions: jax.Array
  attention_mask: jax.Array
  old_logps: jax.Array
  old_values: jax.Array
  advantages: jax.Array
  returns: jax.Array
  completion_mask: jax.Array
  logits_to_keep: int = flax.struct.field(pytree_node=False)


@dataclasses.dataclass(slots=True, kw_only=True)
class PpoTrainingConfig(peft_trainer.TrainingConfig):
  """Configuration for PPO trainer."""

  total_generation_steps: int
  max_prompt_length: int
  num_ppo_epochs: int = 1
  num_mini_batches: int = 1
  gamma: float = 1.0
  lam: float = 1.0
  cliprange: float = 0.2
  cliprange_value: float = 0.2
  vf_coef: float = 0.5
  kl_coef: float = 0.0
  kl_estimator: str = "k1"
  temperature: float = 1.0
  top_p: float = 1.0
  top_k: int | None = None
  whiten_rewards: bool = False


class PpoTrainer(peft_trainer.PeftTrainer):
  """Simplified PPO trainer implemented in JAX."""

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module,
      value_model: nnx.Module,
      sampler: sampler_lib.Sampler,
      reward_fn: RewardFn,
      optimizer: optax.GradientTransformation,
      training_config: PpoTrainingConfig,
  ) -> None:
    super().__init__(model, optimizer, training_config)
    self.ref_model = ref_model
    self.value_model = value_model
    self.sampler = sampler
    self.reward_fn = reward_fn
    self.ppo_config = training_config
    self.loss_fn = ppo_loss_fn
    self.eval_loss_fn = ppo_loss_fn
    self.gen_model_input_fn = lambda x: {
        "train_example": x,
        "cliprange": self.ppo_config.cliprange,
        "cliprange_value": self.ppo_config.cliprange_value,
        "vf_coef": self.ppo_config.vf_coef,
    }
    self._has_aux = True

    if jax.device_count() > 1:
      devices = jax.devices()
      self.model = jax.device_put_replicated(self.model, devices)
      self.optimizer = jax.device_put_replicated(self.optimizer, devices)
      self.ref_model = jax.device_put_replicated(self.ref_model, devices)
      self.value_model = jax.device_put_replicated(self.value_model, devices)

  def jit_train_and_eval_step(self, skip_jit: bool = False):
    """JIT or PMAP the train and eval steps based on device count."""
    if jax.device_count() > 1 and not skip_jit:
      if self._jitted_train_step_fn is None:
        train_step = self.create_train_step_fn()
        eval_step = self.create_eval_step_fn()
        self._jitted_train_step_fn = jax.pmap(train_step, axis_name="batch")
        self._jitted_eval_step_fn = jax.pmap(eval_step, axis_name="batch")
      return self._jitted_train_step_fn, self._jitted_eval_step_fn
    return super().jit_train_and_eval_step(skip_jit)

  def _build_train_example(self, training_input: dict[str, Any]) -> TrainExample:
    """Generate completions and compute GAE."""
    pad_value = self.sampler.tokenizer.pad_id()
    eos_value = self.sampler.tokenizer.eos_id()

    completion_output = self.sampler(
        input_strings=training_input["prompts"],
        total_generation_steps=self.ppo_config.total_generation_steps,
        max_prompt_length=self.ppo_config.max_prompt_length,
        echo=False,
        temperature=self.ppo_config.temperature,
        top_p=self.ppo_config.top_p,
        top_k=self.ppo_config.top_k,
    )

    completion_ids = grpo_trainer.pad_inputs(
        completion_output.tokens,
        target_length=self.ppo_config.total_generation_steps,
        pad_value=pad_value,
        left=False,
    )
    prompt_ids = completion_output.padded_prompt_tokens

    (
        positions,
        prompt_completion_ids,
        completion_mask,
        _,
        prompt_completion_causal_mask,
    ) = grpo_trainer.process_ids(prompt_ids, completion_ids, pad_value, eos_value)

    logits_to_keep = completion_ids.shape[1]
    old_per_token_logps = common.get_per_token_logps(
        self.model,
        input_tokens=prompt_completion_ids,
        positions=positions,
        attn_mask=prompt_completion_causal_mask,
        logits_to_keep=logits_to_keep,
    )
    ref_per_token_logps = common.get_per_token_logps(
        self.ref_model,
        input_tokens=prompt_completion_ids,
        positions=positions,
        attn_mask=prompt_completion_causal_mask,
        logits_to_keep=logits_to_keep,
    )

    value_logits, _ = self.value_model(
        prompt_completion_ids,
        positions=positions,
        cache=None,
        attention_mask=prompt_completion_causal_mask,
    )
    old_values = value_logits[:, -logits_to_keep - 1 : -1, 0]

    common.clear_memory()

    scores = jnp.array(
        self.reward_fn(training_input["prompts"], completion_output.text)
    )
    logr = ref_per_token_logps - old_per_token_logps
    if self.ppo_config.kl_estimator == "k3":
      kl = jnp.exp(logr) - 1 - logr
    else:
      kl = -logr
    non_score_reward = -self.ppo_config.kl_coef * kl
    rewards = non_score_reward
    seq_lens = completion_mask.sum(axis=1) - 1
    rewards = rewards.at[jnp.arange(rewards.shape[0]), seq_lens].add(scores)
    if self.ppo_config.whiten_rewards:
      rewards = common.masked_whiten(rewards, completion_mask, shift_mean=False)

    advantages, returns = common.generalized_advantage_estimation(
        rewards,
        old_values,
        completion_mask,
        self.ppo_config.gamma,
        self.ppo_config.lam,
    )
    advantages = common.masked_whiten(advantages, completion_mask)

    ex = TrainExample(
        input_ids=prompt_completion_ids,
        positions=positions,
        attention_mask=prompt_completion_causal_mask,
        old_logps=old_per_token_logps,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        completion_mask=completion_mask,
        logits_to_keep=logits_to_keep,
    )
    common.clear_memory()
    return ex

  @override
  def _prepare_inputs(self, training_input: dict[str, Any]) -> TrainExample:
    return self._build_train_example(training_input)

  @override
  def train(
      self,
      train_ds: Iterable[dict[str, Any]],
      eval_ds: Iterable[dict[str, Any]] | None = None,
      skip_jit: bool = False,
  ) -> None:
    epochs = self.ppo_config.num_ppo_epochs
    mbs = self.ppo_config.num_mini_batches
    dataset = EpochBatchIter(self, train_ds, epochs, mbs)
    gas = self.ppo_config.get_with_default("gradient_accumulation_steps", 1)
    if gas > 1:
      dataset = RepeatTrainingInputIter(dataset, gas)
    self._eval_ds_raw = eval_ds
    if eval_ds is not None:
      eval_ds = (self._build_train_example(d) for d in eval_ds)
    orig_prepare = self._prepare_inputs
    self._prepare_inputs = lambda x: x
    try:
      super().train(dataset, eval_ds, skip_jit)
    finally:
      self._prepare_inputs = orig_prepare

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    for key, val in aux.items():
      self.metrics_logger.log(key, val, self._mode, self._train_steps)
    common.clear_memory()

  @override
  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step: Callable[..., Any],
  ) -> None:
    super()._run_eval(eval_ds, eval_step)
    if self._eval_ds_raw is not None:
      self.generate_completions(self._eval_ds_raw)

  def generate_completions(
      self, eval_ds: Iterable[dict[str, Any]], num_batches: int = 1
  ) -> None:
    """Generate and log sample completions from eval dataset."""
    for i, batch in enumerate(eval_ds):
      prompts = batch["prompts"]
      out = self.sampler(
          input_strings=prompts,
          total_generation_steps=self.ppo_config.total_generation_steps,
          max_prompt_length=self.ppo_config.max_prompt_length,
          echo=False,
          temperature=0.01,
          top_p=1.0,
          top_k=None,
      )
      for prompt, resp in zip(prompts, out.text):
        logging.info("Prompt: %s\nCompletion: %s", prompt, resp)
      if i + 1 >= num_batches:
        break
    common.clear_memory()

  def save_checkpoint(self) -> None:
    """Save current model state."""
    self.checkpoint_manager.save(
        self._train_steps,
        self.model,
        save_only_lora_params=self._lora_enabled,
        force=True,
    )


@nnx.jit(static_argnums=(3,))
def ppo_loss_fn(model, train_example, cliprange, cliprange_value, vf_coef):
  """Computes PPO loss."""
  logits, _ = model(
      train_example.input_ids,
      positions=train_example.positions,
      cache=None,
      attention_mask=train_example.attention_mask,
  )
  logits = logits[:, -train_example.logits_to_keep - 1 : -1, :]
  new_logps = common.selective_log_softmax(
      logits, train_example.input_ids[:, -train_example.logits_to_keep :]
  )
  new_logps = new_logps * train_example.completion_mask

  ratio = jnp.exp(new_logps - train_example.old_logps)
  adv = train_example.advantages
  pg_losses = -adv * ratio
  pg_losses2 = -adv * jnp.clip(ratio, 1 - cliprange, 1 + cliprange)
  pg_loss = jnp.mean(jnp.maximum(pg_losses, pg_losses2))
  pg_clipfrac = jnp.mean((pg_losses2 > pg_losses).astype(jnp.float32))

  log_prob = jax.nn.log_softmax(logits)
  prob = jnp.exp(log_prob)
  token_entropy = -jnp.sum(prob * log_prob, axis=-1)
  entropy = common.masked_mean(token_entropy, train_example.completion_mask)
  ratio_mean = common.masked_mean(ratio, train_example.completion_mask)

  value_logits, _ = model(
      train_example.input_ids,
      positions=train_example.positions,
      cache=None,
      attention_mask=train_example.attention_mask,
  )
  values = value_logits[:, -train_example.logits_to_keep - 1 : -1, 0]
  vpred_clipped = train_example.old_values + jnp.clip(
      values - train_example.old_values, -cliprange_value, cliprange_value
  )
  vf_losses1 = (values - train_example.returns) ** 2
  vf_losses2 = (vpred_clipped - train_example.returns) ** 2
  vf_loss = 0.5 * jnp.mean(jnp.maximum(vf_losses1, vf_losses2))
  vf_clipfrac = jnp.mean((vf_losses2 > vf_losses1).astype(jnp.float32))

  loss = pg_loss + vf_coef * vf_loss
  kl = jnp.mean((train_example.old_logps - new_logps))
  approx_kl = 0.5 * jnp.mean((new_logps - train_example.old_logps) ** 2)
  return loss, {
      "kl": kl,
      "approx_kl": approx_kl,
      "policy_loss": pg_loss,
      "value_loss": vf_loss,
      "pg_clipfrac": pg_clipfrac,
      "vf_clipfrac": vf_clipfrac,
      "entropy": entropy,
      "ratio": ratio_mean,
  }
