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

"""PPO learner."""

from __future__ import annotations

from concurrent import futures
import dataclasses
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.ppo import ppo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import metrics_logger

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  returns: jax.Array
  old_values: jax.Array
  completion_plus_one_mask: jax.Array


@dataclasses.dataclass(slots=True, kw_only=True)
class PpoConfig:
  """Configuration for PPO learner.

  Attributes:
    num_ppo_epochs: The number of optimization epochs per batch of rollouts.
    mini_batch_size: The batch size on which the actual model updates happen.
      The rollout phase (`generate_and_compute_advantages`) happen on a larger
      batch, which is then split into "mini-batches".
    gamma: The discount factor for future rewards in GAE.
    gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
    beta: The coefficient for the KL divergence penalty.
    epsilon: Epsilon value for clipping the policy objective.
    vf_coef: The coefficient for the value function loss.
    clip_range_value: The range for clipping the value function loss.
  """

  num_ppo_epochs: int = 4
  mini_batch_size: int | None = None

  # PPO loss and advantage computation configs.
  gamma: float = 1.0
  gae_lambda: float = 0.95
  beta: float = 0.04
  epsilon: float = 0.2
  vf_coef: float = 0.1
  clip_range_value: float = 0.2


class PpoLearner:
  """PPO (Proximal Policy Optimization) learner.

  PPO is a reinforcement learning algorithm that fine-tunes models using an
  actor-critic architecture. It optimizes a clipped surrogate objective function
  to ensure stable policy updates, preventing large, destructive changes. The
  actor (policy model) learns what actions to take, while the critic (value
  model) estimates the value of states to help calculate advantages. This
  approach balances exploration and exploitation, making it a robust choice for
  a wide range of RL tasks.

  References:
  - https://arxiv.org/abs/1707.06347
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      ppo_config: PpoConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `PpoLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      ppo_config: An instance of `PpoConfig` containing all training-specific
        configuration options.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      data_shuffle_seed: The seed for shuffling the data.
    """
    self.rl_cluster = rl_cluster
    self.ppo_config = ppo_config
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )

    # ===== RlCluster should have `reward` and `critic` models =====
    if bool(reward_fns) == bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    ):
      raise ValueError(
          "PPO requires one of `reward_fns` or `rl_cluster.reward` to be set. "
          f"Received: reward_fn={reward_fns}, "
          "rl_cluster.reward="
          f"{self.rl_cluster.inference_worker._models['reward']}"
      )
    if not self.rl_cluster.inference_worker._models["critic"]:
      raise ValueError(
          "PPO requires a critic model. Please pass the correct `critic` to "
          "`RlCluster`."
      )
    self._use_reward_model = bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    )

    # ===== Configure the actor (policy) trainer =====
    self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "epsilon": self.ppo_config.epsilon,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )
    self.rl_cluster.actor_trainer.is_managed_externally = True

    # ===== Configure the critic (value) trainer =====
    self.rl_cluster.critic_trainer.with_loss_fn(ppo_value_loss_fn, has_aux=True)
    self.rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": self.ppo_config.clip_range_value,
            "vf_coef": self.ppo_config.vf_coef,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )
    self.rl_cluster.critic_trainer.is_managed_externally = True

    # ===== Configure the metrics logger =====
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "score/mean",
        "reward/mean",
        lambda: "kl" if self.ppo_config.beta != 0.0 else None,
    ])

    # TODO(abheesht): Are we okay with two TQDM bars and two metric loggers?
    self.rl_cluster.critic_trainer.with_rl_metrics_to_log(
        {"loss/vf": "vf_loss"}
    )
    self.rl_cluster.critic_trainer.with_tqdm_metrics_to_display([
        "loss/vf",
    ])

    self._actor_metrics_logger = rl_cluster.actor_trainer.metrics_logger
    self._critic_metrics_logger = rl_cluster.critic_trainer.metrics_logger

    self.grad_acc_steps = (
        rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

    self._data_shuffle_key = data_shuffle_seed
    if data_shuffle_seed is None:
      self._data_shuffle_key = 0
    self._data_shuffle_key = jax.random.PRNGKey(self._data_shuffle_key)

    self._train_steps = 0
    self._eval_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        utils.is_sharing_weights(
            self.rl_cluster.actor_trainer.model,
            self.rl_cluster.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        != self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self._last_train_step = self.rl_cluster.actor_trainer.train_steps

  def _get_metric_logging_steps(self, mode: metrics_logger.Mode) -> int:
    return (
        self._train_steps
        if mode == metrics_logger.Mode.TRAIN
        else self._eval_steps
    )

  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> TrainExample:
    """Generates completions and computes advantages for PPO training.

    Args:
      training_input: A dictionary containing the training input data, including
        the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data for PPO.
    """
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    if isinstance(self.rl_cluster.cluster_config.rollout_config, dict):
      rollout_config = self.rl_cluster.cluster_config.rollout_config[
          rl_cluster_lib.Mode[mode.name]
      ]
    else:
      rollout_config = self.rl_cluster.cluster_config.rollout_config
    max_prompt_length = rollout_config.max_prompt_length

    # ===== Generation ======
    # Generate. We use `model`, i.e., the policy model for generating the
    # "experiences".
    completion_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
    )
    completion_ids = completion_output.tokens
    prompt_ids = completion_output.left_padded_prompt_tokens

    batch_size = completion_ids.shape[0]
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    )
    is_padding_token = jnp.any(~completion_mask, axis=-1)
    completion_plus_one_mask = completion_mask.at[
        jnp.arange(batch_size)[is_padding_token],
        (eos_idx + 1)[is_padding_token],
    ].set(True)

    # ===== Compute log probs ======
    logits_to_keep = completion_ids.shape[1]
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
      # Set log probs to 1 for padding tokens.
      ref_per_token_logps = jnp.where(
          completion_mask,
          ref_per_token_logps,
          jnp.array(1).astype(ref_per_token_logps.dtype),
      )
    else:
      ref_per_token_logps = None

    # Get log probs from the policy before model weights are updated. We use
    # the policy model here. Shape = `[B, T]`.
    # TODO(abheesht): Do we do this only when `self.num_ppo_epochs > 1`? Don't
    # see this condition here:
    # https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L1224-L1233.
    old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
    )
    # Set log probs to 1 for padding tokens.
    old_per_token_logps = jnp.where(
        completion_mask,
        old_per_token_logps,
        jnp.array(1).astype(old_per_token_logps.dtype),
    )

    # ===== Value computation ======
    # Get values from the value model before model weights are updated.
    values = self.rl_cluster.get_values(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
    )
    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    # Set `values` corresponding to padding tokens to 0.
    values = jnp.where(
        completion_plus_one_mask,
        values,
        jnp.array(0).astype(values.dtype),
    )

    # ===== Reward computation ======
    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
      # We use the score corresponding to the last non-padding token.
      # Remember, completion is padded on the right, and the prompt is padded on
      # the left.
      last_token_scores = scores[
          jnp.arange(batch_size),
          eos_idx + max_prompt_length,
      ]
    else:
      last_token_scores = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=completion_output.text,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )

    # This is how rewards are computed. This is in accordance with TRL and
    # with verl's `NaiveRewardManager`. This is a different from GRPO, where
    # we don't consider rewards at the token level.
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. Subtract KL divergence from the reward tensor of all 0s.
    # 3. A positive reward is given only at the final timestep, so we add that
    # to the reward tensor from (2).
    rewards = jnp.zeros_like(completion_ids)
    if self.ppo_config.beta != 0.0:
      kl = common.compute_kl_divergence(
          old_per_token_logps, ref_per_token_logps
      )
      rewards = rewards - self.ppo_config.beta * kl

    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)

    # ===== Metric logging ======
    # TODO(abheesht): Verify metric logging. We should move these to losses,
    # because the rollout batch can be split into mini-batches.
    step = self._get_metric_logging_steps(mode)

    # Log raw scores from the reward model/fn
    self._actor_metrics_logger.log(
        "score/mean", last_token_scores.mean(), mode, step
    )
    self._actor_metrics_logger.log(
        "score/max", last_token_scores.max(), mode, step
    )
    self._actor_metrics_logger.log(
        "score/min", last_token_scores.min(), mode, step
    )

    # Log final rewards (scores + KL penalty)
    sequence_rewards = rewards.sum(-1)
    self._actor_metrics_logger.log(
        "reward/mean", sequence_rewards.mean(), mode, step
    )
    self._actor_metrics_logger.log(
        "reward/max", sequence_rewards.max(), mode, step
    )
    self._actor_metrics_logger.log(
        "reward/min", sequence_rewards.min(), mode, step
    )
    if self.ppo_config.beta != 0.0:
      # Average of the per-sequence mean KL
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
      )
      self._actor_metrics_logger.log(
          "kl/mean", per_sequence_mean_kl.mean(), mode, step
      )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self._actor_metrics_logger.log(
        "completions/mean_length",
        agg_completion_mask.mean(),
        mode,
        self._get_metric_logging_steps(mode),
    )
    self._actor_metrics_logger.log(
        "completions/max_length",
        agg_completion_mask.max(),
        mode,
        self._get_metric_logging_steps(mode),
    )
    self._actor_metrics_logger.log(
        "completions/min_length",
        agg_completion_mask.min(),
        mode,
        self._get_metric_logging_steps(mode),
    )

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantages, returns = ppo_helpers.compute_gae_advantages(
        rewards=rewards,
        values=values,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )
    # Normalize advantages.
    advantages = ppo_helpers.normalize_advantages(advantages, completion_mask)

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_plus_one_mask=completion_plus_one_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_per_token_logps=old_per_token_logps,
        old_values=values,
    )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **kargs,
  ):
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      **kargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts,]`) of scalar rewards for
      each prompt-completion pair. The rewards are computed using the provided
      reward functions (their mean).
    """
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)

    # Take the mean of rewards, because we need one score per element.
    rewards = jnp.nanmean(rewards, axis=1)
    return rewards

  def _prepare_data(
      self,
      iterator: Iterator[_TrainingInputT],
      mini_batch_size: int | None,
      proceed_num_steps: int,
      batch_repeat: int,
      data_queue: queue_lib.AbstractDataQueue[
          list[TrainExample] | common.RepeatIterable | None
      ],
      async_loading: bool = False,
      shuffle_data: bool = False,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> None:
    """Prepares the dataset for training.

    Args:
      iterator: The input iterator of the dataset.
      mini_batch_size: The mini batch size which will be used to slice the
        dataset.
      proceed_num_steps: The number of steps to proceed for the iterator if set
        to a positive number. If it's set to a non positive number, the function
        will exhaust the iterator. If the input iterator is exhausted before the
        number of steps is reached, the function will return the empty result.
      batch_repeat: The number of times to repeat the batch in the final
        dataset.
      data_queue: The data queue to use for putting the examples into.
      async_loading: Whether to load the batch asynchronously, if not async
        loading, then all the examples needed will be processed and then loaded
        into the data queue.
      shuffle_data: Whether to shuffle the data.
      mode: The mode to use for logging metrics.

    Returns:
      None. Examples are put into the data queue.
    """
    example_list = []

    def _put_list_of_examples_to_data_queue():
      if shuffle_data:
        self._data_shuffle_key, _ = jax.random.split(self._data_shuffle_key)

      if not async_loading:
        data_queue.put(
            common.RepeatIterable(
                example_list,
                repeat=batch_repeat,
                mini_batch_size=mini_batch_size,
                shuffle=shuffle_data,
                key=self._data_shuffle_key if shuffle_data else None,
            )
        )
      elif batch_repeat > 1:
        # Since we have already loaded the batch in data_queue once, we only
        # need to repeat batch_repeat - 1 times.
        data_queue.put(
            common.RepeatIterable(
                example_list,
                repeat=batch_repeat - 1,
                mini_batch_size=mini_batch_size,
                shuffle=shuffle_data,
                key=self._data_shuffle_key if shuffle_data else None,
            )
        )

    while True:
      try:
        while (
            mode == metrics_logger.Mode.TRAIN
            and self._train_steps < self._last_train_step
        ):
          next(iterator)
          self._train_steps += 1
        example = next(iterator)

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._train_steps
            if mode == metrics_logger.Mode.TRAIN
            else self._eval_steps,
        ):
          advantage = self._generate_and_compute_advantage(example, mode)
        if async_loading:
          data_queue.put([advantage])

        if mode == metrics_logger.Mode.TRAIN:
          self._train_steps += 1
        else:
          self._eval_steps += 1
        example_list.append(advantage)
        if proceed_num_steps > 0 and len(example_list) == proceed_num_steps:
          _put_list_of_examples_to_data_queue()
          data_queue.put(None)
          return
      except StopIteration as e:
        if proceed_num_steps > 0:
          data_queue.put(None)
          raise e
        else:
          _put_list_of_examples_to_data_queue()
          data_queue.put(None)
          return
      except Exception as e:
        data_queue.put(None)
        raise e

  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """PPO training loop."""
    train_iterator = iter(train_ds)
    while True:  # loop over M
      try:
        # reserve 1 for None and the other for repeated interable
        # if batch_repeat > 1
        train_data_queue = queue_lib.SimpleDataQueue(
            maxsize=self.grad_acc_steps + 2
        )
        # reserve 1 for None
        eval_data_queue = queue_lib.SimpleDataQueue(maxsize=2)
        initial_train_steps = self._train_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            mini_batch_size=self.ppo_config.mini_batch_size,
            proceed_num_steps=self.grad_acc_steps,
            batch_repeat=self.ppo_config.num_ppo_epochs,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=metrics_logger.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_train_steps
        ):
          while True:
            curr_train_ds = train_data_queue.get(block=True)
            if curr_train_ds is None:
              break
            if (
                eval_ds
                and not curr_eval_ds
                and self.rl_cluster.actor_trainer.train_steps
                % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                == 0
            ):
              self._prepare_data(
                  iterator=iter(eval_ds),
                  mini_batch_size=None,
                  proceed_num_steps=-1,
                  batch_repeat=1,
                  data_queue=eval_data_queue,
                  async_loading=False,
                  mode=metrics_logger.Mode.EVAL,
              )
              curr_eval_ds = eval_data_queue.get(block=True)
            self.rl_cluster.update_actor(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
            self.rl_cluster.update_critic(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
        # call to throw stop iteration as a signal to break the loop
        future.result()
        # Sync the train steps with internal trainer, this is based on the
        # assumption that the trainer internally doesn't reset the train steps.
        self._train_steps = self.rl_cluster.actor_trainer.train_steps
        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_train_steps
          ):
            self.rl_cluster.sync_weights()
        if (
            self._train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    self.rl_cluster.actor_trainer.close()
    self.rl_cluster.critic_trainer.close()


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    vf_coef: float,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_plus_one_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_plus_one_mask,
  )
  logits_to_keep = completion_ids.shape[1]

  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns
  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]
  vpreds = jnp.where(
      completion_plus_one_mask,
      vpreds,
      jnp.array(0).astype(vpreds.dtype),
  )
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = ppo_helpers.masked_mean(
      jnp.maximum(vf_losses1, vf_losses2), completion_plus_one_mask
  )
  vf_loss = 0.5 * clipped_vf_losses

  aux = {"vf_loss": vf_loss}
  return vf_coef * vf_loss, aux


def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon: float,
    pad_id: int,
    eos_id: int,
):
  """Computes the policy loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # Get log probs.
  per_token_logps = common.compute_per_token_logps(
      model,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
  )
  # Set log probs to 1 for padding tokens.
  per_token_logps = jnp.where(
      completion_mask,
      per_token_logps,
      jnp.array(1).astype(per_token_logps.dtype),
  )

  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps
  coef_1 = jnp.exp(per_token_logps - old_per_token_logps)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)
  policy_loss = jnp.maximum(
      -coef_1 * jnp.expand_dims(advantages, 1),
      -coef_2 * jnp.expand_dims(advantages, 1),
  )
  # TODO(abheesht): Verify these losses, especially the normalisation part.
  policy_loss = ppo_helpers.masked_mean(policy_loss, completion_mask)

  # TODO(abheesht): We should log entropy.
  aux = {}
  return policy_loss, aux
