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
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.ppo import ppo_helpers
from tunix.rl.queue import data_queue as queue_lib

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  returns: jax.Array
  old_values: jax.Array


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
    epsilon: Epsilon value for clipping the ratio for the policy objective.
    epsilon_low: Lower bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_high: Upper bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_c: Lower bound for clipping for dual-clip PPO. If not provided, we
      don't do dual-clip PPO.
      Reference: https://arxiv.org/abs/1912.09729.
    vf_coef: The coefficient for the value function loss.
    clip_range_value: The range for clipping the value function loss.
  """

  num_ppo_epochs: int = 1
  mini_batch_size: int | None = None

  # PPO loss and advantage computation configs.
  gamma: float = 1.0
  gae_lambda: float = 0.95
  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_low: float | None = None
  epsilon_high: float | None = None
  epsilon_c: float | None = None
  vf_coef: float = 0.1
  clip_range_value: float = 0.2

  def __post_init__(self):
    self.epsilon_low = self.epsilon_low if self.epsilon_low else self.epsilon
    self.epsilon_high = self.epsilon_high if self.epsilon_high else self.epsilon
    self.epsilon = self.epsilon

    if self.epsilon_c is not None and self.epsilon_c <= 1.0:
      raise ValueError(
          f"`epsilon_c` must be greater than 1. Received: {self.epsilon_c}."
      )


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
            "epsilon_low": self.ppo_config.epsilon_low,
            "epsilon_high": self.ppo_config.epsilon_high,
            "epsilon_c": self.ppo_config.epsilon_c,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )
    self.rl_cluster.actor_trainer.is_managed_externally = True

    # adjust global steps based on the number of iterations.
    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.train_steps
        // self.ppo_config.num_ppo_epochs
    )

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
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.ppo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log
    )
    # TODO(tsbao): this need to be fixed, currently it won't display in tqdm
    # since these metrics are logged in rl_cluster and aggregated at global step
    # level.
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "score/mean",
        "reward/mean",
        lambda: "reward_kl_penalty" if self.ppo_config.beta != 0.0 else None,
    ])

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log({
        "vpred_mean": np.mean,
        "vf_clipfrac": np.mean,
    })

    self.grad_acc_steps = (
        rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

    self._data_shuffle_key = data_shuffle_seed
    if data_shuffle_seed is None:
      self._data_shuffle_key = 0
    self._data_shuffle_key = jax.random.PRNGKey(self._data_shuffle_key)

    self._iter_steps = 0
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
    self._last_iter_step = self.rl_cluster.actor_trainer.iter_steps

  def _get_metric_logging_steps(self, mode: rl_cluster_lib.Mode) -> int:
    return (
        self._iter_steps
        if mode == rl_cluster_lib.Mode.TRAIN
        else self._eval_steps
    )

  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
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

    # TODO(abheesht): verl allows specifying different micro batch sizes for
    # computing log probs, values, rewards, etc. We can do that here.

    # ===== Generation ======
    # Generate. We use `model`, i.e., the policy model for generating the
    # "experiences".
    completion_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
    )
    completion_ids = completion_output.tokens
    prompt_ids = completion_output.left_padded_prompt_tokens

    batch_size = completion_ids.shape[0]
    logits_to_keep = completion_ids.shape[1]
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    )

    # ===== Compute log probs ======
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
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
    values = values * completion_mask

    # ===== Reward computation ======
    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )[:, -logits_to_keep:]
      # We use the score corresponding to the last non-padding token.
      last_token_scores = scores[jnp.arange(batch_size), eos_idx]
    else:
      last_token_scores = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=completion_output.text,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )

    # Reward computation is in accordance with TRL and verl's
    # `BatchRewardManager` (token-level rewards).
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. A positive reward is given only at the final timestep, so we add that
    # to the tensor of zeros.
    # 3. Subtract KL divergence from the reward tensor.
    rewards = jnp.zeros_like(completion_ids)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)
    if self.ppo_config.beta != 0.0:
      # TODO(abheesht): Add a toggle - KL can either be added directly to
      # rewards or computed in the loss function.
      kl = common.compute_kl_divergence(
          old_per_token_logps, ref_per_token_logps
      )
      kl = kl * completion_mask
      rewards = rewards - self.ppo_config.beta * kl

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantages, returns = ppo_helpers.compute_gae_advantages(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )

    # ===== Metric logging ======
    # Log raw scores from the reward model fn
    self.rl_cluster.buffer_metrics(
        {
            "score/mean": (np.mean(last_token_scores), np.mean),
            "score/max": (np.max(last_token_scores), np.max),
            "score/min": (np.min(last_token_scores), np.min),
        },
        mode=mode,
    )

    # Log final rewards (scores + KL penalty)
    sequence_rewards = rewards.sum(-1)
    self.rl_cluster.buffer_metrics(
        {
            "reward/mean": (np.mean(sequence_rewards), np.mean),
            "reward/max": (np.max(sequence_rewards), np.max),
            "reward/min": (np.min(sequence_rewards), np.min),
        },
        mode=mode,
    )
    if self.ppo_config.beta != 0.0:
      # Average of the per-sequence mean KL
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
      )
      self.rl_cluster.buffer_metrics(
          {
              "reward_kl_penalty": (
                  per_sequence_mean_kl.mean(),
                  np.mean,
              ),
          },
          mode=mode,
      )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
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
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
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
            mode == rl_cluster_lib.Mode.TRAIN
            and self._iter_steps < self._last_iter_step
        ):
          next(iterator)
          self._iter_steps += 1
        example = next(iterator)

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._iter_steps
            if mode == rl_cluster_lib.Mode.TRAIN
            else self._eval_steps,
        ):
          advantage = self._generate_and_compute_advantage(example, mode)
        if async_loading:
          data_queue.put([advantage])

        if mode == rl_cluster_lib.Mode.TRAIN:
          self._iter_steps += 1
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
        initial_steps = self._iter_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            mini_batch_size=self.ppo_config.mini_batch_size,
            proceed_num_steps=self.grad_acc_steps,
            batch_repeat=self.ppo_config.num_ppo_epochs,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_steps
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
                  mode=rl_cluster_lib.Mode.EVAL,
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
        # Sync the iter steps with internal trainer, this is based on the
        # assumption that the trainer internally doesn't reset the iter steps.
        self._iter_steps = self.rl_cluster.actor_trainer.iter_steps
        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_steps
          ):
            self.rl_cluster.sync_weights()
        else:
          self.rl_cluster.global_steps += (
              1  # manually increment the global steps.
          )
        if (
            self.rl_cluster.actor_trainer.train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    self.rl_cluster.close()


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    vf_coef: float,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
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
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
  # "token mean" style of normalisation.
  vf_loss = ppo_helpers.masked_mean(clipped_vf_losses, completion_mask)
  vf_loss = 0.5 * vf_loss

  aux = {
      "vpred_mean": ppo_helpers.masked_mean(vpreds, completion_mask),
      "vf_clipfrac": ppo_helpers.masked_mean(
          (vf_losses2 > vf_losses1).astype(jnp.float32), completion_mask
      ),
  }
  return vf_coef * vf_loss, aux


def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon_low: float,
    epsilon_high: float,
    epsilon_c: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the policy loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  use_dual_clip_ppo = epsilon_c is not None

  # Get log probs.
  per_token_logps, _ = common.compute_per_token_logps(
      model,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
  )

  advantages = train_example.advantages

  # Compute ratio.
  old_per_token_logps = train_example.old_per_token_logps
  ratio = jnp.exp(per_token_logps - old_per_token_logps)
  ratio_clipped = jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)

  # Vanilla PPO loss
  pg_losses_1 = -ratio * advantages
  pg_losses_2 = -ratio_clipped * advantages
  clip_pg_losses_1 = jnp.maximum(pg_losses_1, pg_losses_2)

  # Dual-clip PPO to avoid negative-advantage policy updates
  pg_losses = clip_pg_losses_1
  if use_dual_clip_ppo:
    pg_losses_3 = -epsilon_c * advantages
    clip_pg_losses_2 = jnp.minimum(pg_losses_3, clip_pg_losses_1)

    pg_losses = jnp.where(advantages < 0.0, clip_pg_losses_2, clip_pg_losses_1)

    # For logging.
    unreduced_pg_clipfrac_lower = (
        (clip_pg_losses_1 > pg_losses_3) & (advantages < 0.0)
    ).astype(jnp.float32)
    pg_clipfrac_lower = ppo_helpers.masked_mean(
        unreduced_pg_clipfrac_lower, completion_mask
    )

  # Logging
  aux = {
      "pg_clipfrac": ppo_helpers.masked_mean(
          (pg_losses_2 > pg_losses_1).astype(jnp.float32), completion_mask
      ),
  }
  if use_dual_clip_ppo:
    aux["pg_clipfrac_lower"] = pg_clipfrac_lower  # pylint: disable=undefined-variable

  # "token mean" style of normalisation
  policy_loss = ppo_helpers.masked_mean(pg_losses, completion_mask)

  return policy_loss, aux
