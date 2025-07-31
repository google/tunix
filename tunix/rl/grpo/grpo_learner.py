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

"""GRPO learner."""

from __future__ import annotations

from concurrent import futures
import dataclasses
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.grpo import grpo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import metrics_logger

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


class RepeatIterable(Iterable[Any]):
  """A simple wrapper on top of one example to repeat it N times."""

  def __init__(self, data: list[Any], repeat: int = 1):
    self._data = data
    self._data_len = len(data)
    self._total_count = repeat * self._data_len
    self._itr_cnt = 0

  def __iter__(self):
    self._itr_cnt = 0
    return self

  def __next__(self):
    if self._itr_cnt >= self._total_count:
      raise StopIteration
    output = self._data[self._itr_cnt % self._data_len]
    self._itr_cnt += 1
    return output


@flax.struct.dataclass(frozen=True)
class TrainExample:
  prompt_ids: jax.Array
  prompt_mask: jax.Array
  completion_ids: jax.Array
  completion_mask: jax.Array
  advantages: jax.Array
  ref_per_token_logps: jax.Array | None
  old_per_token_logps: jax.Array | None


@dataclasses.dataclass(slots=True, kw_only=True)
class GrpoConfig:
  """Configuration for GRPO algorithm.

  Attributes:
    num_generations: The number of times the policy generates multiple responses
      for a given prompt within a single training step. This corresponds to 'G'
      in Algorithm 1 in the paper. A higher value means more samples are used to
      compute relative advantages.
    num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
    beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
      function. This term prevents policy updates from deviating too far from
      the reference model. A value of 0.0 means no KL penalty is applied.
    epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
      PPO, it ensures stable updates.
    loss_algo: use GRPO or GSPO for loss computation. GRPO loss is per-batch
      normalized instead of per-response normalized as mentioned in the paper.
      For GSPO, we use gspo-token loss which is more flexible.

  References:
  - GRPO: https://arxiv.org/abs/2402.03300
  - GSPO: https://www.arxiv.org/pdf/2507.18071
  """

  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2
  loss_algo: str = "grpo"  # grpo or gspo-token

  def __post_init__(self):
    assert self.num_generations > 1, (
        "num_generations must be greater than 1. Received: "
        f"{self.num_generations}"
    )


class GrpoLearner:
  """GRPO (Group Relative Policy Optimization) learner.

  GRPO is a reinforcement learning algorithm designed to enhance the reasoning
  abilities of large language models, like mathematical problem-solving. It is
  a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
  eliminating the need for a separate value function model. GRPO works by
  generating multiple responses for a given prompt, evaluating these responses
  using a reward model, and then calculating a relative advantage based on the
  group's performance to update the policy.

  References:
  - https://arxiv.org/abs/2402.03300
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      reward_fns: RewardFn | List[RewardFn],
      grpo_config: GrpoConfig,
  ):
    """Initializes the `GrpoTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      grpo_config: An instance of `GrpoConfig` containing all GRPO sepecific
        parameters.
    """
    assert grpo_config.loss_algo in ["grpo", "gspo-token"], (
        "loss_algo should be either grpo or gspo-token. Received: "
        f"{grpo_config.loss_algo}"
    )
    self.grpo_config = grpo_config
    self.rl_cluster = rl_cluster
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )

    # Workaround for passing in importance_sampling_algo as jax transforms
    # doesn't like partial functions with kwargs.
    loss_fn = lambda model, train_example, beta, epsilon: grpo_loss_fn(
        model,
        train_example,
        beta=beta,
        epsilon=epsilon,
        loss_algo=self.grpo_config.loss_algo,
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "beta": self.grpo_config.beta,
            "epsilon": self.grpo_config.epsilon,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": "kl"})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "rewards/overall",
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
    ])
    self.rl_cluster.actor_trainer.is_managed_externally = True

    self._metrics_logger = self.rl_cluster.actor_trainer.metrics_logger

    self.grad_acc_steps = (
        self.rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

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
    """Generates text completions and computes the advantages for GRPO training.

    Args:
      training_input: A dictionary containing the training input data,
        containing the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data, including
      prompt IDs, completion IDs, masks, advantages, and per-token log
      probabilities from the reference and policy models.
    """
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    # Generate, and pad output.
    completion_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
    )
    completion_ids = completion_output.tokens
    prompt_ids = completion_output.left_padded_prompt_tokens

    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
        "int32"
    )
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    completion_mask = completion_mask * completion_padding_mask

    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
    else:
      ref_per_token_logps = None

    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids, completion_tokens=completion_ids
      )
    else:
      old_per_token_logps = None

    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_output.text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )

    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
    )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    steps = self._get_metric_logging_steps(mode)
    self._metrics_logger.log(
        "completions/mean_length",
        agg_completion_mask.mean(),
        mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/max_length",
        agg_completion_mask.max(),
        mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/min_length",
        agg_completion_mask.min(),
        mode,
        steps,
    )

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: metrics_logger.Mode,
      **kargs,
  ):
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      **kargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts, num_reward_fns]`) of scalar rewards for
      each prompt-completion pair. The rewards are computed using the provided
      reward functions.
    """
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    steps = self._get_metric_logging_steps(mode)
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)

      self._metrics_logger.log(
          f"rewards/{reward_fn.__name__}",
          r.mean(),
          mode,
          steps,
      )

    rewards = jnp.nansum(rewards, axis=1)

    self._metrics_logger.log(
        "rewards/overall",
        rewards.mean(),
        mode,
        steps,
    )

    return rewards

  def prepare_dataset(
      self,
      iterator: Iterator[_TrainingInputT],
      proceed_num_steps: int,
      sample_repeat: int,
      batch_repeat: int,
      data_queue: queue_lib.AbstractDataQueue[
          list[TrainExample] | RepeatIterable | None
      ],
      async_loading: bool = False,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> None:
    """Prepares the dataset for training.

    Args:
      iterator: The input iterator of the dataset.
      proceed_num_steps: The number of steps to proceed for the iterator if set
        to a positive number. If it's set to a non positive number, the function
        will exhaust the iterator. If the input iterator is exhausted before the
        number of steps is reached, the function will return the empty result.
      sample_repeat: The number of times to repeat the sample within a batch.
      batch_repeat: The number of times to repeat the batch in the final
        dataset.
      data_queue: The data queue to use for putting the examples into.
      async_loading: Whether to load the batch asynchronously, if not async
        loading, then all the examples needed will be processed and then loaded
        into the data queue.
      mode: The mode to use for logging metrics.

    Returns:
      None. Examples are put into the data queue.
    """

    example_list = []

    def _put_list_of_examples_to_data_queue():
      if not async_loading:
        data_queue.put(RepeatIterable(example_list, batch_repeat))
      elif batch_repeat > 1:
        # Since we have already loaded the batch in data_queue once, we only
        # need to repeat batch_repeat - 1 times.
        data_queue.put(RepeatIterable(example_list, batch_repeat - 1))

    try:
      while True:
        while (
            mode == metrics_logger.Mode.TRAIN
            and self._train_steps < self._last_train_step
        ):  # fast forward the iterator if loading from a previous checkpoint.
          next(iterator)
          self._train_steps += 1
        example = next(iterator)
        example = jax.tree.map(
            lambda x: np.repeat(x, sample_repeat, axis=0),
            example,
        )  # [B] -> [B * G]

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
          return
    except StopIteration as e:
      if proceed_num_steps > 0:
        raise e
      else:
        _put_list_of_examples_to_data_queue()
        return
    except Exception as e:
      raise e
    finally:
      # Signal no more iterable to be loaded.
      data_queue.put(None)

  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """GRPO training loop.

    Algorithm as below: extract from https://arxiv.org/abs/2402.03300
    Input initial policy model πθinit; reward models rφ; task prompts D;
    hyperparameters ε, β, μ

    policy model πθ ← πθinit
    for iteration = 1, ..., I do
      reference model πref ← πθ
      for step = 1, ..., M do
        Sample a batch D♭ from D
        Update the old policy model πθold ← πθ
        Sample G outputs {oi}G_i=1 ~ πθold(· | q) for each question q ∈ D♭
        Compute rewards {ri}G_i=1 for each sampled output oi by running rφ
        Compute Âi,t for the t-th token of oi through group relative advantage
        estimation.
        for GRPO iteration = 1, ..., μ do
          Update the policy model πθ by maximizing the GRPO objective (Equation
          21)
      Update rφ through continuous training using a replay mechanism.
    Output πθ

    NOTE:
    1. The outer loop (I) is ignored for now because we never update the
    reference model for now.
    2. Currently sample and train hold the same referece to the model. So we
    also omit the step to update the sampler model.

    Args:
      train_ds: An iterable of training input data, where each element is a
        dictionary containing the key 'prompts'.
      eval_ds: An iterable of evaluation input data, where each element is a
        dictionary containing the key 'prompts'.
      skip_jit: Whether to skip JIT compilation of the training loop.
    """
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
            self.prepare_dataset,
            iterator=train_iterator,
            proceed_num_steps=self.grad_acc_steps,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
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
            if eval_ds and not curr_eval_ds:
              self.prepare_dataset(
                  iterator=iter(eval_ds),
                  proceed_num_steps=-1,
                  sample_repeat=self.grpo_config.num_generations,
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
        # call to throw stop iteration as a singal to break the loop
        future.result()
        # sync the train steps with internel trainer, this is based on the
        # assumption that the trainer internally doesn't reset the train steps.
        # there is current a unit test to ensure this assumption.
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


def grpo_loss_fn(model, train_example, beta, epsilon, loss_algo):
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    beta: The coefficient for the KL divergence penalty. A value of 0.0 means no
      KL penalty is applied.
    epsilon: Epsilon value for clipping.
    loss_algo: The loss algorithm to use. Can be grpo or gspo-token.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  prompt_ids, prompt_mask = (
      train_example.prompt_ids,
      train_example.prompt_mask,
  )
  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)
  prompt_completion_mask = jnp.concat([prompt_mask, completion_mask], axis=-1)
  attention_mask = common.make_causal_attn_mask(prompt_completion_mask)
  logits_to_keep = completion_ids.shape[1]
  positions = common.build_positions_from_mask(prompt_completion_mask)

  per_token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  advantages = train_example.advantages

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = per_token_logps - old_per_token_logps
  if loss_algo == "gspo-token":
    seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
        axis=-1
    ) / jnp.clip(completion_mask.sum(-1), min=1)
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_importance_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)

  coef_1 = jnp.exp(seq_importance_ratio)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)

  # TODO(tsbao): We should handle token level advantages.
  per_token_loss = -jnp.minimum(
      coef_1 * jnp.expand_dims(advantages, 1),
      coef_2 * jnp.expand_dims(advantages, 1),
  )

  if loss_algo == "gspo-token":
    loss_denominator = jnp.clip(completion_mask.sum(axis=-1), min=1)
  else:  # grpo
    loss_denominator = jnp.clip(completion_mask.sum(), min=1)

  aux = {"kl": 0.0}
  if beta != 0.0:
    kl = grpo_helpers.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    per_token_loss = per_token_loss + beta * kl

    # Log mean KL.
    aux["kl"] = (kl * completion_mask).sum() / loss_denominator.mean()

  if loss_algo == "gspo-token":
    loss = (
        (per_token_loss * completion_mask).sum(axis=-1) / loss_denominator
    ).mean()
  else:  # grpo
    loss = (per_token_loss * completion_mask).sum() / loss_denominator

  return loss, aux
