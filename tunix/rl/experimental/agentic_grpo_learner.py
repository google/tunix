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

"""Implements an RLLearner for the Agentic GRPO algorithm.

This learner orchestrates the process of generating multiple text completions
for each prompt from a dataset, computing rewards and advantages according to
the GRPO (Group-wise Reward Policy Optimization) algorithm, and then training
the actor model.

The data flow is designed around an asynchronous producer-consumer pattern:
1. A producer generates rollouts (text generations) in parallel for each prompt.
2. These rollouts are grouped by the original prompt.
3. For each group, rewards and advantages are computed.
4. The resulting training examples are put into a queue.
5. The main training loop consumes these examples to update the model weights.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Sequence, TypeVar

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.experimental import agentic_rl_learner


TrainingInputT = agentic_rl_learner.TrainingInputT
RewardFn = agentic_rl_learner.RewardFn
MetricFn = agentic_rl_learner.MetricFn

TrainExample = agentic_rl_learner.TrainExample


@dataclasses.dataclass(slots=True, kw_only=True)
class GRPOConfig(agentic_rl_learner.AgenticRLConfig):
  """Configuration for GRPO algorithm.

  Parameters:
    num_generations: Number of samples per prompt (G in the paper). Must be > 1.
    num_iterations: Number of GRPO iterations per batch (μ in the paper).
    beta: KL penalty coefficient.
    epsilon: PPO-style clipping epsilon.
    loss_algo: "grpo" or "gspo-token".
    system_prompt: System prompt for the agent.
    max_concurrency: Maximum number of concurrent rollout engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
  """
  algo_variant: str = "agentic_grpo"
  advantage_estimator: str = "agentic_grpo"
  policy_loss_fn: str = "agentic_grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  loss_algo: (
      str
  ) = (  # grpo or gspo-token # TODO(sizhi): Remove this option once gspo is
      # refactored to a separate loss fn.
      "grpo"
  )
  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2
  system_prompt: str = ""
  max_concurrency: int = 16
  epsilon_high: float | None = None  # 0.28 from DAPO.
  off_policy_steps: int = 0

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )
    if self.epsilon_high is None:
      self.epsilon_high = self.epsilon
    if self.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )


TGrpoConfig = TypeVar("TGrpoConfig", bound=GRPOConfig)


class GRPOLearner(agentic_rl_learner.AgenticRLLearner[TGrpoConfig]):
  """An RLLearner that implements the GRPO algorithm in an agentic setting.

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
      algo_config: TGrpoConfig,
      chat_parser: Any,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `GRPOTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept `prompts`, `completions` and optional keyword arguments, and
        return a list of float rewards.
      algo_config: An instance of `GRPOConfig` containing all GRPO specific
        parameters.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      data_shuffle_seed: The seed used to shuffle the training data.
    """  # fmt: skip
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
        algo_config=algo_config,
        chat_parser=chat_parser,
    )

    # Workaround to pass loss fn with algorithm flag
    policy_loss_fn = function_registry.get_policy_loss_fn(
        self.algo_config.policy_loss_fn
    )
    logging.info(
        "algo_config.policy_loss_fn: %s", self.algo_config.policy_loss_fn
    )
    logging.info("type(policy_loss_fn): %s", type(policy_loss_fn))

    # Log the string representation of the callable
    logging.info("repr(policy_loss_fn): %r", policy_loss_fn)
    loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
        model,
        train_example,
        algo_config=self.algo_config,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "algo_config": self.algo_config,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        lambda: "kl" if self.algo_config.beta != 0.0 else None,
    ])

  def _process_results(
      self,
      results: List[Any],
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages.

    This is a core method that performs several steps:
    1. Extracts completions from the raw trajectory results.
    2. Pads prompt and completion tokens to a consistent length.
    3. Computes masks for prompts and completions.
    4. Gets reference and old model log probabilities if required.
    5. Computes rewards for each completion using the provided reward functions.
    6. Computes GRPO-specific advantages from the rewards.
    7. Buffers metrics for logging.
    8. Constructs and returns a list of `TrainExample` objects.

    Args:
      results: A list of trajectory results for a single GRPO group.
      training_input: The merged training input for the group.
      mode: The current mode (TRAIN or EVAL).
      step: The current training step.

    Returns:
      A list of `TrainExample` instances containing all data needed for the
      loss function.
    """
    logging.debug(
        "Processing results to compute advantage for %d items.", len(results)
    )
    # With a full group, sorting by pair_index is not necessary as they all
    # originate from the same initial prompt.
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    # Extract completions and tokens from the group of G results.
    completion_texts = []
    completion_tokens_list = []
    policy_versions_list = []
    for item in results:
      conversation = item.traj.get("conversation_text") or []
      assistant_text = next(
          message["content"]
          for message in conversation
          if message["role"] == "assistant"
      )
      completion_texts.append(assistant_text)
      completion_tokens_list.append(item.traj.get("conversation_tokens"))
      policy_version = item.traj.get("policy_version")
      if policy_version is None:
        raise ValueError("policy_version is missing from trajectory task.")
      policy_versions_list.append(policy_version)

    # All results in a group share the same prompt.
    prompt_tokens = results[0].traj.get("prompt_tokens")

    # Pad all prompts and completions to consistent lengths.
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[mode]
    max_prompt_length = rollout_config.max_prompt_length
    max_tokens_to_generate = rollout_config.max_tokens_to_generate
    all_padded_prompt_ids = []
    all_padded_completion_ids = []
    for completion_tokens in completion_tokens_list:
      padded_prompt, padded_completion, _ = (
          agentic_utils.pad_prompt_and_completion(
              prompt_tokens,
              completion_tokens,
              max_prompt_length,
              max_tokens_to_generate,
              pad_value,
          )
      )
      all_padded_prompt_ids.append(padded_prompt)
      all_padded_completion_ids.append(padded_completion)

    prompt_ids = jnp.asarray(all_padded_prompt_ids)
    completion_ids = jnp.asarray(all_padded_completion_ids)
    logging.debug(
        "Token shapes: prompt_ids=%s, completion_ids=%s",
        prompt_ids.shape,
        completion_ids.shape,
    )

    # Masks
    prompt_mask = prompt_ids != pad_value
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value)
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    completion_mask = completion_mask * completion_padding_mask
    if self.algo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=1,
      )
    else:
      ref_per_token_logps = None
    logging.debug("Ref logps computed.")
    if self.algo_config.num_iterations > 1:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          micro_batch_size=1,
      )
    else:
      old_per_token_logps = None
    logging.debug("Old logps computed.")
    # Rewards & advantages

    # Prepare arguments for reward computation by forwarding all training inputs
    # except for prompts, which is passed explicitly.
    reward_kwargs = {
        key: value for key, value in training_input.items() if key != "prompts"
    }
    # TODO: b/456528861 - Refactor reward computation to happen within the
    # environment during rollout, rather than as a post-processing step. This
    # would align with the standard agentic RL pattern and remove the need for
    # `dummy_reward`.
    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_texts,
        mode=mode,
        **reward_kwargs,
        step=step,
    )

    advantage_estimator = function_registry.get_advantage_estimator(
        self.algo_config.advantage_estimator
    )
    advantages = advantage_estimator(
        rewards=rewards, num_generations=self.algo_config.num_generations
    )

    policy_versions = jnp.array(policy_versions_list, dtype=jnp.int32)

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics_async(
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
        step=step,
    )
    for metric_fn in self.metric_fns:
      user_defined_metric = metric_fn(
          prompts=training_input["prompts"],
          completions=completion_texts,
          advantages=advantages,
          rewards=rewards,
          **{
              key: value
              for key, value in training_input.items()
              if key != "prompts"
          },
      )
      self.rl_cluster.buffer_metrics_async(
          user_defined_metric, mode=mode, step=step
      )

    logging.debug("Advantages computed: %s", advantages)
    combined_batch = TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        policy_version=policy_versions,
    )
    return [
        rl_utils.get_batch_slice(combined_batch, slice(i, i + 1))
        for i in range(self.algo_config.num_generations)
    ]


@function_registry.register_policy_loss_fn("agentic_grpo")
def grpo_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
):
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    algo_config: The algorithm config.
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  loss_algo = algo_config.loss_algo
  epsilon_high = (
      algo_config.epsilon_high
      if hasattr(algo_config, "epsilon_high")
      else epsilon
  )
  loss_aggregation_mode = algo_config.loss_agg_mode

  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # TODO(yangmu): trace this part as "actor_inference_and_training".
  # with perf_tracer.span("...", list(completion_ids.devices())):
  per_token_logps = common.compute_per_token_logps(
      model,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_logits=False,
  )
  advantages = train_example.advantages

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = per_token_logps - old_per_token_logps
  # TODO(sizhi): Refactor this to a separate function.
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
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)

  # TODO(tsbao): We should handle token level advantages.
  per_token_loss = -jnp.minimum(
      coef_1 * jnp.expand_dims(advantages, 1),
      coef_2 * jnp.expand_dims(advantages, 1),
  )

  aux = {"kl": 0.0}
  if beta is not None and beta != 0.0:
    kl = common.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    per_token_loss = per_token_loss + beta * kl

    # Log mean KL.
    aux["kl"] = (kl * completion_mask).sum() / jnp.clip(
        completion_mask.sum(), min=1
    )

  loss = common.aggregate_loss(
      per_token_loss, completion_mask, loss_aggregation_mode
  )

  return loss, aux


@function_registry.register_advantage_estimator("agentic_grpo")
def compute_advantages(rewards: jax.Array, num_generations: int) -> jax.Array:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
  std_grouped_rewards = rewards.reshape(-1, num_generations).std(
      axis=-1, ddof=1
  )

  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
  return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
