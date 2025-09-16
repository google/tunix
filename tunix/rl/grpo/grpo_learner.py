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
"""GRPO Learner implementation."""
from __future__ import annotations

from concurrent import futures
import dataclasses
from itertools import chain  # pylint: disable=g-importing-member
import math
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.grpo import grpo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


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
    """Validates the configuration after initialization."""
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
      metric_fns: (
          Sequence[Callable[..., rl_cluster_lib.MetricsT]] | None
      ) = None,
  ):
    """Initializes the `GrpoTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      grpo_config: An instance of `GrpoConfig` containing all GRPO specific
        parameters.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn):
        >>> def metric_fn(prompts, completions, rewards, advantages, **kargs):
        ...    return { ...        "prompt_min_len": (min(len(p) for p in
        prompts), np.min), ...        ... ...    }
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
    self.metric_fns = metric_fns or []

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
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "rewards/overall",
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
    ])
    self.rl_cluster.actor_trainer.is_managed_externally = True

    # adjust global steps based on the number of iterations.
    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.train_steps
        // self.grpo_config.num_iterations
    )

    self.grad_acc_steps = (
        self.rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

    self._iter_steps = 0
    self._eval_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
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

  def _compute_trajectory_ids(self, example: _TrainingInputT) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch."""
    batch_size = len(example["prompts"]) // self.grpo_config.num_generations
    row_offset = self._iter_steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.grpo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.grpo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
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
    training_config = self.rl_cluster.cluster_config.training_config
    training_input["prompts"] = list(training_input["prompts"])
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            training_config.rollout_micro_batch_size
            * self.grpo_config.num_generations
        ),
    )
    completion_ids = rollout_output.tokens
    prompt_ids = rollout_output.left_padded_prompt_tokens
    completion_text = rollout_output.text

    # Assemble masks
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
        "int32"
    )
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=(
              training_config.ref_logps_micro_batch_size
              * self.grpo_config.num_generations
          ),
      )
    else:
      ref_per_token_logps = None
    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          micro_batch_size=(
              training_config.old_logps_micro_batch_size
              * self.grpo_config.num_generations
          ),
      )
    else:
      old_per_token_logps = None

    # Compute rewards and advantages
    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )

    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
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
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=completion_text,
          advances=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

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
      mode: rl_cluster_lib.Mode,
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
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)
      self.rl_cluster.buffer_metrics(
          {
              f"rewards/{reward_fn.__name__}": (
                  np.mean(r),
                  np.mean,
              ),
          },
          mode=mode,
      )

    rewards = jnp.nansum(rewards, axis=1)
    self.rl_cluster.buffer_metrics(
        {
            "rewards/overall": (
                np.mean(rewards),
                np.mean,
            ),
        },
        mode=mode,
    )
    self.rl_cluster.buffer_metrics(
        {
            "rewards/min": (
                np.min(rewards),
                np.min,
            ),
        },
        mode=mode,
    )
    for p, c in zip(prompts, completions):
      self.rl_cluster.buffer_metrics(
          {
              "prompts": (
                  p,
                  None,
              ),
              "completions": (
                  c,
                  None,
              ),
          },
          mode=mode,
      )

    return rewards

  def aggregate_and_compute_advantages(
      self,
      buf: list[_TrainingInputT],
      buf_sizes: list[int],
      buf_b: int,
      service_target_bs: int,
      sample_repeat: int,
      mode: rl_cluster_lib.Mode,
      force: bool = False,
  ) -> tuple[list[TrainExample], list[_TrainingInputT], list[int], int]:
    """Merges, repeats, and computes advantages for a buffer of examples.

    This function takes a buffer of micro-batches, merges them, repeats the
    samples, runs a single large forward pass to generate completions and
    compute advantages, and then splits the results back into micro-batches.

    Args:
      buf: A list of training micro-batches.
      buf_sizes: A list of the number of samples for each training micro-batch.
      buf_b: The aggregated sample count (before repeating).
      service_target_bs: The target batch size for the service.
      sample_repeat: The number of times each sample is repeated.
      mode: The mode to use for logging metrics.
      force: If True, forces the aggregation and computation even if the buffer
        size is less than `service_target_bs`.

    Returns:
      A tuple containing:
        - A list of small TrainExample chunks, split back by original micro
          boundaries.
        - The updated buffer.
        - The updated buffer sizes.
        - The updated buffer sample count.
    """
    produced: list[TrainExample] = []

    if not buf:
      return produced, buf, buf_sizes, buf_b
    if (not force) and (buf_b < service_target_bs):
      return produced, buf, buf_sizes, buf_b

    # Merge multiple training micro-batches
    merged: _TrainingInputT = {}
    keys = buf[0].keys()
    for k in keys:
      merged[k] = (
          list(buf[0][k])
          if isinstance(buf[0][k], list)
          else np.asarray(buf[0][k])
      )
    for i in range(1, len(buf)):
      for k in keys:
        a, b = merged[k], buf[i][k]
        if isinstance(a, list) and isinstance(b, list):
          a.extend(b)
        else:
          merged[k] = np.concatenate([np.asarray(a), np.asarray(b)], axis=0)

    # Repeat samples (equivalent to repeating each micro, then concat)
    merged_repeated = jax.tree.map(
        lambda x: np.repeat(x, sample_repeat, axis=0),
        merged,
    )

    if mode == rl_cluster_lib.Mode.TRAIN:
      trajectory_ids = self._compute_trajectory_ids(merged_repeated)
      assert "trajectory_ids" not in merged_repeated
      merged_repeated["trajectory_ids"] = trajectory_ids

    # Single large forward pass + advantage computation
    # The batch size of `big_example` is `buf_b * sample_repeat`, where
    # `buf_b` is the number of prompts aggregated so far.
    with jax.profiler.StepTraceAnnotation(
        "sampler",
        step_num=self._iter_steps
        if mode == rl_cluster_lib.Mode.TRAIN
        else self._eval_steps,
    ):
      big_example = self._generate_and_compute_advantage(merged_repeated, mode)

    # Split back to original training micro size
    offset = 0
    for n in buf_sizes:
      # Calculate slice indices
      start_idx = offset * sample_repeat
      end_idx = (offset + n) * sample_repeat
      token_sl = slice(start_idx, end_idx)

      # Create TrainExample for this micro-batch
      te_small = TrainExample(
          prompt_ids=big_example.prompt_ids[token_sl],
          prompt_mask=big_example.prompt_mask[token_sl],
          completion_ids=big_example.completion_ids[token_sl],
          completion_mask=big_example.completion_mask[token_sl],
          ref_per_token_logps=(
              None
              if big_example.ref_per_token_logps is None
              else big_example.ref_per_token_logps[token_sl]
          ),
          advantages=big_example.advantages[token_sl],
          old_per_token_logps=(
              None
              if big_example.old_per_token_logps is None
              else big_example.old_per_token_logps[token_sl]
          ),
      )

      produced.append(te_small)
      offset += n

    return produced, [], [], 0

  def _prepare_data(
      self,
      iterator: Iterator[_TrainingInputT],
      proceed_num_steps: int,
      sample_repeat: int,
      batch_repeat: int,
      data_queue: queue_lib.AbstractDataQueue[list[TrainExample] | None],
      async_loading: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> None:
    """Orchestrates the data preparation pipeline for GRPO.

    This method is designed to efficiently process data in micro-batches while
    accommodating the requirements of different model services (rollout,
    reference log-probabilities and old policy log probabilities) that may have
    different optimal batch sizes.

    The pipeline follows these main steps:
    1. **Merge**: It consumes multiple small micro-batches from the input
       `iterator` and merges them into a single, larger batch. This is done to
       meet the `service_target_bs`, which is the least common multiple of the
       micro-batch sizes for different services, ensuring efficient hardware
       utilization.
    2. **Repeat (Sample)**: Each prompt in the merged batch is repeated
       `sample_repeat` times (which corresponds to `num_generations` in GRPO).
       This is because GRPO generates multiple completions for each prompt to
       compute relative advantages.
    3. **Single Large Forward Pass**: The resulting large batch of repeated
       prompts is then processed in a single call to
       `_generate_and_compute_advantage`. This function handles text generation,
       reward computation, and advantage calculation for the entire batch.
    4. **Split**: After processing, the large `TrainExample` is split back into
       smaller chunks that correspond to the original input micro-batches.
    5. **Enqueue**: These smaller `TrainExample` chunks are then put into the
       `data_queue` to be consumed by the training loop.

    Args:
      iterator: An iterator yielding `_TrainingInputT` examples.
      proceed_num_steps: The number of training micro-batches to process before
        returning. If > 0, the function will stop after consuming this many
        steps. If -1, it will continue until the iterator is exhausted.
      sample_repeat: The number of times each sample in a micro-batch is
        repeated during the advantage computation. This is typically
        `grpo_config.num_generations`.
      batch_repeat: The number of times the produced `TrainExample` batch should
        `grpo_config.num_iterations`.
      data_queue: The queue to which lists of `TrainExample` are added.
      async_loading: If True, enqueue each produced micro-batch immediately in
        async mode. Otherwise, accumulate and enqueue at the boundary.
      mode: The metrics logger mode, either `metrics_logger.Mode.TRAIN` or
        `metrics_logger.Mode.EVAL`.
    """
    training_config = self.rl_cluster.cluster_config.training_config

    def enqueue_examples(examples: list[TrainExample], times: int) -> None:
      """Wrap each TrainExample as [TrainExample] and put it into the queue, repeated `times`."""
      if times <= 0 or not examples:
        return
      for _ in range(times):
        for ex in examples:
          data_queue.put([ex])

    service_target_bs = math.lcm(
        training_config.rollout_micro_batch_size,
        training_config.ref_logps_micro_batch_size,
        training_config.old_logps_micro_batch_size,
    )

    buf: list[_TrainingInputT] = []
    buf_sizes: list[int] = []  # Number of samples for each training micro-batch
    buf_b = 0  # Aggregated sample count (before repeating)
    consumed_steps = 0  # Number of consumed training micro-batches

    pending_examples: list[TrainExample] = []

    try:
      while True:
        while (
            mode == rl_cluster_lib.Mode.TRAIN
            and self._iter_steps < self._last_iter_step
        ):  # fast forward the iterator if loading from a previous checkpoint.
          next(iterator)
          self._iter_steps += 1

        # Fetch one training micro-batch
        example = next(iterator)
        cur_batch_size = len(example["prompts"])
        buf.append(example)
        buf_sizes.append(cur_batch_size)
        buf_b += cur_batch_size
        consumed_steps += 1

        # If the LCM threshold is reached, produce one batch
        produced_now, buf, buf_sizes, buf_b = (
            self.aggregate_and_compute_advantages(
                buf=buf,
                buf_sizes=buf_sizes,
                buf_b=buf_b,
                service_target_bs=service_target_bs,
                sample_repeat=sample_repeat,
                mode=mode,
                force=False,
            )
        )
        if produced_now:
          if async_loading:
            # Async: Enqueue immediately
            enqueue_examples(produced_now, 1)
            if batch_repeat > 1:
              pending_examples.extend(produced_now)
          else:
            # Sync: accumulate; at boundary finalize with batch_repeat times
            pending_examples.extend(produced_now)

        if mode == rl_cluster_lib.Mode.TRAIN:
          self._iter_steps += 1
        else:
          self._eval_steps += 1

        # On proceed boundary: handle tail + enqueue repeats
        if proceed_num_steps > 0 and consumed_steps == proceed_num_steps:
          tail, buf, buf_sizes, buf_b = self.aggregate_and_compute_advantages(
              buf=buf,
              buf_sizes=buf_sizes,
              buf_b=buf_b,
              service_target_bs=service_target_bs,
              sample_repeat=sample_repeat,
              mode=mode,
              force=True,
          )
          if tail:
            if async_loading:
              enqueue_examples(tail, 1)
              if batch_repeat > 1:
                pending_examples.extend(tail)
            else:
              pending_examples.extend(tail)

          if pending_examples:
            if not async_loading:
              enqueue_examples(pending_examples, batch_repeat)
            else:
              rem = batch_repeat - 1
              if rem > 0:
                enqueue_examples(pending_examples, rem)
            pending_examples.clear()

          consumed_steps = 0
          return
    except StopIteration as e:
      if proceed_num_steps > 0:
        raise e
      else:
        tail, buf, buf_sizes, buf_b = self.aggregate_and_compute_advantages(
            buf=buf,
            buf_sizes=buf_sizes,
            buf_b=buf_b,
            service_target_bs=service_target_bs,
            sample_repeat=sample_repeat,
            mode=mode,
            force=True,
        )
        if tail:
          pending_examples.extend(tail)
        if pending_examples:
          enqueue_examples(pending_examples, batch_repeat)
          pending_examples.clear()
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
    print("begin to train")
    train_iterator = iter(train_ds)
    first_item = next(train_iterator)
    input_batch_size = len(first_item["prompts"])
    train_iterator = chain([first_item], train_iterator)
    training_config = self.rl_cluster.cluster_config.training_config
    if training_config.training_micro_batch_size is not None:
      assert training_config.training_micro_batch_size == input_batch_size, (
          "Training micro batch size must be equal to input batch size. "
          f"Got {training_config.training_micro_batch_size} and "
          f"{input_batch_size}."
      )
    else:
      training_config.training_micro_batch_size = input_batch_size
    if training_config.rollout_micro_batch_size is None:
      training_config.rollout_micro_batch_size = input_batch_size
    if training_config.ref_logps_micro_batch_size is None:
      training_config.ref_logps_micro_batch_size = input_batch_size
    if training_config.old_logps_micro_batch_size is None:
      training_config.old_logps_micro_batch_size = input_batch_size
      
    while True:  # loop over M
      try:
        # reserve 1 for None and the other for repeated interable
        # if batch_repeat > 1
        train_data_queue = queue_lib.SimpleDataQueue(
            maxsize=self.grad_acc_steps * self.grpo_config.num_iterations + 1
        )
        # Use an unbounded queue for evaluation data.
        eval_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
        initial_steps = self._iter_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            proceed_num_steps=self.grad_acc_steps,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_steps
        ):
          while True:
            with sft_utils.time_measure(suppress_logging=True) as timer:
              curr_train_ds = train_data_queue.get(block=True)

            if curr_train_ds is None:
              break

            if self.can_enable_async_rollout:
              self.rl_cluster.buffer_metrics(
                  {
                      "actor_dequeue_time": (
                          timer(),
                          np.mean,
                      ),
                  },
                  mode=rl_cluster_lib.Mode.TRAIN,
              )

            if (
                eval_ds
                and not curr_eval_ds
                and self.rl_cluster.actor_trainer.train_steps
                % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                == 0
            ):
              self._prepare_data(
                  iterator=iter(eval_ds),
                  proceed_num_steps=-1,
                  sample_repeat=self.grpo_config.num_generations,
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
        # call to throw stop iteration as a singal to break the loop
        future.result()
        # sync the iter steps with internel trainer, this is based on the
        # assumption that the trainer internally doesn't reset the iter steps.
        # there is current a unit test to ensure this assumption.
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
    kl = common.compute_kl_divergence(
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
