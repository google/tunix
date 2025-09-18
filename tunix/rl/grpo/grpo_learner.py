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
    if self.num_generations <= 1:
      raise ValueError(
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
    if grpo_config.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
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

  def _compute_trajectory_ids(
      self, example: _TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is a string of format {row_offset}_{group_offset} where
    row_offset is the row index of the example data source and
    group_offset is the group index of the example in the generation group.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self.grpo_config.num_generations
    row_offset = steps * batch_size
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
              training_config.compute_logps_micro_batch_size
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
              training_config.compute_logps_micro_batch_size
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
      **kwargs,
  ) -> jax.Array:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts, num_reward_fns]`) of scalar rewards for
      each prompt-completion pair. The rewards are computed using the provided
      reward functions.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kwargs)
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

  def _initialize_micro_batch_sizes(self, input_batch_size: int):
    """Initializes micro batch sizes in training_config if not set."""
    training_config = self.rl_cluster.cluster_config.training_config
    if (
        training_config.training_micro_batch_size
        and training_config.training_micro_batch_size != input_batch_size
    ):
      raise ValueError(
          "Training micro batch size must be equal to input batch size. "
          f"Got {training_config.training_micro_batch_size} and "
          f"{input_batch_size}."
      )
    training_config.training_micro_batch_size = input_batch_size
    for attr in (
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
    ):
      if getattr(training_config, attr) is None:
        setattr(training_config, attr, input_batch_size)

    training_micro_batch_size = training_config.training_micro_batch_size
    for attr in (
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
    ):
      micro_batch_size = getattr(training_config, attr)
      if micro_batch_size < training_micro_batch_size:
        raise ValueError(
            f"{attr} ({micro_batch_size}) must be greater than or equal to "
            f"training_micro_batch_size ({training_micro_batch_size})."
        )
      if micro_batch_size % training_micro_batch_size != 0:
        raise ValueError(
            f"{attr} ({micro_batch_size}) must be a multiple of "
            f"training_micro_batch_size ({training_micro_batch_size})."
        )

  def _process_accumulated_batches(
      self,
      micro_batches: list[_TrainingInputT],
      micro_batch_sizes: list[int],
      sample_repeat: int,
      mode: rl_cluster_lib.Mode,
  ) -> list[TrainExample]:
    """Merges, repeats, and computes advantages for a buffer of examples.

    This function takes a buffer of micro-batches, merges them, repeats the
    samples, runs a single large forward pass to generate completions and
    compute advantages, and then splits the results back into micro-batches.

    Args:
      micro_batches: A list of training micro-batches.
      micro_batch_sizes: A list of the number of samples for each training
        micro-batch.
      sample_repeat: The number of times each sample is repeated.
      mode: The mode to use for logging metrics.

    Returns:
      A list of small TrainExample chunks, split back by original micro
      boundaries.
    """
    if not micro_batches:
      return []

    # Merge multiple training micro-batches
    merged = rl_utils.merge_micro_batches(micro_batches)

    combined_batch = self._generate_and_compute_advantage(merged, mode)

    # Split back to original training micro size
    produced: list[TrainExample] = []
    offset = 0

    for n in micro_batch_sizes:
      # Calculate slice indices
      start_idx = offset * sample_repeat
      end_idx = (offset + n) * sample_repeat
      token_slice = slice(start_idx, end_idx)
      training_example = rl_utils.get_batch_slice(combined_batch, token_slice)
      produced.append(training_example)
      offset += n

    return produced

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

    service_target_batch_size = math.lcm(
        training_config.rollout_micro_batch_size,
        training_config.compute_logps_micro_batch_size,
    )

    # A buffer to accumulate micro-batches before processing them together.
    micro_batches: list[_TrainingInputT] = []
    # Number of samples for each micro-batch
    micro_batch_sizes: list[int] = []
    # Aggregated sample count (before repeating)
    accumulated_samples_num = 0
    # Number of consumed training micro-batches
    consumed_steps = 0

    pending_examples: list[TrainExample] = []

    def enqueue_examples(examples: list[TrainExample], repeats: int) -> None:
      """Wrap each TrainExample as [TrainExample] and put it into the queue, repeated `repeats`."""
      if repeats <= 0 or not examples:
        return
      for _ in range(repeats):
        for example in examples:
          data_queue.put([example])

    def _enqueue_or_buffer_examples(produced: list[TrainExample]):
      """Enqueues produced examples or adds them to a temporary buffer."""
      if not produced:
        return
      if async_loading:
        enqueue_examples(produced, 1)
      if not async_loading or batch_repeat > 1:
        pending_examples.extend(produced)

    def _process_and_enqueue_tail():
      """Processes any remaining micro-batches and enqueues them."""
      tail_examples = self._process_accumulated_batches(
          micro_batches=micro_batches,
          micro_batch_sizes=micro_batch_sizes,
          sample_repeat=sample_repeat,
          mode=mode,
      )
      micro_batches.clear()
      micro_batch_sizes.clear()

      repeats = 1 if mode == rl_cluster_lib.Mode.EVAL else batch_repeat

      # For evaluation, or training without async loading, buffer the tail and
      # tail and enqueue all pending examples.
      if mode == rl_cluster_lib.Mode.EVAL or not async_loading:
        if tail_examples:
          pending_examples.extend(tail_examples)
        if pending_examples:
          enqueue_examples(pending_examples, repeats)
          pending_examples.clear()
        return

      # --- Handle Asynchronous Training ---
      _enqueue_or_buffer_examples(tail_examples)
      if pending_examples:
        remaining_repeats = repeats - 1
        if remaining_repeats > 0:
          enqueue_examples(pending_examples, remaining_repeats)
        pending_examples.clear()

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

        # Buffer the fetched micro-batch. We accumulate micro-batches and track
        # their sizes and the total number of samples. This allows us to form a
        # larger batch for processing once `accumulated_samples_num` reaches the
        # `service_target_batch_size` threshold.
        micro_batch_sizes.append(cur_batch_size)
        accumulated_samples_num += cur_batch_size
        consumed_steps += 1

        example = jax.tree.map(
            lambda x: np.repeat(x, sample_repeat, axis=0),
            example,
        )  # [B] -> [B * G]

        micro_batches.append(example)
        # Compute trajectory ids for the current batch.
        trajectory_ids = self._compute_trajectory_ids(
            example,
            self._iter_steps
            if mode == rl_cluster_lib.Mode.TRAIN
            else self._eval_steps,
        )
        assert "trajectory_ids" not in example
        example["trajectory_ids"] = trajectory_ids

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._iter_steps
            if mode == rl_cluster_lib.Mode.TRAIN
            else self._eval_steps,
        ):
          # If the LCM threshold is reached, produce one batch
          produced_training_examples = []
          if accumulated_samples_num >= service_target_batch_size:
            produced_training_examples = self._process_accumulated_batches(
                micro_batches=micro_batches,
                micro_batch_sizes=micro_batch_sizes,
                sample_repeat=sample_repeat,
                mode=mode,
            )
            micro_batches.clear()
            micro_batch_sizes.clear()
            accumulated_samples_num = 0
          _enqueue_or_buffer_examples(produced_training_examples)

        if mode == rl_cluster_lib.Mode.TRAIN:
          self._iter_steps += 1
        else:
          self._eval_steps += 1

        # On proceed boundary: handle tail + enqueue repeats
        # The "tail" is the current buffer. If we haven't collected a large
        # enough batch of data, we don't process it immediately but instead
        # temporarily store it in the buffer.
        # There are two cases where we need to force a flush of the tail:
        # 1. The dataset is exhausted (see StopIteration handling).
        # 2. The gradient accumulation steps are reached, completing an
        #    effective batch for a parameter update, which requires a forced
        #    flush.
        if proceed_num_steps > 0 and consumed_steps == proceed_num_steps:
          _process_and_enqueue_tail()
          return
    except StopIteration as e:
      if proceed_num_steps > 0:
        raise e
      else:
        _process_and_enqueue_tail()
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
    first_item = next(train_iterator)
    input_batch_size = len(first_item["prompts"])
    train_iterator = chain([first_item], train_iterator)
    self._initialize_micro_batch_sizes(input_batch_size)

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
              self._eval_steps = 0
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
