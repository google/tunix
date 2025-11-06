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

"""Abstract RL learner class."""

from __future__ import annotations

import abc
from concurrent import futures
import itertools
import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

from absl import logging
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

ABC = abc.ABC
abstractmethod = abc.abstractmethod

TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]

MetricFn = Callable[..., rl_cluster_lib.MetricsT]


class RLLearner(ABC):
  """Base class that should be extended by specific RL algorithms."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `RLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn):
        >>> def metric_fn(prompts, completions, rewards, advantages, **kargs):
        ...    return { ...        "prompt_min_len": (min(len(p) for p in
        prompts), np.min), ...        ... ...    }
      data_shuffle_seed: The seed for shuffling the data.
    """
    self.rl_cluster = rl_cluster
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    self.metric_fns = metric_fns or []
    self.rl_cluster.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_cluster, "critic_trainer"):
      self.rl_cluster.critic_trainer.is_managed_externally = True

    self._data_shuffle_seed = (
        jax.random.PRNGKey(data_shuffle_seed)
        if data_shuffle_seed is not None
        else None
    )

    self._training_config = self.rl_cluster.cluster_config.training_config

    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.restored_global_step()
    )

    self._iter_steps = 0
    self._eval_iter_steps = 0

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

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size
    )
    sft_utils.show_hbm_usage(title="RLLearner init")

  @abstractmethod
  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> common.TrainExample:
    pass

  @abstractmethod
  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    pass

  @abstractmethod
  def _num_iterations(self) -> int:
    pass

  @abstractmethod
  def _num_generations(self) -> int:
    pass

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
      A JAX array (shape `[num_prompts]`) of scalar rewards for
      each prompt-completion pair. The rewards are the sum across all the
      provided
      reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    num_prompts = len(prompts)
    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kwargs)

      if r is None:
        raise RuntimeError(
            f"Failed to obtain result from {reward_fn.__name__}. Result is"
            " None."
        )
      if isinstance(r, list) and len(r) != len(prompts):
        raise RuntimeError(
            f"Length mismatch after {reward_fn.__name__}: "
            f"len(r)={len(r)}, len(prompts)={num_prompts}. "
            f"Content of r: {r}"
        )

      rewards[:, i] = np.array(r)

    # Sum rewards across all reward functions for each prompt.
    sum_rewards = np.nansum(rewards, axis=1)

    # Log all metrics in a single loop
    for j, (prompt, completion) in enumerate(zip(prompts, completions)):
      metrics_to_log = {}

      # Log prompts and completions.
      metrics_to_log["prompts"] = (prompt, None)
      metrics_to_log["completions"] = (completion, None)

      # Log the summed rewards for this trajectory.
      trajectory_sum = sum_rewards[j]
      metrics_to_log["rewards/sum"] = (trajectory_sum, np.mean)
      metrics_to_log["rewards/min"] = (np.min(rewards[j]), np.min)
      metrics_to_log["rewards/max"] = (np.max(rewards[j]), np.max)

      # Log individual rewards for this trajectory
      for i, reward_fn in enumerate(self.reward_fns):
        metric_name = f"rewards/{reward_fn.__name__}"
        metrics_to_log[metric_name] = (rewards[j, i], np.mean)

      # Log all metrics for this trajectory in one call
      self.rl_cluster.buffer_metrics(metrics_to_log, mode=mode)

    return jnp.array(sum_rewards)

  def _process_accumulated_batches(
      self,
      micro_batches: list[TrainingInputT],
      micro_batch_sizes: list[int],
      mode: rl_cluster_lib.Mode,
  ) -> list[common.TrainExample]:
    """Merges, repeats, and computes advantages for a buffer of examples.

    This function takes a buffer of micro-batches, merges them, repeats the
    samples, runs a single large forward pass to generate completions and
    compute advantages, and then splits the results back into micro-batches.

    Args:
      micro_batches: A list of training micro-batches.
      micro_batch_sizes: A list of the number of samples for each training
        micro-batch.
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
    produced: list[common.TrainExample] = []
    offset = 0

    for n in micro_batch_sizes:
      cur_slice = slice(offset, offset + n)  # Calculate slice indices
      training_example = rl_utils.get_batch_slice(combined_batch, cur_slice)
      produced.append(training_example)
      offset += n

    return produced

  def _prepare_data(
      self,
      iterator: Iterator[TrainingInputT],
      proceed_num_steps: int,
      sample_repeat: int,
      batch_repeat: int,
      service_target_batch_size: int,
      data_queue: queue_lib.AbstractDataQueue[list[common.TrainExample] | None],
      async_loading: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> None:
    """Orchestrates the data preparation pipeline.

    This method is designed to efficiently process data in micro-batches while
    accommodating the requirements of different model computations (e.g.
    rollout, ref_logps, old_logps, training) that may have different optimal
    batch sizes.

    The pipeline follows these main steps:
    1. **Merge**: It consumes multiple small micro-batches from the input
       `iterator` and merges them into a single, larger batch. This is done to
       meet the `service_target_bs`, which is the least common multiple of the
       micro-batch sizes for different services, ensuring efficient hardware
       utilization.
    2. **Repeat (Sample)**: Each prompt in the merged batch is repeated
       `sample_repeat` times based on the algorithm needs(e.g. G in GRPO).
       For algorithmrs that don't repeat prompts, this will be 1 (e.g. PPO).
    3. **Single Large Forward Pass**: The resulting large batch of repeated
       prompts is then processed in a single call to
       `_generate_and_compute_advantage`. This function handles text generation,
       reward computation, and advantage calculation for the entire batch.
    4. **Split**: After processing, the large `TrainExample` is split back into
       smaller chunks that correspond to the original input micro-batches.
    5. **Enqueue**: These smaller `TrainExample` chunks are then put into the
       `data_queue` to be consumed by the training loop.

    Args:
      iterator: An iterator yielding `TrainingInputT` examples.
      proceed_num_steps: The number of training micro-batches to process before
        returning. If > 0, the function will stop after consuming this many
        steps. If -1, it will continue until the iterator is exhausted.
      sample_repeat: The number of times each sample in a micro-batch is
        repeated during the advantage computation. This is typically
        `grpo_config.num_generations`.
      batch_repeat: The number of times the produced `TrainExample` batch should
        `grpo_config.num_iterations`.
      service_target_batch_size: largest common multiple of rollout and
        compute_logps micro-batch sizes. This is used to accumulate
        micro-batches between the rollout and inference computation.
      data_queue: The queue to which lists of `TrainExample` are added.
      async_loading: If True, enqueue each produced micro-batch immediately in
        async mode. Otherwise, accumulate and enqueue at the boundary.
      mode: The metrics logger mode, either `metrics_logger.Mode.TRAIN` or
        `metrics_logger.Mode.EVAL`.
    """
    # A buffer to accumulate micro-batches before processing them together.
    # Num of examples per micro-batch is train_micro_batch_size * sample_repeat.
    micro_batches: list[TrainingInputT] = []
    # Number of samples for each micro-batch
    micro_batch_sizes: list[int] = []
    # Aggregated sample count (before repeating)
    accumulated_samples_num = 0
    # Number of consumed training micro-batches
    consumed_steps = 0

    pending_examples: list[common.TrainExample] = []

    def enqueue_examples(
        examples: list[common.TrainExample], repeats: int
    ) -> None:
      """Wrap each TrainExample as [TrainExample] and put it into the queue, repeated `repeats`."""
      if repeats <= 0 or not examples:
        return
      for _ in range(repeats):
        if self._data_shuffle_seed is not None:
          shuffle_seed, self._data_shuffle_seed = jax.random.split(
              self._data_shuffle_seed
          )
          shuffled_indices = jax.random.permutation(shuffle_seed, len(examples))
          for i in shuffled_indices:
            data_queue.put([examples[i]])
        else:
          for example in examples:
            data_queue.put([example])

    def _enqueue_or_buffer_examples(produced: list[common.TrainExample]):
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
          if self._iter_steps == self._last_iter_step:
            logging.info("Fast forwarded %d micro-batches.", self._iter_steps)

        # Fetch one training micro-batch
        example = next(iterator)
        cur_batch_size = len(example["prompts"])

        # Buffer the fetched micro-batch. We accumulate micro-batches and track
        # their sizes and the total number of samples. This allows us to form a
        # larger batch for processing once `accumulated_samples_num` reaches the
        # `service_target_batch_size` threshold.
        micro_batch_sizes.append(cur_batch_size * sample_repeat)
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
            else self._eval_iter_steps,
        )
        assert "trajectory_ids" not in example
        example["trajectory_ids"] = trajectory_ids

        for t_id in trajectory_ids:
          self.rl_cluster.buffer_metrics(
              {
                  "trajectory_ids": (t_id, None),
              },
              mode=mode,
          )

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._iter_steps
            if mode == rl_cluster_lib.Mode.TRAIN
            else self._eval_iter_steps,
        ):
          # If the LCM threshold is reached, produce one batch
          produced_training_examples = []
          if accumulated_samples_num >= service_target_batch_size:
            produced_training_examples = self._process_accumulated_batches(
                micro_batches=micro_batches,
                micro_batch_sizes=micro_batch_sizes,
                mode=mode,
            )
            micro_batches.clear()
            micro_batch_sizes.clear()
            accumulated_samples_num = 0
          _enqueue_or_buffer_examples(produced_training_examples)

        if mode == rl_cluster_lib.Mode.TRAIN:
          self._iter_steps += 1
        else:
          self._eval_iter_steps += 1

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

  def _create_micro_batch_iterator(
      self,
      full_batch_iterator: Iterator[TrainingInputT],
      micro_batch_size: int,
  ) -> Iterator[TrainingInputT]:
    """Re-batches large inputs into an iterator of micro-batches.

    Args:
      full_batch_iterator: Iterator yielding large `TrainingInputT` batches.
      micro_batch_size: The desired size of the micro-batches.

    Yields:
      `TrainingInputT` dicts, each with `micro_batch_size` samples.
    """
    buffer = {}

    def get_buffer_len(buf: dict[str, list[Any]]) -> int:
      if not buf:
        return 0
      return len(next(iter(buf.values())))

    for large_batch in full_batch_iterator:
      for key, values in large_batch.items():
        if key not in buffer:
          buffer[key] = []

        if isinstance(values, (np.ndarray, jax.Array)):
          buffer[key].extend(list(values.flatten()))
        elif isinstance(values, (list, tuple)):
          buffer[key].extend(values)
        else:
          buffer[key].append(values)

      while get_buffer_len(buffer) >= micro_batch_size:
        micro_batch = {}
        for key in buffer:
          micro_batch_list_slice = buffer[key][:micro_batch_size]
          micro_batch[key] = np.array(micro_batch_list_slice)
          buffer[key] = buffer[key][micro_batch_size:]

        yield micro_batch

  def train(
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main entry point for the training loop."""
    full_batch_iterator = iter(train_ds)
    first_item = next(full_batch_iterator)
    full_batch_size = len(first_item["prompts"])
    full_batch_iterator = itertools.chain([first_item], full_batch_iterator)
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    self._rollout_micro_batch_size = (
        self._rollout_micro_batch_size or train_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._compute_logps_micro_batch_size or train_micro_batch_size
    )
    for v, n in [
        (self._rollout_micro_batch_size, f"{self._rollout_micro_batch_size=}"),
        (
            self._compute_logps_micro_batch_size,
            f"{self._compute_logps_micro_batch_size=}",
        ),
        (mini_batch_size, f"{mini_batch_size=}"),
    ]:
      rl_utils.check_divisibility(v, full_batch_size, n, f"{full_batch_size=}")
    grad_acc_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=}, {grad_acc_steps=}"
    )

    service_target_batch_size = math.lcm(
        self._rollout_micro_batch_size,
        self._compute_logps_micro_batch_size,
    )

    # if the micro batch size is the same as the full batch size, we can use the
    # full batch iterator directly.
    if train_micro_batch_size == full_batch_size:
      train_iterator = full_batch_iterator
    else:
      train_iterator = self._create_micro_batch_iterator(
          full_batch_iterator, train_micro_batch_size
      )

    while True:  # loop over M
      try:
        initial_steps = self._iter_steps
        for _ in range(full_batch_size // mini_batch_size):
          # reserve 1 for None and the other for repeated interable
          # if batch_repeat > 1
          train_data_queue = queue_lib.SimpleDataQueue(
              maxsize=grad_acc_steps * self._num_iterations() + 1
          )
          # Use an unbounded queue for evaluation data.
          eval_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
          initial_steps = self._iter_steps
          future = self.executor.submit(
              self._prepare_data,
              iterator=train_iterator,
              proceed_num_steps=grad_acc_steps,
              sample_repeat=self._num_generations(),
              batch_repeat=self._num_iterations(),
              service_target_batch_size=service_target_batch_size,
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
                self._eval_iter_steps = 0
                self._prepare_data(
                    iterator=iter(eval_ds),
                    proceed_num_steps=-1,
                    sample_repeat=self._num_generations(),
                    batch_repeat=1,
                    service_target_batch_size=service_target_batch_size,
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
              if hasattr(self.rl_cluster, "critic_trainer"):
                self.rl_cluster.update_critic(
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
