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
import os
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Sequence, TypeVar

from absl import logging
import jax
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import reward_manager
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import sharding_utils
from tunix.sft import utils as sft_utils
from tunix.utils import compat


ABC = abc.ABC
abstractmethod = abc.abstractmethod

TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]

MetricFn = Callable[..., rl_engine_lib.MetricsT]

TConfig = TypeVar("TConfig", bound=algo_config_lib.AlgorithmConfig)


def _env_flag(name: str, default: str = "0") -> bool:
  """True iff the environment variable `name` is set to 1/true/True."""
  return (os.environ.get(name, default) or default).strip() in (
      "1",
      "true",
      "True",
  )


def per_host_learner_enabled() -> bool:
  """Whether the per-host learner is on (TUNIX_PER_HOST_LEARNER=1, >1 process).

  Multi-controller JAX (one process per host, `jax.distributed`, no Pathways)
  with a host-local in-process rollout engine: every process feeds the same
  global batch stream, takes its own 1/N prompt shard, rolls it out on its
  own engine, computes rewards and advantages for it, and feeds it to the
  shared train step as process-local data (`shard_input` assembles the global
  array with `jax.make_array_from_process_local_data`). The train step's
  gradient reduction is the only cross-host step of a cycle besides the
  weight publish. All configured batch sizes stay GLOBAL and are divided by
  the process count, so one optimizer update still covers one global
  mini-batch.
  """
  return _env_flag("TUNIX_PER_HOST_LEARNER") and jax.process_count() > 1


def shard_batch_by_process(
    batch: TrainingInputT,
    process_index: int,
    process_count: int,
    indices: Sequence[int] | None = None,
) -> TrainingInputT:
  """Returns one process's prompt shard of a global batch.

  Shards before the per-prompt repeat (`num_generations`), so every group of a
  prompt stays on one process and GRPO's within-group advantage is unchanged.

  Args:
    batch: A global batch; batch-aligned columns have `len(batch["prompts"])`
      rows, other entries are passed through.
    process_index: This process's index.
    process_count: The number of processes.
    indices: Rows of this process; defaults to the stride
      `[process_index::process_count]`.

  Returns:
    The process-local batch.
  """
  if not isinstance(batch, dict) or "prompts" not in batch:
    raise ValueError("per-host learner: batch must be a dict with 'prompts'.")
  num_rows = len(batch["prompts"])
  if num_rows % process_count:
    raise ValueError(
        f"per-host learner: global batch of {num_rows} prompts is not divisible"
        f" by the {process_count} processes; every process must run the same"
        " number of train steps."
    )
  if indices is None:
    rows = slice(process_index, None, process_count)
  else:
    rows = np.asarray(indices, dtype=np.int64)
  out = {}
  for key, value in batch.items():
    try:
      aligned = len(value) == num_rows and not isinstance(value, (str, bytes))
    except TypeError:
      aligned = False
    if not aligned:
      out[key] = value
    elif isinstance(value, (np.ndarray, jax.Array)):
      out[key] = value[rows]
    else:
      value = list(value)
      out[key] = (
          value[rows] if isinstance(rows, slice) else [value[i] for i in rows]
      )
  return out


def balanced_shard_indices(
    lengths: Sequence[int], process_count: int
) -> List[List[int]]:
  """Equal-count, length-balanced partition of rows over processes.

  Sorts rows by length (longest first, ties by index) and deals them in rounds
  of `process_count`, reversing the direction every round (snake). Every
  process gets exactly `len(lengths) / process_count` rows when that divides,
  and per-process length totals end up nearly equal. Deterministic, so every
  process computes the same partition without communication.

  Args:
    lengths: Per-row length (a proxy for rollout cost).
    process_count: The number of processes.

  Returns:
    For every process, its sorted row indices.
  """
  order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
  bins = [[] for _ in range(process_count)]
  for round_index, start in enumerate(range(0, len(order), process_count)):
    hosts = range(process_count)
    if round_index % 2:
      hosts = reversed(hosts)
    for host, row in zip(hosts, order[start : start + process_count]):
      bins[host].append(row)
  return [sorted(b) for b in bins]


class _ProcessShardedIterable:
  """Re-iterable view of a batch iterable that yields this process's shards."""

  def __init__(
      self,
      iterable: Iterable[TrainingInputT],
      shard_fn: Callable[[TrainingInputT], TrainingInputT],
  ):
    self._iterable = iterable
    self._shard_fn = shard_fn

  def __iter__(self) -> Iterator[TrainingInputT]:
    for batch in self._iterable:
      yield self._shard_fn(batch)


class RLLearner(abc.ABC, Generic[TConfig]):
  """Base class that should be extended by specific RL algorithms."""

  @compat.alias_init_param("rl_cluster", "rl_engine")
  def __init__(
      self,
      rl_engine: rl_engine_lib.RLEngine,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `RLLearner`.

    Args:
      rl_engine: RL engine containing actor, reference and reward models.
      algo_config: An instance of `AlgorithmConfig` containing all
        training-specific configuration options.
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
    self.rl_engine = rl_engine
    self.algo_config = algo_config

    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_manager
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )

    self.metric_fns = metric_fns or []
    self.rl_engine.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_engine, "critic_trainer"):
      self.rl_engine.critic_trainer.is_managed_externally = True

    self._data_shuffle_seed = (
        jax.random.PRNGKey(data_shuffle_seed)
        if data_shuffle_seed is not None
        else None
    )

    self._training_config = self.rl_engine.cluster_config.training_config
    self._packing_enabled = rl_engine_lib.configs.is_sequence_packing_enabled(
        self._training_config
    )

    if self._packing_enabled:
      assert self._training_config.max_seq_token_per_tpu is not None
      # Fail now rather than mid-run: a maximal sequence must fit one packed
      # row. `max_response_length` lives on the algo config for agentic
      # learners and on the rollout config elsewhere.
      rollout_config = self.rl_engine.cluster_config.rollout_config

      if isinstance(rollout_config, dict):
        r_config = rollout_config.get(rl_engine_lib.Mode.TRAIN) or next(
            iter(rollout_config.values())
        )
      else:
        r_config = rollout_config

      rl_utils.validate_packing_budget(
          self._training_config.max_seq_token_per_tpu,
          getattr(r_config, "max_prompt_length", 1000000),
          getattr(algo_config, "max_response_length", None)
          or r_config.max_tokens_to_generate,
      )

    self.rl_engine.global_steps = (
        self.rl_engine.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = 0
    self._eval_iter_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
            self.rl_engine.actor_trainer.model,
            self.rl_engine.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_engine.cluster_config.role_to_mesh[rl_engine_lib.Role.ACTOR]
        != self.rl_engine.cluster_config.role_to_mesh[
            rl_engine_lib.Role.ROLLOUT
        ]
    )
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self._last_iter_step = self.rl_engine.actor_trainer.iter_steps

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
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
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
      mode: rl_engine_lib.Mode,
      step: int | None = None,
      **kwargs,
  ) -> np.ndarray:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      step: The current training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A numpy array (shape `[B]`) of scalar rewards for
      each prompt-completion pair. The rewards are the sum across all the
      provided reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    rewards_info = self.reward_manager(
        prompts=prompts,
        completions=completions,
        **kwargs,
    )

    if step is not None:
      self.rl_engine.buffer_metrics_async(
          rewards_info["log_metrics"], mode=mode, step=step
      )
    else:
      self.rl_engine.buffer_metrics(rewards_info["log_metrics"], mode=mode)

    return rewards_info["rewards"]

  def _process_accumulated_batches(
      self,
      micro_batches: list[TrainingInputT],
      micro_batch_sizes: list[int],
      mode: rl_engine_lib.Mode,
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
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
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

      repeats = 1 if mode == rl_engine_lib.Mode.EVAL else batch_repeat

      # For evaluation, or training without async loading, buffer the tail and
      # tail and enqueue all pending examples.
      if mode == rl_engine_lib.Mode.EVAL or not async_loading:
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
            mode == rl_engine_lib.Mode.TRAIN
            and self._iter_steps < self._last_iter_step
        ):  # fast forward the iterator if loading from a previous checkpoint.
          next(iterator)
          self._iter_steps += 1
          if self._iter_steps == self._last_iter_step:
            logging.info("Fast forwarded %d micro-batches.", self._iter_steps)

        with self.rl_engine.perf.span(
            "data_loading"
        ), self.rl_engine.perf_v2.span(
            perf_constants.DATA_LOADING,
            tags={
                perf_constants.STEP: self.rl_engine.global_steps,
            },
        ):
          # Fetch one training micro-batch
          example = next(iterator)
          cur_batch_size = len(example["prompts"])  # pyrefly: ignore[bad-argument-type]

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
            if mode == rl_engine_lib.Mode.TRAIN
            else self._eval_iter_steps,
        )
        assert "trajectory_ids" not in example
        example["trajectory_ids"] = trajectory_ids

        for t_id in trajectory_ids:
          self.rl_engine.buffer_metrics(
              {
                  "trajectory_ids": (t_id, None),
              },
              mode=mode,
          )

        with jax.profiler.StepTraceAnnotation(
            "sampler",
            step_num=self._iter_steps
            if mode == rl_engine_lib.Mode.TRAIN
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

        if mode == rl_engine_lib.Mode.TRAIN:
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
    per_host = per_host_learner_enabled()
    local_size = self._per_host_size if per_host else (lambda size, _: size)
    if per_host:
      if self._packing_enabled:
        raise NotImplementedError(
            "TUNIX_PER_HOST_LEARNER=1 does not support sequence packing: the"
            " number of packed chunks is data dependent, so processes would"
            " run different numbers of train steps."
        )
      logps_passes = self._per_host_trainer_side_logps()
      if logps_passes and not sharding_utils.per_host_safe_logps_enabled():
        raise ValueError(
            "TUNIX_PER_HOST_LEARNER=1 with "
            + " and ".join(logps_passes)
            + " needs TUNIX_PER_HOST_SAFE_LOGPS=1: the stock"
            " RLCluster.get_ref_per_token_logps/get_actor_per_token_logps"
            " slice the globalised batch with process-local offsets, so every"
            " process but the first would score another process's rows."
        )
      train_ds = _ProcessShardedIterable(
          train_ds, self._shard_batch_for_process
      )
      if eval_ds is not None:
        eval_ds = _ProcessShardedIterable(
            eval_ds, self._shard_batch_for_process
        )
      self._rollout_micro_batch_size = local_size(
          self._training_config.rollout_micro_batch_size,
          "rollout_micro_batch_size",
      )
      self._compute_logps_micro_batch_size = local_size(
          self._training_config.compute_logps_micro_batch_size,
          "compute_logps_micro_batch_size",
      )
    full_batch_iterator = iter(train_ds)
    first_item = next(full_batch_iterator)
    full_batch_size = len(first_item["prompts"])  # pyrefly: ignore[bad-argument-type]
    full_batch_iterator = itertools.chain([first_item], full_batch_iterator)
    # Initialize batch sizes.
    mini_batch_size = (
        local_size(self._training_config.mini_batch_size, "mini_batch_size")
        or full_batch_size
    )
    if self._packing_enabled:
      train_micro_batch_size = None
      grad_acc_steps = None
    else:
      train_micro_batch_size = (
          local_size(
              self._training_config.train_micro_batch_size,
              "train_micro_batch_size",
          )
          or mini_batch_size
      )
      grad_acc_steps = self._training_config.get_with_default(
          "gradient_accumulation_steps", 1
      )
    # Prompts fetched per iterator step (`mini_batch_size` when packing is on)
    # and the number of iterator steps needed to assemble one mini-batch.
    iterator_batch_size = train_micro_batch_size or mini_batch_size
    iterator_steps_per_mini_batch = mini_batch_size // iterator_batch_size
    self._rollout_micro_batch_size = (
        self._rollout_micro_batch_size or iterator_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._compute_logps_micro_batch_size or iterator_batch_size
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

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=}, {grad_acc_steps=},"
        f" {iterator_steps_per_mini_batch=}"
    )

    service_target_batch_size = math.lcm(
        self._rollout_micro_batch_size,
        self._compute_logps_micro_batch_size,
    )
    if per_host:
      self._check_per_host_data_axis(
          iterator_batch_size,
          self._compute_logps_micro_batch_size
          if self._per_host_trainer_side_logps()
          else None,
      )
      logging.info(
          "Per-host learner (TUNIX_PER_HOST_LEARNER=1): process %d/%d trains"
          " on its own prompt shard: full_batch_size=%d mini_batch_size=%d"
          " train_micro_batch_size=%s rollout_micro_batch_size=%d"
          " compute_logps_micro_batch_size=%d (global sizes / %d).",
          jax.process_index(),
          jax.process_count(),
          full_batch_size,
          mini_batch_size,
          train_micro_batch_size,
          self._rollout_micro_batch_size,
          self._compute_logps_micro_batch_size,
          jax.process_count(),
      )

    # if the iterator batch size is the same as the full batch size, we can
    # use the full batch iterator directly.
    if iterator_batch_size == full_batch_size:
      train_iterator = full_batch_iterator
    else:
      train_iterator = self._create_micro_batch_iterator(
          full_batch_iterator, iterator_batch_size
      )

    try:
      while True:  # loop over M
        try:
          initial_steps = self._iter_steps

          with self.rl_engine.perf.span_group("global_step"):
            self._run_global_step(
                full_batch_size,
                mini_batch_size,
                service_target_batch_size,
                iterator_steps_per_mini_batch,
                train_iterator,
                eval_ds,
                skip_jit,
            )

            if self.should_sync_weights:
              logging.debug(
                  "Syncing weights at global step %s mini batch step %s",
                  self.rl_engine.global_steps,
                  self._iter_steps,
              )
              with self.rl_engine.perf.span(
                  "weight_sync", self.rl_engine.perf.all_devices
              ), self.rl_engine.perf_v2.span(
                  perf_constants.WEIGHT_SYNC,
                  self.rl_engine.perf_v2.all_devices,
                  tags={
                      perf_constants.STEP: self.rl_engine.global_steps,
                  },
              ):
                with jax.profiler.StepTraceAnnotation(
                    "sync_sampler_weights", step_num=initial_steps
                ):
                  self.rl_engine.sync_weights()
            else:
              self.rl_engine.global_steps += (
                  1  # manually increment the global steps.
              )

          self.rl_engine.buffer_metrics(
              self.rl_engine.perf.export(),
              mode=rl_engine_lib.Mode.TRAIN,
          )
          self.rl_engine.buffer_metrics(
              self.rl_engine.perf_v2.export(),
              mode=rl_engine_lib.Mode.TRAIN,
          )

          if (
              self.rl_engine.actor_trainer.train_steps  # pyrefly: ignore[unsupported-operation]
              >= self.rl_engine.cluster_config.training_config.max_steps
          ):
            break
        except StopIteration:
          break
    finally:
      self.rl_engine.close()

  def _shard_batch_for_process(self, batch: TrainingInputT) -> TrainingInputT:
    """This process's prompt shard of a global batch (per-host learner).

    With TUNIX_PER_HOST_BALANCED_SHARD=1 the shard is length-balanced instead
    of the `[process_index::process_count]` stride: the per-host rollouts end
    together, so fewer hosts wait at the train step for the host with the
    longest prompts. Same rows per process; the union is unchanged.

    Args:
      batch: A global batch.

    Returns:
      This process's shard.
    """
    indices = None
    if _env_flag("TUNIX_PER_HOST_BALANCED_SHARD"):
      lengths = []
      for prompt in batch["prompts"]:  # pyrefly: ignore[not-iterable]
        try:
          lengths.append(len(prompt))
        except TypeError:
          lengths.append(0)
      indices = balanced_shard_indices(lengths, jax.process_count())[
          jax.process_index()
      ]
    return shard_batch_by_process(
        batch, jax.process_index(), jax.process_count(), indices
    )

  def _per_host_size(self, size: int | None, name: str) -> int | None:
    """Divides a configured global batch size by the process count."""
    if size is None:
      return None
    process_count = jax.process_count()
    if size % process_count:
      raise ValueError(
          f"TUNIX_PER_HOST_LEARNER=1: {name}={size} must be divisible by the"
          f" {process_count} processes (batch sizes are global)."
      )
    return size // process_count

  def _per_host_trainer_side_logps(self) -> list[str]:
    """Trainer-side logprob passes the algorithm config turns on.

    Under the per-host learner they need TUNIX_PER_HOST_SAFE_LOGPS=1. Mirrors
    GRPOLearner: reference logps when beta != 0; trainer (actor) logps when
    use_rollout_logps is off or sampler_is/sampler_rs/
    seq_logprob_error_threshold is set.

    Returns:
      A description of each pass that runs; empty if none does.
    """
    cfg = self.algo_config
    passes = []
    beta = getattr(cfg, "beta", 0.0)
    if beta is not None and beta != 0.0:
      passes.append(f"reference logprobs (beta={beta})")
    reasons = [] if getattr(cfg, "use_rollout_logps", True) else [
        "use_rollout_logps=False"
    ]
    reasons += [
        f"{name}={getattr(cfg, name)!r}"
        for name in ("sampler_is", "sampler_rs", "seq_logprob_error_threshold")
        if getattr(cfg, name, None) is not None
    ]
    if reasons:
      passes.append(f"trainer logprobs ({', '.join(reasons)})")
    return passes

  def _check_per_host_data_axis(
      self,
      train_micro_batch_size: int,
      logps_micro_batch_size: int | None,
  ) -> None:
    """Fails fast where `shard_input` would silently replicate local data.

    `shard_input` falls back to a replicated sharding when the leading
    dimension of the input is not divisible by the global data axis size
    (judged by the process-local rows, or by local rows x process count with
    TUNIX_PER_HOST_SAFE_LOGPS=1). Process-local rows that differ between
    processes must never be treated as replicated.

    Args:
      train_micro_batch_size: Per-host prompts per train micro-batch.
      logps_micro_batch_size: Per-host prompts per reference/trainer logps
        chunk, or None if no trainer-side logps pass runs.
    """
    mesh = self.rl_engine.cluster_config.role_to_mesh[rl_engine_lib.Role.ACTOR]
    axis_size = 1
    for axis in self._training_config.data_sharding_axis:
      for name in axis if isinstance(axis, tuple) else (axis,):
        if name is None:
          continue
        if name not in mesh.shape:
          raise ValueError(
              f"TUNIX_PER_HOST_LEARNER=1: data sharding axis {name!r} is not"
              f" in the actor mesh {dict(mesh.shape)}; shard_input would"
              " replicate process-local data."
          )
        axis_size *= mesh.shape[name]
    generations = self._num_generations() or 1
    global_rows = sharding_utils.per_host_safe_logps_enabled()
    factor = jax.process_count() if global_rows else 1
    checks = [(train_micro_batch_size, "train micro-batch")]
    if logps_micro_batch_size is not None:
      checks.append((logps_micro_batch_size, "reference/trainer logps chunk"))
    for prompts, what in checks:
      rows = prompts * generations
      if (rows * factor) % axis_size:
        counted = (
            "local rows x process count (TUNIX_PER_HOST_SAFE_LOGPS=1)"
            if global_rows
            else "process-local rows"
        )
        raise ValueError(
            f"TUNIX_PER_HOST_LEARNER=1: the per-host {what} has {rows} rows;"
            f" shard_input judges {rows * factor} ({counted}), which is not"
            f" divisible by the data axis size {axis_size}, so it would"
            " replicate process-local data."
        )

  def _run_global_step(
      self,
      full_batch_size: int,
      mini_batch_size: int,
      service_target_batch_size: int,
      iterator_steps_per_mini_batch: int,
      train_iterator: Iterator[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None,
      skip_jit: bool,
  ) -> None:
    """Run one global step."""
    for _ in range(full_batch_size // mini_batch_size):
      initial_steps = self._iter_steps

      with self.rl_engine.perf.span_group("mini_batch_step"):
        self._run_mini_batch_step(
            initial_steps,
            mini_batch_size,
            service_target_batch_size,
            iterator_steps_per_mini_batch,
            train_iterator,
            eval_ds,
            skip_jit,
        )

      # sync the iter steps with internel trainer, this is based on the
      # assumption that the trainer internally doesn't reset the iter steps.
      # there is current a unit test to ensure this assumption.
      self._iter_steps = self.rl_engine.actor_trainer.iter_steps

  def _run_mini_batch_step(
      self,
      initial_steps: int,
      mini_batch_size: int,
      service_target_batch_size: int,
      iterator_steps_per_mini_batch: int,
      train_iterator: Iterator[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None,
      skip_jit: bool,
  ) -> None:
    """Run one mini batch step."""
    with self.rl_engine.perf.span_group("micro_batch_steps"):
      self._run_all_micro_batch_steps(
          initial_steps,
          mini_batch_size,
          service_target_batch_size,
          iterator_steps_per_mini_batch,
          train_iterator,
          eval_ds,
          skip_jit,
      )

  def _run_all_micro_batch_steps(
      self,
      initial_steps: int,
      mini_batch_size: int,
      service_target_batch_size: int,
      iterator_steps_per_mini_batch: int,
      train_iterator: Iterator[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None,
      skip_jit: bool,
  ) -> None:
    """Run all micro batch steps."""

    # reserve 1 for None and the other for repeated interable
    # if batch_repeat > 1
    train_data_queue = queue_lib.SimpleDataQueue(
        maxsize=iterator_steps_per_mini_batch * self._num_iterations() + 1
    )
    # Use an unbounded queue for evaluation data.
    eval_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    future = self.executor.submit(
        self._prepare_data,
        iterator=train_iterator,
        proceed_num_steps=iterator_steps_per_mini_batch,
        sample_repeat=self._num_generations(),
        batch_repeat=self._num_iterations(),
        service_target_batch_size=service_target_batch_size,
        data_queue=train_data_queue,
        async_loading=self.can_enable_async_rollout,
        mode=rl_engine_lib.Mode.TRAIN,
    )

    def queue_iterator():
      while True:
        item = train_data_queue.get(block=True)
        if item is None:
          break
        yield item

    train_data_gen = queue_iterator()
    if self._packing_enabled:
      assert self._training_config.max_seq_token_per_tpu is not None
      mesh = self.rl_engine.cluster_config.role_to_mesh[
          rl_engine_lib.Role.ACTOR
      ]
      # The packed batch size must be a multiple of the FSDP and DP mesh axis
      # sizes.
      pack_size = rl_utils.compute_pack_size(mesh)

      logging.info(
          "Using sequence packing with max_seq_token_per_tpu: %d, "
          " pack_size: %d",
          self._training_config.max_seq_token_per_tpu,
          pack_size,
      )
      # Update boundary in sequences (mini-batch semantics): packing is
      # independent of any micro-batch/streaming granularity.
      train_data_gen = rl_utils.pack_sequences(
          train_data_gen,
          self._training_config.max_seq_token_per_tpu,
          sequences_per_update=mini_batch_size * self._num_generations(),
          pack_size=pack_size,
          max_segments_per_packed_row=getattr(
              self._training_config, "max_segments_per_packed_row", None
          ),
      )

    curr_eval_ds = None
    with jax.profiler.StepTraceAnnotation("trainer", step_num=initial_steps):
      while True:
        with sft_utils.time_measure(suppress_logging=True) as timer:
          try:
            curr_train_ds = next(train_data_gen)
          except StopIteration:
            break

        if self.can_enable_async_rollout:
          self.rl_engine.buffer_metrics(
              {
                  "actor_dequeue_time": (
                      timer(),
                      np.mean,
                  ),
              },
              mode=rl_engine_lib.Mode.TRAIN,
          )

        if (
            eval_ds
            and not curr_eval_ds
            and self.rl_engine.actor_trainer.train_steps
            % self.rl_engine.cluster_config.training_config.eval_every_n_steps
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
              mode=rl_engine_lib.Mode.EVAL,
          )
          curr_eval_ds = eval_data_queue.get(block=True)
        self.rl_engine.update_actor(
            curr_train_ds,
            curr_eval_ds,
            skip_jit,
        )  # loop over μ num_iterations
        if hasattr(self.rl_engine, "critic_trainer"):
          self.rl_engine.update_critic(
              curr_train_ds,
              curr_eval_ds,
              skip_jit,
          )  # loop over μ num_iterations

    # call to throw stop iteration as a signal to break the loop
    future.result()

  @property
  def rl_cluster(self) -> rl_engine_lib.RLEngine:
    """Alias for backward compatibility."""
    return self.rl_engine

  @rl_cluster.setter
  def rl_cluster(self, value: rl_engine_lib.RLEngine) -> None:
    self.rl_engine = value
