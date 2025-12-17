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

"""Base class for Agentic RL Learners."""

from __future__ import annotations

import abc
import asyncio
from concurrent import futures
import contextlib
import dataclasses
import itertools
from typing import Any, Callable, Coroutine, Dict, Generic, Iterable, Iterator, List, Sequence, TypeVar

from absl import logging
import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_cluster_lib.MetricsT]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class AgenticRLConfig(algo_config_lib.AlgorithmConfig):
  """Base configuration for Agentic RL algorithms.

  Parameters:
    system_prompt: System prompt for the agent.
    max_concurrency: Maximum number of concurrent rollout engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
  """

  system_prompt: str = ""
  max_concurrency: int = 16
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn],
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `AgenticRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: Metric functions.
      data_shuffle_seed: Seed for data shuffling.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
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
    # Current iter steps for micro-batch based training.
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
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = 0
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    self._full_batch_size = 0

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      step: int | None = None,
      **kwargs,
  ) -> jax.Array:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      step: The current training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts]`) of scalar rewards for each
      prompt-completion pair. The rewards are the sum across all the provided
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
      if step is not None:
        self.rl_cluster.buffer_metrics_async(
            metrics_to_log, mode=mode, step=step
        )
      else:
        self.rl_cluster.buffer_metrics(metrics_to_log, mode=mode)

    return jnp.array(sum_rewards)

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

  def _make_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int | None = None
  ) -> tuple[model_agent.ModelAgent, task_environment.TaskEnvironment]:
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a group.

    Args:
      single_example: A training input containing a single prompt.
      group_id: An identifier to group generations from the same original
        prompt.

    Returns:
      A tuple containing a configured `ModelAgent` and `TaskEnvironment`.
    """

    question_text = single_example["question"][0]
    # Embed original input to avoid materializing the dataset in producer.
    task = {"question": question_text, "original_input": single_example}
    if group_id is not None:
      task["group_id"] = group_id
    # Pass along other metadata from the original example.
    for key, value in single_example.items():
      if key not in ["prompts", "original_input"]:
        task[key] = value[0]
    agent = model_agent.ModelAgent(system_prompt=self.algo_config.system_prompt)
    # TODO: b/456528861 - Support both single-turn and multi-turn from config.
    env = task_environment.TaskEnvironment(
        task=task,
        reward_fn=reward.dummy_reward,
        max_steps=1,
    )
    return agent, env

  def _model_call(self, chat_lists, env: Any = None):
    """Calls model generation."""
    version = self.policy_version

    if env:
      env.task["policy_version"] = version
    result = self.rl_cluster.generate(
        prompts=chat_lists,
        apply_chat_template=True,
        mode=rl_cluster_lib.Mode.TRAIN,
    )
    return result.text[0]

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_defaults = dict(
        model_call=self._model_call,
        final_reward_fn=reward.dummy_reward,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_defaults=engine_defaults,
        max_concurrency=self.algo_config.max_concurrency,
        rollout_sync_lock=self._rollout_sync_lock,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT],
      num_generations: int = 1,
      collect_mode: str = "Token",
  ):
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").

    Yields:
      A tuple where the first element is a list of trajectory results for a
      group, and the second is a list containing the original `TrainingInputT`
      for that group.
    """

    def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      for i, single_example in enumerate(prompt_iterator):
        agent, env = self._make_agent_env_pair(single_example, group_id=i)
        yield agent, env

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.algo_config.num_generations,
            group_key=lambda i, env, traj: env.task["group_id"],
            num_episodes=num_generations,
            collect_mode=collect_mode,
        )
    )

    # Let the producer start and initialize its manager before consuming.
    await asyncio.sleep(0)

    # Consume full groups and yield them with their original input.
    async_generator = orchestrator.yield_batches(
        batch_size=self.algo_config.num_generations
    )
    try:
      async with contextlib.aclosing(async_generator) as stream:
        async for group in stream:
          if group:
            # Retrieve the original input embedded in the task.
            original_input = group[0].traj["original_input"]
            yield group, [original_input]
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path for a generator.
      return
    finally:
      # Ensure the background producer task is cancelled and cleaned up.
      if not producer_task.done():
        producer_task.cancel()

        async def await_cancellation():
          with contextlib.suppress(asyncio.CancelledError):
            await producer_task

        cancellation_task = asyncio.create_task(await_cancellation())
        del cancellation_task

  def _batch_to_train_example(
      self,
      batch_results: list[Any],
      cached_inputs_for_window: list[TrainingInputT],
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    """Converts a group of trajectories into a list of `TrainExample`s.

    Args:
      batch_results: A list of trajectory results from the orchestrator.
      cached_inputs_for_window: The original input data for this group.
      mode: The current mode (TRAIN or EVAL).

    Returns:
      A list of `TrainExample` instances, ready for training.
    """
    # Create a merged training_input where each field from the original input
    # is repeated G times to align with the G completions.
    num_generations = self.algo_config.num_generations
    micro_batches = [cached_inputs_for_window[0]] * num_generations
    training_input = rl_utils.merge_micro_batches(micro_batches)

    prompt_index = batch_results[0].pair_index
    if mode == rl_cluster_lib.Mode.TRAIN and self._full_batch_size:
      step = prompt_index // self._full_batch_size
    else:
      step = self.rl_cluster.global_steps
    trajectory_ids = self._compute_trajectory_ids(training_input, prompt_index)
    assert "trajectory_ids" not in training_input
    training_input["trajectory_ids"] = trajectory_ids
    for t_id in trajectory_ids:
      self.rl_cluster.buffer_metrics_async(
          {
              "trajectory_ids": (t_id, None),
          },
          mode=mode,
          step=step,
      )
    return self._process_results(
        results=batch_results,
        training_input=training_input,
        mode=mode,
        step=step,
    )

  @abc.abstractmethod
  def _process_results(
      self,
      results: List[Any],
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages."""
    pass

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Unused in AgenticRLLearner."""
    raise NotImplementedError(
        "_generate_and_compute_advantage is not used in AgenticRLLearner"
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, prompt_index: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch."""
    batch_size = len(example["prompts"]) // self.algo_config.num_generations
    if batch_size != 1:
      raise ValueError(
          "_compute_trajectory_ids expects inputs for a single prompt group,"
          f" but got batch_size={batch_size}"
      )
    row_offset = prompt_index
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    """Returns the number of iterations per batch."""
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    """Returns the number of generations per prompt."""
    return self.algo_config.num_generations

  @staticmethod
  def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Runs a coroutine, handling existing event loops correctly."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      # asyncio.get_running_loop() raises RuntimeError if no loop is running.
      # If no loop is running, start a new one using asyncio.run().
      return asyncio.run(coro)
    else:
      # If a loop is already running, use it to run the coroutine.
      return loop.run_until_complete(coro)

  async def _producer(self, orchestrator, dataset_iterator, train_data_queue):
    """Produces training examples from prompts in the dataset_iterator."""

    def _iterate_micro_batches():
      for item in dataset_iterator:
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    prompt_iterator = _iterate_micro_batches()
    try:
      async for batch, cached_inputs in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        try:
          train_examples = self._batch_to_train_example(
              batch_results=batch,
              cached_inputs_for_window=cached_inputs,
              mode=rl_cluster_lib.Mode.TRAIN,
          )
          iterations = self.algo_config.num_iterations
          for _ in range(iterations):
            for train_example in train_examples:
              train_data_queue.put(train_example)
        except Exception as e:
          if not isinstance(e, RuntimeError):
            logging.exception(
                "Exception in _producer while processing batch: %s", e
            )
          raise
    finally:
      # Signal production is complete for this batch, even if errors occurred.
      train_data_queue.put(None)

  def _data_consumer_batch_generator(
      self, queue: queue_lib.AbstractDataQueue, batch_size: int
  ):
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      yield batch

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_cluster.close()
      return

    full_batch_size = len(first_item["prompts"])
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    self._rollout_micro_batch_size = 1
    self._compute_logps_micro_batch_size = 1
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

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    training_config = self.rl_cluster.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    # 1. Start producer thread to generate rollouts and training examples.
    orchestrator = self._build_orchestrator()
    producer_future = self.executor.submit(
        self._run_async,
        self._producer(orchestrator, full_dataset_iterator, train_data_queue),
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, train_micro_batch_size * self._num_generations()
    )
    micro_batches_since_last_sync = 0
    micro_batches_per_full_batch = full_batch_size // train_micro_batch_size
    for train_micro_batch in train_data_gen:
      if self.rl_cluster.global_steps >= self._training_config.max_steps:
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_cluster.global_steps,
            self._training_config.max_steps,
        )
        break
      self._iter_steps += 1

      # Filter out examples that are too old (off-policy).
      filtered_train_micro_batch = []
      for train_example in train_micro_batch:
        if train_example.policy_version is not None and (
            train_example.policy_version[0] == -1
            or (
                self.policy_version - train_example.policy_version[0]
                <= self.algo_config.off_policy_steps
            )
        ):
          filtered_train_micro_batch.append(train_example)
      if not filtered_train_micro_batch:
        logging.warning(
            "Skipping microbatch: all %d examples are too old."
            " Current policy version: %d, data versions: %s,"
            " off_policy_steps: %d",
            len(train_micro_batch),
            self.policy_version,
            str([
                train_example.policy_version[0]
                for train_example in train_micro_batch
            ]),
            self.algo_config.off_policy_steps,
        )
        continue
      train_micro_batch = filtered_train_micro_batch

      merged_train_micro_batch = jax.tree.map(
          lambda *xs: np.concatenate(xs, axis=0), *train_micro_batch
      )

      # --- Evaluation Logic ---
      current_eval_dataset = None
      if (
          all_eval_prompts
          and self.rl_cluster.actor_trainer.train_steps
          % training_config.eval_every_n_steps
          == 0
      ):
        self._eval_iter_steps = 0
        eval_orchestrator = self._build_orchestrator()

        async def _eval_runner_async(current_eval_orchestrator):
          eval_examples = []
          async for batch, cached_inputs in self._orchestrator_producer(
              current_eval_orchestrator,
              all_eval_prompts,
              num_generations=self._num_generations(),
          ):
            train_examples = self._batch_to_train_example(
                batch,
                cached_inputs,
                rl_cluster_lib.Mode.EVAL,
            )
            eval_examples.extend(train_examples)
          return eval_examples

        eval_future = self.executor.submit(
            self._run_async, _eval_runner_async(eval_orchestrator)
        )
        eval_examples = eval_future.result()
        self._eval_iter_steps += 1
        current_eval_dataset = eval_examples

      # --- Training Step ---
      self.rl_cluster.update_actor(
          [merged_train_micro_batch], current_eval_dataset, skip_jit
      )
      if hasattr(self.rl_cluster, "critic_trainer"):
        self.rl_cluster.update_critic(
            train_micro_batch, current_eval_dataset, skip_jit
        )

      # --- Weight Sync Logic ---
      micro_batches_since_last_sync += 1
      if micro_batches_since_last_sync == micro_batches_per_full_batch:
        if self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            self.rl_cluster.sync_weights()
            self.policy_version += 1
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
          finally:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
        else:
          self.rl_cluster.global_steps += 1
        micro_batches_since_last_sync = 0

    _ = producer_future.result()
    self.rl_cluster.close()
