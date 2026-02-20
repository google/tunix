# Copyright 2026 Google LLC
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
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import dataclasses
import itertools
import queue
import threading
from typing import Any, AsyncIterator, Callable, Dict, Generic, Iterable, Iterator, List, Sequence, Type, TypeVar

from absl import logging
import flax
import jax
from jax import typing
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

ArrayLike = typing.ArrayLike
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
    max_response_length: Maximum number of tokens for each episode.
    max_concurrency: Maximum number of concurrent requests to the rollout
      engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
    episode_timeout: Timeout for each episode in seconds.
  """

  system_prompt: str = ""
  # TODO(tsbao): we need to update the scripts that uses max_tokens_to_generate
  # once this new agentic_rl_learner is used.
  max_response_length: int = 1024
  max_concurrency: int = 32
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1
  episode_timeout: float = 1800.0


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  class _AsyncQueueIterator:
    """Async iterator that yields items from a sync queue."""

    def __init__(
        self,
        q: queue.Queue[TrainingInputT | None],
        loop: asyncio.AbstractEventLoop,
    ):
      self.q = q
      self.loop = loop

    def __aiter__(self):
      return self

    async def __anext__(self):
      item = await self.loop.run_in_executor(None, self.q.get)
      if item is None:
        raise StopAsyncIteration
      return item

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn],
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
      agent_class: Type[
          base_agent.ConversationAgentBase
      ] = model_agent.ModelAgent,
      agent_kwargs: Dict[str, Any] | None = None,
      env_class: Type[
          base_environment.BaseTaskEnv
      ] = task_environment.TaskEnvironment,
      env_kwargs: Dict[str, Any] | None = None,
  ):
    """Initializes the `AgenticRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: Metric functions.
      data_shuffle_seed: Seed for data shuffling.
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
    self._patch_rollout_config()

    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_manager
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )
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

    self.agent_class = agent_class
    self.agent_kwargs = agent_kwargs or {}
    self.env_class = env_class
    self.env_kwargs = env_kwargs or {}

    self._training_config = self.rl_cluster.cluster_config.training_config

    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = self.rl_cluster.actor_trainer.iter_steps
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

    loop_queue = queue.Queue()

    def run_loop_forever():
      loop = agentic_utils.get_or_create_loop()
      loop.set_default_executor(
          ThreadPoolExecutor(max_workers=algo_config.max_concurrency + 1)
      )
      loop_queue.put(loop)
      loop.run_forever()

    loop_thread = threading.Thread(target=run_loop_forever, daemon=True)
    loop_thread.start()
    self.loop = loop_queue.get()

  def _patch_rollout_config(self):
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if not isinstance(rollout_config, dict):
      rollout_config = {"train": rollout_config}
    for config in rollout_config.values():
      config.max_tokens_to_generate = self.algo_config.max_response_length

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      expected_step: int | None = None,
      **kwargs,
  ) -> np.ndarray:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      expected_step: The expected training step.
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

    rewards_info = self.reward_manager(
        prompts=prompts,
        completions=completions,
        **kwargs,
    )

    # Log all metrics for this trajectory in one call
    if expected_step is not None:
      # Pass the expected_step explicitly because it is calculated based on
      # the batch index (predicted step) to align metrics with the correct
      # training step in the asynchronous execution.
      self.rl_cluster.buffer_metrics_async(
          rewards_info["log_metrics"], mode=mode, step=expected_step
      )
    else:
      self.rl_cluster.buffer_metrics_async(
          rewards_info["log_metrics"], mode=mode
      )

    return rewards_info["rewards"]

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

  def _create_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int, pair_index: int
  ) -> tuple[base_agent.ConversationAgentBase, base_environment.BaseTaskEnv]:
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a group.

    Args:
      single_example: A training input containing a single prompt.
      group_id: An identifier for group generations from the same original
        prompt.
      pair_index: The index of the pair within the group.

    Returns:
      A tuple of agent and environment.
    """
    agent = self.agent_class(
        **{"system_prompt": self.algo_config.system_prompt, **self.agent_kwargs}
    )  # if agent_kwargs contains "system_prompt", it will be honored.

    assert "group_id" not in self.env_kwargs
    assert "pair_index" not in self.env_kwargs
    env = self.env_class(
        single_example,
        **{"group_id": group_id, "pair_index": pair_index, **self.env_kwargs},
    )
    return agent, env

  def _model_call(
      self, chat_lists: List[Dict[str, str]], env: Any = None
  ) -> str:
    """Calls model generation."""
    version = self.policy_version

    if env:
      env.task["policy_version"] = version

    if self.chat_parser:
      chat_lists = self.chat_parser.parse(
          messages=chat_lists,
          add_generation_prompt=True,
          is_first_msg=True,  # no op if system msg is populated in reset
      )

    with self.rl_cluster.perf.span("rollout"):
      result = self.rl_cluster.generate(
          prompts=chat_lists,
          apply_chat_template=False if self.chat_parser else True,
          mode=rl_cluster_lib.Mode.TRAIN,
      )

    return result.text[0]

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        final_reward_fn=reward.dummy_reward,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
    )
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_kwargs=engine_kwargs,
        max_concurrency=self.algo_config.max_concurrency,
        rollout_sync_lock=self._rollout_sync_lock,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT] | AsyncIterator[TrainingInputT],
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
      A list of trajectories for a group.
    """
    is_async_iterator = hasattr(prompt_iterator, "__aiter__")

    async def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      # TODO (tsbao): fix the group id when we can resume from mid global step
      # with mini-batch.
      group_id = self.rl_cluster.global_steps * self._full_batch_size
      if is_async_iterator:
        async for single_example in prompt_iterator:
          # Create agent-env pairs in parallel for a group to handle potential
          # cold start latency on env creation.
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1
      else:
        for single_example in prompt_iterator:
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.algo_config.num_generations,
            group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
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
            yield group
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
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    """Converts a group of trajectories into a list of `TrainExample`s.

    Args:
      batch_results: A list of trajectories from the same generation group.
      mode: The current mode (TRAIN or EVAL).

    Returns:
      A list of `TrainExample` instances, ready for training.
    """
    # Create a merged training_input where each field from the original input
    # is repeated G times to align with the G completions.
    num_generations = self.algo_config.num_generations
    prompt_index = batch_results[0].pair_index // num_generations
    if mode == rl_cluster_lib.Mode.TRAIN and self._full_batch_size:
      expected_step = prompt_index // self._full_batch_size
    else:
      expected_step = self.rl_cluster.global_steps

    return self._process_results(
        trajectories=batch_results,
        mode=mode,
        expected_step=expected_step,
    )

  @abc.abstractmethod
  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
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

  def _num_iterations(self) -> int:
    """Returns the number of iterations per batch."""
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    """Returns the number of generations per prompt."""
    return self.algo_config.num_generations

  async def _producer(
      self,
      orchestrator,
      prompt_queue: queue.Queue[TrainingInputT | None],
      train_data_queue,
  ):
    """Produces training examples from prompts in the dataset_iterator."""
    loop = asyncio.get_running_loop()
    async_queue_iter = self._AsyncQueueIterator(prompt_queue, loop)

    async def _iterate_micro_batches():
      async for item in async_queue_iter:
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    prompt_iterator = _iterate_micro_batches()
    try:
      async for batch in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        try:
          train_examples = self._batch_to_train_example(
              batch_results=batch,
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
      # Ensure that any background threads waiting on the prompt queue are
      # unblocked.
      prompt_queue.put(None)

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

    if self.rl_cluster.global_steps > 0:
      logging.info(
          "Skipping %d batches from train_dataset to fast-forward to step %d",
          self.rl_cluster.global_steps,
          self.rl_cluster.global_steps,
      )
      # TODO(b/483779605): Current implementation of fast-forwarding does not
      # take into account the mini-batch size. Follow-up CL will address this.
      for _ in range(self.rl_cluster.global_steps):
        try:
          next(full_batch_iterator)
        except StopIteration:
          logging.warning("Train dataset exhausted while skipping batches.")
          self.rl_cluster.close()
          return

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
    # Rollout and compute_logps micro batch sizes have to be 1 since we only
    # process inidividual prompts.
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

    prompt_queue = queue.Queue()
    initial_buffer_size = self.algo_config.off_policy_steps + 1
    logging.info(
        "Prefilling prompt queue with %d batches.", initial_buffer_size
    )
    for _ in range(initial_buffer_size):
      try:
        self._put_prompts_to_queue(prompt_queue, next(full_dataset_iterator))
      except StopIteration:
        prompt_queue.put(None)
        break

    producer_future = asyncio.run_coroutine_threadsafe(
        self._producer(orchestrator, prompt_queue, train_data_queue),
        self.loop,
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, train_micro_batch_size * self._num_generations()
    )
    micro_batches_per_full_batch = full_batch_size // train_micro_batch_size
    data_insufficient = False

    while self.rl_cluster.global_steps < self._training_config.max_steps:
      # Start of Global Step
      with self.rl_cluster.perf.span_group("global_step"):
        # TODO(noghabi): This mini batch is not correct. just a hack to make export work.
        with self.rl_cluster.perf.span_group("mini_batch_step"):
          # TODO(noghabi): Change metrics to single micro-batch not all steps.
          with self.rl_cluster.perf.span("micro_batch_steps"):
            for _ in range(micro_batches_per_full_batch):
              # Start of Micro Batch
              try:
                with self.rl_cluster.perf.span("data_loading"):
                  # with sft_utils.time_measure(suppress_logging=True) as timer:
                  train_micro_batch = next(train_data_gen)

                # self.rl_cluster.buffer_metrics(
                #     {
                #         "actor_dequeue_time": (
                #             timer(),
                #             np.mean,
                #         ),
                #     },
                #     mode=rl_cluster_lib.Mode.TRAIN,
                # )
              except StopIteration:
                data_insufficient = True
                break

              self._iter_steps += 1

              # Filter out examples that are too old (off-policy).
              filtered_train_micro_batch = (
                  self._filter_outdated_offpolicy_examples(train_micro_batch)
              )
              if not filtered_train_micro_batch:
                continue
              train_micro_batch = filtered_train_micro_batch

              merged_train_micro_batch = jax.tree.map(
                  lambda *xs: jnp.concatenate(xs, axis=0), *train_micro_batch
              )

              # --- Evaluation Logic for one micro-batch ---
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
                  async for batch in self._orchestrator_producer(
                      current_eval_orchestrator,
                      all_eval_prompts,
                      num_generations=self._num_generations(),
                  ):
                    eval_example = self._batch_to_train_example(
                        batch,
                        rl_cluster_lib.Mode.EVAL,
                    )
                    eval_examples.extend(eval_example)
                  return eval_examples

                eval_future = asyncio.run_coroutine_threadsafe(
                    _eval_runner_async(eval_orchestrator), self.loop
                )
                eval_examples = eval_future.result()
                self._eval_iter_steps += 1
                current_eval_dataset = eval_examples

              # --- Training Step for one micro-batch ---
              self.rl_cluster.update_actor(
                  [merged_train_micro_batch], current_eval_dataset, skip_jit
              )
              if hasattr(self.rl_cluster, "critic_trainer"):
                self.rl_cluster.update_critic(
                    train_micro_batch, current_eval_dataset, skip_jit
                )

        # End the global step due to data insufficiency.
        if data_insufficient:
          break

          # --- Weight Sync Logic ---
        if self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            with self.rl_cluster.perf.span(
                "weight_sync", self.rl_cluster.perf.all_devices
            ):
              self.rl_cluster.sync_weights()
            self.policy_version += 1
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
            try:
              self._put_prompts_to_queue(
                  prompt_queue, next(full_dataset_iterator)
              )
            except StopIteration:
              prompt_queue.put(None)
          finally:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
        else:
          self.rl_cluster.global_steps += 1
          try:
            self._put_prompts_to_queue(
                prompt_queue, next(full_dataset_iterator)
            )
          except StopIteration:
            prompt_queue.put(None)

        self.rl_cluster.buffer_metrics(
            self.rl_cluster.perf.export(),
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        # End of Global Step

    logging.info(
        "Training loop finished. global_steps: %d, max_steps: %d",
        self.rl_cluster.global_steps,
        self._training_config.max_steps,
    )
    prompt_queue.put(None)

    _ = producer_future.result()
    self.rl_cluster.close()

  def _put_prompts_to_queue(
      self,
      prompt_queue: queue.Queue[TrainingInputT | None],
      batch,
  ):
    """Puts a batch of prompts into the queue.

    If the batch size does not match the expected full batch size, a warning is
    logged, and a StopIteration is raised to signal the end of the dataset.
    A None is put into the queue upon StopIteration to signal completion.

    Args:
      prompt_queue: The queue to put the batch into.
      batch: The batch of prompts (TrainingInputT).
    """
    if len(batch["prompts"]) != self._full_batch_size:
      logging.warning(
          "partial batch %d vs %d detected. The rest of the batch will be"
          " skipped.",
          len(batch["prompts"]),
          self._full_batch_size,
      )
      prompt_queue.put(None)
    else:
      prompt_queue.put(batch)

  def _filter_outdated_offpolicy_examples(
      self,
      train_micro_batch: List[TrainExample],
  ) -> List[TrainExample]:
    """Filters out outdated off-policy examples."""
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
    return filtered_train_micro_batch
