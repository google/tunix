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
import time
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
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import function_registry
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
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
  policy_version: np.ndarray | None = None


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
  reward_manager: str = "agentic-sequence-level"
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
      reward_fns: RewardFn | List[RewardFn] | None = None,
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
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
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
    self._validate_rollout_config()
    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_manager
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )
    self.metric_fns = metric_fns or []
    self.rl_cluster.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_cluster, "critic_trainer"):
      self.rl_cluster.critic_trainer.is_managed_externally = True

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

    # Colocate mode: detected when the actor and rollout meshes are the *same
    # Python object* (set via ``same_as: actor`` in the rollout mesh config).
    # Value-equal meshes that were constructed independently are NOT considered
    # colocate — this lets users share a subset of devices across roles without
    # accidentally triggering colocate mode.
    self.colocate_mode = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        is self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )
    self.can_enable_async_rollout = not self.colocate_mode

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size
    )
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = self.rl_cluster.global_steps
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
    self._global_step_start_time = time.time()

  def _validate_rollout_config(self):
    """Validates that the rollout config is properly aligned with the algo config."""
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if not isinstance(rollout_config, dict):
      configs_to_check = {"train": rollout_config}
    else:
      configs_to_check = rollout_config

    for mode, config in configs_to_check.items():
      if config.max_tokens_to_generate != self.algo_config.max_response_length:
        raise ValueError(
            f"RolloutConfig ({mode}) max_tokens_to_generate "
            f"({config.max_tokens_to_generate}) must match AgenticRLConfig "
            f"max_response_length ({self.algo_config.max_response_length}). "
            "Please align these configurations before initializing RLCluster."
        )
      if not config.return_logprobs:
        raise ValueError(
            f"RolloutConfig ({mode}) must have return_logprobs=True for "
            "AgenticRLLearner. Please set this before initializing RLCluster."
        )
      if (
          self.rl_cluster.cluster_config.rollout_engine == "vllm"
          and not config.rollout_vllm_server_mode
      ):
        raise ValueError(
            f"RolloutConfig ({mode}) must have rollout_vllm_server_mode set to "
            "True for AgenticRLLearner if using vLLM engine. Please set this "
            "before initializing RLCluster."
        )

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

    # Pass the expected_step explicitly because it is calculated based on
    # the batch index (predicted step) to align metrics with the correct
    # training step in the asynchronous execution.
    expected_step = 0 if expected_step is None else expected_step
    self.rl_cluster.buffer_metrics_async(
        rewards_info["log_metrics"], mode=mode, step=expected_step
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
  ) -> base_rollout.RolloutOutput:
    """Calls model generation."""
    if env:
      env.task["policy_version"] = self.policy_version

    if self.chat_parser:
      chat_lists = self.chat_parser.parse(
          messages=chat_lists,
          add_generation_prompt=True,
          is_first_msg=True,  # no op if system msg is populated in reset
      )
    tags = {}
    if env and hasattr(env, "extra_kwargs"):
      if "group_id" in env.extra_kwargs:
        tags[perf_constants.GROUP_ID] = env.extra_kwargs["group_id"]
        if self._full_batch_size > 0:
          tags[perf_constants.STEP] = (
              env.extra_kwargs["group_id"] // self._full_batch_size
          )
      if "pair_index" in env.extra_kwargs:
        tags[perf_constants.PAIR_INDEX] = env.extra_kwargs["pair_index"]

    result = self.rl_cluster.generate(
        prompts=chat_lists,
        apply_chat_template=False if self.chat_parser else True,
        mode=rl_cluster_lib.Mode.TRAIN,
        trace_tags=tags,
    )

    return result

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
        perf_v2=self.rl_cluster.perf_v2,
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
      group_id_offset: int = 0,
  ):
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").
      group_id_offset: Added to the base group ID so that multiple calls
        within the same global step produce unique group IDs (used in colocate
        mode where rollout is chunked into micro batches).

    Yields:
      A list of trajectories for a group.
    """
    is_async_iterator = hasattr(prompt_iterator, "__aiter__")

    async def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      # TODO (tsbao): fix the group id when we can resume from mid global step
      # with mini-batch.
      group_id = (
          self.rl_cluster.global_steps * self._full_batch_size + group_id_offset
      )
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
    if mode == rl_cluster_lib.Mode.TRAIN:
      expected_step = batch_results[0].group_id // self._full_batch_size
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
      raw_group_queue: queue.Queue,
      rollout_mbs: int = 0,
      training_done_event: threading.Event | None = None,
  ):
    """Produces raw trajectory groups from rollout.

    Ref-logprob computation is intentionally left to the consumer so that the
    reference model (which shares devices with the rollout engine in colocate
    mode) is never used concurrently with rollout.

    In colocate mode (``rollout_mbs > 0``) the producer pauses after every
    ``rollout_mbs`` groups and waits for ``training_done_event`` before
    resuming.  This enforces the strict rollout → compute → train → rollout
    sequence required when all roles share the same device mesh.

    Args:
      orchestrator: The RolloutOrchestrator instance.
      prompt_queue: Incoming prompt batches; ``None`` sentinel signals end.
      raw_group_queue: Output queue for raw trajectory groups.
      rollout_mbs: Groups to produce before pausing (0 = no pause, disagg).
      training_done_event: Set by the consumer after compute+train; cleared
        by the producer before each resumed rollout micro-batch.
    """
    loop = asyncio.get_running_loop()
    async_queue_iter = self._AsyncQueueIterator(prompt_queue, loop)

    async def _iterate_micro_batches():
      async for item in async_queue_iter:
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    groups_produced = 0
    try:
      async for group in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=_iterate_micro_batches(),
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        raw_group_queue.put(group)
        groups_produced += 1
        if rollout_mbs > 0 and groups_produced % rollout_mbs == 0:
          # Colocate: pause until consumer finishes ref-logprob compute and
          # training for this micro-batch before starting the next rollout.
          await loop.run_in_executor(None, training_done_event.wait)
          training_done_event.clear()
    finally:
      raw_group_queue.put(None)
      prompt_queue.put(None)

  def _run_eval(
      self,
      all_eval_prompts: list[TrainingInputT],
      training_config,
  ) -> list[Any] | None:
    """Runs eval rollouts and returns eval examples, or None if not due."""
    if (
        not all_eval_prompts
        or self.rl_cluster.actor_trainer.train_steps
        % training_config.eval_every_n_steps
        != 0
    ):
      return None

    self._eval_iter_steps = 0
    eval_orchestrator = self._build_orchestrator()

    async def _eval_runner_async(orch):
      eval_examples = []
      async for batch in self._orchestrator_producer(
          orch,
          all_eval_prompts,
          num_generations=self._num_generations(),
      ):
        eval_example = self._batch_to_train_example(
            batch, rl_cluster_lib.Mode.EVAL
        )
        eval_examples.extend(eval_example)
      return eval_examples

    eval_future = asyncio.run_coroutine_threadsafe(
        _eval_runner_async(eval_orchestrator), self.loop
    )
    eval_examples = eval_future.result()
    self._eval_iter_steps += 1
    return eval_examples

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner.

    A single producer coroutine runs rollouts and deposits raw trajectory
    groups into ``raw_group_queue``.  The consumer (this thread) converts
    those groups into training examples — including ref-logprob computation
    — and applies gradient updates.  Keeping ref-logprob compute in the
    consumer means the reference model is never used concurrently with the
    rollout engine, which is critical in colocate mode where all roles share
    the same device mesh.

    Colocate mode (``self.colocate_mode = True``):
      The producer pauses after every ``rollout_micro_batch_size`` groups and
      waits for ``training_done_event``.  The consumer accumulates those
      groups, computes ref-logprobs + advantages, runs gradient updates
      (``num_iterations`` times in ``train_micro_batch_size`` chunks), then
      sets the event so the producer may start the next rollout micro-batch.

    Disagg mode (``self.colocate_mode = False``):
      The producer runs freely on a separate device submesh.  The consumer
      processes each raw group as soon as it arrives, buffering
      ``train_micro_batch_size`` training examples before each gradient step.
      Rollout and compute+train overlap naturally across device partitions.
    """
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

    full_batch_size = len(next(iter(first_item.values())))
    self._full_batch_size = full_batch_size
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    training_config = self.rl_cluster.cluster_config.training_config

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    # ── Mode-specific setup ────────────────────────────────────────────────
    if self.colocate_mode:
      rollout_mbs = (
          self._training_config.rollout_micro_batch_size or full_batch_size
      )
      self._rollout_micro_batch_size = rollout_mbs
      self._compute_logps_micro_batch_size = (
          self._training_config.compute_logps_micro_batch_size
      )
      rl_utils.check_divisibility(
          rollout_mbs,
          full_batch_size,
          f"{rollout_mbs=}",
          f"{full_batch_size=}",
      )
      training_done_event = threading.Event()
      logging.info(
          "Colocate mode: full_batch_size=%d, rollout_micro_batch_size=%d, "
          "compute_logps_micro_batch_size=%s, train_micro_batch_size=%d, "
          "num_iterations=%d",
          full_batch_size,
          rollout_mbs,
          self._compute_logps_micro_batch_size,
          train_micro_batch_size,
          self.algo_config.num_iterations,
      )
    else:
      rollout_mbs = 0
      training_done_event = None
      grad_acc_steps = self._training_config.get_with_default(
          "gradient_accumulation_steps", 1
      )
      logging.info(
          "Disagg mode: full_batch_size=%d, mini_batch_size=%d, "
          "train_micro_batch_size=%d, grad_acc_steps=%d",
          full_batch_size,
          mini_batch_size,
          train_micro_batch_size,
          grad_acc_steps,
      )

    # ── Producer ────────────────────────────────────────────────────────────
    raw_group_queue = queue_lib.SimpleDataQueue(maxsize=0)
    orchestrator = self._build_orchestrator()
    prompt_queue = queue.Queue()

    initial_buffer_size = self.algo_config.off_policy_steps + 1
    logging.info("Prefilling prompt queue with %d batches.", initial_buffer_size)
    for _ in range(initial_buffer_size):
      try:
        self._put_prompts_to_queue(prompt_queue, next(full_dataset_iterator))
      except StopIteration:
        prompt_queue.put(None)
        break

    producer_future = asyncio.run_coroutine_threadsafe(
        self._producer(
            orchestrator,
            prompt_queue,
            raw_group_queue,
            rollout_mbs,
            training_done_event,
        ),
        self.loop,
    )

    # ── Consumer loop (unified for colocate and disagg) ────────────────────
    # Colocate: accumulate rollout_mbs raw groups before compute+train.
    # Disagg:   process each group immediately; buffer at train_micro_batch_size.
    groups_since_last_step = 0
    raw_group_buffer: list[Any] = []
    train_example_buffer: list[TrainExample] = []

    for raw_group in iter(raw_group_queue.get, None):
      if (
          training_config.max_steps
          and self.rl_cluster.global_steps >= training_config.max_steps
      ):
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_cluster.global_steps,
            training_config.max_steps,
        )
        prompt_queue.put(None)
        break

      if self.colocate_mode:
        # ── Colocate consumer ──────────────────────────────────────────────
        # The producer is paused after depositing rollout_mbs groups, so no
        # rollout is active while we run compute and training below.
        raw_group_buffer.append(raw_group)
        if len(raw_group_buffer) < rollout_mbs:
          continue

        # Phase: ref-logprob compute + advantage estimation.
        micro_train_examples: list[TrainExample] = []
        for group in raw_group_buffer:
          micro_train_examples.extend(
              self._batch_to_train_example(group, rl_cluster_lib.Mode.TRAIN)
          )
        raw_group_buffer = []

        # Phase: gradient updates (num_iterations passes over micro-batch).
        current_eval = self._run_eval(all_eval_prompts, training_config)
        for _ in range(self.algo_config.num_iterations):
          self._iter_steps += 1
          for i in range(0, len(micro_train_examples), train_micro_batch_size):
            micro = micro_train_examples[i : i + train_micro_batch_size]
            merged = jax.tree.map(
                lambda *xs: jnp.concatenate(xs, axis=0), *micro
            )
            self.rl_cluster.update_actor([merged], current_eval, skip_jit)
            if hasattr(self.rl_cluster, "critic_trainer"):
              self.rl_cluster.update_critic([merged], current_eval, skip_jit)
            current_eval = None

        groups_since_last_step += rollout_mbs

        if groups_since_last_step < full_batch_size:
          # Not yet a full step: signal producer to start next rollout
          # micro-batch and continue consuming.
          training_done_event.set()
          continue

      else:
        # ── Disagg consumer ───────────────────────────────────────────────
        # Compute ref-logprobs immediately; the rollout engine is running
        # concurrently on a separate submesh so there is no contention.
        train_examples = self._batch_to_train_example(
            raw_group, rl_cluster_lib.Mode.TRAIN
        )
        train_example_buffer.extend(train_examples)
        groups_since_last_step += 1

        while len(train_example_buffer) >= train_micro_batch_size:
          micro = train_example_buffer[:train_micro_batch_size]
          train_example_buffer = train_example_buffer[train_micro_batch_size:]
          merged = jax.tree.map(
              lambda *xs: jnp.concatenate(xs, axis=0), *micro
          )
          current_eval = self._run_eval(all_eval_prompts, training_config)
          self._iter_steps += 1
          self.rl_cluster.update_actor([merged], current_eval, skip_jit)
          if hasattr(self.rl_cluster, "critic_trainer"):
            self.rl_cluster.update_critic([merged], current_eval, skip_jit)

        if groups_since_last_step < full_batch_size:
          continue

      # ── End-of-step bookkeeping (shared) ────────────────────────────────
      global_step_time = time.time() - self._global_step_start_time
      logging.info(
          "Global step %d completed in %.2f s.",
          self.rl_cluster.global_steps,
          global_step_time,
      )
      self.rl_cluster.buffer_metrics_async(
          {"perf/global_step_time": (global_step_time, np.mean)},
          mode=rl_cluster_lib.Mode.TRAIN,
          step=self.rl_cluster.global_steps,
      )

      if self.should_sync_weights:
        if not self.colocate_mode:
          # Disagg: hold the rollout sync lock so the concurrent producer
          # does not dispatch model calls during the weight copy.
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
        try:
          with self.rl_cluster.perf_v2.span(
              perf_constants.WEIGHT_SYNC,
              self.rl_cluster.perf_v2.all_devices,
              tags={perf_constants.STEP: self.rl_cluster.global_steps},
          ):
            self.rl_cluster.sync_weights()
          self.policy_version += 1
          logging.info(
              "Weights synced. Policy version: %d.", self.policy_version
          )
        finally:
          if not self.colocate_mode:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
      else:
        self.rl_cluster.global_steps += 1

      self.rl_cluster.buffer_metrics(
          self.rl_cluster.perf_v2.export(),
          mode=rl_cluster_lib.Mode.TRAIN,
      )
      groups_since_last_step = 0
      self._global_step_start_time = time.time()

      # Load next batch into prompt_queue, then unblock the producer.
      try:
        with self.rl_cluster.perf_v2.span(
            perf_constants.DATA_LOADING,
            tags={perf_constants.STEP: self.rl_cluster.global_steps},
        ):
          batch = next(full_dataset_iterator)
        self._put_prompts_to_queue(prompt_queue, batch)
      except StopIteration:
        prompt_queue.put(None)

      if self.colocate_mode:
        # Weights are synced and next batch is queued: producer may now
        # start rolling out the first micro-batch of the next step.
        training_done_event.set()

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
    current_batch_size = len(next(iter(batch.values())))
    if (
        self._training_config.max_steps
        and self.rl_cluster.global_steps >= self._training_config.max_steps
    ):
      logging.info(
          "Reached max_steps: %d >= %d",
          self.rl_cluster.global_steps,
          self._training_config.max_steps,
      )
      prompt_queue.put(None)
    elif current_batch_size != self._full_batch_size:
      logging.warning(
          "partial batch %d vs %d detected. The rest of the batch will be"
          " skipped.",
          current_batch_size,
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
