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
import asyncio
import abc
from concurrent import futures as concurrent_futures
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import dataclasses
import itertools
import queue
import threading
import time
from typing import Any, AsyncIterator, Callable, Dict, Generic, Iterable, Iterator, List, Sequence, Type, TypeVar, Optional, Set

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


# Marks the end of one logical global batch while keeping the stage pipeline
# alive. Downstream stages must flush any buffered work for the current batch
# when they see this sentinel, but they should continue waiting for more items
# from the same queue afterward. This is what prevents sequence packing and
# train micro-batching from crossing a global-step boundary.
_GLOBAL_BATCH_BARRIER = object()

# Marks the terminal end of a stage queue. Unlike _GLOBAL_BATCH_BARRIER, this
# means no more items will ever arrive on the queue, so downstream stages
# should flush any remaining buffered work and then exit instead of waiting for
# another batch.
_QUEUE_CLOSED = object()


def _mesh_device_keys(mesh) -> frozenset[Any]:
  return frozenset(
      getattr(device, "id", device)
      for device in mesh.devices.flatten().tolist()
  )


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class TrainStageItem:
  train_examples: List[TrainExample] | None
  end_of_global_batch: bool = False


@dataclasses.dataclass(frozen=True)
class StageBoundaryConfig:
  """Describes how a stage should buffer at its input and output edges."""

  buffer_inputs_until_barrier: bool = False
  buffer_outputs_until_barrier: bool = False


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
  filter_statuses: Optional[Set] = None
  overlong_filter: bool = False


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
    self._train_micro_batch_size = 1

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
      self, chat_lists: List[Dict[str, str]], env: Any = None,
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
        overlong_filter=self.algo_config.overlong_filter,
        filter_statuses=self.algo_config.filter_statuses,
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

  def _get_role_mesh(
      self,
      role: rl_cluster_lib.Role,
      fallback_role: rl_cluster_lib.Role | None = None,
  ):
    """Returns the mesh for a role, optionally falling back to another role."""
    mesh = self.rl_cluster.cluster_config.role_to_mesh.get(role)
    if mesh is not None:
      return mesh
    if fallback_role is None:
      raise KeyError(f"No mesh configured for role: {role}")
    return self.rl_cluster.cluster_config.role_to_mesh[fallback_role]

  def _roles_share_devices(
      self,
      upstream_role: rl_cluster_lib.Role,
      downstream_role: rl_cluster_lib.Role,
      *,
      upstream_fallback_role: rl_cluster_lib.Role | None = None,
      downstream_fallback_role: rl_cluster_lib.Role | None = None,
  ) -> bool:
    """Returns whether two stage roles resolve to the same device set."""
    upstream_mesh = self._get_role_mesh(upstream_role, upstream_fallback_role)
    downstream_mesh = self._get_role_mesh(
        downstream_role,
        downstream_fallback_role,
    )
    return _mesh_device_keys(upstream_mesh) == _mesh_device_keys(downstream_mesh)

  def _should_buffer_stage_boundary(
      self,
      upstream_role: rl_cluster_lib.Role,
      downstream_role: rl_cluster_lib.Role,
      *,
      upstream_fallback_role: rl_cluster_lib.Role | None = None,
      downstream_fallback_role: rl_cluster_lib.Role | None = None,
  ) -> bool:
    """Returns whether two adjacent stages must serialize on one boundary."""
    if not self.rl_cluster.cluster_config.offload_to_cpu:
      return False
    return self._roles_share_devices(
        upstream_role,
        downstream_role,
        upstream_fallback_role=upstream_fallback_role,
        downstream_fallback_role=downstream_fallback_role,
    )

  def _create_stage_boundary_config(
      self,
      *,
      upstream_role: rl_cluster_lib.Role,
      stage_role: rl_cluster_lib.Role,
      downstream_role: rl_cluster_lib.Role,
      upstream_fallback_role: rl_cluster_lib.Role | None = None,
      stage_fallback_role: rl_cluster_lib.Role | None = None,
      downstream_fallback_role: rl_cluster_lib.Role | None = None,
  ) -> StageBoundaryConfig:
    """Builds a reusable boundary config for a pipeline stage."""
    return StageBoundaryConfig(
        buffer_inputs_until_barrier=self._should_buffer_stage_boundary(
            upstream_role,
            stage_role,
            upstream_fallback_role=upstream_fallback_role,
            downstream_fallback_role=stage_fallback_role,
        ),
        buffer_outputs_until_barrier=self._should_buffer_stage_boundary(
            stage_role,
            downstream_role,
            upstream_fallback_role=stage_fallback_role,
            downstream_fallback_role=downstream_fallback_role,
        ),
    )

  async def _rollout_stage_producer(
      self,
      orchestrator,
      prompt_queue: queue.Queue[TrainingInputT | None],
      rollout_queue: queue_lib.AbstractDataQueue,
  ):
    """Produces rollout groups and emits a barrier per global batch."""
    loop = asyncio.get_running_loop()
    async_queue_iter = self._AsyncQueueIterator(prompt_queue, loop)
    full_batch_size = max(getattr(self, "_full_batch_size", 0), 1)
    prompts_in_current_batch = 0

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
        rollout_queue.put(batch)
        prompts_in_current_batch += 1
        if prompts_in_current_batch == full_batch_size:
          rollout_queue.put(_GLOBAL_BATCH_BARRIER)
          prompts_in_current_batch = 0
    finally:
      if prompts_in_current_batch:
        rollout_queue.put(_GLOBAL_BATCH_BARRIER)
      rollout_queue.put(_QUEUE_CLOSED)
      prompt_queue.put(None)

  def _global_rollout_batch_to_train_examples(
      self,
      rollout_batches: List[list[Any]],
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    """Converts one rollout global batch into raw train examples."""
    train_examples = []
    for batch in rollout_batches:
      train_examples.extend(
          self._batch_to_train_example(
              batch_results=batch,
              mode=mode,
          )
      )
    return train_examples

  def _emit_train_examples(
      self,
      train_data_queue: queue_lib.AbstractDataQueue,
      train_examples: List[TrainExample],
  ) -> None:
    """Emits train examples for all configured training iterations."""
    for _ in range(self.algo_config.num_iterations):
      for train_example in train_examples:
        train_data_queue.put(train_example)

  def _reference_stage_producer(
      self,
      rollout_queue: queue_lib.AbstractDataQueue,
      train_data_queue: queue_lib.AbstractDataQueue,
      boundary_config: StageBoundaryConfig,
  ) -> None:
    """Consumes rollout batches and emits raw train examples per global batch."""
    rollout_batches = []
    pending_train_examples = []
    try:
      while True:
        item = rollout_queue.get(block=True)
        if item is _QUEUE_CLOSED:
          break
        if item is _GLOBAL_BATCH_BARRIER:
          if rollout_batches:
            train_examples = self._global_rollout_batch_to_train_examples(
                rollout_batches,
                rl_cluster_lib.Mode.TRAIN,
            )
            if boundary_config.buffer_outputs_until_barrier:
              pending_train_examples.extend(train_examples)
            else:
              self._emit_train_examples(train_data_queue, train_examples)
            rollout_batches = []
          if pending_train_examples:
            self._emit_train_examples(
                train_data_queue,
                pending_train_examples,
            )
            pending_train_examples = []
          train_data_queue.put(_GLOBAL_BATCH_BARRIER)
          continue
        if boundary_config.buffer_inputs_until_barrier:
          rollout_batches.append(item)
          continue
        train_examples = self._global_rollout_batch_to_train_examples(
            [item],
            rl_cluster_lib.Mode.TRAIN,
        )
        if boundary_config.buffer_outputs_until_barrier:
          pending_train_examples.extend(train_examples)
        else:
          self._emit_train_examples(train_data_queue, train_examples)
    finally:
      if rollout_batches:
        train_examples = self._global_rollout_batch_to_train_examples(
            rollout_batches,
            rl_cluster_lib.Mode.TRAIN,
        )
        if boundary_config.buffer_outputs_until_barrier:
          pending_train_examples.extend(train_examples)
        else:
          self._emit_train_examples(train_data_queue, train_examples)
      if pending_train_examples:
        self._emit_train_examples(train_data_queue, pending_train_examples)
        train_data_queue.put(_GLOBAL_BATCH_BARRIER)
      train_data_queue.put(_QUEUE_CLOSED)

  def _request_stage_pipeline_stop(
      self,
      *,
      prompt_queue: queue.Queue[TrainingInputT | None] | None,
      rollout_queue: queue_lib.AbstractDataQueue | None,
      train_data_queue: queue_lib.AbstractDataQueue | None,
      rollout_future: concurrent_futures.Future | None = None,
  ) -> None:
    """Best-effort cooperative stop for the staged pipeline."""
    if prompt_queue is not None:
      prompt_queue.put(None)
    if rollout_queue is not None:
      rollout_queue.put(_QUEUE_CLOSED)
    if train_data_queue is not None:
      train_data_queue.put(_QUEUE_CLOSED)
    if rollout_future is not None and not rollout_future.done():
      rollout_future.cancel()

  def _attach_stage_failure_callbacks(
      self,
      *,
      rollout_future: concurrent_futures.Future,
      reference_future: concurrent_futures.Future,
      rollout_queue: queue_lib.AbstractDataQueue,
      train_data_queue: queue_lib.AbstractDataQueue,
  ) -> None:
    """Wakes blocked stages when an upstream worker fails or is cancelled."""

    def _wake_rollout_consumer_on_failure(
        future: concurrent_futures.Future,
    ) -> None:
      if not future.cancelled():
        try:
          future.result()
          return
        except BaseException:
          pass
      rollout_queue.put(_QUEUE_CLOSED)

    def _wake_train_consumer_on_failure(
        future: concurrent_futures.Future,
    ) -> None:
      if not future.cancelled():
        try:
          future.result()
          return
        except BaseException:
          pass
      train_data_queue.put(_QUEUE_CLOSED)

    rollout_future.add_done_callback(_wake_rollout_consumer_on_failure)
    reference_future.add_done_callback(_wake_train_consumer_on_failure)

  def _data_consumer_batch_generator(
      self,
      queue: queue_lib.AbstractDataQueue,
      batch_size: int,
  ) -> Iterator[TrainStageItem]:
    """Yields train batches and marks the global-batch boundary explicitly."""

    class _GlobalBatchQueueIterator:

      def __init__(
          self,
          data_queue: queue_lib.AbstractDataQueue,
      ):
        self._data_queue = data_queue
        self.hit_barrier = False
        self.hit_queue_closed = False

      def __iter__(self):
        return self

      def __next__(self):
        item = self._data_queue.get(block=True)
        if item is _GLOBAL_BATCH_BARRIER:
          self.hit_barrier = True
          raise StopIteration
        if item is _QUEUE_CLOSED:
          self.hit_queue_closed = True
          raise StopIteration
        return [item]

    while True:
      global_batch_iter = _GlobalBatchQueueIterator(
          queue,
      )
      if self._training_config.max_seq_token_per_tpu is not None:
        logging.info(
            "Using sequence packing with max_seq_token_per_tpu: %d",
            self._training_config.max_seq_token_per_tpu,
        )
        batch_iterator = rl_utils.pack_sequences(
            global_batch_iter,
            self._training_config.max_seq_token_per_tpu,
        )
      else:
        batch_iterator = self._iter_train_micro_batches_from_stream(
            global_batch_iter,
            batch_size,
        )

      last_batch = None
      for batch in batch_iterator:
        if last_batch is not None:
          yield TrainStageItem(last_batch, end_of_global_batch=False)
        last_batch = batch

      if last_batch is not None:
        yield TrainStageItem(last_batch, end_of_global_batch=True)
      elif global_batch_iter.hit_barrier:
        yield TrainStageItem(None, end_of_global_batch=True)

      if global_batch_iter.hit_queue_closed:
        return

  def _iter_train_micro_batches_from_stream(
      self,
      train_example_iterator: Iterable[List[TrainExample]],
      batch_size: int,
  ) -> Iterator[List[TrainExample]]:
    """Re-batches a stream of raw train examples into train micro-batches."""
    pending_train_examples = []
    for train_examples in train_example_iterator:
      pending_train_examples.extend(train_examples)
      while len(pending_train_examples) >= batch_size:
        yield pending_train_examples[:batch_size]
        pending_train_examples = pending_train_examples[batch_size:]
    if pending_train_examples:
      yield pending_train_examples

  def _iter_train_micro_batches(
      self,
      train_examples: List[TrainExample],
      batch_size: int,
  ) -> Iterator[List[TrainExample]]:
    """Splits a full training batch into trainer micro-batches."""
    for start in range(0, len(train_examples), batch_size):
      yield train_examples[start : start + batch_size]

  def _finalize_global_step(
      self,
      full_dataset_iterator: Iterator[TrainingInputT],
      prompt_queue: queue.Queue[TrainingInputT | None],
  ) -> None:
    """Completes sync and advances input state after one training batch."""
    global_step_time = time.time() - self._global_step_start_time
    logging.info(
        f"Global step {self.rl_cluster.global_steps} completed in"
        f" {global_step_time:.2f} seconds."
    )
    self.rl_cluster.buffer_metrics_async(
        {"perf/global_step_time": (global_step_time, np.mean)},
        mode=rl_cluster_lib.Mode.TRAIN,
        step=self.rl_cluster.global_steps,
    )
    if self.should_sync_weights:
      logging.info("Requesting sync lock to sync weights...")
      self._rollout_sync_lock.acquire_weight_sync()
      try:
        logging.info("Sync lock acquired. Syncing weights.")
        with self.rl_cluster.perf_v2.span(
            perf_constants.WEIGHT_SYNC,
            self.rl_cluster.perf_v2.all_devices,
            tags={
                perf_constants.STEP: self.rl_cluster.global_steps,
            },
        ):
          self.rl_cluster.sync_weights()
        self.policy_version += 1
        logging.info(
            "Weights synced. Policy version incremented to %d.",
            self.policy_version,
        )
        try:
          with self.rl_cluster.perf_v2.span(
              perf_constants.DATA_LOADING,
              tags={
                  perf_constants.STEP: self.rl_cluster.global_steps,
              },
          ):
            batch = next(full_dataset_iterator)
          self._put_prompts_to_queue(prompt_queue, batch)
        except StopIteration:
          prompt_queue.put(None)
      finally:
        self._rollout_sync_lock.release_weight_sync()
        logging.info("Sync lock released.")
    else:
      self.rl_cluster.global_steps += 1
      try:
        with self.rl_cluster.perf_v2.span(
            perf_constants.DATA_LOADING,
            tags={
                perf_constants.STEP: self.rl_cluster.global_steps,
            },
        ):
          batch = next(full_dataset_iterator)
        self._put_prompts_to_queue(prompt_queue, batch)
      except StopIteration:
        prompt_queue.put(None)

    self.rl_cluster.buffer_metrics(
        self.rl_cluster.perf_v2.export(),
        mode=rl_cluster_lib.Mode.TRAIN,
    )
    self._global_step_start_time = time.time()

  def _setup_train_stage_pipeline(
      self,
      *,
      orchestrator,
      prompt_queue: queue.Queue[TrainingInputT | None],
      rollout_queue: queue_lib.AbstractDataQueue,
      train_data_queue: queue_lib.AbstractDataQueue,
      reference_executor: ThreadPoolExecutor,
  ) -> tuple[concurrent_futures.Future, concurrent_futures.Future]:
    """Starts the rollout and reference stages for the train pipeline.

    This helper only performs stage startup:
    - submits the reference-stage worker to the single-thread executor
    - schedules the rollout-stage coroutine on the background event loop
    - installs failure callbacks so downstream blocked consumers wake up

    It deliberately does not own shutdown. `train()` is the lifecycle owner for
    the staged pipeline, so any startup failure or later runtime failure is
    unwound there via `_request_stage_pipeline_stop(...)` and executor cleanup.

    Returns:
      The `(rollout_future, reference_future)` pair for the started upstream
      stages. `train()` awaits these futures after the train-stage consumer
      drains the pipeline.
    """
    reference_stage_boundary_config = self._create_stage_boundary_config(
        upstream_role=rl_cluster_lib.Role.ROLLOUT,
        stage_role=rl_cluster_lib.Role.REFERENCE,
        downstream_role=rl_cluster_lib.Role.ACTOR,
        stage_fallback_role=rl_cluster_lib.Role.ACTOR,
    )
    reference_future = reference_executor.submit(
        self._reference_stage_producer,
        rollout_queue,
        train_data_queue,
        reference_stage_boundary_config,
    )
    rollout_future = asyncio.run_coroutine_threadsafe(
        self._rollout_stage_producer(
            orchestrator,
            prompt_queue,
            rollout_queue,
        ),
        self.loop,
    )
    self._attach_stage_failure_callbacks(
        rollout_future=rollout_future,
        reference_future=reference_future,
        rollout_queue=rollout_queue,
        train_data_queue=train_data_queue,
    )
    return rollout_future, reference_future

  def _run_train_stage_pipeline(
      self,
      *,
      prompt_queue: queue.Queue[TrainingInputT | None],
      train_data_queue: queue_lib.AbstractDataQueue,
      full_dataset_iterator: Iterator[TrainingInputT],
      all_eval_prompts: List[TrainingInputT],
      train_micro_batch_size: int,
      training_config,
      skip_jit: bool,
  ) -> None:
    """Consumes train-stage items until the staged pipeline reaches completion."""
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue,
        train_micro_batch_size,
    )
    for train_stage_item in train_data_gen:
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
        break

      train_micro_batch = train_stage_item.train_examples
      if train_micro_batch is not None:
        self._iter_steps += 1

        # TODO(tsbao): Re-enable this once off-policy filtering is needed.
        # Filter out examples that are too old (off-policy).
        # filtered_train_micro_batch = self._filter_outdated_offpolicy_examples(
        #     train_micro_batch
        # )
        # if not filtered_train_micro_batch:
        #   continue
        # train_micro_batch = filtered_train_micro_batch

        merged_train_micro_batch = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *train_micro_batch
        )

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

        self.rl_cluster.update_actor(
            [merged_train_micro_batch], current_eval_dataset, skip_jit
        )
        if hasattr(self.rl_cluster, "critic_trainer"):
          self.rl_cluster.update_critic(
              [merged_train_micro_batch], current_eval_dataset, skip_jit
          )

      if train_stage_item.end_of_global_batch:
        self._finalize_global_step(full_dataset_iterator, prompt_queue)

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

    full_batch_size = len(next(iter(first_item.values())))
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    self._train_micro_batch_size = train_micro_batch_size
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

    rollout_queue = queue_lib.SimpleDataQueue(maxsize=0)
    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
    reference_executor = None
    rollout_future = None
    reference_future = None

    # 1. Start the rollout and reference stages explicitly.
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

    try:
      reference_executor = ThreadPoolExecutor(max_workers=1)
      rollout_future, reference_future = self._setup_train_stage_pipeline(
          orchestrator=orchestrator,
          prompt_queue=prompt_queue,
          rollout_queue=rollout_queue,
          train_data_queue=train_data_queue,
          reference_executor=reference_executor,
      )
      self._run_train_stage_pipeline(
          prompt_queue=prompt_queue,
          train_data_queue=train_data_queue,
          full_dataset_iterator=full_dataset_iterator,
          all_eval_prompts=all_eval_prompts,
          train_micro_batch_size=train_micro_batch_size,
          training_config=training_config,
          skip_jit=skip_jit,
      )
      rollout_future.result()
      reference_future.result()
    except BaseException:
      self._request_stage_pipeline_stop(
          prompt_queue=prompt_queue,
          rollout_queue=rollout_queue,
          train_data_queue=train_data_queue,
          rollout_future=rollout_future,
      )
      raise
    finally:
      if reference_executor is not None:
        reference_executor.shutdown(wait=False, cancel_futures=True)
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
