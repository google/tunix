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
from typing import Any, AsyncIterator, Callable, Dict, Generic, get_args, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Type, TypeVar

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
from tunix.rl import packing
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import dynamic_sampling
from tunix.rl.agentic import group_filters as group_filters_lib
from tunix.rl.agentic import prompt_feed as prompt_feed_lib
from tunix.rl.agentic import submission_order
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.utils import compat
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward  # pylint: disable=unused-import
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

ArrayLike = typing.ArrayLike
TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_engine_lib.MetricsT]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: np.ndarray | None = None

  # Includes environment/template tokens; distinct from sampled-token loss mask.
  completion_attention_mask: jax.Array | None = None


# How kept groups fill the mini-batches of a global step; see
# `AgenticRLConfig.minibatch_order`.
MinibatchOrder = Literal["completion", "submission"]


@dataclasses.dataclass(slots=True, kw_only=True)
class AgenticRLConfig(algo_config_lib.AlgorithmConfig):
  """Base configuration for Agentic RL algorithms.

  Attributes:
    system_prompt: System prompt for the agent.
    reward_mode: The reward computation mode. Defaults to `"env"`, which uses
      environment `trajectory_rewards` (`AgenticSequenceRewardManager`) and
      sums `reward_fns` on top when `reward_fns` are provided.
    max_response_length: Maximum number of tokens for each episode.
    max_concurrency: Maximum number of concurrent requests to the rollout
      engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
    episode_timeout: Timeout for each episode in seconds.
    group_filters: Dynamic sampling (DAPO). Names from
      `group_filters.GROUP_FILTERS` applied to every scored training group; a
      dropped group is not trained on and one replacement prompt is admitted
      in its place, within the `off_policy_steps` staleness window. Empty
      (default) keeps every group. Reward metrics cover every scored group,
      dropped ones included.
    max_consecutive_filtered_groups: Raise `DynamicSamplingStalledError` after
      this many groups in a row are dropped by `group_filters` (the data no
      longer yields trainable groups). -1 (default) disables the guard.
    max_in_flight_groups: Optional cap on the prompt groups admitted ahead of
      the trainer, between `batch_size` and the default `(off_policy_steps +
      1) * batch_size` (e.g. 24 for `batch_size=16, off_policy_steps=1`).
      Setting it, or `group_filters`, switches the prompt feed from one batch
      per global step to per-prompt admission.
    minibatch_order: How kept groups fill the mini-batches of a global step
      when `mini_batch_size < batch_size` (several optimizer updates per
      step). `"completion"` (default): groups train in the order they finish,
      which puts the longest groups in the last (most off-policy, most
      clipped) mini-batch. `"submission"`: mini-batch `m` is the `m`-th block
      of `mini_batch_size` kept groups in prompt (`group_id`) order, so the
      mini-batch a group trains in depends on its dataset position, not on how
      long it took to generate; a dropped group gives its place to the next
      prompt, and members of the open mini-batch are released as soon as they
      finish. Ignored with one optimizer update per global step.
  """

  system_prompt: str = ""
  # TODO(tsbao): we need to update the scripts that uses max_tokens_to_generate
  # once this new agentic_rl_learner is used.
  reward_mode: str = "env"
  max_response_length: int = 1024
  max_concurrency: int = 32
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1
  episode_timeout: float = 1800.0
  filter_statuses: Optional[Set] = None
  overlong_filter: bool = False
  use_rollout_logps: bool = True
  group_filters: Sequence[str] = ()
  max_consecutive_filtered_groups: int = -1
  max_in_flight_groups: int | None = None
  minibatch_order: MinibatchOrder = "completion"

  def __post_init__(self):
    # `slots=True` recreates the class, which breaks zero-argument super().
    super(AgenticRLConfig, self).__post_init__()  # pylint: disable=super-with-arguments
    self._validate_dynamic_sampling_options()
    self._validate_minibatch_order()

  def _validate_minibatch_order(self) -> None:
    """Checks `minibatch_order`.

    Raises:
      ValueError: If `minibatch_order` is not a `MinibatchOrder`.
    """
    if self.minibatch_order not in get_args(MinibatchOrder):
      raise ValueError(
          f"minibatch_order must be one of {get_args(MinibatchOrder)}, got"
          f" {self.minibatch_order!r}."
      )

  def _validate_dynamic_sampling_options(self) -> None:
    """Checks `group_filters` and the dynamic-sampling limits.

    Raises:
      ValueError: If an option is invalid.
    """
    group_filters_lib.validate_group_filters(self.group_filters)
    if (
        self.max_consecutive_filtered_groups == 0
        or self.max_consecutive_filtered_groups < -1
    ):
      raise ValueError(
          "max_consecutive_filtered_groups must be positive or -1 to disable,"
          f" got {self.max_consecutive_filtered_groups}."
      )
    if self.max_in_flight_groups is not None and self.max_in_flight_groups <= 0:
      raise ValueError(
          "max_in_flight_groups must be positive when set, got"
          f" {self.max_in_flight_groups}."
      )


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  # Whether `_process_results` calls `_check_group_filters`, which is what
  # `AgenticRLConfig.group_filters` needs.
  _SUPPORTS_GROUP_FILTERS = False

  @compat.alias_init_param("rl_cluster", "rl_engine")
  def __init__(
      self,
      rl_engine: rl_engine_lib.RLEngine,
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
      rl_engine: RL engine containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return a
        dictionary of metric names to tuples of ``(metric_value,
        aggregation_fn)``:  >>> def metric_fn( ...     prompts, completions,
        rewards, advantages, **kargs ... ): ...     return { ...       # ... ...
        "prompt_min_len": (min(len(p) for p in prompts), np.min), ...       #
        ... }
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    if algo_config.group_filters and not self._SUPPORTS_GROUP_FILTERS:
      raise ValueError(
          f"{type(self).__name__} does not support group_filters: its"
          " _process_results does not call _check_group_filters."
      )
    self.rl_engine = rl_engine
    self.algo_config = algo_config
    self._training_config = self.rl_engine.cluster_config.training_config
    self._packing_enabled = rl_engine_lib.configs.is_sequence_packing_enabled(
        self._training_config
    )
    self._validate_rollout_config()
    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_mode
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )
    self.metric_fns = metric_fns or []
    self.rl_engine.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_engine, "critic_trainer"):
      self.rl_engine.critic_trainer.is_managed_externally = True

    self.agent_class = agent_class
    self.agent_kwargs = agent_kwargs or {}
    self.env_class = env_class
    self.env_kwargs = env_kwargs or {}

    self.rl_engine.global_steps = (
        self.rl_engine.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = self.rl_engine.actor_trainer.iter_steps
    self._eval_iter_steps = 0
    # Tracks the last train_step value at which evaluation was run. The
    # optimizer is wrapped in ``optax.MultiSteps(grad_accum_steps)``, which
    # keeps ``actor_trainer.train_steps`` constant for ``grad_accum_steps``
    # consecutive micro-iterations. Without this guard, the
    # ``train_steps % eval_every_n_steps == 0`` check would fire at every
    # micro-iteration during an eval boundary, causing the full evaluation
    # rollout to be replayed ``grad_accum_steps`` times for the same step.
    self._last_eval_train_step = -1

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

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size or 1
    )
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_engine.tokenizer
    self.policy_version = self.rl_engine.global_steps
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    # Set by `train` when `group_filters` or `max_in_flight_groups` is set.
    self._dynamic_sampler: dynamic_sampling.DynamicSampler | None = None
    # Set by `train` when mini-batches are filled in submission order.
    self._submission_emitter: (
        submission_order.SubmissionOrderEmitter[list[Any]] | None
    ) = None
    self._background_tasks: Set[asyncio.Task] = set()
    self._full_batch_size = 0
    self._process_in_consumer: bool = False

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

    # Per-step reward accumulators populated inside ``_compute_rewards``.
    # Drained at the global-step boundary to emit a one-line per-step
    # summary that mirrors what an external metric logger would show.
    # Each bin keeps at most ``full_batch_size``-worth of recent values
    # so a producer that races one batch ahead of the consumer does not
    # double-count.
    self._train_rewards_window: List[float] = []
    self._eval_rewards_window: List[float] = []
    self._rewards_window_lock = threading.Lock()

  def _validate_rollout_config(self):
    """Validates that the rollout config is properly aligned with the algo config."""
    rollout_config = self.rl_engine.cluster_config.rollout_config
    if not isinstance(rollout_config, dict):
      configs_to_check = {"train": rollout_config}
    else:
      configs_to_check = rollout_config

    for mode, config in configs_to_check.items():
      if self.algo_config.use_rollout_logps and not config.return_logprobs:
        raise ValueError(
            f"RolloutConfig ({mode}) must have return_logprobs=True for "
            "AgenticRLLearner when use_rollout_logps=True. Please set this "
            "before initializing RLEngine."
        )
      if self.algo_config.exact_token_continuity:
        if not config.return_logprobs:
          raise ValueError("exact_token_continuity requires sampled logprobs")
        if config.return_routed_experts and self._packing_enabled:
          raise ValueError(
              "exact_token_continuity does not replay expert routing when"
              " sequence packing (max_seq_token_per_tpu) is enabled"
          )
      if config.max_tokens_to_generate != self.algo_config.max_response_length:
        raise ValueError(
            f"RolloutConfig ({mode}) max_tokens_to_generate "
            f"({config.max_tokens_to_generate}) must match AgenticRLConfig "
            f"max_response_length ({self.algo_config.max_response_length}). "
            "Please align these configurations before initializing RLEngine."
        )
      if (
          self.rl_engine.cluster_config.rollout_engine == "vllm"
          and not config.rollout_vllm_server_mode
      ):
        raise ValueError(
            f"RolloutConfig ({mode}) must have rollout_vllm_server_mode set to "
            "True for AgenticRLLearner if using vLLM engine. Please set this "
            "before initializing RLEngine."
        )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_engine_lib.Mode,
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
    self.rl_engine.buffer_metrics_async(
        rewards_info["log_metrics"], mode=mode, step=expected_step
    )

    rewards_array = np.asarray(rewards_info["rewards"])
    with self._rewards_window_lock:
      target = (
          self._train_rewards_window
          if mode == rl_engine_lib.Mode.TRAIN
          else self._eval_rewards_window
      )
      target.extend(rewards_array.tolist())
      # Cap train window at full_batch_size * num_generations (one full step's
      # worth of per-sequence rewards) to bound the producer-vs-consumer
      # race: the producer can race up to ``off_policy_steps + 1`` batches
      # ahead, so without a cap the window would over-count next-step rewards
      # at the current step's boundary.
      if mode == rl_engine_lib.Mode.TRAIN and self._full_batch_size > 0:
        cap = self._full_batch_size * self.algo_config.num_generations
        excess = len(target) - cap
        if excess > 0:
          del target[:excess]

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
        **{"group_id": group_id, "pair_index": pair_index, **self.env_kwargs},  # pyrefly: ignore[bad-argument-type]
    )

    return agent, env

  def _model_call(
      self,
      chat_lists: List[Dict[str, str]] | None,
      env: Any = None,
      max_generation_steps: int | None = None,
      *,
      prompt_token_ids: np.ndarray | None = None,
  ) -> base_rollout.RolloutOutput:
    """Calls model generation from chat messages or from recorded token ids."""
    if env:
      env.task["policy_version"] = self.policy_version

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

    if prompt_token_ids is None:
      if chat_lists is None:
        raise ValueError(
            "chat_lists is required when prompt_token_ids is None."
        )
      # `_create_micro_batch_iterator` leaves task fields as 1-element arrays
      # and agents copy them into messages as is. Unwrap them as
      # `tokenize_and_generate_masks` does, so that the sampled and the trained
      # prompts are built from the same strings.
      messages = [
          agentic_utils.convert_messages_to_string(message)
          for message in chat_lists
      ]
      if self.chat_parser:
        prompt = self.chat_parser.parse(
            messages=messages,
            add_generation_prompt=True,
            is_first_msg=True,  # no op if system msg is populated in reset
        )
      else:
        prompt = messages
      return self.rl_engine.generate(
          prompts=[prompt],  # pyrefly: ignore[bad-argument-type]
          apply_chat_template=not self.chat_parser,
          mode=rl_engine_lib.Mode.TRAIN,
          trace_tags=tags,
          max_generation_steps=max_generation_steps,
      )

    return self.rl_engine.generate(
        prompts=None,
        apply_chat_template=False,
        mode=rl_engine_lib.Mode.TRAIN,
        trace_tags=tags,
        max_generation_steps=max_generation_steps,
        prompt_token_ids=[packing.as_token_ids(prompt_token_ids)],
    )

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
        max_response_length=self.algo_config.max_response_length,
        overlong_filter=self.algo_config.overlong_filter,
        filter_statuses=self.algo_config.filter_statuses,
        perf_v2=self.rl_engine.perf_v2,
        exact_token_continuity=self.algo_config.exact_token_continuity,
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
      group_id = self.rl_engine.global_steps * self._full_batch_size
      if is_async_iterator:
        async for single_example in prompt_iterator:  # pyrefly: ignore[not-iterable]
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
        for single_example in prompt_iterator:  # pyrefly: ignore[not-iterable]
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
            num_generations=self.algo_config.num_generations,
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
        self._background_tasks.add(cancellation_task)
        cancellation_task.add_done_callback(self._background_tasks.discard)

  def _batch_to_train_example(
      self,
      batch_results: list[Any],
      mode: rl_engine_lib.Mode,
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
    if mode != rl_engine_lib.Mode.TRAIN:
      expected_step = self.rl_engine.global_steps
    elif self._dynamic_sampler is not None:
      # Dropped groups are replaced, so group ids run ahead of the trainer.
      # Kept groups are counted in completion order, so with submission order
      # a reordered group may be attributed to the neighboring step (metrics
      # and perf tags only).
      expected_step = self._dynamic_sampler.expected_step
    else:
      expected_step = batch_results[0].group_id // self._full_batch_size

    return self._process_results(
        trajectories=batch_results,
        mode=mode,
        expected_step=expected_step,
    )

  def _compute_packed_logps(self, example: TrainExample) -> TrainExample:
    # pack-first hook: algorithms that defer old/ref logp under packing compute
    # them on the packed buffer here. Base is a no-op.
    return example

  @abc.abstractmethod
  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages.

    Learners that support `group_filters` set `_SUPPORTS_GROUP_FILTERS = True`
    and call `_check_group_filters` once the advantages are known, before any
    reference / actor forward pass, letting its `GroupDroppedError` propagate.

    Args:
      trajectories: The trajectory items of one group.
      mode: The current mode.
      expected_step: The global step the group is trained in.

    Returns:
      The group's TrainExamples.

    Raises:
      dynamic_sampling.GroupDroppedError: If `group_filters` drop the group.
    """

  def _check_group_filters(
      self,
      trajectories: list[Any],
      advantages: typing.ArrayLike,
      mode: rl_engine_lib.Mode,
  ) -> None:
    """Raises `GroupDroppedError` if dynamic sampling drops a training group.

    Args:
      trajectories: The group's trajectory items.
      advantages: The group's advantages.
      mode: The current mode; only training groups are filtered.

    Raises:
      dynamic_sampling.GroupDroppedError: If a filter drops the group.
    """
    if mode == rl_engine_lib.Mode.TRAIN and self._dynamic_sampler is not None:
      self._dynamic_sampler.check_group(trajectories, advantages)

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
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
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_feed: prompt_feed_lib.PromptFeed,
      train_data_queue: queue_lib.AbstractDataQueue,
  ):
    """Produces training examples from the prompts of `prompt_feed`.

    Args:
      orchestrator: The rollout orchestrator.
      prompt_feed: Source of the prompts; closed when production ends.
      train_data_queue: Output queue of TrainExamples (or raw groups).
    """
    try:
      async for batch in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_feed.prompts(),
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        if self._reached_max_steps():
          prompt_feed.close()
          break
        if self._can_skip_post_max_steps_group():
          prompt_feed.close()
          continue
        try:
          if self._process_in_consumer:
            # Wrap `batch` (list of trajectories) in a 1-element list so
            # `_queue_payloads` enqueues the raw trajectory group as one item
            # for the consumer, which handles iterations.
            self._emit_group(batch, [batch], train_data_queue)
          else:
            self._queue_train_examples(batch, train_data_queue)
        except Exception as e:
          if not isinstance(e, RuntimeError):
            logging.exception(
                "Exception in _producer while processing batch: %s", e
            )
          raise
      self._release_held_groups(train_data_queue)
    finally:
      # Signal production is complete for this batch, even if errors occurred.
      train_data_queue.put(None)
      # Ensure that any background threads waiting on the prompt feed are
      # unblocked.
      prompt_feed.close()

  def _queue_train_examples(
      self, group: list[Any], train_data_queue: queue_lib.AbstractDataQueue
  ) -> None:
    """Scores a group and queues its TrainExamples unless it is dropped.

    Args:
      group: The group's trajectory items.
      train_data_queue: Output queue of TrainExamples.
    """
    try:
      train_examples = self._batch_to_train_example(
          batch_results=group,
          mode=rl_engine_lib.Mode.TRAIN,
      )
    except dynamic_sampling.GroupDroppedError as dropped:
      sampler = self._dynamic_sampler
      assert sampler is not None  # Only a sampler's filters drop groups.
      self.rl_engine.buffer_metrics_async(
          {f"dynamic_sampling/dropped_groups/{dropped.reason}": (1, np.sum)},
          mode=rl_engine_lib.Mode.TRAIN,
          step=sampler.expected_step,
      )
      sampler.on_group_dropped(group, dropped.reason)
      self._emit_group(group, None, train_data_queue)
      return
    if self._dynamic_sampler is not None:
      self._dynamic_sampler.on_group_kept()
      if (
          self._training_config.max_steps
          and self._dynamic_sampler.expected_step
          >= self._training_config.max_steps
      ):
        self._dynamic_sampler.close()
    self._emit_group(group, train_examples, train_data_queue)

  def _can_skip_post_max_steps_group(self) -> bool:
    """Whether in-flight groups may be skipped because `max_steps` are queued.

    With submission-order mini-batches, kept lookahead groups wait in
    `_submission_emitter` until earlier groups resolve, so earlier groups must
    still be scored while any group is held.
    """
    return bool(
        self._dynamic_sampler is not None
        and (
            self._submission_emitter is None
            or not self._submission_emitter.held
        )
        and self._training_config.max_steps
        and self._dynamic_sampler.expected_step
        >= self._training_config.max_steps
    )

  def _emit_group(
      self,
      group: list[Any],
      items: list[Any] | None,
      train_data_queue: queue_lib.AbstractDataQueue,
  ) -> None:
    """Queues the items of a scored group, in submission order if configured.

    Args:
      group: The group's trajectory items.
      items: What to queue for the group, or None if it was dropped.
      train_data_queue: Output queue.
    """
    emitter = self._submission_emitter
    if emitter is None:
      payloads = [] if items is None else [items]
    elif items is None:
      payloads = emitter.drop(group[0].group_id)
    else:
      payloads = emitter.complete(group[0].group_id, items)
    self._queue_payloads(payloads, train_data_queue)

  def _queue_payloads(
      self,
      payloads: list[list[Any]],
      train_data_queue: queue_lib.AbstractDataQueue,
  ) -> None:
    """Queues the items of released groups, one payload per group.

    Args:
      payloads: The items of each released group, in release order.
      train_data_queue: Output queue.
    """
    for payload in payloads:
      for item in payload:
        train_data_queue.put(item)

  def _release_held_groups(
      self, train_data_queue: queue_lib.AbstractDataQueue
  ) -> None:
    """Queues the groups still held for submission order at the stream end.

    Groups are only held at the end if a lower-id group never completed;
    training on them is better than losing them.

    Args:
      train_data_queue: Output queue.
    """
    if (
        self._submission_emitter is None
        or not self._submission_emitter.held
        or self._reached_max_steps()
    ):
      return
    logging.warning(
        "Releasing %d groups held for submission order at the end of the"
        " prompt stream.",
        self._submission_emitter.held,
    )
    self._queue_payloads(self._submission_emitter.flush(), train_data_queue)

  def _start_submission_order(
      self, full_batch_size: int, mini_batch_size: int
  ) -> submission_order.SubmissionOrderEmitter[list[Any]] | None:
    """Creates the submission-order emitter of a `train` call, if needed.

    Args:
      full_batch_size: Groups per global step.
      mini_batch_size: Groups per optimizer update.

    Returns:
      The emitter, or None when groups are queued in completion order.
    """
    if (
        self.algo_config.minibatch_order != "submission"
        or mini_batch_size >= full_batch_size
    ):
      return None
    logging.info(
        "Mini-batches of %d groups are filled in submission order (%d"
        " optimizer updates per global step of %d groups).",
        mini_batch_size,
        full_batch_size // mini_batch_size,
        full_batch_size,
    )
    # Same numbering as `_orchestrator_producer`'s pair stream.
    return submission_order.SubmissionOrderEmitter(
        window_size=mini_batch_size,
        first_group_id=self.rl_engine.global_steps * full_batch_size,
    )

  def _check_batch_sizes(
      self, full_batch_size: int, mini_batch_size: int
  ) -> None:
    """Rejects algorithm options that the batch sizes cannot support.

    Called by `train` before any rollout; subclasses override it.

    Args:
      full_batch_size: Groups per global step.
      mini_batch_size: Groups per optimizer update.
    """
    del full_batch_size, mini_batch_size

  def _create_prompt_feed(
      self,
      batches: Iterator[TrainingInputT],
      full_batch_size: int,
  ) -> prompt_feed_lib.PromptFeed:
    """Creates the prompt feed of a `train` call.

    Args:
      batches: Full prompt batches.
      full_batch_size: Groups per global step.

    Returns:
      A `DynamicSampler` (also stored in `_dynamic_sampler`) when
      `group_filters` or `max_in_flight_groups` is set, else a
      `QueuePromptFeed` that admits one batch per global step.

    Raises:
      ValueError: If dynamic sampling is configured but groups would be
        scored in the consumer.
    """

    def split(batch: TrainingInputT) -> Iterator[TrainingInputT]:
      return self._create_micro_batch_iterator(iter([batch]), 1)

    self._dynamic_sampler = None
    if (
        not self.algo_config.group_filters
        and self.algo_config.max_in_flight_groups is None
    ):
      return prompt_feed_lib.QueuePromptFeed(
          batches,
          put_batch=self._put_prompts_to_queue,
          split=split,
          prefill_batches=self.algo_config.off_policy_steps + 1,
          loading_span=lambda: self.rl_engine.perf_v2.span(
              perf_constants.DATA_LOADING,
              tags={perf_constants.STEP: self.rl_engine.global_steps},
          ),
      )
    if self._process_in_consumer:
      raise ValueError(
          "group_filters / max_in_flight_groups need groups scored in the"
          " producer; they do not support compute_logps_micro_batch_size > 1"
          " without packing."
      )
    self._dynamic_sampler = dynamic_sampling.DynamicSampler(
        batches=batches,
        split=split,
        loop=self.loop,
        group_filters=self.algo_config.group_filters,
        max_consecutive_filtered_groups=(
            self.algo_config.max_consecutive_filtered_groups
        ),
        off_policy_steps=self.algo_config.off_policy_steps,
        full_batch_size=full_batch_size,
        max_in_flight_groups=self.algo_config.max_in_flight_groups,
        global_step_fn=lambda: self.rl_engine.global_steps,
    )
    self._warn_if_concurrency_caps_window(self._dynamic_sampler)
    return self._dynamic_sampler

  def _warn_if_concurrency_caps_window(
      self, sampler: dynamic_sampling.DynamicSampler
  ) -> None:
    """Logs the admission window and warns if max_concurrency caps it.

    Args:
      sampler: The dynamic sampler.
    """
    episodes = sampler.window_capacity * self.algo_config.num_generations
    if self.algo_config.max_concurrency < episodes:
      logging.warning(
          "max_concurrency=%d caps the %d episodes (%d groups x %d"
          " generations) that per-prompt admission keeps in flight.",
          self.algo_config.max_concurrency,
          episodes,
          sampler.window_capacity,
          self.algo_config.num_generations,
      )
    logging.info(
        "Per-prompt admission: group_filters=%s, %d groups in flight.",
        list(self.algo_config.group_filters),
        sampler.window_capacity,
    )

  def _reached_max_steps(self) -> bool:
    """Whether the trainer has completed `max_steps` global steps."""
    training_config = getattr(self, "_training_config", None)
    max_steps = training_config.max_steps if training_config else None
    return bool(max_steps and self.rl_engine.global_steps >= max_steps)

  def _data_consumer_batch_generator(
      self, queue: queue_lib.AbstractDataQueue, batch_size: int
  ):
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      if len(batch) < batch_size and self._dynamic_sampler is not None:
        # Per-prompt admission has no batch boundary at the end of the
        # dataset; drop any trailing partial micro-batch.
        logging.info(
            "Dropping trailing partial batch of %d < %d groups.",
            len(batch),
            batch_size,
        )
        return
      yield batch

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    if self.rl_engine.global_steps > 0:
      logging.info(
          "Skipping %d batches from train_dataset to fast-forward to step %d",
          self.rl_engine.global_steps,
          self.rl_engine.global_steps,
      )
      # TODO(b/483779605): Current implementation of fast-forwarding does not
      # take into account the mini-batch size. Follow-up CL will address this.
      for _ in range(self.rl_engine.global_steps):
        try:
          next(full_batch_iterator)
        except StopIteration:
          logging.warning("Train dataset exhausted while skipping batches.")
          self.rl_engine.close()
          return

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_engine.close()
      return

    full_batch_size = len(next(iter(first_item.values())))  # pyrefly: ignore[bad-argument-type]
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    if self._packing_enabled:
      train_micro_batch_size = None
      grad_acc_steps = None
    else:
      train_micro_batch_size = (
          self._training_config.train_micro_batch_size or mini_batch_size
      )
      grad_acc_steps = self._training_config.get_with_default(
          "gradient_accumulation_steps", 1
      )
    # Prompt groups dequeued per consumer step (`mini_batch_size` when packed).
    iterator_batch_size = train_micro_batch_size or mini_batch_size
    # Rollout micro batch size has to be 1 since we only process individual
    # prompts.
    self._rollout_micro_batch_size = 1
    self._process_in_consumer = False

    # pack-first: with packing on, the pack is the logp batch, so logp is
    # computed on the packed buffer after pack_sequences; do not defer
    # conversion to the consumer (which would enqueue raw lists pack_sequences
    # cannot consume).
    if self._compute_logps_micro_batch_size > 1 and not self._packing_enabled:
      if self._compute_logps_micro_batch_size != train_micro_batch_size:
        raise ValueError(
            "compute_logps_micro_batch_size"
            f" ({self._compute_logps_micro_batch_size}) must be equal to"
            f" train_micro_batch_size ({train_micro_batch_size})"
        )
      self._process_in_consumer = True

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
        f" {iterator_batch_size=}"
    )

    self._check_batch_sizes(full_batch_size, mini_batch_size)
    self._submission_emitter = self._start_submission_order(
        full_batch_size, mini_batch_size
    )

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    training_config = self.rl_engine.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    # 1. Start producer thread to generate rollouts and training examples.
    orchestrator = self._build_orchestrator()

    prompt_feed = self._create_prompt_feed(
        full_dataset_iterator, full_batch_size
    )
    producer_future = asyncio.run_coroutine_threadsafe(
        self._producer(orchestrator, prompt_feed, train_data_queue),
        self.loop,
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, iterator_batch_size
    )
    if self._process_in_consumer:
      # Convert raw Trajectory groups into TrainExamples up front, before
      # `pack_sequences` wraps the generator below. Otherwise `pack_sequences`
      # (via `unpad_train_example`) receives raw lists and crashes on
      # `.prompt_ids`.
      def _to_train_examples(raw_gen):
        for group_batch in raw_gen:
          all_trajectories = [t for group in group_batch for t in group]
          yield self._batch_to_train_example(
              batch_results=all_trajectories,
              mode=rl_engine_lib.Mode.TRAIN,
          )

      train_data_gen = _to_train_examples(train_data_gen)
    if self._packing_enabled:
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
          self._training_config.max_seq_token_per_tpu,  # pyrefly: ignore[bad-argument-type]
          sequences_per_update=mini_batch_size * self._num_generations(),
          pack_size=pack_size,
          max_segments_per_packed_row=getattr(
              self._training_config, "max_segments_per_packed_row", None
          ),
      )
    update_steps_since_last_sync = 0
    update_steps_per_full_batch = full_batch_size // mini_batch_size
    unpacked_micro_step_counter = 0
    did_eval_this_global_step = False
    full_batch_chunks = []
    for train_micro_batch in train_data_gen:
      if self._reached_max_steps():
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_engine.global_steps,
            self._training_config.max_steps,
        )
        prompt_feed.close()
        break
      self._iter_steps += 1

      # TODO(tsbao): Re-enable this once off-policy filtering is needed.
      # Filter out examples that are too old (off-policy).
      # filtered_train_micro_batch = self._filter_outdated_offpolicy_examples(
      #     train_micro_batch
      # )
      # if not filtered_train_micro_batch:
      #   continue
      # train_micro_batch = filtered_train_micro_batch

      # `train_micro_batch` is always a Sequence[TrainExample] now:
      #  - _process_in_consumer: converted up front (GRPO -> single-element list)
      #  - producer-side processing: TrainExamples straight from the queue
      #  - _packing_enabled: a single packed TrainExample from pack_sequences
      # jax.tree.map(concatenate) over a single-element list is a no-op, so this
      # equals the old `train_examples[0]` for the GRPO consumer path.
      merged_train_micro_batch = jax.tree.map(
          lambda *xs: jnp.concatenate(xs, axis=0), *train_micro_batch
      )

      if self._packing_enabled:
        # pack-first: old/ref logp were deferred (left None) so they can be
        # computed here on the packed buffer (segment-aware forward), sharing
        # the same packed representation training uses.
        merged_train_micro_batch = self._compute_packed_logps(
            merged_train_micro_batch
        )
        # Packing efficiency: segment_ids==0 marks padding and dummy packs, so
        # this is the wasted fraction. As a WeightedMetric it reduces by
        # Sum(pad)/Sum(total), which stays correct if pack shapes ever differ.
        seg = np.asarray(merged_train_micro_batch.segment_ids)
        dummy_ratio = sft_utils.WeightedMetric(
            jnp.array(float((seg == 0).sum())), jnp.array(float(seg.size))
        )
        self.rl_engine.buffer_metrics_async(
            {
                "packing/dummy_ratio": (  # pyrefly: ignore[bad-assignment]
                    dummy_ratio,
                    common.global_weighted_mean,
                ),
                "packing/seqs_per_pack": (
                    float(seg.max(axis=-1).mean()),
                    np.mean,
                ),
            },
            mode=rl_engine_lib.Mode.TRAIN,
            step=self.rl_engine.global_steps,
        )

      # When ``train_micro_batch_size < mini_batch_size`` we want the trainer
      # to invoke ``train_step`` multiple times per outer iteration so the
      # optimizer (which fires every ``gradient_accumulation_steps`` micro-
      # steps) sees ``mini_batch_size``-shaped gradients while peak HBM is
      # only ``train_micro_batch_size``-shaped. Slice the merged train
      # example along its batch axis into chunks sized to one micro-step,
      # and pass the list to ``update_actor``; ``peft_trainer.train``
      # iterates the list and calls ``train_step`` once per chunk.
      is_packed = (
          hasattr(merged_train_micro_batch, "segment_ids")
          and getattr(merged_train_micro_batch, "segment_ids") is not None
      )
      if not is_packed:
        assert train_micro_batch_size is not None
        seqs_per_chunk = (
            train_micro_batch_size * self.algo_config.num_generations
        )
        n_total = merged_train_micro_batch.completion_ids.shape[0]
        if n_total > seqs_per_chunk:
          chunked_train_micro_batch = [
              jax.tree_util.tree_map(
                  lambda x: (
                      x[i : i + seqs_per_chunk]
                      if hasattr(x, "shape")
                      and x.shape
                      and x.shape[0] == n_total
                      else x
                  ),
                  merged_train_micro_batch,
              )
              for i in range(0, n_total, seqs_per_chunk)
          ]
        else:
          chunked_train_micro_batch = [merged_train_micro_batch]
      else:
        chunked_train_micro_batch = [merged_train_micro_batch]

      full_batch_chunks.extend(chunked_train_micro_batch)

      # --- Evaluation Logic on FIRST microbatch ---
      current_eval_dataset = None
      if update_steps_since_last_sync == 0:
        current_train_step = self.rl_engine.actor_trainer.train_steps
        if (
            all_eval_prompts
            and current_train_step % training_config.eval_every_n_steps == 0
            and current_train_step != self._last_eval_train_step
        ):
          self._last_eval_train_step = current_train_step
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
                  rl_engine_lib.Mode.EVAL,
              )
              eval_examples.extend(eval_example)
            return eval_examples

          eval_future = asyncio.run_coroutine_threadsafe(
              _eval_runner_async(eval_orchestrator), self.loop
          )
          eval_examples = eval_future.result()
          self._eval_iter_steps += 1
          current_eval_dataset = eval_examples
          did_eval_this_global_step = True

      # --- First iteration Training Step (Parallelized with Rollout) ---
      # Note: Suppose one full batch has m minibatches, each minibatch has n
      # microbatches, and #iterations=K, we will:
      #   1. Train on the m * n microbatches once as we get them from rollout.
      #   2. When we get the full batch, repeat K-1 times on the entire batch.
      self.rl_engine.update_actor(
          chunked_train_micro_batch, current_eval_dataset, skip_jit
      )
      if hasattr(self.rl_engine, "critic_trainer"):
        self.rl_engine.update_critic(
            chunked_train_micro_batch, current_eval_dataset, skip_jit
        )

      # --- Weight Sync Logic ---
      if is_packed:
        # `merged_train_micro_batch.is_update_step` is a 0-d jax scalar set
        # by `pack_sequences`; pull the host-side value before deciding.
        is_update = bool(
            np.asarray(merged_train_micro_batch.is_update_step).item()
        )
      else:
        # Mirror `peft_trainer._train_step`'s derivation:
        # `is_update_step` flips True every `grad_acc_steps` micro-batches.
        assert grad_acc_steps is not None
        unpacked_micro_step_counter += 1
        is_update = unpacked_micro_step_counter % grad_acc_steps == 0

      if is_update:
        update_steps_since_last_sync += 1

      if update_steps_since_last_sync == update_steps_per_full_batch:
        # --- Remaining Iterations Training Step ---
        iterations = self._num_iterations()

        for i in range(1, iterations):
          # TODO(b/483779605) Sub-step checkpointing.
          self._iter_steps += len(full_batch_chunks)

          # TODO(yixuanm): Eval during iteration too. Skipping for now as we
          # will refactor the learner soon.
          self.rl_engine.update_actor(full_batch_chunks, None, skip_jit)
          if hasattr(self.rl_engine, "critic_trainer"):
            self.rl_engine.update_critic(full_batch_chunks, None, skip_jit)
        full_batch_chunks.clear()

        global_step_time = time.time() - self._global_step_start_time
        logging.info(
            f"Global step {self.rl_engine.global_steps} completed in"
            f" {global_step_time:.2f} seconds."
        )
        # One-line per-step diagnostic: raw rewards, solve rate, completion
        # length, advantage scale, and eval (when an eval just fired this
        # step). Mirrors the per-iter view a wandb dashboard would show
        # without depending on the async metric logger pipeline.
        with self._rewards_window_lock:
          train_rewards = np.asarray(
              self._train_rewards_window, dtype=np.float32
          )
          eval_rewards = np.asarray(self._eval_rewards_window, dtype=np.float32)
          self._train_rewards_window.clear()
          if did_eval_this_global_step:
            self._eval_rewards_window.clear()
        adv = np.asarray(merged_train_micro_batch.advantages, dtype=np.float32)
        cmask = np.asarray(
            merged_train_micro_batch.completion_mask, dtype=np.float32
        )
        compl_len = cmask.sum(axis=-1).mean() if cmask.size else 0.0
        adv_abs_mean = float(np.abs(adv).mean()) if adv.size else float("nan")
        train_r_mean = (
            float(train_rewards.mean()) if train_rewards.size else float("nan")
        )
        train_solve = (
            float((train_rewards > 0.1).mean())
            if train_rewards.size
            else float("nan")
        )
        if eval_rewards.size and did_eval_this_global_step:
          eval_r_mean = float(eval_rewards.mean())
          eval_solve = float((eval_rewards > 0.1).mean())
          eval_str = (
              f" eval_reward={eval_r_mean:.3f}"
              f" eval_solve={eval_solve:.3f}"
              f" eval_n={eval_rewards.size}"
          )
        else:
          eval_str = ""
        # Best-effort read of trainer-side per-step metrics (grad_norm,
        # pg_loss, entropy, kl) directly from the actor trainer's metric
        # buffer so they appear in the per-step absl log alongside the
        # rollout metrics, independently of any external metric logger.
        trainer_str = ""
        try:
          actor_trainer = self.rl_engine.actor_trainer
          trainer_buf = getattr(
              actor_trainer, "_prev_buffered_train_metrics", None
          ) or getattr(actor_trainer, "_buffered_train_metrics", None)
          if trainer_buf is not None:
            extras = []
            if trainer_buf.losses:
              extras.append(f"loss={float(trainer_buf.loss):.4f}")
            am = trainer_buf.additional_metrics
            for key, label in (
                ("grad_norm", "grad_norm"),
                ("reduced_pg_loss", "reduced_pg_loss"),
                ("entropy", "entropy"),
                ("kl", "kl"),
                ("log_ratio/abs_mean", "log_ratio_abs"),
                ("pg_clipfrac", "clipfrac"),
            ):
              if key in am:
                vals, _ = am[key]
                if vals:
                  v = float(
                      np.mean(
                          [np.asarray(common._metric_scalar(x)) for x in vals]
                      )
                  )
                  extras.append(f"{label}={v:.4f}")
            if extras:
              trainer_str = " " + " ".join(extras)
        except Exception as e:  # pylint: disable=broad-except
          logging.debug("Failed to read trainer buffered metrics: %s", e)
        logging.info(
            "[step %d] train_reward=%.3f train_solve=%.3f n=%d"
            " adv_abs_mean=%.3f compl_len=%.1f time=%.1fs%s%s",
            self.rl_engine.global_steps,
            train_r_mean,
            train_solve,
            int(train_rewards.size),
            adv_abs_mean,
            float(compl_len),
            global_step_time,
            trainer_str,
            eval_str,
        )
        did_eval_this_global_step = False
        self.rl_engine.buffer_metrics_async(
            {"perf/global_step_time": (global_step_time, np.mean)},
            mode=rl_engine_lib.Mode.TRAIN,
            step=self.rl_engine.global_steps,
        )
        if self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            with self.rl_engine.perf_v2.span(
                perf_constants.WEIGHT_SYNC,
                self.rl_engine.perf_v2.all_devices,
                tags={
                    perf_constants.STEP: self.rl_engine.global_steps,
                },
            ):
              self.rl_engine.sync_weights()
            self.policy_version += 1
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
          finally:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
        else:
          self.rl_engine.global_steps += 1
        prompt_feed.on_global_step(self._reached_max_steps())

        self.rl_engine.buffer_metrics(
            self.rl_engine.perf_v2.export(),
            mode=rl_engine_lib.Mode.TRAIN,
        )
        update_steps_since_last_sync = 0
        did_eval_this_global_step = False
        self._global_step_start_time = time.time()

    _ = producer_future.result()
    if update_steps_since_last_sync > 0:
      logging.warning(
          "Dataset ended mid-step after %d/%d optimizer updates at global"
          " step %d (max_steps=%s); the incomplete step was not synced.",
          update_steps_since_last_sync,
          update_steps_per_full_batch,
          self.rl_engine.global_steps,
          self._training_config.max_steps,
      )
    elif self._training_config.max_steps and not self._reached_max_steps():
      logging.warning(
          "Dataset ended after %d global steps before reaching max_steps=%d.",
          self.rl_engine.global_steps,
          self._training_config.max_steps,
      )
    self.rl_engine.close()

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
    if self._reached_max_steps():
      logging.info(
          "Reached max_steps: %d >= %d",
          self.rl_engine.global_steps,
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
              train_example.policy_version[0]  # pyrefly: ignore[unsupported-operation]
              for train_example in train_micro_batch
          ]),
          self.algo_config.off_policy_steps,
      )
    return filtered_train_micro_batch
