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

import asyncio
import contextlib
import dataclasses
import itertools
from typing import Any, Coroutine, Iterable, List, Sequence, TypeVar

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib


TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn

@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  policy_version: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class GRPOConfig(algo_config_lib.AlgorithmConfig):
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
  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
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


class GRPOLearner(rl_learner.RLLearner[TGrpoConfig]):
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
    self.algo_config = algo_config
    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = 0
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
        algo_config=self.algo_config,
    )
    self._full_batch_size = 0

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

  def _make_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int | None = None
  ) -> tuple[model_agent.ModelAgent, task_environment.TaskEnvironment]:
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a GRPO group.

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
    """Generates trajectory groups for GRPO using the orchestrator pattern.

    For each single-item input example, this function launches
    `G=num_generations` rollouts in parallel. It then yields a full group of G
    trajectories together with the original input for downstream advantage
    computation.

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

    This method takes the results from a group of `num_generations` rollouts
    (all from the same prompt) and processes them into individual
    `TrainExample` instances, one for each rollout.

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
    return self._process_results_and_compute_advantage(
        results=batch_results,
        training_input=training_input,
        mode=mode,
        step=step,
    )

  def _process_results_and_compute_advantage(
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

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generate text and compute advantages using Agentic RL framework.

    Note: This method is a placeholder from the base class and is not used
    in the GRPOLearner's asynchronous data pipeline. It returns None.

    Args:
      training_input: The input data for training.
      mode: The current mode (TRAIN or EVAL).
    """
    raise NotImplementedError(
        "_generate_and_compute_advantage is not used in AgenticGRPOLearner"
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, prompt_index: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is a string of format {row_offset}_{group_offset} where
    row_offset is the row index of the example data source and
    group_offset is the group index of the example in the generation group.

    In agentic GRPO, this method is called when processing rollouts for a
    single prompt, so `len(example["prompts"])` == `num_generations`,
    meaning `batch_size` will be 1.

    Args:
      example: The training input data for one prompt group.
      prompt_index: The index of the prompt in the dataset.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
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
    """Returns the number of GRPO iterations per batch."""
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
    """Main training loop for the GRPOLearner.

    This method orchestrates the entire training process using a
    producer-consumer
    pattern with asynchronous data generation.

    The loop proceeds as follows for each batch from the dataset:
    1. An asynchronous producer (`_producer`) is started. It consumes prompts
       from the dataset, generates `num_generations` rollouts for each using
       the orchestrator, computes advantages, and puts `TrainExample`s into
       a `train_data_queue`.
    2. The main loop consumes `TrainExample`s from the `train_data_queue` in
       micro-batches.
    3. For each micro-batch, it runs an evaluation cycle if needed and then
       calls `rl_cluster.update_actor` to perform a training step.
    4. After processing a full batch, model weights are synced.

    Args:
      train_dataset: An iterable of training data batches.
      eval_dataset: An optional iterable of evaluation data batches.
      skip_jit: If True, JIT compilation is skipped for the training step.
    """
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

    logging.info("Starting GRPOLearner training loop.")
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


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
