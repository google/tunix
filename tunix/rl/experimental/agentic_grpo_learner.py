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
from typing import Any, Coroutine, Iterable, List, Sequence

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.grpo import grpo_helpers
from tunix.rl.queue import data_queue as queue_lib


TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


@dataclasses.dataclass(slots=True, kw_only=True)
class GRPOConfig:
  """Configuration for GRPO algorithm.

  Parameters:
    num_generations: Number of samples per prompt (G in the paper). Must be > 1.
    num_iterations: Number of GRPO iterations per batch (μ in the paper).
    beta: KL penalty coefficient.
    epsilon: PPO-style clipping epsilon.
    loss_algo: "grpo" or "gspo-token".
    system_prompt: System prompt for the agent.
    max_concurrency: Maximum number of concurrent rollout engines.
  """

  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2
  loss_algo: str = "grpo"  # grpo or gspo-token
  system_prompt: str = ""
  max_concurrency: int = 16

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )
    if self.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )


class GRPOLearner(rl_learner.RLLearner):
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
      grpo_config: GRPOConfig,
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
      grpo_config: An instance of `GRPOConfig` containing all GRPO specific
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
    self.grpo_config = grpo_config
    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    # Workaround to pass loss fn with algorithm flag
    loss_fn = lambda model, train_example, beta, epsilon: grpo_loss_fn(
        model,
        train_example,
        beta=beta,
        epsilon=epsilon,
        pad_id=self.rl_cluster.rollout.pad_id(),
        eos_id=self.rl_cluster.rollout.eos_id(),
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
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
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
    task = {"question": question_text, "_original_input": single_example}
    if group_id is not None:
      task["group_id"] = group_id
    # Pass along other metadata from the original example.
    for key, value in single_example.items():
      if key not in ["prompts", "_original_input"]:
        task[key] = value[0]
    agent = model_agent.ModelAgent(system_prompt=self.grpo_config.system_prompt)
    # TODO: b/456528861 - Support both single-turn and multi-turn from config.
    env = task_environment.TaskEnvironment(
        task=task,
        reward_fn=reward.dummy_reward,
        max_steps=1,
    )
    return agent, env

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_defaults=dict(
            model_call=lambda chat_lists: self.rl_cluster.generate(
                prompts=chat_lists,
                apply_chat_template=True,
                mode=rl_cluster_lib.Mode.TRAIN,
            ).text[0],
            final_reward_fn=reward.dummy_reward,
            tokenizer=self.tokenizer,
            chat_parser=self.chat_parser,
        ),
        max_concurrency=self.grpo_config.max_concurrency,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT],
      episodes_per_pair: int = 1,
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
      episodes_per_pair: The number of episodes to run per agent-environment
        pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").

    Yields:
      A tuple where the first element is a list of trajectory results for a
      group, and the second is a list containing the original `TrainingInputT`
      for that group.
    """

    def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      for i, single_example in enumerate(prompt_iterator):
        for _ in range(self.grpo_config.num_generations):
          # group_id=i ensures all generations for the same prompt are grouped
          yield self._make_agent_env_pair(single_example, group_id=i)

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.grpo_config.num_generations,
            group_key=lambda i, env, traj: env.task["group_id"],
            episodes_per_pair=episodes_per_pair,
            collect_mode=collect_mode,
        )
    )

    # Let the producer start and initialize its manager before consuming.
    await asyncio.sleep(0)

    # Consume full groups and yield them with their original input.
    async_generator = orchestrator.yield_batches(
        batch_size=self.grpo_config.num_generations
    )
    try:
      async with contextlib.aclosing(async_generator) as stream:
        async for group in stream:
          if group:
            # Retrieve the original input embedded in the task.
            original_input = group[0].traj["_original_input"]
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
    num_generations = self.grpo_config.num_generations
    micro_batches = [cached_inputs_for_window[0]] * num_generations
    training_input = rl_utils.merge_micro_batches(micro_batches)

    steps = (
        self._iter_steps
        if mode == rl_cluster_lib.Mode.TRAIN
        else self._eval_iter_steps
    )
    trajectory_ids = self._compute_trajectory_ids(training_input, steps)
    assert "trajectory_ids" not in training_input
    training_input["trajectory_ids"] = trajectory_ids
    for t_id in trajectory_ids:
      self.rl_cluster.buffer_metrics(
          {
              "trajectory_ids": (t_id, None),
          },
          mode=mode,
      )
    return self._process_results_and_compute_advantage(
        results=batch_results, training_input=training_input, mode=mode
    )

  def _process_results_and_compute_advantage(
      self,
      results: List[Any],
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
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
    for item in results:
      conversation = item.traj.get("conversation_text") or []
      assistant_text = next(
          message["content"]
          for message in conversation
          if message["role"] == "assistant"
      )
      completion_texts.append(assistant_text)
      completion_tokens_list.append(item.traj.get("conversation_tokens"))

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
    if self.grpo_config.beta != 0.0:
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
    if self.grpo_config.num_iterations > 1:
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
    )

    logging.debug("Rewards computed: %s", rewards)
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
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    logging.debug("Advantages computed: %s", advantages)
    combined_batch = TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )
    return [
        rl_utils.get_batch_slice(combined_batch, slice(i, i + 1))
        for i in range(self.grpo_config.num_generations)
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
      self, example: TrainingInputT, steps: int
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

  def _num_iterations(self) -> int:
    """Returns the number of GRPO iterations per batch."""
    return self.grpo_config.num_iterations

  def _num_generations(self) -> int:
    """Returns the number of generations per prompt."""
    return self.grpo_config.num_generations

  @staticmethod
  def _run_async(coro: Coroutine) -> Any:
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

  async def _producer(self, orchestrator, prompt_queue, train_data_queue):
    """Produces training examples from prompts in the prompt_queue."""
    prompt_iterator = iter(lambda: prompt_queue.get(block=True), None)
    try:
      async for batch, cached_inputs in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          episodes_per_pair=1,
          collect_mode="Token",
      ):
        try:
          train_examples = self._batch_to_train_example(
              batch_results=batch,
              cached_inputs_for_window=cached_inputs,
              mode=rl_cluster_lib.Mode.TRAIN,
          )
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
    1. Prompts from the batch are added to a `prompt_queue`.
    2. An asynchronous producer (`_producer`) is started. It consumes prompts,
       generates `num_generations` rollouts for each using the orchestrator,
       computes advantages, and puts `TrainExample`s into a `train_data_queue`.
    3. The main loop consumes `TrainExample`s from the `train_data_queue` in
       micro-batches.
    4. For each micro-batch, it runs an evaluation cycle if needed and then
       calls `rl_cluster.update_actor` to perform a training step.
    5. After processing a full batch, model weights are synced.

    Args:
      train_dataset: An iterable of training data batches.
      eval_dataset: An optional iterable of evaluation data batches.
      skip_jit: If True, JIT compilation is skipped for the training step.
    """
    full_batch_iterator = iter(train_dataset)
    first_item = next(full_batch_iterator)
    full_batch_size = len(first_item["prompts"])
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

    prompt_queue = queue_lib.SimpleDataQueue(maxsize=full_batch_size + 1)
    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    for full_batch in full_dataset_iterator:
      # 1. Fill prompt queue for the current full batch.
      for prompt in self._create_micro_batch_iterator(iter([full_batch]), 1):
        prompt_queue.put(prompt)
      prompt_queue.put(None)  # Signal end of prompts.

      # 2. Start producer for this batch.
      orchestrator = self._build_orchestrator()
      producer_future = self.executor.submit(
          self._run_async,
          self._producer(orchestrator, prompt_queue, train_data_queue),
      )

      train_data_gen = self._data_consumer_batch_generator(
          train_data_queue, train_micro_batch_size * self._num_generations()
      )

      for train_micro_batch in train_data_gen:
        self._iter_steps += 1
        merged_train_micro_batch = jax.tree.map(
            lambda *xs: np.concatenate(xs, axis=0), *train_micro_batch
        )

        # --- Evaluation Logic ---
        current_eval_dataset = None
        # Run eval based on actor train steps, not this learner's steps
        if (
            all_eval_prompts
            and self.rl_cluster.actor_trainer.train_steps
            % training_config.eval_every_n_steps
            == 0
        ):
          # Create a new orchestrator for eval to not interfere
          eval_orchestrator = self._build_orchestrator()

          async def _eval_runner(current_eval_orchestrator):
            eval_examples = []
            self._eval_iter_steps = 0
            async for batch, cached_inputs in self._orchestrator_producer(
                current_eval_orchestrator, all_eval_prompts, episodes_per_pair=1
            ):
              train_examples = self._batch_to_train_example(
                  batch, cached_inputs, rl_cluster_lib.Mode.EVAL
              )
              eval_examples.extend(train_examples)
              self._eval_iter_steps += len(train_examples)
            return eval_examples

          eval_examples = self._run_async(_eval_runner(eval_orchestrator))
          if eval_examples:
            current_eval_dataset = eval_examples

        # --- Training Step ---
        self.rl_cluster.update_actor(
            [merged_train_micro_batch], current_eval_dataset, skip_jit
        )
        if hasattr(self.rl_cluster, "critic_trainer"):
          self.rl_cluster.update_critic(
              train_micro_batch, current_eval_dataset, skip_jit
          )

      _ = producer_future.result()
      # --- Weight Sync Logic ---
      if self.should_sync_weights:
        logging.info("Syncing weights after processing full batch.")
        self.rl_cluster.sync_weights()
      self.rl_cluster.global_steps += 1

    self.rl_cluster.close()


def grpo_loss_fn(
    model,
    train_example,
    beta,
    epsilon,
    loss_algo,
    pad_id,
    eos_id,
):
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
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )

  per_token_logps = common.compute_per_token_logps(
      model,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_logits=False,
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


GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
