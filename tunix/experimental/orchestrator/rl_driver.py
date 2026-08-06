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

"""RLDriver implementation for algorithm strategy and batch assembly (Layer 3)."""

import asyncio
import inspect
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.rl import algorithm_config
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.rewards import reward  # pylint: disable=unused-import

TrainExample = agentic_rl_learner.TrainExample
RewardFn = Callable[..., list[float]]
MetricFn = Callable[..., Mapping[str, Any]]


def _sync_or_async(coro: Any) -> Any:
  """Executes coroutine synchronously if no loop is running, else returns coro."""
  if inspect.iscoroutine(coro):
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      return coro
    return asyncio.run(coro)
  return coro


class RLDriver:
  """RL Algorithm Driver (Layer 3) and canonical facade for RL programs.

  Composes a DistributedRLEngine with algorithm math (advantage estimation,
  rewards, TIS weights) and batch assembly.
  """

  def __init__(
      self,
      rl_engine: distributed_rl_engine.DistributedRLEngine,
      algo_config: algorithm_config.AlgorithmConfig,
      reward_fns: RewardFn | list[RewardFn] | None = None,
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      tokenizer: Any | None = None,
      adapter: algorithm_adapter.AlgorithmAdapter | None = None,
  ):
    """Initializes RLDriver.

    Args:
      rl_engine: The DistributedRLEngine providing compute.
      algo_config: Algorithm configuration (e.g. GRPOConfig).
      reward_fns: Callable or list of callables for evaluating rewards.
      chat_parser: Optional parser for chat templating.
      metric_fns: Sequence of user-defined metric functions.
      tokenizer: Optional tokenizer adapter.
      adapter: Optional AlgorithmAdapter for algorithm math & batch assembly.
    """
    self.rl_engine = rl_engine
    self.algo_config = algo_config
    self.chat_parser = chat_parser
    self.metric_fns = list(metric_fns) if metric_fns else []
    self.tokenizer = tokenizer
    self.adapter = adapter or algorithm_adapter.get_algorithm_adapter(
        algo_config
    )
    self._policy_version = 0

    reward_manager_fn = function_registry.get_reward_manager(
        getattr(algo_config, "reward_manager", "agentic-sequence-level")
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )

  @property
  def policy_version(self) -> int:
    """Returns the current policy version (global steps)."""
    return self._policy_version

  @policy_version.setter
  def policy_version(self, value: int) -> None:
    self._policy_version = value

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    """Generates completions for prompts using the underlying engine."""
    coro = self.rl_engine.generate(
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        mode=mode,
        micro_batch_size=micro_batch_size,
        trace_tags=trace_tags,
        max_generation_steps=max_generation_steps,
    )
    return _sync_or_async(coro)

  async def dispatch_rollouts(
      self,
      requests: Sequence[datatypes.RolloutRequest] | Sequence[Any],
  ) -> None:
    """Dispatches rollout requests across workers asynchronously."""
    await self.rl_engine.dispatch_generate(requests)

  async def poll_rollouts(self, timeout_s: float = 0.1) -> list[Any]:
    """Polls completed rollout responses across workers."""
    return await self.rl_engine.poll_rollouts(timeout_s=timeout_s)

  async def score_async(
      self,
      group: Any,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> Any:
    """Scores a group of trajectories asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.process_results(
            trajectories=group,
            mode=mode,
            expected_step=expected_step,
        ),
    )

  def sync_weights(self) -> Any:
    """Synchronizes actor policy weights to rollout and inference replicas."""
    return _sync_or_async(self.rl_engine.sync_weights())

  def train(
      self,
      role: datatypes.Role,
      train_ds: Any,
      eval_ds: Any = None,
      skip_jit: bool = False,
  ) -> None:
    """Executes a training update on the underlying engine."""
    self.rl_engine.train(
        role=role, train_ds=train_ds, eval_ds=eval_ds, skip_jit=skip_jit
    )

  def train_step(
      self,
      batch: Any,
      role: datatypes.Role = datatypes.Role.ACTOR,
      skip_jit: bool = False,
  ) -> Any:
    """Executes a single atomic gradient update step over a batch."""
    coro = self.rl_engine.train_step(batch=batch, role=role, skip_jit=skip_jit)
    return _sync_or_async(coro)

  def create_queue_manager(
      self,
      group_size: int | None = None,
      max_staleness: int | None = None,
      filter_fn: Any | None = None,
  ) -> trajectory_queue_manager.TrajectoryQueueManager:
    """Creates a TrajectoryQueueManager for out-of-order prompt grouping.

    Args:
      group_size: Target number of trajectories per ready group (defaults to
        algo_config.num_generations or 1).
      max_staleness: Optional maximum allowed policy lag. If set, trajectories
        with policy_version < (current_expected_version - max_staleness) are
        filtered out.
      filter_fn: Optional custom filtering function.

    Returns:
      A TrajectoryQueueManager configured for grouping and filtering.
    """
    target_group_size = (
        group_size
        if group_size is not None
        else getattr(self.algo_config, "num_generations", 1)
    )

    combined_filter_fn = filter_fn
    if max_staleness is not None:

      def _staleness_filter(group: Sequence[Any]) -> Any:
        current_version = getattr(self, "expected_policy_version", 0)
        min_allowed = current_version - max_staleness
        valid = [
            item
            for item in group
            if getattr(item, "policy_version", 0) >= min_allowed
        ]
        filtered = [
            item
            for item in group
            if getattr(item, "policy_version", 0) < min_allowed
        ]
        if filter_fn is not None:
          res = filter_fn(valid)
          if isinstance(res, tuple):
            return res[0], list(res[1]) + filtered
          return res, filtered
        return valid, filtered

      combined_filter_fn = _staleness_filter

    return trajectory_queue_manager.TrajectoryQueueManager(
        group_size=target_group_size,
        filter_fn=combined_filter_fn,
    )

  async def process_ready_groups(
      self,
      queue: trajectory_queue_manager.TrajectoryQueueManager,
      num_groups: int = 1,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
      expected_step: int = 0,
  ) -> list[TrainExample]:
    """Dequeues ready prompt groups and converts them to TrainExamples.

    Args:
      queue: The TrajectoryQueueManager instance.
      num_groups: Number of prompt groups to dequeue and process.
      mode: Mode (TRAIN or EVAL).
      expected_step: Current training step.

    Returns:
      A list of TrainExample instances ready for train_step.
    """
    group_size = queue.group_size or 1
    items = await queue.get_batch(num_groups * group_size)
    if not items:
      return []

    train_examples = []
    for i in range(0, len(items), group_size):
      group_items = items[i : i + group_size]
      if len(group_items) == group_size:
        examples = self.process_results(
            trajectories=group_items,
            mode=mode,
            expected_step=expected_step,
        )
        train_examples.extend(examples)
    return train_examples

  def _compute_rewards(
      self,
      prompts: list[str],
      completions: list[str],
      mode: rl_engine_lib.Mode,
      expected_step: int | None = None,
      **kwargs: Any,
  ) -> np.ndarray:
    """Computes rewards using the reward manager."""
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    rewards_info = self.reward_manager(
        prompts=prompts,
        completions=completions,
        **kwargs,
    )
    expected_step = 0 if expected_step is None else expected_step
    if hasattr(self.rl_engine, "buffer_metrics_async"):
      self.rl_engine.buffer_metrics_async(
          rewards_info["log_metrics"], mode=mode, step=expected_step
      )
    return np.asarray(rewards_info["rewards"])

  def _sampler_trainer_agreement(
      self,
      rollout_per_token_logps: Any,
      trainer_per_token_logps: Any,
      completion_mask: Any,
  ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Computes sampler-vs-trainer agreement metrics and TIS weights."""
    metrics = {}
    sampler_is_weights = None
    if rollout_per_token_logps is None or trainer_per_token_logps is None:
      return metrics, sampler_is_weights

    mask = completion_mask.astype(jnp.bool_)
    mask_f = mask.astype(jnp.float32)
    mask_sum = jnp.maximum(mask_f.sum(), 1.0)
    diff = jnp.abs(rollout_per_token_logps - trainer_per_token_logps)
    diff_mean = float((diff * mask_f).sum() / mask_sum)
    diff_max = float(jnp.where(mask, diff, 0.0).max())

    rp = jnp.exp(rollout_per_token_logps)
    tp = jnp.exp(trainer_per_token_logps)
    prob_diff = jnp.abs(rp - tp)
    prob_diff_mean = float((prob_diff * mask_f).sum() / mask_sum)
    prob_diff_max = float(jnp.where(mask, prob_diff, 0.0).max())

    rp_flat, tp_flat, mf = rp.reshape(-1), tp.reshape(-1), mask_f.reshape(-1)
    rp_mean = (rp_flat * mf).sum() / mask_sum
    tp_mean = (tp_flat * mf).sum() / mask_sum
    rp_d = (rp_flat - rp_mean) * mf
    tp_d = (tp_flat - tp_mean) * mf
    cov = (rp_d * tp_d).sum() / mask_sum
    rp_var = (rp_d * rp_d).sum() / mask_sum
    tp_var = (tp_d * tp_d).sum() / mask_sum
    pearson = float(cov / jnp.sqrt(jnp.maximum(rp_var * tp_var, 1e-12)))

    metrics.update({
        "sampler_trainer/logp_diff_mean": (diff_mean, np.mean),
        "sampler_trainer/logp_diff_max": (diff_max, np.max),
        "sampler_trainer/prob_diff_mean": (prob_diff_mean, np.mean),
        "sampler_trainer/prob_diff_max": (prob_diff_max, np.max),
        "sampler_trainer/probs_pearson_corr": (pearson, np.mean),
    })

    if getattr(self.algo_config, "sampler_is", "") == "token":
      asst_mask_f = completion_mask.astype(jnp.float32)
      log_ratio = trainer_per_token_logps - rollout_per_token_logps
      log_ratio = jnp.clip(log_ratio, min=-20.0, max=20.0)
      threshold = getattr(self.algo_config, "sampler_is_threshold", 5.0)
      sampler_is_weights = jax.lax.stop_gradient(
          jnp.minimum(jnp.exp(log_ratio), threshold) * asst_mask_f
      )
      is_mask_sum = jnp.maximum(asst_mask_f.sum(), 1.0)
      is_mean = float((sampler_is_weights * asst_mask_f).sum() / is_mask_sum)
      is_max = float(jnp.where(asst_mask_f > 0, sampler_is_weights, 0.0).max())
      metrics.update({
          "sampler_is/weight_mean": (is_mean, np.mean),
          "sampler_is/weight_max": (is_max, np.max),
      })
    return metrics, sampler_is_weights

  def process_results(
      self,
      trajectories: Sequence[Any],
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> list[TrainExample]:
    """Processes generation results, evaluates rewards, and computes advantages.

    Args:
      trajectories: A sequence of trajectory results for a prompt group.
      mode: Current mode (TRAIN or EVAL).
      expected_step: Expected training step index.

    Returns:
      A list containing the formatted TrainExample ready for training.
    """
    if not trajectories:
      return []

    if self.tokenizer is None:
      raise ValueError("RLDriver.process_results requires a valid tokenizer.")
    pad_value = self.tokenizer.pad_id()
    eos_value = self.tokenizer.eos_id()

    completion_texts: list[str] = []
    prompt_tokens_list: list[np.ndarray] = []
    completion_tokens_list: list[np.ndarray] = []
    completion_masks_list: list[np.ndarray] = []
    old_logprobs_list: list[np.ndarray] = []
    policy_versions_list: list[int] = []
    trajectory_rewards_list: list[float] = []

    for item in trajectories:
      traj_dict = getattr(item, "traj", item)
      conversation = traj_dict.get("conversation_text") or []
      assistant_text = next(
          (
              message["content"]
              for message in conversation
              if message.get("role") == "assistant"
          ),
          "",
      )
      completion_texts.append(assistant_text)
      prompt_tokens_list.append(traj_dict.get("prompt_tokens"))
      completion_tokens_list.append(traj_dict.get("conversation_tokens"))
      completion_masks_list.append(traj_dict.get("conversation_masks"))
      old_logprobs_list.append(traj_dict.get("old_logprobs"))
      policy_version = traj_dict.get("policy_version", 0)
      policy_versions_list.append(policy_version)
      trajectory_rewards_list.append(traj_dict.get("trajectory_reward", 0.0))

    rollout_config = getattr(self.algo_config, "rollout_config", None)
    if rollout_config is None and hasattr(self.rl_engine, "get_rollout_config"):
      rollout_config = self.rl_engine.get_rollout_config(mode)
    elif isinstance(rollout_config, dict):
      rollout_config = rollout_config.get(mode, rollout_config)

    padded_prompt_ids = []
    padded_completion_ids = []
    padded_completion_masks = []
    padded_old_logprobs = []

    max_response_len = getattr(self.algo_config, "max_response_length", 1024)
    for prompt_tokens, completion_tokens, completion_mask, old_logprobs in zip(
        prompt_tokens_list,
        completion_tokens_list,
        completion_masks_list,
        old_logprobs_list,
    ):
      padded_prompt, padded_completion, _ = (
          agentic_utils.pad_prompt_and_completion(
              prompt_tokens,
              completion_tokens,
              rollout_config.max_prompt_length,
              max_response_len,
              pad_value,
          )
      )
      padded_prompt_ids.append(padded_prompt)
      padded_completion_ids.append(padded_completion[:max_response_len])
      padded_completion_masks.append(
          agentic_utils.right_pad(completion_mask, max_response_len, 0)[
              :max_response_len
          ]
      )
      if getattr(self.algo_config, "use_rollout_logps", True):
        if old_logprobs is not None:
          padded_old_logprobs.append(
              agentic_utils.right_pad(
                  old_logprobs,
                  length=max_response_len,
                  pad=0.0,
                  dtype=old_logprobs.dtype,
              )[:max_response_len]
          )
        else:
          padded_old_logprobs.append(
              np.zeros(max_response_len, dtype=np.float32)
          )

    prompt_ids = jnp.asarray(padded_prompt_ids)
    prompt_mask = prompt_ids != pad_value
    completion_ids = jnp.asarray(padded_completion_ids)
    completion_mask = jnp.asarray(padded_completion_masks)

    configured_compute_logps = getattr(
        self.algo_config,
        "compute_logps_micro_batch_size",
        None,
    )
    num_generations = getattr(
        self.algo_config, "num_generations", len(trajectories)
    )
    compute_logps_micro_batch_size = (
        configured_compute_logps * num_generations
        if configured_compute_logps
        else len(trajectories)
    )

    rollout_per_token_logps = None
    trainer_per_token_logps = None
    if (
        getattr(self.algo_config, "use_rollout_logps", True)
        and padded_old_logprobs
    ):
      rollout_per_token_logps = jnp.asarray(padded_old_logprobs)
      old_per_token_logps = rollout_per_token_logps
      if getattr(self.algo_config, "sampler_is", "") == "token":
        trainer_per_token_logps = self.rl_engine.per_token_logps(
            role=datatypes.Role.ACTOR,
            prompt_tokens=prompt_ids,
            completion_tokens=completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=compute_logps_micro_batch_size,
        )
        old_per_token_logps = trainer_per_token_logps
    else:
      trainer_per_token_logps = self.rl_engine.per_token_logps(
          role=datatypes.Role.ACTOR,
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=compute_logps_micro_batch_size,
      )
      old_per_token_logps = trainer_per_token_logps

    ref_per_token_logps = None
    if (
        getattr(self.algo_config, "force_compute_kl", False)
        or getattr(self.algo_config, "beta", 0.0) != 0.0
    ):
      ref_per_token_logps = self.rl_engine.per_token_logps(
          role=datatypes.Role.REFERENCE,
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=compute_logps_micro_batch_size,
      )

    original_inputs_list = [
        getattr(item, "traj", item).get("original_input", {})
        for item in trajectories
    ]
    original_inputs = rl_utils.merge_micro_batches(original_inputs_list)

    reward_kwargs = {
        key: value for key, value in original_inputs.items() if key != "prompts"
    }
    reward_kwargs["trajectory_rewards"] = trajectory_rewards_list
    rewards = self._compute_rewards(
        prompts=original_inputs["prompts"],
        completions=completion_texts,
        mode=mode,
        expected_step=expected_step,
        **reward_kwargs,
    )

    advantages = jnp.asarray(
        self.adapter.compute_advantages(
            rewards=rewards, num_generations=num_generations
        )
    )

    agreement_metrics, sampler_is_weights = self._sampler_trainer_agreement(
        rollout_per_token_logps, trainer_per_token_logps, completion_mask
    )
    if agreement_metrics and hasattr(self.rl_engine, "buffer_metrics_async"):
      self.rl_engine.buffer_metrics_async(
          agreement_metrics, mode=mode, step=expected_step
      )

    policy_versions = np.array(policy_versions_list, dtype=np.int32)
    combined_batch = self.adapter.assemble_train_example(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=advantages,
        ref_per_token_logps=ref_per_token_logps,
        old_per_token_logps=old_per_token_logps,
        policy_version=policy_versions,
        sampler_is_weights=sampler_is_weights,
    )
    return [combined_batch]
