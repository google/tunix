# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A production-shaped single-turn GRPO loop over `RLOrchestrator`.

`GRPOLoop` is the layer that owns training-loop choreography for the
orchestrator stack. It keeps GRPO flow control out of the trainer worker:

  prompt groups -> rollout trajectories -> rewards -> advantages ->
  TrainExample -> train micro-batches -> actor update -> weight sync.

Algorithm-specific math remains in the injected `AlgorithmAdapter` behind the
orchestrator, while trainer workers only execute configured objectives.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Mapping, Sequence

import numpy as np


RewardFn = Callable[..., Sequence[float] | np.ndarray]


@dataclasses.dataclass(frozen=True)
class GRPOStepResult:
  """Summary returned by one GRPO train step."""

  step: int | None
  train_step: Any
  global_step: int
  num_prompt_groups: int
  num_trajectories: int
  num_chunks: int
  rewards: np.ndarray
  advantages: np.ndarray
  completions: Sequence[str]

  @property
  def reward_mean(self) -> float:
    return float(self.rewards.mean()) if self.rewards.size else 0.0

  @property
  def reward_std(self) -> float:
    return float(self.rewards.std()) if self.rewards.size else 0.0


def _encode(tokenizer: Any, text: str) -> list[int]:
  try:
    return tokenizer.encode(text, add_special_tokens=False)
  except TypeError:
    return tokenizer.encode(text)


def _slice_or_none(value: Any, start: int, end: int) -> Any:
  if value is None:
    return None
  return value[start:end]


def _pad_1d_right(
    values: Any,
    *,
    target_length: int,
    pad_value: float = 0.0,
    dtype: Any = np.float32,
) -> np.ndarray:
  arr = np.asarray(values, dtype=dtype).reshape(-1)
  out = np.full((target_length,), pad_value, dtype=dtype)
  copy_len = min(arr.shape[0], target_length)
  if copy_len:
    out[:copy_len] = arr[:copy_len]
  return out


def _padded_rollout_logps(
    rollout_logprobs: Any,
    *,
    num_trajectories: int,
    max_response_length: int,
) -> np.ndarray | None:
  """Pads rollout per-token logprobs to the completion tensor shape."""
  if rollout_logprobs is None:
    return None
  if len(rollout_logprobs) != num_trajectories:
    raise ValueError(
        "rollout returned a different number of logprob sequences than "
        f"prompts: {len(rollout_logprobs)} vs {num_trajectories}."
    )
  if any(logprobs is None for logprobs in rollout_logprobs):
    return None
  return np.stack([
      _pad_1d_right(
          logprobs,
          target_length=max_response_length,
          pad_value=0.0,
          dtype=np.float32,
      )
      for logprobs in rollout_logprobs
  ])


def _replace_train_example(
    example: Any,
    **updates: Any,
) -> Any:
  replace = getattr(example, "replace", None)
  if callable(replace):
    return replace(**updates)
  return dataclasses.replace(example, **updates)


def split_train_example(
    example: Any,
    micro_batch_size: int,
) -> list[Any]:
  """Splits a trajectory-level TrainExample into train micro-batches."""
  batch = int(example.prompt_ids.shape[0])
  if micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")

  chunks = []
  for start in range(0, batch, micro_batch_size):
    end = min(start + micro_batch_size, batch)
    chunks.append(
        _replace_train_example(
            example,
            prompt_ids=example.prompt_ids[start:end],
            prompt_mask=example.prompt_mask[start:end],
            completion_ids=example.completion_ids[start:end],
            completion_mask=example.completion_mask[start:end],
            advantages=example.advantages[start:end],
            ref_per_token_logps=_slice_or_none(
                example.ref_per_token_logps, start, end
            ),
            old_per_token_logps=_slice_or_none(
                example.old_per_token_logps, start, end
            ),
            policy_version=_slice_or_none(example.policy_version, start, end),
            sampler_is_weights=_slice_or_none(
                example.sampler_is_weights, start, end
            ),
        )
    )
  return chunks


def _expand_prompt_groups(
    prompt_groups: Sequence[str],
    num_generations: int,
) -> list[str]:
  return [
      prompt
      for prompt in prompt_groups
      for _ in range(num_generations)
  ]


def _expand_reward_kwargs(
    reward_kwargs: Mapping[str, Any],
    *,
    num_prompt_groups: int,
    num_trajectories: int,
    num_generations: int,
) -> dict[str, Any]:
  expanded = {}
  for key, value in reward_kwargs.items():
    if isinstance(value, (str, bytes, MappingABC)):
      expanded[key] = value
      continue
    try:
      value_len = len(value)
    except TypeError:
      expanded[key] = value
      continue

    if value_len == num_prompt_groups:
      expanded[key] = [
          item
          for item in value
          for _ in range(num_generations)
      ]
    elif value_len == num_trajectories:
      expanded[key] = value
    else:
      expanded[key] = value
  return expanded


class GRPOLoop:
  """Single-turn GRPO learner loop built on orchestrator primitives.

  The loop owns group/trajectory choreography and remains independent of where
  compute runs. Swapping the orchestrator's cluster changes single-process vs
  distributed execution without changing this loop.
  """

  def __init__(
      self,
      orchestrator: Any,
      *,
      reward_fn: RewardFn,
      tokenizer: Any,
      num_generations: int,
      max_prompt_length: int,
      max_response_length: int,
      train_micro_batch_size: int,
      pad_id: int,
      eos_id: int,
      sync_weights: bool = True,
      configure_trainer: bool = True,
      max_generation_steps: int | None = None,
  ):
    if num_generations <= 1:
      raise ValueError("num_generations must be greater than 1 for GRPO.")
    if train_micro_batch_size <= 0:
      raise ValueError("train_micro_batch_size must be positive.")

    self._orch = orchestrator
    self._reward_fn = reward_fn
    self._tokenizer = tokenizer
    self._num_generations = num_generations
    self._max_prompt_length = max_prompt_length
    self._max_response_length = max_response_length
    self._train_micro_batch_size = train_micro_batch_size
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._sync_weights = sync_weights
    self._max_generation_steps = max_generation_steps or max_response_length

    if configure_trainer:
      self._orch.configure_trainer()

  @property
  def num_generations(self) -> int:
    return self._num_generations

  def train(
      self,
      prompt_batches: Sequence[Sequence[str]],
      *,
      reward_kwargs_batches: Sequence[Mapping[str, Any] | None] | None = None,
      eval_ds: Any = None,
      skip_jit: bool = False,
  ) -> list[GRPOStepResult]:
    """Runs one GRPO step per prompt batch."""
    if reward_kwargs_batches is None:
      reward_kwargs_batches = [None] * len(prompt_batches)
    if len(reward_kwargs_batches) != len(prompt_batches):
      raise ValueError(
          "reward_kwargs_batches must have the same length as prompt_batches."
      )

    results = []
    for step, (prompt_groups, reward_kwargs) in enumerate(
        zip(prompt_batches, reward_kwargs_batches)
    ):
      results.append(
          self.train_step(
              prompt_groups,
              reward_kwargs=reward_kwargs,
              step=step,
              eval_ds=eval_ds,
              skip_jit=skip_jit,
          )
      )
    return results

  def train_step(
      self,
      prompt_groups: Sequence[str],
      *,
      reward_kwargs: Mapping[str, Any] | None = None,
      step: int | None = None,
      eval_ds: Any = None,
      skip_jit: bool = False,
  ) -> GRPOStepResult:
    """Runs rollout, reward, train, and optional weight sync for one batch."""
    if not prompt_groups:
      raise ValueError("prompt_groups must be non-empty.")

    reward_kwargs = reward_kwargs or {}
    expanded_prompts = _expand_prompt_groups(
        prompt_groups, self._num_generations
    )
    rollout_output = self._orch.generate(
        expanded_prompts,
        max_generation_steps=self._max_generation_steps,
    )
    completions = list(rollout_output.text)
    completion_tokens = rollout_output.tokens
    rollout_logps = getattr(rollout_output, "logprobs", None)
    if len(completions) != len(expanded_prompts):
      raise ValueError(
          "rollout returned a different number of completions than prompts: "
          f"{len(completions)} vs {len(expanded_prompts)}."
      )
    if len(completion_tokens) != len(expanded_prompts):
      raise ValueError(
          "rollout returned a different number of token sequences than "
          f"prompts: {len(completion_tokens)} vs {len(expanded_prompts)}."
      )
    old_per_token_logps = self._maybe_get_rollout_logps(
        rollout_logps, len(expanded_prompts)
    )

    expanded_reward_kwargs = _expand_reward_kwargs(
        reward_kwargs,
        num_prompt_groups=len(prompt_groups),
        num_trajectories=len(expanded_prompts),
        num_generations=self._num_generations,
    )
    rewards = np.asarray(
        self._reward_fn(
            prompts=expanded_prompts,
            completions=completions,
            **expanded_reward_kwargs,
        ),
        dtype=np.float32,
    )
    if rewards.shape[0] != len(expanded_prompts):
      raise ValueError(
          "reward_fn must return one reward per trajectory: "
          f"{rewards.shape[0]} vs {len(expanded_prompts)}."
      )

    advantages = self._orch.compute_advantages(
        rewards, num_generations=self._num_generations
    )
    prompt_tokens = [_encode(self._tokenizer, prompt) for prompt in expanded_prompts]
    train_example = self._orch.assemble_train_example(
        prompt_tokens,
        completion_tokens,
        advantages,
        max_prompt_length=self._max_prompt_length,
        max_response_length=self._max_response_length,
        pad_id=self._pad_id,
        policy_version=np.full(
            (len(expanded_prompts),), self._orch.global_steps, dtype=np.int32
        ),
        old_per_token_logps=old_per_token_logps,
    )
    train_example = self._maybe_add_ref_logps(train_example)
    chunks = split_train_example(train_example, self._train_micro_batch_size)
    train_step = self._orch.train_step(chunks, eval_ds=eval_ds, skip_jit=skip_jit)

    start_global_step = self._orch.global_steps
    if self._sync_weights:
      self._orch.sync_weights()
    if self._orch.global_steps == start_global_step:
      self._orch.global_steps = start_global_step + 1

    return GRPOStepResult(
        step=step,
        train_step=train_step,
        global_step=self._orch.global_steps,
        num_prompt_groups=len(prompt_groups),
        num_trajectories=len(expanded_prompts),
        num_chunks=len(chunks),
        rewards=rewards,
        advantages=np.asarray(advantages),
        completions=completions,
    )

  def _maybe_get_rollout_logps(
      self,
      rollout_logps: Any,
      num_trajectories: int,
  ) -> np.ndarray | None:
    algo_config = getattr(self._orch.algorithm, "algo_config", None)
    if not getattr(algo_config, "use_rollout_logps", False):
      return None
    old_per_token_logps = _padded_rollout_logps(
        rollout_logps,
        num_trajectories=num_trajectories,
        max_response_length=self._max_response_length,
    )
    if (
        getattr(algo_config, "num_iterations", 1) > 1
        and old_per_token_logps is None
    ):
      raise RuntimeError(
          "old_per_token_logps is not available for off-policy GRPO. Enable "
          "return_logprobs on the rollout worker or use one policy iteration."
      )
    return old_per_token_logps

  def _maybe_add_ref_logps(
      self,
      train_example: Any,
  ) -> Any:
    algo_config = getattr(self._orch.algorithm, "algo_config", None)
    beta = getattr(algo_config, "beta", 0.0)
    force_compute_kl = getattr(algo_config, "force_compute_kl", False)
    if beta == 0.0 and not force_compute_kl:
      return train_example

    ref_logps = self._orch.reference_logps(
        train_example.prompt_ids,
        train_example.completion_ids,
        self._pad_id,
        self._eos_id,
        micro_batch_size=self._train_micro_batch_size,
    )
    return _replace_train_example(
        train_example, ref_per_token_logps=ref_logps
    )
