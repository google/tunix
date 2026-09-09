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

"""Trajectory Collector Engine wrapping TrajectoryCollectEngine with pause/resume/cancel control."""

import logging
from typing import Any, List
import zlib
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import trajectory_collect_engine as rl_collect_engine
from tunix.rl.rollout import base_rollout

_DEFAULT_EPISODE_TIMEOUT_SECS: float = 600.0


def generate_vanilla_rollout_seed(
    prompt_id: str | int,
    group_index: int = 0,
) -> int:
  """Generates a deterministic rollout seed for vanilla samplers.

  Computes `seed = (crc32(prompt_id) & 0x7FFFFFFF + group_index) & 0x7FFFFFFF`.

  Args:
    prompt_id: Prompt identifier string or integer (e.g. 'prompt_0', 42,
      'gsm8k_q1').
    group_index: The index of the rollout within its group (group_index >= 0).

  Returns:
    A deterministic 31-bit non-negative integer seed for the rollout request.
  """
  prompt_hash = zlib.crc32(str(prompt_id).encode("utf-8")) & 0x7FFFFFFF
  return (prompt_hash + group_index) & 0x7FFFFFFF


def _build_prompt(chat_parser: Any, chat_completions: Any) -> Any:
  """Vanilla samplers take a string; parse chat messages when needed."""
  if chat_parser and not isinstance(chat_completions, str):
    return chat_parser.parse(
        chat_completions, add_generation_prompt=True, is_first_msg=True
    )
  return chat_completions


class TrajectoryCollectorEngine:
  """Wrapper around TrajectoryCollectEngine providing lifecycle controls and Trajectory conversion."""

  def __init__(
      self,
      traj_id: str,
      request: datatypes.RolloutRequest,
      sampler: sampler_lib.Sampler,
      env_client: Any,
      agent: Any,
      tokenizer: Any,
      chat_parser: Any,
  ):
    if (
        sampler is None
        or env_client is None
        or agent is None
        or tokenizer is None
        or chat_parser is None
    ):
      raise ValueError(
          "TrajectoryCollectorEngine requires valid sampler, env_client, agent,"
          " tokenizer, and chat_parser arguments (none can be None)."
      )
    self.traj_id = traj_id
    self.request = request
    self.sampler = sampler
    self.env = env_client
    self.agent = agent
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self.is_paused: bool = False
    self.is_cancelled: bool = False
    self.is_done: bool = False
    self.max_response_length = request.generation_kwargs.get(
        "max_response_length"
    )
    metadata = request.metadata or {}
    timeout = metadata.get("episode_timeout")
    self.episode_timeout = float(
        timeout if timeout is not None else _DEFAULT_EPISODE_TIMEOUT_SECS
    )
    if self.episode_timeout <= 0:
      raise ValueError("episode_timeout must be positive.")
    overlong_filter = metadata.get("overlong_filter")
    if overlong_filter is None:
      self.overlong_filter = False
    elif isinstance(overlong_filter, bool):
      self.overlong_filter = overlong_filter
    else:
      raise TypeError(
          "overlong_filter must be a boolean, got"
          f" {type(overlong_filter).__name__}: {overlong_filter!r}."
      )

  async def run_episode(self) -> agent_types.TrajectoryItem:
    """Executes multi-turn agentic rollout episode and returns TrajectoryItem."""
    # Note: model_call is an async coroutine callback invoked directly by
    # TrajectoryCollectEngine on the asyncio event loop without blocking
    # threads.
    async def model_call(
        chat_completions, env=None, max_generation_steps=None, **kwargs
    ):
      del env, kwargs
      generation_kwargs = dict(self.request.generation_kwargs)
      request_max_generation_steps = generation_kwargs.pop(
          "max_generation_steps", None
      )

      if max_generation_steps is not None:
        effective_max_tokens = max_generation_steps
      elif request_max_generation_steps is not None:
        effective_max_tokens = request_max_generation_steps
      else:
        raise ValueError(
            "TrajectoryCollectorEngine requires either"
            " request.generation_kwargs or the model_call callback to specify"
            " max_generation_steps."
        )

      generation_kwargs["max_tokens"] = effective_max_tokens

      seed = generation_kwargs.get("seed", None)
      if isinstance(
          self.sampler, vanilla_sampler_adapter.VanillaSamplerAdapter
      ):
        # TODO(tunix-dev): make vanilla sampler stateful with internal RNG key
        if seed is None and self.request.prompt_id is not None:
          seed = generate_vanilla_rollout_seed(
              self.request.prompt_id, self.request.group_index
          )
        if seed is None:
          raise ValueError(
              "Vanilla sampler requires a seed or valid prompt_id to generate"
              " diverse rollouts, but got seed=None."
          )

      sampling_params = sampler_lib.SamplingParams(
          max_tokens=effective_max_tokens,
          temperature=generation_kwargs.get("temperature", 0.0),
          top_p=generation_kwargs.get("top_p", None),
          top_k=generation_kwargs.get("top_k", None),
          seed=seed,
          return_logprobs=generation_kwargs.get("return_logprobs", False),
      )
      sampling_req = sampler_lib.SamplingRequest(
          request_id=self.traj_id,
          prompt=_build_prompt(self.chat_parser, chat_completions),
          sampling_params=sampling_params,
      )
      res = await self.sampler.sample(sampling_req, **generation_kwargs)
      text = res if isinstance(res, str) else getattr(res, "text", str(res))
      tokens = getattr(res, "token_ids", np.array([], dtype=np.int32))
      logprobs = getattr(res, "logprobs", None)
      prompt_tokens = np.asarray(
          getattr(res, "prompt_token_ids", np.array([], dtype=np.int32)),
          dtype=np.int32,
      ).reshape(-1)
      if prompt_tokens.size:
        prompt_tokens = prompt_tokens.reshape(1, -1)
      else:
        prompt_tokens = np.array([[0]], dtype=np.int32)

      return base_rollout.RolloutOutput(
          text=[text],
          logits=None,
          tokens=[tokens],
          left_padded_prompt_tokens=prompt_tokens,
          logprobs=[logprobs] if logprobs is not None else None,
      )

    if not self.agent or not self.env:
      raise RuntimeError(
          "RolloutCollector requires valid registered agent and env instances"
          " to run an episode."
      )
    inner_engine = rl_collect_engine.TrajectoryCollectEngine(
        agent=self.agent,
        env=self.env,
        model_call=model_call,  # pyrefly: ignore[bad-argument-type]
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        max_response_length=self.max_response_length,
        timeout=self.episode_timeout,
        overlong_filter=self.overlong_filter,
    )
    rl_traj = await inner_engine.collect(mode="Token")
    self.is_done = True
    return self._convert_to_trajectory(rl_traj)

  def _convert_to_trajectory(
      self, rl_traj: dict[str, Any]
  ) -> agent_types.TrajectoryItem:
    """Converts internal Token-mode rollout trajectory to agent_types.TrajectoryItem."""
    if not isinstance(rl_traj, dict):
      raise TypeError(
          f"Expected rl_traj to be a dict, got {type(rl_traj).__name__}"
      )

    metadata = dict(self.request.metadata or {})
    metadata["prompt_id"] = self.request.prompt_id
    metadata["group_index"] = self.request.group_index
    metadata.setdefault("text", rl_traj.get("conversation_text", ""))
    metadata["reward"] = float(rl_traj.get("trajectory_reward", 0.0) or 0.0)
    metadata["status"] = rl_traj.get("status", "")
    policy_version = getattr(
        self.request,
        "target_policy_version",
        rl_traj.get("policy_version", 0),
    )
    metadata["policy_version"] = int(policy_version or 0)
    prompt_tokens = rl_traj.get("prompt_tokens")
    if prompt_tokens is not None:
      metadata["prompt_tokens"] = np.asarray(prompt_tokens, dtype=np.int32)

    steps_history = []
    agent_traj = getattr(self.agent, "trajectory", None)
    if agent_traj is not None and hasattr(agent_traj, "steps"):
      for step in agent_traj.steps:
        steps_history.append({
            "model_response": str(getattr(step, "model_response", "")),
            "thought": str(getattr(step, "thought", "")),
            "action": str(getattr(step, "action", "")),
            "observation": str(getattr(step, "observation", "")),
            "reward": float(getattr(step, "reward", 0.0) or 0.0),
            "done": bool(getattr(step, "done", False)),
        })
    metadata["steps_history"] = steps_history
    logging.info(
        "[RolloutCollector] Completed trajectory %s (prompt_id=%s, reward=%.2f,"
        " turns=%d)",
        self.traj_id,
        self.request.prompt_id,
        metadata["reward"],
        len(steps_history),
    )

    return agent_types.TrajectoryItem(
        prompt_id=self.request.prompt_id,
        group_index=self.request.group_index,
        start_step=0,
        traj=rl_traj,
        metadata=metadata,
    )

  def pause(self) -> None:
    self.is_paused = True

  def resume(self) -> None:
    self.is_paused = False

  def cancel(self) -> None:
    self.is_cancelled = True
    self.is_done = True

  def get_accumulated_token_ids(self) -> List[int]:
    """Returns token IDs of historical turns for Raiden KV-cache transfer."""
    return []
