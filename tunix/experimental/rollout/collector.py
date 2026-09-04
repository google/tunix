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
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.trajectory import trajectory_collect_engine as rl_collect_engine
from tunix.rl.rollout import base_rollout

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

  async def run_episode(self) -> trajectory_lib.Trajectory:
    """Executes multi-turn agentic rollout episode and returns standardized Trajectory."""
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

      sampling_params = sampler_lib.SamplingParams(
          max_tokens=effective_max_tokens,
          temperature=generation_kwargs.get("temperature", 0.0),
          top_p=generation_kwargs.get("top_p", None),
          top_k=generation_kwargs.get("top_k", None),
          seed=generation_kwargs.get("seed", None),
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

      # TEMPORARY instrumentation: nothing else on this path records what the
      # sampler actually produced, so a run of empty completions reports the
      # same "rollouts=N" summary as a real one. Remove once the vllm sampler
      # path is trusted.
      logging.info(
          "[collector] traj=%s completion_tokens=%d prompt_tokens=%d"
          " logprobs=%s text=%r",
          self.traj_id,
          np.asarray(tokens).size,
          prompt_tokens.size,
          "none" if logprobs is None else np.asarray(logprobs).size,
          text[:160],
      )

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
    )
    rl_traj = await inner_engine.collect(mode="Trajectory")
    self.is_done = True
    return self._convert_to_trajectory(rl_traj)

  def _convert_to_trajectory(self, rl_traj: Any) -> trajectory_lib.Trajectory:
    """Converts internal rollout trajectory to standardized Trajectory format."""
    metadata = dict(self.request.metadata or {})
    metadata["prompt_id"] = self.request.prompt_id
    metadata["group_index"] = self.request.group_index
    assistant_text = "\n".join(
        str(getattr(step, "model_response", ""))
        for step in getattr(rl_traj, "steps", [])
        if getattr(step, "model_response", "")
    )
    metadata.setdefault("text", assistant_text)
    metadata["prompt_tokens"] = np.asarray(
        getattr(rl_traj, "prompt_tokens", np.zeros(0, dtype=np.int32)),
        dtype=np.int32,
    )
    metadata["reward"] = float(getattr(rl_traj, "reward", 0.0) or 0.0)
    trajectory = trajectory_lib.Trajectory(
        trajectory_id=self.traj_id,
        agent=trajectory_lib.Agent(
            name=getattr(self.agent, "name", "agent"),
            version="1.0",
        ),
        extra=metadata,
    )
    if hasattr(rl_traj, "steps"):
      for step in rl_traj.steps:
        obs_val = getattr(step, "observation", None)
        obs_obj = None
        if obs_val:
          obs_obj = trajectory_lib.Observation(
              results=[trajectory_lib.ObservationResult(content=str(obs_val))]
          )
        new_step = trajectory.add_step(
            source=trajectory_lib.Source.AGENT,
            message=getattr(step, "model_response", str(step)),
            observation=obs_obj,
        )
        extra_dict = {}
        for attr in (
            "assistant_tokens",
            "assistant_masks",
            "env_tokens",
            "env_masks",
            "logprobs",
        ):
          val = getattr(step, attr, None)
          if val is not None:
            extra_dict[attr] = val
            try:
              setattr(new_step, attr, val)
            except (AttributeError, ValueError):
              pass
        if extra_dict:
          new_step.extra = extra_dict
    return trajectory

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
