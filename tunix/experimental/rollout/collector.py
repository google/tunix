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

import asyncio
from typing import Any, Callable, List, Optional
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import converter as converter_lib
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types
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
      trajectory_store: Optional[store.TrajectoryWriter] = None,
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
    self.trajectory_store = trajectory_store
    self.is_paused: bool = False
    self.is_cancelled: bool = False
    self.is_done: bool = False
    target_policy_versions = None
    target_policy_version = getattr(self.request, "target_policy_version", None)
    if target_policy_version is not None:
      target_policy_versions = [target_policy_version]

    self.metadata = converter_lib.create_trajectory_metadata(
        self.traj_id,
        self.request,
        self.agent,
        target_policy_versions=target_policy_versions,
        status=agent_types.TrajectoryStatus.RUNNING,
    )

  def _sync_metadata(
      self, status: str | agent_types.TrajectoryStatus | None = None
  ) -> None:
    """Syncs metadata status and agent trajectory timing/reward in place."""
    if self.metadata is not None:
      policy_version = getattr(self.request, "target_policy_version", None)
      converter_lib.update_trajectory_metadata(
          metadata=self.metadata,
          agent=self.agent,
          policy_version=policy_version,
          status=status,
      )

  async def run_episode(
      self,
  ) -> trajectory_lib.TunixTrajectory:
    """Executes multi-turn agentic rollout episode and returns standardized Trajectory."""
    # Note: model_call is an async coroutine callback invoked directly by
    # TrajectoryCollectEngine on the asyncio event loop without blocking
    # threads.
    async def model_call(
        chat_completions, env=None, max_generation_steps=None, **kwargs
    ):
      del env, kwargs
      generation_kwargs = dict(self.request.generation_kwargs)
      if max_generation_steps is not None:
        generation_kwargs["max_tokens"] = max_generation_steps

      sampling_params = sampler_lib.SamplingParams(
          max_tokens=generation_kwargs.get("max_tokens", 64),
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
        policy_version=getattr(self.request, "target_policy_version", None),
        trajectory_store=self.trajectory_store,
        metadata=self.metadata,
    )
    try:
      rl_traj = await inner_engine.collect(mode="Trajectory")
      self.is_done = True
      self._sync_metadata()
      steps = []
      task = getattr(rl_traj, "task", None) or getattr(
          self.request, "prompt", None
      )
      task_step = converter_lib.create_task_step(task)
      if task_step is not None:
        steps.append(task_step)
      for i, step in enumerate(getattr(rl_traj, "steps", []) or []):
        agent_step = converter_lib.create_agent_step(
            step,
            tunix_step_id=i,
            policy_version=getattr(self.request, "target_policy_version", None),
        )
        if agent_step is not None:
          steps.append(agent_step)
        env_step = converter_lib.create_env_step(step, tunix_step_id=i)
        if env_step is not None:
          steps.append(env_step)
      return trajectory_lib.TunixTrajectory(
          **self.metadata.model_dump(),
          steps=steps,
      )
    except asyncio.TimeoutError:
      self._sync_metadata(status=agent_types.TrajectoryStatus.TIMEOUT)
      raise
    except Exception:
      self._sync_metadata(status=agent_types.TrajectoryStatus.FAILED)
      raise
    finally:
      if self.trajectory_store is not None:
        self.trajectory_store.update_metadata(self.metadata)
        self.trajectory_store.flush()

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
