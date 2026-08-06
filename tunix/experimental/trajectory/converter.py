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

"""Converter for translating between Tunix RL Step and ATIF Step representations."""

from typing import Any
import numpy as np
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


def _extract_tool_calls(
    action: agent_types.Action | None,
) -> list[trajectory_lib.ToolCall] | None:
  """Extracts ATIF ToolCalls from an RL Action."""
  if action is None or action.action is None:
    return None
  payload = action.action
  if isinstance(payload, list):
    calls = []
    for idx, item in enumerate(payload, 1):
      if isinstance(item, trajectory_lib.ToolCall):
        calls.append(item)
      elif isinstance(item, dict):
        calls.append(
            trajectory_lib.ToolCall(
                tool_call_id=str(
                    item.get("id", item.get("tool_call_id", f"call_{idx}"))
                ),
                function_name=str(
                    item.get("name", item.get("function_name", "action"))
                ),
                arguments=item.get("arguments", item.get("args", item)),
            )
        )
    return calls or None
  if isinstance(payload, dict):
    return [
        trajectory_lib.ToolCall(
            tool_call_id=str(
                payload.get("id", payload.get("tool_call_id", "call_1"))
            ),
            function_name=str(
                payload.get("name", payload.get("function_name", "action"))
            ),
            arguments=payload.get("arguments", payload.get("args", payload)),
        )
    ]
  return None


def _extract_metrics(
    assistant_tokens: np.ndarray | None,
    logprobs: np.ndarray | None,
) -> trajectory_lib.Metrics | None:
  """Extracts metrics, validating token/logprob lengths."""
  if assistant_tokens is not None and logprobs is not None:
    if len(assistant_tokens) != len(logprobs):
      raise ValueError(
          "Length mismatch: assistant_tokens has length"
          f" {len(assistant_tokens)}, but logprobs has length {len(logprobs)}"
      )

  completion_tokens = None
  completion_token_ids = None
  if assistant_tokens is not None:
    completion_tokens = len(assistant_tokens)
    completion_token_ids = assistant_tokens.tolist()

  logprobs_list = None
  if logprobs is not None:
    logprobs_list = logprobs.tolist()

  if completion_tokens is None and logprobs_list is None:
    return None

  return trajectory_lib.Metrics(
      completion_tokens=completion_tokens,
      completion_token_ids=completion_token_ids,
      logprobs=logprobs_list,
  )


def _extract_observation(obs_val: Any) -> trajectory_lib.Observation | None:
  """Extracts an ATIF Observation container from an observation value."""
  if obs_val is None:
    return None
  if isinstance(obs_val, trajectory_lib.Observation):
    return obs_val
  if isinstance(obs_val, list):
    return trajectory_lib.Observation(
        results=[
            trajectory_lib.ObservationResult(content=str(x)) for x in obs_val
        ]
    )
  return trajectory_lib.Observation(
      results=[trajectory_lib.ObservationResult(content=str(obs_val))]
  )


def tunix_to_atif_step(
    step: agent_types.Step | None,
    step_id: int = 1,
    source: trajectory_lib.Source = trajectory_lib.Source.AGENT,
) -> trajectory_lib.Step | None:
  """Converts a Tunix agent_types.Step into an ATIF Step.

  Args:
    step: The Tunix RL step to convert, or None.
    step_id: 1-based sequential step ID for the resulting ATIF step.
    source: The Source role (AGENT for model turns, SYSTEM/USER for env turns).

  Returns:
    The converted ATIF Step, or None if step is None.

  Raises:
    ValueError: If assistant_tokens and logprobs lengths do not match.
  """
  if step is None:
    return None

  if source == trajectory_lib.Source.AGENT:
    agent_extra = {}
    if step.assistant_tokens is not None:
      agent_extra["assistant_tokens"] = step.assistant_tokens
    if step.assistant_masks is not None:
      agent_extra["assistant_masks"] = step.assistant_masks
    if step.logprobs is not None:
      agent_extra["logprobs"] = step.logprobs
    if step.mc_return:
      agent_extra["mc_return"] = step.mc_return
    if step.info:
      agent_extra.update(step.info)

    return trajectory_lib.Step(
        step_id=step_id,
        source=trajectory_lib.Source.AGENT,
        message=step.model_response,
        reasoning_content=step.thought or None,
        tool_calls=_extract_tool_calls(step.action),
        metrics=_extract_metrics(step.assistant_tokens, step.logprobs),
        extra=agent_extra or None,
    )

  # Non-agent step (SYSTEM or USER)
  env_extra = {
      "reward": step.reward,
      "done": step.done,
  }
  if step.env_tokens is not None:
    env_extra["env_tokens"] = step.env_tokens
  if step.env_masks is not None:
    env_extra["env_masks"] = step.env_masks
  if step.info:
    env_extra.update(step.info)

  obs_str = str(step.observation) if step.observation is not None else ""
  return trajectory_lib.Step(
      step_id=step_id,
      source=source,
      message=obs_str,
      observation=_extract_observation(step.observation),
      extra=env_extra,
  )


def atif_to_tunix_step(
    agent_step: trajectory_lib.Step | None = None,
    env_step: trajectory_lib.Step | None = None,
) -> agent_types.Step:
  """Converts ATIF agent and/or env steps into a single Tunix Step.

  Args:
    agent_step: Optional ATIF Step for the agent turn.
    env_step: Optional ATIF Step for the environment/system turn.

  Returns:
    A reconstructed Tunix agent_types.Step (empty if both inputs are None).
  """
  if agent_step is None and env_step is None:
    return agent_types.Step()

  thought = ""
  model_response = ""
  action = None
  info: dict[str, Any] = {}
  assistant_tokens = None
  assistant_masks = None
  logprobs = None
  mc_return = 0.0

  if agent_step is not None:
    thought = agent_step.reasoning_content or ""
    model_response = agent_step.message or ""
    if agent_step.tool_calls:
      if len(agent_step.tool_calls) == 1:
        tc = agent_step.tool_calls[0]
        action = agent_types.Action(
            action={"name": tc.function_name, "arguments": tc.arguments}
        )
      else:
        action = agent_types.Action(
            action=[
                {"name": tc.function_name, "arguments": tc.arguments}
                for tc in agent_step.tool_calls
            ]
        )
    elif agent_step.extra and "action" in agent_step.extra:
      act = agent_step.extra["action"]
      if isinstance(act, agent_types.Action):
        action = act
      else:
        action = agent_types.Action(action=act)

    if agent_step.extra:
      assistant_tokens = agent_step.extra.get("assistant_tokens")
      assistant_masks = agent_step.extra.get("assistant_masks")
      logprobs = agent_step.extra.get("logprobs")
      mc_return = float(agent_step.extra.get("mc_return", 0.0))
      for k, v in agent_step.extra.items():
        if k not in (
            "assistant_tokens",
            "assistant_masks",
            "logprobs",
            "mc_return",
            "action",
        ):
          info[k] = v

    if agent_step.metrics is not None:
      if (
          assistant_tokens is None
          and agent_step.metrics.completion_token_ids is not None
      ):
        assistant_tokens = np.asarray(agent_step.metrics.completion_token_ids)
      if logprobs is None and agent_step.metrics.logprobs is not None:
        logprobs = np.asarray(agent_step.metrics.logprobs)

  obs_source = env_step if env_step is not None else agent_step
  observation = None
  if obs_source is not None and obs_source.observation is not None:
    if obs_source.observation.results:
      results = obs_source.observation.results
      if len(results) == 1:
        observation = results[0].content
      else:
        observation = [r.content for r in results]
  elif env_step is not None and env_step.message:
    observation = env_step.message

  reward = 0.0
  done = False
  env_tokens = None
  env_masks = None
  target_env = env_step if env_step is not None else agent_step
  if target_env is not None and target_env.extra:
    reward = float(target_env.extra.get("reward", 0.0))
    done = bool(target_env.extra.get("done", False))
    env_tokens = target_env.extra.get("env_tokens")
    env_masks = target_env.extra.get("env_masks")
    for k, v in target_env.extra.items():
      if k not in ("reward", "done", "env_tokens", "env_masks"):
        info[k] = v

  return agent_types.Step(
      model_response=model_response,
      thought=thought,
      action=action,
      observation=observation,
      reward=reward,
      done=done,
      mc_return=mc_return,
      assistant_tokens=assistant_tokens,
      assistant_masks=assistant_masks,
      env_tokens=env_tokens,
      env_masks=env_masks,
      logprobs=logprobs,
      info=info,
  )
