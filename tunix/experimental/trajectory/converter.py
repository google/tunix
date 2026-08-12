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

"""Converter for translating between Tunix RL Step and Trajectory Step representations."""

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
      if isinstance(item, dict):
        args = item.get("arguments")
        if args is None:
          args = item.get("args", item)
        if not isinstance(args, dict):
          args = {}
        calls.append(
            trajectory_lib.ToolCall(
                tool_call_id=str(
                    item.get("id", item.get("tool_call_id", f"call_{idx}"))
                ),
                function_name=str(
                    item.get("name", item.get("function_name", "action"))
                ),
                arguments=args,
            )
        )
    return calls or None
  if isinstance(payload, dict):
    args = payload.get("arguments")
    if args is None:
      args = payload.get("args", payload)
    if not isinstance(args, dict):
      args = {}
    return [
        trajectory_lib.ToolCall(
            tool_call_id=str(
                payload.get("id", payload.get("tool_call_id", "call_1"))
            ),
            function_name=str(
                payload.get("name", payload.get("function_name", "action"))
            ),
            arguments=args,
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
  if isinstance(obs_val, list):
    return trajectory_lib.Observation(
        results=[
            trajectory_lib.ObservationResult(content=str(x)) for x in obs_val
        ]
    )
  return trajectory_lib.Observation(
      results=[trajectory_lib.ObservationResult(content=str(obs_val))]
  )


def create_task_step(
    task: Any,
) -> trajectory_lib.Step | None:
  """Creates an initial ATIF user task Step from a task prompt or object.

  Args:
    task: The task prompt string, dictionary with 'prompts', or None.

  Returns:
    An ATIF Step with step_id=0 and source=USER containing the task prompt,
    or None if task is None or empty.
  """
  if task is None:
    return None

  prompt_str = None
  if isinstance(task, str):
    prompt_str = task or None
  elif isinstance(task, dict):
    prompts = task.get("prompts")
    if isinstance(prompts, list) and prompts:
      prompt_str = str(prompts[0])
    elif isinstance(prompts, str):
      prompt_str = prompts or None

  if not prompt_str:
    return None

  return trajectory_lib.Step(
      step_id=0,
      source=trajectory_lib.Source.USER,
      message=prompt_str,
  )


def create_agent_step(
    step: agent_types.Step | None,
    step_id: int = 0,
) -> trajectory_lib.Step | None:
  """Converts a Tunix agent_types.Step into an ATIF agent turn Step.

  Args:
    step: The Tunix RL step to convert, or None.
    step_id: 0-based index of the step.

  Returns:
    The converted Trajectory Step (with 1-based step_id), or None if step is
    None.

  Raises:
    ValueError: If assistant_tokens and logprobs lengths do not match.
  """
  if step is None:
    return None

  trajectory_step_id = step_id + 1  # offset the task step
  return trajectory_lib.Step(
      step_id=trajectory_step_id,
      source=trajectory_lib.Source.AGENT,
      message=step.model_response,
      reasoning_content=step.thought or None,
      tool_calls=_extract_tool_calls(step.action),
      metrics=_extract_metrics(step.assistant_tokens, step.logprobs),
      assistant_tokens=step.assistant_tokens,
      assistant_masks=step.assistant_masks,
      logprobs=step.logprobs,
      mc_return=step.mc_return or None,
      extra=step.info or None,
  )


def create_env_step(
    step: agent_types.Step | None,
    step_id: int = 0,
) -> trajectory_lib.Step | None:
  """Converts a Tunix agent_types.Step into an ATIF environment turn Step.

  Args:
    step: The Tunix RL step to convert, or None.
    step_id: 0-based index of the step.

  Returns:
    The converted Trajectory Step (with 1-based step_id), or None if step is
    None.
  """
  if step is None:
    return None

  trajectory_step_id = step_id + 1  # offset the task step
  obs_str = str(step.observation) if step.observation is not None else ""
  return trajectory_lib.Step(
      step_id=trajectory_step_id,
      source=trajectory_lib.Source.SYSTEM,
      message=obs_str,
      observation=_extract_observation(step.observation),
      reward=step.reward,
      done=step.done,
      env_tokens=step.env_tokens,
      env_masks=step.env_masks,
      extra=step.info or None,
  )


def to_tunix_step(
    agent_step: trajectory_lib.Step | None = None,
    env_step: trajectory_lib.Step | None = None,
) -> agent_types.Step:
  """Converts Trajectory agent and/or env steps into a single Tunix Step.

  Args:
    agent_step: Optional Trajectory Step for the agent turn.
    env_step: Optional Trajectory Step for the environment/system turn.

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
            action={
                "id": tc.tool_call_id,
                "name": tc.function_name,
                "arguments": tc.arguments,
            }
        )
      else:
        action = agent_types.Action(
            action=[
                {
                    "id": tc.tool_call_id,
                    "name": tc.function_name,
                    "arguments": tc.arguments,
                }
                for tc in agent_step.tool_calls
            ]
        )

    assistant_tokens = agent_step.assistant_tokens
    assistant_masks = agent_step.assistant_masks
    logprobs = agent_step.logprobs
    if agent_step.mc_return is not None:
      mc_return = float(agent_step.mc_return)

    if agent_step.metrics is not None:
      if (
          assistant_tokens is None
          and agent_step.metrics.completion_token_ids is not None
      ):
        assistant_tokens = np.asarray(agent_step.metrics.completion_token_ids)
      if logprobs is None and agent_step.metrics.logprobs is not None:
        logprobs = np.asarray(agent_step.metrics.logprobs)

    if agent_step.extra:
      info.update(agent_step.extra)

  observation = None
  reward = 0.0
  done = False
  env_tokens = None
  env_masks = None

  if env_step is not None:
    if env_step.observation is not None and env_step.observation.results:
      results = env_step.observation.results
      if len(results) == 1:
        observation = results[0].content
      else:
        observation = [r.content for r in results]
    elif env_step.message:
      observation = env_step.message

    if env_step.reward is not None:
      reward = float(env_step.reward)
    if env_step.done is not None:
      done = bool(env_step.done)
    env_tokens = env_step.env_tokens
    env_masks = env_step.env_masks
    if env_step.extra:
      info.update(env_step.extra)

  if assistant_tokens is not None and not isinstance(
      assistant_tokens, np.ndarray
  ):
    assistant_tokens = np.asarray(assistant_tokens)
  if assistant_masks is not None and not isinstance(
      assistant_masks, np.ndarray
  ):
    assistant_masks = np.asarray(assistant_masks)
  if env_tokens is not None and not isinstance(env_tokens, np.ndarray):
    env_tokens = np.asarray(env_tokens)
  if env_masks is not None and not isinstance(env_masks, np.ndarray):
    env_masks = np.asarray(env_masks)
  if logprobs is not None and not isinstance(logprobs, np.ndarray):
    logprobs = np.asarray(logprobs)

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
