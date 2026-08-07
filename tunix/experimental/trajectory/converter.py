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

"""Converter for translating between Tunix RL and ATIF representations."""

from typing import Any
import numpy as np
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
  """Extracts an attribute or dictionary key safely without type narrowing."""
  if obj is None:
    return default
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


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


def to_atif_metadata(
    traj_id: str,
    request: Any,
    agent: Any,
    status: str = "RUNNING",
) -> trajectory_lib.TrajectoryMetadata:
  """Constructs ATIF TrajectoryMetadata from rollout request and agent state.

  Args:
    traj_id: Unique identifier for the trajectory.
    request: RolloutRequest or request-like object containing generation info.
    agent: Agent instance or agent-like object executing the rollout.
    status: Current status string of the trajectory (default: "RUNNING").

  Returns:
    An ATIF TrajectoryMetadata instance.
  """
  meta_extra = {
      "prompt_id": _get_field(request, "prompt_id", ""),
      "group_id": _get_field(request, "group_id", ""),
      "target_policy_version": _get_field(request, "target_policy_version", 0),
  }

  req_meta = _get_field(request, "metadata", None)
  if isinstance(req_meta, dict):
    meta_extra.update(req_meta)
  meta_extra["status"] = status

  gen_kwargs = _get_field(request, "generation_kwargs", None)
  if isinstance(gen_kwargs, dict):
    meta_extra["hyperparams"] = gen_kwargs

  traj_obj = _get_field(agent, "trajectory", None)
  if traj_obj is not None:
    total_reward = _get_field(traj_obj, "reward", None)
    if total_reward is not None:
      meta_extra["total_reward"] = total_reward

  agent_name = _get_field(agent, "name", "agent")

  return trajectory_lib.TrajectoryMetadata(
      trajectory_id=traj_id,
      agent=trajectory_lib.Agent(
          name=str(agent_name),
          version="1.0",
      ),
      extra=meta_extra,
  )


def to_atif_trajectory(
    traj: (
        agent_types.Trajectory
        | agent_types.TrajectoryItem
        | dict[str, Any]
        | None
    ),
    traj_id: str = "traj_1",
    request: Any = None,
    agent: Any = None,
    status: str = "RUNNING",
    metadata: dict[str, Any] | None = None,
) -> trajectory_lib.Trajectory:
  """Converts a Tunix Trajectory or TrajectoryItem into an ATIF Trajectory.

  Args:
    traj: Tunix Trajectory, TrajectoryItem, or dictionary representation.
    traj_id: Canonical trajectory identifier (default: "traj_1").
    request: Optional RolloutRequest or generation request object.
    agent: Optional Agent instance or agent-like object executing rollout.
    status: Trajectory status string (default: "RUNNING").
    metadata: Optional extra metadata dictionary.

  Returns:
    An ATIF Trajectory instance containing converted steps and metadata.
  """
  meta_dict: dict[str, Any] = {}
  if metadata:
    meta_dict.update(metadata)

  raw_traj = traj
  if isinstance(traj, agent_types.TrajectoryItem):
    if traj.metadata:
      meta_dict.update(traj.metadata)
    raw_traj = traj.traj

  steps_data = []
  if isinstance(raw_traj, agent_types.Trajectory):
    steps_data = raw_traj.steps
    if raw_traj.task is not None:
      meta_dict["task"] = raw_traj.task
    meta_dict["total_reward"] = float(raw_traj.reward)
    meta_dict["status"] = (
        raw_traj.status.name
        if hasattr(raw_traj.status, "name")
        else str(raw_traj.status)
    )
    if raw_traj.env_time:
      meta_dict["env_time"] = raw_traj.env_time
    if raw_traj.reward_time:
      meta_dict["reward_time"] = raw_traj.reward_time
  elif isinstance(raw_traj, dict):
    steps_data = raw_traj.get("steps", [])
    if "task" in raw_traj:
      meta_dict["task"] = raw_traj["task"]
    if "reward" in raw_traj:
      meta_dict["total_reward"] = float(raw_traj["reward"])
    if "status" in raw_traj:
      meta_dict["status"] = str(raw_traj["status"])
    if "env_time" in raw_traj:
      meta_dict["env_time"] = raw_traj["env_time"]
    if "reward_time" in raw_traj:
      meta_dict["reward_time"] = raw_traj["reward_time"]

  if request is not None or agent is not None:
    meta_obj = to_atif_metadata(
        traj_id=traj_id,
        request=request or {},
        agent=agent or {},
        status=status,
    )
    if meta_dict:
      if meta_obj.extra is None:
        meta_obj.extra = {}
      meta_obj.extra.update(meta_dict)
  else:
    meta_extra = dict(meta_dict)
    meta_extra["status"] = meta_extra.get("status", status)
    agent_name = _get_field(agent, "name", "agent")
    model_name = _get_field(agent, "model_name", None)
    agent_obj = trajectory_lib.Agent(
        name=str(agent_name),
        version="1.0",
        model_name=model_name,
    )
    meta_obj = trajectory_lib.TrajectoryMetadata(
        trajectory_id=traj_id,
        agent=agent_obj,
        extra=meta_extra or None,
    )

  converted_steps: list[trajectory_lib.Step] = []
  step_idx = 1
  for s in steps_data:
    step_obj = s
    if isinstance(s, dict):
      step_obj = agent_types.Step(**s)

    agent_turn = tunix_to_atif_step(
        step_obj, step_id=step_idx, source=trajectory_lib.Source.AGENT
    )
    if agent_turn is not None:
      converted_steps.append(agent_turn)
      step_idx += 1

    has_env = (
        step_obj.observation is not None
        or step_obj.reward != 0.0
        or step_obj.done
        or step_obj.env_tokens is not None
    )
    if has_env:
      env_turn = tunix_to_atif_step(
          step_obj, step_id=step_idx, source=trajectory_lib.Source.SYSTEM
      )
      if env_turn is not None:
        converted_steps.append(env_turn)
        step_idx += 1

  return trajectory_lib.Trajectory(
      trajectory_id=meta_obj.trajectory_id,
      session_id=meta_obj.session_id,
      schema_version=meta_obj.schema_version,
      agent=meta_obj.agent,
      notes=meta_obj.notes,
      final_metrics=meta_obj.final_metrics,
      continued_trajectory_ref=meta_obj.continued_trajectory_ref,
      extra=meta_obj.extra,
      steps=converted_steps,
  )


def atif_to_tunix_trajectory(
    atif_traj: trajectory_lib.Trajectory | dict[str, Any],
) -> agent_types.Trajectory:
  """Converts an ATIF Trajectory into a Tunix agent_types.Trajectory.

  Args:
    atif_traj: ATIF Trajectory or dictionary representation.

  Returns:
    A reconstructed Tunix agent_types.Trajectory instance.
  """
  if isinstance(atif_traj, dict):
    atif_traj = trajectory_lib.Trajectory.from_json_dict(atif_traj)

  dto_steps: list[agent_types.Step] = []
  step_idx = 0
  num_steps = len(atif_traj.steps)
  while step_idx < num_steps:
    curr_step = atif_traj.steps[step_idx]
    if curr_step.source == trajectory_lib.Source.AGENT:
      next_step = None
      if (
          step_idx + 1 < num_steps
          and atif_traj.steps[step_idx + 1].source
          != trajectory_lib.Source.AGENT
      ):
        next_step = atif_traj.steps[step_idx + 1]
      dto_step = atif_to_tunix_step(agent_step=curr_step, env_step=next_step)
      dto_steps.append(dto_step)
      if next_step is not None:
        step_idx += 2
      else:
        step_idx += 1
    else:
      dto_step = atif_to_tunix_step(agent_step=None, env_step=curr_step)
      dto_steps.append(dto_step)
      step_idx += 1

  reward = 0.0
  status_enum = agent_types.TrajectoryStatus.RUNNING
  task = None
  env_time: dict[str, float] = {}
  reward_time: dict[str, float] = {}

  if atif_traj.extra:
    if "total_reward" in atif_traj.extra:
      reward = float(atif_traj.extra["total_reward"])
    elif "reward" in atif_traj.extra:
      reward = float(atif_traj.extra["reward"])
    if "task" in atif_traj.extra:
      task = atif_traj.extra["task"]
    if "status" in atif_traj.extra:
      st = str(atif_traj.extra["status"])
      if hasattr(agent_types.TrajectoryStatus, st):
        status_enum = getattr(agent_types.TrajectoryStatus, st)
    if "env_time" in atif_traj.extra and isinstance(
        atif_traj.extra["env_time"], dict
    ):
      env_time = atif_traj.extra["env_time"]
    if "reward_time" in atif_traj.extra and isinstance(
        atif_traj.extra["reward_time"], dict
    ):
      reward_time = atif_traj.extra["reward_time"]

  return agent_types.Trajectory(
      task=task,
      steps=dto_steps,
      reward=reward,
      status=status_enum,
      env_time=env_time,
      reward_time=reward_time,
  )
