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
from tunix.experimental.trajectory import action_converter
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


def _extract_metrics(
    assistant_tokens: Any,
    logprobs: Any,
) -> trajectory_lib.Metrics | None:
  """Builds a Metrics object from assistant_tokens and logprobs if present."""
  completion_tokens = None
  completion_token_ids = None
  logprobs_list = None

  if assistant_tokens is not None and isinstance(
      assistant_tokens, (list, tuple, np.ndarray)
  ):
    completion_tokens = len(assistant_tokens)
    completion_token_ids = [int(t) for t in assistant_tokens]

  if logprobs is not None and isinstance(logprobs, (list, tuple, np.ndarray)):
    logprobs_list = [float(lp) for lp in logprobs]

  if (
      completion_token_ids is not None
      and logprobs_list is not None
      and len(completion_token_ids) != len(logprobs_list)
  ):
    raise ValueError(
        f"Length of assistant_tokens ({len(completion_token_ids)}) must match"
        f" length of logprobs ({len(logprobs_list)})."
    )

  if completion_tokens is None and logprobs_list is None:
    return None

  return trajectory_lib.Metrics(
      completion_tokens=completion_tokens,
      completion_token_ids=completion_token_ids,
      logprobs=logprobs_list,
  )


def _extract_observation(obs_val: Any) -> trajectory_lib.Observation | None:
  """Extracts an Observation container from an observation value."""
  if obs_val is None:
    return None
  items = obs_val if isinstance(obs_val, list) else [obs_val]
  return trajectory_lib.Observation(
      results=[trajectory_lib.ObservationResult(content=str(x)) for x in items]
  )


def create_task_step(
    task: Any,
) -> trajectory_lib.TunixEnvStep | None:
  """Creates an initial user task Step from a task prompt or object.

  Args:
    task: The task prompt string, dictionary with 'prompts', or None.

  Returns:
    A TunixEnvStep with step_id=0 and source=USER containing the task prompt,
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

  return trajectory_lib.TunixEnvStep(
      step_id=0,
      source=trajectory_lib.Source.USER,
      message=prompt_str,
  )


def create_agent_step(
    step: agent_types.Step | None,
    tunix_step_id: int,
    policy_version: int | None = None,
) -> trajectory_lib.TunixAgentStep | None:
  """Converts a Tunix agent_types.Step into an agent turn Step.

  Maps the 0-based Tunix interaction turn index (`tunix_step_id`) to the
  converted step ID: `converted_step_id = 2 * tunix_step_id + 1`.
  (e.g., Tunix turn 0 -> step 1, Tunix turn 1 -> step 3).

  Args:
    step: The Tunix RL step to convert, or None.
    tunix_step_id: 0-based turn index of the step in the Tunix trajectory.
    policy_version: Optional policy version override for the agent step.

  Returns:
    The converted TunixAgentStep (with step_id = 2 * tunix_step_id + 1), or None
    if step is None.

  Raises:
    ValueError: If assistant_tokens and logprobs lengths do not match.
  """
  if step is None:
    return None

  extra = dict(step.info) if step.info else {}
  if step.action is not None:
    raw_action = (
        step.action.action
        if isinstance(step.action, agent_types.Action)
        else step.action
    )
    if raw_action is not None:
      extra["raw_action"] = raw_action

  effective_policy_version = (
      policy_version
      if policy_version is not None
      else getattr(step, "policy_version", None)
  )
  if effective_policy_version is None and "policy_version" in extra:
    effective_policy_version = extra["policy_version"]

  converted_step_id = 2 * tunix_step_id + 1
  return trajectory_lib.TunixAgentStep(
      step_id=converted_step_id,
      source=trajectory_lib.Source.AGENT,
      message=step.model_response,
      reasoning_content=(
          step.thought if step.thought is not None else None
      ),
      tool_calls=action_converter.extract_tool_calls(step.action),
      metrics=_extract_metrics(step.assistant_tokens, step.logprobs),
      assistant_tokens=step.assistant_tokens,
      assistant_masks=step.assistant_masks,
      logprobs=step.logprobs,
      policy_version=effective_policy_version,
      mc_return=(
          float(step.mc_return) if step.mc_return is not None else None
      ),
      extra=extra or None,
  )


def create_env_step(
    step: agent_types.Step | None,
    tunix_step_id: int,
) -> trajectory_lib.TunixEnvStep | None:
  """Converts a Tunix agent_types.Step into an environment turn Step.

  Maps the 0-based Tunix interaction turn index (`tunix_step_id`) to the
  converted step ID: `converted_step_id = 2 * tunix_step_id + 2`.
  (e.g., Tunix turn 0 -> step 2, Tunix turn 1 -> step 4).

  Args:
    step: The Tunix RL step to convert, or None.
    tunix_step_id: 0-based turn index of the step in the Tunix trajectory.

  Returns:
    The converted TunixEnvStep (with step_id = 2 * tunix_step_id + 2), or None
    if step is None.
  """
  if step is None:
    return None

  obs_str = str(step.observation) if step.observation is not None else ""
  converted_step_id = 2 * tunix_step_id + 2
  return trajectory_lib.TunixEnvStep(
      step_id=converted_step_id,
      source=trajectory_lib.Source.SYSTEM,
      message=obs_str,
      observation=_extract_observation(step.observation),
      reward=step.reward,
      done=step.done,
      env_tokens=step.env_tokens,
      env_masks=step.env_masks,
      extra=step.info or None,
  )


def _filter_extra_info(extra: dict[str, Any] | None) -> dict[str, Any]:
  """Filters action keys out of step extra metadata."""
  if not extra:
    return {}
  return {k: v for k, v in extra.items() if k not in ("raw_action", "action")}


def _to_numpy_or_none(arr: Any) -> np.ndarray | None:
  """Converts an array-like object to a numpy array if not None."""
  return np.asarray(arr) if arr is not None else None


def to_tunix_step(
    agent_step: (
        trajectory_lib.TunixAgentStep | trajectory_lib.Step | None
    ) = None,
    env_step: trajectory_lib.TunixEnvStep | trajectory_lib.Step | None = None,
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

    # Restore action
    if agent_step.extra and "raw_action" in agent_step.extra:
      action = agent_types.Action(action=agent_step.extra["raw_action"])
    elif agent_step.extra and "action" in agent_step.extra:
      action = agent_types.Action(action=agent_step.extra["action"])
    elif agent_step.tool_calls:
      calls = [
          {
              "id": tc.tool_call_id,
              "name": tc.function_name,
              "arguments": tc.arguments,
          }
          for tc in agent_step.tool_calls
      ]
      action = agent_types.Action(action=calls[0] if len(calls) == 1 else calls)

    assistant_tokens = getattr(agent_step, "assistant_tokens", None)
    assistant_masks = getattr(agent_step, "assistant_masks", None)
    logprobs = getattr(agent_step, "logprobs", None)
    mc_return_val = getattr(agent_step, "mc_return", None)
    if mc_return_val is not None:
      mc_return = float(mc_return_val)

    if agent_step.metrics is not None:
      if (
          assistant_tokens is None
          and agent_step.metrics.completion_token_ids is not None
      ):
        assistant_tokens = np.asarray(agent_step.metrics.completion_token_ids)
      if logprobs is None and agent_step.metrics.logprobs is not None:
        logprobs = np.asarray(agent_step.metrics.logprobs)

    info.update(_filter_extra_info(agent_step.extra))
    policy_version = getattr(agent_step, "policy_version", None)
    if policy_version is not None:
      info["policy_version"] = policy_version

  observation = None
  reward = 0.0
  done = False
  env_tokens = None
  env_masks = None

  if env_step is not None:
    if env_step.observation is not None and env_step.observation.results:
      results = env_step.observation.results
      observation = (
          results[0].content
          if len(results) == 1
          else [r.content for r in results]
      )
    elif env_step.message:
      observation = env_step.message

    reward_val = getattr(env_step, "reward", None)
    if reward_val is not None:
      reward = float(reward_val)
    done_val = getattr(env_step, "done", None)
    if done_val is not None:
      done = bool(done_val)
    env_tokens = getattr(env_step, "env_tokens", None)
    env_masks = getattr(env_step, "env_masks", None)
    info.update(_filter_extra_info(env_step.extra))

  return agent_types.Step(
      model_response=model_response,
      thought=thought,
      action=action,
      observation=observation,
      reward=reward,
      done=done,
      mc_return=mc_return,
      assistant_tokens=_to_numpy_or_none(assistant_tokens),
      assistant_masks=_to_numpy_or_none(assistant_masks),
      env_tokens=_to_numpy_or_none(env_tokens),
      env_masks=_to_numpy_or_none(env_masks),
      logprobs=_to_numpy_or_none(logprobs),
      info=info,
  )


def to_tunix_trajectory(
    traj: (
        trajectory_lib.TunixTrajectory
        | trajectory_lib.Trajectory
        | dict[str, Any]
    ),
) -> agent_types.Trajectory:
  """Converts a Trajectory into a Tunix agent_types.Trajectory.

  Reconstructs Tunix RL interaction steps from sequential converted steps:
  - Step 0 (source USER or SYSTEM) -> Tunix task prompt dictionary.
  - Subsequent steps (Agent step at 2i+1, Env step at 2i+2) -> Paired into
    Tunix `agent_types.Step` instances for turn i.

  Args:
    traj: Trajectory or dictionary representation.

  Returns:
    A reconstructed Tunix agent_types.Trajectory instance.
  """
  if isinstance(traj, dict):
    if (
        "prompt_id" in traj
        or "status" in traj
        or "total_reward" in traj
        or traj.get("schema_version", "").startswith("Tunix")
        or any(
            step.get("step_id") == 0
            or "mc_return" in step
            or "reward" in step
            for step in traj.get("steps", [])
            if isinstance(step, dict)
        )
    ):
      traj_obj = trajectory_lib.TunixTrajectory.from_json_dict(traj)
    else:
      traj_obj = trajectory_lib.Trajectory.from_json_dict(traj)
  else:
    traj_obj = traj

  dto_steps: list[agent_types.Step] = []
  converted_step_idx = 0
  num_converted_steps = len(traj_obj.steps)

  task_val = None
  # If step 0 is the initial prompt (source USER or SYSTEM), extract it.
  if num_converted_steps > 0 and traj_obj.steps[0].source in (
      trajectory_lib.Source.USER,
      trajectory_lib.Source.SYSTEM,
  ):
    task_val = {"prompts": [traj_obj.steps[0].message]}
    converted_step_idx = 1

  # Iterate through steps and pair AGENT + ENV steps into Tunix turns.
  while converted_step_idx < num_converted_steps:
    curr_step = traj_obj.steps[converted_step_idx]
    if curr_step.source == trajectory_lib.Source.AGENT:
      next_step: trajectory_lib.Step | None = None
      if (
          converted_step_idx + 1 < num_converted_steps
          and traj_obj.steps[converted_step_idx + 1].source
          != trajectory_lib.Source.AGENT
      ):
        next_step = traj_obj.steps[converted_step_idx + 1]
      dto_step = to_tunix_step(agent_step=curr_step, env_step=next_step)
      dto_steps.append(dto_step)
      converted_step_idx += 2 if next_step is not None else 1
    else:
      dto_step = to_tunix_step(agent_step=None, env_step=curr_step)
      dto_steps.append(dto_step)
      converted_step_idx += 1

  total_reward = getattr(traj_obj, "total_reward", None)
  reward = float(total_reward) if total_reward is not None else 0.0

  status_enum = agent_types.TrajectoryStatus.RUNNING
  traj_obj_status = getattr(traj_obj, "status", None)
  if traj_obj_status is not None and hasattr(
      agent_types.TrajectoryStatus, str(traj_obj_status)
  ):
    status_enum = getattr(agent_types.TrajectoryStatus, str(traj_obj_status))

  env_time = getattr(traj_obj, "env_time", None) or {}
  reward_time = getattr(traj_obj, "reward_time", None) or {}

  return agent_types.Trajectory(
      task=task_val,
      steps=dto_steps,
      reward=reward,
      status=status_enum,
      env_time=env_time,
      reward_time=reward_time,
  )


def create_trajectory_metadata(
    traj_id: str,
    request: Any = None,
    agent: Any = None,
    target_policy_versions: list[int] | None = None,
    status: (
        str | agent_types.TrajectoryStatus
    ) = agent_types.TrajectoryStatus.RUNNING,
    extra: dict[str, Any] | None = None,
) -> trajectory_lib.TunixTrajectoryMetadata:
  """Constructs TunixTrajectoryMetadata from rollout request and agent state."""
  meta_extra = dict(getattr(request, "metadata", None) or {})
  if extra:
    meta_extra.update(extra)

  if isinstance(agent, trajectory_lib.Agent):
    agent_obj = agent
  else:
    agent_obj = trajectory_lib.Agent(
        name=getattr(agent, "name", "agent"),
        version=getattr(agent, "version", "1.0"),
    )
  traj_obj = getattr(agent, "trajectory", None)

  effective_status = status
  if effective_status is None:
    effective_status = getattr(
        traj_obj, "status", agent_types.TrajectoryStatus.RUNNING
    )

  status_str = (
      effective_status.name
      if hasattr(effective_status, "name")
      else str(effective_status)
  )

  return trajectory_lib.TunixTrajectoryMetadata(
      trajectory_id=traj_id,
      agent=agent_obj,
      prompt_id=getattr(request, "prompt_id", None),
      group_index=getattr(request, "group_index", 0),
      target_policy_versions=target_policy_versions,
      status=status_str,
      total_reward=getattr(traj_obj, "reward", None),
      hyperparams=getattr(request, "generation_kwargs", None),
      env_time=getattr(traj_obj, "env_time", None),
      reward_time=getattr(traj_obj, "reward_time", None),
      extra=meta_extra or None,
  )


def update_trajectory_metadata(
    metadata: (
        trajectory_lib.TunixTrajectoryMetadata
        | trajectory_lib.TrajectoryMetadata
    ),
    agent: Any = None,
    policy_version: int | None = None,
    target_policy_versions: list[int] | None = None,
    status: str | agent_types.TrajectoryStatus | None = None,
    env_time: dict[str, Any] | None = None,
    reward_time: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> trajectory_lib.TrajectoryMetadata:
  """Updates an existing TrajectoryMetadata with latest agent/trajectory state.

  Args:
    metadata: The TrajectoryMetadata instance to update in place.
    agent: Optional agent containing trajectory status, reward, and timings.
    policy_version: Optional policy version to append to target_policy_versions.
    target_policy_versions: Optional explicit list of target policy versions.
    status: Optional explicit status override.
    env_time: Optional explicit environment timing dictionary.
    reward_time: Optional explicit reward timing dictionary.
    extra: Optional dictionary of extra metadata fields to merge.

  Returns:
    The updated TrajectoryMetadata instance.
  """
  traj_obj = getattr(agent, "trajectory", None)
  if status is not None:
    status_str = status.name if hasattr(status, "name") else str(status)
  elif traj_obj is not None and getattr(traj_obj, "status", None) is not None:
    traj_status = traj_obj.status
    status_str = (
        traj_status.name if hasattr(traj_status, "name") else str(traj_status)
    )
  else:
    status_str = None

  if status_str is not None:
    if hasattr(metadata, "status"):
      setattr(metadata, "status", status_str)
    elif metadata.extra is not None:
      metadata.extra["status"] = status_str

  if target_policy_versions is not None:
    if hasattr(metadata, "target_policy_versions"):
      setattr(metadata, "target_policy_versions", list(target_policy_versions))
    elif metadata.extra is not None:
      metadata.extra["target_policy_versions"] = list(target_policy_versions)
  elif policy_version is not None:
    if hasattr(metadata, "target_policy_versions"):
      current_versions = getattr(metadata, "target_policy_versions", None)
      if current_versions is None:
        setattr(metadata, "target_policy_versions", [policy_version])
      elif policy_version not in current_versions:
        current_versions.append(policy_version)
    elif metadata.extra is not None:
      existing = metadata.extra.setdefault("target_policy_versions", [])
      if policy_version not in existing:
        existing.append(policy_version)

  if traj_obj is not None:
    if hasattr(metadata, "total_reward"):
      setattr(metadata, "total_reward", getattr(traj_obj, "reward", None))
    effective_env_time = (
        env_time
        if env_time is not None
        else getattr(traj_obj, "env_time", None)
    )
    if hasattr(metadata, "env_time") and effective_env_time is not None:
      setattr(metadata, "env_time", effective_env_time)
    effective_reward_time = (
        reward_time
        if reward_time is not None
        else getattr(traj_obj, "reward_time", None)
    )
    if hasattr(metadata, "reward_time") and effective_reward_time is not None:
      setattr(metadata, "reward_time", effective_reward_time)
  else:
    if hasattr(metadata, "env_time") and env_time is not None:
      setattr(metadata, "env_time", env_time)
    if hasattr(metadata, "reward_time") and reward_time is not None:
      setattr(metadata, "reward_time", reward_time)

  if extra:
    if metadata.extra is None:
      metadata.extra = {}
    metadata.extra.update(extra)

  return metadata
