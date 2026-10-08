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

"""Token-mode training dict and its Trajectory Store record.

`build_token_dict` is the single definition of how a finished
`agent_types.Trajectory` is flattened into the dict the agentic learners train
on. `TrajectoryCollectEngine` calls it when collecting in "Token" mode, and the
sub-batch checkpoint restore path calls it again on a trajectory read back from
the Trajectory Store, so the two can never drift apart.

`to_store_record` / `write_trajectory` persist every per-turn input of
`build_token_dict` exactly once; `from_store_record` reverses them. Rebuilding
is cheap (array concatenation), so the flattened dict itself is never stored.
"""

from collections.abc import Hashable
import dataclasses
from typing import Any, Final

import numpy as np
from tunix.experimental.trajectory import converter
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types

# Bump whenever `build_token_dict` or the record layout changes in a way that
# makes previously written records rebuild to a different dict. Records with a
# different version are rejected instead of silently rebuilt.
FLATTEN_VERSION: Final[int] = 1

# Key under `TunixTrajectoryMetadata.extra` holding the flatten context.
TOKEN_DICT_EXTRA_KEY: Final[str] = "tunix_token_dict"

# Key under which `build_token_dict` reports the store trajectory id.
TRAJECTORY_ID_KEY: Final[str] = "trajectory_id"

_FLATTEN_VERSION_KEY: Final[str] = "flatten_version"
_GROUP_ID_KEY: Final[str] = "group_id"
_MASKED_OUT_KEY: Final[str] = "masked_out"
_EXACT_TOKEN_CONTINUITY_KEY: Final[str] = "exact_token_continuity"

_RECORD_AGENT: Final[trajectory_lib.Agent] = trajectory_lib.Agent(
    name="tunix_trajectory_collect_engine", version=str(FLATTEN_VERSION)
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenDictContext:
  """Inputs of `build_token_dict` that live outside `agent_types.Trajectory`.

  Attributes:
    chat_completions: The agent's full chat history (OpenAI Chat API format).
    policy_version: Policy version the trajectory was sampled with.
    group_id: The GRPO group the trajectory belongs to.
    masked_out: Whether overlong filtering zeroes the conversation masks.
    exact_token_continuity: Whether the prompt was recorded left-padded with an
      explicit `prompt_length`.
  """

  chat_completions: list[dict[str, Any]]
  policy_version: int | None
  group_id: Hashable | None
  masked_out: bool
  exact_token_continuity: bool


def build_token_dict(
    trajectory: agent_types.Trajectory,
    context: TokenDictContext,
    *,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
  """Flattens all steps of `trajectory` into a single training dict.

  Args:
    trajectory: A finished trajectory; `env_time` and `reward_time` must already
      hold the episode's timings.
    context: Flatten inputs not stored on `trajectory`.
    trajectory_id: The Trajectory Store id of `trajectory`, reported under
      `TRAJECTORY_ID_KEY` when not None.

  Returns:
    The Token-mode dict consumed by the agentic learners.

  Raises:
    ValueError: If routed experts are present but missing or misaligned for any
      tokens.
  """
  conversation_tokens, conversation_masks, logprobs = [], [], []
  routed_experts = []
  prompt_tokens = trajectory.prompt_tokens
  prompt_routed = trajectory.prompt_routed_experts
  has_routed_experts = prompt_routed is not None or any(
      step.assistant_routed_experts is not None
      or step.env_routed_experts is not None
      for step in trajectory.steps
  )

  for idx, step in enumerate(trajectory.steps):
    # Keep tokens/masks/logprobs/routed_experts appended in lockstep.
    assistant_tokens = step.assistant_tokens
    env_tokens = step.env_tokens
    step_logprobs = step.logprobs
    step_routed = step.assistant_routed_experts
    step_env_routed = step.env_routed_experts
    if assistant_tokens is not None:
      conversation_tokens.append(assistant_tokens)
      conversation_masks.append(step.assistant_masks)
      if step_logprobs is not None:
        assert len(step_logprobs) == len(assistant_tokens), (
            f"Logprobs length {len(step_logprobs)} does not match assistant"
            f" tokens length {len(assistant_tokens)}"
        )
        logprobs.append(step_logprobs)
      else:
        logprobs.append(np.zeros(len(assistant_tokens)))
      if has_routed_experts:
        if step_routed is None:
          raise ValueError(
              f"Step {idx} has assistant_tokens (len"
              f" {len(assistant_tokens)}) but missing"
              " assistant_routed_experts while routed_experts is active."
          )
        if len(step_routed) != len(assistant_tokens):
          raise ValueError(
              f"Step {idx} assistant_routed_experts length"
              f" {len(step_routed)} does not match assistant_tokens length"
              f" {len(assistant_tokens)}."
          )
        routed_experts.append(np.asarray(step_routed, dtype=np.int16))
    if env_tokens is not None:
      conversation_tokens.append(env_tokens)
      conversation_masks.append(step.env_masks)
      logprobs.append(np.zeros(len(env_tokens)))
      if has_routed_experts:
        if step_env_routed is None:
          raise ValueError(
              f"Step {idx} has env_tokens (len {len(env_tokens)}) but"
              " missing env_routed_experts while routed_experts is active."
          )
        if len(step_env_routed) != len(env_tokens):
          raise ValueError(
              f"Step {idx} env_routed_experts length"
              f" {len(step_env_routed)} does not match env_tokens length"
              f" {len(env_tokens)}."
          )
        routed_experts.append(np.asarray(step_env_routed, dtype=np.int16))

  conversation_tokens = [
      np.asarray(tokens) for tokens in conversation_tokens if len(tokens) > 0
  ]
  conversation_masks = [
      np.asarray(masks) for masks in conversation_masks if len(masks) > 0
  ]
  logprobs = [
      np.asarray(step_logprobs)
      for step_logprobs in logprobs
      if len(step_logprobs) > 0
  ]
  conversation_masks = (
      np.concatenate(conversation_masks, axis=0)
      if conversation_masks
      else np.array([], dtype=np.int32)
  )
  conversation_tokens = (
      np.concatenate(conversation_tokens, axis=0)
      if conversation_tokens
      else np.array([], dtype=np.int32)
  )
  final_masks = (
      np.zeros_like(conversation_masks)
      if context.masked_out
      else conversation_masks
  )

  final_routed_experts = None
  if has_routed_experts:
    sample_arr = next(iter(routed_experts), prompt_routed)
    sample_shape = sample_arr.shape[1:] if sample_arr is not None else (0, 0)
    conv_routed = (
        np.concatenate(routed_experts, axis=0)
        if routed_experts
        else np.zeros((0,) + sample_shape, dtype=np.int16)
    )
    prompt_len = (
        (trajectory.prompt_length or 0)
        if context.exact_token_continuity
        else (len(prompt_tokens) if prompt_tokens is not None else 0)
    )
    if prompt_len > 0:
      if prompt_routed is None:
        raise ValueError(
            f"Trajectory has prompt_tokens (len {prompt_len}) but missing"
            " prompt_routed_experts while routed_experts is active."
        )
      if prompt_routed.shape[0] != prompt_len:
        raise ValueError(
            f"prompt_routed_experts shape {prompt_routed.shape} does not"
            f" match prompt_tokens length {prompt_len}."
        )
      prompt_routed_arr = np.asarray(prompt_routed, dtype=np.int16)
    else:
      prompt_routed_arr = np.zeros((0,) + sample_shape, dtype=np.int16)

    final_routed_experts = np.concatenate(
        [prompt_routed_arr, conv_routed], axis=0
    )

  result = {
      "conversation_text": context.chat_completions,
      "prompt_tokens": prompt_tokens,
      "conversation_tokens": conversation_tokens,
      "conversation_masks": final_masks,
      "status": trajectory.status.name,
      "trajectory_reward": trajectory.reward,
      "env_time": trajectory.env_time,
      "reward_time": trajectory.reward_time,
      "old_logprobs": (np.concatenate(logprobs, axis=0) if logprobs else None),
      "routed_experts": final_routed_experts,
      "policy_version": context.policy_version,
      "original_input": trajectory.task,
      "group_id": context.group_id,
  }
  if trajectory.prompt_length is not None:
    # Set only by exact token continuity; lets training unpad by length.
    result["prompt_length"] = trajectory.prompt_length
  if trajectory_id is not None:
    result[TRAJECTORY_ID_KEY] = trajectory_id
  return result


def to_store_record(
    trajectory: agent_types.Trajectory,
    context: TokenDictContext,
    *,
    trajectory_id: str,
) -> tuple[
    trajectory_lib.TunixTrajectoryMetadata,
    list[trajectory_lib.TunixAgentStep | trajectory_lib.TunixEnvStep],
]:
  """Converts a finished trajectory into Trajectory Store metadata and steps.

  Step 0 is the task prompt; turn i becomes an agent step at 2i+1 and an
  environment step at 2i+2, matching `converter.to_tunix_trajectory`.

  Args:
    trajectory: A finished trajectory, as passed to `build_token_dict`.
    context: The flatten context, as passed to `build_token_dict`.
    trajectory_id: The Trajectory Store id to write under.

  Returns:
    The trajectory metadata and its 0-indexed steps.

  Raises:
    TypeError: If the task is not a dict or the group id is not JSON-native.
  """
  if not isinstance(trajectory.task, dict):
    raise TypeError(
        "Only dict tasks can be written to the Trajectory Store, got"
        f" {type(trajectory.task).__name__}."
    )
  group_id = context.group_id
  if isinstance(group_id, np.integer):
    group_id = int(group_id)
  if group_id is not None and not isinstance(group_id, (int, str)):
    raise TypeError(
        "group_id must be an int or str to be written to the Trajectory"
        f" Store, got {type(group_id).__name__}."
    )
  task_step = converter.create_task_step(trajectory.task)
  if task_step is None:
    # Step 0 is always the prompt slot, so turn i lands at 2i+1 / 2i+2.
    task_step = trajectory_lib.TunixEnvStep(
        step_id=0, source=trajectory_lib.Source.USER, message=""
    )
  steps: list[trajectory_lib.TunixAgentStep | trajectory_lib.TunixEnvStep] = [
      task_step
  ]
  for turn, step in enumerate(trajectory.steps):
    agent_step = converter.create_agent_step(step, turn)
    env_step = converter.create_env_step(step, turn)
    assert agent_step is not None and env_step is not None
    steps.append(agent_step)
    steps.append(env_step)

  metadata = trajectory_lib.TunixTrajectoryMetadata(
      trajectory_id=trajectory_id,
      agent=_RECORD_AGENT,
      status=trajectory.status.name,
      total_reward=trajectory.reward,
      env_time=trajectory.env_time,
      reward_time=trajectory.reward_time,
      prompt_tokens=trajectory.prompt_tokens,
      prompt_length=trajectory.prompt_length,
      prompt_routed_experts=trajectory.prompt_routed_experts,
      policy_version=context.policy_version,
      task=trajectory.task,
      chat_completions=context.chat_completions,
      extra={
          TOKEN_DICT_EXTRA_KEY: {
              _FLATTEN_VERSION_KEY: FLATTEN_VERSION,
              _GROUP_ID_KEY: group_id,
              _MASKED_OUT_KEY: context.masked_out,
              _EXACT_TOKEN_CONTINUITY_KEY: context.exact_token_continuity,
          }
      },
  )
  return metadata, steps


def write_trajectory(
    store: store_lib.TrajectoryWriter,
    trajectory: agent_types.Trajectory,
    context: TokenDictContext,
    *,
    trajectory_id: str,
) -> None:
  """Writes a finished trajectory to `store` under `trajectory_id`.

  Steps are logged against a header-only metadata, and the full metadata is
  written once at the end, so that writers that snapshot the metadata with
  every step (the file store deep copies it) copy the prompt payload once.

  Args:
    store: The Trajectory Store to write to.
    trajectory: A finished trajectory, as passed to `build_token_dict`.
    context: The flatten context, as passed to `build_token_dict`.
    trajectory_id: The Trajectory Store id to write under.
  """
  metadata, steps = to_store_record(
      trajectory, context, trajectory_id=trajectory_id
  )
  header = trajectory_lib.TunixTrajectoryMetadata(
      trajectory_id=trajectory_id, agent=metadata.agent
  )
  for step in steps:
    store.add_step(step, header)
  store.update_metadata(metadata)


def from_store_record(
    record: trajectory_lib.Trajectory,
) -> tuple[agent_types.Trajectory, TokenDictContext]:
  """Reverses `to_store_record` for a trajectory read from a store.

  Args:
    record: A trajectory as returned by `TrajectoryReader.get_trajectories`
      (1-indexed ATIF or 0-indexed Tunix).

  Returns:
    The trajectory and flatten context to pass to `build_token_dict`.

  Raises:
    ValueError: If `record` was not written by `write_trajectory`, or was
      written by an incompatible `FLATTEN_VERSION`.
  """
  tunix_record = trajectory_lib.TunixTrajectory.from_atif_trajectory(record)
  if tunix_record.extra is None or TOKEN_DICT_EXTRA_KEY not in tunix_record.extra:
    raise ValueError(
        f"Trajectory {tunix_record.trajectory_id!r} has no"
        f" {TOKEN_DICT_EXTRA_KEY!r} metadata; it was not written by"
        " write_trajectory."
    )
  flatten_info = tunix_record.extra[TOKEN_DICT_EXTRA_KEY]
  if flatten_info[_FLATTEN_VERSION_KEY] != FLATTEN_VERSION:
    raise ValueError(
        f"Trajectory {tunix_record.trajectory_id!r} was written with flatten"
        f" version {flatten_info[_FLATTEN_VERSION_KEY]}, but this binary"
        f" rebuilds version {FLATTEN_VERSION}."
    )
  if tunix_record.chat_completions is None:
    raise ValueError(
        f"Trajectory {tunix_record.trajectory_id!r} has no chat_completions."
    )
  trajectory = converter.to_tunix_trajectory(tunix_record)
  context = TokenDictContext(
      chat_completions=tunix_record.chat_completions,
      policy_version=tunix_record.policy_version,
      group_id=flatten_info[_GROUP_ID_KEY],
      masked_out=flatten_info[_MASKED_OUT_KEY],
      exact_token_continuity=flatten_info[_EXACT_TOKEN_CONTINUITY_KEY],
  )
  return trajectory, context
