# Copyright 2025 Google LLC
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

"""Context util for RL agents."""

import json
from typing import Any, Callable, Optional
from tunix.rl.agentic.agents import agent_types

TokenizerFunction = Callable[[str], int]

# Define a flexible type for media costs: maps "key_name" -> token_cost
MediaCostConfig = dict[str, int]

DEFAULT_MEDIA_COSTS: MediaCostConfig = {
    "image": 576,
    "screenshot": 576,
    "visual": 576,
}


def safe_serialize(obj: Any) -> str:
  """Reliably converts arbitrary objects to a string for token counting."""
  if isinstance(obj, str):
    return obj
  try:
    return json.dumps(obj, default=str, sort_keys=True, ensure_ascii=False)
  except TypeError:
    return str(obj)


def count_multimodal_observation(
    obs: Any,
    tokenizer: TokenizerFunction,
    media_costs: Optional[MediaCostConfig] = None,
) -> int:
  """Counts tokens for an observation using a configurable media cost map.

  Args:
      obs: The observation object (usually a dict).
      tokenizer: Function to count text tokens.
      media_costs: Dictionary mapping keys (e.g., 'screenshot') to token costs.
        If None, uses DEFAULT_MEDIA_COSTS.
  """
  costs = media_costs if media_costs is not None else DEFAULT_MEDIA_COSTS

  if isinstance(obs, dict):
    token_count = 0
    text_content = {}

    for key, value in obs.items():
      lower_key = key.lower()
      if lower_key in costs and value is not None:
        token_count += costs[lower_key]
      else:
        text_content[key] = value

    text_str = safe_serialize(text_content)
    token_count += tokenizer(text_str)
    return token_count

  return tokenizer(safe_serialize(obs))


def count_step_tokens(
    step: agent_types.Step,
    tokenizer: TokenizerFunction,
    media_costs: Optional[MediaCostConfig] = None,
) -> int:
  """Estimates size of a step, passing down media configuration."""
  total = 0

  if step.thought:
    total += tokenizer(step.thought)

  if step.model_response and step.model_response != step.thought:
    total += tokenizer(step.model_response)

  if step.action:
    action_str = safe_serialize(step.action.action)
    total += tokenizer(action_str)

  if step.observation:
    total += count_multimodal_observation(
        step.observation, tokenizer, media_costs
    )

  return total


def calculate_trajectory_tokens(
    traj: agent_types.Trajectory,
    tokenizer: TokenizerFunction,
    media_costs: Optional[MediaCostConfig] = None,
) -> int:
  """Calculates full trajectory tokens with custom media costs."""
  total = 0
  if traj.task:
    total += tokenizer(safe_serialize(traj.task))

  for step in traj.steps:
    total += count_step_tokens(step, tokenizer, media_costs)

  return total
