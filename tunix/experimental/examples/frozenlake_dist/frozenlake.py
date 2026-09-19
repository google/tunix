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

"""Dataset and registry wiring for the distributed FrozenLake demo."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from examples.frozenlake import agent as frozenlake_agent
from examples.frozenlake import data as frozenlake_data
from examples.frozenlake import env as frozenlake_env
import numpy as np
from tunix.experimental.rl.agentic import registry

FROZENLAKE_ENV_NAME = "frozenlake_env"
FROZENLAKE_AGENT_NAME = "frozenlake_agent"


def _python_scalar(value: Any) -> Any:
  if isinstance(value, np.generic):
    return value.item()
  return value


def create_dataset(
    size: int,
    seed: int,
    *,
    shuffle_seed: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
  """Builds the same generated and shuffled dataset as the agentic recipe.

  Args:
    size: Number of environment configurations to generate. Required so the
      caller (e.g. the `--dataset_size` flag) stays the single source of truth.
    seed: Seed for the environment parameter generator. Required so a run's
      data is never generated from an implicit, hidden default.
    shuffle_seed: If set, shuffles the dataset with this seed, matching
      `datasets.Dataset.shuffle(seed=...)` ordering.
    limit: If set, truncates the dataset to this many entries after shuffling.

  Returns:
    The list of serializable FrozenLake environment configurations.

  Raises:
    ValueError: If `size` or `limit` is not positive.
  """
  if size <= 0:
    raise ValueError("dataset size must be positive.")
  seeds, sizes, probabilities = frozenlake_data.generate_dataset_parameters(
      size, random_seed=seed
  )
  dataset = [
      frozenlake_data.get_frozenlake_dict(
          env_seed, sizes[index], probabilities[index]
      )
      for index, env_seed in enumerate(seeds)
  ]
  if shuffle_seed is not None:
    # Hugging Face Dataset.shuffle(seed=...) uses default_rng(seed).permutation.
    # Reproduce that order without materializing Arrow/Grain in the
    # orchestrator process.
    order = np.random.default_rng(shuffle_seed).permutation(len(dataset))
    dataset = [dataset[index] for index in order]
  if limit is not None:
    if limit <= 0:
      raise ValueError("dataset limit must be positive.")
    dataset = dataset[:limit]
  return dataset


def build_prompt_item(
    *,
    entry: dict[str, Any],
    prompt_idx: int,
    max_turns: int,
    max_response_length: int,
    episode_timeout_secs: int,
    temperature: float,
    top_p: float,
    top_k: int,
    is_slippery: bool,
    use_multistep_prompt: bool,
) -> dict[str, Any]:
  """Builds one serializable distributed rollout request input."""
  normalized_entry = {
      key: _python_scalar(value) for key, value in entry.items()
  }
  prompt_id = f"frozenlake_{prompt_idx}"
  return {
      # The registered environment supplies the first user observation.
      "prompt": "",
      "prompt_id": prompt_id,
      "max_turns": max_turns,
      "max_response_length": max_response_length,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": {
          "episode_timeout": episode_timeout_secs,
          "env_config": {
              "entry": normalized_entry,
              "max_steps": max_turns,
              "is_slippery": is_slippery,
          },
          "agent_config": {
              "use_multistep_prompt": use_multistep_prompt,
          },
      },
  }


def iter_prompt_items(
    *,
    dataset: list[dict[str, Any]],
    max_steps: int,
    batch_size: int,
    max_turns: int,
    max_response_length: int,
    episode_timeout_secs: int,
    temperature: float,
    top_p: float,
    top_k: int,
    is_slippery: bool,
    use_multistep_prompt: bool,
) -> Iterator[dict[str, Any]]:
  """Yields exactly one full-batch worth of prompt groups per RL step."""
  if not dataset:
    raise ValueError("FrozenLake dataset is empty.")
  for prompt_idx in range(max_steps * batch_size):
    yield build_prompt_item(
        entry=dataset[prompt_idx % len(dataset)],
        prompt_idx=prompt_idx,
        max_turns=max_turns,
        max_response_length=max_response_length,
        episode_timeout_secs=episode_timeout_secs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        is_slippery=is_slippery,
        use_multistep_prompt=use_multistep_prompt,
    )


# Both recipe classes already accept the configuration emitted above, so the
# distributed registry can use them without adapter subclasses.
FrozenLakeEnv = registry.register_env(FROZENLAKE_ENV_NAME)(
    frozenlake_env.FrozenLakeEnv
)
FrozenLakeAgent = registry.register_agent(FROZENLAKE_AGENT_NAME)(
    frozenlake_agent.FrozenLakeAgent
)
