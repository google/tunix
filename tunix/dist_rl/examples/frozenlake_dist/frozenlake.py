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

"""FrozenLake components for the distributed GRPO example.

* The rollout node imports this module (`cluster_config.registry_module`),
  which registers the recipe's `FrozenLakeEnv` and `FrozenLakeAgent`.
* The orchestrator calls `create_dataset(**data_config)` (`data_module`) for
  the task records.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import grain
import numpy as np
from tunix.dist_rl.common import registry
from examples.frozenlake import agent as frozenlake_agent
from examples.frozenlake import data as frozenlake_data
from examples.frozenlake import env as frozenlake_env

FROZENLAKE_ENV_NAME = "frozenlake_env"
FROZENLAKE_AGENT_NAME = "frozenlake_agent"


def generate_entries(
    size: int,
    seed: int,
    *,
    shuffle_seed: int | None = None,
    limit: int | None = None,
    grid_size_range: tuple[int, int] = (2, 9),
) -> list[dict[str, Any]]:
  """Generates the same shuffled map entries as the agentic recipe.

  Each entry configures one `FrozenLakeEnv` map; `create_dataset` wraps them in
  task records.

  Args:
    size: Number of environment configurations to generate. Required so the
      caller (e.g. the `--dataset_size` flag) stays the single source of truth.
    seed: Seed for the environment parameter generator. Required so a run's data
      is never generated from an implicit, hidden default.
    shuffle_seed: If set, shuffles the dataset with this seed, matching
      `datasets.Dataset.shuffle(seed=...)` ordering.
    limit: If set, truncates the dataset to this many entries after shuffling.
    grid_size_range: Lower (inclusive) and upper (exclusive) grid size bounds.

  Returns:
    The list of serializable FrozenLake environment configurations.

  Raises:
    ValueError: If `size` or `limit` is not positive.
  """
  if size <= 0:
    raise ValueError("dataset size must be positive.")
  print(f"Creating dataset with grid size range {grid_size_range}")
  seeds, sizes, probabilities = frozenlake_data.generate_dataset_parameters(
      size, random_seed=seed, grid_size_range=grid_size_range
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


def create_dataset(
    *,
    split: str,
    shuffle: bool,
    seed: int,
    size: int,
    limit: int | None = None,
    grid_size_range: Sequence[int] = (2, 9),
) -> grain.MapDataset[dict[str, Any]]:
  """Returns the FrozenLake task records for the distributed orchestrator.

  The orchestrator calls `create_dataset(**data_config)` (`data_module`). Each
  record has an empty `prompt` and the `metadata.env_config` that the rollout
  worker builds `FrozenLakeEnv` from: the map `entry`, from which the
  environment renders the first observation. The run-wide `max_steps` and
  `is_slippery` come from `env_kwargs`, and the agent settings from
  `agent_kwargs`; records set no `agent_config`, which would replace
  `agent_kwargs`.

  The records keep the previous FrozenLake launcher's fixed order:
  `generate_entries` shuffles the maps once, as
  `datasets.Dataset.shuffle(seed=seed)` does, and the orchestrator repeats them
  in that order, without reshuffling each epoch.

  Args:
    split: Must be "train"; FrozenLake generates a single training set.
    shuffle: Whether to shuffle the generated maps (once, with `seed`).
    seed: Seed for the map parameters and the shuffle.
    size: Number of maps to generate.
    limit: If set, keeps only the first `limit` maps after the shuffle.
    grid_size_range: Lower (inclusive) and upper (exclusive) grid size bounds,
      with `2 <= low < high`: a 1x1 map has no distinct goal cell, so
      `generate_random_map` would never return.

  Returns:
    The `{"prompt", "metadata"}` records.

  Raises:
    ValueError: If `split` is not "train", `grid_size_range` is not two ints
      (not bools) with `2 <= low < high`, or `size` or `limit` is not positive.
  """
  if split != "train":
    raise ValueError(
        f"FrozenLake generates a single 'train' split, got split={split!r}."
    )
  # `bool` is an `int` subclass, so YAML `true` would otherwise pass as 1.
  if (
      len(grid_size_range) != 2
      or not all(
          isinstance(bound, int) and not isinstance(bound, bool)
          for bound in grid_size_range
      )
      or not 2 <= grid_size_range[0] < grid_size_range[1]
  ):
    raise ValueError(
        "grid_size_range must be two ints [low, high) with 2 <= low < high,"
        f" got {grid_size_range!r}."
    )
  low, high = grid_size_range
  entries = generate_entries(
      size,
      seed,
      shuffle_seed=seed if shuffle else None,
      limit=limit,
      grid_size_range=(low, high),
  )
  return grain.MapDataset.source([
      {"prompt": "", "metadata": {"env_config": {"entry": entry}}}
      for entry in entries
  ])


# Both recipe classes already accept the configuration the rollout worker
# builds them from (`env_kwargs` with a record's `env_config`, and
# `agent_kwargs`), so the distributed registry can use them without adapter
# subclasses.
FrozenLakeEnv = registry.register_env(FROZENLAKE_ENV_NAME)(
    frozenlake_env.FrozenLakeEnv
)
FrozenLakeAgent = registry.register_agent(FROZENLAKE_AGENT_NAME)(
    frozenlake_agent.FrozenLakeAgent
)
