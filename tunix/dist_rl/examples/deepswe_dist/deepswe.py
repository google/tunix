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

"""DeepSWE components for the distributed GRPO example.

* The rollout node imports this module (`cluster_config.registry_module`),
  which registers `DeepSWEEnv` and `DeepSWEAgent`.
* The orchestrator calls `create_dataset(**data_config)` (`data_module`) for
  the task records.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import contextlib
import json
import logging
import math
import os
from typing import Any

import grain
import numpy as np
from tunix.dist_rl.common import registry
from tunix.dist_rl.orchestrator import dataset_utils
from examples.deepswe import deepswe_data
from examples.deepswe import sandbox_utils
from examples.deepswe import swe_agent
from examples.deepswe import swe_env

DEEPSWE_ENV_NAME = "deepswe_env"
DEEPSWE_AGENT_NAME = "deepswe_agent"
DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"


def normalize_example_value(value: Any) -> Any:
  """Converts dataset scalar wrappers into plain Python values."""
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return normalize_example_value(flat[0])
    return [normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, list):
    return [normalize_example_value(v) for v in value]
  return value


def as_text(value: Any) -> str:
  value = normalize_example_value(value)
  return value if isinstance(value, str) else str(value)


def _jsonify_lists(entry: Mapping[str, object]) -> dict[str, object]:
  """Matches the legacy DeepSWE recipe's heterogeneous dataset normalization."""
  normalized: dict[str, object] = {}
  for key, value in entry.items():
    value = normalize_example_value(value)
    normalized[key] = json.dumps(value) if isinstance(value, list) else value
  return normalized


def load_deepswe_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = "train",
    dataset_path: str = "",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Any:
  """Loads the R2E-Gym dataset used by the DeepSWE recipe."""
  if dataset_path:
    from datasets import DatasetDict  # pylint: disable=g-import-not-at-top
    from datasets import load_from_disk  # pylint: disable=g-import-not-at-top

    logging.info("Loading DeepSWE dataset from disk: %s", dataset_path)
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
      dataset = dataset[dataset_split]
    dataset = dataset.map(_jsonify_lists, keep_in_memory=True)
    if shuffle:
      dataset = dataset.shuffle(seed=seed)
    return dataset

  logging.info(
      "Loading DeepSWE dataset %s split=%s cache_dir=%s shuffle=%s seed=%d.",
      dataset_name,
      dataset_split,
      cache_dir,
      shuffle,
      seed,
  )
  return deepswe_data.create_dataset(
      dataset_name=dataset_name,
      dataset_split=dataset_split,
      cache_dir=cache_dir,
      shuffle=shuffle,
      seed=seed,
  )


def _validate_str(
    value: object, name: str, *, allow_empty: bool = False
) -> str:
  """Validates a string option, rejecting non-str and whitespace-only values."""
  if not isinstance(value, str):
    raise TypeError(f"{name} must be a str, got {type(value).__name__}.")
  if not value.strip() and (not allow_empty or value):
    raise ValueError(f"{name} must be a non-empty string.")
  return value


def _validate_int(value: object, name: str, *, positive: bool = False) -> int:
  """Validates an integer option, rejecting booleans."""
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
  if positive and value <= 0:
    raise ValueError(f"{name} must be positive, got {value}.")
  return value


def _problem_statement(entry: Mapping[str, object]) -> str:
  for key in ("problem_statement", "prompt", "instance_id"):
    if key in entry and entry[key] is not None:
      return as_text(entry[key])
  return ""


def _to_task_record(index: int, raw_entry: object) -> dict[str, object]:
  """Maps a DeepSWE dataset entry to an orchestrator task record."""
  if not isinstance(raw_entry, Mapping):
    raise TypeError(
        "DeepSWE dataset item must be a mapping, got"
        f" {type(raw_entry).__name__}."
    )
  entry = _jsonify_lists(raw_entry)
  problem = _problem_statement(entry)
  raw_id = entry["instance_id"] if "instance_id" in entry else None
  instance_id = as_text(raw_id).strip() if raw_id is not None else ""
  prompt_id = instance_id or f"deepswe_{index}"
  return {
      "prompt": problem,
      "metadata": {
          "instance_id": prompt_id,
          "problem_statement": problem,
          "env_config": {
              "entry": entry,
              "prompt_id": prompt_id,
          },
      },
  }


def create_dataset(
    *,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_path: str = "",
    cache_dir: str | None = None,
    limit: int | None = None,
) -> grain.MapDataset[dict[str, object]]:
  """Returns the DeepSWE task records for the distributed orchestrator.

  The orchestrator calls `create_dataset(**data_config)` (`data_module`). Each
  record holds `prompt` (the problem statement) and `metadata` with
  `instance_id`, `problem_statement`, and `env_config` (`entry` and `prompt_id`)
  for `DeepSWEEnv`. Run-wide environment and agent settings come from
  `env_kwargs` and `agent_kwargs`; records omit `agent_config`, which would
  replace `agent_kwargs` in `RolloutManager`.

  Args:
    split: Dataset split name (for example, `"train"`).
    shuffle: Whether to shuffle the dataset once with `seed` on load.
    seed: Random seed for shuffling.
    dataset_name: HuggingFace dataset identifier used when `dataset_path` is
      empty.
    dataset_path: Optional local HuggingFace `save_to_disk` directory path.
    cache_dir: Optional local cache directory for HuggingFace downloads. Empty
      string `""` is normalized to `None`.
    limit: Optional positive cap on the number of records returned after
      optional shuffling.

  Returns:
    A `grain.MapDataset` of `{"prompt", "metadata"}` task records.

  Raises:
    TypeError: If any argument has an invalid type (including `bool` for `seed`
      or `limit`).
    ValueError: If `split` is empty, both `dataset_path` and `dataset_name` are
      empty, any string option is whitespace-only, `limit` is not positive, or
      the loaded dataset is empty.
  """
  _validate_str(split, "split")
  if not isinstance(shuffle, bool):
    raise TypeError(f"shuffle must be a bool, got {type(shuffle).__name__}.")
  _validate_int(seed, "seed")
  _validate_str(dataset_name, "dataset_name", allow_empty=True)
  _validate_str(dataset_path, "dataset_path", allow_empty=True)
  if not dataset_path and not dataset_name:
    raise ValueError("Either dataset_path or dataset_name must be non-empty.")
  effective_cache_dir: str | None = None
  if cache_dir is not None:
    effective_cache_dir = (
        _validate_str(cache_dir, "cache_dir", allow_empty=True) or None
    )
  if limit is not None:
    _validate_int(limit, "limit", positive=True)

  dataset = load_deepswe_dataset(
      dataset_name=dataset_name,
      dataset_split=split,
      dataset_path=dataset_path,
      cache_dir=effective_cache_dir,
      shuffle=shuffle,
      seed=seed,
  )
  raw: grain.MapDataset[object] = (
      dataset
      if isinstance(dataset, grain.MapDataset)
      else grain.MapDataset.source(dataset)
  )
  if len(raw) <= 0:
    raise ValueError("DeepSWE dataset is empty.")
  if limit is not None:
    raw = raw[:limit]
  logging.info(
      "Loaded DeepSWE dataset: source=%s split=%s size=%d.",
      dataset_path or dataset_name,
      split,
      len(raw),
  )
  return raw.map_with_index(_to_task_record)


def _entry_at(dataset: Any, index: int) -> dict[str, Any]:
  entry = dataset[index]
  if not isinstance(entry, dict):
    raise TypeError(f"DeepSWE dataset item must be a dict, got {type(entry)}")
  return _jsonify_lists(dict(entry))


# TODO(noghabi): Remove build_prompt_item and iter_prompt_items with
# run_deepswe_dist.py.
def build_prompt_item(
    *,
    entry: dict[str, Any],
    prompt_idx: int,
    max_turns: int,
    max_response_length: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    step_timeout_secs: int,
    reward_timeout_secs: int,
    env_backend: str,
    use_agent_sandbox: bool,
    scaffold: str,
    env_verbose: bool,
    overlong_filter: bool = False,
    episode_timeout_secs: int | None = None,
) -> dict[str, Any]:
  """Builds one StandardRLProgram prompt item for a DeepSWE task."""
  problem = _problem_statement(entry)
  prompt_id = as_text(entry.get("instance_id") or f"deepswe_{prompt_idx}")
  env_config = {
      "entry": entry,
      "prompt_id": prompt_id,
      "max_steps": max_turns,
      "step_timeout": step_timeout_secs,
      "reward_timeout": reward_timeout_secs,
      "backend": env_backend,
      "use_agent_sandbox": use_agent_sandbox,
      "scaffold": scaffold,
      "verbose": env_verbose,
  }
  agent_config = {"scaffold": scaffold}
  metadata: dict[str, Any] = {
      "instance_id": prompt_id,
      "problem_statement": problem,
      "env_config": env_config,
      "agent_config": agent_config,
      # Read by TrajectoryCollectorEngine to decide which terminal statuses
      # get their policy loss mask zeroed out.
      "overlong_filter": overlong_filter,
  }
  # When `episode_timeout_secs` is None, `"episode_timeout"` is omitted from
  # `metadata` so `TrajectoryCollectorEngine` uses its default
  # (`timeouts.DEFAULT_EPISODE_TIMEOUT_S`).
  if episode_timeout_secs is not None:
    metadata["episode_timeout"] = episode_timeout_secs
  return {
      "prompt": problem,
      "prompt_id": prompt_id,
      "task_name": "deepswe",
      "max_turns": max_turns,
      "max_response_length": max_response_length,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": metadata,
  }


def iter_prompt_items(
    *,
    dataset: Any,
    max_steps: int,
    batch_size: int,
    max_turns: int,
    max_response_length: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    step_timeout_secs: int,
    reward_timeout_secs: int,
    env_backend: str,
    use_agent_sandbox: bool,
    scaffold: str,
    env_verbose: bool,
    overlong_filter: bool = False,
    max_staleness: int = 0,
    episode_timeout_secs: int | None = None,
) -> Iterator[dict[str, Any]]:
  """Yields one full-batch worth of prompt groups per RL step plus off-policy lookahead."""
  dataset_size = len(dataset)
  if dataset_size <= 0:
    raise ValueError("DeepSWE dataset is empty.")

  total_prompts = dataset_utils.total_prompt_groups(
      max_steps, batch_size, max_staleness
  )
  for prompt_idx in range(total_prompts):
    yield build_prompt_item(
        entry=_entry_at(dataset, prompt_idx % dataset_size),
        prompt_idx=prompt_idx,
        max_turns=max_turns,
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        step_timeout_secs=step_timeout_secs,
        reward_timeout_secs=reward_timeout_secs,
        episode_timeout_secs=episode_timeout_secs,
        env_backend=env_backend,
        use_agent_sandbox=use_agent_sandbox,
        scaffold=scaffold,
        env_verbose=env_verbose,
        overlong_filter=overlong_filter,
    )


@contextlib.contextmanager
def prewarm_sandboxes(
    prompt_items: grain.MapDataset[dict[str, object]],
    *,
    batch_size: int,
    num_generations: int,
    max_staleness: int,
    env_kwargs: Mapping[str, object],
    max_concurrency: int,
    max_warmpool_replicas: int,
) -> Iterator[Iterable[dict[str, object]]]:
  """Warms job-scoped `SandboxFleet` pools ahead of dispatch when enabled."""
  _validate_int(batch_size, "batch_size", positive=True)
  use_agent_sandbox = env_kwargs["use_agent_sandbox"]
  if not isinstance(use_agent_sandbox, bool):
    raise TypeError(
        "env_kwargs['use_agent_sandbox'] must be a bool, got"
        f" {type(use_agent_sandbox).__name__}."
    )
  if not use_agent_sandbox:
    yield prompt_items
    return

  scaffold = _validate_str(env_kwargs["scaffold"], "env_kwargs.scaffold")
  fleet = sandbox_utils.init_global_fleet(
      tasks=None,
      max_concurrency=max_concurrency,
      num_generations=num_generations,
      batch_size=batch_size,
      max_warmpool_replicas=max_warmpool_replicas,
      scaffold=scaffold,
      job_id=os.environ["ORCHESTRATOR_ID"],
      warm_on_init=False,
      teardown_on_exit=False,
  )
  try:
    with contextlib.closing(
        sandbox_utils.PrewarmDatasetIterator(
            prompt_items,
            fleet=fleet,
            num_generations=num_generations,
            batch_size=batch_size,
            max_warmpool_replicas=max_warmpool_replicas,
            max_staleness=max_staleness + math.ceil(2 / batch_size),
            unwarm_on_exhaustion=True,
            scaffold=scaffold,
        )
    ) as prewarm:
      yield prewarm
  finally:
    sandbox_utils.teardown_global_fleet()


@registry.register_env(DEEPSWE_ENV_NAME)
class DeepSWEEnv(swe_env.SWEEnv):
  """Registry adapter that lets RolloutWorker construct SWEEnv per request."""

  def __init__(
      self,
      entry: dict[str, Any] | None = None,
      prompt_id: str = "",
      group_index: int = 0,
      num_generations: int = 1,
      policy_version: int = 0,
      group_id: Any = None,
      pair_index: int | None = None,
      max_warmpool_replicas: int | None = None,
      scaffold: str = "r2egym",
      use_agent_sandbox: bool = False,
      fleet: Any | None = None,
      **kwargs: Any,
  ):
    entry = dict(entry or kwargs.pop("task", {}) or {})
    if prompt_id and "instance_id" not in entry:
      entry["instance_id"] = prompt_id
    if group_id is None:
      group_id = prompt_id or None
    if pair_index is None:
      pair_index = group_index
    if max_warmpool_replicas is None:
      max_warmpool_replicas = num_generations
    if use_agent_sandbox and fleet is None:
      raw_job_id = (
          os.environ["ORCHESTRATOR_ID"].strip()
          if "ORCHESTRATOR_ID" in os.environ
          else ""
      )
      job_id = raw_job_id or None
      fleet = sandbox_utils.init_global_fleet(
          tasks=[entry] if job_id else None,
          max_concurrency=num_generations,
          num_generations=num_generations,
          max_warmpool_replicas=max_warmpool_replicas,
          scaffold=scaffold,
          job_id=job_id,
          teardown_on_exit=job_id is None,
      )

    super().__init__(
        entry=entry,
        group_id=group_id,
        pair_index=pair_index,
        scaffold=scaffold,
        use_agent_sandbox=use_agent_sandbox,
        fleet=fleet,
        **kwargs,
    )
    self.task = {
        **entry,
        "prompt_id": prompt_id,
        "group_index": group_index,
        "num_generations": num_generations,
        "policy_version": policy_version,
    }


@registry.register_agent(DEEPSWE_AGENT_NAME)
class DeepSWEAgent(swe_agent.SWEAgent):
  """Registry adapter for the legacy DeepSWE XML-tool agent."""

  name = DEEPSWE_AGENT_NAME
