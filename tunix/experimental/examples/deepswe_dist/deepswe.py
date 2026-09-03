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

"""DeepSWE components for the experimental distributed GRPO example."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import os
from typing import Any

import numpy as np
from tunix.experimental.rl.agentic import registry

from examples.deepswe import swe_env


DEEPSWE_ENV_NAME = "deepswe_env"
DEEPSWE_AGENT_NAME = "deepswe_agent"
DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-V1"


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


def _jsonify_lists(entry: dict[str, Any]) -> dict[str, Any]:
  """Matches the legacy DeepSWE recipe's heterogeneous dataset normalization."""
  normalized = {}
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

  from examples.deepswe import deepswe_data  # pylint: disable=g-import-not-at-top

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


def _entry_at(dataset: Any, index: int) -> dict[str, Any]:
  entry = dataset[index]
  if not isinstance(entry, dict):
    raise TypeError(f"DeepSWE dataset item must be a dict, got {type(entry)}")
  return _jsonify_lists(dict(entry))


def _problem_statement(entry: dict[str, Any]) -> str:
  for key in ("problem_statement", "prompt", "instance_id"):
    if key in entry and entry[key] is not None:
      return as_text(entry[key])
  return ""


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
  return {
      "prompt": problem,
      "prompt_id": prompt_id,
      "max_turns": max_turns,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": {
          "instance_id": prompt_id,
          "problem_statement": problem,
          "prefix_hash": prompt_id,
          "env_config": env_config,
          "agent_config": agent_config,
      },
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
) -> Iterator[dict[str, Any]]:
  """Yields exactly the prompt groups needed for the requested training run."""
  dataset_size = len(dataset)
  if dataset_size <= 0:
    raise ValueError("DeepSWE dataset is empty.")

  for prompt_idx in range(max_steps * batch_size):
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
        env_backend=env_backend,
        use_agent_sandbox=use_agent_sandbox,
        scaffold=scaffold,
        env_verbose=env_verbose,
  )


def _env_bool(name: str, default: bool) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
  value = os.getenv(name)
  if value is None or value == "":
    return default
  return int(value)


def _sandbox_tasks_from_env() -> list[dict[str, Any]]:
  dataset = load_deepswe_dataset(
      dataset_name=os.getenv("DATASET_NAME", DEFAULT_DATASET_NAME),
      dataset_split=os.getenv("DATASET_SPLIT", "train"),
      dataset_path=os.getenv("DATASET_PATH", ""),
      cache_dir=os.getenv("DATASET_CACHE_DIR") or None,
      shuffle=_env_bool("SHUFFLE", True),
      seed=_env_int("SEED", 42),
  )
  return [_entry_at(dataset, i) for i in range(len(dataset))]


def _init_sandbox_fleet_from_env(
    entry: dict[str, Any], group_size: int
) -> Any:
  """Initializes DeepSWE's process-wide SandboxFleet from rollout metadata."""
  max_concurrency = _env_int(
      "SANDBOX_MAX_CONCURRENCY",
      _env_int("ROLLOUT_MAX_CONCURRENCY", group_size),
  )
  try:
    tasks = _sandbox_tasks_from_env()
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Failed to load the full DeepSWE dataset for SandboxFleet. Falling "
        "back to the current task only."
    )
    tasks = [entry]
  logging.info(
      "Initializing DeepSWE SandboxFleet in rollout worker with %d task(s) "
      "(max_concurrency=%d).",
      len(tasks),
      max_concurrency,
  )
  return swe_env._init_global_fleet(  # pylint: disable=protected-access
      tasks=tasks,
      max_concurrency=max_concurrency,
  )


@registry.register_env(DEEPSWE_ENV_NAME)
class DeepSWEEnv(swe_env.SWEEnv):
  """Registry adapter that lets RolloutWorker construct SWEEnv per request."""

  def __init__(
      self,
      entry: dict[str, Any] | None = None,
      prompt_id: str = "",
      group_index: int = 0,
      group_size: int = 1,
      policy_version: int = 0,
      group_id: Any = None,
      pair_index: int | None = None,
      **kwargs: Any,
  ):
    entry = dict(entry or kwargs.pop("task", {}) or {})
    if prompt_id and "instance_id" not in entry:
      entry["instance_id"] = prompt_id
    if group_id is None:
      group_id = prompt_id or None
    if pair_index is None:
      pair_index = group_index
    if kwargs.get("use_agent_sandbox") and kwargs.get("fleet") is None:
      kwargs["fleet"] = _init_sandbox_fleet_from_env(entry, group_size)

    super().__init__(
        entry=entry,
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )
    self.task = {
        **entry,
        "prompt_id": prompt_id,
        "group_index": group_index,
        "group_size": group_size,
        "policy_version": policy_version,
    }


@registry.register_agent(DEEPSWE_AGENT_NAME)
class DeepSWEAgent:
  """Lazy registry adapter for the legacy DeepSWE XML-tool agent."""

  name = DEEPSWE_AGENT_NAME

  def __init__(self, **kwargs: Any):
    from examples.deepswe import swe_agent  # pylint: disable=g-import-not-at-top

    self._agent = swe_agent.SWEAgent(**kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._agent, name)
