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

"""Lightweight dataset, environment, and telemetry helpers for data_plane.

Kept free of rollout-node (`automodel`, `maxtext_utils`, `raiden_preload`) and
worker/server imports so importing this module on the stock CPU Orchestrator
(`data_module` and `benchmark_rewards`) or as `cluster_config.registry_module`
never inflates baseline process RSS/HWM. Mock Rollout and Trainer node
entrypoints live in `mock_rollout.py` and `mock_trainer.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
import pathlib
import random
import resource
import sys

import grain
from tunix.dist_rl.common import registry
# Importing test_utils registers mock_agent when loaded as registry_module.
from tunix.dist_rl.common import test_utils as _test_utils
from tunix.rl.agentic.environments import base_environment

del _test_utils

_PROC_STATUS_PATH = pathlib.Path("/proc/self/status")
_MIN_PROMPT_TOKENS = 4


def read_process_memory_mib() -> tuple[float, float]:
  """Returns `(current_rss_mib, peak_hwm_mib)` for the current OS process."""
  if _PROC_STATUS_PATH.is_file():
    rss_kib = 0
    hwm_kib = 0
    for line in _PROC_STATUS_PATH.read_text(encoding="utf-8").splitlines():
      if line.startswith("VmRSS:"):
        rss_kib = int(line.split()[1])
      elif line.startswith("VmHWM:"):
        hwm_kib = int(line.split()[1])
    return rss_kib / 1024.0, hwm_kib / 1024.0
  # macOS `getrusage` reports `ru_maxrss` in bytes; Linux/BSD report KiB.
  ru_divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
  maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  return 0.0, maxrss / ru_divisor


def kwarg_int(kwargs: Mapping[str, object], key: str, default: int) -> int:
  """Extracts an `int` (rejecting `bool`) from `kwargs`, or `default`."""
  if key not in kwargs:
    return default
  val = kwargs[key]
  if isinstance(val, bool) or not isinstance(val, int):
    raise TypeError(f"Expected int for kwargs[{key!r}], got {val!r}.")
  return val


def kwarg_float(
    kwargs: Mapping[str, object], key: str, default: float
) -> float:
  """Extracts a `float` (rejecting `bool`) from `kwargs`, or `default`."""
  if key not in kwargs:
    return default
  val = kwargs[key]
  if isinstance(val, bool) or not isinstance(val, (int, float)):
    raise TypeError(f"Expected float for kwargs[{key!r}], got {val!r}.")
  return float(val)


def create_dataset(
    split: str = "train",
    shuffle: bool = False,
    seed: int = 42,
    num_records: int = 256,
    prompt_tokens: int = 512,
    cache_dir: str = "",
) -> grain.MapDataset[dict[str, object]]:
  """Creates synthetic prompt items for the data-plane benchmark."""
  del cache_dir
  if num_records <= 0 or prompt_tokens < _MIN_PROMPT_TOKENS:
    raise ValueError(
        f"Expected num_records > 0 and prompt_tokens >= {_MIN_PROMPT_TOKENS},"
        f" got num_records={num_records}, prompt_tokens={prompt_tokens}."
    )
  records: list[dict[str, object]] = [
      {
          "prompt": f"{split}-{i}",
          "task_name": "data_plane_benchmark",
          "metadata": {
              "env_config": {
                  "seed": seed + i,
                  "prompt_tokens": prompt_tokens,
              }
          },
      }
      for i in range(num_records)
  ]
  if shuffle:
    random.Random(seed).shuffle(records)
  return grain.MapDataset.source(records)


@registry.register_env("mock_data_plane_env")
class MockDataPlaneEnv(base_environment.BaseTaskEnv):
  """Synthetic environment emitting configurable prompt lengths and rewards."""

  def __init__(
      self,
      seed: int = 0,
      prompt_tokens: int = 512,
      max_steps: int = 1,
      prompt_id: str = "",
      group_index: int = 0,
      num_generations: int = 2,
      policy_version: int = 0,
      **kwargs: object,
  ) -> None:
    del prompt_id, num_generations, kwargs
    self.seed = int(seed)
    self.group_index = int(group_index)
    self.prompt_tokens = int(prompt_tokens)
    if self.prompt_tokens < _MIN_PROMPT_TOKENS:
      raise ValueError(
          f"prompt_tokens must be >= {_MIN_PROMPT_TOKENS}, got"
          f" {self.prompt_tokens}."
      )
    body = " ".join(["tok"] * (self.prompt_tokens - 2))
    prompt_text = f"prompt {self.seed % 10} {body}"
    super().__init__(
        task={"prompts": prompt_text, "policy_version": int(policy_version)},
        max_steps=max(1, int(max_steps)),
    )

  def _initial_observation(self) -> dict[str, str]:
    return {"prompts": str(self.task["prompts"])}

  def _step_impl(self, action: str) -> base_environment.EnvStepResult:
    # `BaseTaskEnv.step()` increments `self.step_count` before `_step_impl`.
    done = "FINAL_ANSWER" in action or self.step_count >= self.max_steps
    # Deterministic non-constant reward across `group_index` ensures non-zero
    # within-group advantage variance so no group is dropped as degenerate.
    reward = float((self.group_index % 4) + 1) / 4.0 if done else 0.0
    obs_text = (
        "bench done" if done else f"step {self.step_count % 10} tok tok"
    )
    return base_environment.EnvStepResult(
        observation={"prompts": obs_text},
        reward=reward,
        done=done,
        info={"steps": self.step_count},
    )


def _validate_routed_num_experts(mode: str, raw_num_experts: object) -> int:
  """Validates `vllm_config.routed_experts_num_experts` against `mode`."""
  if (
      raw_num_experts is None
      or isinstance(raw_num_experts, bool)
      or not isinstance(raw_num_experts, int)
  ):
    raise ValueError(
        "vllm_config.routed_experts_num_experts must be set to a positive int"
        " when vllm_config.return_routed_experts is true; got"
        f" {raw_num_experts!r}."
    )
  if raw_num_experts <= 0:
    raise ValueError(
        f"routed_experts_num_experts must be positive, got {raw_num_experts}."
    )
  if mode == "uint8" and raw_num_experts > 255:
    raise ValueError(
        "routed_experts_encode_mode='uint8' requires"
        f" routed_experts_num_experts <= 255, got {raw_num_experts}."
    )
  if mode == "uint10" and raw_num_experts > 1023:
    raise ValueError(
        "routed_experts_encode_mode='uint10' requires"
        f" routed_experts_num_experts <= 1023, got {raw_num_experts}."
    )
  return raw_num_experts


def resolve_routed_experts_spec(
    vllm_cfg: Mapping[str, object],
    model_kwargs: Mapping[str, object],
) -> tuple[tuple[int, int] | None, int, int | None]:
  """Returns `(shape_spec, num_experts, expected_batched_ndim)` for MoE mocks."""
  if not vllm_cfg["return_routed_experts"]:
    return None, 256, None
  mode = str(vllm_cfg["routed_experts_encode_mode"])
  num_experts = _validate_routed_num_experts(
      mode, vllm_cfg["routed_experts_num_experts"]
  )
  num_layers = kwarg_int(model_kwargs, "mock_num_layers", 61)
  top_k = kwarg_int(model_kwargs, "mock_top_k", 8)
  is_uint10 = mode == "uint10" or (
      mode == "auto" and 256 <= num_experts <= 1023
  )
  return (num_layers, top_k), num_experts, (5 if is_uint10 else 4)
