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

"""Distributed-RL Rollout Subsystem."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# pylint: disable=g-importing-member
from tunix.dist_rl.common.datatypes import RolloutRequest
from tunix.dist_rl.common.datatypes import WeightSyncMetadata
from tunix.dist_rl.rollout.sampler import Sampler
from tunix.dist_rl.trajectory.trajectory import Trajectory
from tunix.dist_rl.trajectory.trajectory import TrajectoryError

if TYPE_CHECKING:
  from tunix.dist_rl.rollout.collector import TrajectoryCollectorEngine
  from tunix.dist_rl.rollout.inprocess_vllm_sampler import InprocessVllmSampler
  from tunix.dist_rl.rollout.manager import RolloutManager
  from tunix.dist_rl.rollout.vanilla_sampler import VanillaSampler
  from tunix.dist_rl.rollout.vllm_sampler import VllmSampler

_LAZY_ROLLOUT_EXPORTS: dict[str, tuple[str, str]] = {
    "InprocessVllmSampler": (
        "tunix.dist_rl.rollout.inprocess_vllm_sampler",
        "InprocessVllmSampler",
    ),
    "RolloutManager": ("tunix.dist_rl.rollout.manager", "RolloutManager"),
    "TrajectoryCollectorEngine": (
        "tunix.dist_rl.rollout.collector",
        "TrajectoryCollectorEngine",
    ),
    "VanillaSampler": (
        "tunix.dist_rl.rollout.vanilla_sampler",
        "VanillaSampler",
    ),
    "VllmSampler": ("tunix.dist_rl.rollout.vllm_sampler", "VllmSampler"),
}

__all__ = [
    "InprocessVllmSampler",
    "RolloutManager",
    "RolloutRequest",
    "Sampler",
    "Trajectory",
    "TrajectoryCollectorEngine",
    "TrajectoryError",
    "VanillaSampler",
    "VllmSampler",
    "WeightSyncMetadata",
]


def __getattr__(  # pylint: disable=invalid-name
    name: str,
) -> type[
    InprocessVllmSampler
    | RolloutManager
    | TrajectoryCollectorEngine
    | VanillaSampler
    | VllmSampler
]:
  if name not in _LAZY_ROLLOUT_EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_path, attr_name = _LAZY_ROLLOUT_EXPORTS[name]
  cls = getattr(importlib.import_module(module_path), attr_name)
  globals()[name] = cls
  return cls


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(globals()) | set(_LAZY_ROLLOUT_EXPORTS))
