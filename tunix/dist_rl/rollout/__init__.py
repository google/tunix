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

from collections.abc import Mapping
import importlib
from typing import Any, Final, TYPE_CHECKING

if TYPE_CHECKING:
  # pylint: disable=g-importing-member, g-import-not-at-top
  from tunix.dist_rl.common.datatypes import RolloutRequest
  from tunix.dist_rl.rollout.collector import TrajectoryCollectorEngine
  from tunix.dist_rl.rollout.inprocess_vllm_sampler import InprocessVllmSampler
  from tunix.dist_rl.rollout.manager import RolloutManager
  from tunix.dist_rl.rollout.sampler import Sampler
  from tunix.dist_rl.rollout.vanilla_sampler import VanillaSampler
  from tunix.dist_rl.rollout.vllm_sampler import VllmSampler
  from tunix.dist_rl.trajectory.trajectory import Trajectory
  from tunix.dist_rl.trajectory.trajectory import TrajectoryError
  # pylint: enable=g-importing-member, g-import-not-at-top

_LAZY_EXPORTS: Final[Mapping[str, tuple[str, str]]] = {
    "RolloutRequest": ("tunix.dist_rl.common.datatypes", "RolloutRequest"),
    "TrajectoryCollectorEngine": (
        "tunix.dist_rl.rollout.collector",
        "TrajectoryCollectorEngine",
    ),
    "InprocessVllmSampler": (
        "tunix.dist_rl.rollout.inprocess_vllm_sampler",
        "InprocessVllmSampler",
    ),
    "RolloutManager": ("tunix.dist_rl.rollout.manager", "RolloutManager"),
    "Sampler": ("tunix.dist_rl.rollout.sampler", "Sampler"),
    "VanillaSampler": (
        "tunix.dist_rl.rollout.vanilla_sampler",
        "VanillaSampler",
    ),
    "VllmSampler": ("tunix.dist_rl.rollout.vllm_sampler", "VllmSampler"),
    "Trajectory": ("tunix.dist_rl.trajectory.trajectory", "Trajectory"),
    "TrajectoryError": (
        "tunix.dist_rl.trajectory.trajectory",
        "TrajectoryError",
    ),
}

__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  if name not in _LAZY_EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_name, attr_name = _LAZY_EXPORTS[name]
  val = getattr(importlib.import_module(module_name), attr_name)
  globals()[name] = val
  return val


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(globals()) | set(__all__))
