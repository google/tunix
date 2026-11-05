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

"""Worker subsystem for Distributed-RL (RolloutWorker, TrainerWorker, etc.)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# pylint: disable=g-importing-member
from tunix.dist_rl.worker.abstract_worker import Worker
from tunix.dist_rl.worker.remote_execution import ActorHandle
from tunix.dist_rl.worker.remote_execution import ExecutionRequest
from tunix.dist_rl.worker.remote_execution import ExecutionResponse
from tunix.dist_rl.worker.remote_execution import InProcessActorHandle
from tunix.dist_rl.worker.remote_execution import InProcessRemoteExecutionServer
from tunix.dist_rl.worker.remote_execution import RemoteExecutionServer
from tunix.dist_rl.worker.task_session import DefaultRouter
from tunix.dist_rl.worker.task_session import RouteFn
from tunix.dist_rl.worker.task_session import stable_route_hash
from tunix.dist_rl.worker.task_session import TaskSession

if TYPE_CHECKING:
  from tunix.dist_rl.worker.rollout_worker import RolloutWorker
  from tunix.dist_rl.worker.trainer_worker import TrainerWorker

_LAZY_WORKER_EXPORTS: dict[str, tuple[str, str]] = {
    "RolloutWorker": ("tunix.dist_rl.worker.rollout_worker", "RolloutWorker"),
    "TrainerWorker": ("tunix.dist_rl.worker.trainer_worker", "TrainerWorker"),
}

__all__ = [
    "ActorHandle",
    "DefaultRouter",
    "ExecutionRequest",
    "ExecutionResponse",
    "InProcessActorHandle",
    "InProcessRemoteExecutionServer",
    "RemoteExecutionServer",
    "RolloutWorker",
    "RouteFn",
    "TaskSession",
    "TrainerWorker",
    "Worker",
    "stable_route_hash",
]


def __getattr__(  # pylint: disable=invalid-name
    name: str,
) -> type[RolloutWorker | TrainerWorker]:
  if name not in _LAZY_WORKER_EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_path, attr_name = _LAZY_WORKER_EXPORTS[name]
  cls = getattr(importlib.import_module(module_path), attr_name)
  globals()[name] = cls
  return cls


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(globals()) | set(_LAZY_WORKER_EXPORTS))
