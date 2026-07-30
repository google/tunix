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

"""Worker abstractions for role-based isolation.

Defines the base interface for all RL pipeline workers. The Worker is the unit
that the Orchestrator talks to, and is a wrapper around the core logic of the
pipeline (e.g. TrainerWorker is a wrapper around the Trainer).
"""

import abc
from typing import Any

from tunix.experimental.common import datatypes


class Worker(abc.ABC):
  """Base interface for all Workers."""

  # Default initial state for a worker
  _state: datatypes.WorkerState = datatypes.WorkerState.PENDING

  @property
  def state(self) -> datatypes.WorkerState:
    """Gets the current lifecycle state of the worker."""
    return self._state

  @state.setter
  def state(self, new_state: datatypes.WorkerState) -> None:
    """Sets a new state, validating the transition."""
    if self._state == new_state:
      return
    if self._state and not self._state.can_transition_to(new_state):
      current_name = (
          self._state.name if hasattr(self._state, "name") else self._state
      )
      target_name = new_state.name if hasattr(new_state, "name") else new_state
      raise RuntimeError(
          f"Invalid transition from {current_name} to {target_name}"
      )
    self._state = new_state

  @abc.abstractmethod
  def initialize(self) -> datatypes.Response:
    """Initializes the worker.

    Allocates memory, loads model weights, and sets up mesh/sharding
    constraints, etc.
    """
    pass

  @abc.abstractmethod
  def compile(self, dummy_data: Any) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    pass

  @abc.abstractmethod
  def start(self) -> datatypes.Response:
    """Starts the worker's main loop."""
    pass

  @abc.abstractmethod
  def stop(self) -> datatypes.Response:
    """Gracefully stops the worker."""
    pass

  @abc.abstractmethod
  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's identification and roles."""
    pass

  @abc.abstractmethod
  def heartbeat(self) -> datatypes.HealthReport:
    """Returns the current health status of the worker."""
    pass
