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

from tunix.experimental.common import datatypes


class Worker(abc.ABC):
  """Base interface for all Workers."""

  @abc.abstractmethod
  def initialize(self) -> None:
    """Initializes the worker.

    Allocates memory, loads model weights, and sets up mesh/sharding
    constraints, etc.
    """
    pass

  @abc.abstractmethod
  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    """Triggers JIT compilation.

    Role-specific and worker-driven: the worker synthesizes its own warmup
    dummies from `shape_config` (plus its tokenizer/geometry), since the
    orchestrator cannot fabricate engine-private warmup shapes.
    """
    pass

  @abc.abstractmethod
  def start(self) -> None:
    """Starts the worker's main loop."""
    pass

  @abc.abstractmethod
  def stop(self) -> None:
    """Gracefully stops the worker."""
    pass

  @abc.abstractmethod
  def health(self) -> datatypes.HealthReport:
    """Returns a liveness/status snapshot for the orchestrator health monitor."""
    pass

  @abc.abstractmethod
  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's static description (roles, capabilities, resources)."""
    pass

  def pause(self) -> None:
    """Stops admitting new work (capability-gated; default: unsupported)."""
    raise NotImplementedError(f"{type(self).__name__} does not support pause().")

  def resume(self) -> None:
    """Resumes a paused worker (capability-gated; default: unsupported)."""
    raise NotImplementedError(
        f"{type(self).__name__} does not support resume()."
    )
