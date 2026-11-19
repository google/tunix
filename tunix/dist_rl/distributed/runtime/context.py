# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The context API provides a platform-agnostic way for a process to interact with its runtime environment."""

from collections.abc import Callable
import importlib
import os
import sys
from typing import Any

from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import env_utils
from tunix.dist_rl.common import execution_target

_TARGET_ENV_DEFAULTS: dict[
    execution_target.ExecutionTarget, tuple[tuple[str, str], ...]
] = {
    execution_target.JAX_TPU: (),
    execution_target.JAX_GPU: (
        ("JAX_PLATFORMS", "cuda,cpu"),
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
        ("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85"),
    ),
    execution_target.TORCH_TPU: (
        ("JAX_PLATFORMS", "cpu"),
        ("PJRT_DEVICE", "TPU"),
    ),
}


class JaxContext:
  """Abstract interface for initializing and managing the JAX distributed runtime."""

  def initialize(self) -> None:
    """Initializes the JAX distributed runtime for this process."""
    pass


class DiscoveryContext:
  """Abstract interface for peer discovery registration and callbacks."""

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    """Registers a callback invoked when a peer node registers itself.

    Args:
      callback: A function accepting (hostname, port, metadata) from registering
        peers.
    """
    pass

  def register(self, metadata: bytes) -> None:
    """Registers this process with the discovery service.

    Args:
      metadata: Serialized metadata bytes describing this worker node.
    """
    pass


class IpcContext:
  """Abstract interface providing inter-process communication contexts."""

  @property
  def discovery(self) -> DiscoveryContext:
    """Returns the discovery context associated with this IPC context."""
    return DiscoveryContext()


class ProcessContext:
  """Abstract interface representing the combined execution context of a worker process."""

  def __enter__(self) -> "ProcessContext":
    """Enters the process context manager scope."""
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Exits the process context manager scope."""
    pass

  @property
  def jax(self) -> JaxContext:
    """Returns the JAX context for this process."""
    return JaxContext()

  @property
  def ipc(self) -> IpcContext:
    """Returns the IPC context for this process."""
    return IpcContext()

  def prepare_env(
      self,
      target: execution_target.ExecutionTarget,
      *,
      weight_sync_mode: common_datatypes.WeightSyncMode = (
          common_datatypes.WeightSyncMode.NONE
      ),
  ) -> None:
    """Applies pre-import runtime environment defaults for `target`."""
    if target not in _TARGET_ENV_DEFAULTS:
      raise ValueError(
          f"Unsupported execution target '{target}' for runtime"
          " environment preparation."
      )
    for key, default in _TARGET_ENV_DEFAULTS[target]:
      if key not in os.environ:
        os.environ[key] = default
    if target == execution_target.JAX_TPU and (
        env_utils.parse_env_bool("USE_RAIDEN")
        or weight_sync_mode == common_datatypes.WeightSyncMode.RAIDEN
    ):
      mod_name = "tunix.dist_rl.weight_sync.raiden_preload"
      preload = (
          sys.modules[mod_name]
          if mod_name in sys.modules
          else importlib.import_module(mod_name)
      )
      preload.import_raiden()

  def initialize_runtime(
      self, target: execution_target.ExecutionTarget
  ) -> None:
    """Initializes the runtime backend for `target`."""
    if target not in execution_target.SUPPORTED_EXECUTION_TARGETS:
      raise ValueError(
          f"Unsupported execution target '{target}' for runtime"
          " initialization."
      )
    if target.framework == execution_target.Framework.JAX:
      self.jax.initialize()
      return
    importlib.import_module("torch")
