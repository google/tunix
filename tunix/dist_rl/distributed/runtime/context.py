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

from typing import Any, Callable


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
