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

"""Local execution distributed runtime context implementations."""

import argparse
import logging
import os
from typing import Any, Callable

from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.discovery import discovery


def resolve_discovery_address(discovery_address: str) -> str:
  """Resolves a local discovery server address to localhost.

  Args:
    discovery_address: Discovery address string formatted as 'id:port'.

  Returns:
    Resolved address formatted as 'localhost:port'.
  """
  # hostname is always "localhost", just extract the port
  _, discovery_port = discovery_address.split(":")
  return f"localhost:{discovery_port}"


class LocalDiscoveryContext(context.DiscoveryContext):
  """Discovery context for local single-machine execution."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the local discovery context.

    Args:
      args: Parsed command-line arguments containing discovery options.
    """
    self._args = args
    self._server = discovery.DiscoveryServer()

  def __enter__(self) -> "LocalDiscoveryContext":
    """Enters the discovery context manager scope."""
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Stops the local discovery server if started."""
    if self._server.is_started():
      self._server.stop()
      logging.info("discovery server stopped")

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    """Starts the local discovery server on the configured port.

    Args:
      callback: Invoked when a peer registers with this server.
    """
    self._server.start(self._args.discovery_port, callback)
    logging.info(
        "discovery server started on port %s", self._args.discovery_port
    )

  def register(self, metadata: bytes) -> None:
    """Registers this process with the local discovery server.

    Args:
      metadata: Serialized metadata describing this worker.
    """
    server_address = resolve_discovery_address(self._args.discovery_addrs)

    hostname = "localhost"

    logging.info("register to discovery server at %s", server_address)
    discovery.register(
        server_address, hostname, self._args.discovery_port, metadata
    )
    logging.info("registered to discovery server at %s", server_address)


class LocalIpcContext(context.IpcContext):
  """Inter-process communication context for local execution."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the local IPC context.

    Args:
      args: Parsed command-line arguments containing discovery options.
    """
    self._discovery = LocalDiscoveryContext(args)

  def __enter__(self) -> "LocalIpcContext":
    """Enters the IPC context manager scope."""
    self._discovery.__enter__()
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Exits the IPC context manager scope."""
    self._discovery.__exit__(exc_type, exc, tb)

  @property
  def discovery(self) -> context.DiscoveryContext:
    """Returns the local discovery context."""
    return self._discovery


class LocalJaxContext(context.JaxContext):
  """JAX runtime context for local execution.

  Mirrors the Pathways half of K8sProcessContext's JaxContext. When a process is
  pointed at a Pathways proxy, pathwaysutils has to register the "proxy" backend
  factory before anything resolves a backend; otherwise JAX fails with

    Unable to initialize backend 'proxy': Backend 'proxy' is not in the list of
    known backends: ['cpu', 'tpu']

  which reads like the proxy is unreachable rather than unregistered. The base
  JaxContext does nothing, so before this class a local Pathways run could not
  work no matter how the daemons were configured.

  Deliberately not the mcJAX branch k8s_context also carries: a local run is a
  single process that needs no jax.distributed.initialize(), and calling it here
  would change the behaviour of the existing non-Pathways local tests.
  """

  def initialize(self) -> None:
    """Registers the Pathways proxy backend when this process targets one."""
    if "proxy" not in os.environ.get("JAX_PLATFORMS", ""):
      return
    if not os.environ.get("JAX_BACKEND_TARGET"):
      return

    logging.info("Initializing the Pathways runtime for local execution.")
    import pathwaysutils  # pylint: disable=g-import-not-at-top

    pathwaysutils.initialize()


class LocalProcessContext(context.ProcessContext):
  """Handles the implementation differences across platforms for local execution."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the local process context.

    Args:
      args: Parsed command-line arguments.
    """
    self._jax = LocalJaxContext()
    self._ipc = LocalIpcContext(args)

  def __enter__(self) -> "LocalProcessContext":
    """Enters the process context manager scope."""
    self._ipc.__enter__()
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Exits the process context manager scope."""
    self._ipc.__exit__(exc_type, exc, tb)

  @property
  def jax(self) -> context.JaxContext:
    """Returns the JAX runtime context."""
    return self._jax

  @property
  def ipc(self) -> context.IpcContext:
    """Returns the local IPC context."""
    return self._ipc
