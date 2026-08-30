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

"""Borg and direct multi-host execution distributed runtime context implementations."""

import argparse
import logging
import os
import socket
from typing import Any, Callable

from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.discovery import discovery


def resolve_local_ip() -> str:
  """Returns the local IP address for this host."""
  for family, target in [
      (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
      (socket.AF_INET, ("8.8.8.8", 80)),
  ]:
    try:
      s = socket.socket(family, socket.SOCK_DGRAM)
      try:
        s.connect(target)
        return s.getsockname()[0]
      finally:
        s.close()
    except Exception:
      pass
  try:
    return socket.gethostbyname(socket.gethostname())
  except Exception:
    return "127.0.0.1"


class BorgJaxContext(context.JaxContext):
  """JAX distributed runtime initializer for Borg and direct execution."""

  def initialize(self) -> None:
    """Initializes Pathways or standard multi-controller JAX runtime."""
    if "proxy" in os.environ.get("JAX_PLATFORMS", "") and os.environ.get(
        "JAX_BACKEND_TARGET"
    ):
      logging.info("Initializing Pathways runtime...")
      try:
        import pathwaysutils  # pylint: disable=g-import-not-at-top # pyrefly: ignore[missing-import]

        pathwaysutils.initialize()
      except ImportError:
        pass
    else:
      logging.info("Initializing multi-controller JAX runtime...")
      try:
        import jax  # pylint: disable=g-import-not-at-top # pyrefly: ignore[missing-import]

        jax.distributed.initialize()
      except Exception:
        pass


class BorgDiscoveryContext(context.DiscoveryContext):
  """Borg discovery context managing registration and server hosting."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Borg discovery context."""
    self._args = args
    self._server = discovery.DiscoveryServer()

  def __enter__(self) -> "BorgDiscoveryContext":
    """Enters the discovery context manager scope."""
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Stops the discovery server if started."""
    if self._server.is_started():
      self._server.stop()
      logging.info("Discovery server stopped.")

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    """Starts the discovery server on the configured port."""
    discovery_port = getattr(self._args, "discovery_port", 0)
    if discovery_port:
      self._server.start(discovery_port, callback)
      logging.info("Discovery server started on port %s", discovery_port)

  def register(self, metadata: bytes) -> None:
    """Registers this process with the remote discovery server."""
    discovery_addrs = getattr(self._args, "discovery_addrs", "")
    if not discovery_addrs:
      raise ValueError(
          "discovery_addrs must be non-empty. Did you set --discovery_addrs?"
      )

    hostname = resolve_local_ip()
    port = getattr(self._args, "port", 0) or getattr(self._args, "discovery_port", 0) or 0
    logging.info("Registering to discovery server at %s from host %s port %d", discovery_addrs, hostname, port)
    discovery.register(discovery_addrs, hostname, port, metadata)
    logging.info("Registered to discovery server at %s", discovery_addrs)


class BorgIpcContext(context.IpcContext):
  """Borg inter-process communication context."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Borg IPC context."""
    self._discovery = BorgDiscoveryContext(args)

  def __enter__(self) -> "BorgIpcContext":
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
    """Returns the Borg discovery context."""
    return self._discovery


class BorgProcessContext(context.ProcessContext):
  """Process context for Borg and direct multi-host execution."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Borg process context."""
    self._jax = BorgJaxContext()
    self._ipc = BorgIpcContext(args)

  def __enter__(self) -> "BorgProcessContext":
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
    """Returns the Borg IPC context."""
    return self._ipc
