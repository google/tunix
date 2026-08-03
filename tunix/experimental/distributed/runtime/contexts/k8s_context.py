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

"""Kubernetes-specific distributed runtime context implementations."""

import argparse
import logging
import os
from typing import Any, Callable

from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.discovery import discovery


def resolve_discovery_address(discovery_address: str) -> str:
  """Resolves the discovery server hostname and port from a discovery address string.

  Args:
    discovery_address: Address formatted as 'discovery_id:port'.

  Returns:
    Fully resolved discovery server address formatted as 'hostname:port'.
  """
  discovery_id, discovery_port = discovery_address.split(":")
  # this should be consistent with distributed/deployment/yaml_generator.py and
  # templates in distributed/deployment/yamls/.
  # the discovery server runs in the same pod as the main process, that is, run
  # on the "proc" replicated job at pod index 0-0.
  hostname = f"{discovery_id}-proc-0-0.{discovery_id}"
  return f"{hostname}:{discovery_port}"


def resolve_self_hostname() -> str:
  """Resolves the FQDN hostname of this pod from Kubernetes environment variables.

  Returns:
    Fully qualified domain name (FQDN) for this pod.

  Raises:
    ValueError: If required Kubernetes pod environment variables are missing.
  """
  required_envs = [
      "JOBSET_NAME",
      "REPLICATED_JOB_NAME",
      "JOB_INDEX",
      "POD_INDEX",
  ]
  missing_envs = [env for env in required_envs if env not in os.environ]
  if missing_envs:
    raise ValueError(
        f"Missing required environment variable(s): {', '.join(missing_envs)}"
    )

  jobset_name = os.environ["JOBSET_NAME"]
  replicated_job = os.environ["REPLICATED_JOB_NAME"]
  job_index = os.environ["JOB_INDEX"]
  pod_index = os.environ["POD_INDEX"]

  # Constructing a fully qualified domain name (FQDN)
  fqdn = f"{jobset_name}-{replicated_job}-{job_index}-{pod_index}.{jobset_name}"
  return fqdn


class K8sJaxContext(context.JaxContext):
  """JAX distributed runtime initializer for Kubernetes pods."""

  def initialize(self) -> None:
    """Initializes Pathways or standard JAX distributed runtime based on environment."""
    if "proxy" in os.environ.get("JAX_PLATFORMS") and os.environ.get(
        "JAX_BACKEND_TARGET"
    ):
      logging.info("initializing Pathways runtime")

      import pathwaysutils

      pathwaysutils.initialize()
    else:
      logging.info("initializing multi-controller JAX runtime")

      import jax

      jax.distributed.initialize()


class K8sDiscoveryContext(context.DiscoveryContext):
  """Kubernetes discovery context managing registration and server hosting."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Kubernetes discovery context.

    Args:
      args: Parsed command-line arguments containing discovery options.
    """
    self._args = args
    self._server = discovery.DiscoveryServer()

  def __enter__(self) -> "K8sDiscoveryContext":
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
      logging.info("discovery server stopped")

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    """Starts the discovery server on the configured port and registers the callback.

    Args:
      callback: Invoked when a peer registers with this server.
    """
    self._server.start(self._args.discovery_port, callback)
    logging.info(
        "discovery server started on port %s", self._args.discovery_port
    )

  def register(self, metadata: bytes) -> None:
    """Registers this Kubernetes pod with the remote discovery server.

    Args:
      metadata: Serialized metadata describing this pod.
    """
    server_address = resolve_discovery_address(self._args.discovery_addrs)

    hostname = resolve_self_hostname()

    logging.info("register to discovery server at %s", server_address)
    discovery.register(
        server_address, hostname, self._args.discovery_port, metadata
    )
    logging.info("registered to discovery server at %s", server_address)


class K8sIpcContext(context.IpcContext):
  """Kubernetes inter-process communication context."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Kubernetes IPC context.

    Args:
      args: Parsed command-line arguments containing discovery options.
    """
    self._discovery = K8sDiscoveryContext(args)

  def __enter__(self) -> "K8sIpcContext":
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
    """Returns the Kubernetes discovery context."""
    return self._discovery


class K8sProcessContext(context.ProcessContext):
  """Handles the implementation differences across platforms for Kubernetes."""

  def __init__(self, args: argparse.Namespace) -> None:
    """Initializes the Kubernetes process context.

    Args:
      args: Parsed command-line arguments.
    """
    self._jax = K8sJaxContext()
    self._ipc = K8sIpcContext(args)

  def __enter__(self) -> "K8sProcessContext":
    """Enters the Kubernetes process context manager scope."""
    self._ipc.__enter__()
    return self

  def __exit__(
      self,
      exc_type: Any | None,
      exc: Any | None,
      tb: Any | None,
  ) -> None:
    """Exits the Kubernetes process context manager scope."""
    self._ipc.__exit__(exc_type, exc, tb)

  @property
  def jax(self) -> context.JaxContext:
    """Returns the Kubernetes JAX runtime context."""
    return self._jax

  @property
  def ipc(self) -> context.IpcContext:
    """Returns the Kubernetes IPC context."""
    return self._ipc
