import argparse
import grpc
import logging
import os
import time

from concurrent import futures
from typing import Callable

from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.discovery import discovery

def resolve_discovery_address(discovery_address) -> str:
  discovery_id, discovery_port = discovery_address.split(":")
  # this should be consistent with distributed/deployment/yaml_generator.py
  hostname = f"{discovery_id}-proc-0-0.{discovery_id}"
  return f"{hostname}:{discovery_port}"

def resolve_self_hostname() -> str:
  required_envs = ["JOBSET_NAME", "REPLICATED_JOB_NAME", "JOB_INDEX", "POD_INDEX"]
  missing_envs = [env for env in required_envs if env not in os.environ]
  if missing_envs:
    raise ValueError(f"Missing required environment variable(s): {', '.join(missing_envs)}")

  jobset_name = os.environ["JOBSET_NAME"]
  replicated_job = os.environ["REPLICATED_JOB_NAME"]
  job_index = os.environ["JOB_INDEX"]
  pod_index = os.environ["POD_INDEX"]

  # Constructing a fully qualified domain name (FQDN)
  # Format: <pod-hostname>.<headless-service-name>.<namespace>.svc.cluster.local
  fqdn = f"{jobset_name}-{replicated_job}-{job_index}-{pod_index}.{jobset_name}"
  return fqdn


class K8sJaxContext(context.JaxContext):
  def initialize(self) -> None:
    if os.environ.get("JAX_PLATFORMS") and os.environ.get("JAX_BACKEND_TARGET"):
      import pathwaysutils
      pathwaysutils.initialize()
    else:
      import jax
      jax.distributed.initialize()


class K8sDiscoveryContext(context.DiscoveryContext):
  def __init__(self, args: argparse.Namespace) -> None:
    self._args = args
    self._server = discovery.DiscoveryServer()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._server.is_started():
      self._server.stop()
      logging.info("discovery server stopped")

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    self._server.start(self._args.discovery_port, callback)
    logging.info(f"discovery server started on port {self._args.discovery_port}")

  def register(self, metadata: bytes) -> None:
    server_address = resolve_discovery_address(self._args.discovery_addrs)

    hostname = resolve_self_hostname()

    logging.info(f"register to discovery server at {server_address}")
    discovery.register(self, server_address, hostname, self._args.discovery_port, metadata)
    logging.info(f"registered to discovery server at {server_address}")


class K8sIpcContext(context.IpcContext):
  def __init__(self, args: argparse.Namespace) -> None:
    self._discovery = K8sDiscoveryContext(args)

  def __enter__(self):
    self._discovery.__enter__()
    return self

  def __exit__(self, exc_type, exc, tb):
    self._discovery.__exit__(exc_type, exc, tb)

  @property
  def discovery(self) -> context.DiscoveryContext:
    return self._discovery


class K8sProcessContext(context.ProcessContext):
  """Handles the implementation differences across platforms."""

  def __init__(self, args: argparse.Namespace) -> None:
    self._jax = K8sJaxContext()
    self._ipc = K8sIpcContext(args)

  def __enter__(self):
    self._ipc.__enter__()
    return self

  def __exit__(self, exc_type, exc, tb):
    self._ipc.__exit__(exc_type, exc, tb)

  @property
  def jax(self) -> context.JaxContext:
    return self._jax

  @property
  def ipc(self) -> context.IpcContext:
    return self._ipc
