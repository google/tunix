import argparse
import grpc
import logging
import time
from concurrent import futures
from typing import Callable

from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.discovery import discovery

def resolve_discovery_address(discovery_address) -> str:
  # hostname is always "localhost", just extract the port
  _, discovery_port = discovery_address.split(":")
  return f"localhost:{discovery_port}"

class LocalDiscoveryContext(context.DiscoveryContext):
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
  
    hostname = "localhost"

    logging.info(f"register to discovery server at {server_address}")
    discovery.register(self, server_address, hostname, self._args.discovery_port, metadata)
    logging.info(f"registered to discovery server at {server_address}")


class LocalIpcContext(context.IpcContext):
  def __init__(self, args: argparse.Namespace) -> None:
    self._discovery = LocalDiscoveryContext(args)

  def __enter__(self):
    self._discovery.__enter__()
    return self

  def __exit__(self, exc_type, exc, tb):
    self._discovery.__exit__(exc_type, exc, tb)

  @property
  def discovery(self) -> context.DiscoveryContext:
    return self._discovery


class LocalProcessContext(context.ProcessContext):
  """Handles the implementation differences across platforms."""

  def __init__(self, args: argparse.Namespace) -> None:
    self._jax = context.JaxContext()
    self._ipc = LocalIpcContext(args)

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
