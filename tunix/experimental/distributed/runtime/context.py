"""The context API provides a platform agnostic way for the process to interact with the running environment."""

import argparse
import grpc
import time
from concurrent import futures
from typing import Callable

class JaxContext:
  def initialize(self) -> None:
    pass


class DiscoveryContext:
  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    pass

  def register(self, metadata: bytes) -> None:
    pass


class IpcContext:
  @property
  def discovery(self) -> DiscoveryContext:
    pass


class ProcessContext:
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    pass

  @property
  def jax(self) -> JaxContext:
    pass

  @property
  def ipc(self) -> IpcContext:
    pass
