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

"""gRPC-based peer discovery server and client helper functions."""

from concurrent import futures
import logging
import time
from typing import Callable

import grpc
from tunix.dist_rl.distributed.runtime.discovery import discovery_service_pb2 as pb2
from tunix.dist_rl.distributed.runtime.discovery import discovery_service_pb2_grpc as pb2_grpc


class RegistrationUnavailableError(RuntimeError):
  """The discovery server cannot accept a registration right now.

  A registration callback raises it when the failure is transient, e.g. while
  the server is shutting down and a restarted server will take its place. The
  server returns `UNAVAILABLE` for it, so `register` retries instead of
  failing the worker for good.
  """


class DiscoveryServer:
  """Lightweight gRPC server for registering distributed worker nodes."""

  def __init__(self) -> None:
    """Initializes an unstarted discovery server instance."""
    self._server: grpc.Server | None = None

  def is_started(self) -> bool:
    """Returns True if the discovery server is running."""
    return self._server is not None

  def start(
      self, port: int, callback: Callable[[str, int, bytes], None]
  ) -> None:
    """Starts the discovery gRPC server on the given port.

    Args:
      port: Network port on which the gRPC discovery server listens.
      callback: Function invoked when a peer node registers via RPC.

    Raises:
      ValueError: If `port` is zero or invalid.
      RuntimeError: If the server has already been started.
    """
    if not port:
      raise ValueError("port must be non-zero. did you set --discovery_port ?")
    if self._server is not None:
      raise RuntimeError("server already started")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # define and register handler
    class _handler(pb2_grpc.DiscoveryServiceServicer):

      def Register(
          self, request: pb2.RegisterRequest, context: grpc.ServicerContext
      ):
        try:
          callback(request.hostname, request.port, request.metadata)
        except RegistrationUnavailableError as e:
          context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        return pb2.RegisterResponse()

    pb2_grpc.add_DiscoveryServiceServicer_to_server(_handler(), server)

    # start server
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    self._server = server

  def stop(self, timeout: float | None = None) -> None:
    """Stops the discovery gRPC server and waits for termination.

    Args:
      timeout: Grace period in seconds to wait for active RPCs to terminate.
    """
    if self._server:
      self._server.stop(timeout)
      self._server.wait_for_termination(timeout)
      self._server = None


def register(
    server_address: str,
    hostname: str,
    port: int,
    metadata: bytes,
    address_resolver: Callable[[], str] | None = None,
) -> None:
  """Registers a node with the remote discovery server using exponential backoff.

  Retries while the server returns `UNAVAILABLE`: it is not up yet, it moved,
  or it cannot accept registrations right now (`RegistrationUnavailableError`).

  Args:
    server_address: Host and port of the target discovery server (e.g.
      'host:port').
    hostname: Hostname or address of the registering node.
    port: Port of the registering node.
    metadata: Custom serialized metadata bytes to pass to the server.
    address_resolver: Returns the server's current address, for servers that can
      move (e.g. a restarted server behind a service name that resolves to its
      current host). Called before each retry; when its address differs from the
      current one, the next attempt goes to the new address, starting over from
      the initial backoff. If it raises, the current address is kept.

  Raises:
    ValueError: If `server_address` is empty.
    RuntimeError: If registration fails with a non-retryable gRPC error.
  """
  if not server_address:
    raise ValueError(
        "server_address must be non-empty. did you set --discovery_addrs ?"
    )

  request = pb2.RegisterRequest(hostname=hostname, port=port, metadata=metadata)
  channel = grpc.insecure_channel(server_address)
  stub = pb2_grpc.DiscoveryServiceStub(channel)
  try:
    delay = 1
    count = 0
    while True:
      try:
        stub.Register(request)
        return
      except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.UNAVAILABLE:  # pyrefly: ignore[missing-attribute]
          raise RuntimeError(
              f"discovery register failed: {e.code()} - {e.details()}"  # pyrefly: ignore[missing-attribute]
          ) from e
      time.sleep(delay)
      count += 1
      if count >= 60:
        delay = min(delay * 2, 300)
        count = 0
      if address_resolver is None:
        continue
      try:
        new_address = address_resolver()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Could not re-resolve the discovery server address; retrying %s:"
            " %s",
            server_address,
            e,
        )
        continue
      if new_address and new_address != server_address:
        logging.info(
            "Discovery server moved from %s to %s; registering there.",
            server_address,
            new_address,
        )
        channel.close()
        server_address = new_address
        channel = grpc.insecure_channel(server_address)
        stub = pb2_grpc.DiscoveryServiceStub(channel)
        # The new server may still be starting; don't make it wait out the
        # backoff accumulated against the old one.
        delay = 1
        count = 0
  finally:
    channel.close()
