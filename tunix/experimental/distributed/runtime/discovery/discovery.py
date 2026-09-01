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
import dataclasses
import logging
import threading
import time
from typing import Any, Callable
import uuid

import grpc
from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2 as pb2
from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2_grpc as pb2_grpc


class _RegistryServicer(pb2_grpc.DiscoveryServiceServicer):
  """gRPC servicer for one-shot fire-and-forget worker registration."""

  def __init__(
      self, callback: Callable[[str, int, bytes], None] | None = None
  ) -> None:
    self._callback = callback

  def Register(
      self, request: pb2.RegisterRequest, context: grpc.ServicerContext
  ) -> pb2.RegisterResponse:
    if self._callback is not None:
      try:
        self._callback(request.hostname, request.port, request.metadata)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Error in discovery server callback: %s", e)
    return pb2.RegisterResponse()

  def Connect(
      self, request: pb2.ConnectRequest, context: grpc.ServicerContext
  ) -> pb2.ConnectResponse:
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED,
        "Discovery server is running in register mode. Use Register"
        " RPC instead of Connect.",
    )
    return pb2.ConnectResponse()

  def Heartbeat(
      self, request: pb2.HeartbeatRequest, context: grpc.ServicerContext
  ) -> pb2.HeartbeatResponse:
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED,
        "Discovery server is running in register mode. Heartbeats not"
        " supported.",
    )
    return pb2.HeartbeatResponse()


class _ConnectionServicer(pb2_grpc.DiscoveryServiceServicer):
  """gRPC servicer for managed persistent client connections with heartbeats and lease eviction."""

  @dataclasses.dataclass
  class ClientConnection:
    client_id: str
    hostname: str
    port: int
    metadata: bytes
    last_seen: float

  def __init__(
      self,
      on_client_connected: (
          Callable[[str, str, int, bytes, bool], None] | None
      ) = None,
      on_client_disconnected: (
          Callable[[str, str, int, str], None] | None
      ) = None,
      heartbeat_sec: int = 5,
  ) -> None:
    self._on_client_connected = on_client_connected
    self._on_client_disconnected = on_client_disconnected
    self._heartbeat_sec = heartbeat_sec
    self._server_epoch = str(uuid.uuid4())
    self._connected_clients: dict[str, _ConnectionServicer.ClientConnection] = (
        {}
    )
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._evictor_thread: threading.Thread | None = None
    evictor_thread = threading.Thread(
        target=self._run_evictor_loop, daemon=True
    )
    evictor_thread.start()
    self._evictor_thread = evictor_thread

  def Register(
      self, request: pb2.RegisterRequest, context: grpc.ServicerContext
  ) -> pb2.RegisterResponse:
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED,
        "Discovery server is running in connect mode. Use Connect"
        " RPC instead of Register.",
    )
    return pb2.RegisterResponse()

  def Connect(
      self, request: pb2.ConnectRequest, context: grpc.ServicerContext
  ) -> pb2.ConnectResponse:
    client_id = request.client_id or f"{request.hostname}:{request.port}"
    now = time.time()

    with self._lock:
      is_reconnect = client_id in self._connected_clients
      self._connected_clients[client_id] = _ConnectionServicer.ClientConnection(
          client_id=client_id,
          hostname=request.hostname,
          port=request.port,
          metadata=request.metadata,
          last_seen=now,
      )

    if self._on_client_connected is not None:
      try:
        self._on_client_connected(
            client_id,
            request.hostname,
            request.port,
            request.metadata,
            is_reconnect,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.exception(
            "Error in discovery server on_client_connected callback: %s", e
        )

    return pb2.ConnectResponse(
        server_epoch=self._server_epoch,
        heartbeat_sec=self._heartbeat_sec,
    )

  def Heartbeat(
      self, request: pb2.HeartbeatRequest, context: grpc.ServicerContext
  ) -> pb2.HeartbeatResponse:
    with self._lock:
      if (
          request.server_epoch != self._server_epoch
          or request.client_id not in self._connected_clients
      ):
        return pb2.HeartbeatResponse(
            action=pb2.HEARTBEAT_ACTION_RE_REGISTER,
            server_epoch=self._server_epoch,
            heartbeat_sec=self._heartbeat_sec,
        )

      self._connected_clients[request.client_id].last_seen = time.time()
      return pb2.HeartbeatResponse(
          action=pb2.HEARTBEAT_ACTION_OK,
          server_epoch=self._server_epoch,
          heartbeat_sec=self._heartbeat_sec,
      )

  def _run_evictor_loop(self) -> None:
    while not self._stop_event.is_set():
      self._stop_event.wait(timeout=float(self._heartbeat_sec))
      if self._stop_event.is_set():
        break

      now = time.time()
      evicted: list[_ConnectionServicer.ClientConnection] = []

      with self._lock:
        timeout_threshold = 3 * self._heartbeat_sec
        stale_ids = [
            cid
            for cid, reg in self._connected_clients.items()
            if now - reg.last_seen > timeout_threshold
        ]
        for cid in stale_ids:
          evicted.append(self._connected_clients.pop(cid))

      if self._on_client_disconnected is not None:
        for reg in evicted:
          try:
            self._on_client_disconnected(
                reg.client_id, reg.hostname, reg.port, "heartbeat_timeout"
            )
          except Exception as e:  # pylint: disable=broad-except
            logging.exception(
                "Error in discovery server on_client_disconnected callback: %s",
                e,
            )

  def stop(self, timeout: float | None = None) -> None:
    """Stops the persistent servicer evictor thread."""
    self._stop_event.set()
    if self._evictor_thread and self._evictor_thread.is_alive():
      self._evictor_thread.join(timeout=timeout)
      self._evictor_thread = None


class DiscoveryServer:
  """Lightweight gRPC server for discovery, operating in either register or connect mode."""

  def __init__(self, heartbeat_sec: int = 5) -> None:
    """Initializes an unstarted discovery server instance."""
    self._server: grpc.Server | None = None
    self._executor: futures.ThreadPoolExecutor | None = None
    self._servicer: _RegistryServicer | _ConnectionServicer | None = None
    self._mode: str | None = None  # "register" or "connect"
    self._heartbeat_sec: int = heartbeat_sec

    # Registered configuration arguments
    self._on_client_register: Callable[[str, int, bytes], None] | None = None
    self._on_client_connected: (
        Callable[[str, str, int, bytes, bool], None] | None
    ) = None
    self._on_client_disconnected: (
        Callable[[str, str, int, str], None] | None
    ) = None

  def is_started(self) -> bool:
    """Returns True if the discovery server is running."""
    return self._server is not None

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    """Configures register mode for the server."""
    if self._mode == "connect":
      raise RuntimeError(
          "Cannot configure on_register when on_connect is already configured."
      )
    self._mode = "register"
    self._on_client_register = callback

  def on_connect(
      self,
      on_client_connected: (
          Callable[[str, str, int, bytes, bool], None] | None
      ) = None,
      *,
      on_client_disconnected: (
          Callable[[str, str, int, str], None] | None
      ) = None,
  ) -> None:
    """Configures connect mode for the server."""
    if self._mode == "register":
      raise RuntimeError(
          "Cannot configure on_connect when on_register is already configured."
      )
    self._mode = "connect"
    self._on_client_connected = on_client_connected
    self._on_client_disconnected = on_client_disconnected

  def start(
      self,
      port: int,
      heartbeat_sec: int = 5,
  ) -> None:
    """Starts the discovery gRPC server on the given port."""
    if not port:
      raise ValueError("port must be non-zero. did you set --discovery_port ?")
    if self._server is not None:
      raise RuntimeError("server already started")
    if self._mode is None:
      raise RuntimeError(
          "Discovery server mode not configured. Call on_register() or"
          " on_connect() before starting the server."
      )

    if self._mode == "connect":
      self._servicer = _ConnectionServicer(
          on_client_connected=self._on_client_connected,
          on_client_disconnected=self._on_client_disconnected,
          heartbeat_sec=heartbeat_sec or self._heartbeat_sec,
      )
    else:
      self._servicer = _RegistryServicer(self._on_client_register)

    self._executor = futures.ThreadPoolExecutor(max_workers=10)
    server = grpc.server(self._executor)
    pb2_grpc.add_DiscoveryServiceServicer_to_server(self._servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    self._server = server

  def stop(self, timeout: float | None = None) -> None:
    """Stops the discovery gRPC server and waits for termination."""
    if isinstance(self._servicer, _ConnectionServicer):
      self._servicer.stop(timeout=timeout)

    if self._server:
      self._server.stop(timeout)
      self._server.wait_for_termination(timeout)
      self._server = None
    if self._executor:
      self._executor.shutdown(wait=True)
      self._executor = None
    self._servicer = None


def register(
    server_address: str, hostname: str, port: int, metadata: bytes
) -> None:
  """Registers a node with the remote discovery server using exponential backoff (one-shot)."""
  if not server_address:
    raise ValueError(
        "server_address must be non-empty. did you set --discovery_addrs ?"
    )

  with grpc.insecure_channel(server_address) as channel:
    stub = pb2_grpc.DiscoveryServiceStub(channel)

    request = pb2.RegisterRequest(
        hostname=hostname, port=port, metadata=metadata
    )

    delay = 1
    count = 0
    while True:
      try:
        stub.Register(request)
        break
      except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
          time.sleep(delay)
          count += 1
          if count >= 60:
            delay = min(delay * 2, 300)
            count = 0
          continue
        else:
          raise RuntimeError(
              f"discovery register failed: {e.code()} - {e.details()}"  # pytype: disable=attribute-error
          )


class DiscoveryClient:
  """Manages persistent worker registration and background heartbeat loop."""

  def __init__(
      self,
      server_address: str,
      hostname: str,
      port: int,
      metadata: bytes,
      client_id: str,
      *,
      on_connected: Callable[[str, bool], None] | None = None,
      on_disconnected: Callable[[str, str], None] | None = None,
  ) -> None:
    self._server_address = server_address
    self._hostname = hostname
    self._port = port
    self._metadata = metadata
    self._client_id = client_id
    self._on_connected = on_connected
    self._on_disconnected = on_disconnected

    self._server_epoch: str = ""
    self._heartbeat_sec: int = 5
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._channel: grpc.Channel | None = None
    self._stub: pb2_grpc.DiscoveryServiceStub | None = None

  def start(self) -> None:
    """Starts the discovery client, performs initial connect, and launches background heartbeats."""
    if not self._server_address:
      raise ValueError(
          "server_address must be non-empty. did you set --discovery_addrs ?"
      )

    self._channel = grpc.insecure_channel(self._server_address)
    self._stub = pb2_grpc.DiscoveryServiceStub(self._channel)

    response = self._connect_with_backoff()
    self._server_epoch = response.server_epoch
    self._heartbeat_sec = response.heartbeat_sec or 5

    if self._on_connected is not None:
      try:
        self._on_connected(self._server_epoch, False)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception(
            "Error in discovery client on_connected callback: %s", e
        )

    self._stop_event.clear()
    self._thread = threading.Thread(
        target=self._run_heartbeat_loop, daemon=True
    )
    self._thread.start()

  def _connect_with_backoff(self) -> pb2.ConnectResponse:
    request = pb2.ConnectRequest(
        client_id=self._client_id,
        hostname=self._hostname,
        port=self._port,
        metadata=self._metadata,
    )
    delay = 1
    while not self._stop_event.is_set():
      try:
        assert self._stub is not None
        return self._stub.Connect(request)
      except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
          if self._stop_event.wait(delay):
            break
          delay = min(delay * 2, 60)
          continue
        else:
          raise RuntimeError(
              f"discovery connect failed: {e.code()} - {e.details()}"  # pytype: disable=attribute-error
          )
    raise RuntimeError("discovery client stopped during connect")

  def _run_heartbeat_loop(self) -> None:
    while not self._stop_event.is_set():
      self._stop_event.wait(timeout=float(self._heartbeat_sec))
      if self._stop_event.is_set():
        break

      try:
        assert self._stub is not None
        req = pb2.HeartbeatRequest(
            client_id=self._client_id, server_epoch=self._server_epoch
        )
        resp = self._stub.Heartbeat(req)

        if resp.action == pb2.HEARTBEAT_ACTION_OK:
          if resp.heartbeat_sec:
            self._heartbeat_sec = resp.heartbeat_sec
        elif resp.action == pb2.HEARTBEAT_ACTION_RE_REGISTER:
          old_epoch = self._server_epoch
          if self._on_disconnected is not None:
            try:
              self._on_disconnected(old_epoch, "epoch_mismatch")
            except Exception as e:  # pylint: disable=broad-except
              logging.exception(
                  "Error in discovery client on_disconnected callback: %s", e
              )

          new_resp = self._connect_with_backoff()
          self._server_epoch = new_resp.server_epoch
          if new_resp.heartbeat_sec:
            self._heartbeat_sec = new_resp.heartbeat_sec

          if self._on_connected is not None:
            try:
              self._on_connected(self._server_epoch, True)
            except Exception as e:  # pylint: disable=broad-except
              logging.exception(
                  "Error in discovery client on_connected callback: %s", e
              )
      except grpc.RpcError as e:
        old_epoch = self._server_epoch
        if self._on_disconnected is not None:
          try:
            self._on_disconnected(old_epoch, "rpc_error")
          except Exception as ex:  # pylint: disable=broad-except
            logging.exception(
                "Error in discovery client on_disconnected callback: %s", ex
            )

        try:
          new_resp = self._connect_with_backoff()
          self._server_epoch = new_resp.server_epoch
          if new_resp.heartbeat_sec:
            self._heartbeat_sec = new_resp.heartbeat_sec

          if self._on_connected is not None:
            try:
              self._on_connected(self._server_epoch, True)
            except Exception as ex:  # pylint: disable=broad-except
              logging.exception(
                  "Error in discovery client on_connected callback: %s", ex
              )
        except Exception as ex:  # pylint: disable=broad-except
          logging.exception(
              "Failed to re-connect with discovery server: %s", ex
          )

  def stop(self, timeout: float | None = None) -> None:
    """Stops the discovery client and background heartbeat loop."""
    self._stop_event.set()
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=timeout)
      self._thread = None
    if self._channel:
      self._channel.close()
      self._channel = None
      self._stub = None


def connect(
    server_address: str,
    hostname: str,
    port: int,
    metadata: bytes,
    client_id: str,
    *,
    on_connected: Callable[[str, bool], None] | None = None,
    on_disconnected: Callable[[str, str], None] | None = None,
) -> DiscoveryClient:
  """Establishes a persistent connection with heartbeats."""
  client = DiscoveryClient(
      server_address=server_address,
      hostname=hostname,
      port=port,
      metadata=metadata,
      client_id=client_id,
      on_connected=on_connected,
      on_disconnected=on_disconnected,
  )
  client.start()
  return client
