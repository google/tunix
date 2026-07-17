import argparse
import grpc
import time
from concurrent import futures
from typing import Callable

from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2 as pb2
from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2_grpc as pb2_grpc

class DiscoveryServer:
  def __init__(self) -> None:
    self._server: grpc.Server | None = None

  def is_started(self) -> bool:
    return self._server is not None

  def start(self, port: int, callback: Callable[[str, int, bytes], None]) -> None:
    if not port:
      raise ValueError("port must be non-zero. did you set --discovery_port ?")
    if self._server is not None:
      raise RuntimeError("server already started")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # define and register handler
    class _handler(pb2_grpc.DiscoveryServiceServicer):
      def Register(self, request: pb2.RegisterRequest, context: grpc.ServicerContext):
        callback(request.hostname, request.port, request.metadata)
        return pb2.RegisterResponse()
    pb2_grpc.add_DiscoveryServiceServicer_to_server(_handler(), server)

    # start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    self._server = server

  def stop(self, timeout: float | None = None) -> None:
    if self._server:
      self._server.stop(timeout)
      self._server.wait_for_termination(timeout)

def register(self, server_address: str, hostname: str, port: int, metadata: bytes) -> None:
  if not server_address:
    raise ValueError("server_address must be non-empty. did you set --discovery_addrs ?")

  with grpc.insecure_channel(server_address) as channel:
    stub = pb2_grpc.DiscoveryServiceStub(channel)

    request = pb2.RegisterRequest(hostname=hostname, port=port, metadata=metadata)

    delay = 1
    while True:
      try:
        stub.Register(request)
        break
      except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
          time.sleep(delay)
          delay = min(delay * 2, 300)
          continue
        else:
          raise RuntimeError(f"discovery register failed: {e.code()} - {e.details()}")
