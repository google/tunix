import argparse
import grpc
import logging
import pickle
import random
import time
from concurrent import futures
from typing import Callable

import jax
import jax.numpy as jnp

from tunix.experimental.distributed.runtime.context import ProcessContext

from examples.distributed.rl import service_pb2 as pb2
from examples.distributed.rl import service_pb2_grpc as pb2_grpc

class RolloutServer:
  def __init__(self) -> None:
    self._server: grpc.Server | None = None

  def start(self, port: int, on_generate: Callable[[str], str]) -> None:
    if self._server is not None:
      raise RuntimeError("server already started")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # define and register handler
    class _handler(pb2_grpc.RolloutServiceServicer):
      def Generate(self, request: pb2.GenerateRequest, context: grpc.ServicerContext):
        return pb2.GenerateResponse(completion=on_generate(request.prompt))
    pb2_grpc.add_RolloutServiceServicer_to_server(_handler(), server)

    # start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    self._server = server

    logging.info(f"rollout server started on port {port}")

  def stop(self, timeout: float | None = None) -> None:
    if self._server:
      self._server.stop(timeout)
      self._server.wait_for_termination(timeout)

      logging.info("rollout server stopped")

def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="this is rollout!", help="")
  parser.add_argument("--server_id", type=str, default="", help="")
  parser.add_argument("--server_port", type=int, default=11111, help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  # context.jax.initialize()
  # logging.info(f"jax initialized: {jax.devices()}")

  def on_generate(prompt: str) -> str:
    x, op, y = prompt.split(' ')
    expr = f"{x} {op} {y}"
    random_error = random.randint(0, 2) - 1
    completion = f"= {eval(expr) + random_error}"
    logging.info(f"[{args.server_id}] generate({prompt}) -> {completion}")
    return completion

  server = RolloutServer()
  server.start(args.server_port, on_generate=on_generate)

  context.ipc.discovery.register(metadata=pickle.dumps({
    "service_type": "rollout",
    "server_port": args.server_port,
    "server_id": args.server_id,
  }))

  print("Press Ctrl+C to exit...")
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    pass

  server.stop()
