import argparse
from concurrent import futures
import logging
import pickle
import time
from typing import Callable

import grpc
import jax
import jax.numpy as jnp
from tunix.experimental.distributed.examples.rl import service_pb2 as pb2
from tunix.experimental.distributed.examples.rl import service_pb2_grpc as pb2_grpc
from tunix.experimental.distributed.runtime.context import ProcessContext


class TrainerServer:

  def __init__(self) -> None:
    self._server: grpc.Server | None = None

  def start(self, port: int, on_train: Callable[[str, str], str]) -> None:
    if self._server is not None:
      raise RuntimeError("server already started")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # define and register handler
    class _handler(pb2_grpc.TrainerServiceServicer):

      def Train(self, request: pb2.TrainRequest, context: grpc.ServicerContext):
        return pb2.TrainResponse(
            weights=on_train(request.prompt, request.completion)
        )

    pb2_grpc.add_TrainerServiceServicer_to_server(_handler(), server)

    # start server
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    self._server = server

    logging.info(f"trainer server started on port {port}")

  def stop(self, timeout: float | None = None) -> None:
    if self._server:
      self._server.stop(timeout)
      self._server.wait_for_termination(timeout)

      logging.info("trainer server stopped")


def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--message", type=str, default="this is trainer!", help=""
  )
  parser.add_argument("--server_id", type=str, default="", help="")
  parser.add_argument("--server_port", type=int, default=11111, help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  # context.jax.initialize()
  # logging.info(f"jax initialized: {jax.devices()}")

  weights = [5.0]

  def on_train(prompt: str, completion: str) -> str:
    # if completion does the math correctly, learn a lot
    # otherwise, learn little
    if eval(prompt) == int(completion.split()[1]):
      weights[0] = 0.99 * weights[0] + 0.01 * float(completion.split()[1])
    else:
      weights[0] = 0.9999 * weights[0] + 0.0001 * float(completion.split()[1])
    logging.info(
        f"[{args.server_id}] train({prompt}, {completion}) ->"
        f" [{weights[0]:.2f}]"
    )
    return f"[{weights[0]:.2f}]"

  server = TrainerServer()
  server.start(args.server_port, on_train=on_train)
  assert context is not None
  context.ipc.discovery.register(
      metadata=pickle.dumps({
          "service_type": "trainer",
          "server_port": args.server_port,
          "server_id": args.server_id,
      })
  )

  print("Press Ctrl+C to exit...")
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    pass

  server.stop()
