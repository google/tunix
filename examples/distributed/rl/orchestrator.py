import argparse
from concurrent import futures
import grpc
import logging
import pickle
import queue
import random
import time

from tunix.experimental.distributed.runtime.context import ProcessContext

from examples.distributed.rl import service_pb2 as pb2
from examples.distributed.rl import service_pb2_grpc as pb2_grpc

class RolloutClient:
  def __init__(self, service_addr: str) -> None:
    self._service_addr = service_addr

  def generate(self, prompt: str) -> str:
    with grpc.insecure_channel(self._service_addr) as channel:
      stub = pb2_grpc.RolloutServiceStub(channel)

      request = pb2.GenerateRequest(prompt=prompt)

      try:
        response = stub.Generate(request)
        return response.completion
      except grpc.RpcError as e:
        raise RuntimeError(f"generate failed: {e.code()} - {e.details()}")

class TrainerClient:
  def __init__(self, service_addr: str) -> None:
    self._service_addr = service_addr

  def train(self, prompt: str, completion: str) -> str:
    with grpc.insecure_channel(self._service_addr) as channel:
      stub = pb2_grpc.TrainerServiceStub(channel)

      request = pb2.TrainRequest(prompt=prompt, completion=completion)

      try:
        response = stub.Train(request)
        return response.weights
      except grpc.RpcError as e:
        raise RuntimeError(f"train failed: {e.code()} - {e.details()}")

def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="this is orchestrator!", help="")
  parser.add_argument("--max_train_step", type=int, default=100, help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  # setup discovery for workers
  rollout_client_futures = queue.Queue()
  trainer_client_future = futures.Future()

  def accept_worker(hostname: str, discovery_port: int, metadata: bytes) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    server_address = f"{hostname}:{md["server_port"]}"
    server_id = md["server_id"]

    logging.info(f"discovered {service_type} service {server_id} at {server_address}")

    match service_type:
      case "rollout":
        rollout_client_future = futures.Future()
        rollout_client_future.set_result(RolloutClient(server_address))
        rollout_client_futures.put((server_id, rollout_client_future))
      case "trainer":
        trainer_client_future.set_result(TrainerClient(server_address))
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

  context.ipc.discovery.on_register(accept_worker)

  def pick_rollout_client():
    # wait at least two rollout clients
    while rollout_client_futures.qsize() < 2:
      time.sleep(1)
    server_id, rollout_client_future = random.choice(list(rollout_client_futures.queue))
    return server_id, rollout_client_future.result()

  trainer_client = trainer_client_future.result()

  try:
    # just to simulate the data flow
    # don't relate this code to actual RL algorithms
    logging.info("run simulated RL training steps...")
    for i in range(args.max_train_step):
      logging.info(f"\n------ iteration {i} ------\n")

      prompt = f"{random.randint(0, 10)} + {random.randint(0, 10)}"
      logging.info(f"[loader] prompt: {prompt}")

      server_id, rollout_client = pick_rollout_client()
      completion = rollout_client.generate(prompt)
      logging.info(f"[{server_id}] completion: {completion}")

      weights = trainer_client.train(prompt, completion)
      logging.info(f"[trainer] weights: {weights}")
  except KeyboardInterrupt:
    pass

  print("Press Ctrl+C to exit...")
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    pass
