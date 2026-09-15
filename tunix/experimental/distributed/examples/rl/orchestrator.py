import argparse
from concurrent import futures
import logging
import os
import pickle
import queue
import random
import time

import grpc
from tunix.experimental.distributed.examples.rl import service_pb2 as pb2
from tunix.experimental.distributed.examples.rl import service_pb2_grpc as pb2_grpc
from tunix.experimental.distributed.runtime.context import ProcessContext


class RolloutClient:

  def __init__(self, service_addr: str) -> None:
    self._service_addr = service_addr

  def generate(self, prompt: str) -> tuple[str, dict[str, float]]:
    with grpc.insecure_channel(self._service_addr) as channel:
      stub = pb2_grpc.RolloutServiceStub(channel)

      request = pb2.GenerateRequest(prompt=prompt)

      try:
        response = stub.Generate(request)
        return response.completion, dict(response.metrics)
      except grpc.RpcError as e:
        raise RuntimeError(
            f"generate failed: {e.code()} - {e.details()}"  # pytype: disable=attribute-error
        )


class TrainerClient:

  def __init__(self, service_addr: str) -> None:
    self._service_addr = service_addr

  def train(self, prompt: str, completion: str) -> tuple[str, dict[str, float]]:
    with grpc.insecure_channel(self._service_addr) as channel:
      stub = pb2_grpc.TrainerServiceStub(channel)

      request = pb2.TrainRequest(prompt=prompt, completion=completion)

      try:
        response = stub.Train(request)
        return response.weights, dict(response.metrics)
      except grpc.RpcError as e:
        raise RuntimeError(
            f"train failed: {e.code()} - {e.details()}"  # pytype: disable=attribute-error
        )


def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--message", type=str, default="this is orchestrator!", help=""
  )
  parser.add_argument("--max_train_step", type=int, default=100, help="")
  parser.add_argument(
      "--wandb_project",
      type=str,
      default=os.environ.get("WANDB_PROJECT", ""),
      help="Weights & Biases project name for live dashboard logging.",
  )
  parser.add_argument(
      "--wandb_name",
      type=str,
      default=os.environ.get("WANDB_NAME", ""),
      help="Weights & Biases run name.",
  )
  args = parser.parse_args(argv)

  logging.info(args.message)

  # Initialize Weights & Biases if project flag or env var is specified
  if args.wandb_project:
    try:
      import wandb  # pytype: disable=import-error

      wandb.init(
          project=args.wandb_project,
          name=args.wandb_name or None,
      )
      logging.info(
          "Initialized W&B run '%s' under project '%s'",
          args.wandb_name or "auto",
          args.wandb_project,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to initialize W&B: %s", e)

  # Initialize MetricsLogger with OpenTelemetry double-write enabled
  from tunix.sft import metrics_logger

  options = metrics_logger.MetricsLoggerOptions(
      log_dir="/tmp/tunix_logs",
      project_name=args.wandb_project,
      run_name=args.wandb_name,
      enable_opentelemetry=True,  # pytype: disable=wrong-keyword-args  # Pending PR #1686
  )
  logger = metrics_logger.MetricsLogger(metrics_logger_options=options)

  # setup discovery for workers
  rollout_client_futures = queue.Queue()
  trainer_client_future = futures.Future()

  def accept_worker(
      hostname: str, discovery_port: int, metadata: bytes
  ) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    server_address = f"{hostname}:{md["server_port"]}"
    server_id = md["server_id"]

    logging.info(
        f"discovered {service_type} service {server_id} at {server_address}"
    )

    match service_type:
      case "rollout":
        rollout_client_future = futures.Future()
        rollout_client_future.set_result(RolloutClient(server_address))
        rollout_client_futures.put((server_id, rollout_client_future))
      case "trainer":
        trainer_client_future.set_result(TrainerClient(server_address))
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

  assert context is not None
  context.ipc.discovery.on_register(accept_worker)

  def pick_rollout_client():
    # wait at least two rollout clients
    while rollout_client_futures.qsize() < 2:
      time.sleep(1)
    server_id, rollout_client_future = random.choice(
        list(rollout_client_futures.queue)
    )
    return server_id, rollout_client_future.result()

  trainer_client = trainer_client_future.result()

  try:
    # just to simulate the data flow
    # don't relate this code to actual RL algorithms
    logging.info("run simulated RL training steps with OpenTelemetry logging...")
    for i in range(args.max_train_step):
      logging.info(f"\n------ iteration {i} ------\n")

      prompt = f"{random.randint(0, 10)} + {random.randint(0, 10)}"
      logging.info(f"[loader] prompt: {prompt}")

      step_start_time = time.time()

      server_id, rollout_client = pick_rollout_client()
      completion, rollout_metrics = rollout_client.generate(prompt)
      logging.info(f"[{server_id}] completion: {completion} (metrics: {rollout_metrics})")

      weights, trainer_metrics = trainer_client.train(prompt, completion)
      logging.info(f"[trainer] weights: {weights} (metrics: {trainer_metrics})")

      global_step_time = time.time() - step_start_time

      # Centralized Aggregation & OpenTelemetry Double-Write Logging per design doc
      reward = rollout_metrics.get("reward", 0.0)
      rollout_time = rollout_metrics.get("rollout_time", 0.0)
      reward_calc_time = rollout_metrics.get("reward_calc_time", 0.0)

      loss = trainer_metrics.get("loss", 0.0)
      kl_div = trainer_metrics.get("kl_divergence", 0.0)
      grad_norm = trainer_metrics.get("grad_norm", 0.0)
      lr = trainer_metrics.get("learning_rate", 0.0)
      actor_train_time = trainer_metrics.get("actor_train_time", 0.0)

      # Log OTel & W&B metrics adhering to tunix metrics specification
      logger.log("rewards", "score", reward, mode="train", step=i)
      logger.log("train", "loss", loss, mode="train", step=i)
      logger.log("train", "kl_divergence", kl_div, mode="train", step=i)
      logger.log("train", "gradient_norm", grad_norm, mode="train", step=i)
      logger.log("train", "learning_rate", lr, mode="train", step=i)
      logger.log("perf", "global_step_time", global_step_time, mode="train", step=i)
      logger.log("perf", "rollout_time", rollout_time, mode="train", step=i)
      logger.log("perf", "reward_calc_time", reward_calc_time, mode="train", step=i)
      logger.log("perf", "actor_train_time", actor_train_time, mode="train", step=i)
  except KeyboardInterrupt:
    pass

  print("Press Ctrl+C to exit...")
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    pass
