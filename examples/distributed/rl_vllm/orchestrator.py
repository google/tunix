import argparse
import asyncio
import dataclasses
import logging
import pickle
import queue
import random
import time
from typing import List

from tunix.experimental.distributed.runtime.context import ProcessContext

from tunix.experimental.orchestrator import driver
from tunix.experimental.rollout import data_types
from tunix.experimental.worker import remote_execution


@dataclasses.dataclass(frozen=True)
class RolloutSpec:
  worker_id: str
  address: str
  model_name: str


async def orchestrator_main(rollout_specs: List[RolloutSpec]):
  orchestrator = driver.GlobalOrchestrator(
      orchestrator_id="orchestrator-vllm",
      rollout_actors=[spec.address for spec in rollout_specs],
  )

  for _ in range(3):
    index = random.randrange(len(rollout_specs))

    rollout_spec = rollout_specs[index]
    rollout_handle = orchestrator.actor_handles[index]

    worker_id = await rollout_handle.asubmit("get_worker_id")
    assert worker_id == rollout_spec.worker_id

    request = data_types.RequestInput(
      prompt_id="long_q1",
      prompt=(
        "What is 123 + 456? Please explain each step in detail and verify"
        " the result."
      ),
      generation_kwargs={
        "max_tokens": 128,
        "temperature": 0.0,
        "model": rollout_spec.model_name,
      },
      max_turns=2,
      metadata={"system_prompt": "You are a helpful mathematical assistant."},
    )
    logging.info(f"submitting request to rollout worker {worker_id}: {request}")
    response = await rollout_handle.asubmit("generate", request)
    logging.info(f"received response from rollout worker {worker_id}: {response}")


def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--min_rollout_workers", type=int, default=1, help="")
  args = parser.parse_args(argv)

  # setup discovery for workers
  rollout_spec_queue = queue.Queue()
  def accept_worker(hostname: str, _: int, metadata: bytes) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    service_address = f"grpc://{hostname}:{md["service_port"]}"
    worker_id = md["worker_id"]
    model_name = md["model_name"]

    logging.info(f"discovered {service_type} service {worker_id} at {service_address}")

    match service_type:
      case "rollout":
        rollout_spec = RolloutSpec(
          worker_id=worker_id,
          address=service_address,
          model_name=model_name,
        )
        rollout_spec_queue.put(rollout_spec)
      case _:
        raise RuntimeError(f"unknown service type {service_type}")
  context.ipc.discovery.on_register(accept_worker)

  # wait at least two rollout clients
  have_workers = -1
  while rollout_spec_queue.qsize() < args.min_rollout_workers:
    if have_workers < rollout_spec_queue.qsize():
      have_workers = rollout_spec_queue.qsize()
      logging.info(f"waiting for rollout workers (need: {args.min_rollout_workers}, have: {rollout_spec_queue.qsize()})")
    time.sleep(1)
  logging.info(f"all {args.min_rollout_workers} rollout workers ready")

  # start main loop
  asyncio.run(orchestrator_main(list(rollout_spec_queue.queue)))
