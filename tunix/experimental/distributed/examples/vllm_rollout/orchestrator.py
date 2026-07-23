# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator script for distributed RL generation with vLLM workers."""

import argparse
import concurrent.futures
import dataclasses
import logging
import pickle
import queue
import random
import threading
import time
from typing import Any, Sequence

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.rollout import data_types
from tunix.experimental.worker import remote_execution


@dataclasses.dataclass(frozen=True)
class RolloutSpec:
  """Specification and metadata for a discovered rollout worker.

  Attributes:
    worker_id: Unique identifier for the rollout worker.
    address: gRPC service address of the worker.
    model_name: Name of the model being served by the worker.
  """

  worker_id: str
  address: str
  model_name: str


class RequestStatsMonitor:
  """Tracks request statistics and periodically logs summary reports.

  Attributes:
    total_requests: Total number of requests recorded across all workers since
      initialization.
  """

  def __init__(self, log_interval_seconds: float = 10.0) -> None:
    """Initializes the stats monitor.

    Args:
      log_interval_seconds: Time interval in seconds between summary logs.
    """
    self._lock = threading.Lock()
    self._log_interval_seconds = log_interval_seconds
    self._last_log_time = time.monotonic()
    self._total_requests = 0
    self._interval_requests = 0
    self._worker_counts = {}

  @property
  def total_requests(self) -> int:
    """Returns the total requests recorded since startup."""
    with self._lock:
      return self._total_requests

  def record_request(self, worker_id: str) -> None:
    """Records a single completed request for the specified worker.

    Args:
      worker_id: The ID of the worker that processed the request.
    """
    with self._lock:
      self._total_requests += 1
      self._interval_requests += 1
      self._worker_counts[worker_id] = self._worker_counts.get(worker_id, 0) + 1

  def maybe_log_summary(
      self, sample_request: Any, sample_response: Any
  ) -> None:
    """Logs a summary of request statistics if the log interval has elapsed.

    Args:
      sample_request: A sample request object to display in the log summary.
      sample_response: A sample response object to display in the log summary.
    """
    with self._lock:
      current_time = time.monotonic()
      elapsed_since_log = current_time - self._last_log_time
      if elapsed_since_log >= self._log_interval_seconds:
        rps = self._interval_requests / elapsed_since_log
        worker_stats_str = ", ".join(
            f"{wid}={cnt}" for wid, cnt in sorted(self._worker_counts.items())
        )
        logging.info(
            "\n\n--- %d Seconds Summary ---\n"
            "\nRequests per second (RPS): %.2f\n"
            "\nWorker Request Counts: %s\n"
            "\nSample Request: %s\n"
            "\nSample Response: %s\n"
            "\n------------------------\n",
            self._log_interval_seconds,
            rps,
            worker_stats_str,
            sample_request.prompt,
            sample_response.turns[0].action.split("\n")[0] + "...",
        )
        self._interval_requests = 0
        self._worker_counts.clear()
        self._last_log_time = current_time


def orchestrator_main(
    rollout_spec_queue: queue.Queue, parallelism: int
) -> None:
  """Runs the main orchestrator loop to submit generation requests to workers.

  Args:
    rollout_spec_queue: Thread-safe queue containing connected rollout worker
      specifications.
    parallelism: Number of concurrent generation requests to submit using a
      thread pool.
  """
  rollout_handles: dict[str, remote_execution.ActorHandle] = {}
  worker_lock = threading.Lock()

  def pick_rollout_worker() -> tuple[RolloutSpec, remote_execution.ActorHandle]:
    """Randomly selects a rollout worker and its remote actor handle."""
    with worker_lock:
      rollout_specs = list(rollout_spec_queue.queue)
      assert len(rollout_specs) > 0
      index = random.randrange(len(rollout_specs))
      rollout_spec = rollout_specs[index]
      if rollout_spec.worker_id not in rollout_handles:
        rollout_handles[rollout_spec.worker_id] = (
            remote_execution.ActorHandle.from_address(rollout_spec.address)
        )
      return rollout_spec, rollout_handles[rollout_spec.worker_id]

  logging.info(
      "starting generation loop with parallelism=%d...",
      parallelism,
  )

  stats = RequestStatsMonitor()

  def generation_task(request_id: int) -> None:
    rollout_spec, rollout_handle = pick_rollout_worker()
    request = data_types.RequestInput(
        prompt_id=f"long_q_{request_id}",
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
    response = rollout_handle.submit("generate", request)
    stats.record_request(rollout_spec.worker_id)
    stats.maybe_log_summary(request, response)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=parallelism
  ) as executor:
    in_flight = set()
    request_id = 0
    while True:
      while len(in_flight) < parallelism:
        future = executor.submit(generation_task, request_id)
        in_flight.add(future)
        request_id += 1

      done, in_flight = concurrent.futures.wait(
          in_flight, return_when=concurrent.futures.FIRST_COMPLETED
      )
      for future in done:
        try:
          future.result()
        except Exception as e:
          logging.error("Generation request failed: %s", e)


def main(argv: Sequence[str], context: ProcessContext | None) -> None:
  """Main entry point for the orchestrator service.

  Args:
    argv: Command-line arguments.
    context: Process context for IPC discovery and distributed communication.
  """
  # Step 1: Parse command-line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--min_rollout_workers",
      type=int,
      default=1,
      help="Minimum number of rollout workers required before starting.",
  )
  parser.add_argument(
      "--parallelism",
      type=int,
      default=64,
      help="Number of concurrent generation requests sent to rollout workers.",
  )
  args = parser.parse_args(argv)

  # Step 2: Configure IPC discovery and register the worker acceptance callback.
  rollout_spec_queue = queue.Queue()

  def accept_worker(hostname: str, _: int, metadata: bytes) -> None:
    md = pickle.loads(metadata)

    service_type = md["service_type"]
    service_address = f"grpc://{hostname}:{md['service_port']}"
    worker_id = md["worker_id"]
    model_name = md["model_name"]

    logging.info(
        "discovered %s service %s at %s",
        service_type,
        worker_id,
        service_address,
    )

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

  if context and context.ipc and context.ipc.discovery:
    context.ipc.discovery.on_register(accept_worker)

  # Step 3: Wait for the minimum required number of rollout workers to connect.
  have_workers = -1
  while rollout_spec_queue.qsize() < args.min_rollout_workers:
    if have_workers < rollout_spec_queue.qsize():
      have_workers = rollout_spec_queue.qsize()
      logging.info(
          "waiting for rollout workers (minimum: %d, have: %d)",
          args.min_rollout_workers,
          rollout_spec_queue.qsize(),
      )
    time.sleep(1)
  logging.info("all %d rollout workers ready", args.min_rollout_workers)

  # Step 4: Start the orchestrator main loop with discovered workers.
  orchestrator_main(rollout_spec_queue, args.parallelism)
