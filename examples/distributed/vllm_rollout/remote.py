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

"""A standalone demo script for testing gRPC remote execution in Tunix."""

import asyncio
import logging
from typing import Sequence

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.worker import remote_execution


async def server_main(done: asyncio.Future) -> None:
  """Starts a gRPC remote execution server with a dummy worker.

  Args:
    done: An asyncio Future used to signal when the client has finished its work
      and the server should shut down.
  """

  class Worker:
    """A simple worker class demonstrating remote method invocation."""

    def get_worker_id(self) -> str:
      """Returns a dummy worker ID."""
      return "hello"

  worker = Worker()
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(12345)
  await done
  await server.stop_serving(5)


async def client_main(done: asyncio.Future) -> None:
  """Connects to the remote worker, invokes a method, and signals completion.

  Args:
    done: An asyncio Future set by this client once remote execution finishes.
  """
  client = remote_execution.ActorHandle.from_address("grpc://localhost:12345")

  worker_id = await client.asubmit("get_worker_id")
  logging.info("received %s", f"{worker_id=}")

  done.set_result(None)


async def run_loop() -> None:
  """Runs the server and client concurrently until execution completes."""
  loop = asyncio.get_running_loop()

  done = loop.create_future()
  server_task = loop.create_task(server_main(done))
  client_task = loop.create_task(client_main(done))

  await asyncio.gather(server_task, client_task)


def main(argv: Sequence[str], context: ProcessContext | None) -> None:
  """Entry point for running the remote execution demo.

  Args:
    argv: Command-line arguments passed to the script.
    context: Optional runtime context provided by the distributed runtime.
  """
  del argv, context  # Unused in this standalone demo.
  asyncio.run(run_loop())
