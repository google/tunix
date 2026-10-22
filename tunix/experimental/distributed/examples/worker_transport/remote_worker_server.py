# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone remote worker process for worker_transport demo."""

import argparse
import asyncio
import pickle
from typing import Sequence

from absl import app, flags
from tunix.experimental.distributed.examples.worker_transport.worker import Worker
from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.distributed.runtime.contexts.local_context import LocalProcessContext
from tunix.experimental.worker import remote_execution

FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 12345, "Port for remote worker gRPC server.")
flags.DEFINE_string("discovery_id", "remote_worker", "Discovery ID.")
flags.DEFINE_integer("discovery_port", 0, "Discovery port.")
flags.DEFINE_string("discovery_addrs", "", "Discovery server addresses.")


async def run_server_async(context: ProcessContext | None) -> None:
  """Starts the gRPC worker server, registers with discovery after it starts, and waits for termination."""
  worker = Worker("remote")
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(FLAGS.port)

  if context and context.ipc and context.ipc.discovery:
    context.ipc.discovery.register(
        metadata=pickle.dumps({"service_port": FLAGS.port})
    )

  if server._server is not None:
    await server._server.wait_for_termination()


def main(argv: Sequence[str], context: ProcessContext | None) -> None:
  """Distributed process entry point for the remote worker."""
  del argv
  if context is None and FLAGS.discovery_addrs:
    args = argparse.Namespace(
        discovery_id=FLAGS.discovery_id,
        discovery_port=FLAGS.discovery_port or 0,
        discovery_addrs=FLAGS.discovery_addrs,
    )
    context = LocalProcessContext(args)
    context.__enter__()

  asyncio.run(run_server_async(context))


if __name__ == "__main__":
  app.run(lambda argv: main(argv, context=None))
