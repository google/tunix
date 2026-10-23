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

"""Distributed process entry point for RolloutWorker.

Usage Examples:
  1. Local host execution (via LocalExecutor):
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor \
        --process_main=tunix.experimental.worker.rollout_worker_process.main \
        --worker_id=rollout-0 \
        --service_port=12347
    ```

  2. Remote host / Kubernetes execution (via yaml_generator.py):
    ```shell
    python -m tunix.experimental.distributed.deployment.yaml_generator \
        tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
        --jobset_name=rollout-worker \
        --tpu_slice=tpuv5e:2x2x4 \
        --worker_container_port=12347 \
        --worker_startup_command="python -m tunix.experimental.distributed.runtime.main --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=tunix.experimental.worker.rollout_worker_process.main --worker_id=rollout-0 --service_port=12347" \
        > deployment.yaml
    kubectl apply -f deployment.yaml
    ```
"""

import argparse
import asyncio
import pickle
from typing import Sequence

import portpicker
from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker


def main(
    argv: Sequence[str] | list[str], context: ProcessContext | None
) -> None:
  """Distributed process entry point for RolloutWorker."""
  parser = argparse.ArgumentParser(description="RolloutWorker process")
  parser.add_argument(
      "--worker_id",
      type=str,
      default="rollout_worker",
      help="Unique identifier for this worker.",
  )
  parser.add_argument(
      "--service_port",
      type=int,
      default=0,
      help="Port for gRPC remote execution server (0 to pick unused port).",
  )
  args = parser.parse_args(argv)

  worker = rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
  )

  service_port = args.service_port or portpicker.pick_unused_port()

  async def run_server_async() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker)
    await server.start_serving_async(service_port)

    if context and context.ipc and context.ipc.discovery:
      context.ipc.discovery.register(
          metadata=pickle.dumps({
              "service_type": "rollout",
              "service_port": service_port,
              "worker_id": args.worker_id,
          })
      )

    if server._server is not None:
      await server._server.wait_for_termination()

  asyncio.run(run_server_async())
