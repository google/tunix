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

"""Distributed process entry point for InferenceWorker.

Usage Examples:
  1. Local host execution (via LocalExecutor):
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor \
        --process_main=tunix.experimental.worker.inference_worker_process.main \
        --worker_id=inference-0 \
        --service_port=12345
    ```

  2. Remote host / Kubernetes execution (via yaml_generator.py):
    ```shell
    python -m tunix.experimental.distributed.deployment.yaml_generator \
        tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
        --jobset_name=inference-worker \
        --tpu_slice=tpuv5e:2x2x4 \
        --worker_container_port=12345 \
        --worker_startup_command="python -m tunix.experimental.distributed.runtime.main --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=tunix.experimental.worker.inference_worker_process.main --worker_id=inference-0 --service_port=12345" \
        > deployment.yaml
    kubectl apply -f deployment.yaml
    ```
"""

import argparse
import asyncio
import pickle
from typing import Sequence

import jax.numpy as jnp
import portpicker
from tunix.experimental.common import import_utils
from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.worker import inference_worker
from tunix.experimental.worker import remote_execution


class _DefaultCore:
  """Default reference scoring core fallback."""

  def get_ref_per_token_logps(
      self, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0
  ):
    del prompt_tokens, pad_id, eos_id, temperature
    return jnp.zeros_like(completion_tokens, dtype=jnp.float32)

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del completion_tokens, pad_id, eos_id
    return jnp.zeros((prompt_tokens.shape[0],), dtype=jnp.float32)


def main(
    argv: Sequence[str] | list[str], context: ProcessContext | None
) -> None:
  """Distributed process entry point for InferenceWorker."""
  parser = argparse.ArgumentParser(description="InferenceWorker process")
  parser.add_argument(
      "--worker_id",
      type=str,
      default="inference_worker",
      help="Unique identifier for this worker.",
  )
  parser.add_argument(
      "--service_port",
      type=int,
      default=0,
      help="Port for gRPC remote execution server (0 to pick unused port).",
  )
  parser.add_argument(
      "--pad_id",
      type=int,
      default=0,
      help="Padding token ID.",
  )
  parser.add_argument(
      "--eos_id",
      type=int,
      default=1,
      help="End of sequence token ID.",
  )
  parser.add_argument(
      "--model_version",
      type=int,
      default=0,
      help="Version tag for the hosted weights.",
  )
  parser.add_argument(
      "--chunk_size",
      type=int,
      default=0,
      help="Maximum batch size for scoring (0 or negative means None).",
  )
  parser.add_argument(
      "--core_factory",
      type=str,
      default="",
      help="Fully qualified name of callable returning a ReferenceScoringCore.",
  )
  args = parser.parse_args(argv)

  if args.core_factory:
    core = import_utils.import_symbol(args.core_factory)()
  else:
    core = _DefaultCore()

  chunk_size = args.chunk_size if args.chunk_size > 0 else None

  worker = inference_worker.InferenceWorker(
      core=core,
      worker_id=args.worker_id,
      pad_id=args.pad_id,
      eos_id=args.eos_id,
      model_version=args.model_version,
      chunk_size=chunk_size,
  )

  service_port = args.service_port or portpicker.pick_unused_port()

  async def run_server_async() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker)
    await server.start_serving_async(service_port)

    if context and context.ipc and context.ipc.discovery:
      context.ipc.discovery.register(
          metadata=pickle.dumps({
              "service_type": "inference",
              "service_port": service_port,
              "worker_id": args.worker_id,
          })
      )

    if server._server is not None:
      await server._server.wait_for_termination()

  asyncio.run(run_server_async())
