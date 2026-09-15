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

"""Distributed process entry point for TrainerWorker.

Usage Examples:
  1. Local host execution (via LocalExecutor):
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor \
        --process_main=tunix.experimental.worker.trainer_worker_process.main \
        --worker_id=trainer-0 \
        --service_port=12346
    ```

  2. Remote host / Kubernetes execution (via yaml_generator.py):
    ```shell
    python -m tunix.experimental.distributed.deployment.yaml_generator \
        tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
        --jobset_name=trainer-worker \
        --tpu_slice=tpuv5e:2x2x4 \
        --worker_container_port=12346 \
        --worker_startup_command="python -m tunix.experimental.distributed.runtime.main --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=tunix.experimental.worker.trainer_worker_process.main --worker_id=trainer-0 --service_port=12346" \
        > deployment.yaml
    kubectl apply -f deployment.yaml
    ```
"""

import argparse
import asyncio
import pickle
from typing import Sequence

import portpicker
from tunix.experimental.common import datatypes
from tunix.experimental.common import import_utils
from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.metrics import metrics
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class _DefaultTrainer(abstract_trainer.AbstractTrainer):
  """Default stub trainer fallback."""

  def __init__(self, config=None):
    del config

  def with_loss_fn(self, loss_fn, has_aux=False):
    del loss_fn, has_aux
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn):
    del gen_model_input_fn
    return self

  def compile(self, dummy_data):
    del dummy_data

  def fwd_bwd(self, payload, **kwargs):
    del payload, kwargs

  def update(self, **kwargs):
    del kwargs
    return 1

  def eval_step(self, payload, **kwargs):
    del payload, kwargs

  def save_checkpoint(self, metadata, **kwargs):
    del metadata, kwargs

  def restore_checkpoint(self, **kwargs):
    del kwargs
    return {}

  def prepare_weight_sync(self, **kwargs):
    del kwargs

  def get_metrics(self):
    return metrics.MetricsBuffer(id="default")

  def close(self):
    pass


def main(
    argv: Sequence[str] | list[str], context: ProcessContext | None
) -> None:
  """Distributed process entry point for TrainerWorker."""
  parser = argparse.ArgumentParser(description="TrainerWorker process")
  parser.add_argument(
      "--worker_id",
      type=str,
      default="trainer_worker",
      help="Unique identifier for this worker.",
  )
  parser.add_argument(
      "--service_port",
      type=int,
      default=0,
      help="Port for gRPC remote execution server (0 to pick unused port).",
  )
  parser.add_argument(
      "--trainer_factory",
      type=str,
      default="",
      help="Fully qualified name of callable returning an AbstractTrainer.",
  )
  args = parser.parse_args(argv)

  if args.trainer_factory:
    trainer_factory = import_utils.import_symbol(args.trainer_factory)
  else:
    trainer_factory = lambda: _DefaultTrainer()

  worker = trainer_worker.TrainerWorker(
      trainer_factory=trainer_factory,
      worker_id=args.worker_id,
  )

  service_port = args.service_port or portpicker.pick_unused_port()

  async def run_server_async() -> None:
    server = remote_execution.GrpcRemoteExecutionServer(worker)
    await server.start_serving_async(service_port)

    if context and context.ipc and context.ipc.discovery:
      context.ipc.discovery.register(
          metadata=pickle.dumps({
              "service_type": "trainer",
              "service_port": service_port,
              "worker_id": args.worker_id,
          })
      )

    if server._server is not None:
      await server._server.wait_for_termination()

  asyncio.run(run_server_async())
