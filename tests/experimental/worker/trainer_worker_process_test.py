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

"""Unit tests for trainer_worker_process."""

import argparse
import asyncio
import os
import pickle
import sys
import threading
import time
from unittest import mock

from absl.testing import absltest
import grpc
from tunix.experimental.metrics import metrics
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker_process


class _CustomTrainer(abstract_trainer.AbstractTrainer):

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
    return 42

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
    return metrics.MetricsBuffer(id="custom_trainer_metrics")

  def close(self):
    pass


def _custom_trainer_factory():
  return _CustomTrainer()


def _get_test_module_fqn(symbol_name: str) -> str:
  test_dir = os.path.dirname(os.path.abspath(__file__))
  if test_dir not in sys.path:
    sys.path.insert(0, test_dir)
  module_name = os.path.splitext(os.path.basename(__file__))[0]
  return f"{module_name}.{symbol_name}"


class MockDiscovery:

  def __init__(self):
    self.registered_metadata = []

  def register(self, metadata: bytes):
    self.registered_metadata.append(pickle.loads(metadata))


class MockProcessContext:

  def __init__(self):
    self.ipc = argparse.Namespace(discovery=MockDiscovery())


class TrainerWorkerProcessTest(absltest.TestCase):

  def test_initialization(self):
    """Tests initialization of the trainer worker with custom factory."""
    context = MockProcessContext()
    factory_fqn = _get_test_module_fqn("_custom_trainer_factory")
    argv = [
        "--worker_id=test_trainer_init",
        "--service_port=0",
        f"--trainer_factory={factory_fqn}",
    ]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      trainer_worker_process.main(argv, context)

  def test_discovery(self):
    """Tests discovery metadata registration."""
    context = MockProcessContext()
    argv = ["--worker_id=test_trainer_disc", "--service_port=12345"]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      trainer_worker_process.main(argv, context)

    self.assertEqual(len(context.ipc.discovery.registered_metadata), 1)
    meta = context.ipc.discovery.registered_metadata[0]
    self.assertEqual(meta["service_type"], "trainer")
    self.assertEqual(meta["worker_id"], "test_trainer_disc")
    self.assertEqual(meta["service_port"], 12345)

  def test_api(self):
    """Tests worker API execution flow over live gRPC remote execution server."""
    context = MockProcessContext()
    factory_fqn = _get_test_module_fqn("_custom_trainer_factory")
    argv = [
        "--worker_id=test_trainer_flow",
        "--service_port=0",
        f"--trainer_factory={factory_fqn}",
    ]

    stop_event = threading.Event()

    async def mock_wait_for_termination(self):
      while not stop_event.is_set():
        await asyncio.sleep(0.01)

    with mock.patch.object(
        grpc.aio.Server, "wait_for_termination", mock_wait_for_termination
    ):
      thread = threading.Thread(
          target=trainer_worker_process.main,
          args=(argv, context),
          daemon=True,
      )
      thread.start()

      deadline = time.time() + 10.0
      while not context.ipc.discovery.registered_metadata:
        if time.time() > deadline:
          self.fail("Timed out waiting for server discovery registration.")
        time.sleep(0.05)

      meta = context.ipc.discovery.registered_metadata[0]
      port = meta["service_port"]

      handle = remote_execution.GrpcRemoteActorHandle(
          target_address=f"grpc://localhost:{port}"
      )
      # Fetch worker metadata and verify worker role.
      info = handle.submit("info")
      self.assertEqual(info.worker_id, "test_trainer_flow")
      self.assertIn("trainer", info.roles)

      # Initialize worker and underlying trainer state.
      init_resp = handle.submit("initialize")
      self.assertIsNotNone(init_resp)

      # Execute training update step.
      step_count = handle.submit("update")
      self.assertEqual(step_count, 42)

      # Retrieve buffered training metrics.
      metrics_buf = handle.submit("get_metrics")
      self.assertEqual(metrics_buf.id, "custom_trainer_metrics")

      stop_event.set()
      thread.join(timeout=5.0)


if __name__ == "__main__":
  absltest.main()
