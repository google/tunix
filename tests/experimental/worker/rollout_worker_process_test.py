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

"""Unit tests for rollout_worker_process."""

import argparse
import asyncio
import pickle
import threading
import time
from unittest import mock

from absl.testing import absltest
import grpc
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker_process


class MockDiscovery:

  def __init__(self):
    self.registered_metadata = []

  def register(self, metadata: bytes):
    self.registered_metadata.append(pickle.loads(metadata))


class MockProcessContext:

  def __init__(self):
    self.ipc = argparse.Namespace(discovery=MockDiscovery())


class RolloutWorkerProcessTest(absltest.TestCase):

  def test_initialization(self):
    """Tests initialization of the rollout worker process."""
    context = MockProcessContext()
    argv = ["--worker_id=test_rollout_init", "--service_port=0"]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      rollout_worker_process.main(argv, context)

  def test_discovery(self):
    """Tests discovery metadata registration."""
    context = MockProcessContext()
    argv = ["--worker_id=test_rollout_disc", "--service_port=12345"]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      rollout_worker_process.main(argv, context)

    self.assertEqual(len(context.ipc.discovery.registered_metadata), 1)
    meta = context.ipc.discovery.registered_metadata[0]
    self.assertEqual(meta["service_type"], "rollout")
    self.assertEqual(meta["worker_id"], "test_rollout_disc")
    self.assertEqual(meta["service_port"], 12345)

  def test_api(self):
    """Tests worker API execution flow over live gRPC remote execution server."""
    context = MockProcessContext()
    argv = ["--worker_id=test_rollout_flow", "--service_port=0"]

    stop_event = threading.Event()

    async def mock_wait_for_termination(self):
      while not stop_event.is_set():
        await asyncio.sleep(0.01)

    with mock.patch.object(
        grpc.aio.Server, "wait_for_termination", mock_wait_for_termination
    ):
      thread = threading.Thread(
          target=rollout_worker_process.main,
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
      # Retrieve unique worker identifier.
      worker_id = handle.submit("get_worker_id")
      self.assertEqual(worker_id, "test_rollout_flow")

      # Fetch worker metadata and verify worker role.
      info = handle.submit("info")
      self.assertEqual(info.worker_id, "test_rollout_flow")
      self.assertIn("rollout", info.roles)

      # Initialize rollout worker state.
      init_resp = handle.submit("initialize")
      self.assertIsNotNone(init_resp)

      # Query worker health report via heartbeat.
      hb = handle.submit("heartbeat")
      self.assertIsNotNone(hb)

      stop_event.set()
      thread.join(timeout=5.0)


if __name__ == "__main__":
  absltest.main()
