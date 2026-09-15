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

"""Unit tests for inference_worker_process."""

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
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.worker import inference_worker_process
from tunix.experimental.worker import remote_execution


class _CustomCore:

  def get_ref_per_token_logps(
      self, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0
  ):
    del prompt_tokens, pad_id, eos_id, temperature
    return np.ones(completion_tokens.shape, dtype=np.float32)

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del completion_tokens, pad_id, eos_id
    return np.ones((prompt_tokens.shape[0],), dtype=np.float32)


def _custom_core_factory():
  return _CustomCore()


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


class InferenceWorkerProcessTest(absltest.TestCase):

  def test_initialization(self):
    """Tests initialization of the worker with custom core factory and CLI args."""
    context = MockProcessContext()
    factory_fqn = _get_test_module_fqn("_custom_core_factory")
    argv = [
        "--worker_id=test_inference_init",
        "--service_port=0",
        f"--core_factory={factory_fqn}",
        "--pad_id=10",
        "--eos_id=11",
        "--model_version=2",
        "--chunk_size=16",
    ]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      inference_worker_process.main(argv, context)

  def test_discovery(self):
    """Tests discovery metadata registration."""
    context = MockProcessContext()
    argv = ["--worker_id=test_inference_disc", "--service_port=12345"]
    with mock.patch.object(
        remote_execution.GrpcRemoteExecutionServer,
        "start_serving_async",
        new_callable=mock.AsyncMock,
    ):
      inference_worker_process.main(argv, context)

    self.assertEqual(len(context.ipc.discovery.registered_metadata), 1)
    meta = context.ipc.discovery.registered_metadata[0]
    self.assertEqual(meta["service_type"], "inference")
    self.assertEqual(meta["worker_id"], "test_inference_disc")
    self.assertEqual(meta["service_port"], 12345)

  def test_api(self):
    """Tests worker API execution flow over live gRPC remote execution server."""
    context = MockProcessContext()
    factory_fqn = _get_test_module_fqn("_custom_core_factory")
    argv = [
        "--worker_id=test_inference_flow",
        "--service_port=0",
        f"--core_factory={factory_fqn}",
    ]

    stop_event = threading.Event()

    async def mock_wait_for_termination(self):
      while not stop_event.is_set():
        await asyncio.sleep(0.01)

    with mock.patch.object(
        grpc.aio.Server, "wait_for_termination", mock_wait_for_termination
    ):
      thread = threading.Thread(
          target=inference_worker_process.main,
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
      self.assertEqual(info.worker_id, "test_inference_flow")
      self.assertIn("inference", info.roles)

      # Initialize worker state.
      init_resp = handle.submit("initialize")
      self.assertIsNotNone(init_resp)

      # Compute per-token reference model log-probabilities.
      req = datatypes.LogprobsRequest(
          request_id="req_101",
          model_role="reference",
          prompt_tokens=np.array([[1, 2, 3]], dtype=np.int32),
          completion_tokens=np.array([[4, 5]], dtype=np.int32),
          temperature=1.0,
      )
      resp = handle.submit("compute_logps", req)
      self.assertEqual(resp.request_id, "req_101")
      self.assertEqual(resp.per_token_logps.shape, (1, 2))

      # Compute scalar reward scores.
      score_req = datatypes.ScoreRequest(
          request_id="req_102",
          model_role="reward",
          prompt_tokens=np.array([[1, 2, 3]], dtype=np.int32),
          completion_tokens=np.array([[4, 5]], dtype=np.int32),
      )
      score_resp = handle.submit("score", score_req)
      self.assertEqual(score_resp.request_id, "req_102")
      self.assertEqual(score_resp.scores.shape, (1,))

      stop_event.set()
      thread.join(timeout=5.0)


if __name__ == "__main__":
  absltest.main()
