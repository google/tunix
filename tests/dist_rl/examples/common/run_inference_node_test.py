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

"""Unit tests for reference inference node runner (`run_inference_node.py`)."""

import asyncio
import contextlib
import os
import signal
from unittest import mock

from absl.testing import absltest
import transformers
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.examples.common import run_inference_node
from tunix.dist_rl.worker import inference_worker as exp_inference_worker
from tunix.dist_rl.worker import remote_execution
from tunix.models import automodel
from tunix.rl.inference import inference_worker as rl_inference_worker


class RunInferenceNodeTest(absltest.TestCase):

  def _setup_mocks(self):
    context = mock.MagicMock()
    mock_tokenizer = mock.MagicMock(
        pad_token_id=0, eos_token_id=1, eos_token="<eos>"
    )
    mock_mesh = mock.MagicMock()
    mock_worker_service = mock.MagicMock(
        spec=exp_inference_worker.InferenceWorker
    )
    registered_callbacks = []
    mock_worker_service.set_on_stop.side_effect = registered_callbacks.append

    mock_server = mock.MagicMock(
        spec=remote_execution.GrpcRemoteExecutionServer
    )
    mock_server.start_serving_async = mock.AsyncMock()
    mock_server.stop_serving = mock.AsyncMock()
    return (
        context,
        mock_tokenizer,
        mock_mesh,
        mock_worker_service,
        registered_callbacks,
        mock_server,
    )

  @contextlib.contextmanager
  def _patched_runtime(
      self, mock_tokenizer, mock_mesh, mock_worker_service, mock_server
  ):
    with (
        mock.patch.object(otel_setup, "setup_metrics"),
        mock.patch.object(
            run_inference_node,
            "_ensure_model_dir",
            return_value="/tmp/fake_model",
        ),
        mock.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            return_value=mock_tokenizer,
        ),
        mock.patch.object(
            run_inference_node, "_create_mesh", return_value=mock_mesh
        ),
        mock.patch.object(
            automodel.AutoModel,
            "from_pretrained",
            return_value=(mock.MagicMock(), None),
        ),
        mock.patch.object(rl_inference_worker, "InferenceWorker"),
        mock.patch.object(
            exp_inference_worker,
            "InferenceWorker",
            return_value=mock_worker_service,
        ),
        mock.patch.object(
            remote_execution,
            "GrpcRemoteExecutionServer",
            return_value=mock_server,
        ),
    ):
      yield

  def test_rpc_stop_callback_drains_in_order_and_exits_nonzero_on_abort(self):
    (
        context,
        mock_tokenizer,
        mock_mesh,
        mock_worker_service,
        registered_callbacks,
        mock_server,
    ) = self._setup_mocks()
    shutdown_order: list[str] = []

    async def _stop_serving(grace: float = 0.5):
      self.assertEqual(grace, 5.0)
      shutdown_order.append("server.stop_serving")

    def _stop_worker():
      shutdown_order.append("worker_service.stop")

    mock_server.stop_serving = mock.AsyncMock(side_effect=_stop_serving)
    mock_worker_service.stop.side_effect = _stop_worker

    with self._patched_runtime(
        mock_tokenizer, mock_mesh, mock_worker_service, mock_server
    ):

      def _register_and_clean_stop(*, metadata):
        del metadata
        self.assertTrue(callable(registered_callbacks[0]))
        registered_callbacks[0](False)

      context.ipc.discovery.register.side_effect = _register_and_clean_stop
      run_inference_node.main(
          ["--port=20002", "--worker_id=inference-0"],
          context=context,
      )
      self.assertIsNone(registered_callbacks[-1])
      self.assertEqual(
          shutdown_order, ["server.stop_serving", "worker_service.stop"]
      )
      mock_server.stop_serving.assert_awaited_once_with(grace=5.0)

      shutdown_order.clear()
      mock_server.stop_serving.reset_mock()
      mock_worker_service.stop.reset_mock()
      registered_callbacks.clear()

      def _register_and_abort_stop(*, metadata):
        del metadata
        self.assertTrue(callable(registered_callbacks[0]))
        registered_callbacks[0](True)

      context.ipc.discovery.register.side_effect = _register_and_abort_stop
      with self.assertRaises(SystemExit) as cm:
        run_inference_node.main(
            ["--port=20002", "--worker_id=inference-0"],
            context=context,
        )
      self.assertEqual(cm.exception.code, 1)
      self.assertIsNone(registered_callbacks[-1])
      self.assertEqual(
          shutdown_order, ["server.stop_serving", "worker_service.stop"]
      )
      mock_server.stop_serving.assert_awaited_once_with(grace=5.0)

  def test_sigterm_shutdown_drains_worker_even_if_stop_serving_raises(self):
    (
        context,
        mock_tokenizer,
        mock_mesh,
        mock_worker_service,
        _,
        mock_server,
    ) = self._setup_mocks()
    mock_server.stop_serving = mock.AsyncMock(
        side_effect=RuntimeError("gRPC stop_serving failed")
    )

    def _schedule_sigterm(*, metadata):
      del metadata
      asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)

    context.ipc.discovery.register.side_effect = _schedule_sigterm

    with self._patched_runtime(
        mock_tokenizer, mock_mesh, mock_worker_service, mock_server
    ):
      run_inference_node.main(
          ["--port=20002", "--worker_id=inference-0"],
          context=context,
      )

    mock_server.stop_serving.assert_awaited_once_with(grace=5.0)
    mock_worker_service.stop.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
