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

"""Unit tests for run_trainer_node.py."""

import asyncio
import contextlib
from pathlib import Path
import pickle
import signal
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax
from jax.sharding import Mesh
from tunix.experimental.examples.math_gsm8k_dist import run_trainer_node
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class MeshBoundTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_trainer = mock.MagicMock(spec=peft_trainer_v2.PeftTrainer)
    self.mock_mesh = mock.MagicMock(spec=Mesh)
    self.mesh_trainer = run_trainer_node._MeshBoundTrainer(
        self.mock_trainer, self.mock_mesh
    )

  def test_save_checkpoint_runs_within_mesh_context_and_propagates_kwargs(self):
    call_order = []
    self.mock_mesh.__enter__.side_effect = (
        lambda: call_order.append("enter_mesh")
    )
    self.mock_mesh.__exit__.side_effect = (
        lambda *args: call_order.append("exit_mesh")
    )
    self.mock_trainer.save_checkpoint.side_effect = (
        lambda *args, **kwargs: call_order.append("save_checkpoint")
    )

    metadata = {"step": 10, "policy_version": 2, "num_rollouts": 4}
    self.mesh_trainer.save_checkpoint(
        metadata=metadata, force=True, save_only_lora_params=True
    )

    self.mock_trainer.save_checkpoint.assert_called_once_with(
        metadata, force=True, save_only_lora_params=True
    )
    self.assertEqual(
        call_order, ["enter_mesh", "save_checkpoint", "exit_mesh"]
    )

  def test_save_checkpoint_default_metadata(self):
    self.mesh_trainer.save_checkpoint()
    self.mock_trainer.save_checkpoint.assert_called_once_with(None)

  def test_save_checkpoint_propagates_exception_and_exits_mesh(self):
    self.mock_trainer.save_checkpoint.side_effect = RuntimeError("Disk full")
    with self.assertRaisesRegex(RuntimeError, "Disk full"):
      self.mesh_trainer.save_checkpoint(metadata={"step": 1})

    self.mock_mesh.__enter__.assert_called_once()
    self.mock_mesh.__exit__.assert_called_once()

  def test_fwd_bwd_runs_within_mesh(self):
    self.mesh_trainer.fwd_bwd("payload", skip_jit=False)
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.fwd_bwd.assert_called_once_with(
        "payload", skip_jit=False
    )
    self.mock_mesh.__exit__.assert_called_once()

  def test_update_runs_within_mesh(self):
    self.mock_trainer.update.return_value = 5
    result = self.mesh_trainer.update(custom_kw=True)
    self.assertEqual(result, 5)
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.update.assert_called_once_with(custom_kw=True)
    self.mock_mesh.__exit__.assert_called_once()

  def test_eval_step_runs_within_mesh(self):
    self.mesh_trainer.eval_step("eval_payload", arg1=1)
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.eval_step.assert_called_once_with(
        "eval_payload", arg1=1
    )
    self.mock_mesh.__exit__.assert_called_once()

  def test_eval_context_runs_within_mesh(self):
    mock_ctx = mock.MagicMock()
    self.mock_trainer.eval_context.return_value = mock_ctx

    with self.mesh_trainer.eval_context():
      mock_ctx.__enter__.assert_called_once()

    mock_ctx.__exit__.assert_called_once()
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_mesh.__exit__.assert_called_once()

  def test_compile_runs_within_mesh(self):
    self.mesh_trainer.compile("dummy_data")
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.compile.assert_called_once_with("dummy_data")
    self.mock_mesh.__exit__.assert_called_once()

  def test_prepare_weight_sync_runs_within_mesh(self):
    self.mock_trainer.prepare_weight_sync.return_value = {"weights": "synced"}
    result = self.mesh_trainer.prepare_weight_sync(sync_request="req1")
    self.assertEqual(result, {"weights": "synced"})
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.prepare_weight_sync.assert_called_once_with(
        sync_request="req1"
    )
    self.mock_mesh.__exit__.assert_called_once()

  def test_close_runs_within_mesh(self):
    self.mesh_trainer.close()
    self.mock_mesh.__enter__.assert_called_once()
    self.mock_trainer.close.assert_called_once()
    self.mock_mesh.__exit__.assert_called_once()

  def test_getattr_delegates_to_underlying_trainer(self):
    self.mock_trainer.custom_attr = "custom_value"
    self.assertEqual(self.mesh_trainer.custom_attr, "custom_value")


class RunTrainerNodeMainAndShutdownTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_context = mock.MagicMock()
    self.mock_context.ipc.discovery.register = mock.MagicMock()
    self.mock_context.jax.initialize = mock.MagicMock()

    self.mock_worker_service = mock.MagicMock(spec=trainer_worker.TrainerWorker)
    self.mock_server = mock.MagicMock(
        spec=remote_execution.GrpcRemoteExecutionServer
    )
    self.mock_server.start_serving_async = mock.AsyncMock()
    self.mock_server.stop_serving = mock.AsyncMock()

  def _setup_main_mocks(
      self,
      mock_ensure_model_dir,
      mock_create_mesh,
      mock_load_actor_model,
      mock_create_trainer_factory,
      mock_trainer_worker_cls,
      mock_grpc_server_cls,
  ):
    mock_ensure_model_dir.return_value = "/tmp/mock_model_dir"
    mock_mesh = mock.MagicMock(spec=Mesh)
    mock_create_mesh.return_value = mock_mesh
    mock_load_actor_model.return_value = mock.MagicMock()
    mock_create_trainer_factory.return_value = mock.MagicMock()
    mock_trainer_worker_cls.return_value = self.mock_worker_service
    mock_grpc_server_cls.return_value = self.mock_server

  def _patch_signal_handlers(self, handler_fn):
    loop = asyncio.new_event_loop()
    loop_cls = type(loop)
    loop.close()
    classes_to_patch = {
        asyncio.AbstractEventLoop,
        asyncio.BaseEventLoop,
        loop_cls,
    }
    stack = contextlib.ExitStack()
    for cls in classes_to_patch:
      if hasattr(cls, "add_signal_handler"):
        stack.enter_context(
            mock.patch.object(cls, "add_signal_handler", new=handler_fn)
        )
    return stack

  @mock.patch.object(run_trainer_node, "_ensure_model_dir_for_trainer")
  @mock.patch.object(run_trainer_node, "_create_mesh")
  @mock.patch.object(run_trainer_node, "_load_actor_model")
  @mock.patch.object(run_trainer_node, "_create_trainer_factory")
  @mock.patch.object(trainer_worker, "TrainerWorker")
  @mock.patch.object(remote_execution, "GrpcRemoteExecutionServer")
  def test_shutdown_handler_drains_worker_on_sigterm(
      self,
      mock_grpc_server_cls,
      mock_trainer_worker_cls,
      mock_create_trainer_factory,
      mock_load_actor_model,
      mock_create_mesh,
      mock_ensure_model_dir,
  ):
    self._setup_main_mocks(
        mock_ensure_model_dir,
        mock_create_mesh,
        mock_load_actor_model,
        mock_create_trainer_factory,
        mock_trainer_worker_cls,
        mock_grpc_server_cls,
    )

    signal_handlers = {}

    def mock_add_signal_handler(loop_self, sig, callback):
      signal_handlers[sig] = callback
      if sig == signal.SIGTERM:
        loop_self.call_soon(callback)

    with self._patch_signal_handlers(mock_add_signal_handler):
      run_trainer_node.main(
          ["--port", "20000", "--worker_id", "trainer-0"],
          context=self.mock_context,
      )

    self.assertIn(signal.SIGINT, signal_handlers)
    self.assertIn(signal.SIGTERM, signal_handlers)
    self.mock_context.jax.initialize.assert_called_once()
    self.mock_server.start_serving_async.assert_called_once_with(20000)
    self.mock_context.ipc.discovery.register.assert_called_once()
    self.mock_worker_service.stop.assert_called_once()
    self.mock_server.stop_serving.assert_called_once()

  @mock.patch.object(run_trainer_node, "_ensure_model_dir_for_trainer")
  @mock.patch.object(run_trainer_node, "_create_mesh")
  @mock.patch.object(run_trainer_node, "_load_actor_model")
  @mock.patch.object(run_trainer_node, "_create_trainer_factory")
  @mock.patch.object(trainer_worker, "TrainerWorker")
  @mock.patch.object(remote_execution, "GrpcRemoteExecutionServer")
  def test_shutdown_handler_drains_worker_on_sigint(
      self,
      mock_grpc_server_cls,
      mock_trainer_worker_cls,
      mock_create_trainer_factory,
      mock_load_actor_model,
      mock_create_mesh,
      mock_ensure_model_dir,
  ):
    self._setup_main_mocks(
        mock_ensure_model_dir,
        mock_create_mesh,
        mock_load_actor_model,
        mock_create_trainer_factory,
        mock_trainer_worker_cls,
        mock_grpc_server_cls,
    )

    signal_handlers = {}

    def mock_add_signal_handler(loop_self, sig, callback):
      signal_handlers[sig] = callback
      if sig == signal.SIGINT:
        loop_self.call_soon(callback)

    with self._patch_signal_handlers(mock_add_signal_handler):
      run_trainer_node.main(
          ["--port", "20000", "--worker_id", "trainer-0"],
          context=self.mock_context,
      )

    self.assertIn(signal.SIGINT, signal_handlers)
    self.assertIn(signal.SIGTERM, signal_handlers)
    self.mock_worker_service.stop.assert_called_once()
    self.mock_server.stop_serving.assert_called_once()

  @mock.patch.object(run_trainer_node, "_ensure_model_dir_for_trainer")
  @mock.patch.object(run_trainer_node, "_create_mesh")
  @mock.patch.object(run_trainer_node, "_load_actor_model")
  @mock.patch.object(run_trainer_node, "_create_trainer_factory")
  @mock.patch.object(trainer_worker, "TrainerWorker")
  @mock.patch.object(remote_execution, "GrpcRemoteExecutionServer")
  def test_shutdown_handler_handles_drain_exception_and_stops_server(
      self,
      mock_grpc_server_cls,
      mock_trainer_worker_cls,
      mock_create_trainer_factory,
      mock_load_actor_model,
      mock_create_mesh,
      mock_ensure_model_dir,
  ):
    self._setup_main_mocks(
        mock_ensure_model_dir,
        mock_create_mesh,
        mock_load_actor_model,
        mock_create_trainer_factory,
        mock_trainer_worker_cls,
        mock_grpc_server_cls,
    )
    self.mock_worker_service.stop.side_effect = RuntimeError(
        "Worker drain exception"
    )

    def mock_add_signal_handler(loop_self, sig, callback):
      if sig == signal.SIGTERM:
        loop_self.call_soon(callback)

    with self._patch_signal_handlers(mock_add_signal_handler):
      run_trainer_node.main(
          ["--port", "20000", "--worker_id", "trainer-0"],
          context=self.mock_context,
      )

    self.mock_worker_service.stop.assert_called_once()
    self.mock_server.stop_serving.assert_called_once()

  @mock.patch.object(run_trainer_node, "_ensure_model_dir_for_trainer")
  @mock.patch.object(run_trainer_node, "_create_mesh")
  @mock.patch.object(run_trainer_node, "_load_actor_model")
  @mock.patch.object(run_trainer_node, "_create_trainer_factory")
  @mock.patch.object(trainer_worker, "TrainerWorker")
  @mock.patch.object(remote_execution, "GrpcRemoteExecutionServer")
  def test_shutdown_ignores_not_implemented_error_on_add_signal_handler(
      self,
      mock_grpc_server_cls,
      mock_trainer_worker_cls,
      mock_create_trainer_factory,
      mock_load_actor_model,
      mock_create_mesh,
      mock_ensure_model_dir,
  ):
    self._setup_main_mocks(
        mock_ensure_model_dir,
        mock_create_mesh,
        mock_load_actor_model,
        mock_create_trainer_factory,
        mock_trainer_worker_cls,
        mock_grpc_server_cls,
    )

    canceled = False

    async def _cancel_soon():
      nonlocal canceled
      if canceled:
        return
      canceled = True
      await asyncio.sleep(0.001)
      for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
          task.cancel()

    def mock_add_signal_handler(loop_self, sig, callback):
      del sig, callback
      if not canceled:
        loop_self.create_task(_cancel_soon())
      raise NotImplementedError("Signal handler not supported")

    with self._patch_signal_handlers(mock_add_signal_handler):
      run_trainer_node.main(
          ["--port", "20000", "--worker_id", "trainer-0"],
          context=self.mock_context,
      )

    self.mock_worker_service.stop.assert_called_once()
    self.mock_server.stop_serving.assert_called_once()

  @mock.patch.object(run_trainer_node, "_create_tunix_trainer_factory")
  @mock.patch.object(run_trainer_node, "_create_maxtext_trainer_factory")
  def test_create_trainer_factory_delegates_by_backend(
      self, mock_create_maxtext, mock_create_tunix
  ):
    args_tunix = mock.MagicMock(trainer_backend="tunix")
    run_trainer_node._create_trainer_factory(args_tunix)
    mock_create_tunix.assert_called_once_with(args_tunix)
    mock_create_maxtext.assert_not_called()

    mock_create_tunix.reset_mock()
    args_maxtext = mock.MagicMock(trainer_backend="maxtext")
    run_trainer_node._create_trainer_factory(args_maxtext)
    mock_create_maxtext.assert_called_once_with(args_maxtext)
    mock_create_tunix.assert_not_called()

  def test_main_raises_without_discovery_context(self):
    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      run_trainer_node.main([], context=None)

    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      run_trainer_node.main([], context=mock.MagicMock(ipc=None))

  @mock.patch.object(run_trainer_node, "_ensure_model_dir_for_trainer")
  @mock.patch.object(run_trainer_node, "_create_mesh")
  @mock.patch.object(run_trainer_node, "_load_actor_model")
  def test_main_raises_on_non_positive_micro_batch_size(
      self, mock_load_actor_model, mock_create_mesh, mock_ensure_model_dir
  ):
    mock_ensure_model_dir.return_value = "/tmp/mock_model_dir"
    mock_create_mesh.return_value = mock.MagicMock(spec=Mesh)
    mock_load_actor_model.return_value = mock.MagicMock()

    with self.assertRaisesRegex(
        ValueError, "--train_micro_batch_size must be positive"
    ):
      run_trainer_node.main(
          ["--train_micro_batch_size", "0"],
          context=self.mock_context,
      )

  def test_parse_args_defaults_and_custom(self):
    args = run_trainer_node._parse_args([])
    self.assertEqual(args.port, 20000)
    self.assertEqual(args.worker_id, "trainer-0")
    self.assertEqual(args.model_name, "Qwen3-1.7B")
    self.assertEqual(args.mesh_fsdp, 2)
    self.assertEqual(args.mesh_tp, 1)
    self.assertEqual(args.checkpoint_save_interval_steps, 1)
    self.assertEqual(args.checkpoint_max_to_keep, 10)
    self.assertFalse(args.use_lora)
    self.assertEqual(args.weight_sync_use_ffi, "auto")

    custom_argv = [
        "--port",
        "20050",
        "--worker_id",
        "trainer-1",
        "--model_name",
        "Qwen3-32B",
        "--mesh_fsdp",
        "4",
        "--mesh_tp",
        "2",
        "--checkpoint_save_interval_steps",
        "5",
        "--checkpoint_max_to_keep",
        "3",
        "--checkpoint_root_directory",
        "/checkpoints/test",
        "--use_lora",
        "--lora_rank",
        "32",
        "--lora_alpha",
        "64.0",
        "--weight_sync_use_ffi",
        "false",
    ]
    args_custom = run_trainer_node._parse_args(custom_argv)
    self.assertEqual(args_custom.port, 20050)
    self.assertEqual(args_custom.worker_id, "trainer-1")
    self.assertEqual(args_custom.model_name, "Qwen3-32B")
    self.assertEqual(args_custom.mesh_fsdp, 4)
    self.assertEqual(args_custom.mesh_tp, 2)
    self.assertEqual(args_custom.checkpoint_save_interval_steps, 5)
    self.assertEqual(args_custom.checkpoint_max_to_keep, 3)
    self.assertEqual(args_custom.checkpoint_root_directory, "/checkpoints/test")
    self.assertTrue(args_custom.use_lora)
    self.assertEqual(args_custom.lora_rank, 32)
    self.assertEqual(args_custom.lora_alpha, 64.0)
    self.assertEqual(args_custom.weight_sync_use_ffi, "false")

  def test_create_mesh_validates_device_count(self):
    args = mock.MagicMock(mesh_fsdp=2, mesh_tp=2)
    with mock.patch.object(jax, "device_count", return_value=2):
      with self.assertRaisesRegex(
          ValueError, "Trainer mesh dimensions must multiply"
      ):
        run_trainer_node._create_mesh(args)

  def test_has_direct_safetensors(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      p = Path(tmp_dir)
      self.assertFalse(run_trainer_node._has_direct_safetensors(p))
      (p / "model.safetensors").touch()
      self.assertTrue(run_trainer_node._has_direct_safetensors(p))

  def test_ensure_model_dir_raises_for_empty_or_file(self):
    with self.assertRaisesRegex(ValueError, "--model_dir is required"):
      run_trainer_node._ensure_model_dir_for_trainer("", "Qwen/Qwen3-1.7B")

    with tempfile.NamedTemporaryFile() as tmp_file:
      with self.assertRaisesRegex(
          ValueError, "--model_dir must point to an existing local directory"
      ):
        run_trainer_node._ensure_model_dir_for_trainer(
            tmp_file.name, "Qwen/Qwen3-1.7B"
        )


if __name__ == "__main__":
  absltest.main()
