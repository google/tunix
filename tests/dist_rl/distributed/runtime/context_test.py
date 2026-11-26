# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for base distributed runtime context interfaces."""

import importlib
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.distributed.runtime import context


class ContextTest(parameterized.TestCase):

  def test_jax_context_defaults(self):
    jax_ctx = context.JaxContext()
    self.assertIsNone(jax_ctx.initialize())

  def test_discovery_context_defaults(self):
    disc_ctx = context.DiscoveryContext()
    self.assertIsNone(disc_ctx.on_register(lambda host, port, meta: None))
    self.assertIsNone(disc_ctx.register(b"metadata"))

  def test_ipc_context_defaults(self):
    ipc_ctx = context.IpcContext()
    self.assertIsInstance(ipc_ctx.discovery, context.DiscoveryContext)

  def test_process_context_defaults(self):
    proc_ctx = context.ProcessContext()
    with proc_ctx as entered:
      self.assertIs(entered, proc_ctx)
    self.assertIsInstance(proc_ctx.jax, context.JaxContext)
    self.assertIsInstance(proc_ctx.ipc, context.IpcContext)
    self.assertIsInstance(proc_ctx.ipc.discovery, context.DiscoveryContext)

  @parameterized.named_parameters(
      (
          "raiden_mode",
          {},
          common_datatypes.WeightSyncMode.RAIDEN,
          True,
      ),
      (
          "use_raiden_env_1",
          {"USE_RAIDEN": "1"},
          common_datatypes.WeightSyncMode.NONE,
          True,
      ),
      (
          "use_raiden_env_true",
          {"USE_RAIDEN": " TRUE "},
          common_datatypes.WeightSyncMode.NONE,
          True,
      ),
      (
          "no_raiden",
          {"USE_RAIDEN": "0"},
          common_datatypes.WeightSyncMode.NONE,
          False,
      ),
  )
  def test_prepare_env_jax_tpu_raiden_preload(
      self,
      env: dict[str, str],
      weight_sync_mode: common_datatypes.WeightSyncMode,
      expect_preload: bool,
  ):
    proc_ctx = context.ProcessContext()
    mod_name = "tunix.dist_rl.weight_sync.raiden_preload"
    fake_preload = types.ModuleType(mod_name)
    fake_preload.import_raiden = mock.MagicMock()
    env_modules = {k: v for k, v in sys.modules.items() if k != mod_name}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict(sys.modules, env_modules, clear=True),
        mock.patch.object(
            importlib, "import_module", return_value=fake_preload
        ) as mock_import,
    ):
      proc_ctx.prepare_env(
          execution_target.JAX_TPU, weight_sync_mode=weight_sync_mode
      )
    if expect_preload:
      mock_import.assert_called_once_with(mod_name)
      fake_preload.import_raiden.assert_called_once_with()
    else:
      mock_import.assert_not_called()
      fake_preload.import_raiden.assert_not_called()

  def test_prepare_env_jax_gpu_sets_defaults_without_overwriting(self):
    proc_ctx = context.ProcessContext()
    with mock.patch.dict(os.environ, {}, clear=True):
      proc_ctx.prepare_env(execution_target.JAX_GPU)
      self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda,cpu")
      self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
      self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85")

    preset = {
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    }
    with mock.patch.dict(os.environ, preset, clear=True):
      proc_ctx.prepare_env(execution_target.JAX_GPU)
      self.assertEqual(dict(os.environ), preset)

  def test_prepare_env_torch_tpu_gates_jax_to_cpu_and_sets_pjrt_device(self):
    proc_ctx = context.ProcessContext()
    with mock.patch.dict(os.environ, {}, clear=True):
      proc_ctx.prepare_env(execution_target.TORCH_TPU)
      self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
      self.assertEqual(os.environ["PJRT_DEVICE"], "TPU")

    preset = {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "TPU_V2"}
    with mock.patch.dict(os.environ, preset, clear=True):
      proc_ctx.prepare_env(execution_target.TORCH_TPU)
      self.assertEqual(dict(os.environ), preset)

  def test_prepare_env_rejects_unsupported_target(self):
    proc_ctx = context.ProcessContext()
    with self.assertRaisesRegex(
        ValueError, "Unsupported execution target 'torch:gpu'"
    ):
      proc_ctx.prepare_env(execution_target.TORCH_GPU)

  @parameterized.named_parameters(
      ("jax_tpu", execution_target.JAX_TPU),
      ("jax_gpu", execution_target.JAX_GPU),
  )
  def test_initialize_runtime_jax_targets_call_jax_initialize(
      self, target: execution_target.ExecutionTarget
  ):
    proc_ctx = context.ProcessContext()
    with mock.patch.object(context.JaxContext, "initialize") as mock_jax_init:
      proc_ctx.initialize_runtime(target)
    mock_jax_init.assert_called_once_with()

  def test_initialize_runtime_torch_tpu_imports_torch(self):
    proc_ctx = context.ProcessContext()
    with (
        mock.patch.object(context.JaxContext, "initialize") as mock_jax_init,
        mock.patch.object(importlib, "import_module") as mock_import,
    ):
      proc_ctx.initialize_runtime(execution_target.TORCH_TPU)
    mock_jax_init.assert_not_called()
    self.assertEqual([c.args[0] for c in mock_import.call_args_list], ["torch"])

  def test_initialize_runtime_rejects_unsupported_target(self):
    proc_ctx = context.ProcessContext()
    with self.assertRaisesRegex(
        ValueError, "Unsupported execution target 'torch:gpu'"
    ):
      proc_ctx.initialize_runtime(execution_target.TORCH_GPU)


if __name__ == "__main__":
  absltest.main()
