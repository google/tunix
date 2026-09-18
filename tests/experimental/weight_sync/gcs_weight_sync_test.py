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

"""Unit tests for GCS weight sync handler and runner application."""

import sys
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from tunix.experimental.common import datatypes
from tunix.experimental.weight_sync import gcs_weight_sync
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.weight_sync import weight_sync_coordinator


class GcsSyncHandlerTest(unittest.TestCase):

  def test_handler_registration_and_transfer(self):
    handler = gcs_weight_sync.GcsSyncHandler()
    self.assertFalse(handler.needs_manifest_preflight)

    unit_src = weight_sync.WorkUnitId(job_name="trainer")
    unit_dst = weight_sync.WorkUnitId(job_name="rollout_0")
    handler.register_work_unit(weight_sync.WorkUnitMetadata(unit=unit_src))
    handler.register_work_unit(weight_sync.WorkUnitMetadata(unit=unit_dst))

    self.assertIn(unit_src, handler.registered_units)
    self.assertIn(unit_dst, handler.registered_units)

    result = handler.transfer(
        src_units=[unit_src],
        dst_units=[unit_dst],
        req_id="test_req",
    )
    self.assertTrue(result.success)
    self.assertEqual(result.req_id, "test_req")

  def test_create_default_handler_gcs(self):
    handler = weight_sync_coordinator.create_default_handler(mode="gcs")
    self.assertIsInstance(handler, gcs_weight_sync.GcsSyncHandler)

  @mock.patch("orbax.checkpoint.Checkpointer")
  def test_apply_checkpoint_to_runner(self, mock_cp_cls):
    mock_cp = mock.MagicMock()
    mock_cp_cls.return_value = mock_cp

    # Runner state with 2 parameters
    w1 = jnp.zeros((4, 4))
    w2 = jnp.zeros((2, 2))
    runner = mock.MagicMock()
    runner.state = {
        "model": {
            "layers_0": {"attention": {"q_proj": w1}},
            "layers_1": {"mlp": {"dense": w2}},
        }
    }
    runner.state_leaves = (w1, w2)

    # Restored checkpoint arrays with matching names/shapes
    new_w1 = jnp.ones((4, 4))
    new_w2 = jnp.ones((2, 2))
    mock_cp.restore.return_value = {
        "base": {
            "layers_0": {"attention": {"q_proj": new_w1}},
            "layers_1": {"mlp": {"dense": new_w2}},
        }
    }

    matched = gcs_weight_sync.apply_checkpoint_to_runner(
        runner, "/tmp/fake_ckpt"
    )

    self.assertEqual(matched, 2)
    self.assertEqual(len(runner.state_leaves), 2)
    self.assertTrue(jnp.array_equal(runner.state_leaves[0], new_w1))
    self.assertTrue(jnp.array_equal(runner.state_leaves[1], new_w2))

  def test_patch_tpu_worker_gcs_sync(self):
    fake_mod = types.ModuleType("tpu_inference.worker.tpu_worker")

    class FakeTPUWorker:
      pass

    fake_mod.TPUWorker = FakeTPUWorker
    with mock.patch.dict(sys.modules, {"tpu_inference.worker.tpu_worker": fake_mod}):
      gcs_weight_sync.patch_tpu_worker_gcs_sync()
      self.assertTrue(hasattr(FakeTPUWorker, "load_gcs_weights"))

      # Test method invocation
      worker = FakeTPUWorker()
      worker.model_runner = mock.MagicMock()
      worker.model_runner.state = {"a": jnp.zeros((2,))}
      worker.model_runner.state_leaves = (jnp.zeros((2,)),)

      with mock.patch(
          "tunix.experimental.weight_sync.gcs_weight_sync.apply_checkpoint_to_runner"
      ) as mock_apply:
        worker.load_gcs_weights("/tmp/path")
        mock_apply.assert_called_once_with(worker.model_runner, "/tmp/path")


if __name__ == "__main__":
  unittest.main()
