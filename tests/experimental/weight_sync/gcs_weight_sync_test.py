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

"""Unit and integration tests for GCSWeightSync and GCSWeightSyncHandler."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.weight_sync import gcs_weight_sync
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.weight_sync import weight_sync_coordinator


class _DummyRunner:
  """Minimal runner holding `.state` and `.state_leaves` like vLLM's TPU model runner."""

  def __init__(self, state: dict):
    self.state = state
    self.state_leaves = tuple(jax.tree_util.tree_leaves(state))
    self.refreshed = False

  def refresh_state_leaves(self) -> None:
    self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
    self.refreshed = True


class GCSWeightSyncTest(absltest.TestCase):

  def test_unified_weight_synchronizer_hierarchy(self):
    staging_dir = self.create_tempdir().full_path
    gcs_sync = weight_sync.create_weight_synchronizer(
        "gcs", "trainer", staging_dir=staging_dir
    )
    raiden_sync = weight_sync.create_weight_synchronizer("raiden", "trainer")
    self.assertIsInstance(gcs_sync, weight_sync.WeightSynchronizer)
    self.assertIsInstance(raiden_sync, weight_sync.WeightSynchronizer)
    self.assertIsInstance(gcs_sync, gcs_weight_sync.GCSWeightSync)
    self.assertIsInstance(raiden_sync, raiden_synchronizer.RaidenSynchronizer)

  def test_d2h_h2d_apply_to_runner_and_verify_checksums(self):
    staging_dir = self.create_tempdir().full_path
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("tp",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None)
    )

    src_state = {
        "base": {
            "decoder": {
                "layers_0": {
                    "mlp": {
                        "wi_0": {
                            "kernel": jax.device_put(
                                jnp.arange(16, dtype=jnp.bfloat16).reshape(4, 4)
                                + 1.0,
                                sharding,
                            )
                        }
                    }
                }
            }
        }
    }
    dst_state = {
        "model": {
            "decoder": {
                "layers_0": {
                    "mlp": {
                        "wi_0": {
                            "kernel": jax.device_put(
                                jnp.zeros((4, 4), dtype=jnp.bfloat16),
                                sharding,
                            )
                        }
                    }
                }
            }
        }
    }
    runner = _DummyRunner(dst_state)

    with mock.patch.dict(os.environ, {"VERIFY_WEIGHTS": "true"}):
      src_sync = gcs_weight_sync.GCSWeightSync(
          job_name="trainer",
          state=src_state,
          staging_dir=staging_dir,
          max_to_keep=0,
      )
      req = datatypes.WeightSyncRequest(
          policy_version=1,
          extra_config={"req_id": "wsync-v1-r0", "uuid": 1},
      )
      src_sync.d2h(sync_request=req)
      src_meta = src_sync.work_unit_metadata()
      self.assertIsNotNone(src_meta.artifact_uri)
      self.assertTrue(os.path.exists(src_meta.artifact_uri))
      self.assertIsNotNone(src_meta.checksums)

      dst_sync = gcs_weight_sync.GCSWeightSync(
          job_name="replica_vllm-0",
          state=runner.state,
          staging_dir=staging_dir,
      )
      dst_meta = dst_sync.work_unit_metadata()

      # Manifest preflight between trainer and rollout must find 0 mismatches
      mismatches = weight_sync_coordinator._manifest_mismatches(  # pylint: disable=protected-access
          [src_meta], [dst_meta]
      )
      self.assertEmpty(mismatches)

      # Restore and apply to runner
      handler = gcs_weight_sync.GCSWeightSyncHandler()
      handler.register_work_unit(src_meta)
      handler.register_work_unit(dst_meta)
      extra = handler.build_extra_config([src_meta])
      prepared_req = datatypes.WeightSyncRequest(
          policy_version=1,
          source_metadata=[src_meta],
          extra_config=extra,
      )
      transfer_res = handler.transfer(
          [src_meta.unit], [dst_meta.unit], req_id="wsync-v1-r0", generation=1
      )
      self.assertTrue(transfer_res.success)

      dst_sync.h2d(sync_request=prepared_req)
      dst_sync.apply_to_runner(runner)
      self.assertTrue(runner.refreshed)

      # Re-bind to runner's updated live state and verify checksum parity
      dst_sync.bind(runner.state)
      dst_checksums = dst_sync.checksums(sample=None)
      summary = weight_sync.verify_weight_checksums(
          src_meta.checksums, dst_checksums
      )
      self.assertTrue(summary["verified"])
      self.assertEqual(summary["max_rel_err"], 0.0)

      updated_arr = runner.state["model"]["decoder"]["layers_0"]["mlp"]["wi_0"][
          "kernel"
      ]
      np.testing.assert_allclose(
          np.asarray(updated_arr, dtype=np.float32),
          np.asarray(
              src_state["base"]["decoder"]["layers_0"]["mlp"]["wi_0"]["kernel"],
              dtype=np.float32,
          ),
      )

      # Verify release() cleans up staging checkpoint when max_to_keep=0
      ckpt_uri = src_meta.artifact_uri
      src_sync.release(sync_request=prepared_req)
      self.assertFalse(os.path.exists(ckpt_uri))

  def test_verify_weight_checksums_detects_corruption(self):
    src = {
        "decoder.layers_0.mlp.wi_0.kernel": 120.0,
        "__grand_total__": 120.0,
        "__tensor_count__": 1,
        "__element_count__": 16,
    }
    dst_bad = {
        "decoder.layers_0.mlp.wi_0.kernel": 99.0,
        "__grand_total__": 99.0,
        "__tensor_count__": 1,
        "__element_count__": 16,
    }
    with mock.patch.dict(os.environ, {"VERIFY_WEIGHTS": "true"}):
      with self.assertRaisesRegex(RuntimeError, "WEIGHT VERIFICATION FAILED"):
        weight_sync.verify_weight_checksums(src, dst_bad)


if __name__ == "__main__":
  absltest.main()
