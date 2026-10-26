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

"""Tests for tunix.experimental.weight_sync.weight_sync compatibility shim."""

from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.weight_sync import raiden_synchronizer
from tunix.dist_rl.weight_sync import weight_sync as dist_weight_sync
from tunix.experimental.weight_sync import weight_sync as exp_weight_sync


class ExperimentalWeightSyncShimTest(absltest.TestCase):

  def test_reexports_contract_and_contains_legacy_factory(self):
    self.assertIs(
        exp_weight_sync.WeightSyncMode, dist_weight_sync.WeightSyncMode
    )
    self.assertFalse(hasattr(dist_weight_sync, "create_weight_synchronizer"))
    self.assertFalse(
        hasattr(raiden_synchronizer, "create_weight_synchronizer")
    )

    for mode in ("raiden", "  RAIDEN ", dist_weight_sync.WeightSyncMode.RAIDEN):
      with self.subTest(mode=mode):
        sync = exp_weight_sync.create_weight_synchronizer(
            mode=mode,
            job_name="trainer",
            worker_index=2,
            staging_dir="/tmp/unused_staging",
            auto_h2d=False,
            parallelism=4,
        )
        self.assertIsInstance(sync, raiden_synchronizer.RaidenSynchronizer)
        self.assertEqual(sync.job_name, "trainer")
        self.assertEqual(sync.worker_index, 2)
        self.assertEqual(sync.work_unit_metadata().unit.job_replica_id, "2")

    for bad_mode in (
        "gcs",
        "file",
        "fallback",
        "none",
        "invalid",
        dist_weight_sync.WeightSyncMode.FALLBACK,
        dist_weight_sync.WeightSyncMode.NONE,
    ):
      with self.subTest(bad_mode=bad_mode):
        with self.assertRaisesRegex(ValueError, "Unsupported WeightSyncMode"):
          exp_weight_sync.create_weight_synchronizer(bad_mode, "trainer")

  def test_supports_mock_patch_on_shim_module(self):
    sentinel = object()
    with mock.patch.object(
        exp_weight_sync, "create_weight_synchronizer", return_value=sentinel
    ) as mocked:
      self.assertIs(
          exp_weight_sync.create_weight_synchronizer("raiden", "trainer"),
          sentinel,
      )
      mocked.assert_called_once_with("raiden", "trainer")


if __name__ == "__main__":
  absltest.main()
