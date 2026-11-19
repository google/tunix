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

"""Unit tests for `GCSHandler`."""

from __future__ import annotations

import asyncio
import os
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync
from tunix.experimental.weight_sync import gcs_handler
from tunix.experimental.weight_sync import gcs_synchronizer

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SRC = weight_sync.WorkUnitId("trainer")
_DST = weight_sync.WorkUnitId("rollout")


class GCSHandlerTest(absltest.TestCase):

  def _stage_checkpoint(self) -> weight_sync.WorkUnitMetadata:
    """Saves a real finalized Orbax checkpoint and returns source metadata."""
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=self.create_tempdir().full_path
        ),
    )
    sync.bind({"layers.0.w": jax.device_put(np.ones((2, 2), np.float32))})
    sync.d2h(
        datatypes.WeightSyncRequest(
            policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
        )
    )
    return sync.work_unit_metadata()

  def _handler(
      self, src_meta: weight_sync.WorkUnitMetadata
  ) -> gcs_handler.GCSHandler:
    handler = gcs_handler.GCSHandler()
    handler.register_work_unit(weight_sync.WorkUnitMetadata(unit=_DST))
    handler.register_work_unit(src_meta)
    return handler

  def test_transfer_validates_registered_units(self):
    handler = self._handler(
        weight_sync.WorkUnitMetadata(unit=_SRC, artifact_uri="/tmp/x")
    )
    self.assertIn(
        "at least one source", handler.transfer([], [_DST]).message
    )
    self.assertIn(
        "at least one destination", handler.transfer([_SRC], []).message
    )
    unknown = weight_sync.WorkUnitId("unknown")
    self.assertIn(
        "Unregistered destination", handler.transfer([_SRC], [unknown]).message
    )
    self.assertIn(
        "Unregistered source", handler.transfer([unknown], [_DST]).message
    )

  def test_transfer_requires_fresh_artifact_each_round(self):
    handler = self._handler(weight_sync.WorkUnitMetadata(unit=_SRC))
    res = handler.transfer([_SRC], [_DST], req_id="no-artifact")
    self.assertFalse(res.success)
    self.assertIn("fresh artifact_uri", res.message)

    handler.register_work_unit(self._stage_checkpoint())
    first = handler.transfer([_SRC], [_DST], req_id="r1")
    self.assertTrue(first.success, first.message)
    self.assertEqual(first.req_id, "r1")
    stale = handler.transfer([_SRC], [_DST], req_id="r2")
    self.assertFalse(stale.success)
    self.assertIn("fresh artifact_uri", stale.message)

  def test_transfer_rejects_conflicting_source_artifacts(self):
    other_src = weight_sync.WorkUnitId("trainer", job_replica_id="1")
    handler = self._handler(
        weight_sync.WorkUnitMetadata(unit=_SRC, artifact_uri="/tmp/a")
    )
    handler.register_work_unit(
        weight_sync.WorkUnitMetadata(unit=other_src, artifact_uri="/tmp/b")
    )
    res = handler.transfer([_SRC, other_src], [_DST])
    self.assertFalse(res.success)
    self.assertIn("conflicting artifact_uri", res.message)

  def test_transfer_rejects_missing_or_unfinalized_checkpoint(self):
    root = epath.Path(self.create_tempdir().full_path)
    missing = self._handler(
        weight_sync.WorkUnitMetadata(
            unit=_SRC, artifact_uri=str(root / "missing")
        )
    ).transfer([_SRC], [_DST])
    self.assertFalse(missing.success)
    self.assertIn("does not exist", missing.message)

    tmp_ckpt = root / "step_1_r1.orbax-checkpoint-tmp-0"
    tmp_ckpt.mkdir()
    unfinalized = self._handler(
        weight_sync.WorkUnitMetadata(unit=_SRC, artifact_uri=str(tmp_ckpt))
    ).transfer([_SRC], [_DST])
    self.assertFalse(unfinalized.success)
    self.assertIn("not finalized", unfinalized.message)

  def test_transfer_inside_running_event_loop(self):
    handler = self._handler(self._stage_checkpoint())

    async def _transfer() -> weight_sync.TransferResult:
      return handler.transfer([_SRC], [_DST], req_id="in-loop")

    res = asyncio.run(_transfer())
    self.assertTrue(res.success, res.message)

  def test_transfer_reports_storage_errors_as_failed_result(self):
    handler = self._handler(self._stage_checkpoint())
    with mock.patch.object(
        gcs_handler.ocp.path.step,
        "is_path_finalized",
        autospec=True,
        side_effect=OSError("gcs timeout"),
    ):
      res = handler.transfer([_SRC], [_DST], req_id="req-io")
    self.assertFalse(res.success)
    self.assertEqual(res.req_id, "req-io")
    self.assertIn("Failed checking checkpoint artifact", res.message)


if __name__ == "__main__":
  absltest.main()
