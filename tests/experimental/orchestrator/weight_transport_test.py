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

"""Tests for the InProcessTransport."""

import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import weight_transport


def _spec(version):
  return datatypes.WeightSyncSpec(version=version)


def _meta(version):
  return datatypes.WeightSyncMetadata(version=version, method="in_process")


class InProcessTransportTest(absltest.TestCase):

  def test_stage_returns_metadata_with_live_locator(self):
    transport = weight_transport.InProcessTransport()
    state = {"w": np.arange(4)}
    meta = transport.stage(state, _spec(3))
    self.assertEqual(meta.version, 3)
    self.assertEqual(meta.method, "in_process")
    self.assertIs(meta.locator, state)  # zero-copy hand-off

  def test_fetch_round_trips_staged_state(self):
    transport = weight_transport.InProcessTransport()
    state = {"w": np.arange(4)}
    meta = transport.stage(state, _spec(1))
    chunks = list(transport.fetch(meta))
    self.assertLen(chunks, 1)
    self.assertIs(chunks[0], state)

  def test_fetch_unknown_version_raises(self):
    transport = weight_transport.InProcessTransport()
    with self.assertRaises(KeyError):
      list(transport.fetch(_meta(9)))

  def test_release_drops_only_that_version(self):
    transport = weight_transport.InProcessTransport()
    transport.stage({"w": 1}, _spec(1))
    transport.stage({"w": 2}, _spec(2))
    transport.release(1)
    self.assertEqual(transport.staged_versions(), [2])
    with self.assertRaises(KeyError):
      list(transport.fetch(_meta(1)))
    # A later version stays fetchable (retry / crash-recovery re-fetch).
    self.assertLen(list(transport.fetch(_meta(2))), 1)

  def test_release_is_idempotent(self):
    transport = weight_transport.InProcessTransport()
    transport.release(99)  # no such version; must not raise


if __name__ == "__main__":
  absltest.main()
