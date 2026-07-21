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

"""Tests for the MetricsPump seal-id drain-once policy."""

from absl.testing import absltest
from tunix.experimental.metrics import metrics
from tunix.experimental.orchestrator import metrics_pump


def _buffer(seal_id, *, mode="train", scalars=None):
  return metrics.MetricsBuffer(
      id=seal_id, scalar_metrics=scalars or {}, mode=mode
  )


class MetricsPumpTest(absltest.TestCase):

  def test_accepts_new_seal_ids_and_stamps_step(self):
    pump = metrics_pump.MetricsPump()
    self.assertTrue(pump.pull("t0", _buffer(0, scalars={"loss": 1.0}), step=0))
    self.assertTrue(pump.pull("t0", _buffer(1, scalars={"loss": 0.5}), step=1))
    records = pump.records()
    self.assertEqual([r.step for r in records], [0, 1])
    self.assertEqual(records[0].scalars["loss"], 1.0)
    self.assertEqual(records[1].seal_id, 1)

  def test_drops_redelivered_seal_id(self):
    pump = metrics_pump.MetricsPump()
    self.assertTrue(pump.pull("t0", _buffer(5), step=0))
    self.assertFalse(pump.pull("t0", _buffer(5), step=1))  # at-least-once retry
    self.assertLen(pump.records(), 1)

  def test_same_seal_id_across_workers_is_independent(self):
    pump = metrics_pump.MetricsPump()
    self.assertTrue(pump.pull("t0", _buffer(0), step=0))
    self.assertTrue(pump.pull("r0", _buffer(0), step=0))
    self.assertLen(pump.records(), 2)
    self.assertLen(pump.records_for("t0"), 1)

  def test_string_seal_ids_are_deduped(self):
    pump = metrics_pump.MetricsPump()
    self.assertTrue(pump.pull("t0", _buffer("eval-0", mode="eval"), step=0))
    self.assertFalse(pump.pull("t0", _buffer("eval-0", mode="eval"), step=0))
    self.assertEqual(pump.records()[0].mode, "eval")


if __name__ == "__main__":
  absltest.main()
