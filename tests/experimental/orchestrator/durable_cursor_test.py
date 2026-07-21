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

"""Tests for the durable cursor and checkpoint coordinator."""

import json
import pathlib
import tempfile

from absl.testing import absltest
from tunix.experimental.orchestrator import durable_cursor
from tunix.experimental.testing import fake_trainer_worker


class DurableCursorTest(absltest.TestCase):

  def test_to_from_dict_round_trip(self):
    cursor = durable_cursor.DurableCursor(
        global_step=3, weight_version=2, incarnation=1, dataset_cursor=10, seed=7
    )
    self.assertEqual(
        durable_cursor.DurableCursor.from_dict(cursor.to_dict()), cursor
    )

  def test_from_dict_ignores_unknown_keys(self):
    cursor = durable_cursor.DurableCursor.from_dict(
        {"global_step": 1, "unknown": 99}
    )
    self.assertEqual(cursor.global_step, 1)


class CheckpointCoordinatorTest(absltest.TestCase):

  def _coordinator(self, **kwargs):
    trainer = fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    path = pathlib.Path(tempfile.mkdtemp()) / "cursor.json"
    coord = durable_cursor.CheckpointCoordinator(trainer, path, **kwargs)
    return coord, trainer, path

  def test_save_writes_cursor_and_checkpoints_trainer(self):
    coord, _, path = self._coordinator()
    saved = coord.maybe_save(
        durable_cursor.DurableCursor(global_step=1, weight_version=1)
    )
    self.assertTrue(saved)
    self.assertTrue(path.exists())
    self.assertEqual(json.loads(path.read_text())["global_step"], 1)

  def test_cadence_skips_off_boundary_steps(self):
    coord, _, path = self._coordinator(save_every_n_steps=2)
    self.assertFalse(
        coord.maybe_save(durable_cursor.DurableCursor(global_step=1))
    )
    self.assertFalse(path.exists())
    self.assertTrue(
        coord.maybe_save(durable_cursor.DurableCursor(global_step=2))
    )

  def test_resume_returns_none_when_absent(self):
    coord, _, _ = self._coordinator()
    self.assertIsNone(coord.resume())

  def test_resume_restores_trainer_and_returns_cursor(self):
    coord, _, _ = self._coordinator()
    coord.save(durable_cursor.DurableCursor(global_step=5, weight_version=3))
    restored = coord.resume()
    self.assertEqual(restored.global_step, 5)
    self.assertEqual(restored.weight_version, 3)


if __name__ == "__main__":
  absltest.main()
