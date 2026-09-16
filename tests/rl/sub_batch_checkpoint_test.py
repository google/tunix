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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types

KEY_BASE = sub_batch_checkpoint.KEY_BASE
GEOMETRY = sub_batch_checkpoint.BatchGeometry(
    full_batch_size=4,
    mini_batch_size=2,
    train_micro_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    num_iterations=1,
)


def _group(prompt_id, num_generations=2):
  return [
      agent_types.TrajectoryItem(
          prompt_id=prompt_id,
          group_index=i,
          traj={
              "conversation_tokens": np.array([1, 2, prompt_id]),
              "old_logps": jnp.array([0.5, 0.25]),
              "reward": float(i),
          },
          metadata={"generation_id": i},
      )
      for i in range(num_generations)
  ]


def _prompt_ids(state):
  return [group[0].prompt_id for group in state.active_groups]


class SubBatchCheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path

  def _manager(self, enable_async=False):
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=enable_async
    )
    self.addCleanup(manager.close)
    manager.set_geometry(GEOMETRY)
    return manager

  def _keys(self, manager):
    manager.wait()
    return manager._run(
        lambda: sorted(ck.step for ck in manager._checkpointer.checkpoints)
    )

  def _write(self, global_step, prompt_ids):
    """Saves each group in a snapshot of its own."""
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=False
    )
    manager.set_geometry(GEOMETRY)
    for prompt_id in prompt_ids:
      manager.record_rollout_group(global_step, _group(prompt_id))
      manager.wait()
    manager.close()

  @parameterized.parameters(False, True)
  def test_restore_round_trips_groups(self, enable_async):
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=enable_async
    )
    manager.set_geometry(GEOMETRY)
    invalid = _group(12)
    invalid[0].is_valid = False
    # Groups finish out of order; training consumes them as recorded.
    manager.record_rollout_group(3, _group(13))
    manager.record_rollout_group(3, invalid)
    # Closing saves what was recorded.
    manager.close()

    state = self._manager().restore(global_step=3)

    self.assertEqual(state.global_step, 3)
    self.assertEqual(state.geometry, GEOMETRY)
    self.assertEqual(
        [
            [(t.prompt_id, t.group_index, t.is_valid) for t in group]
            for group in state.active_groups
        ],
        [[(13, 0, True), (13, 1, True)], [(12, 0, False), (12, 1, True)]],
    )
    item = state.active_groups[0][1]
    self.assertEqual(item.metadata, {"generation_id": 1})
    self.assertEqual(item.traj["reward"], 1.0)
    np.testing.assert_array_equal(item.traj["conversation_tokens"], [1, 2, 13])
    self.assertIsInstance(item.traj["old_logps"], np.ndarray)
    np.testing.assert_array_equal(item.traj["old_logps"], [0.5, 0.25])

  def test_snapshots_hold_only_new_groups(self):
    manager = self._manager()
    manager.record_rollout_group(2, _group(8))
    self.assertEqual(self._keys(manager), [2 * KEY_BASE])
    manager.record_rollout_group(2, _group(9))
    manager.record_rollout_group(2, _group(10))
    keys = self._keys(manager)
    # Groups 9 and 10 may or may not share a snapshot; either way, each group
    # is saved exactly once.
    saved = [
        group[0].prompt_id
        for key in keys
        for group in manager._run(manager._load, key).groups
    ]
    self.assertEqual(saved, [8, 9, 10])
    self.assertEqual(
        _prompt_ids(self._manager().restore(global_step=2)), [8, 9, 10]
    )

  def test_retention_keeps_the_two_latest_steps(self):
    manager = self._manager()
    for global_step in (1, 2, 3):
      for prompt_id in range(2):
        manager.record_rollout_group(global_step, _group(prompt_id))
        manager.wait()
    self.assertEqual(
        self._keys(manager),
        [2 * KEY_BASE, 2 * KEY_BASE + 1, 3 * KEY_BASE, 3 * KEY_BASE + 1],
    )

  def test_snapshot_converts_nested_jax_arrays(self):
    manager = self._manager()
    traj = agent_types.Trajectory(
        steps=[agent_types.Step(info={"logprobs": jnp.array([-0.5])})]
    )
    manager.record_rollout_group(
        0,
        [
            agent_types.TrajectoryItem(
                prompt_id=0,
                traj=traj,
                metadata={"turns": {0: jnp.array([1]), "count": 1}},
            )
        ],
    )
    manager.wait()

    item = self._manager().restore(global_step=0).active_groups[0][0]

    self.assertIsInstance(item.traj.steps[0].info["logprobs"], np.ndarray)
    self.assertIsInstance(item.metadata["turns"][0], np.ndarray)
    self.assertEqual(item.metadata["turns"]["count"], 1)

  def test_restore_without_snapshots_returns_none(self):
    manager = self._manager()
    self.assertFalse(manager.has_snapshots())
    self.assertIsNone(manager.restore(global_step=5))

  def test_restore_ignores_earlier_steps(self):
    self._write(global_step=3, prompt_ids=[12])
    manager = self._manager()
    self.assertTrue(manager.has_snapshots())
    self.assertIsNone(manager.restore(global_step=4))

  def test_restore_deletes_snapshots_of_later_steps(self):
    self._write(global_step=3, prompt_ids=[12, 13])
    self._write(global_step=4, prompt_ids=[16])
    manager = self._manager()

    state = manager.restore(global_step=3)

    self.assertEqual(_prompt_ids(state), [12, 13])
    self.assertEqual(self._keys(manager), [3 * KEY_BASE, 3 * KEY_BASE + 1])
    # The step's new groups are saved after the restored ones.
    manager.record_rollout_group(3, _group(14))
    self.assertEqual(
        self._keys(manager),
        [3 * KEY_BASE, 3 * KEY_BASE + 1, 3 * KEY_BASE + 2],
    )
    self.assertEqual(
        _prompt_ids(self._manager().restore(global_step=3)), [12, 13, 14]
    )

  def test_only_restored_groups_are_skipped(self):
    self._write(global_step=3, prompt_ids=[12])
    manager = self._manager()
    manager.restore(global_step=3)
    manager.record_rollout_group(3, _group(13))
    self.assertTrue(manager.should_skip_group(12))
    self.assertFalse(manager.should_skip_group(13))

  def test_set_geometry_rejects_a_change_from_the_snapshot(self):
    self._write(global_step=3, prompt_ids=[12])
    manager = self._manager()
    manager.restore(global_step=3)
    with self.assertRaisesRegex(ValueError, "geometry"):
      manager.set_geometry(dataclasses.replace(GEOMETRY, mini_batch_size=4))
    manager.set_geometry(GEOMETRY)

  def test_failed_saves_are_raised(self):
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.root, enable_async=False
    )
    manager.set_geometry(GEOMETRY)
    with mock.patch.object(
        manager._checkpointer,
        "save_checkpointables",
        side_effect=OSError("disk full"),
    ):
      manager.record_rollout_group(0, _group(0))
      manager._run(lambda: None)  # The save has failed by now.
      with self.assertRaisesRegex(OSError, "disk full"):
        manager.record_rollout_group(0, _group(1))
      # The save of the group just recorded fails too.
      with self.assertRaisesRegex(OSError, "disk full"):
        manager.close()

  def test_updates_per_step(self):
    self.assertEqual(GEOMETRY.updates_per_step, 2)
    self.assertEqual(
        dataclasses.replace(GEOMETRY, num_iterations=3).updates_per_step, 6
    )


if __name__ == "__main__":
  absltest.main()
