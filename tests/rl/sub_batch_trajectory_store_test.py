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

"""Tests for sub_batch_trajectory_store."""

import dataclasses
import enum
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl import sub_batch_checkpoint
from tunix.rl import sub_batch_trajectory_store
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import token_dict
from tunix.rl.agentic.trajectory import token_dict_testing
from tunix.sft import checkpoint_options

digest = sub_batch_trajectory_store.canonical_digest
_make_trajectory = token_dict_testing.make_trajectory
_collect = token_dict_testing.stored_token_dict


def _open_store(
    root: str, run_id: str = "run"
) -> file_store.FileTrajectoryStore[trajectory_lib.TunixTrajectoryMetadata]:
  """Opens a file Trajectory Store reading Tunix metadata, as the learner's."""
  return file_store.FileTrajectoryStore(
      root, run_id=run_id, metadata_cls=trajectory_lib.TunixTrajectoryMetadata
  )


class _Color(enum.Enum):
  RED = 1
  BLUE = 2


class CanonicalDigestTest(absltest.TestCase):

  def test_equivalent_values_digest_equal(self):
    for a, b in (
        ([1, 2], (1, 2)),
        (np.int64(3), 3),
        (np.float32(0.5), 0.5),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (jnp.array([1, 2], dtype=jnp.int32), np.array([1, 2], dtype=np.int32)),
    ):
      with self.subTest(repr(a)):
        self.assertEqual(digest(a), digest(b))

  def test_training_relevant_differences_digest_differently(self):
    for a, b in (
        (np.array([1, 2], np.int32), np.array([1, 2], np.int64)),
        (np.array([1, 2, 3, 4]), np.array([[1, 2], [3, 4]])),
        (1, 1.0),
        (1, True),
        ("a", b"a"),
        (None, 0),
        (_Color.RED, _Color.BLUE),
        ({"a": 1}, {"a": 2}),
        ([1, [2]], [[1], 2]),
    ):
      with self.subTest(repr(a)):
        self.assertNotEqual(digest(a), digest(b))

  def test_values_without_canonical_encoding_raise(self):
    for value in (np.array([object()]), object()):
      with self.subTest(repr(value)):
        with self.assertRaisesRegex(TypeError, "Cannot digest"):
          digest(value)


class TrajectoryStorePayloadsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    self.store = _open_store(self.root)
    self.addCleanup(self.store.close)
    self.payloads = sub_batch_trajectory_store.TrajectoryStorePayloads(
        self.store
    )

  def assertSameDict(self, actual, expected):
    self.assertEqual(list(actual), list(expected))
    self.assertEqual(digest(actual), digest(expected))

  def test_commit_then_load_in_a_fresh_process_round_trips(self):
    live = [_collect(self.store, "t0"), _collect(self.store, "t1", 0.5)]
    # Item metadata rides the ref as JSON, which keeps ints of any size.
    metadata = [
        {"generation_id": 0},
        {"generation_id": 1, "trace_id": (1 << 122) + 17},
    ]
    refs = self.payloads.commit(list(zip(live, metadata)))
    for ref in refs:
      self.assertTrue(all(isinstance(v, str) for v in ref.values()))

    reopened = _open_store(self.root)
    self.addCleanup(reopened.close)
    loaded = sub_batch_trajectory_store.TrajectoryStorePayloads(
        reopened
    ).load(refs)

    self.assertLen(loaded, 2)
    for (traj, got_metadata), expected, want_metadata in zip(
        loaded, live, metadata
    ):
      self.assertSameDict(traj, expected)
      self.assertEqual(got_metadata, want_metadata)

  def test_commit_requires_store_ids(self):
    live = _collect(self.store, "t0")
    del live[token_dict.TRAJECTORY_ID_KEY]
    with self.assertRaisesRegex(ValueError, "tagged with 'trajectory_id'"):
      self.payloads.commit([(live, {})])

  def test_commit_detects_unwritten_trajectory(self):
    live = token_dict.build_token_dict(
        _make_trajectory(),
        token_dict_testing.token_dict_context(),
        trajectory_id="never_written",
    )
    with self.assertRaises(store_lib.TrajectoryNotFoundError):
      self.payloads.commit([(live, {})])

  def test_commit_detects_drift_and_names_the_key(self):
    live = _collect(self.store, "t0")
    live["trajectory_reward"] = 99.0
    with self.assertRaisesRegex(ValueError, "differing keys: trajectory_reward"):
      self.payloads.commit([(live, {})])

  def test_commit_rejects_metadata_that_does_not_round_trip_json(self):
    live = _collect(self.store, "t0")
    for metadata in ({1: "int key"}, {"a": np.array([1])}):
      with self.subTest(repr(metadata)):
        with self.assertRaises((ValueError, TypeError)):
          self.payloads.commit([(live, metadata)])

  def test_commit_verifies_each_id_once(self):
    live = _collect(self.store, "t0")
    with mock.patch.object(
        self.store, "get_trajectories", wraps=self.store.get_trajectories
    ) as get:
      first = self.payloads.commit([(live, {})])
      second = self.payloads.commit([(live, {})])
    self.assertEqual(get.call_count, 1)
    self.assertEqual(first, second)

  def test_load_rejects_digest_mismatch(self):
    live = _collect(self.store, "t0")
    (ref,) = self.payloads.commit([(live, {})])
    ref["digest"] = digest("something else")
    with self.assertRaisesRegex(ValueError, "does not match the digest"):
      self.payloads.load([ref])

  def test_degenerate_rollouts_round_trip(self):
    """Rollouts with an empty response or no turns survive commit and load.

    Commit rebuilds every new id from the store and compares digests, so a
    rebuild that differs for these shapes would fail the first snapshot.
    """
    empty_response = agent_types.Step(
        model_response="",
        done=True,
        reward=0.0,
        assistant_tokens=np.array([], dtype=np.int32),
        assistant_masks=np.array([], dtype=np.int32),
        logprobs=np.array([], dtype=np.float32),
    )
    for name, steps in (("empty_response", [empty_response]), ("no_turns", [])):
      with self.subTest(name):
        trajectory = dataclasses.replace(_make_trajectory(), steps=steps)
        context = token_dict_testing.token_dict_context()
        token_dict.write_trajectory(
            self.store, trajectory, context, trajectory_id=name
        )
        live = token_dict.build_token_dict(
            trajectory, context, trajectory_id=name
        )
        refs = self.payloads.commit([(live, {})])

        reopened = _open_store(self.root)
        self.addCleanup(reopened.close)
        payloads = sub_batch_trajectory_store.TrajectoryStorePayloads(reopened)
        ((traj, metadata),) = payloads.load(refs)
        self.assertSameDict(traj, live)
        self.assertEqual(metadata, {})


class ManagerIntegrationTest(absltest.TestCase):
  """The manager with a real Trajectory Store and Orbax checkpointer."""

  def _manager(self, root, store):
    mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
        root_directory=root,
        options=checkpoint_options.TunixCheckpointingOptions(
            save_decision_policy=(
                ocp.training.save_decision_policies.FixedIntervalPolicy(
                    interval=1
                )
            ),
            preservation_policy=ocp.training.preservation_policies.LatestN(
                n=10
            ),
            step_name_format=ocp.path.step.standard_name_format(),
            enable_async_checkpointing=True,
        ),
        payload_store=sub_batch_trajectory_store.TrajectoryStorePayloads(
            store
        ),
    )
    self.addCleanup(mgr.close)
    return mgr

  def _save_snapshot(self, ckpt_root, store_root):
    """Saves one mid-step snapshot whose active trajectory is in the store.

    Args:
      ckpt_root: The manager's root directory.
      store_root: The root directory of the run's file Trajectory Store.

    Returns:
      The live Token-mode dict of the snapshot's active trajectory.
    """
    store = _open_store(store_root)
    self.addCleanup(store.close)
    live = _collect(store, "t0")
    mgr = self._manager(ckpt_root, store)
    mgr.save(
        1,
        1,
        iter_steps=3,
        global_step=1,
        grad_accum_steps=2,
        step_complete=False,
        completed_group_ids=[5],
        trained_trajectory_counts={(5, 0): 1},
        active_group_trajectories=[
            agent_types.TrajectoryItem(
                prompt_id=5,
                group_index=0,
                start_step=0,
                traj=live,
                metadata={"generation_id": 0},
            )
        ],
        training_state=None,
    )
    mgr.wait()
    return live

  def test_snapshot_restores_trajectories_from_the_store(self):
    ckpt_root = self.create_tempdir().full_path
    store_root = self.create_tempdir().full_path
    live = self._save_snapshot(ckpt_root, store_root)

    reopened = _open_store(store_root)
    self.addCleanup(reopened.close)
    state = self._manager(ckpt_root, reopened).try_restore(
        train_steps=1, grad_accum_steps=2
    )

    self.assertIsNotNone(state)
    assert state is not None
    (item,) = state.active_group_trajectories
    self.assertEqual((item.prompt_id, item.group_index), (5, 0))
    self.assertEqual(list(item.traj), list(live))
    self.assertEqual(digest(item.traj), digest(live))
    self.assertEqual(item.metadata, {"generation_id": 0})

  def test_restore_from_another_store_raises_and_keeps_the_snapshot(self):
    """A relaunch pointed at the wrong store fails loud.

    The snapshot survives the failed restore, so a later relaunch with the
    store that wrote it still resumes.
    """
    ckpt_root = self.create_tempdir().full_path
    store_root = self.create_tempdir().full_path
    self._save_snapshot(ckpt_root, store_root)

    for root, run_id in (
        (self.create_tempdir().full_path, "run"),
        (store_root, "other_run"),
    ):
      with self.subTest(run_id=run_id):
        other = _open_store(root, run_id=run_id)
        self.addCleanup(other.close)
        with self.assertRaisesRegex(
            sub_batch_checkpoint.SubBatchUnreadableError,
            "TrajectoryNotFoundError",
        ):
          self._manager(ckpt_root, other).try_restore(
              train_steps=1, grad_accum_steps=2
          )

    reopened = _open_store(store_root)
    self.addCleanup(reopened.close)
    state = self._manager(ckpt_root, reopened).try_restore(
        train_steps=1, grad_accum_steps=2
    )
    self.assertIsNotNone(state)


if __name__ == "__main__":
  absltest.main()
