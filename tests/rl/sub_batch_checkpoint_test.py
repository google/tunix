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


"""Tests for sub_batch_checkpoint (encoded parent-train_step keying).

Keys are train_steps * KEY_BASE + local. The fixed-k tests drive the API
through the _save helper (which decodes an iter_steps-style position); the
ragged-window tests call save() directly with locals no fixed-k window could
produce.
"""

import shutil
import tempfile
from types import SimpleNamespace

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_options


def _make_trajectory_item(
    group_id: int,
    pair_index: int,
    reward: float = 1.0,
    with_env: bool = False,
) -> agent_types.TrajectoryItem:
  """Creates a dummy `TrajectoryItem` for testing."""
  env_kwargs = (
      dict(env_tokens=np.array([7, 8]), env_masks=np.array([1, 0]))
      if with_env
      else {}
  )
  step = agent_types.Step(
      chat_completions=[{"role": "user", "content": "test"}],
      thought="thinking",
      model_response="response",
      reward=reward,
      done=True,
      assistant_tokens=np.array([1, 2, 3]),
      assistant_masks=np.array([1, 1, 1]),
      logprobs=np.array([0.1, 0.2, 0.3]),
      **env_kwargs,
  )
  traj = agent_types.Trajectory(
      task="test_task",
      steps=[step],
      reward=reward,
      status=agent_types.TrajectoryStatus.SUCCEEDED,
  )
  return agent_types.TrajectoryItem(
      group_id=group_id,
      pair_index=pair_index,
      start_step=0,
      traj=traj,
      metadata={"test_key": "test_value"},
  )


def _state(acc: float = 1.0, mini_step: int = 1):
  return {
      "acc_grads": {"w": np.array([acc])},
      "mini_step": np.array([mini_step]),
  }


class SubBatchCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
        root_directory=self.test_dir,
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
    )

  def tearDown(self):
    self.mgr.close()
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def _save(
      self,
      iter_steps,
      *,
      global_step=1,
      k=2,
      step_complete=False,
      counts=None,
      items=None,
      generated=None,
      training_state=None,
      omit_training_state=False,
      num_generations=None,
  ):
    """Saves a fixed-k-shaped snapshot from an iter_steps-style position.

    In standard mode iter_steps == train_steps * k + local, so the helper
    decodes (train_steps, local) = divmod(iter_steps, k) and drives the real
    encoded-key save API with it. Packed-mode (ragged) key placement is
    exercised by the tests that call `self.mgr.save` directly.
    """
    train_steps, local = divmod(iter_steps, k)
    self.mgr.save(
        train_steps,
        local,
        iter_steps=iter_steps,
        global_step=global_step,
        grad_accum_steps=k,
        step_complete=step_complete,
        completed_group_ids=generated or [],
        trained_trajectory_counts=counts or {},
        active_group_trajectories=items or [],
        training_state=(
            None if omit_training_state else (training_state or _state())
        ),
        num_generations=num_generations,
    )

  def test_save_and_restore_roundtrip(self):
    item = _make_trajectory_item(group_id=10, pair_index=0, with_env=True)
    state = _state(acc=2.5, mini_step=1)
    # k=2, snapshot after micro-step 3 (mid-window of train_steps=1).
    self._save(
        3,
        global_step=4,
        k=2,
        counts={(10, 0): 1},
        items=[item],
        generated=[10, 20],
        training_state=state,
    )

    restored = self.mgr.try_restore(
        train_steps=1,
        grad_accum_steps=2,
        target_training_state=state,
    )
    self.assertIsNotNone(restored)
    assert restored is not None
    self.assertEqual(restored.iter_steps, 3)
    self.assertEqual(restored.global_step, 4)
    self.assertEqual(restored.grad_accum_steps, 2)
    self.assertFalse(restored.step_complete)
    self.assertEqual(restored.completed_group_ids, [10, 20])
    self.assertEqual(restored.trained_trajectory_counts, {(10, 0): 1})
    self.assertLen(restored.active_group_trajectories, 1)
    r = restored.active_group_trajectories[0]
    self.assertEqual((r.group_id, r.pair_index), (10, 0))
    np.testing.assert_array_equal(
        r.traj.steps[0].assistant_tokens, np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(r.traj.steps[0].env_masks, np.array([1, 0]))
    np.testing.assert_equal(
        restored.training_state["acc_grads"]["w"], np.array([2.5])
    )

  def test_benchmark_scalars_ride_custom_metadata(self):
    """BENCHMARK instrumentation (temporary): sub_step_elapsed and

    experiment_start_time round-trip via custom_metadata -- no
    checkpointables schema change -- and default to 0.0 when a snapshot
    predates the instrumentation (save called without them).
    """
    state = _state()
    self.mgr.save(
        1,
        0,
        iter_steps=2,
        global_step=1,
        grad_accum_steps=2,
        step_complete=False,
        completed_group_ids=[],
        trained_trajectory_counts={},
        active_group_trajectories=[],
        training_state=state,
        sub_step_elapsed=12.5,
        experiment_start_time=1000.0,
    )
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.sub_step_elapsed, 12.5)
    self.assertEqual(st.experiment_start_time, 1000.0)
    # Defaulted when not passed (pre-instrumentation compatibility).
    self._save(3, k=2, training_state=state)
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertEqual(st.sub_step_elapsed, 0.0)
    self.assertEqual(st.experiment_start_time, 0.0)

  def test_window_selection_picks_latest_within_window(self):
    """For restored weights at train_steps=T, only snapshots whose key decodes

    to parent T are valid (accumulated on exactly those weights); the latest
    (max local) loses the least work. Other parents must never be selected:
    earlier ones are stale, later ones contain gradients the restored weights
    never received. Exercised here through the fixed-k-shaped helper; the
    ragged case is test_ragged_window_selection_has_no_k_dependence.
    """
    k = 4
    state = _state()
    # Window of T=2 is [8, 12). Also plant a snapshot in the next window (12)
    # simulating the sub-batch stream having run durably ahead of the trainer.
    for j in (8, 9, 10, 12):
      self._save(j, k=k, counts={(1, 0): j}, training_state=state)
    self.mgr.wait()

    # Ordering matters: each restore purges keys above its resume point, so
    # check the highest window first.
    with self.subTest("apply-boundary key belongs to its own window"):
      st = self.mgr.try_restore(
          train_steps=3, grad_accum_steps=k, target_training_state=state
      )
      self.assertIsNotNone(st)
      self.assertEqual(st.iter_steps, 12)
    with self.subTest("latest within window"):
      st = self.mgr.try_restore(
          train_steps=2, grad_accum_steps=k, target_training_state=state
      )
      self.assertIsNotNone(st)
      self.assertEqual(st.iter_steps, 10)
      self.assertEqual(st.trained_trajectory_counts, {(1, 0): 10})
    with self.subTest("no snapshot in window -> None"):
      st = self.mgr.try_restore(
          train_steps=1, grad_accum_steps=k, target_training_state=state
      )
      self.assertIsNone(st)

  def test_default_options_are_sub_batch_defaults_not_peft_defaults(self):
    """Constructing with options=None must resolve to the SUB-BATCH defaults.

    Inheriting the base class blindly would hand this stream the PEFT defaults,
    whose ContinuousCheckpointingPolicy(180s) silently SKIPS most saves -- a
    skipped snapshot is a missing key the restore window can never select. This
    is the trap the dedicated resolver closes.
    """
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(d)
      policy = mgr._options.save_decision_policy
      self.assertIsInstance(
          policy, ocp.training.save_decision_policies.FixedIntervalPolicy
      )
      preserve = mgr._options.preservation_policy
      self.assertIsInstance(  # window-counting, not snapshot-counting
          preserve, sub_batch_checkpoint.SubBatchPreservationPolicy
      )
      self.assertEqual(preserve.windows_to_keep, 2)
      self.assertTrue(mgr._options.enable_async_checkpointing)
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_partial_options_fill_only_none_fields(self):
    """A caller-set field (the drill's enable_async_checkpointing=False)

    must survive resolution; everything left None gets sub-batch defaults.
    """
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
          d,
          options=checkpoint_options.TunixCheckpointingOptions(
              enable_async_checkpointing=False,
          ),
      )
      self.assertFalse(mgr._options.enable_async_checkpointing)  # respected
      self.assertIsInstance(  # filled
          mgr._options.save_decision_policy,
          ocp.training.save_decision_policies.FixedIntervalPolicy,
      )
      self.assertIsInstance(
          mgr._options.preservation_policy,
          sub_batch_checkpoint.SubBatchPreservationPolicy,
      )
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_run_options_carry_async_but_never_policies(self):
    """The manager accepts the RUN's (trainer-stream) options and extracts

    only the stream-neutral fields itself: the async toggle carries over,
    the run's save/preservation policies never do -- they are trainer
    tuning, and the drill's config even holds a non-orbax placeholder
    policy that would crash the Checkpointer if trusted.
    """
    d = tempfile.mkdtemp()
    try:
      run_options = SimpleNamespace(
          enable_async_checkpointing=False,
          save_decision_policy=SimpleNamespace(
              interval=1
          ),  # NOT an orbax policy
          preservation_policy=SimpleNamespace(n=1),
      )
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
          d, run_options=run_options
      )
      self.assertFalse(mgr._options.enable_async_checkpointing)  # carried
      self.assertIsInstance(  # NOT carried: sub-batch's own policy
          mgr._options.save_decision_policy,
          ocp.training.save_decision_policies.FixedIntervalPolicy,
      )
      self.assertIsInstance(
          mgr._options.preservation_policy,
          sub_batch_checkpoint.SubBatchPreservationPolicy,
      )
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_preservation_policy_keeps_top_two_windows_whole(self):
    """The window-counting retention: every snapshot of the newest two

    parents survives -- however many each holds -- and older parents drop
    entirely. This is the requirement LatestN could only approximate with a
    per-mode snapshot bound that packing cannot supply tightly.
    """
    policy = sub_batch_checkpoint.SubBatchPreservationPolicy()
    KB = sub_batch_checkpoint.KEY_BASE
    cks = [
        SimpleNamespace(step=st)
        for st in (
            0,
            1,  # parent 0: 2 snapshots
            1 * KB,
            1 * KB + 1,
            1 * KB + 17,  # parent 1: ragged, 3 snapshots
            2 * KB,  # parent 2: 1 snapshot
        )
    ]
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, False, True, True, True, True])

  def test_preservation_policy_counts_distinct_parents_not_adjacency(self):
    """Gapped parents (coarse trainer cadence, declined windows) must not

    strand retention: the newest two DISTINCT parents are kept, whatever
    their numeric distance.
    """
    policy = sub_batch_checkpoint.SubBatchPreservationPolicy()
    KB = sub_batch_checkpoint.KEY_BASE
    cks = [SimpleNamespace(step=st) for st in (3 * KB, 7 * KB, 7 * KB + 4)]
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [True, True, True])
    cks.append(SimpleNamespace(step=9 * KB))
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, True, True, True])  # parent 3 aged out

  def test_ragged_retention_end_to_end_through_real_manager(self):
    """Against the real checkpointer: a ragged window with more snapshots

    than any fixed-k bound would predict survives INTACT while an older
    window is garbage-collected -- the case a count-based default handled
    only by gross over-retention.
    """
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(d)

      def save(t, local):
        mgr.save(
            t,
            local,
            iter_steps=t * 10 + local,
            global_step=t,
            grad_accum_steps=2,
            step_complete=False,
            completed_group_ids=[],
            trained_trajectory_counts={},
            active_group_trajectories=[],
            training_state=None,
        )

      for local in (0, 1):
        save(0, local)
      for local in (0, 1, 5, 17):  # ragged: far beyond k=2
        save(1, local)
      save(2, 0)
      mgr.wait()
      KB = sub_batch_checkpoint.KEY_BASE
      on_disk = sorted(ck.step for ck in mgr._checkpointer.checkpoints)
      self.assertEqual(
          on_disk,
          [1 * KB, 1 * KB + 1, 1 * KB + 5, 1 * KB + 17, 2 * KB],
      )
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_ragged_window_selection_has_no_k_dependence(self):
    """The packed-mode case the encoding exists for: a window's micro-step

    count is data-dependent (the packer flushes on a token budget), so locals
    can run far past grad_accum_steps. Selection must key purely on the
    parent train_step -- any k-derived window arithmetic would reject these
    keys or, worse, misfile them under a neighboring apply.
    """
    state = _state()
    for local in (0, 1, 5, 17):  # 17 >> k=2: impossible under fixed-k keying
      self.mgr.save(
          1,
          local,
          iter_steps=10 + local,
          global_step=1,
          grad_accum_steps=2,
          step_complete=False,
          completed_group_ids=[],
          trained_trajectory_counts={(1, 0): local},
          active_group_trajectories=[],
          training_state=state,
      )
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.sub_batch_key, 1 * sub_batch_checkpoint.KEY_BASE + 17)
    self.assertEqual(st.iter_steps, 27)  # payload, not derived from the key
    self.assertEqual(st.trained_trajectory_counts, {(1, 0): 17})
    # A different parent sees none of them.
    self.assertIsNone(self.mgr._select_step(2))

  def test_local_overflow_raises_at_save(self):
    """local >= KEY_BASE would corrupt the parent train_step's digits and

    reconcile the snapshot onto the wrong weights; it can only mean the
    caller's per-window counter is not resetting at applies.
    """
    with self.assertRaisesRegex(ValueError, "local index"):
      self.mgr.save(
          1,
          sub_batch_checkpoint.KEY_BASE,
          iter_steps=1,
          global_step=1,
          grad_accum_steps=2,
          step_complete=False,
          completed_group_ids=[],
          trained_trajectory_counts={},
          active_group_trajectories=[],
          training_state=None,
      )

  def test_overwrite_allows_resume_reexecution(self):
    """A resumed run re-executes the same micro-steps and re-saves the same

    iter_steps keys. That must replace the crashed run's snapshot instead of
    raising StepAlreadyExistsError (the failure mode that killed real TPU
    runs under the previous encoded-step scheme).
    """
    state = _state()
    self._save(5, k=1, global_step=5, counts={(1, 0): 1}, training_state=state)
    self.mgr.wait()
    # Same key, new content -- as a resumed run would produce.
    self._save(5, k=1, global_step=5, counts={(1, 0): 2}, training_state=state)
    self.mgr.wait()

    st = self.mgr.try_restore(
        train_steps=5, grad_accum_steps=1, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.trained_trajectory_counts, {(1, 0): 2})

  def test_grad_accum_mismatch_declines(self):
    """A config change to gradient_accumulation_steps across a restart changes

    the window's item composition; restore must decline rather than inject a
    buffer built for a different apply schedule. Under the encoded keying the
    parent train_step still matches across the change (that is the point of
    reading it from the trainer), so the meta compat check is the only line
    of defense -- selection cannot save us.
    """
    state = _state()
    self._save(3, k=2, training_state=state)  # parent T=1, saved under k=2
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=4, target_training_state=state
    )
    self.assertIsNone(st)
    # The decline must also purge the old-k lineage. Otherwise those keys
    # survive to win max-selection under this parent, so every subsequent
    # restore declines identically and the run forfeits mid-step resume for
    # good.
    self.assertIsNone(self.mgr._select_step(1))

  def test_step_complete_roundtrip(self):
    state = _state()
    self._save(6, k=2, global_step=3, step_complete=True, training_state=state)
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=3, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertTrue(st.step_complete)
    self.assertEqual(st.global_step, 3)

  def test_any_fields_round_trip_exactly(self):
    """End-to-end against the real Orbax checkpointer: the fidelity contract

    is one tier -- every trajectory component restores exactly as saved.
    Types the old leaf-conversion layer degraded one-way (unicode arrays,
    bytes, tuples) now come back identical through the pickled payload.
    """
    item = _make_trajectory_item(group_id=3, pair_index=1, with_env=True)
    item.traj.steps[0].observation = np.array(["state text"])  # <U array
    item.traj.steps[0].info = {"raw": np.array([b"a", b"bb"])}  # |S array
    item.traj.steps[0].action = agent_types.Action(action=b"act")  # bytes
    item.metadata = {"pair": (1, "two"), "blob": b"\x00\xff"}
    state = _state()

    self._save(2, k=2, counts={(3, 1): 1}, items=[item], training_state=state)
    self.mgr.wait()  # re-raises any background save failure

    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    r = st.active_group_trajectories[0]
    self.assertEqual((r.group_id, r.pair_index), (3, 1))
    np.testing.assert_array_equal(
        r.traj.steps[0].assistant_tokens, np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(
        r.traj.steps[0].observation, np.array(["state text"])
    )
    np.testing.assert_array_equal(
        r.traj.steps[0].info["raw"], np.array([b"a", b"bb"])
    )
    self.assertEqual(r.traj.steps[0].action.action, b"act")  # bytes stay bytes
    self.assertEqual(r.metadata["pair"], (1, "two"))  # tuples stay tuples
    self.assertEqual(r.metadata["blob"], b"\x00\xff")

  def test_serialized_item_is_native_identity_plus_pickled_payload(self):
    """The wire format: identity fields stay native Orbax leaves (restore

    logic compares them without unpickling anything), everything else is one
    uint8 pickle blob.
    """
    item = _make_trajectory_item(group_id=1, pair_index=2, with_env=True)
    item.metadata = {"obj": {"nested": (1, 2)}}

    ser = sub_batch_checkpoint._trajectory_item_to_serializable(item)
    self.assertEqual(
        set(ser), {"group_id", "pair_index", "start_step", "payload"}
    )
    self.assertEqual(ser["group_id"], 1)
    self.assertEqual(ser["pair_index"], 2)
    self.assertIsInstance(ser["payload"], np.ndarray)
    self.assertEqual(ser["payload"].dtype, np.uint8)
    back = sub_batch_checkpoint._trajectory_item_from_serializable(ser)
    self.assertEqual(back.metadata, {"obj": {"nested": (1, 2)}})
    np.testing.assert_array_equal(
        back.traj.steps[0].assistant_tokens, item.traj.steps[0].assistant_tokens
    )

  def test_retention_purges_old_snapshots(self):
    """LatestN retention drops the oldest keys; the restore window for recent

    weights survives.
    """
    state = _state()
    for j in range(1, 13):  # LatestN(10) -> 1 and 2 purged
      self._save(j, k=2, counts={(1, 0): j}, training_state=state)
    self.mgr.wait()

    st = self.mgr.try_restore(
        train_steps=5, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 11)
    # The purged window (train_steps=0 -> keys 0,1) is gone.
    st = self.mgr.try_restore(
        train_steps=0, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNone(st)

  def test_restore_purges_dead_lineage_above_resume_point(self):
    """Keys above the resume point belong to a crashed run whose weights past

    that point never became durable. If they survive, a later restore whose
    window reaches them injects a buffer accumulated on divergent weights
    (silent training corruption). try_restore must delete them.
    """
    k = 4
    state = _state()
    # Run A: window [8,12) plus run-ahead keys 12, 13 (its T=3 weights never
    # became durable). Trainer restores T=2.
    for j in (9, 10, 11, 12, 13):
      self._save(j, k=k, counts={(1, 0): j}, training_state=state)
    self.mgr.wait()

    st = self.mgr.try_restore(
        train_steps=2, grad_accum_steps=k, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 11)
    # Run B crashes before re-reaching key 13; a restore at T=3 must not see
    # run A's stale 12/13.
    st = self.mgr.try_restore(
        train_steps=3, grad_accum_steps=k, target_training_state=state
    )
    self.assertIsNone(st)

  def test_declined_restore_purges_stale_lineage(self):
    """A fresh-start decline (empty window) must also purge keys above the

    resume point so they cannot be selected by a later restore.
    """
    k = 4
    state = _state()
    self._save(13, k=k, training_state=state)  # stale run-ahead key
    self.mgr.wait()

    # Window of T=2 is [8,12): empty -> decline, and 13 must be purged.
    self.assertIsNone(
        self.mgr.try_restore(
            train_steps=2, grad_accum_steps=k, target_training_state=state
        )
    )
    self.assertIsNone(
        self.mgr.try_restore(
            train_steps=3, grad_accum_steps=k, target_training_state=state
        )
    )

  def test_bufferless_snapshot_roundtrips_without_state(self):
    """Snapshots taken at apply boundaries (or on runs with no MultiSteps

    accumulator) carry no `state` checkpointable at all. Restore must work
    with or without a caller-provided target and return training_state None,
    never attempting to load the absent checkpointable.
    """
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    with self.subTest("no target provided"):
      st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
      self.assertIsNotNone(st)
      self.assertIsNone(st.training_state)
      self.assertEqual(st.trained_trajectory_counts, {(1, 0): 1})
    with self.subTest("target provided but ignored"):
      st = self.mgr.try_restore(
          train_steps=1, grad_accum_steps=2, target_training_state=_state()
      )
      self.assertIsNotNone(st)
      self.assertIsNone(st.training_state)

  def test_buffer_snapshot_without_target_declines(self):
    """A snapshot that carries a buffer cannot be restored without a shaped

    abstract target: an unshaped load stringifies the buffer's tuple path
    keys (verified: ('w',) comes back as the string \"('w',)\"), which would
    corrupt injection downstream. Restore must decline, not crash and not
    return mangled keys.
    """
    self._save(
        3,
        k=2,
        training_state={
            "acc_grads": {("w",): np.ones(4)},
            "mini_step": np.array(1),
        },
    )
    self.mgr.wait()
    self.assertIsNone(self.mgr.try_restore(train_steps=1, grad_accum_steps=2))

  def test_dict_trajectory_roundtrips_verbatim(self):
    """Token-mode collection produces plain-dict trajectories with their own

    schema; restore must not rebuild them as Trajectory dataclasses (which
    silently discards every field).
    """
    item = agent_types.TrajectoryItem(
        group_id=7,
        pair_index=0,
        start_step=0,
        traj={
            "conversation_text": "hello",
            "prompt_tokens": np.array([1, 2]),
            "conversation_masks": np.array([1, 1]),
            "trajectory_reward": 0.5,
        },
        metadata={},
    )
    state = _state()
    self._save(2, k=2, items=[item], training_state=state)
    self.mgr.wait()

    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    r = st.active_group_trajectories[0]
    self.assertIsInstance(r.traj, dict)
    self.assertEqual(r.traj["conversation_text"], "hello")
    np.testing.assert_array_equal(r.traj["prompt_tokens"], np.array([1, 2]))
    self.assertEqual(r.traj["trajectory_reward"], 0.5)

  def test_non_roundtrippable_group_id_fails_at_save(self):
    """Tuple group ids restore as lists (unhashable ledger keys) and bytes

    cannot be stored at all; save must fail loudly instead of bricking the
    restore.
    """
    state = _state()
    with self.assertRaises(ValueError):
      self._save(1, k=1, counts={(("a", 3), 0): 1}, training_state=state)
    with self.assertRaises(ValueError):
      self._save(1, k=1, generated=[b"gid"], training_state=state)
    # numpy ints are normalized, not rejected.
    self._save(1, k=1, counts={(np.int64(5), 0): 1}, training_state=state)
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=1, target_training_state=state
    )
    self.assertEqual(st.trained_trajectory_counts, {(5, 0): 1})

  def test_jax_arrays_normalize_and_exotic_leaves_round_trip(self):
    """jax array leaves are normalized to numpy before pickling (payload

    fidelity must not depend on jax's pickle behavior across versions);
    datetime64 arrays and non-str dict keys -- which the old conversion layer
    degraded to strings -- now round-trip exactly.
    """
    item = _make_trajectory_item(group_id=1, pair_index=0)
    when = np.array(["2026-07-11T10:00"], dtype="datetime64[s]")
    item.metadata = {"jax_leaf": jnp.array([1, 2, 3]), "when": when, 7: "x"}
    back = sub_batch_checkpoint._trajectory_item_from_serializable(
        sub_batch_checkpoint._trajectory_item_to_serializable(item)
    )
    self.assertIsInstance(back.metadata["jax_leaf"], np.ndarray)
    np.testing.assert_array_equal(back.metadata["jax_leaf"], [1, 2, 3])
    np.testing.assert_array_equal(back.metadata["when"], when)
    self.assertEqual(back.metadata[7], "x")  # int key stays an int key

  def test_decline_purges_poison_key(self):
    """A snapshot the current run cannot restore (here: carries a buffer but

    no abstract target is supplied) must be purged on decline, or it stays
    the window's max and every subsequent restore declines identically -- a
    fresh-start livelock until the rerun happens to overwrite it.
    """
    state = {
        "acc_grads": {("w",): np.ones(4)},
        "mini_step": np.array(1),
    }
    self._save(3, k=2, counts={(1, 0): 1}, training_state=state)
    self._save(2, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    # First restore declines on the buffer-without-target key 3 and purges it.
    self.assertIsNone(self.mgr.try_restore(train_steps=1, grad_accum_steps=2))
    # Second restore now selects the usable bufferless key 2 underneath.
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 2)

  def test_structural_drift_decline_purges_instead_of_crashing(self):
    """A buffer snapshot whose tree no longer matches the caller's abstract

    target (e.g. an optimizer refactor across a restart) must decline+purge,
    not propagate: the load exception would otherwise crash __init__ on
    every subsequent restart until the snapshot dir is hand-deleted.
    """
    state = {
        "acc_grads": {("w",): np.ones(4)},
        "mini_step": np.array(1),
    }
    self._save(3, k=2, counts={(1, 0): 1}, training_state=state)
    self._save(2, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    drifted_target = {
        "acc_grads": {("v",): np.zeros(8)},  # wrong key AND wrong shape
        "mini_step": np.array(0),
    }
    self.assertIsNone(
        self.mgr.try_restore(
            train_steps=1,
            grad_accum_steps=2,
            target_training_state=drifted_target,
        )
    )
    # The poison key 3 was purged; the bufferless key 2 restores next.
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 2)

  def test_out_of_int64_metadata_int_round_trips_exactly(self):
    """A 128-bit int (uuid4().int-style trace ids in Any-typed metadata) used

    to crash the save one micro-step late unless stringified; inside the
    pickled payload it now round-trips as the exact int.
    """
    big = (1 << 122) + 17
    item = _make_trajectory_item(group_id=10, pair_index=0)
    item.metadata["trace_id"] = big
    self._save(
        2, k=2, counts={(10, 0): 1}, items=[item], omit_training_state=True
    )
    self.mgr.wait()  # the save (incl. async finalize) must not raise
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    self.assertEqual(st.active_group_trajectories[0].metadata["trace_id"], big)

  def test_zero_size_arrays_round_trip_with_dtype_and_shape(self):
    """Orbax raises 'Cannot save arrays with zero size', and the rollout

    engine really emits these (conversation_tokens/masks are empty int32
    arrays on an immediate-EOS rollout). Inside the pickled payload they are
    just arrays -- no placeholder machinery -- but dtype and shape surviving
    is still the load-bearing property: a float64 come-back would upcast the
    token arrays it gets concatenated with.
    """
    dict_item = agent_types.TrajectoryItem(
        group_id=1,
        pair_index=0,
        start_step=0,
        traj={
            "conversation_tokens": np.array([], dtype=np.int32),
            "conversation_masks": np.array([], dtype=np.int32),
            "trajectory_reward": 0.0,
        },
        metadata={
            "empty_2d": np.zeros((0, 3), dtype=np.float32),
            "empty_bf16": jnp.zeros((0,), dtype=jnp.bfloat16),
        },
    )
    dataclass_item = _make_trajectory_item(group_id=2, pair_index=0)
    dataclass_item.traj.steps[0].assistant_tokens = np.array([], dtype=np.int32)

    self._save(
        2, k=2, items=[dict_item, dataclass_item], omit_training_state=True
    )
    self.mgr.wait()  # the save itself must not raise

    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    restored_dict, restored_dc = st.active_group_trajectories
    for key in ("conversation_tokens", "conversation_masks"):
      arr = restored_dict.traj[key]
      self.assertIsInstance(arr, np.ndarray)
      self.assertEqual(arr.shape, (0,))
      self.assertEqual(arr.dtype, np.int32)
    # Shape and dtype survive inside Any-typed fields too, custom ml_dtypes
    # (bfloat16) included.
    meta_arr = restored_dict.metadata["empty_2d"]
    self.assertEqual(meta_arr.shape, (0, 3))
    self.assertEqual(meta_arr.dtype, np.float32)
    self.assertEqual(restored_dict.metadata["empty_bf16"].dtype, jnp.bfloat16)
    # ...and inside a reconstructed Trajectory dataclass's Step.
    self.assertEqual(restored_dc.traj.steps[0].assistant_tokens.shape, (0,))

  def test_out_of_int64_group_id_raises_at_save(self):
    """Group ids must round-trip verbatim (== with live producer ids), so an

    int beyond int64 cannot be silently stringified -- reject loudly at save
    time. np.uint64 above 2**63-1 normalizes into exactly this case.
    """
    for bad in ((1 << 64) + 3, np.uint64(2**63 + 5)):
      item = _make_trajectory_item(group_id=0, pair_index=0)
      item.group_id = bad
      with self.assertRaisesRegex(ValueError, "int64 range"):
        self._save(2, k=2, items=[item], omit_training_state=True)

  def test_empty_window_reports_surviving_evidence_before_purging(self):
    """An empty restore window with surviving snapshots must be reported with

    the on-disk FACTS, BEFORE the fresh-path purge deletes them (the earlier
    detector inspected only below-window keys after the purge, which made the
    retention-outran-cadence case structurally unreportable). The message is
    hedged: the evidence cannot distinguish the durability race from
    retention from the inverse race where the step actually completed.
    """
    self._save(
        3, k=2, global_step=7, counts={(1, 0): 1}, omit_training_state=True
    )
    # An above-window dead-lineage key too: the report must see it BEFORE the
    # fresh-path purge deletes it (the old detector ran post-purge and could
    # therefore never report the retention-outran-cadence case).
    self._save(
        13, k=2, global_step=8, counts={(2, 0): 1}, omit_training_state=True
    )
    self.mgr.wait()
    with self.assertLogs(level="WARNING") as logs:
      # Trainer restored T=5: no key with parent 5; parents 1 and 6 survive.
      self.assertIsNone(self.mgr.try_restore(train_steps=5, grad_accum_steps=2))
    joined = "\n".join(logs.output)
    self.assertIn("no usable snapshot", joined)
    # Latest survivor (key 13 = parent 6, local 1), captured pre-purge and
    # reported in decoded form.
    self.assertIn("train_step 6, local 1", joined)
    self.assertIn("benign", joined)  # hedged, not asserting a single cause
    # The purge then removed the above-window dead lineage (below-window keys
    # are unreachable by forward-moving restores and left to retention).
    self.assertIsNone(self.mgr._select_step(6))

  def test_empty_window_with_clean_boundary_reports_info_not_warning(self):
    """When the latest survivor marks a cleanly completed step, the report

    must not cry wolf: INFO, not WARNING (an operator triaging preemptions
    should not be told training work was discarded when none was).
    """
    self._save(
        4, k=2, global_step=7, step_complete=True, omit_training_state=True
    )
    self.mgr.wait()
    with self.assertRaises(AssertionError):
      with self.assertLogs(level="WARNING"):
        self.mgr.try_restore(train_steps=5, grad_accum_steps=2)

  def test_torn_meta_declines_instead_of_crashing_startup(self):
    """A step directory whose meta checkpointable is unreadable (crash

    mid-delete leaves torn directories that still enumerate) must decline
    and purge, not raise out of try_restore: raising would crash __init__ on
    every restart until someone hand-deletes the directory.
    """
    import os

    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    # Tear the step: remove its meta subdirectory only.
    step_dir = None
    for name in os.listdir(self.test_dir):
      candidate = os.path.join(self.test_dir, name, "meta")
      if os.path.isdir(candidate):
        step_dir = candidate
    self.assertIsNotNone(step_dir, "expected a committed step with meta/")
    shutil.rmtree(step_dir)

    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNone(st)  # declined, not raised
    # And purged: a second restore finds nothing rather than repeating.
    self.assertIsNone(self.mgr._select_step(1))

  def test_num_generations_mismatch_declines(self):
    """The ledger's per-pair accounting is only valid under the saved

    num_generations; a config change must decline like a k change.
    """
    self._save(
        3, k=2, counts={(1, 0): 1}, omit_training_state=True, num_generations=2
    )
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, num_generations=4
    )
    self.assertIsNone(st)
    # As with the k-mismatch decline, the stale-geometry key must be purged or
    # it stays the window's max and poisons every later restore.
    self.assertIsNone(self.mgr._select_step(1))


if __name__ == "__main__":
  absltest.main()
