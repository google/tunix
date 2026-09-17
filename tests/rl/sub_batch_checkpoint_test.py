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

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_options


def _make_trajectory_item(
    prompt_id: int,
    group_index: int,
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
      prompt_id=prompt_id,
      group_index=group_index,
      start_step=0,
      traj=traj,
      metadata={"test_key": "test_value"},
  )


KB = sub_batch_checkpoint.KEY_BASE


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
    marker = self.mgr.enable_marker_path
    if marker.exists():
      marker.unlink()
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
      num_iterations=None,
      mini_batch_size=None,
      geometry=None,
      anchor_step=None,
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
        num_iterations=num_iterations,
        mini_batch_size=mini_batch_size,
        geometry=geometry,
        anchor_step=anchor_step,
    )

  def _keys(self, mgr=None):
    """Keys the manager enumerates (its checkpoints listing is cached per
    manager, so a hand-deleted directory shows only through a new one)."""
    mgr = mgr or self.mgr
    return sorted(ck.step for ck in mgr._checkpointer.checkpoints)

  def test_save_and_restore_roundtrip(self):
    item = _make_trajectory_item(prompt_id=10, group_index=0, with_env=True)
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
    self.assertEqual((r.prompt_id, r.group_index), (10, 0))
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
    with self.subTest("no snapshot in window -> LedgerMissing"):
      # Window 1 is empty while window 2 survives: a trainer at 1 whose
      # ledger 1.0 is gone is a lineage bug, not a fresh start.
      with self.assertRaises(sub_batch_checkpoint.SubBatchLedgerMissingError):
        self.mgr.try_restore(
            train_steps=1, grad_accum_steps=k, target_training_state=state
        )

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
      self.assertIsInstance(  # durable-floor, not snapshot-counting
          preserve, sub_batch_checkpoint.SubBatchPreservationPolicy
      )
      self.assertIsNone(preserve.durable_train_steps)  # preserve-all
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

  def test_preservation_policy_keeps_newest_per_parent_above_floor(self):
    """The durable-floor retention: parents below the reported floor drop

    entirely; every parent at or above it keeps only its newest key (the one
    restore selects), however ragged the window. Nothing reported => no
    parent drops, still one key each.
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
    # Every non-newest key of a window is dead weight in every case.
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, True, False, False, True, True])
    policy.durable_train_steps = 1
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, False, False, False, True, True])
    policy.durable_train_steps = 2
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, False, False, False, False, True])

  def test_preservation_policy_floor_is_numeric_not_positional(self):
    """Gapped parents (coarse trainer cadence, declined windows): a parent

    below the floor drops however close it is; one above stays whatever its
    distance. The floor is a train_steps value, not a count of parents.
    """
    policy = sub_batch_checkpoint.SubBatchPreservationPolicy()
    KB = sub_batch_checkpoint.KEY_BASE
    cks = [
        SimpleNamespace(step=st)
        for st in (3 * KB, 7 * KB, 7 * KB + 4, 9 * KB)
    ]
    policy.durable_train_steps = 7
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, False, True, True])
    policy.durable_train_steps = 8
    keep = policy.should_preserve(cks, context=None)
    self.assertEqual(keep, [False, False, False, True])

  def test_ragged_retention_end_to_end_through_real_manager(self):
    """Against the real checkpointer: a ragged window keeps only its newest

    key as it grows (each save retires the previous one, so no window is
    ever deleted whole), and the floor then drops an older window's last
    key at the next save.
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
            anchor_step=t,
        )

      for local in (0, 1):
        save(0, local)
      for local in (0, 1, 5, 17):  # ragged: far beyond k=2
        save(1, local)
      mgr.wait()
      KB = sub_batch_checkpoint.KEY_BASE
      on_disk = sorted(ck.step for ck in mgr._checkpointer.checkpoints)
      # Nothing reported yet: no parent drops, newest key each.
      self.assertEqual(on_disk, [1, 1 * KB + 17])
      # The learner's steady state: W(1) proven durable, floor reported,
      # then (T+1).0 precommitted -- that save drops parent 0.
      mgr.set_durable_train_steps(1)
      save(2, 0)
      mgr.wait()
      on_disk = sorted(ck.step for ck in mgr._checkpointer.checkpoints)
      self.assertEqual(on_disk, [1 * KB + 17, 2 * KB])
      # The newest key of the window is what restore selects.
      self.assertEqual(mgr._select_step(1), 1 * KB + 17)
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def _save_boundary(self, mgr, parent):
    mgr.save(
        parent,
        0,
        iter_steps=parent,
        global_step=0,
        grad_accum_steps=1,
        step_complete=False,
        completed_group_ids=[],
        trained_trajectory_counts={},
        active_group_trajectories=[],
        training_state=None,
        anchor_step=parent,
    )

  def test_retention_keeps_ledger_for_last_durable_trainer_checkpoint(self):
    """Reviewer regression: a standalone manager that was never told what is

    durable must not delete the only ledger matching the durable weights.
    Legal schedule: W1 durable; L2 precommitted; W2 saving; L3 precommitted
    before update_actor starts W3. Crash while W2 pending -> restore at 1.
    """
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
          d,
          options=checkpoint_options.TunixCheckpointingOptions(
              enable_async_checkpointing=False,
          ),
      )
      for parent in (1, 2, 3):
        self._save_boundary(mgr, parent)
        mgr.wait()
      state = mgr.try_restore(train_steps=1, grad_accum_steps=1)
      self.assertIsNotNone(
          state, "Retention deleted L1 while W1 remains the recovery anchor."
      )
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_floor_report_orders_gc(self):
    """Through the real async manager: the floor takes effect at the NEXT

    save (orbax evaluates retention inside save()), drops exactly the
    parents below it, and the restore window for the floor itself survives.
    """
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(d)
      for parent in (1, 2, 3):
        self._save_boundary(mgr, parent)
      mgr.set_durable_train_steps(2)
      self._save_boundary(mgr, 4)
      mgr.wait()
      KB = sub_batch_checkpoint.KEY_BASE
      on_disk = sorted(ck.step // KB for ck in mgr._checkpointer.checkpoints)
      self.assertEqual(on_disk, [2, 3, 4])
      st = mgr.try_restore(train_steps=2, grad_accum_steps=1)
      self.assertIsNotNone(st)
      self.assertEqual(st.sub_batch_key, 2 * KB)
      # Dead lineage above the chosen key is purged as usual; parent 1 was
      # dropped by the floor, so weights restored at 1 have no ledger while
      # parent 2 survives: that is the missing-window stop, not a decline.
      on_disk = sorted(ck.step // KB for ck in mgr._checkpointer.checkpoints)
      self.assertEqual(on_disk, [2])
      with self.assertRaises(sub_batch_checkpoint.SubBatchLedgerMissingError):
        mgr.try_restore(train_steps=1, grad_accum_steps=1)
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_set_durable_train_steps_never_lowers_the_floor(self):
    d = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(d)
      policy = mgr._options.preservation_policy
      mgr.set_durable_train_steps(3)
      self.assertEqual(policy.durable_train_steps, 3)
      mgr.set_durable_train_steps(1)  # a stale report must not re-widen
      self.assertEqual(policy.durable_train_steps, 3)
      mgr.set_durable_train_steps(5)
      self.assertEqual(policy.durable_train_steps, 5)
      mgr.close()
    finally:
      shutil.rmtree(d, ignore_errors=True)

  def test_set_durable_train_steps_noop_for_foreign_policy(self):
    """A caller-supplied preservation policy owns retention: the report is

    accepted and ignored, and the policy object is left untouched.
    """
    self.assertIsInstance(
        self.mgr._options.preservation_policy,
        ocp.training.preservation_policies.LatestN,
    )
    before = repr(self.mgr._options.preservation_policy)
    self.mgr.set_durable_train_steps(5)
    self.assertEqual(repr(self.mgr._options.preservation_policy), before)
    self.assertFalse(
        hasattr(self.mgr._options.preservation_policy, "durable_train_steps")
    )

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
          anchor_step=1,
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

  def test_grad_accum_mismatch_raises_and_preserves(self):
    """A config change to gradient_accumulation_steps across a restart changes
    the window's item composition; restore must RAISE rather than inject a
    buffer built for a different apply schedule (resume assumes relaunch
    with the same configuration). Under the encoded keying the parent
    train_step still matches across the change (that is the point of reading
    it from the trainer), so the meta geometry check is the only line of
    defense -- selection cannot save us. The snapshots must survive the
    raise: a corrected relaunch resumes from them."""
    state = _state()
    self._save(3, k=2, training_state=state)  # parent T=1, saved under k=2
    self.mgr.wait()
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "grad_accum_steps"
    ):
      self.mgr.try_restore(
          train_steps=1, grad_accum_steps=4, target_training_state=state
      )
    # Nothing purged: relaunching with the ORIGINAL k resumes normally.
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 3)

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
    item = _make_trajectory_item(prompt_id=3, group_index=1, with_env=True)
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
    self.assertEqual((r.prompt_id, r.group_index), (3, 1))
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
    item = _make_trajectory_item(prompt_id=1, group_index=2, with_env=True)
    item.metadata = {"obj": {"nested": (1, 2)}}

    ser = sub_batch_checkpoint._trajectory_item_to_serializable(item)
    self.assertEqual(
        set(ser), {"prompt_id", "group_index", "start_step", "payload"}
    )
    self.assertEqual(ser["prompt_id"], 1)
    self.assertEqual(ser["group_index"], 2)
    self.assertIsInstance(ser["payload"], np.ndarray)
    self.assertEqual(ser["payload"].dtype, np.uint8)
    back = sub_batch_checkpoint._trajectory_item_from_serializable(ser)
    self.assertEqual(back.metadata, {"obj": {"nested": (1, 2)}})
    np.testing.assert_array_equal(
        back.traj.steps[0].assistant_tokens, item.traj.steps[0].assistant_tokens
    )

  def test_serialize_reads_canonical_fields_not_aliases(self):
    """TrajectoryItem swallows unknown keywords into metadata and its

    __getattr__ falls back to traj-dict keys then metadata, so a serializer
    reading the legacy names (group_id / pair_index) would silently pick up
    aliases. Identity must come from the canonical fields only -- on a
    Trajectory-dataclass traj too, where no traj-dict fallback exists.
    """
    item = _make_trajectory_item(prompt_id=7, group_index=1)
    item.metadata["group_id"] = 99
    item.metadata["pair_index"] = 5
    self.assertEqual(item.group_id, 99)  # the alias resolves...
    ser = sub_batch_checkpoint._trajectory_item_to_serializable(item)
    self.assertEqual(ser["prompt_id"], 7)  # ...but is never read
    self.assertEqual(ser["group_index"], 1)
    self.assertNotIn("group_id", ser)
    self.assertNotIn("pair_index", ser)
    back = sub_batch_checkpoint._trajectory_item_from_serializable(ser)
    self.assertEqual(back.traj_id, "traj_7_g1")
    self.assertEqual((back.prompt_id, back.group_index), (7, 1))
    # Constructor keywords take the same route (kwargs -> metadata).
    aliased = agent_types.TrajectoryItem(
        prompt_id=7, group_index=1, group_id=99, pair_index=5, traj={}
    )
    ser = sub_batch_checkpoint._trajectory_item_to_serializable(aliased)
    self.assertEqual((ser["prompt_id"], ser["group_index"]), (7, 1))

  def test_token_payload_item_without_pair_index_serializes(self):
    """An item exactly as the rollout orchestrator builds it in Token mode:

    the traj dict carries `group_id` (env kwarg) and nothing carries
    `pair_index`; identity is only in the canonical fields.
    """
    item = agent_types.TrajectoryItem(
        prompt_id=7,
        group_index=1,
        start_step=0,
        traj={
            "group_id": 7,
            "conversation_tokens": np.array([1, 2, 3]),
            "conversation_masks": np.array([1, 1, 1]),
            "trajectory_reward": 1.0,
        },
        metadata={"generation_id": 1},
    )
    self._save(2, k=2, items=[item], omit_training_state=True)
    self.mgr.wait()
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    r = st.active_group_trajectories[0]
    self.assertEqual((r.prompt_id, r.group_index), (7, 1))
    self.assertIsInstance(r.prompt_id, int)
    self.assertEqual(r.traj_id, "traj_7_g1")
    self.assertEqual(r.traj["group_id"], 7)

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
    # The purged window (train_steps=0 -> keys 0,1) is gone. A trainer at
    # the anchor with parents >= 2 surviving is the retention-outran-the-
    # trainer signature: None (step 0 reruns from the anchor), but WARNING
    # naming retention, and the dead lineage above 0 is purged.
    with self.assertLogs(level="WARNING") as logs:
      st = self.mgr.try_restore(
          train_steps=0, grad_accum_steps=2, target_training_state=state
      )
    self.assertIsNone(st)
    self.assertIn("retention", "\n".join(logs.output))
    self.assertEqual(self._keys(), [])

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
    # run A's stale 12/13. Without run B's own 3.0 precommit that window is
    # empty while window 2 survives: a stop, nothing purged.
    with self.assertRaises(sub_batch_checkpoint.SubBatchLedgerMissingError):
      self.mgr.try_restore(
          train_steps=3, grad_accum_steps=k, target_training_state=state
      )
    self.assertEqual(self.mgr._select_step(2), 2 * KB + 3)
    # Run B precommits 3.0 before its apply; a restore at 3 then lands on it.
    self._save(12, k=k, counts={(1, 0): 12}, omit_training_state=True)
    self.mgr.wait()
    st = self.mgr.try_restore(train_steps=3, grad_accum_steps=k)
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 12)

  def test_missing_window_raises_with_evidence(self):
    """Trainer restored T>0 whose window is empty while other ledgers
    survive: T.0 is written durably before every apply, so this is a
    retention/lineage bug or a foreign root. The restore must STOP with the
    on-disk facts (latest survivor decoded, its meta) and purge nothing --
    a silent None here would start the next step and drop the remainder of
    the in-progress one."""
    self._save(
        3, k=2, global_step=7, counts={(1, 0): 1}, omit_training_state=True
    )
    self._save(
        13, k=2, global_step=8, counts={(2, 0): 1}, omit_training_state=True
    )
    self.mgr.wait()
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchLedgerMissingError,
        r"train_steps=5.*train_step 6, local 1",
    ) as ctx:
      # Trainer restored T=5: no key with parent 5; parents 1 and 6 survive.
      self.mgr.try_restore(train_steps=5, grad_accum_steps=2)
    msg = str(ctx.exception)
    self.assertIsInstance(
        ctx.exception, sub_batch_checkpoint.SubBatchRestoreError
    )
    self.assertIn("global_step=8", msg)
    self.assertIn(self.test_dir, msg)  # what to move or delete
    self.assertNotIn("fresh checkpoint root", msg)
    # Nothing purged, above or below the window.
    self.assertEqual(self._keys(), [KB + 1, 6 * KB + 1])

  def test_first_enable_restart_reuses_durable_baseline(self):
    with self.assertLogs(level="WARNING"):
      self.assertIsNone(self.mgr.try_restore(2, 2))
    self.assertEqual(self.mgr.enable_marker_path.read_text(), "2")
    self.mgr.close()
    self.mgr = sub_batch_checkpoint.SubBatchCheckpointManager(self.test_dir)
    self.assertIsNone(self.mgr.try_restore(2, 2))
    # A speculative next-parent ledger cannot advance the durable baseline.
    self._save(6, k=2, omit_training_state=True)
    self.assertIsNone(self.mgr.try_restore(2, 2))
    self.assertEqual(self._keys(), [])
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchLedgerMissingError, "ledger was lost"
    ):
      self.mgr.try_restore(3, 2)

  def test_whole_ledger_loss_keeps_enrollment_evidence(self):
    self.mgr.mark_enabled()
    self._save(3, k=2, omit_training_state=True)
    self.mgr.close()
    shutil.rmtree(self.test_dir)
    self.mgr = sub_batch_checkpoint.SubBatchCheckpointManager(self.test_dir)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchLedgerMissingError, "ledger was lost"
    ):
      self.mgr.try_restore(1, 2)
    self.assertEqual(self.mgr.enable_marker_path.read_text(), "0")

  def test_enrollment_baseline_is_immutable(self):
    self.mgr.mark_enabled(2)
    self.mgr.mark_enabled(2)
    with self.assertRaises(sub_batch_checkpoint.SubBatchLedgerMissingError):
      self.mgr.mark_enabled(3)
    self.assertEqual(self.mgr.enable_marker_path.read_text(), "2")

  def test_torn_enrollment_stops_before_purge(self):
    self.mgr.mark_enabled()
    self._save(3, k=2, omit_training_state=True)
    self.mgr.wait()
    self.mgr.enable_marker_path.write_text("")
    with self.assertRaises(sub_batch_checkpoint.SubBatchUnreadableError):
      self.mgr.try_restore(0, 2)
    self.assertEqual(self._keys(), [KB + 1])

  def test_marker_is_scoped_to_ledger_and_outside_it(self):
    self.mgr.mark_enabled()
    self.assertNotEqual(self.mgr.enable_marker_path.parent,
                        self.mgr._checkpointer.directory)
    self._save(3, k=2, omit_training_state=True)
    self.mgr.wait()
    self.assertEqual(self._keys(), [KB + 1])
    fresh = sub_batch_checkpoint.SubBatchCheckpointManager(self.test_dir)
    try:
      self.assertEqual(fresh._select_step(1), KB + 1)
      self.assertEqual(fresh._enabled_at(), 0)
    finally:
      fresh.close()

  def test_window_zero_empty_with_dead_lineage_returns_none(self):
    """Anchor restored (T=0), window 0 empty, a 1.0 precommit left over:
    the k=1 crash-before-the-first-apply-is-durable case. Step 0 reruns
    from the anchor and the leftover is dead lineage: purged, no WARNING
    (parent 1 is the healthy precommit). A leftover parent >= 2 can only
    mean retention dropped window 0 under the durable anchor: still None,
    still purged, but a WARNING naming retention."""
    with self.subTest("healthy precommit"):
      self.mgr.save(
          1, 0, iter_steps=1, global_step=0, grad_accum_steps=1,
          step_complete=False, completed_group_ids=[],
          trained_trajectory_counts={}, active_group_trajectories=[],
          training_state=None, anchor_step=0,
      )
      self.mgr.wait()
      with self.assertRaises(AssertionError):
        with self.assertLogs(level="WARNING"):
          self.assertIsNone(
              self.mgr.try_restore(train_steps=0, grad_accum_steps=1)
          )
      self.assertIsNone(self.mgr._select_step(1))
      self.assertEqual(self._keys(), [])
    with self.subTest("retention signature"):
      self._save(5, k=2, omit_training_state=True)  # parent 2
      self.mgr.wait()
      with self.assertLogs(level="WARNING") as logs:
        self.assertIsNone(
            self.mgr.try_restore(train_steps=0, grad_accum_steps=2)
        )
      self.assertIn("retention", "\n".join(logs.output))
      self.assertEqual(self._keys(), [])

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

  def test_buffer_snapshot_without_target_raises(self):
    """A snapshot that carries a buffer cannot be restored without a shaped

    abstract target: an unshaped load stringifies the buffer's tuple path
    keys (verified: ('w',) comes back as the string \"('w',)\"), which would
    corrupt injection downstream. A missing target means the trainer stack
    changed across the restart -- a geometry violation, so restore must
    RAISE, not decline and not return mangled keys."""
    self._save(
        3,
        k=2,
        training_state={
            "acc_grads": {("w",): np.ones(4)},
            "mini_step": np.array(1),
        },
    )
    self.mgr.wait()
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "no accumulator target"
    ):
      self.mgr.try_restore(train_steps=1, grad_accum_steps=2)

  def test_dict_trajectory_roundtrips_verbatim(self):
    """Token-mode collection produces plain-dict trajectories with their own

    schema; restore must not rebuild them as Trajectory dataclasses (which
    silently discards every field).
    """
    item = agent_types.TrajectoryItem(
        prompt_id=7,
        group_index=0,
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

  def test_non_roundtrippable_prompt_id_fails_at_save(self):
    """Tuple prompt ids restore as lists (unhashable ledger keys) and bytes

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
    item = _make_trajectory_item(prompt_id=1, group_index=0)
    when = np.array(["2026-07-11T10:00"], dtype="datetime64[s]")
    item.metadata = {"jax_leaf": jnp.array([1, 2, 3]), "when": when, 7: "x"}
    back = sub_batch_checkpoint._trajectory_item_from_serializable(
        sub_batch_checkpoint._trajectory_item_to_serializable(item)
    )
    self.assertIsInstance(back.metadata["jax_leaf"], np.ndarray)
    np.testing.assert_array_equal(back.metadata["jax_leaf"], [1, 2, 3])
    np.testing.assert_array_equal(back.metadata["when"], when)
    self.assertEqual(back.metadata[7], "x")  # int key stays an int key

  def test_geometry_mismatches_raise_per_field(self):
    """Every recorded geometry field raises on mismatch, none purges, and a
    field unknown on either side (older snapshot / caller without the value)
    is skipped -- None means 'not recorded', never 'changed'."""
    self._save(
        3, k=2, counts={(1, 0): 1}, omit_training_state=True,
        num_generations=4, num_iterations=2, mini_batch_size=8,
    )
    self.mgr.wait()
    for field, kwargs in (
        ("num_generations", dict(num_generations=2)),
        ("num_iterations", dict(num_iterations=1)),
        ("mini_batch_size", dict(mini_batch_size=4)),
    ):
      with self.subTest(field):
        with self.assertRaisesRegex(
            sub_batch_checkpoint.SubBatchGeometryError, field
        ):
          self.mgr.try_restore(train_steps=1, grad_accum_steps=2, **kwargs)
    with self.subTest("unknown fields are skipped, snapshots preserved"):
      st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
      self.assertIsNotNone(st)
      self.assertEqual(st.iter_steps, 3)
      self.assertEqual(st.geometry, {})  # saved without geometry

  def test_structural_drift_raises_and_preserves(self):
    """A buffer snapshot whose tree no longer matches the caller's abstract

    target (e.g. an optimizer refactor across a restart) must STOP, labelled
    as an incompatible tree, and purge nothing: silently falling back to an
    older key would drop trained chunks, and the original trainer stack
    restores it as-is.
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
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchUnreadableError, "incompatible"
    ) as ctx:
      self.mgr.try_restore(
          train_steps=1,
          grad_accum_steps=2,
          target_training_state=drifted_target,
      )
    # Names the same-window fallback the hand-delete would land on.
    self.assertIn(f"fall back to key {KB}", str(ctx.exception))
    self.assertEqual(self._keys(), [KB, KB + 1])
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 3)

  def test_out_of_int64_metadata_int_round_trips_exactly(self):
    """A 128-bit int (uuid4().int-style trace ids in Any-typed metadata) used

    to crash the save one micro-step late unless stringified; inside the
    pickled payload it now round-trips as the exact int.
    """
    big = (1 << 122) + 17
    item = _make_trajectory_item(prompt_id=10, group_index=0)
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
        prompt_id=1,
        group_index=0,
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
    dataclass_item = _make_trajectory_item(prompt_id=2, group_index=0)
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

  def test_out_of_int64_prompt_id_raises_at_save(self):
    """Prompt ids must round-trip verbatim (== with live producer ids), so an

    int beyond int64 cannot be silently stringified -- reject loudly at save
    time. np.uint64 above 2**63-1 normalizes into exactly this case.
    """
    for bad in ((1 << 64) + 3, np.uint64(2**63 + 5)):
      item = _make_trajectory_item(prompt_id=0, group_index=0)
      item.prompt_id = bad
      with self.assertRaisesRegex(ValueError, "int64 range"):
        self._save(2, k=2, items=[item], omit_training_state=True)

  def test_torn_meta_raises_and_preserves(self):
    """A step directory whose meta checkpointable is missing (a crash
    mid-delete leaves torn directories that still enumerate) must STOP,
    labelled torn, purge nothing, and name the same-window key a
    hand-delete falls back to (same weights, fewer trained chunks)."""
    self._save(2, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    torn = os.path.join(self.test_dir, str(KB + 1))
    shutil.rmtree(os.path.join(torn, "meta"))

    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchUnreadableError, "torn"
    ) as ctx:
      self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    msg = str(ctx.exception)
    self.assertIn(torn, msg)
    self.assertIn(f"fall back to key {KB}", msg)
    self.assertEqual(self.mgr._select_step(1), KB + 1)  # nothing purged
    # The operator's hand-delete, seen by a fresh process: the previous key
    # of the window restores.
    shutil.rmtree(torn)
    fresh = sub_batch_checkpoint.SubBatchCheckpointManager(self.test_dir)
    try:
      st = fresh.try_restore(train_steps=1, grad_accum_steps=2)
      self.assertIsNotNone(st)
      self.assertEqual(st.iter_steps, 2)
    finally:
      fresh.close()

  def test_torn_precommit_has_no_fallback(self):
    """A torn precommit (local 0) has no earlier key in its window, so the
    message must not suggest a hand-delete (at T>0 that lands on the
    missing-window stop, at T=0 on a silent rerun of step 0)."""
    self._save(2, k=2, omit_training_state=True)  # key 1.0
    self.mgr.wait()
    shutil.rmtree(os.path.join(self.test_dir, str(KB), "meta"))
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchUnreadableError, "torn"
    ) as ctx:
      self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    msg = str(ctx.exception)
    self.assertIn("no earlier key in window 1", msg)
    self.assertNotIn("fall back to key", msg)
    self.assertIn(self.test_dir, msg)

  def test_io_failure_raises_and_preserves(self):
    """A transient read failure is labelled I/O, must not be mistaken for a
    torn directory (the operator would delete recoverable evidence), and
    the same key restores once the I/O problem is gone."""
    if os.geteuid() == 0:
      self.skipTest("root ignores directory permissions")
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    meta_dir = os.path.join(self.test_dir, str(KB + 1), "meta")
    os.chmod(meta_dir, 0)
    try:
      with self.assertRaisesRegex(
          sub_batch_checkpoint.SubBatchUnreadableError, "I/O"
      ) as ctx:
        self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
      self.assertNotIn("Delete that directory", str(ctx.exception))
    finally:
      os.chmod(meta_dir, 0o755)
    self.assertEqual(self._keys(), [KB + 1])
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 3)

  def test_purge_failure_stops_recovery(self):
    """A dead-lineage key that cannot be removed stops the restore: left on
    disk it would be selected over the rerun's own snapshots of its window
    (and newest-per-window retention would retire those). Both keys stay
    on disk; once the permission is fixed the same restore succeeds."""
    if os.geteuid() == 0:
      self.skipTest("root ignores directory permissions")
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self._save(5, k=2, counts={(1, 0): 1}, omit_training_state=True)
    self.mgr.wait()
    dead = os.path.join(self.test_dir, str(2 * KB + 1))
    os.chmod(dead, 0o500)  # contents cannot be unlinked
    try:
      with self.assertRaisesRegex(
          sub_batch_checkpoint.SubBatchPurgeError, str(2 * KB + 1)
      ) as ctx:
        self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
      self.assertIsInstance(
          ctx.exception, sub_batch_checkpoint.SubBatchRestoreError
      )
      self.assertTrue(os.path.isdir(dead))
      self.assertEqual(self._keys(), [KB + 1, 2 * KB + 1])
    finally:
      os.chmod(dead, 0o755)
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertIsNotNone(st)
    self.assertEqual(st.iter_steps, 3)
    self.assertEqual(self._keys(), [KB + 1])

  def test_purge_fallback_uses_name_format_and_refreshes_step_cache(self):
    self.mgr.close()
    self.mgr = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.test_dir,
        options=checkpoint_options.TunixCheckpointingOptions(
            step_name_format=ocp.path.step.standard_name_format(
                step_prefix="sb", step_format_fixed_length=8
            ),
            enable_async_checkpointing=False,
        ),
    )
    self._save(2, k=2, omit_training_state=True)
    self._save(4, k=2, omit_training_state=True)
    dead = os.path.join(self.test_dir, "sb_02000000")
    with mock.patch.object(
        self.mgr._checkpointer._manager, "delete",
        side_effect=OSError("injected deletion failure"),
    ):
      self.mgr.purge_steps_above(KB)
    self.assertFalse(os.path.exists(dead))
    self.assertEqual(self._keys(), [KB])
    self.assertIsNone(self.mgr._select_step(2))
    # A retry can now create and select its own lineage under that parent.
    self._save(4, k=2, counts={(3, 0): 1}, omit_training_state=True)
    state = self.mgr.try_restore(2, 2)
    self.assertEqual(state.trained_trajectory_counts, {(3, 0): 1})

  def test_stale_key_would_win_selection_without_the_purge(self):
    """Why a failed purge must stop: with abandoned 2.3 left on disk, the
    rerun's 2.0 loses _select_step(2) to it and newest-per-window
    retention keeps 2.3 over 2.0."""
    self._save_boundary(self.mgr, 2)  # 2.0, the rerun's precommit
    self.mgr.save(
        2, 3, iter_steps=7, global_step=1, grad_accum_steps=1,
        step_complete=False, completed_group_ids=[],
        trained_trajectory_counts={}, active_group_trajectories=[],
        training_state=None, anchor_step=2,
    )
    self.mgr.wait()
    self.assertEqual(self.mgr._select_step(2), 2 * KB + 3)
    policy = sub_batch_checkpoint.SubBatchPreservationPolicy()
    cks = [SimpleNamespace(step=2 * KB), SimpleNamespace(step=2 * KB + 3)]
    self.assertEqual(policy.should_preserve(cks, context=None), [False, True])

  def test_num_generations_mismatch_raises_and_preserves(self):
    """The ledger's per-pair accounting is only valid under the saved
    num_generations; a config change must raise like a k change, leaving
    the snapshots intact for a corrected relaunch.
    """
    self._save(
        3, k=2, counts={(1, 0): 1}, omit_training_state=True, num_generations=2
    )
    self.mgr.wait()
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "num_generations"
    ):
      self.mgr.try_restore(
          train_steps=1, grad_accum_steps=2, num_generations=4
      )
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, num_generations=2
    )
    self.assertIsNotNone(st)

  def _load_meta(self, key):
    return dict(
        self.mgr._checkpointer.load_checkpointables(
            key, abstract_checkpointables={"meta": None}
        )["meta"]
    )

  def test_geometry_keys_round_trip_and_mismatch_raises_per_key(self):
    """The recorded geometry dict: every key present on both sides must be
    equal (None is a VALUE), a key absent on either side is skipped, and
    the dict round-trips through Orbax verbatim (str/bool/None/float)."""
    saved = {
        "max_seq_token_per_tpu": 4096,
        "sampler_is": None,
        "use_rollout_logps": False,
        "beta": 0.0,
        "dataset_id": "gsm8k-v1",
        "pack_size": 2,
    }
    self._save(3, k=2, counts={(1, 0): 1}, geometry=saved)
    self.mgr.wait()
    state = _state()
    for field, geometry in (
        ("max_seq_token_per_tpu", {"max_seq_token_per_tpu": None}),
        ("sampler_is", {"sampler_is": "token"}),
        ("beta", {"beta": 0.04}),
        ("dataset_id", {"dataset_id": "gsm8k-v2"}),
        ("use_rollout_logps", {"use_rollout_logps": True}),
    ):
      with self.subTest(field):
        with self.assertRaisesRegex(
            sub_batch_checkpoint.SubBatchGeometryError, field
        ):
          self.mgr.try_restore(
              train_steps=1,
              grad_accum_steps=2,
              target_training_state=state,
              geometry=geometry,
          )
    for name, geometry in (
        ("absent on the saved side", {"max_segments_per_packed_row": 3}),
        ("absent on the current side", {"beta": 0.0}),
        ("empty", {}),
        ("none", None),
    ):
      with self.subTest(name):
        st = self.mgr.try_restore(
            train_steps=1,
            grad_accum_steps=2,
            target_training_state=state,
            geometry=geometry,
        )
        self.assertIsNotNone(st)
        self.assertEqual(st.geometry, saved)

  def test_objective_keys_checked_for_every_snapshot(self):
    """A retuned objective refuses the resume whether or not the snapshot
    carries a gradient buffer."""
    geo = {"max_seq_token_per_tpu": None, "beta": 0.0, "loss_algo": "grpo"}
    self._save(2, k=2, omit_training_state=True, geometry=geo)  # 1.0
    self.mgr.wait()
    for field, current in (("beta", 0.04), ("loss_algo", "gspo-token")):
      with self.subTest(f"bufferless snapshot rejects {field}"):
        with self.assertRaisesRegex(
            sub_batch_checkpoint.SubBatchGeometryError, field
        ):
          self.mgr.try_restore(
              train_steps=1, grad_accum_steps=2, geometry={field: current}
          )
    self.assertEqual(self._keys(), [KB])  # nothing purged
    st = self.mgr.try_restore(train_steps=1, grad_accum_steps=2, geometry=geo)
    self.assertIsNotNone(st)

  def test_precommit_survives_raise_then_corrected_relaunch_purges_it(self):
    """A geometry raise leaves the crashed run's (T+1).0
    precommit on disk; a corrected relaunch then restores T.local and only
    then purges it as dead lineage (invariant: purge after success)."""
    geo = {"max_seq_token_per_tpu": None}
    self._save(3, k=2, counts={(1, 0): 1}, geometry=geo)  # 1.1, buffer
    self._save(4, k=2, omit_training_state=True, geometry=geo)  # 2.0
    self.mgr.wait()
    with self.assertRaises(sub_batch_checkpoint.SubBatchGeometryError):
      self.mgr.try_restore(
          train_steps=1,
          grad_accum_steps=2,
          target_training_state=_state(),
          geometry={"max_seq_token_per_tpu": 4096},
      )
    self.assertEqual(self._keys(), [KB + 1, 2 * KB])
    st = self.mgr.try_restore(
        train_steps=1,
        grad_accum_steps=2,
        target_training_state=_state(),
        geometry=geo,
    )
    self.assertIsNotNone(st)
    self.assertEqual(st.sub_batch_key, KB + 1)
    self.assertEqual(self._keys(), [KB + 1])

  def _write_raw(self, key, meta, rollout):
    """Writes a snapshot bypassing save(): whatever `meta`/`rollout` say."""
    self.mgr._save_checkpointables(
        key,
        {"meta": meta, "rollout": rollout},
        force=True,
        custom_metadata=None,
        overwrite=True,
    )
    self.mgr.wait()

  def test_invalid_payload_keeps_dead_lineage_on_disk(self):
    """A readable orbax tree is not yet a valid ledger: a trajectory payload
    that does not unpickle raises SubBatchUnreadableError and the abandoned
    key above the resume point is still on disk (validate, then purge)."""
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)  # 1.1
    self._save_boundary(self.mgr, 2)  # 2.0
    self.mgr.wait()
    meta = self._load_meta(KB + 1)
    rollout = {
        "completed_group_ids": [],
        "active_group_trajectories": [{
            "prompt_id": 1,
            "group_index": 0,
            "start_step": 0,
            "payload": np.frombuffer(b"not a pickle", dtype=np.uint8),
        }],
        "trained_trajectory_counts": [],
    }
    self._write_raw(KB + 1, meta, rollout)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchUnreadableError, "does not validate"
    ):
      self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
    self.assertEqual(self._keys(), [KB + 1, 2 * KB])

  def test_missing_required_meta_keeps_dead_lineage_on_disk(self):
    """has_training_state and anchor_step are required: a meta lacking
    either is invalid (not defaulted), and nothing above it is purged."""
    self._save(3, k=2, counts={(1, 0): 1}, omit_training_state=True)  # 1.1
    self._save_boundary(self.mgr, 2)  # 2.0
    self.mgr.wait()
    good = self._load_meta(KB + 1)
    rollout = {
        "completed_group_ids": [],
        "active_group_trajectories": [],
        "trained_trajectory_counts": [],
    }
    for field in ("has_training_state", "anchor_step"):
      with self.subTest(field):
        meta = dict(good)
        del meta[field]
        self._write_raw(KB + 1, meta, rollout)
        with self.assertRaisesRegex(
            sub_batch_checkpoint.SubBatchUnreadableError, field
        ):
          self.mgr.try_restore(train_steps=1, grad_accum_steps=2)
        self.assertEqual(self._keys(), [KB + 1, 2 * KB])

  def test_anchor_step_round_trips(self):
    """The step-start train_steps rides meta and comes back on the state;
    a save without it records the snapshot's own parent (test convenience
    only)."""
    state = _state()
    self._save(3, k=2, training_state=state, anchor_step=7)  # 1.1
    self.mgr.wait()
    self.assertEqual(self._load_meta(KB + 1)["anchor_step"], 7)
    st = self.mgr.try_restore(
        train_steps=1, grad_accum_steps=2, target_training_state=state
    )
    self.assertEqual(st.anchor_step, 7)
    self._save(4, k=2, training_state=state)  # 2.0, anchor defaulted
    self.mgr.wait()
    st = self.mgr.try_restore(
        train_steps=2, grad_accum_steps=2, target_training_state=state
    )
    self.assertEqual(st.anchor_step, 2)


if __name__ == "__main__":
  absltest.main()
