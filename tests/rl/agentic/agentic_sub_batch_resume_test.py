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


"""Live-loop tests for sub-batch checkpointing in AgenticRLLearner (v2: keyed

by the trainer's own iter_steps/train_steps, epoch-1-streaming + K-1-replay
loop). Drives the real bound `_sb_*` methods on a minimal concrete learner
instance (stubs only the one abstract method), not a re-implementation, so
wiring regressions are caught rather than masked.
"""

import queue
import shutil
import tempfile
from types import SimpleNamespace

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_options
from tunix.sft import peft_trainer


class _TinyModel(nnx.Module):
  """Minimal nnx module standing in for the actor: one param, real forward

  pass, so nnx.value_and_grad produces properly-shaped grads (a hand-built
  plain-dict grads tree was verified NOT to match what nnx.Optimizer.update
  expects internally -- see memory).
  """

  def __init__(self, rngs):
    self.w = nnx.Param(jnp.zeros((4,)))

  def __call__(self, x):
    return jnp.sum(self.w * x)


def _loss_fn(model, x, target):
  return (model(x) - target) ** 2


def _make_trainer_parts():
  """Builds the real trainer-side accumulation stack the learner targets:

  a model, the REAL peft_trainer.GradientAccumulator (the single
  mode-agnostic home of in-flight gradient state on this trainer build --
  optax.MultiSteps is gone), and a plain Adam optimizer applied from the
  accumulator at window boundaries, matching peft_trainer._train_step's
  add/get/update/reset cycle exactly.
  """
  model = _TinyModel(nnx.Rngs(0))
  accumulator = peft_trainer.GradientAccumulator(model, nnx.Param)
  optimizer = nnx.Optimizer(model, optax.adam(0.1), wrt=nnx.Param)
  return model, accumulator, optimizer


def _real_micro_step(model, accumulator, x, target):
  """One real accumulation micro-step: nnx.value_and_grad + accumulator.add,

  matching peft_trainer._train_step's non-apply arm.
  """
  _, grads = nnx.value_and_grad(_loss_fn, argnums=0)(model, x, target)
  accumulator.add(grads)


def _real_apply(model, accumulator, optimizer):
  """The apply arm: optimizer.update from accumulator.get(), then reset --

  peft_trainer._train_step's apply_updates.
  """
  optimizer.update(model, accumulator.get())
  accumulator.reset()


def _acc_state(w=1.0, denom=1.0):
  """A flat accumulator payload as _sb_extract_accumulator produces it:

  tuple paths ('grads', <param>) plus ('denom',).
  """
  return {
      ("grads", "w"): np.array([w, w, w, w], dtype=np.float32),
      ("denom",): np.array(denom, dtype=np.float32),
  }


def _traj(group_id, pair_index, reward=1.0):
  step = agent_types.Step(
      chat_completions=[{"role": "user", "content": "hi"}],
      thought="t",
      model_response="r",
      reward=reward,
      done=True,
      assistant_tokens=np.array([1, 2, 3]),
      assistant_masks=np.array([1, 1, 1]),
      logprobs=np.array([0.1, 0.2, 0.3]),
  )
  traj = agent_types.Trajectory(
      task="task",
      steps=[step],
      reward=reward,
      status=agent_types.TrajectoryStatus.SUCCEEDED,
  )
  return agent_types.TrajectoryItem(
      group_id=group_id,
      pair_index=pair_index,
      start_step=0,
      traj=traj,
      metadata={},
  )


class _Learner(agentic_rl_learner.AgenticRLLearner):
  """Concrete subclass: stub the one abstract method so __new__ works."""

  def _process_results(self, *a, **k):
    return []


def _make_learner(
    *,
    num_iterations=1,
    num_generations=2,
    mgr=None,
    global_steps=0,
    process_in_consumer=False,
    train_steps=0,
    k=1,
    checkpoint_manager_latest_step=0,
    full_batch_size=0,
    critic_train_steps=None,
    critic_checkpoint_manager=None,
):
  learner = _Learner.__new__(_Learner)
  # Benchmark-instrumentation anchors (commit 4): _sb_snapshot reads the
  # time anchors; the restore path reads _bench_anchor_pinned.
  learner._global_step_start_time = 0.0
  learner._experiment_start_time = 0.0
  learner._bench_anchor_pinned = False
  learner._sb_chaos_prob = 0.0
  learner._bench_metrics = False
  learner.algo_config = SimpleNamespace(
      num_iterations=num_iterations,
      num_generations=num_generations,
      off_policy_steps=0,
      sub_batch_checkpointing=True,
  )
  learner._process_in_consumer = process_in_consumer
  learner._full_batch_size = full_batch_size
  learner._sb_mgr = mgr
  learner._sb_lock = __import__("threading").Lock()
  learner._sb_completed = set()
  learner._sb_counts = {}
  learner._sb_active = []
  learner._sb_pending_state = None
  learner._sb_identity_queue = None
  learner._sb_last_snapshot_key = -1
  learner._sb_window_train_steps = -1
  learner._sb_local = -1
  learner._sb_last_snapshot_trainer_iter = -1
  learner._sb_active_gids = set()
  learner._sb_step_complete_for_step = None
  learner._sb_restored_trainer_state = False
  learner._sb_restored_full_batch_size = None
  model, accumulator, optimizer = _make_trainer_parts()
  learner._training_config = SimpleNamespace(
      get_with_default=lambda key, default: k
      if key == "gradient_accumulation_steps"
      else default,
      checkpoint_root_directory="/tmp/unused",
      max_seq_token_per_tpu=None,
      checkpointing_options=SimpleNamespace(
          save_decision_policy=SimpleNamespace(interval=1)
      ),
  )
  learner.rl_engine = SimpleNamespace(
      global_steps=global_steps,
      cluster_config=SimpleNamespace(offload_to_cpu=False),
      actor_trainer=SimpleNamespace(
          model=model,
          optimizer=optimizer,
          grad_accumulator=accumulator,
          _train_steps=train_steps,
          train_steps=train_steps,
          iter_steps=train_steps * k,
          checkpoint_manager=SimpleNamespace(
              latest_step=lambda: checkpoint_manager_latest_step,
              maybe_restore=lambda *a, **kw: (kw.get("step", 0), {}),
          ),
      ),
  )
  if critic_train_steps is not None:
    # hasattr(rl_engine, "critic_trainer") is the real presence signal, so
    # the attribute only exists when a critic is requested.
    critic_model, critic_accumulator, critic_optimizer = _make_trainer_parts()
    learner.rl_engine.critic_trainer = SimpleNamespace(
        model=critic_model,
        optimizer=critic_optimizer,
        grad_accumulator=critic_accumulator,
        _train_steps=critic_train_steps,
        train_steps=critic_train_steps,
        iter_steps=critic_train_steps * k,
        checkpoint_manager=(
            critic_checkpoint_manager
            or SimpleNamespace(
                latest_step=lambda: critic_train_steps,
                maybe_restore=lambda *a, **kw: (kw.get("step", 0), {}),
            )
        ),
    )
  return learner


def _make_manager(tmp_dir, k=2, n=10):
  return sub_batch_checkpoint.SubBatchCheckpointManager(
      root_directory=tmp_dir,
      options=checkpoint_options.TunixCheckpointingOptions(
          save_decision_policy=(
              ocp.training.save_decision_policies.FixedIntervalPolicy(
                  interval=1
              )
          ),
          preservation_policy=ocp.training.preservation_policies.LatestN(n=n),
          step_name_format=ocp.path.step.standard_name_format(),
          enable_async_checkpointing=True,
      ),
  )


class SubBatchLedgerTest(absltest.TestCase):
  """Ledger mechanics: register, skip, snapshot/eviction, chunk epoch/split,

  step boundary. No real Orbax involved (mgr=None -> _sb_enabled False for
  the manager-touching parts); these test the pure in-memory bookkeeping.
  """

  def test_skip_group_checks_completed_and_active(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)  # _sb_enabled True, no I/O
    learner._sb_completed = {5}
    # Register through the real hook so the O(1) gid mirror stays in sync
    # (direct _sb_active writes bypass it, which is exactly why the mirror is
    # maintained inside the hooks).
    learner._sb_register([_traj(7, 0)])
    self.assertTrue(learner._sb_skip_group(5))
    self.assertTrue(learner._sb_skip_group(7))
    self.assertFalse(learner._sb_skip_group(9))

  def test_skip_group_false_when_disabled(self):
    learner = _make_learner(mgr=None)
    self.assertFalse(learner._sb_skip_group(5))

  def test_register_appends_to_active(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    batch = [_traj(1, 0), _traj(1, 1)]
    learner._sb_register(batch)
    self.assertEqual(learner._sb_active, batch)

  def test_chunk_epoch_min_across_identities(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 2, (1, 1): 1}
    self.assertEqual(learner._sb_chunk_epoch([(1, 0), (1, 1)]), 1)
    self.assertEqual(learner._sb_chunk_epoch([]), 0)

  def test_split_by_epoch_noop_when_uniform(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 0, (2, 0): 0}
    chunk = {"x": jnp.array([10, 20])}
    ids = [(1, 0), (2, 0)]
    out = learner._sb_split_by_epoch(chunk, ids)
    self.assertLen(out, 1)
    self.assertEqual(out[0][1], ids)
    np.testing.assert_array_equal(out[0][0]["x"], [10, 20])

  def test_split_by_epoch_splits_mixed_chunk(self):
    """Regression for the cross-group mixing hazard: a chunk batching a

    group already at epoch 2 alongside a group still at epoch 1 (the
    "prefix at e+1, suffix at e" crash signature under train_micro_batch_
    size > 1) must never be trained as one unit -- there is no way to
    partially backprop a batch, so it must split before update_actor sees
    it.
    """
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 2, (1, 1): 2, (2, 0): 1, (2, 1): 1}
    chunk = {"x": jnp.array([10, 11, 20, 21])}
    ids = [(1, 0), (1, 1), (2, 0), (2, 1)]
    out = learner._sb_split_by_epoch(chunk, ids)
    self.assertLen(out, 2)
    self.assertEqual(out[0][1], [(1, 0), (1, 1)])
    np.testing.assert_array_equal(out[0][0]["x"], [10, 11])
    self.assertEqual(out[1][1], [(2, 0), (2, 1)])
    np.testing.assert_array_equal(out[1][0]["x"], [20, 21])

  def test_split_by_epoch_three_way(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 0, (2, 0): 1, (3, 0): 0}
    chunk = {"x": jnp.array([1, 2, 3])}
    ids = [(1, 0), (2, 0), (3, 0)]
    out = learner._sb_split_by_epoch(chunk, ids)
    self.assertLen(out, 3)
    self.assertEqual([o[1] for o in out], [[(1, 0)], [(2, 0)], [(3, 0)]])

  def test_snapshot_completion_retains_active_and_counts(self):
    """Deadlock regression.

    A group reaching num_iterations on every pair is marked completed but its
    trajectories and counts MUST be retained until the step boundary: a resumed
    step re-feeds the queue from the active set, and the consumer's full-batch
    accounting expects one micro-batch per group. Dropping the payload at
    completion (the previous behavior) made that group's micro-batch
    unproducible on resume (payload gone, producer skips it) so
    micro_batches_since_last_sync could never reach the step boundary: a hang,
    not an error.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(num_iterations=2, num_generations=2, mgr=mgr, k=1)
      trajs = [_traj(1, 0), _traj(1, 1)]
      learner._sb_active = list(trajs)
      ids = [(1, 0), (1, 1)]
      learner.rl_engine.actor_trainer.iter_steps = 1
      learner._sb_snapshot(ids, step_complete=False)
      self.assertEqual(learner._sb_counts, {(1, 0): 1, (1, 1): 1})
      self.assertNotIn(1, learner._sb_completed)
      learner.rl_engine.actor_trainer.iter_steps = 2
      learner._sb_snapshot(ids, step_complete=True)
      # Both pairs now at mu=2 -> marked completed, but payload and counts
      # survive for resume re-feeding (skipped via the capped counts).
      self.assertIn(1, learner._sb_completed)
      self.assertEqual(learner._sb_counts, {(1, 0): 2, (1, 1): 2})
      self.assertEqual(learner._sb_active, trajs)
      # The retained state still routes correctly: producer skips the group,
      # and every chunk of it skips training at every epoch.
      self.assertTrue(learner._sb_skip_group(1))
      self.assertEqual(learner._sb_chunk_epoch(ids), 2)
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_snapshot_partial_group_not_completed(self):
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(num_iterations=2, num_generations=2, mgr=mgr, k=1)
      # Only pair 0 trained; pair 1 untouched -> group must stay incomplete.
      learner.rl_engine.actor_trainer.iter_steps = 1
      learner._sb_snapshot([(1, 0)], step_complete=False)
      learner.rl_engine.actor_trainer.iter_steps = 2
      learner._sb_snapshot([(1, 0)], step_complete=False)
      self.assertEqual(learner._sb_counts, {(1, 0): 2})
      self.assertNotIn(1, learner._sb_completed)
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_snapshot_key_must_advance_within_a_run(self):
    """Saves use overwrite=True (resume re-execution legitimately reuses a

    crashed run's keys), so a same-run duplicate key would be silently
    clobbered. The learner must fail loudly instead.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(num_iterations=1, num_generations=1, mgr=mgr, k=1)
      learner.rl_engine.actor_trainer.iter_steps = 1
      learner._sb_snapshot([(1, 0)], step_complete=False)
      with self.assertRaisesRegex(RuntimeError, "wiring bug"):
        learner._sb_snapshot([(2, 0)], step_complete=False)  # iter unchanged
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_snapshot_at_apply_boundary_omits_buffer(self):
    """At an apply boundary the accumulator is provably empty (denom == 0,

    it was just reset), so the parameter-sized payload is omitted and
    restore returns training_state None. With k == 1 every snapshot is a
    boundary, so the buffer is never written at all.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      # k=1: the chunk that just trained ended in an apply, so the trainer
      # sits at T=1 with a freshly reset accumulator when the snapshot runs.
      learner = _make_learner(
          num_iterations=1, num_generations=1, mgr=mgr, k=1, train_steps=1
      )
      # Fresh accumulator: denom == 0 (boundary condition).
      learner._sb_snapshot([(1, 0)], step_complete=False)
      mgr.wait()
      st = mgr.try_restore(train_steps=1, grad_accum_steps=1)
      self.assertIsNotNone(st)
      self.assertIsNone(st.training_state)
      self.assertEqual(st.trained_trajectory_counts, {(1, 0): 1})
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_counts_capped_at_num_iterations(self):
    """Counts never exceed mu, so a completed group's chunks skip at every

    epoch check without drifting past the cap on repeated snapshots.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(num_iterations=2, num_generations=1, mgr=mgr, k=1)
      for it in range(1, 5):  # 4 snapshots, mu = 2
        learner.rl_engine.actor_trainer.iter_steps = it
        learner._sb_snapshot([(1, 0)], step_complete=False)
      self.assertEqual(learner._sb_counts, {(1, 0): 2})  # capped, not 4
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_snapshot_excludes_raced_ahead_next_step_groups(self):
    """Producer run-ahead regression (off_policy_steps > 0): the producer can

    register NEXT-step groups while the consumer is mid-CURRENT-step. Those
    must not be serialized into the current step's snapshots (a mid-step
    resume would re-inject another step's rollouts into this one and inflate
    its micro-batch accounting).
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(
          num_iterations=1,
          num_generations=1,
          mgr=mgr,
          k=1,
          global_steps=2,
          full_batch_size=4,
          train_steps=2,
      )
      current = _traj(2 * 4 + 1, 0)  # group of step 2 (current)
      raced = _traj(3 * 4 + 0, 0)  # group of step 3 (prefetched ahead)
      learner._sb_active = [current, raced]
      learner.rl_engine.actor_trainer.iter_steps = 2
      learner._sb_snapshot([(2 * 4 + 1, 0)], step_complete=False)
      mgr.wait()
      st = mgr.try_restore(train_steps=2, grad_accum_steps=1)
      self.assertIsNotNone(st)
      self.assertEqual(
          [t.group_id for t in st.active_group_trajectories], [2 * 4 + 1]
      )
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_step_boundary_resets_pillars(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_completed = {1, 2}
    learner._sb_counts = {(3, 0): 1}
    learner._sb_active = [_traj(4, 0)]
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, set())
    self.assertEqual(learner._sb_counts, {})
    self.assertEqual(learner._sb_active, [])

  def test_step_boundary_retains_raced_ahead_next_step_groups(self):
    """The boundary clear must drop only the finished step's registrations.

    Wiping a prefetched next-step group would strand it: its payload could
    neither be re-fed after a later crash nor regenerated (its counts would
    wrongly skip fresh rollouts).
    """
    learner = _make_learner(global_steps=2, full_batch_size=4)
    learner._sb_mgr = SimpleNamespace(enabled=True)
    current = _traj(2 * 4 + 1, 0)  # step 2 (finishing)
    raced = _traj(3 * 4 + 0, 0)  # step 3 (prefetched ahead)
    learner._sb_completed = {2 * 4 + 1}
    learner._sb_counts = {(2 * 4 + 1, 0): 1}
    learner._sb_active = [current, raced]
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, set())
    self.assertEqual(learner._sb_counts, {})
    self.assertEqual([t.group_id for t in learner._sb_active], [3 * 4 + 0])
    # And the retained registration still shields the group from
    # regeneration when its step arrives.
    self.assertTrue(learner._sb_skip_group(3 * 4 + 0))

  def test_step_boundary_noop_when_disabled(self):
    learner = _make_learner(mgr=None)
    learner._sb_completed = {1}
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, {1})  # untouched


class SubBatchBufferTest(absltest.TestCase):
  """Accumulator extract/inject against the REAL peft_trainer

  GradientAccumulator (the trainer's single mode-agnostic home of in-flight
  gradient state; optax.MultiSteps is gone from this trainer build).
  Verified mechanism: nnx.State's pure-dict round-trip and naive
  cross-instance tree_map both silently produced structurally invalid module
  state (see memory); flat_state()-path in-place mutation on the live
  Variable boxes is the one mechanism that survived. These tests exercise
  exactly that failure mode: inject, then perform REAL further micro-steps
  and a real apply, and compare against an uninterrupted baseline
  bit-for-bit.
  """

  def test_extract_returns_none_without_accumulator(self):
    learner = _make_learner(k=1)
    del learner.rl_engine.actor_trainer.grad_accumulator
    self.assertIsNone(learner._sb_extract_training_state())

  def test_inject_without_accumulator_warns_and_noops(self):
    learner = _make_learner(k=1)
    del learner.rl_engine.actor_trainer.grad_accumulator
    with self.assertLogs(level="WARNING"):
      learner._sb_inject_training_state(_acc_state())

  def test_extract_inject_roundtrip_bit_exact_against_baseline(self):
    """The core correctness claim: inject an accumulator state extracted

    mid-window from one learner instance into a DIFFERENT (freshly
    constructed) instance, finish the window there with real micro-steps and
    a real apply, and require the result to exactly match an uninterrupted
    run over the identical 3 micro-steps -- weights AND the inner Adam
    state, which injection must leave untouched (it is frozen within the
    window and restored by the trainer's own weight checkpoint, not by this
    buffer).
    """
    xs = [
        jnp.array([1.0, 2.0, 3.0, 4.0]),
        jnp.array([2.0, 1.0, 0.0, 1.0]),
        jnp.array([0.0, 1.0, 2.0, 3.0]),
    ]
    targets = [5.0, 3.0, 2.0]

    crashed_learner = _make_learner(k=3)
    tr_a = crashed_learner.rl_engine.actor_trainer
    _real_micro_step(tr_a.model, tr_a.grad_accumulator, xs[0], targets[0])
    diff = crashed_learner._sb_extract_training_state()
    self.assertIsNotNone(diff)
    self.assertEqual(float(diff[("denom",)]), 1.0)  # one micro-step banked

    resumed_learner = _make_learner(k=3)
    tr_b = resumed_learner.rl_engine.actor_trainer
    resumed_learner._sb_inject_training_state(diff)
    _real_micro_step(tr_b.model, tr_b.grad_accumulator, xs[1], targets[1])
    _real_micro_step(tr_b.model, tr_b.grad_accumulator, xs[2], targets[2])
    _real_apply(tr_b.model, tr_b.grad_accumulator, tr_b.optimizer)

    base_model, base_acc, base_opt = _make_trainer_parts()
    for x, t in zip(xs, targets):
      _real_micro_step(base_model, base_acc, x, t)
    _real_apply(base_model, base_acc, base_opt)

    np.testing.assert_array_equal(
        nnx.state(tr_b.model)["w"][...], nnx.state(base_model)["w"][...]
    )
    resumed_mu = jax.tree_util.tree_leaves(
        nnx.state(tr_b.optimizer, nnx.optimizer.OptState)
    )[0][...]
    baseline_mu = jax.tree_util.tree_leaves(
        nnx.state(base_opt, nnx.optimizer.OptState)
    )[0][...]
    np.testing.assert_array_equal(resumed_mu, baseline_mu)
    # Apply fired: the accumulator was reset, denom back to zero.
    self.assertEqual(float(np.asarray(tr_b.grad_accumulator.denom[...])), 0.0)

  def test_build_abstract_training_state_shapes(self):
    learner = _make_learner(k=2)
    tr = learner.rl_engine.actor_trainer
    _real_micro_step(
        tr.model, tr.grad_accumulator, jnp.array([1.0, 2.0, 3.0, 4.0]), 5.0
    )
    target = learner._sb_build_abstract_training_state()
    self.assertIsInstance(target[("grads", "w")], jax.ShapeDtypeStruct)
    self.assertEqual(target[("grads", "w")].shape, (4,))
    self.assertIsInstance(target[("denom",)], jax.ShapeDtypeStruct)
    self.assertEqual(target[("denom",)].shape, ())


class SubBatchInitResumeTest(absltest.TestCase):
  """_init_sub_batch_checkpointing: T=0 ambiguity fix, rewind vs no-rewind,

  the trainer iter_steps fixup, and end-to-end restore through the real
  manager (real Orbax save + this learner's real restore path).
  """

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp)
    super().tearDown()

  def test_nothing_restored_when_no_checkpoint_exists(self):
    """latest_step() is None (never checkpointed) must be treated as 'no

    checkpoint', distinct from 'restored a checkpoint at train_steps=0':
    both report train_steps=0, so this is the only way to tell them apart.
    A stray leftover snapshot in root_directory must not be injected onto
    fresh-init weights.


    Exercises the real `_init_sub_batch_checkpointing` guard end to end
    (constructs its own manager from checkpoint_root_directory), not just
    the manager's own try_restore -- an earlier version of this test only
    checked try_restore directly and would have passed even if the guard in
    _init_sub_batch_checkpointing were missing entirely.
    """
    root = tempfile.mkdtemp()
    try:
      # Plant a snapshot at the exact path _init_sub_batch_checkpointing
      # will construct its manager against (root/sub_batch), as if from an
      # unrelated earlier run sharing this checkpoint_root_directory.
      planted = _make_manager(f"{root}/sub_batch", k=1)
      state = _acc_state(w=9.0, denom=1.0)
      planted.save(
          0,
          0,
          iter_steps=0,
          global_step=0,
          grad_accum_steps=1,
          step_complete=False,
          completed_group_ids=[],
          trained_trajectory_counts={},
          active_group_trajectories=[],
          training_state=state,
      )
      planted.wait()
      planted.close()

      learner = _make_learner(
          mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
      )
      learner._training_config.checkpoint_root_directory = root
      orig_w = nnx.state(learner.rl_engine.actor_trainer.model)["w"][
          ...
      ].copy()

      learner._init_sub_batch_checkpointing()

      self.assertIsNone(learner._sb_pending_state)  # guard declined to restore
      self.assertEqual(learner._sb_active, [])
      np.testing.assert_array_equal(
          nnx.state(learner.rl_engine.actor_trainer.model)["w"][...], orig_w
      )
      # The guard's purge is load-bearing, not hygiene: try_restore never runs
      # on this path, so nothing else removes the planted key -- if it
      # survived, this run's own window would eventually reach it and
      # max-selection would inject a foreign ledger+buffer onto live weights.
      self.assertIsNone(learner._sb_mgr._select_step(0))
      # Nothing was restored, so the resync gate must stay closed: firing
      # sync_weights here would push fresh-init weights to the engine and
      # burn a global_steps increment for a step that never happened.
      self.assertFalse(learner._sb_restored_trainer_state)
      learner._sb_mgr.close()
    finally:
      shutil.rmtree(root)

  def test_resume_mid_step_rewinds_global_steps(self):
    """step_complete=False means the trainer's reported global_steps (which

    is stamped global_steps+1, see rl_cluster.py) is one ahead of the step
    actually in progress; must rewind to resume that same step.


    Saves directly against the path _init_sub_batch_checkpointing will
    construct its own manager against (checkpoint_root_directory/sub_batch),
    then drives the real init method end to end -- not the manager's
    try_restore in isolation -- so this proves the full wiring, including
    the trainer iter_steps fixup, which only _init_sub_batch_checkpointing
    performs.
    """
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    # Tuple-path keys, matching what _sb_extract_training_state actually
    # produces (flat_state() paths, not plain attribute-name strings) --
    # a hand-written plain-string key here silently makes the abstract
    # target's structure not match what was saved, and try_restore declines
    # (verified against the real checkpointer while debugging this test).
    state = _acc_state(w=1.0, denom=1.0)
    planted.save(
        1,
        1,
        iter_steps=3,
        global_step=4,
        grad_accum_steps=2,
        step_complete=False,
        completed_group_ids=[7],
        trained_trajectory_counts={(9, 0): 1},
        active_group_trajectories=[_traj(9, 0)],
        training_state=state,
    )
    planted.wait()
    planted.close()

    learner = _make_learner(mgr=None, k=2, train_steps=1, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNotNone(learner._sb_pending_state)  # restore actually hit
    self.assertEqual(learner.rl_engine.global_steps, 4)  # rewound from 5
    self.assertEqual(learner._sb_completed, {7})
    self.assertEqual(learner._sb_counts, {(9, 0): 1})
    self.assertLen(learner._sb_active, 1)
    self.assertEqual(learner.rl_engine.actor_trainer._iter_steps, 3)
    # Producer side of the disaggregated-resync gate. SubBatchResyncTest sets
    # this flag by hand, so without asserting it here the assignment can be
    # deleted with every suite green -- and the rollout engine would then
    # generate a whole step from boot-stale weights while tagging it with the
    # current policy_version: off-policy data the learner believes is
    # on-policy, with nothing raising and no metric moving.
    self.assertTrue(learner._sb_restored_trainer_state)
    ga = learner.rl_engine.actor_trainer.grad_accumulator
    np.testing.assert_array_equal(
        np.asarray(ga.grads["w"][...]), [1.0, 1.0, 1.0, 1.0]
    )
    self.assertEqual(float(np.asarray(ga.denom[...])), 1.0)
    learner._sb_mgr.close()

  def test_resume_step_complete_starts_fresh_without_phantom_step(self):
    """Phantom-step regression.

    The final apply's step_complete=True snapshot carries the ENTIRE finished
    step's retained ledger (that retention is deliberate, see the
    completion-retention test). If resume seeded and re-injected it, the
    consumer would count full_batch_size re-fed groups whose chunks all skip via
    the mu-capped counts, fire a boundary after ZERO training, and produce a
    phantom global step: +1 global_steps drift, a spurious weight sync, and a
    silently skipped dataset batch. Resume must start the next step completely
    fresh.
    """
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    # A REALISTIC finished-step payload: full retained active set + capped
    # counts (the earlier version of this test planted an empty payload and
    # could not catch the phantom-step bug).
    items = [_traj(8, 0), _traj(8, 1), _traj(9, 0), _traj(9, 1)]
    planted.save(
        2,
        0,
        iter_steps=4,
        global_step=4,
        grad_accum_steps=2,
        step_complete=True,
        completed_group_ids=[8, 9],
        trained_trajectory_counts={(8, 0): 1, (8, 1): 1, (9, 0): 1, (9, 1): 1},
        active_group_trajectories=items,
        training_state=None,
    )
    planted.wait()
    planted.close()

    learner = _make_learner(mgr=None, k=2, train_steps=2, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertEqual(learner.rl_engine.global_steps, 5)  # no rewind
    # Nothing seeded, nothing armed for re-injection: the next step is fresh.
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_completed, set())
    self.assertEqual(learner._sb_counts, {})
    self.assertEqual(learner._sb_active, [])
    self.assertFalse(learner._sb_skip_group(8))  # producer regenerates freely
    # Monotonicity guard still continues from the restored key.
    self.assertEqual(
        learner._sb_last_snapshot_key, 2 * sub_batch_checkpoint.KEY_BASE
    )
    q = queue.Queue()
    learner._sb_reinject(q)
    self.assertTrue(q.empty())  # nothing re-injected
    learner._sb_mgr.close()

  def test_no_snapshot_in_window_leaves_state_clean(self):
    learner = _make_learner(mgr=None, k=2, train_steps=5, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_active, [])
    learner._sb_mgr.close()

  def test_geometry_rollback_is_consistent_and_next_snapshot_advances(self):
    """full_batch_size changed across the restart: the rollback must leave

    EVERY piece of resume state agreeing on 'fresh step at the window
    start' -- pillars, pending injection, the injected accumulator, the
    trainer counter, the on-disk window, AND the monotonicity guard. The
    guard is the regression here: it was seeded with the restored snapshot's
    key in __init__, and a rollback that forgets to rewind it makes the
    fresh step's first _sb_snapshot raise 'did not advance ... wiring bug'
    on a legitimate, supported config change.
    """
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    state = _acc_state(w=1.0, denom=1.0)
    planted.save(
        1,
        1,
        iter_steps=3,
        global_step=4,
        grad_accum_steps=2,
        step_complete=False,
        completed_group_ids=[16],
        trained_trajectory_counts={(17, 0): 1},
        active_group_trajectories=[_traj(17, 0)],
        training_state=state,
        full_batch_size=4,
    )
    planted.wait()
    planted.close()

    learner = _make_learner(mgr=None, k=2, train_steps=1, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNotNone(learner._sb_pending_state)  # mid-window restore hit
    self.assertEqual(
        learner._sb_last_snapshot_key, 1 * sub_batch_checkpoint.KEY_BASE + 1
    )

    learner._sb_validate_batch_geometry(8)  # crashed run saved 4

    trainer = learner.rl_engine.actor_trainer
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_completed, set())
    self.assertEqual(learner._sb_counts, {})
    self.assertEqual(learner._sb_active, [])
    self.assertEqual(learner._sb_active_gids, set())
    self.assertEqual(trainer._iter_steps, 2)  # window start
    self.assertEqual(  # guard rewound below the fresh window's first key
        learner._sb_last_snapshot_key, 1 * sub_batch_checkpoint.KEY_BASE - 1
    )
    np.testing.assert_array_equal(  # injected accumulator zeroed
        np.asarray(trainer.grad_accumulator.grads["w"][...]),
        [0.0, 0.0, 0.0, 0.0],
    )
    self.assertEqual(
        float(np.asarray(trainer.grad_accumulator.denom[...])), 0.0
    )
    self.assertIsNone(learner._sb_mgr._select_step(1))  # key 3 purged

    # The fresh step's first trained chunk must snapshot cleanly.
    trainer._iter_steps = 3
    trainer.iter_steps = 3
    learner._sb_snapshot([(0, 0)], step_complete=False)
    self.assertEqual(
        learner._sb_last_snapshot_key, 1 * sub_batch_checkpoint.KEY_BASE
    )
    learner._sb_mgr.close()


class SubBatchResyncTest(absltest.TestCase):
  """_sb_resync_rollout_weights: the disaggregated-restart weight refresh

  (mandatory on EVERY restart that restored trainer
  weights). The production topology is vLLM server mode, where
  should_sync_weights is True and rl_engine.sync_weights increments
  global_steps as a side effect (rl_cluster.py); the refresh must reuse that
  tested transfer path while compensating the increment, or every preemption
  shifts the dataset fast-forward and the producer's group-id base by one.
  """

  def _learner(self, *, restored, disaggregated):
    learner = _make_learner(k=1, global_steps=5)
    learner._sb_mgr = SimpleNamespace(enabled=True)  # _sb_enabled True
    learner._sb_restored_trainer_state = restored
    learner.should_sync_weights = disaggregated
    calls = []
    cluster = learner.rl_engine

    def sync_weights():
      calls.append(1)
      cluster.global_steps += 1  # mirrors rl_engine.sync_weights

    cluster.sync_weights = sync_weights
    return learner, calls

  def test_disaggregated_restart_syncs_once_without_global_steps_drift(self):
    learner, calls = self._learner(restored=True, disaggregated=True)
    learner._sb_resync_rollout_weights()
    self.assertEqual(len(calls), 1)  # rollout engine actually refreshed
    self.assertEqual(learner.rl_engine.global_steps, 5)  # increment undone

  def test_colocated_restart_does_not_sync(self):
    learner, calls = self._learner(restored=True, disaggregated=False)
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [])

  def test_nothing_restored_does_not_sync(self):
    learner, calls = self._learner(restored=False, disaggregated=True)
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [])


class SubBatchStepCompleteGuardTest(absltest.TestCase):
  """The step-boundary guard behind step_complete.

  step_complete rides the last chunk actually TRAINED (not the last list
  index): after a resume the count-skipped chunks need not be a prefix, and a
  skipped chunk writes no snapshot. If the flag ever lands on no snapshot, the
  finished step restores as mid-step and burns a zero-training phantom step.
  The boundary is the one place that can notice, since reaching it means the
  step finished.
  """

  def _learner(self):
    learner = _make_learner(k=1, global_steps=3)
    learner._sb_mgr = SimpleNamespace(enabled=True)  # _sb_enabled True, no I/O
    return learner

  def test_boundary_without_step_complete_snapshot_is_reported(self):
    learner = self._learner()
    self.assertIsNone(learner._sb_step_complete_for_step)
    with self.assertLogs(level="ERROR") as logs:
      learner._sb_step_boundary()
    joined = "\n".join(logs.output)
    self.assertIn("step_complete", joined)
    self.assertIn("phantom step", joined)

  def test_boundary_after_step_complete_snapshot_is_quiet(self):
    learner = self._learner()
    # As _sb_snapshot stamps it: the step currently being finished.
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
    with self.assertNoLogs(level="ERROR"):
      learner._sb_step_boundary()

  def test_guard_is_self_arming_across_steps(self):
    """A stamp, not a flag, so the guard cannot be silently disarmed.

    With a boolean re-armed inside _sb_step_boundary, dropping that call --
    the exact regression the guard exists to catch -- would latch it True at
    step 0 and never evaluate again. Comparing stamps means a stale one from
    an earlier step still trips it.
    """
    learner = self._learner()
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
    with self.assertNoLogs(level="ERROR"):
      learner._sb_step_boundary()  # step 3: correctly marked
    learner.rl_engine.global_steps += 1  # step 4 finishes unmarked
    with self.assertLogs(level="ERROR") as logs:
      learner._sb_step_boundary()
    self.assertIn("step_complete", "\n".join(logs.output))

  def test_snapshot_stamps_step_complete_for_the_boundary_guard(self):
    """The guard is only meaningful if _sb_snapshot actually stamps it."""
    tmp = tempfile.mkdtemp()
    try:
      learner = _make_learner(k=1, global_steps=3, full_batch_size=4)
      learner._sb_mgr = _make_manager(tmp, k=1)
      learner._sb_snapshot([(12, 0)], step_complete=False)
      self.assertIsNone(learner._sb_step_complete_for_step)
      learner.rl_engine.actor_trainer.iter_steps = 1
      learner._sb_snapshot([(13, 0)], step_complete=True)
      self.assertEqual(learner._sb_step_complete_for_step, 3)
      with self.assertNoLogs(level="ERROR"):
        learner._sb_step_boundary()
      learner._sb_mgr.close()
    finally:
      shutil.rmtree(tmp, ignore_errors=True)


class SubBatchReinjectTest(absltest.TestCase):

  def test_reinject_true_path_queues_raw_groups(self):
    learner = _make_learner(process_in_consumer=True)
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_pending_state = object()  # truthy marker
    learner._sb_active = [_traj(1, 0), _traj(1, 1), _traj(2, 0)]
    q = queue.Queue()
    learner._sb_reinject(q)
    items = []
    while not q.empty():
      items.append(q.get())
    self.assertLen(items, 2)  # two groups
    all_items = [t for group in items for t in group]
    self.assertCountEqual(
        [(t.group_id, t.pair_index) for t in all_items],
        [(1, 0), (1, 1), (2, 0)],
    )

  def test_reinject_noop_when_nothing_pending(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_pending_state = None
    q = queue.Queue()
    learner._sb_reinject(q)
    self.assertTrue(q.empty())


class SubBatchGuardsTest(absltest.TestCase):

  def test_packing_raises_until_part_two_lands(self):
    """Sequence packing + sub-batch is guarded out in this commit (Part II

    wiring is the next commit of the series); a packed config must fail
    loudly at init, not silently mis-account at the row level.
    """
    learner = _make_learner(mgr=None, k=2)
    learner._training_config.max_seq_token_per_tpu = 4096
    with self.assertRaisesRegex(ValueError, "sequence packing"):
      learner._init_sub_batch_checkpointing()

  def test_missing_checkpoint_root_directory_raises(self):
    learner = _make_learner(mgr=None)
    learner.algo_config.sub_batch_checkpointing = True
    learner._training_config.checkpoint_root_directory = None
    with self.assertRaises(ValueError):
      learner._init_sub_batch_checkpointing()

  def test_critic_trainer_raises(self):
    """Critic support is deferred."""
    learner = _make_learner(mgr=None, critic_train_steps=0)
    learner.algo_config.sub_batch_checkpointing = True
    with self.assertRaisesRegex(ValueError, "critic"):
      learner._init_sub_batch_checkpointing()

  def test_not_per_apply_raises(self):
    learner = _make_learner(mgr=None)
    learner.algo_config.sub_batch_checkpointing = True
    learner._training_config.checkpointing_options.save_decision_policy.interval = 5
    with self.assertRaisesRegex(ValueError, "FixedIntervalPolicy\\(interval=1\\)"):
      learner._init_sub_batch_checkpointing()

  def test_disabled_feature_fully_inert(self):
    learner = _make_learner(mgr=None)
    learner.algo_config.sub_batch_checkpointing = False
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_mgr)
    self.assertFalse(learner._sb_enabled)
    self.assertFalse(learner._sb_skip_group(1))
    learner._sb_register([_traj(1, 0)])
    self.assertEqual(learner._sb_active, [])
    learner._sb_snapshot([(1, 0)], step_complete=True)  # no-op, no crash


if __name__ == "__main__":
  absltest.main()
