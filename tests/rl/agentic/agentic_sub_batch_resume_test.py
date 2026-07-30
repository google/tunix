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

import asyncio
import os
import collections
import queue
import shutil
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.rl import rl_cluster
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.queue_manager import group_queue_manager
from tunix.rl.queue import data_queue as queue_lib
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


def _traj(prompt_id, group_index, reward=1.0):
  """A canonical TrajectoryItem: identity only in prompt_id/group_index (no
  legacy keywords hiding an alias in metadata), Trajectory-dataclass traj.
  """
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
      prompt_id=prompt_id,
      group_index=group_index,
      start_step=0,
      traj=traj,
      metadata={},
  )


class _Trainer(SimpleNamespace):
  """Fake trainer with PeftTrainer's counter surface: `iter_steps` is a
  property over `_iter_steps`, which the restore path writes directly.
  """

  @property
  def iter_steps(self):
    return self._iter_steps

  @iter_steps.setter
  def iter_steps(self, value):
    self._iter_steps = value


# `_make_learner(checkpoint_manager_latest_step=...)` default: the stub
# reports the trainer's own train_steps (durable by construction); pass
# None for "never checkpointed".
_LATEST_IS_TRAIN_STEPS = object()


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
    checkpoint_manager_latest_step=_LATEST_IS_TRAIN_STEPS,
    full_batch_size=0,
    critic_train_steps=None,
    critic_checkpoint_manager=None,
):
  learner: Any = _Learner.__new__(_Learner)
  # Benchmark-instrumentation anchors (commit 4): _sb_snapshot reads the
  # time anchors; the restore path reads _bench_anchor_pinned.
  learner._global_step_start_time = 0.0
  learner._experiment_start_time = 0.0
  learner._bench_anchor_pinned = False
  learner._sb_chaos_prob = 0.0
  learner._bench_metrics = False
  if checkpoint_manager_latest_step is _LATEST_IS_TRAIN_STEPS:
    checkpoint_manager_latest_step = train_steps
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
  # Identity staging store (both queue paths write it; __init__ builds it).
  learner._sb_item_fifo = collections.deque()
  learner._sb_last_snapshot_key = -1
  learner._sb_window_train_steps = -1
  learner._sb_local = -1
  learner._sb_last_snapshot_trainer_iter = -1
  learner._sb_active_gids = set()
  learner._sb_step_complete_for_step = None
  learner._sb_restored_trainer_state = False
  learner._sb_restored_full_batch_size = None
  learner._sb_restored_geometry = None
  learner._sb_resumed_mid_step = False
  learner._sb_anchor_step = 0
  # Disaggregated by default: the step-start pin is active.
  learner.should_sync_weights = True
  model, accumulator, optimizer = _make_trainer_parts()
  learner._training_config = SimpleNamespace(
      get_with_default=lambda key, default: k
      if key == "gradient_accumulation_steps"
      else default,
      checkpoint_root_directory="/tmp/unused",
      max_seq_token_per_tpu=None,
      mini_batch_size=None,
      checkpointing_options=SimpleNamespace(
          save_decision_policy=SimpleNamespace(interval=1)
      ),
  )
  saved: list = []  # the fresh-run step-0 anchor save lands here
  pinned: list = []  # every CheckpointManager.pin_step call, in order
  learner.rl_engine = SimpleNamespace(
      global_steps=global_steps,
      cluster_config=SimpleNamespace(offload_to_cpu=False),
      actor_trainer=_Trainer(
          model=model,
          optimizer=optimizer,
          grad_accumulator=accumulator,
          _train_steps=train_steps,
          train_steps=train_steps,
          # SimpleNamespace writes kwargs straight into __dict__, so the
          # `iter_steps` entry only keeps vars() shaped like a plain
          # namespace for callers that copy it; the property reads
          # `_iter_steps`.
          iter_steps=train_steps * k,
          _iter_steps=train_steps * k,
          checkpoint_manager=SimpleNamespace(
              latest_step=lambda: checkpoint_manager_latest_step,
              # Every step up to the latest is on disk.
              has_step=lambda step: (
                  checkpoint_manager_latest_step is not None
                  and 0 <= step <= checkpoint_manager_latest_step
              ),
              maybe_restore=lambda *a, **kw: (kw.get("step", 0), {}),
              saved=saved,
              pinned=pinned,
              pin_step=pinned.append,
              save=lambda *a, **kw: saved.append((a, kw)),
              wait=lambda *a, **kw: None,
              # The RESOLVED options the learner's cadence check reads.
              _options=SimpleNamespace(
                  save_decision_policy=SimpleNamespace(interval=1),
              ),
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


def _load_meta(mgr, key):
  """The meta checkpointable of `key`, as the manager would read it."""
  return dict(
      mgr._checkpointer.load_checkpointables(
          key, abstract_checkpointables={"meta": None}
      )["meta"]
  )


def _plant(tmp, train_steps, local, *, k=2, geometry=None, buffer=True,
           global_step=4, step_complete=False, full_batch_size=4,
           anchor_step=None):
  """Writes one snapshot under <tmp>/sub_batch (closed afterwards): a
  mid-window buffer snapshot by default, a bufferless boundary otherwise.
  The anchor defaults to the snapshot's own parent (see save())."""
  planted = _make_manager(f"{tmp}/sub_batch", k=k)
  planted.save(
      train_steps, local, iter_steps=train_steps * k + local,
      global_step=global_step, grad_accum_steps=k,
      step_complete=step_complete, completed_group_ids=[16],
      trained_trajectory_counts={(17, 0): 1},
      active_group_trajectories=[_traj(17, 0)],
      training_state=_acc_state(w=1.0, denom=1.0) if buffer else None,
      full_batch_size=full_batch_size, geometry=geometry,
      anchor_step=anchor_step,
  )
  planted.wait()
  planted.close()


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

  def test_uniform_epoch_chunk_passes(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 1, (2, 0): 1}
    learner._sb_require_uniform_epoch([(1, 0), (2, 0)])  # no raise
    learner._sb_require_uniform_epoch([])  # empty is fine

  def test_mixed_epoch_chunk_raises(self):
    """A chunk batching a group at epoch 2 with one at epoch 1 can neither
    be trained as a unit (over-trains half of it) nor split (an extra
    micro-step desyncs the apply cadence); it means re-injection order
    diverged from the crashed run's, so it must fail loudly."""
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_counts = {(1, 0): 2, (1, 1): 2, (2, 0): 1, (2, 1): 1}
    with self.assertRaisesRegex(RuntimeError, "different trained-epoch"):
      learner._sb_require_uniform_epoch([(1, 0), (1, 1), (2, 0), (2, 1)])

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

  def test_snapshot_records_geometry(self):
    """Every snapshot's meta carries the resolved geometry: the init-known
    keys (_sb_geometry; the SimpleNamespace algo config defines no
    objective key, so none is recorded) merged with the train-time keys
    (_sb_geo_train)."""
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(mgr=mgr, k=1, train_steps=1)
      learner._sb_geo_train = {"train_micro_batch_size": 2}
      learner._sb_snapshot([(1, 0)], step_complete=False)
      mgr.wait()
      meta = _load_meta(mgr, 1 * sub_batch_checkpoint.KEY_BASE)
      self.assertEqual(meta["geometry"], {"train_micro_batch_size": 2})
      # The objective keys ride along once the algo config defines them.
      learner.algo_config.beta = 0.04
      learner.algo_config.sub_batch_dataset_id = "ds-v1"
      learner.rl_engine.actor_trainer._iter_steps += 1
      learner._sb_snapshot([(1, 0)], step_complete=False)
      mgr.wait()
      meta = _load_meta(mgr, 1 * sub_batch_checkpoint.KEY_BASE + 1)
      self.assertEqual(meta["geometry"]["beta"], 0.04)
      self.assertEqual(meta["geometry"]["dataset_id"], "ds-v1")
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_anticipated_apply_snapshot_precommits_boundary_key(self):
    """For a chunk whose training applies, the snapshot is written BEFORE
    update_actor: keyed under train_steps + 1 (local 0), iter_steps one
    ahead, no buffer even though the accumulator still holds the window's
    gradients, and made durable -- so the trainer's apply checkpoint can
    never be durable ahead of it."""
    learner = _make_learner(mgr=None, k=2, train_steps=0)
    _real_micro_step(
        learner.rl_engine.actor_trainer.model,
        learner.rl_engine.actor_trainer.grad_accumulator,
        jnp.ones((4,)),
        1.0,
    )
    saves, waits, floors = [], [], []
    learner._sb_mgr = SimpleNamespace(
        save=lambda *a, **k: saves.append((a, k)),
        wait=lambda: waits.append(1),
        set_durable_train_steps=lambda t: floors.append(t),
        enabled=True,
    )
    learner._sb_snapshot(
        [(1, 0)], step_complete=False, anticipated_apply=True
    )
    (args, kwargs), = saves
    self.assertEqual(args[:2], (1, 0))  # window T+1, local 0
    self.assertEqual(kwargs["iter_steps"], 1)  # post-training counter
    self.assertIsNone(kwargs["training_state"])  # the apply resets it
    self.assertEqual(waits, [1])  # durable before the apply runs
    self.assertEqual(floors, [0])  # the pre-apply T, reported before save
    self.assertEqual(learner._sb_window_train_steps, 1)
    self.assertEqual(learner._sb_last_snapshot_trainer_iter, 1)
    self.assertEqual(
        learner._sb_last_snapshot_key, 1 * sub_batch_checkpoint.KEY_BASE
    )

  def test_snapshot_writes_anchor_step(self):
    """Every snapshot records the step-start train_steps the learner
    tracks, so a mid-step resume knows which checkpoint to pin."""
    learner = _make_learner(mgr=None, k=2, train_steps=3)
    learner._sb_anchor_step = 1
    saves = []
    learner._sb_mgr = SimpleNamespace(
        save=lambda *a, **k: saves.append((a, k)), wait=lambda: None,
        set_durable_train_steps=lambda t: None, enabled=True,
    )
    learner.rl_engine.actor_trainer.iter_steps = 7
    learner._sb_snapshot([(1, 0)], step_complete=False)
    (_, kwargs), = saves
    self.assertEqual(kwargs["anchor_step"], 1)

  def test_step_boundary_advances_anchor(self):
    """The boundary runs right after the step's final apply: the trainer's
    train_steps there is the next step's behavior policy."""
    learner = _make_learner(train_steps=1, global_steps=2)
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_step_complete_for_step = 2
    learner._sb_anchor_step = 1
    learner.rl_engine.actor_trainer.train_steps = 4
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_anchor_step, 4)
    pinned = learner.rl_engine.actor_trainer.checkpoint_manager.pinned
    self.assertEqual(pinned, [4])

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
          [t.prompt_id for t in st.active_group_trajectories], [2 * 4 + 1]
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
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
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
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, set())
    self.assertEqual(learner._sb_counts, {})
    self.assertEqual([t.prompt_id for t in learner._sb_active], [3 * 4 + 0])
    # And the retained registration still shields the group from
    # regeneration when its step arrives.
    self.assertTrue(learner._sb_skip_group(3 * 4 + 0))

  def test_step_boundary_noop_when_disabled(self):
    learner = _make_learner(mgr=None)
    learner._sb_completed = {1}
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, {1})  # untouched


class SubBatchBufferTest(parameterized.TestCase):
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

  def test_extract_is_a_host_copy_independent_of_the_next_micro_step(self):
    """The jitted train step donates the accumulator, so the extract must
    be a copy the async save can keep reading: per-shard, on the
    pinned_host memory kind, still a jax array (no numpy gather, which a
    multi-host shard cannot provide). It must survive the accumulator
    moving on, and restore bit-exactly through the real manager onto the
    accumulator's own sharding."""
    learner = _make_learner(k=2)
    tr = learner.rl_engine.actor_trainer
    _real_micro_step(
        tr.model, tr.grad_accumulator, jnp.array([1.0, 2.0, 3.0, 4.0]), 5.0
    )
    diff = learner._sb_extract_accumulator(tr)
    expected = {k: np.asarray(v) for k, v in diff.items()}
    for leaf in diff.values():
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.sharding.memory_kind, "pinned_host")
    # The accumulator moves on (as the next micro-step would); the copy
    # does not.
    _real_micro_step(
        tr.model, tr.grad_accumulator, jnp.array([9.0, 9.0, 9.0, 9.0]), 1.0
    )
    for k, v in diff.items():
      np.testing.assert_array_equal(np.asarray(v), expected[k])
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=2)
      mgr.save(
          1, 1, iter_steps=3, global_step=0, grad_accum_steps=2,
          step_complete=False, completed_group_ids=[],
          trained_trajectory_counts={}, active_group_trajectories=[],
          training_state=diff, anchor_step=1,
      )
      mgr.wait()
      target = learner._sb_build_abstract_training_state()
      st = mgr.try_restore(
          train_steps=1, grad_accum_steps=2, target_training_state=target
      )
      for k, v in st.training_state.items():
        np.testing.assert_array_equal(np.asarray(v), expected[k])
      learner._sb_inject_training_state(st.training_state)
      for path, var in nnx.state(tr.grad_accumulator).flat_state():
        np.testing.assert_array_equal(np.asarray(var[...]), expected[path])
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  @parameterized.named_parameters(
      ("device", "device"), ("already_host", "pinned_host")
  )
  def test_snapshot_survives_donated_input(self, memory_kind):
    learner = _make_learner(k=2)
    trainer = learner.rl_engine.actor_trainer
    acc = trainer.grad_accumulator
    state = nnx.state(acc)
    for _, var in state.flat_state():
      value = var[...]
      var.set_value(jax.device_put(
          value, value.sharding.with_memory_kind(memory_kind)
      ))
    nnx.update(acc, state)
    snapshot = learner._sb_extract_accumulator(trainer)
    expected = {p: np.array(v, copy=True) for p, v in snapshot.items()}
    for _, var in nnx.state(acc).flat_state():
      value = var[...]
      donated_update = jax.jit(
          lambda x: x + 1, donate_argnums=(0,),
          in_shardings=value.sharding, out_shardings=value.sharding,
      )
      donated_update(value).block_until_ready()
      self.assertTrue(value.is_deleted())
    for path, value in snapshot.items():
      self.assertFalse(value.is_deleted())
      np.testing.assert_array_equal(value, expected[path])

  def test_abstract_target_does_not_stage_accumulator_contents(self):
    learner = _make_learner(k=2)
    with mock.patch.object(
        learner, "_sb_extract_accumulator",
        side_effect=AssertionError("restore metadata must not copy values"),
    ):
      target = learner._sb_build_abstract_training_state()
    self.assertEqual(target[("grads", "w")].shape, (4,))
    self.assertEqual(target[("denom",)].shape, ())

  @parameterized.named_parameters(
      ("valid", ("data",), (4,), False),
      ("unknown_axis", ("unknown",), (4,), True),
      ("insufficient_rank", ("data", None), (4,), True),
      ("scalar", ("data",), (), True),
      ("indivisible", ("data",), (3,), True),
  )
  def test_restore_placement_matches_trainer_fallbacks(
      self, spec, shape, replicated
  ):
    # Run this suite with four CPU devices to exercise indivisible dimensions.
    if shape == (3,) and len(jax.devices()) != 4:
      self.skipTest("requires four devices for indivisible-axis coverage")
    learner = _make_learner(k=2)
    trainer = learner.rl_engine.actor_trainer
    trainer.model.w.set_metadata(sharding=spec)
    trainer.grad_accumulator = peft_trainer.GradientAccumulator(
        trainer.model, nnx.Param
    )
    trainer.grad_accumulator.grads["w"].set_value(jnp.ones(shape))
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
    learner.rl_engine.cluster_config.role_to_mesh = {
        rl_cluster.Role.ACTOR: mesh
    }
    target = learner._sb_build_abstract_training_state()
    expected = jax.sharding.PartitionSpec(
        *(() if replicated else spec)
    )
    self.assertEqual(target[("grads", "w")].sharding,
                     jax.sharding.NamedSharding(mesh, expected))
    self.assertEqual(target[("denom",)].sharding.spec,
                     jax.sharding.PartitionSpec())


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
          anchor_step=0,
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
      self.assertFalse(learner._sb_resumed_mid_step)
      # The fresh run anchors window 0 with a durable step-0 weight
      # checkpoint stamped "resume at step 0", so a preemption before the
      # first apply resumes instead of retraining the window.
      saved = learner.rl_engine.actor_trainer.checkpoint_manager.saved
      self.assertEqual(len(saved), 1)
      args, kwargs = saved[0]
      self.assertEqual(args[0], 0)
      self.assertTrue(kwargs["force"])
      self.assertEqual(kwargs["custom_metadata"]["global_step"], 0)
      # The anchor is durable, so the floor is set before any save.
      self.assertEqual(
          learner._sb_mgr._options.preservation_policy.durable_train_steps, 0
      )
      learner._sb_mgr.close()
    finally:
      shutil.rmtree(root)

  def test_window_zero_resumes_from_anchored_trainer(self):
    """A preemption BEFORE the first apply: the trainer restored the step-0
    anchor (latest_step() == 0, train_steps == 0), so the window-0 snapshot
    must be selected and resumed like any other window -- ledger seeded,
    buffer injected, trainer counter advanced, global_steps at 0."""
    planted = _make_manager(f"{self.tmp}/sub_batch", k=4)
    state = _acc_state(w=3.0, denom=3.0)
    planted.save(
        0, 2, iter_steps=3, global_step=0, grad_accum_steps=4,
        step_complete=False, completed_group_ids=[0, 1],
        trained_trajectory_counts={(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        active_group_trajectories=[_traj(0, 0), _traj(0, 1)],
        training_state=state,
        anchor_step=0,
    )
    planted.wait()
    planted.close()

    learner = _make_learner(
        mgr=None, k=4, train_steps=0, global_steps=1,
        checkpoint_manager_latest_step=0,
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()

    self.assertIsNotNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_completed, {0, 1})
    self.assertEqual(learner.rl_engine.actor_trainer._iter_steps, 3)
    self.assertEqual(learner.rl_engine.global_steps, 0)
    self.assertTrue(learner._sb_resumed_mid_step)  # suppresses the re-eval
    acc = learner.rl_engine.actor_trainer.grad_accumulator
    np.testing.assert_array_equal(
        np.asarray(acc.grads["w"][...]), np.array([3.0] * 4)
    )
    self.assertEqual(float(np.asarray(acc.denom[...])), 3.0)
    # No second anchor: the trainer already had a checkpoint.
    self.assertEqual(
        learner.rl_engine.actor_trainer.checkpoint_manager.saved, []
    )
    # Restored train_steps is durable by construction: floor set at init.
    self.assertEqual(
        learner._sb_mgr._options.preservation_policy.durable_train_steps, 0
    )
    learner._sb_mgr.close()

  def test_missing_window_raises(self):
    """Trainer checkpoint T=1 stamped for step 0 (global_step 1) with no
    window-1 snapshot, while window 0's latest snapshot is mid-step 0: the
    commit order that _sb_train_chunk guarantees was violated and the rest
    of step 0 would be lost. Init must STOP (the manager's
    SubBatchLedgerMissingError propagates, the orbax checkpointer is
    closed on the way out) and leave the window-0 evidence intact."""
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    planted.save(
        0, 1, iter_steps=1, global_step=0, grad_accum_steps=2,
        step_complete=False, completed_group_ids=[],
        trained_trajectory_counts={(0, 0): 1, (0, 1): 1},
        active_group_trajectories=[_traj(0, 0), _traj(0, 1)],
        training_state=_acc_state(),
        anchor_step=0,
    )
    planted.wait()
    planted.close()
    learner = _make_learner(
        mgr=None, k=2, train_steps=1, global_steps=1,
        checkpoint_manager_latest_step=1,
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchLedgerMissingError, "train_steps=1"
    ):
      learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner.rl_engine.global_steps, 1)  # untouched
    evidence = _make_manager(f"{self.tmp}/sub_batch", k=2)
    try:
      self.assertEqual(evidence._select_step(0), 1)
    finally:
      evidence.close()

  def test_empty_root_on_existing_run_warns_and_starts_fresh(self):
    """The feature enabled on a run that already has trainer checkpoints
    (T>0) but no ledger at all: the one no-ledger case that is not a stop.
    Stock restart semantics, said explicitly at WARNING."""
    learner = _make_learner(
        mgr=None, k=2, train_steps=5, global_steps=5,
        checkpoint_manager_latest_step=5,
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    with self.assertLogs(level="WARNING") as logs:
      learner._init_sub_batch_checkpointing()
    joined = "\n".join(logs.output)
    self.assertIn("existing run", joined)
    self.assertIn("NOT resumed", joined)
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner.rl_engine.global_steps, 5)
    self.assertEqual(
        learner._sb_mgr._options.preservation_policy.durable_train_steps, 5
    )
    learner._sb_mgr.close()
    # No update happened. A second initialization reuses the same baseline.
    self.assertTrue(learner._sb_mgr.enable_marker_path.exists())
    learner = _make_learner(
        mgr=None, k=2, train_steps=5, global_steps=5,
        checkpoint_manager_latest_step=5,
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    learner._sb_mgr.close()

  def test_fresh_start_writes_enable_marker(self):
    learner = _make_learner(
        mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertEqual(learner._sb_mgr.enable_marker_path.read_text(), "0")
    self.assertEqual(str(learner._sb_mgr.enable_marker_path.parent), self.tmp)
    learner._sb_mgr.close()

  def test_missing_trainer_does_not_purge_another_enrollment(self):
    _plant(self.tmp, 6, 0, buffer=False)
    mgr = _make_manager(f"{self.tmp}/sub_batch", k=1)
    mgr.mark_enabled(5)
    mgr.close()
    learner = _make_learner(
        mgr=None, k=1, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchLedgerMissingError, "another baseline"
    ):
      learner._init_sub_batch_checkpointing()
    evidence = _make_manager(f"{self.tmp}/sub_batch", k=1)
    try:
      self.assertEqual(evidence._select_step(6), 6_000_000)
    finally:
      evidence.close()

  def test_clean_restart_still_validates_batch_geometry(self):
    """A completed-step snapshot seeds nothing, but its recorded geometry
    still guards the relaunch: a different full_batch_size would
    fast-forward a different prompt partition."""
    _plant(self.tmp, 2, 0, step_complete=True, buffer=False,
           full_batch_size=4,
           geometry={"max_seq_token_per_tpu": None,
                     "train_micro_batch_size": 2})
    learner = _make_learner(mgr=None, k=2, train_steps=2, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "full_batch_size"
    ):
      learner._sb_validate_batch_geometry(8)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "train_micro_batch_size"
    ):
      learner._sb_validate_batch_geometry(4, train_micro_batch_size=1)
    learner._sb_validate_batch_geometry(4, train_micro_batch_size=2)
    learner._sb_mgr.close()

  def test_colocated_trainer_old_logps_rejected(self):
    """Colocated runs cannot re-anchor across a restart: configurations
    that consume trainer-recomputed old logps are refused at init."""
    for algo in ({"use_rollout_logps": False}, {"sampler_is": "token"}):
      with self.subTest(**algo):
        learner = _make_learner(
            mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
        )
        learner._training_config.checkpoint_root_directory = self.tmp
        learner.should_sync_weights = False
        for name, value in algo.items():
          setattr(learner.algo_config, name, value)
        with self.assertRaisesRegex(ValueError, "colocated"):
          learner._init_sub_batch_checkpointing()
    # Disaggregated, or colocated with rollout logps: fine.
    learner = _make_learner(
        mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner.should_sync_weights = False
    learner.algo_config.use_rollout_logps = True
    learner._init_sub_batch_checkpointing()
    learner._sb_mgr.close()

  def test_close_closes_sub_batch_manager(self):
    learner = _make_learner(
        mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    mgr = learner._sb_mgr
    closed = []
    mgr.close = lambda: closed.append(True)
    learner.close()
    self.assertEqual(closed, [True])
    self.assertIsNone(learner._sb_mgr)
    self.assertFalse(learner._sb_enabled)

  def test_unreadable_snapshot_raises_instead_of_new_step(self):
    """The selected key (the 3.0 precommit) is torn: init must STOP, not
    purge it and silently start the next step at the trainer's stamp."""
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    planted.save(
        2, 1, iter_steps=5, global_step=4, grad_accum_steps=2,
        step_complete=False, completed_group_ids=[16],
        trained_trajectory_counts={(17, 0): 1},
        active_group_trajectories=[_traj(17, 0)],
        training_state=_acc_state(),
        anchor_step=2,
    )
    planted.save(
        3, 0, iter_steps=6, global_step=4, grad_accum_steps=2,
        step_complete=False, completed_group_ids=[16, 17],
        trained_trajectory_counts={(17, 0): 1, (17, 1): 1},
        active_group_trajectories=[_traj(17, 0), _traj(17, 1)],
        training_state=None,
        anchor_step=2,
    )
    planted.wait()
    planted.close()
    KB = sub_batch_checkpoint.KEY_BASE
    shutil.rmtree(f"{self.tmp}/sub_batch/{3 * KB}/meta")

    learner = _make_learner(
        mgr=None, k=2, train_steps=3, global_steps=5,
        checkpoint_manager_latest_step=3,
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchUnreadableError, "torn"
    ):
      learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner.rl_engine.global_steps, 5)
    evidence = _make_manager(f"{self.tmp}/sub_batch", k=2)
    try:
      self.assertEqual(
          sorted(ck.step for ck in evidence._checkpointer.checkpoints),
          [2 * KB + 1, 3 * KB],
      )
    finally:
      evidence.close()

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
    # target's structure not match what was saved, and try_restore raises
    # SubBatchUnreadableError (verified against the real checkpointer
    # while debugging this test).
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
        anchor_step=1,
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
        anchor_step=2,
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

  def test_full_batch_size_change_raises_and_preserves_snapshots(self):
    """full_batch_size changed across the restart: the deferred check in
    _sb_validate_batch_geometry must RAISE (resume assumes relaunch with the
    same configuration -- there is no cross-geometry adaptation) and must
    leave the on-disk snapshots intact, so a relaunch with the ORIGINAL
    configuration resumes from them normally."""
    planted = _make_manager(f"{self.tmp}/sub_batch", k=2)
    state = _acc_state(w=1.0, denom=1.0)
    planted.save(
        1, 1, iter_steps=3, global_step=4, grad_accum_steps=2,
        step_complete=False, completed_group_ids=[16],
        trained_trajectory_counts={(17, 0): 1},
        active_group_trajectories=[_traj(17, 0)], training_state=state,
        full_batch_size=4,
        anchor_step=1,
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

    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "full_batch_size=4"
    ):
      learner._sb_validate_batch_geometry(8)  # crashed run saved 4

    # Matching geometry passes, and nothing was purged by the failed check:
    # the mid-window key is still on disk for the corrected relaunch.
    learner._sb_validate_batch_geometry(4)
    self.assertIsNotNone(learner._sb_pending_state)
    self.assertEqual(
        learner._sb_mgr._select_step(1),
        1 * sub_batch_checkpoint.KEY_BASE + 1,
    )
    learner._sb_mgr.close()

  def test_mid_step_resume_reads_anchor_step(self):
    """A crash after an apply inside the step: the snapshot's parent (3)
    is the restored weights, its anchor (1) is the step start, and the
    learner tracks the latter for the behavior-policy pin."""
    _plant(self.tmp, 3, 1, anchor_step=1)
    learner = _make_learner(mgr=None, k=2, train_steps=3, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNotNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_pending_state.anchor_step, 1)
    self.assertEqual(learner._sb_anchor_step, 1)
    pinned = learner.rl_engine.actor_trainer.checkpoint_manager.pinned
    self.assertEqual(pinned[-1], 1)  # the step start, not the restored 3
    learner._sb_mgr.close()

  def test_colocated_records_anchor_without_pinning(self):
    """Colocated rollouts are never re-anchored, so the step start is
    recorded (a disaggregated relaunch can still use it) but not pinned on
    the trainer's checkpoint manager, whose retention stays as configured."""
    learner = _make_learner(
        mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
    )
    learner.should_sync_weights = False
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertEqual(learner._sb_anchor_step, 0)
    learner.rl_engine.actor_trainer.train_steps = 3
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
    learner._sb_step_boundary()
    self.assertEqual(learner._sb_anchor_step, 3)
    pinned = learner.rl_engine.actor_trainer.checkpoint_manager.pinned
    self.assertEqual(pinned, [])
    learner._sb_mgr.close()

  def test_step_complete_and_fresh_start_anchor_is_train_steps(self):
    """Where nothing is resumed mid-step the restored (or initial) weights
    ARE the step start: fresh run, clean-boundary restart (whatever the
    finished step's snapshot recorded), and an empty window."""
    learner = _make_learner(
        mgr=None, k=1, train_steps=0, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertEqual(learner._sb_anchor_step, 0)
    pinned = learner.rl_engine.actor_trainer.checkpoint_manager.pinned
    self.assertEqual(pinned, [0])
    learner._sb_mgr.close()

    _plant(self.tmp, 2, 0, step_complete=True, buffer=False,
           anchor_step=1)
    learner = _make_learner(mgr=None, k=2, train_steps=2, global_steps=5)
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_pending_state)
    self.assertEqual(learner._sb_anchor_step, 2)
    pinned = learner.rl_engine.actor_trainer.checkpoint_manager.pinned
    self.assertEqual(pinned, [2])
    learner._sb_mgr.close()

    # First-enable is a separate run, not removal of an enrolled ledger.
    learner = _make_learner(mgr=None, k=2, train_steps=5, global_steps=5)
    learner._training_config.checkpoint_root_directory = (
        f"{self.tmp}/first_enable"
    )
    learner._init_sub_batch_checkpointing()
    self.assertEqual(learner._sb_anchor_step, 5)
    learner._sb_mgr.close()

  def test_objective_change_rejected(self):
    """A retuned objective refuses the resume (None is a value), with or
    without a gradient buffer in the snapshot; an algo config without the
    keys resumes."""
    geo = {
        "max_seq_token_per_tpu": None, "use_rollout_logps": False,
        "sampler_is": "token", "beta": 0.0,
    }
    _plant(self.tmp, 1, 1, geometry=geo)

    def relaunch(**algo):
      learner = _make_learner(mgr=None, k=2, train_steps=1, global_steps=5)
      learner._training_config.checkpoint_root_directory = self.tmp
      for name, value in algo.items():
        setattr(learner.algo_config, name, value)
      learner._init_sub_batch_checkpointing()
      return learner

    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "use_rollout_logps"
    ):
      relaunch(use_rollout_logps=True)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "sampler_is"
    ):
      relaunch(use_rollout_logps=False, sampler_is=None)
    learner = relaunch(use_rollout_logps=False, sampler_is="token", beta=0.0)
    self.assertIsNotNone(learner._sb_pending_state)
    learner._sb_mgr.close()
    learner = relaunch()  # no objective keys defined: skipped
    self.assertIsNotNone(learner._sb_pending_state)
    learner._sb_mgr.close()

    # A bufferless boundary snapshot refuses it too.
    shutil.rmtree(f"{self.tmp}/sub_batch")
    _plant(self.tmp, 1, 1, geometry=geo, buffer=False)
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "beta"
    ):
      relaunch(use_rollout_logps=False, sampler_is="token", beta=0.04)

  def test_dataset_id_checked_only_when_both_present(self):
    """sub_batch_dataset_id is caller-asserted: compared only when both the
    snapshot and the relaunch carry one."""
    _plant(self.tmp, 1, 1, geometry={
        "max_seq_token_per_tpu": None, "dataset_id": "ds-v1",
    })

    def relaunch(dataset_id):
      learner = _make_learner(mgr=None, k=2, train_steps=1, global_steps=5)
      learner._training_config.checkpoint_root_directory = self.tmp
      if dataset_id is not None:
        learner.algo_config.sub_batch_dataset_id = dataset_id
      learner._init_sub_batch_checkpointing()
      self.assertIsNotNone(learner._sb_pending_state)
      learner._sb_mgr.close()

    relaunch(None)
    relaunch("ds-v1")
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "dataset_id"
    ):
      relaunch("ds-v2")
    shutil.rmtree(f"{self.tmp}/sub_batch")
    _plant(self.tmp, 1, 1, geometry={"max_seq_token_per_tpu": None})
    relaunch("ds-v1")  # absent on the saved side: skipped

  def test_accum_window_must_be_one_mini_batch(self):
    """mini_batch_size == train_micro_batch_size * k is what the apply
    prediction assumes; checked at init when the config carries both
    sizes."""
    learner = _make_learner(
        mgr=None, k=2, checkpoint_manager_latest_step=None
    )
    learner._training_config.checkpoint_root_directory = self.tmp
    learner._training_config.mini_batch_size = 4
    learner._training_config.train_micro_batch_size = 1
    with self.assertRaisesRegex(ValueError, "4 != 1 \\* 2"):
      learner._init_sub_batch_checkpointing()
    self.assertIsNone(learner._sb_mgr)  # raised before construction
    learner._training_config.train_micro_batch_size = 2
    learner._init_sub_batch_checkpointing()
    learner._sb_mgr.close()


class SubBatchTrainChunkTest(absltest.TestCase):
  """_sb_train_chunk's crash-safe order (P1.2): the previous apply's weight

  checkpoint is proven durable BEFORE the boundary ledger is precommitted
  (that precommit is where retention may drop the window a crash would
  otherwise restore to), the floor moves only from that anticipated
  snapshot, and an apply the prediction missed fails at once.
  """

  def _stubbed(self, learner, *, bump_on_apply):
    events = []
    tr = learner.rl_engine.actor_trainer
    tr.checkpoint_manager.wait = lambda: events.append("trainer_wait")
    learner._sb_snapshot = lambda *a, **kw: events.append("ledger_save")

    def apply(*a):
      events.append("apply")
      tr.iter_steps += 1
      if bump_on_apply:
        tr.train_steps += 1

    learner.rl_engine.update_actor = apply
    return events

  def _chunk(self, apply):
    return SimpleNamespace(is_update_step=np.array(apply))

  def test_apply_chunk_waits_for_trainer_then_precommits_then_applies(self):
    learner = _make_learner(mgr=object(), k=1)
    events = self._stubbed(learner, bump_on_apply=True)
    learner._sb_train_chunk(
        self._chunk(True), [], None, False, step_complete=False
    )
    self.assertEqual(events, ["trainer_wait", "ledger_save", "apply"])
    self.assertEqual(learner.rl_engine.actor_trainer.train_steps, 1)

  def test_apply_chunk_raises_when_trainer_checkpoint_lags(self):
    """After the wait the trainer's latest durable step must be its

    train_steps; a lagging stream means the per-apply cadence is broken
    and the ledger could never be reconciled. Nothing is precommitted.
    """
    learner = _make_learner(
        mgr=object(), k=1, train_steps=1, checkpoint_manager_latest_step=0
    )
    events = self._stubbed(learner, bump_on_apply=True)
    with self.assertRaisesRegex(RuntimeError, "durable checkpoint step is 0"):
      learner._sb_train_chunk(
          self._chunk(True), [], None, False, step_complete=False
      )
    self.assertEqual(events, ["trainer_wait"])

  def test_train_chunk_raises_on_unpredicted_apply(self):
    """An apply the prediction missed: its ledger was written (or would be)
    under the wrong window, so fail at once rather than snapshot."""
    learner = _make_learner(mgr=object(), k=1)
    events = self._stubbed(learner, bump_on_apply=True)
    with self.assertRaisesRegex(RuntimeError, "predicted apply=False"):
      learner._sb_train_chunk(
          self._chunk(False), [], None, False, step_complete=False
      )
    self.assertEqual(events, ["apply"])  # no wait, no ledger_save

  def test_train_chunk_raises_when_predicted_apply_did_not_happen(self):
    """A predicted apply that the trainer did not perform: its ledger was
    precommitted for a window the weights never reached, and there may be
    no next chunk to notice (step completion), so it fails right here."""
    learner = _make_learner(mgr=object(), k=1)
    events = self._stubbed(learner, bump_on_apply=False)
    with self.assertRaisesRegex(RuntimeError, "predicted apply=True"):
      learner._sb_train_chunk(
          self._chunk(True), [], None, False, step_complete=True
      )
    self.assertEqual(events, ["trainer_wait", "ledger_save", "apply"])

  def test_train_chunk_rejects_missing_micro_step(self):
    learner = _make_learner(mgr=object(), k=2)
    self._stubbed(learner, bump_on_apply=False)
    learner.rl_engine.update_actor = lambda *args: None
    with self.assertRaisesRegex(RuntimeError, "exactly one micro-step"):
      learner._sb_train_chunk(
          self._chunk(False), [], None, False, step_complete=False
      )

  def test_mid_window_chunk_does_not_wait_on_trainer(self):
    learner = _make_learner(mgr=object(), k=2)
    events = self._stubbed(learner, bump_on_apply=False)
    learner._sb_train_chunk(
        self._chunk(False), [], None, False, step_complete=False
    )
    self.assertEqual(events, ["apply", "ledger_save"])

  def test_anticipated_snapshot_raises_floor_mid_window_does_not(self):
    """Only the anticipated (precommit) snapshot reports the floor, with the

    PRE-apply train_steps: a mid-window snapshot's train_steps already
    counts an apply whose weight checkpoint may still be in flight.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(tmp)
      policy = mgr._options.preservation_policy
      learner = _make_learner(
          num_iterations=1, num_generations=1, mgr=mgr, k=2, train_steps=3
      )
      learner._sb_snapshot([], step_complete=False)
      self.assertIsNone(policy.durable_train_steps)
      learner._sb_snapshot([], step_complete=False, anticipated_apply=True)
      self.assertEqual(policy.durable_train_steps, 3)
      self.assertEqual(
          learner._sb_last_snapshot_key, 4 * sub_batch_checkpoint.KEY_BASE
      )
      mgr.close()
    finally:
      shutil.rmtree(tmp)

  def test_anticipated_snapshot_asserts_trainer_durability(self):
    """The floor raise is tied to the durability proof: an anticipated

    snapshot whose trainer checkpoint lags train_steps is a wiring bug and
    must not report anything.
    """
    learner = _make_learner(
        mgr=None, k=2, train_steps=3, checkpoint_manager_latest_step=2
    )
    floors = []
    learner._sb_mgr = SimpleNamespace(
        save=lambda *a, **k: None,
        wait=lambda: None,
        set_durable_train_steps=lambda t: floors.append(t),
        enabled=True,
    )
    with self.assertRaises(AssertionError):
      learner._sb_snapshot([], step_complete=False, anticipated_apply=True)
    self.assertEqual(floors, [])

  def test_previous_apply_durability_gates_retention(self):
    """Against the real async manager: with W(1) still in flight the apply

    chunk for window 2 blocks in the trainer wait and parent 0 -- the pair
    a crash would restore to -- stays on disk; once W(1) is durable the
    (2).0 precommit drops it. k=2 so window 0 has a ledger of its own.
    """
    tmp = tempfile.mkdtemp()
    try:
      mgr = sub_batch_checkpoint.SubBatchCheckpointManager(tmp)
      mgr.set_durable_train_steps(0)  # as the fresh-start anchor does
      learner = _make_learner(
          num_iterations=1, num_generations=1, mgr=mgr, k=2, train_steps=0
      )
      tr = learner.rl_engine.actor_trainer
      w_durable = threading.Event()
      w_durable.set()
      tr.checkpoint_manager.wait = w_durable.wait
      tr.checkpoint_manager.latest_step = lambda: tr.train_steps

      def update_actor(batch, *a):
        tr.iter_steps += 1
        if bool(np.asarray(batch[0].is_update_step).item()):
          tr.train_steps += 1

      learner.rl_engine.update_actor = update_actor
      KB = sub_batch_checkpoint.KEY_BASE

      def train(apply):
        learner._sb_train_chunk(
            self._chunk(apply), [], None, False, step_complete=False
        )

      def on_disk():
        mgr.wait()
        return sorted(ck.step for ck in mgr._checkpointer.checkpoints)

      train(False)  # L0.0
      train(True)  # W0 durable -> L1.0 precommit -> apply: W1 in flight
      w_durable.clear()
      train(False)  # L1.1 (retires L1.0: newest key per window)
      self.assertEqual(on_disk(), [0, 1 * KB + 1])
      t = threading.Thread(target=train, args=(True,), daemon=True)
      t.start()
      t.join(0.5)
      self.assertTrue(t.is_alive())  # blocked on W1, nothing precommitted
      self.assertEqual(on_disk(), [0, 1 * KB + 1])
      self.assertEqual(mgr._options.preservation_policy.durable_train_steps, 0)
      w_durable.set()
      t.join(10)
      self.assertFalse(t.is_alive())
      self.assertEqual(on_disk(), [1 * KB + 1, 2 * KB])
      self.assertEqual(mgr._options.preservation_policy.durable_train_steps, 1)
      self.assertEqual(tr.train_steps, 2)
      mgr.close()
    finally:
      shutil.rmtree(tmp)


class SubBatchResyncTest(absltest.TestCase):
  """_sb_resync_rollout_weights: the disaggregated-restart weight refresh

  (mandatory on EVERY restart that restored trainer
  weights). The production topology is vLLM server mode, where
  should_sync_weights is True and rl_engine.sync_weights increments
  global_steps as a side effect (rl_cluster.py); the refresh must reuse that
  tested transfer path while compensating the increment, or every preemption
  shifts the dataset fast-forward and the producer's group-id base by one.
  """

  def _learner(self, *, restored, disaggregated, train_steps=0, anchor=0,
               mid_step=False):
    """`calls` records the engine's sync_weights / pin_behavior_policy
    calls and the learner's anchor-checkpoint loads, in order."""
    learner = _make_learner(k=1, global_steps=5, train_steps=train_steps)
    learner._sb_mgr = SimpleNamespace(enabled=True)  # _sb_enabled True
    learner._sb_restored_trainer_state = restored
    learner.should_sync_weights = disaggregated
    learner._sb_anchor_step = anchor
    learner._sb_pending_state = object() if mid_step else None
    calls = []
    cluster = learner.rl_engine

    def sync_weights():
      calls.append("sync")
      cluster.global_steps += 1  # mirrors rl_engine.sync_weights

    cluster.sync_weights = sync_weights
    cluster.pin_behavior_policy = lambda p: calls.append(("pin", p))
    learner._sb_load_anchor_params = (
        lambda step: calls.append(("load", step)) or f"W({step})"
    )
    return learner, calls

  def test_disaggregated_restart_syncs_once_without_global_steps_drift(self):
    learner, calls = self._learner(restored=True, disaggregated=True)
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, ["sync"])  # rollout engine actually refreshed
    self.assertEqual(learner.rl_engine.global_steps, 5)  # increment undone

  def test_colocated_restart_does_not_sync(self):
    learner, calls = self._learner(restored=True, disaggregated=False)
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [])

  def test_nothing_restored_does_not_sync(self):
    learner, calls = self._learner(restored=False, disaggregated=True)
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [])

  def test_mid_step_resume_pins_step_start_checkpoint(self):
    """Resumed at train_steps=3 inside a step that started at 1: the
    step-start checkpoint is loaded once and pinned as anchor + rollout
    policy; no sync (that would push W(3)), global_steps untouched."""
    learner, calls = self._learner(
        restored=True, disaggregated=True, train_steps=3, anchor=1,
        mid_step=True,
    )
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [("load", 1), ("pin", "W(1)")])
    self.assertEqual(learner.rl_engine.global_steps, 5)

  def test_mid_step_resume_missing_anchor_checkpoint_raises(self):
    """W(1) is gone from disk despite the pin: raise before touching
    anything."""
    learner, calls = self._learner(
        restored=True, disaggregated=True, train_steps=3, anchor=1,
        mid_step=True,
    )
    learner.rl_engine.actor_trainer.checkpoint_manager.has_step = (
        lambda step: step == 3
    )
    with self.assertRaisesRegex(RuntimeError, "train_steps=1.*pin"):
      learner._sb_resync_rollout_weights()
    self.assertEqual(calls, [])

  def test_mid_step_resume_at_step_start_syncs(self):
    """Crash before the step's first apply: the restored weights are the
    step start, so today's sync (increment undone) is the whole refresh."""
    learner, calls = self._learner(
        restored=True, disaggregated=True, train_steps=1, anchor=1,
        mid_step=True,
    )
    learner._sb_resync_rollout_weights()
    self.assertEqual(calls, ["sync"])
    self.assertEqual(learner.rl_engine.global_steps, 5)

  def test_colocated_mid_step_resume_does_not_pin(self):
    """Colocated rollouts share the live weights: nothing is pushed and
    the anchor is not re-pinned (upstream never re-anchors there)."""
    learner, calls = self._learner(
        restored=True, disaggregated=False, train_steps=3, anchor=1,
        mid_step=True,
    )
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

  def test_boundary_without_step_complete_snapshot_raises(self):
    """Raised BEFORE the ledger reset: the last snapshot on disk is still a
    matched recovery pair for this step."""
    learner = self._learner()
    learner._sb_completed = {1}
    self.assertIsNone(learner._sb_step_complete_for_step)
    with self.assertRaisesRegex(RuntimeError, "phantom step"):
      learner._sb_step_boundary()
    self.assertEqual(learner._sb_completed, {1})  # ledger untouched

  def test_boundary_after_step_complete_snapshot_is_quiet(self):
    learner = self._learner()
    # As _sb_snapshot stamps it: the step currently being finished.
    learner._sb_step_complete_for_step = learner.rl_engine.global_steps
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
    learner._sb_step_boundary()  # step 3: correctly marked
    learner.rl_engine.global_steps += 1  # step 4 finishes unmarked
    with self.assertRaisesRegex(RuntimeError, "step_complete"):
      learner._sb_step_boundary()

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
        [(t.prompt_id, t.group_index) for t in all_items],
        [(1, 0), (1, 1), (2, 0)],
    )

  def test_reinject_false_path_wraps_identity(self):
    # process_in_consumer=False: each group's TrainExample travels as an
    # _SbItem carrying its rows' identities, in ledger order.
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_pending_state = object()
    learner._sb_active = [_traj(1, 0), _traj(1, 1), _traj(2, 0)]
    learner._batch_to_train_example = lambda batch_results, mode: [
        f"te{batch_results[0].prompt_id}"
    ]
    q = queue.Queue()
    learner._sb_reinject(q)
    items = []
    while not q.empty():
      items.append(q.get())
    self.assertEqual(
        [type(i) for i in items], [agentic_rl_learner._SbItem] * 2
    )
    self.assertEqual([i.example for i in items], ["te1", "te2"])
    self.assertEqual([i.ids for i in items], [[(1, 0), (1, 1)], [(2, 0)]])

  def test_reinject_noop_when_nothing_pending(self):
    learner = _make_learner()
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_pending_state = None
    q = queue.Queue()
    learner._sb_reinject(q)
    self.assertTrue(q.empty())


class SubBatchQueueTest(absltest.TestCase):
  """Identity rides the one train_data_queue item (no side queue)."""

  @staticmethod
  def _four_groups(**kwargs):
    async def groups(**unused):
      for gid in range(4):
        yield [_traj(gid, 0), _traj(gid, 1)]

    return groups

  def test_single_queue_four_group_microbatch(self):
    # A micro-batch of four groups must arrive whatever the queue capacity:
    # with a bounded side queue the producer stalled on the identity put
    # that the consumer only relieved after four data gets.
    for capacity in (0, 2):
      with self.subTest(capacity=capacity):
        learner = _make_learner(mgr=object(), k=1)
        data = queue_lib.SimpleDataQueue(maxsize=capacity)
        learner._sb_reinject = lambda q: None
        learner._batch_to_train_example = lambda batch_results, mode: [
            batch_results[0].prompt_id
        ]
        learner._orchestrator_producer = self._four_groups()
        errors = []

        def produce():
          try:
            asyncio.run(learner._producer(None, queue.Queue(), data))
          except BaseException as exc:  # pylint: disable=broad-except
            errors.append(exc)

        received = []
        finished = threading.Event()

        def consume():
          gen = learner._sb_unwrap(
              learner._data_consumer_batch_generator(data, 4)
          )
          received.extend(next(gen))
          finished.set()
          list(gen)  # drain the producer's terminal sentinel

        threads = [
            threading.Thread(target=produce, daemon=True),
            threading.Thread(target=consume, daemon=True),
        ]
        for t in threads:
          t.start()
        succeeded = finished.wait(2)
        for t in threads:
          t.join(2)
        self.assertEqual(errors, [])
        self.assertTrue(succeeded, f"capacity={capacity}: deadlocked")
        self.assertEqual(received, [0, 1, 2, 3])
        self.assertEqual(
            list(learner._sb_item_fifo),
            [(g, p) for g in range(4) for p in range(2)],
        )

  def test_sb_off_producer_enqueues_bare_examples(self):
    # Feature off: the queue carries bare TrainExamples exactly as upstream.
    learner = _make_learner(mgr=None, k=1)
    data = queue_lib.SimpleDataQueue(maxsize=0)
    learner._sb_reinject = lambda q: None
    learner._batch_to_train_example = lambda batch_results, mode: [
        f"te{batch_results[0].prompt_id}"
    ]

    async def groups(**unused):
      for gid in range(2):
        yield [_traj(gid, 0)]

    learner._orchestrator_producer = groups
    asyncio.run(learner._producer(None, queue.Queue(), data))
    out = []
    while (item := data.get()) is not None:
      out.append(item)
    self.assertEqual(out, ["te0", "te1"])
    self.assertEmpty(learner._sb_item_fifo)


class SubBatchGuardsTest(absltest.TestCase):

  def test_packing_raises_until_packing_support_lands(self):
    """Sequence packing + sub-batch is guarded out in this layer (the
    packing wiring is the next layer of the series); a packed config must
    fail loudly at init, not silently mis-account at the row level."""
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
    """Read from the trainer manager's RESOLVED options (a v0
    save_interval_steps maps to a FixedIntervalPolicy there), not the raw
    run options."""
    learner = _make_learner(mgr=None)
    learner.algo_config.sub_batch_checkpointing = True
    manager = learner.rl_engine.actor_trainer.checkpoint_manager
    manager._options.save_decision_policy.interval = 5
    with self.assertRaisesRegex(
        ValueError, "FixedIntervalPolicy\\(interval=1\\)"
    ):
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


class _TokenEngine:
  """Token-mode payload as TrajectoryCollectEngine emits it: the traj dict
  carries `group_id` (an env kwarg) and never `pair_index`, so an item
  identity read through TrajectoryItem.__getattr__ would half-resolve.
  """

  def __init__(self, agent, env, **kw):
    del agent, kw
    self._env = env

  async def collect(self, mode=None):
    del mode
    return {
        "conversation_tokens": np.array([1, 2, 3]),
        "conversation_masks": np.array([1, 1, 1]),
        "trajectory_reward": 1.0,
        "group_id": self._env.extra_kwargs["group_id"],
    }


def _orchestrator_items(gid, pairs):
  """One group's items, built by the REAL orchestrator episode path into a
  REAL GroupQueueManager (its default key_fn is `id`, so the prompt_id key
  is passed explicitly, as the orchestrator does).
  """

  async def go():
    orch = rollout_orchestrator.RolloutOrchestrator(
        rollout_sync_lock=SimpleNamespace(),
        engine_cls=_TokenEngine,
        engine_kwargs={},
        max_concurrency=1,
    )
    qm = group_queue_manager.GroupQueueManager(
        num_generations=pairs, key_fn=lambda it: it.prompt_id
    )
    for p in range(pairs):
      env = SimpleNamespace(
          extra_kwargs={"group_id": gid, "pair_index": p}, task={}
      )
      await orch._run_and_queue_one_episode(
          None,
          env,
          qm,
          group_key_fn=lambda i, e, t: e.extra_kwargs["group_id"],
          start_step_fn=None,
          collect_mode="Token",
      )
    return await qm.get_batch(pairs)

  return asyncio.run(go())


class CanonicalItemTest(absltest.TestCase):
  """Real-orchestrator items (prompt_id/group_index, traj_id) through the
  ledger, a snapshot, restore and re-injection: identity never depends on
  the legacy names or the traj/metadata attribute fallback.
  """

  def test_orchestrator_item_round_trips_through_ledger_and_snapshot(self):
    full_batch_size = 4
    gid = 2 * full_batch_size + 1
    items = _orchestrator_items(gid=gid, pairs=2)
    self.assertEqual([t.traj_id for t in items], ["traj_9_g0", "traj_9_g1"])
    tmp = tempfile.mkdtemp()
    try:
      mgr = _make_manager(tmp, k=1)
      learner = _make_learner(
          mgr=mgr,
          k=1,
          full_batch_size=full_batch_size,
          global_steps=2,
          train_steps=2,
          process_in_consumer=True,
      )
      learner.rl_engine.actor_trainer.iter_steps = 2
      learner._sb_register(items)
      self.assertEqual(learner._sb_active_gids, {gid})
      self.assertEqual(learner._sb_item_step(items[0]), 2)
      self.assertTrue(learner._sb_skip_group(gid))
      learner._sb_snapshot([(gid, 0), (gid, 1)], step_complete=False)
      mgr.wait()
      st = mgr.try_restore(train_steps=2, grad_accum_steps=1)
      self.assertEqual(
          [t.traj_id for t in st.active_group_trajectories],
          ["traj_9_g0", "traj_9_g1"],
      )
      self.assertEqual(st.trained_trajectory_counts, {(gid, 0): 1, (gid, 1): 1})
      self.assertEqual(st.completed_group_ids, [gid])
      # Re-injection re-feeds the restored items as one group.
      learner._sb_pending_state = st
      learner._sb_active = list(st.active_group_trajectories)
      q = queue.Queue()
      learner._sb_reinject(q)
      self.assertEqual(
          [t.traj_id for t in q.get()], ["traj_9_g0", "traj_9_g1"]
      )
      self.assertTrue(q.empty())
      mgr.close()
    finally:
      shutil.rmtree(tmp)


if __name__ == "__main__":
  absltest.main()
