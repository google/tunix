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

"""Behavior-policy pin across a sub-batch mid-step resume (P1.5), on the
REAL RLEngine (CPU, ToyTransformer, vanilla rollout, real trainer
CheckpointManager) and the learner's real `_sb_resync_rollout_weights`.

A global step starts at trainer train_steps 0 (anchor = W0). Two applies
happen inside the step (W0 -> W2), then a crash. The restart restores W2
into the trainer, but the step's old logps, TIS weights and remaining
generations must come from W0: the resume loads the step-0 checkpoint into
a fresh host tree and pins it as the anchor and rollout policy, leaving
the live model at W2 and global_steps untouched.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

from absl.testing import absltest
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.generate import tokenizer_adapter
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.sft import checkpoint_options
from tunix.tests import test_common

try:
  from tunix.rl.agentic import agentic_sub_batch_resume_test as _rt
except ImportError:
  import agentic_sub_batch_resume_test as _rt

_make_learner = _rt._make_learner

PAD, EOS = 0, 2
PROMPTS = np.array([[0, 0, 3, 4, 5, 6, 7, 8]] * 2, dtype=jnp.int32)
COMPLETIONS = np.array([[9, 10, 11, 2], [12, 13, 2, 0]], dtype=jnp.int32)


def _ckpt_options(n):
  return checkpoint_options.TunixCheckpointingOptions(
      save_decision_policy=(
          ocp.training.save_decision_policies.FixedIntervalPolicy(1)
      ),
      preservation_policy=ocp.training.preservation_policies.LatestN(n=n),
      enable_async_checkpointing=False,
  )


def _make_engine(root, model, ref_model, *, keep=10, temperature=1.0):
  """As agentic_grpo_learner_test builds it, plus a per-apply trainer
  checkpoint stream under `root` (the trainer restores its latest step in
  __init__, exactly as a relaunch does)."""
  tokenizer = tokenizer_adapter.TokenizerAdapter(test_common.MockVocab())
  mesh = pxla.thread_resources.env.physical_mesh
  cfg = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.REFERENCE: mesh,
          rl_engine_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-1),
          eval_every_n_steps=100,
          max_steps=10,
          mini_batch_size=1,
          train_micro_batch_size=1,
          checkpoint_root_directory=root,
          checkpointing_options=_ckpt_options(keep),
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_prompt_length=8,
          max_tokens_to_generate=4,
          return_logprobs=True,
          kv_cache_size=64,
          temperature=temperature,
      ),
  )
  return rl_engine_lib.RLEngine(
      actor=model, reference=ref_model, tokenizer=tokenizer, cluster_config=cfg
  )


def _anchor_logps(engine):
  """Old logps: computed from the engine's anchor policy state."""
  return np.asarray(
      engine.get_actor_per_token_logps(
          prompt_tokens=PROMPTS,
          completion_tokens=COMPLETIONS,
          pad_id=PAD,
          eos_id=EOS,
      )
  )


def _params(model):
  return jax.tree.map(np.asarray, nnx.state(model, nnx.Param))


def _apply(trainer, scale, step, global_step):
  """Stands in for one optimizer apply plus its per-apply checkpoint."""
  st = nnx.state(trainer.model, nnx.Param)
  nnx.update(trainer.model, jax.tree.map(lambda x: x + scale, st))
  trainer.checkpoint_manager.save(
      step,
      trainer.model,
      trainer.optimizer,
      force=True,
      custom_metadata={"global_step": global_step, "role": "actor"},
  )
  trainer.checkpoint_manager.wait()


def _assert_trees(a, b, *, equal):
  if jax.tree.structure(a) != jax.tree.structure(b):
    raise AssertionError("parameter tree structures differ")
  for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
    if equal:
      np.testing.assert_allclose(x, y)
    else:
      if np.allclose(x, y):
        raise AssertionError("trees unexpectedly equal")


class SubBatchAnchorPinTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.config = test_common.ModelConfig(
        vocab_size=test_common.MockVocab().GetPieceSize()
    )

  def tearDown(self):
    shutil.rmtree(self.tmp, ignore_errors=True)
    super().tearDown()

  def _model(self):
    return test_common.ToyTransformer(config=self.config, rngs=nnx.Rngs(0))

  def _grpo_temperature_restart(self, resumed_temperature):
    root = os.path.join(self.tmp, "temperature")
    engine = _make_engine(root, self._model(), self._model(), temperature=0.7)

    def config():
      return agentic_grpo_learner.GRPOConfig(
          sub_batch_checkpointing=True, max_response_length=4,
          num_generations=2, num_iterations=1,
      )

    learner = agentic_grpo_learner.GRPOLearner(engine, config())
    try:
      self.assertEqual(learner.algo_config.temperature, 0.7)
      # A completed apply uses the resolved geometry from the live learner.
      learner._sb_mgr.save(
          1, 0, iter_steps=1, global_step=0, grad_accum_steps=1,
          step_complete=True, completed_group_ids=[],
          trained_trajectory_counts={}, active_group_trajectories=[],
          training_state=None, anchor_step=0, geometry=learner._sb_geometry(),
      )
      learner._sb_mgr.wait()
      _apply(engine.actor_trainer, 0.001, 1, 1)
    finally:
      learner.close()
      engine.close()
    engine = _make_engine(
        root, self._model(), self._model(), temperature=resumed_temperature
    )
    resumed = None
    try:
      cfg = config()
      self.assertIsNone(cfg.temperature)
      resumed = agentic_grpo_learner.GRPOLearner(engine, cfg)
      self.assertEqual(resumed.algo_config.temperature, resumed_temperature)
      self.assertEqual(engine.actor_trainer.iter_steps, 1)
    finally:
      if resumed is not None:
        resumed.close()
      engine.close()

  def test_grpo_init_resumes_with_same_resolved_temperature(self):
    self._grpo_temperature_restart(0.7)

  def test_grpo_init_rejects_changed_resolved_temperature(self):
    with self.assertRaisesRegex(
        sub_batch_checkpoint.SubBatchGeometryError, "temperature"
    ):
      self._grpo_temperature_restart(0.3)

  def _crashed_step(self, *, keep=10, pin=False):
    """Step start at W0 (checkpoint 0), two applies (checkpoints 1, 2).
    `pin` pins checkpoint 0 as the learner does for the running step.
    Returns (W0 params, anchor logps under W0, W2 params)."""
    root = os.path.join(self.tmp, "run")
    eng = _make_engine(root, self._model(), self._model(), keep=keep)
    trainer = eng.actor_trainer
    trainer.checkpoint_manager.save(
        0, trainer.model, trainer.optimizer, force=True,
        custom_metadata={"global_step": 0, "role": "actor"},
    )
    trainer.checkpoint_manager.wait()
    if pin:
      trainer.checkpoint_manager.pin_step(0)
    w0 = _params(trainer.model)
    logps_w0 = _anchor_logps(eng)
    _apply(trainer, 2e-3, 1, 1)
    _apply(trainer, 2e-3, 2, 1)
    w2 = _params(trainer.model)
    _assert_trees(w0, w2, equal=False)
    eng.close()
    return root, w0, logps_w0, w2

  def _restarted_learner(self, root, *, anchor, mid_step=True):
    """A relaunch on the same root: the trainer restores W2 (train_steps
    2); the learner is the resume-test stub around the real engine, in
    the state _init_sub_batch_checkpointing leaves after a mid-step
    restore whose step started at `anchor`."""
    eng = _make_engine(root, self._model(), self._model())
    self.assertEqual(eng.actor_trainer.train_steps, 2)
    learner = _make_learner(k=1, global_steps=3)
    learner._sb_mgr = SimpleNamespace(enabled=True)
    learner._sb_restored_trainer_state = True
    learner._sb_pending_state = object() if mid_step else None
    learner._sb_anchor_step = anchor
    learner.should_sync_weights = True
    learner.rl_engine = eng
    eng.global_steps = 3
    return learner, eng

  def test_mid_step_resume_pins_step_start_policy(self):
    root, w0, logps_w0, w2 = self._crashed_step()
    learner, eng = self._restarted_learner(root, anchor=0)
    # Before the resync the engine anchored on the restored W2.
    self.assertFalse(np.allclose(_anchor_logps(eng), logps_w0))

    learner._sb_resync_rollout_weights()

    self.assertEqual(eng.global_steps, 3)
    np.testing.assert_allclose(_anchor_logps(eng), logps_w0, atol=1e-6)
    _assert_trees(_params(eng.actor_trainer.model), w2, equal=True)
    _assert_trees(_params(eng.rollout.model()), w0, equal=True)
    self.assertTrue(
        eng._anchor_policy_state is not None
        and all(
            leaf.sharding.memory_kind == "pinned_host"
            for leaf in jax.tree.leaves(eng._anchor_policy_state)
        )
    )
    # Aliasing guard: a later in-place change of the live params (the
    # step's next apply) must not move the anchor.
    st = nnx.state(eng.actor_trainer.model, nnx.Param)
    nnx.update(eng.actor_trainer.model, jax.tree.map(lambda x: x + 5e-2, st))
    np.testing.assert_allclose(_anchor_logps(eng), logps_w0, atol=1e-6)
    eng.close()

  def test_resume_at_step_start_syncs_restored_weights(self):
    """Step started at the restored weights (anchor == train_steps, or a
    clean-boundary restart with no pending state): the plain sync path,
    W2 everywhere, global_steps untouched."""
    root, unused_w0, logps_w0, w2 = self._crashed_step()
    for mid_step, anchor in ((True, 2), (False, 0)):
      with self.subTest(mid_step=mid_step, anchor=anchor):
        learner, eng = self._restarted_learner(
            root, anchor=anchor, mid_step=mid_step
        )
        learner._sb_resync_rollout_weights()
        self.assertEqual(eng.global_steps, 3)
        self.assertFalse(np.allclose(_anchor_logps(eng), logps_w0))
        _assert_trees(_params(eng.rollout.model()), w2, equal=True)
        _assert_trees(_params(eng.actor_trainer.model), w2, equal=True)
        eng.close()

  def test_pinned_step_start_survives_retention(self):
    """LatestN(1) would keep only checkpoint 2; the pin keeps the step
    start, so the resume restores W0 as the behavior policy."""
    root, unused_w0, logps_w0, unused_w2 = self._crashed_step(keep=1, pin=True)
    learner, eng = self._restarted_learner(root, anchor=0)
    self.assertTrue(eng.actor_trainer.checkpoint_manager.has_step(0))
    learner._sb_resync_rollout_weights()
    np.testing.assert_allclose(_anchor_logps(eng), logps_w0, atol=1e-6)
    eng.close()

  def test_evicted_step_start_checkpoint_raises(self):
    """Without the pin, LatestN(2) evicted checkpoint 0 under the two
    applies: the resume cannot preserve the objective and says so instead
    of silently anchoring on W2."""
    root, *_ = self._crashed_step(keep=2)
    learner, eng = self._restarted_learner(root, anchor=0)
    self.assertFalse(eng.actor_trainer.checkpoint_manager.has_step(0))
    self.assertTrue(eng.actor_trainer.checkpoint_manager.has_step(2))
    with self.assertRaisesRegex(RuntimeError, "train_steps=0.*pin"):
      learner._sb_resync_rollout_weights()
    self.assertEqual(eng.global_steps, 3)
    eng.close()


if __name__ == "__main__":
  absltest.main()