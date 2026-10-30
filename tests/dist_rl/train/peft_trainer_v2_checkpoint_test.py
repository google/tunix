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

"""Checkpoint finalization at externally committed training boundaries."""

import copy
import tempfile
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.sft import checkpoint_options


class _TinyModel(nnx.Module):

  def __init__(self):
    self.w = nnx.Param(jnp.array([1.0, -1.0]))


def _loss(model, target):
  return jnp.mean(jnp.square(model.w[...] - target))


class ExternalCheckpointTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = directory.name

  def _trainer(self):
    config = peft_trainer_v2.TrainingConfig(
        eval_every_n_steps=1000,
        checkpoint_root_directory=self.root,
        checkpointing_options=checkpoint_options.create_checkpointing_options(
            save_decision_policy=(
                ocp.training.save_decision_policies.FixedIntervalPolicy(25)
            ),
            enable_async_checkpointing=False,
        ),
        resume_from_checkpoint_on_init=False,
    )
    trainer = peft_trainer_v2.PeftTrainer(
        _TinyModel(), optax.adam(0.1), config, metrics_logger=mock.Mock()
    )
    return trainer.with_loss_fn(_loss)

  def _worker(self):
    trainer = self._trainer()
    worker = trainer_worker.TrainerWorker(lambda: trainer)
    worker.initialize()
    self.assertTrue(trainer.is_managed_externally)
    return worker, trainer

  def _update(self, trainer):
    trainer.fwd_bwd({"target": jnp.zeros(2)}, skip_jit=True)
    return trainer.update(skip_jit=True)

  def _snapshot(self, trainer):
    return jax.tree.map(
        lambda x: np.asarray(x).copy(),
        (
            nnx.state(trainer.model),
            nnx.state(trainer.optimizer, nnx.optimizer.OptState),
        ),
    )

  def _assert_state_equal(self, trainer, expected):
    actual = self._snapshot(trainer)
    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
      np.testing.assert_array_equal(left, right)

  def test_close_force_saves_skipped_boundary_with_immutable_metadata(self):
    worker, trainer = self._worker()
    self._update(trainer)
    trainer.save_checkpoint(
        {"step": 1, "cursor": {"next_group": 4}}, force=True
    )
    self._update(trainer)
    metadata = {
        "step": 2,
        "policy_version": 2,
        "identity": {"student": "fixed-student", "teacher": "fixed-teacher"},
        "cursor": {"next_group": 8, "pending": ["group-8"]},
    }
    committed_metadata = copy.deepcopy(metadata)
    expected = self._snapshot(trainer)
    trainer.save_checkpoint(metadata)
    self.assertEqual(trainer.checkpoint_manager.latest_step(), 1)

    metadata["cursor"]["next_group"] = 999
    metadata["cursor"]["pending"].append("uncommitted-group")
    metadata["identity"]["teacher"] = "different-teacher"
    worker.stop()

    resumed_worker, resumed = self._worker()
    self.assertEqual(resumed.restore_checkpoint(), committed_metadata)
    self.assertEqual(resumed.train_steps, 2)
    self._assert_state_equal(resumed, expected)
    resumed_worker.stop()

  def test_partial_batch_shutdown_keeps_previous_persisted_boundary(self):
    worker, trainer = self._worker()
    self._update(trainer)
    metadata = {"step": 1, "identity": "fixed", "cursor": {"next_group": 4}}
    expected = self._snapshot(trainer)
    trainer.save_checkpoint(metadata, force=True)
    self._update(trainer)
    trainer.save_checkpoint(
        {"step": 2, "identity": "fixed", "cursor": {"next_group": 8}}
    )
    self.assertEqual(trainer.checkpoint_manager.latest_step(), 1)
    self._update(trainer)
    self.assertFalse(
        np.array_equal(trainer.model.w[...], jax.tree.leaves(expected[0])[0])
    )
    worker.stop()

    resumed_worker, resumed = self._worker()
    self.assertEqual(resumed.restore_checkpoint(), metadata)
    self.assertEqual(resumed.train_steps, 1)
    self._assert_state_equal(resumed, expected)
    resumed_worker.stop()

  def test_shutdown_without_external_commit_does_not_create_checkpoint(self):
    worker, trainer = self._worker()
    self._update(trainer)
    worker.stop()

    resumed_worker, resumed = self._worker()
    self.assertIsNone(resumed.checkpoint_manager.latest_step())
    self.assertEqual(resumed.restore_checkpoint(), {"step": 0})
    resumed_worker.stop()

  def test_builtin_train_still_saves_final_weights(self):
    trainer = self._trainer()
    self.assertFalse(trainer.is_managed_externally)
    trainer.train([{"target": jnp.zeros(2)}] * 2, skip_jit=True)
    expected = self._snapshot(trainer)

    resumed_worker, resumed = self._worker()
    metadata = resumed.restore_checkpoint()
    self.assertEqual(metadata["step"], 2)
    self.assertEqual(resumed.train_steps, 2)
    self._assert_state_equal(resumed, expected)
    resumed_worker.stop()

  def test_mismatched_metadata_step_cannot_be_reused_at_shutdown(self):
    worker, trainer = self._worker()
    self._update(trainer)
    trainer.save_checkpoint({"step": 999, "cursor": "different-step"}, step=1)
    self.assertIsNone(trainer.checkpoint_manager.latest_step())
    worker.stop()

    resumed_worker, resumed = self._worker()
    self.assertIsNone(resumed.checkpoint_manager.latest_step())
    resumed_worker.stop()

  def test_restore_invalidates_previously_cached_commit(self):
    worker, trainer = self._worker()
    self._update(trainer)
    trainer.save_checkpoint({"step": 1, "cursor": "persisted"}, force=True)
    self._update(trainer)
    trainer.save_checkpoint({"step": 2, "cursor": "discarded-future"})
    self.assertEqual(trainer.restore_checkpoint(step=1)["step"], 1)
    self._update(trainer)
    worker.stop()

    resumed_worker, resumed = self._worker()
    self.assertEqual(
        resumed.restore_checkpoint(), {"step": 1, "cursor": "persisted"}
    )
    resumed_worker.stop()

  def test_worker_accepts_trainer_without_management_flag(self):
    trainer = types.SimpleNamespace(close=mock.Mock())
    worker = trainer_worker.TrainerWorker(lambda: trainer)
    worker.initialize()
    self.assertFalse(hasattr(trainer, "is_managed_externally"))
    worker.stop()
    trainer.close.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
