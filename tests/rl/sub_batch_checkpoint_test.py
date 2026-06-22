# Copyright 2025 Google LLC
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

"""Tests for sub_batch_checkpoint."""

import os
import shutil
import tempfile

import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from orbax.checkpoint import v1 as ocp
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types
from tunix.sft import checkpoint_options


def _make_trajectory_item(
    group_id: int,
    pair_index: int,
    reward: float = 1.0,
) -> agent_types.TrajectoryItem:
  """Creates a dummy `TrajectoryItem` for testing."""
  step = agent_types.Step(
      chat_completions=[{"role": "user", "content": "test"}],
      thought="thinking",
      model_response="response",
      reward=reward,
      done=True,
      assistant_tokens=np.array([1, 2, 3]),
      assistant_masks=np.array([1, 1, 1]),
      logprobs=np.array([0.1, 0.2, 0.3]),
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
      metadata={"test_key": "test_value"}
  )


class SubBatchCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.checkpoint_manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        root_directory=self.test_dir,
        options=checkpoint_options.TunixCheckpointingOptions(
            save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
                interval=1
            ),
            preservation_policy=sub_batch_checkpoint.GlobalStepPreservationPolicy(
                latest_n=3
            ),
        ),
    )

  def tearDown(self):
    self.checkpoint_manager.close()
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_trajectory_item_serialization_roundtrip(self):
    """Tests serialization and deserialization of `TrajectoryItem`."""
    item = _make_trajectory_item(group_id=1, pair_index=2)
    serializable_item = sub_batch_checkpoint._trajectory_item_to_serializable(
        item
    )
    restored_item = sub_batch_checkpoint._trajectory_item_from_serializable(
        serializable_item
    )

    with self.subTest(name="TopLevelFields"):
      self.assertEqual(item.group_id, restored_item.group_id)
      self.assertEqual(item.pair_index, restored_item.pair_index)
      self.assertEqual(item.start_step, restored_item.start_step)
      self.assertEqual(item.metadata, restored_item.metadata)
    with self.subTest(name="TrajectoryFields"):
      self.assertEqual(item.traj.status, restored_item.traj.status)
      self.assertEqual(item.traj.task, restored_item.traj.task)
      self.assertEqual(item.traj.reward, restored_item.traj.reward)
      self.assertEqual(item.traj.env_time, restored_item.traj.env_time)
      self.assertEqual(item.traj.reward_time, restored_item.traj.reward_time)
    with self.subTest(name="TrajectorySteps"):
      self.assertLen(item.traj.steps, len(restored_item.traj.steps))
      np.testing.assert_equal(
          item.traj.steps[0].assistant_tokens,
          restored_item.traj.steps[0].assistant_tokens,
      )
      np.testing.assert_equal(
          item.traj.steps[0].assistant_masks,
          restored_item.traj.steps[0].assistant_masks,
      )
      np.testing.assert_equal(
          item.traj.steps[0].logprobs,
          restored_item.traj.steps[0].logprobs,
      )

  def test_save_and_restore(self):
    """Tests saving and restoring sub-batch checkpoint."""
    dummy_item = _make_trajectory_item(group_id=10, pair_index=0)
    dummy_state = {
        "acc_grads": np.array([1.0, 2.0]),
        "mini_step": np.array([1]),
    }
    dummy_token_count = np.array([100.0])

    self.checkpoint_manager.save(
        global_step=1,
        grad_accum_steps=2,
        completed_group_ids=[10, 20],
        trained_trajectory_counts={(10, 0): 1},
        active_group_trajectories=[dummy_item],
        training_state=dummy_state,
        valid_token_count=dummy_token_count,
    )

    state = self.checkpoint_manager.try_restore(
        global_step=1,
        target_training_state=dummy_state,
        target_valid_token_count=dummy_token_count,
    )
    self.assertIsNotNone(state)
    assert state is not None
    self.assertEqual(state.global_step, 1)
    self.assertEqual(state.grad_accum_steps, 2)
    self.assertEqual(state.completed_group_ids, [10, 20])
    self.assertEqual(state.trained_trajectory_counts, {(10, 0): 1})
    self.assertLen(state.active_group_trajectories, 1)
    self.assertEqual(state.active_group_trajectories[0].group_id, 10)
    np.testing.assert_equal(
        state.training_state["acc_grads"], np.array([1.0, 2.0])
    )
    np.testing.assert_equal(state.valid_token_count, np.array(100.0))

  def test_global_step_preservation_policy(self):
    """Tests automatic deletion of old global steps via GlobalStepPreservationPolicy."""
    dummy_item = _make_trajectory_item(group_id=10, pair_index=0)
    dummy_state = {
        "acc_grads": np.array([1.0, 2.0]),
        "mini_step": np.array([1]),
    }

    # Save step 1
    self.checkpoint_manager.save(
        global_step=1,
        grad_accum_steps=2,
        completed_group_ids=[10],
        trained_trajectory_counts={},
        active_group_trajectories=[dummy_item],
        training_state=dummy_state,
    )
    self.checkpoint_manager._checkpointer.wait()

    # Save step 2 (this runs retention cleanup and deletes step 1)
    self.checkpoint_manager.save(
        global_step=2,
        grad_accum_steps=3,
        completed_group_ids=[20],
        trained_trajectory_counts={},
        active_group_trajectories=[dummy_item],
        training_state=dummy_state,
    )
    self.checkpoint_manager._checkpointer.wait()

    restored = self.checkpoint_manager.try_restore(
        global_step=2, target_training_state=dummy_state
    )
    self.assertIsNotNone(restored)
    assert restored is not None
    self.assertEqual(restored.global_step, 2)

    self.checkpoint_manager.close()

    step1_dir = self.checkpoint_manager._checkpointer.directory / "1000002"
    self.assertFalse(os.path.exists(step1_dir))

  def test_restores_only_diff_over_full_step_checkpoint(self):
    """Tests optax MultiSteps diff extraction and injection."""
    opt_state = optax.MultiStepsState(
        mini_step=jnp.array(1),
        gradient_step=jnp.array(10),
        inner_opt_state={"mu": jnp.array([0.5])},
        acc_grads={"w": jnp.array([0.1])},
        skip_state=jnp.array(False),
    )

    diff = sub_batch_checkpoint._extract_multisteps_diff(opt_state)
    self.assertEqual(
        diff,
        {"acc_grads": {"w": jnp.array([0.1])}, "mini_step": jnp.array([1])},
    )

    fresh_opt_state = optax.MultiStepsState(
        mini_step=jnp.array(0),
        gradient_step=jnp.array(10),
        inner_opt_state={"mu": jnp.array([0.5])},
        acc_grads={"w": jnp.array([0.0])},
        skip_state=jnp.array(False),
    )

    restored_opt_state = sub_batch_checkpoint._inject_multisteps_diff(
        fresh_opt_state, diff, 1
    )
    self.assertEqual(restored_opt_state.mini_step, jnp.array(1))
    self.assertEqual(restored_opt_state.acc_grads, {"w": jnp.array([0.1])})
    self.assertEqual(
        restored_opt_state.inner_opt_state, {"mu": jnp.array([0.5])}
    )


if __name__ == "__main__":
  absltest.main()
