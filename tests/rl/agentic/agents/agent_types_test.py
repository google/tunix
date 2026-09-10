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

"""Tests for agent_types, especially TrajectoryItem serialization and attributes."""

import pickle
from absl.testing import absltest
import cloudpickle
import numpy as np
from tunix.rl.agentic.agents import agent_types


class TrajectoryItemTest(absltest.TestCase):

  def test_pickle_roundtrip(self):
    traj = {
        "prompt_tokens": np.array([1, 2, 3], dtype=np.int32),
        "conversation_tokens": np.array([4, 5], dtype=np.int32),
        "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
        "reward": 1.5,
        "status": agent_types.TrajectoryStatus.SUCCEEDED,
        "policy_version": 2,
    }
    item = agent_types.TrajectoryItem(
        prompt_id="prompt_42",
        group_index=1,
        start_step=0,
        traj=traj,
        metadata={"user_tag": "test_tag"},
    )

    serialized = pickle.dumps(item)
    restored = pickle.loads(serialized)

    self.assertEqual(restored.prompt_id, "prompt_42")
    self.assertEqual(restored.group_index, 1)
    self.assertEqual(restored.group_id, "prompt_42")
    self.assertEqual(restored.pair_index, 1)
    self.assertEqual(restored.start_step, 0)
    self.assertEqual(restored.reward, 1.5)
    self.assertEqual(restored.status, agent_types.TrajectoryStatus.SUCCEEDED)
    self.assertEqual(restored.policy_version, 2)
    self.assertEqual(restored.user_tag, "test_tag")
    self.assertEqual(restored.metadata, {"user_tag": "test_tag"})
    np.testing.assert_array_equal(restored.prompt_tokens, item.prompt_tokens)
    np.testing.assert_array_equal(
        restored.conversation_tokens, item.conversation_tokens
    )
    np.testing.assert_array_equal(
        restored.conversation_masks, item.conversation_masks
    )

  def test_cloudpickle_roundtrip(self):
    """Ensures cloudpickle does not cause infinite recursion during unpickling."""
    traj = {
        "prompt_tokens": np.array([10, 20], dtype=np.int32),
        "reward": 2.0,
        "policy_version": 3,
    }
    item = agent_types.TrajectoryItem(
        prompt_id="cloud_prompt",
        group_index=2,
        traj=traj,
        metadata={"meta_key": "meta_val"},
    )

    serialized = cloudpickle.dumps(item)
    restored = cloudpickle.loads(serialized)

    self.assertIsInstance(restored, agent_types.TrajectoryItem)
    self.assertEqual(restored.prompt_id, "cloud_prompt")
    self.assertEqual(restored.group_index, 2)
    self.assertEqual(restored.reward, 2.0)
    self.assertEqual(restored.policy_version, 3)
    self.assertEqual(restored.meta_key, "meta_val")
    np.testing.assert_array_equal(restored.prompt_tokens, [10, 20])

  def test_cloudpickle_roundtrip_empty_and_none_traj(self):
    item_none = agent_types.TrajectoryItem(prompt_id="empty_1", traj=None)
    restored_none = cloudpickle.loads(cloudpickle.dumps(item_none))
    self.assertEqual(restored_none.prompt_id, "empty_1")
    self.assertIsNone(restored_none.traj)

    item_empty = agent_types.TrajectoryItem(prompt_id="empty_2", traj={})
    restored_empty = cloudpickle.loads(cloudpickle.dumps(item_empty))
    self.assertEqual(restored_empty.prompt_id, "empty_2")
    self.assertEqual(restored_empty.traj, {})

  def test_dunder_attribute_lookup_raises_attribute_error(self):
    """Verifies that undefined dunder attributes never fall through to __getattr__."""
    item = agent_types.TrajectoryItem(
        prompt_id="p1",
        traj={"__custom__": "bad", "__deepcopy__": "bad"},
        metadata={"__meta_custom__": "bad"},
    )
    # Defined state methods exist on the object
    self.assertTrue(callable(item.__getstate__))
    self.assertTrue(callable(item.__setstate__))

    # Undefined dunders raise AttributeError even if present in traj or metadata
    with self.assertRaises(AttributeError):
      _ = item.__custom__
    with self.assertRaises(AttributeError):
      _ = item.__deepcopy__
    with self.assertRaises(AttributeError):
      _ = item.__meta_custom__

  def test_dynamic_getattr_precedence(self):
    item = agent_types.TrajectoryItem(
        prompt_id="p1",
        group_index=0,
        traj={"shared": "from_traj", "only_traj": 100},
        metadata={"shared": "from_meta", "only_meta": 200},
    )
    # Traj takes precedence over metadata
    self.assertEqual(item.shared, "from_traj")
    self.assertEqual(item.only_traj, 100)
    self.assertEqual(item.only_meta, 200)

    with self.assertRaises(AttributeError):
      _ = item.non_existent_key

  def test_identifier_aliases(self):
    # Initializing with standard names populates legacy aliases
    item1 = agent_types.TrajectoryItem(
        prompt_id="prompt_a", group_index=3, traj={}
    )
    self.assertEqual(item1.prompt_id, "prompt_a")
    self.assertEqual(item1.group_id, "prompt_a")
    self.assertEqual(item1.group_index, 3)
    self.assertEqual(item1.pair_index, 3)

    # Initializing with legacy names populates standard names
    item2 = agent_types.TrajectoryItem(
        group_id="prompt_b", pair_index=5, traj={}
    )
    self.assertEqual(item2.prompt_id, "prompt_b")
    self.assertEqual(item2.group_id, "prompt_b")
    self.assertEqual(item2.group_index, 5)
    self.assertEqual(item2.pair_index, 5)

  def test_traj_id_property(self):
    item = agent_types.TrajectoryItem(
        prompt_id="task_99", group_index=4, traj={}
    )
    self.assertEqual(item.traj_id, "traj_task_99_g4")

  def test_to_dict_and_from_dict(self):
    traj = {
        "reward": 3.0,
        "status": agent_types.TrajectoryStatus.SUCCEEDED,
        "prompt_tokens": np.array([1, 2], dtype=np.int32),
    }
    item = agent_types.TrajectoryItem(
        prompt_id="p_dict",
        group_index=1,
        start_step=0,
        traj=traj,
        metadata={"extra": "data"},
    )
    d = item.to_dict()
    self.assertEqual(d["prompt_id"], "p_dict")
    self.assertEqual(d["group_index"], 1)
    self.assertEqual(d["start_step"], 0)
    self.assertEqual(d["traj"]["reward"], 3.0)
    self.assertEqual(d["metadata"], {"extra": "data"})

    restored = agent_types.TrajectoryItem.from_dict(d)
    self.assertEqual(restored.prompt_id, "p_dict")
    self.assertEqual(restored.group_index, 1)
    self.assertEqual(restored.reward, 3.0)
    self.assertEqual(restored.metadata, {"extra": "data"})
    np.testing.assert_array_equal(restored.prompt_tokens, [1, 2])


if __name__ == "__main__":
  absltest.main()
