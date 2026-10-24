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

"""Unit tests for Household text game environment, agent, and data generator."""

import json
import os
import tempfile
from absl.testing import absltest
from examples.household.agent import HouseholdAgent
from examples.household.data import create_dataset, generate_task
from examples.household.env import HouseholdEnv


def create_sample_task():
  return {
      "instruction": (
          "Find the silver key in the kitchen drawer, unlock the wooden chest"
          " in the bedroom, and take the golden trophy."
      ),
      "start_room": "living_room",
      "initial_inventory": [],
      "rooms": {
          "living_room": {
              "description": "a central living room",
              "exits": {"north": "kitchen", "east": "bedroom"},
              "containers": {},
              "floor_items": [],
          },
          "kitchen": {
              "description": "a kitchen with tiled floor",
              "exits": {"south": "living_room"},
              "containers": {
                  "kitchen_drawer": {
                      "is_open": False,
                      "is_locked": False,
                      "items": ["silver_key"],
                  }
              },
              "floor_items": ["apple"],
          },
          "bedroom": {
              "description": "a quiet bedroom",
              "exits": {"west": "living_room"},
              "containers": {
                  "wooden_chest": {
                      "is_open": False,
                      "is_locked": True,
                      "key_name": "silver_key",
                      "items": ["golden_trophy"],
                  }
              },
              "floor_items": [],
          },
      },
      "goal": {
          "type": "take",
          "item": "golden_trophy",
      },
  }


class HouseholdEnvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.task = create_sample_task()
    self.env = HouseholdEnv(task=self.task, max_steps=20)

  def test_initial_observation(self):
    obs, info = self.env.reset()
    self.assertIn("Goal: Find the silver key", obs)
    self.assertIn("You are in the living_room", obs)
    self.assertIn("north to kitchen", obs)
    self.assertIn("east to bedroom", obs)

  def test_navigation_and_look(self):
    self.env.reset()
    obs, reward, done, info = self.env.step("go north")
    self.assertFalse(done)
    self.assertIn("kitchen", obs)
    self.assertIn("kitchen drawer (closed)", obs)
    self.assertIn("apple", obs)

    obs, _, _, _ = self.env.step("look")
    self.assertIn("kitchen", obs)

    obs, reward, _, info = self.env.step("go east")
    self.assertFalse(info["valid"])
    self.assertIn("cannot go 'east'", obs)

  def test_container_and_locking_mechanics(self):
    self.env.reset()
    self.env.step("go east")

    obs, reward, done, info = self.env.step("open wooden chest")
    self.assertFalse(info["valid"])
    self.assertIn("is locked", obs)

    self.env.step("go west")
    self.env.step("go north")

    obs, reward, done, info = self.env.step("open kitchen_drawer")
    self.assertTrue(info["valid"])
    self.assertIn("silver key", obs)
    self.assertIn("opened_kitchen_drawer", info["achieved_milestones"])

    obs, reward, done, info = self.env.step(
        "take silver_key from kitchen_drawer"
    )
    self.assertTrue(info["valid"])
    self.assertIn("silver_key", self.env._inventory)
    self.assertIn("took_silver_key", info["achieved_milestones"])

    obs, _, _, _ = self.env.step("inventory")
    self.assertIn("silver key", obs)

    self.env.step("go south")
    self.env.step("go east")

    obs, reward, done, info = self.env.step(
        "unlock wooden chest with silver key"
    )
    self.assertTrue(info["valid"])
    self.assertIn("unlocked_wooden_chest", info["achieved_milestones"])

    obs, reward, done, info = self.env.step("open wooden chest")
    self.assertTrue(info["valid"])
    self.assertIn("golden trophy", obs)

    obs, reward, done, info = self.env.step(
        "take golden_trophy from wooden chest"
    )
    self.assertTrue(done)
    self.assertGreater(reward, 0.5)
    self.assertIn("Congratulations!", obs)

  def test_max_steps_truncation(self):
    env = HouseholdEnv(task=self.task, max_steps=3)
    env.reset()
    _, _, done1, _ = env.step("look")
    self.assertFalse(done1)
    _, _, done2, _ = env.step("look")
    self.assertFalse(done2)
    _, _, done3, _ = env.step("look")
    self.assertTrue(done3)

  def test_env_from_dataset_record(self):
    raw_task = generate_task(seed=99)
    record = {
        "env_name": "household",
        "seed": raw_task["seed"],
        "task_json": json.dumps(raw_task),
        "task": raw_task,
        "prompts": "",
    }
    env = HouseholdEnv(task=record, max_steps=15)
    obs, info = env.reset()
    self.assertIn("Goal:", obs)
    self.assertIn(raw_task["instruction"], obs)
    # Check that rooms are properly extracted (not the single fallback room!)
    self.assertLen(env._rooms, len(raw_task["rooms"]))
    self.assertEqual(env._current_room, raw_task["start_room"])

  def test_env_natural_language_variations(self):
    self.env.reset()
    # Test multi-word room navigation: go to kitchen
    obs, _, _, info = self.env.step("go to kitchen")
    self.assertTrue(info["valid"])
    self.assertEqual(self.env._current_room, "kitchen")

    # Test leading article in open command: open the kitchen drawer
    obs, _, _, info = self.env.step("open the kitchen drawer")
    self.assertTrue(info["valid"])
    self.assertIn("silver key", obs)

    # Test leading article in take command: take the silver key
    obs, _, _, info = self.env.step("take the silver key from the kitchen drawer")
    self.assertTrue(info["valid"])
    self.assertIn("silver_key", self.env._inventory)

    # Test examine with article: examine the wooden chest
    self.env.step("go to living room")
    self.env.step("go to bedroom")
    obs, _, _, info = self.env.step("examine the wooden chest")
    self.assertTrue(info["valid"])
    self.assertIn("locked", obs)

  def test_env_put_and_open_goals(self):
    # Test 'open' goal type
    open_task = {
        "instruction": "Open the kitchen drawer.",
        "start_room": "kitchen",
        "rooms": {
            "kitchen": {
                "description": "the kitchen",
                "exits": {},
                "containers": {
                    "kitchen_drawer": {"is_open": False, "is_locked": False}
                },
                "floor_items": [],
            }
        },
        "goal": {"type": "open", "container": "kitchen_drawer"},
    }
    env_open = HouseholdEnv(task=open_task, max_steps=5)
    env_open.reset()
    obs, reward, done, _ = env_open.step("open kitchen drawer")
    self.assertTrue(done)
    self.assertGreater(reward, 0.5)

    # Test 'put' goal type
    put_task = {
        "instruction": "Put the apple into the refrigerator.",
        "start_room": "kitchen",
        "initial_inventory": ["apple"],
        "rooms": {
            "kitchen": {
                "description": "the kitchen",
                "exits": {},
                "containers": {
                    "refrigerator": {"is_open": True, "is_locked": False, "items": []}
                },
                "floor_items": [],
            }
        },
        "goal": {"type": "put", "item": "apple", "container": "refrigerator"},
    }
    env_put = HouseholdEnv(task=put_task, max_steps=5)
    env_put.reset()
    obs, reward, done, _ = env_put.step("put apple in refrigerator")
    self.assertTrue(done)
    self.assertGreater(reward, 0.5)


class HouseholdAgentTest(absltest.TestCase):

  def test_agent_action_extraction(self):
    agent = HouseholdAgent()
    obs = "You are in the living room. Exits: north to kitchen."
    agent.update_from_env(obs, reward=0.0, done=False)

    model_output = (
        "I need to go north to the kitchen to find the key.\nAction: ```go"
        " north```"
    )
    action = agent.update_from_model(model_output)

    self.assertEqual(action.action, "go north")
    self.assertLen(agent.trajectory.steps, 1)
    step = agent.trajectory.steps[0]
    self.assertEqual(step.action.action, "go north")
    self.assertIn("I need to go north", step.thought)

  def test_tagged_code_block_action(self):
    agent = HouseholdAgent()
    agent.update_from_env("Room description", reward=0.0, done=False)

    # Test tagged ```action\n...```
    action = agent.update_from_model(
        "I will check the chest.\n```action\nopen wooden chest```"
    )
    self.assertEqual(action.action, "open wooden chest")

    # Test tagged ```text\n...```
    action2 = agent.update_from_model(
        "I will take the key.\n```text\ntake silver key```"
    )
    self.assertEqual(action2.action, "take silver key")

  def test_stagnation_warning(self):
    agent = HouseholdAgent()
    obs = "You are in the living room."
    agent.update_from_env(obs, reward=0.0, done=False)
    self.assertNotIn("Warning: Your position or state did not change", agent.last_observation)

    # Same observation fed again (e.g. invalid action resulted in same state)
    agent.update_from_env(obs, reward=-0.05, done=False)
    self.assertIn("Warning: Your position or state did not change", agent.last_observation)

    # Feed same observation third time: warning should persist without flickering off
    agent.update_from_env(obs, reward=-0.05, done=False)
    self.assertIn("Warning: Your position or state did not change", agent.last_observation)


class HouseholdDataTest(absltest.TestCase):

  def test_generate_task_structure(self):
    task = generate_task(seed=123)
    self.assertEqual(task["env_name"], "household")
    self.assertIn(task["start_room"], task["rooms"])
    self.assertNotEmpty(task["instruction"])
    self.assertEqual(task["goal"]["type"], "take")

    # Determinism
    task2 = generate_task(seed=123)
    self.assertEqual(task["instruction"], task2["instruction"])
    self.assertEqual(task["start_room"], task2["start_room"])

  def test_create_dataset_grain_iteration(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      dataset = create_dataset(
          split="train", data_dir=tmp_dir, train_size=5, seed=42
      )
      self.assertEqual(len(dataset), 5)
      first_item = dataset[0]
      self.assertEqual(first_item["env_name"], "household")
      self.assertIn("task_json", first_item)
      task = json.loads(first_item["task_json"])
      self.assertIn("rooms", task)
      self.assertIn("instruction", task)

  def test_dataset_batching(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      dataset = create_dataset(
          split="train", data_dir=tmp_dir, train_size=4, seed=42
      )
      batched = dataset.to_iter_dataset().batch(batch_size=2)
      batch = next(iter(batched))
      self.assertEqual(len(batch["env_name"]), 2)
      self.assertEqual(len(batch["task_json"]), 2)


if __name__ == "__main__":
  absltest.main()
