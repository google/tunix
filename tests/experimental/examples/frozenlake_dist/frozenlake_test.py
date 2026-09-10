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

"""Tests for the self-contained distributed FrozenLake recipe."""

from pathlib import Path

from absl.testing import absltest
from examples.frozenlake import agent as frozenlake_agent
from examples.frozenlake import env as frozenlake_env
from tunix.experimental.examples.common import models
from tunix.experimental.examples.frozenlake_dist import frozenlake
from tunix.experimental.examples.frozenlake_dist import run_frozenlake_dist
from tunix.experimental.rl.agentic import registry


class FrozenLakeDistTest(absltest.TestCase):

  def test_registered_components(self):
    self.assertIs(
        registry.ENV_REGISTRY.get(frozenlake.FROZENLAKE_ENV_NAME),
        frozenlake.FrozenLakeEnv,
    )
    self.assertIs(
        registry.AGENT_REGISTRY.get(frozenlake.FROZENLAKE_AGENT_NAME),
        frozenlake.FrozenLakeAgent,
    )
    self.assertIs(frozenlake.FrozenLakeEnv, frozenlake_env.FrozenLakeEnv)
    self.assertIs(frozenlake.FrozenLakeAgent, frozenlake_agent.FrozenLakeAgent)

  def test_package_reuses_recipe_agent_and_env(self):
    package_dir = Path(frozenlake.__file__).parent
    source = Path(frozenlake.__file__).read_text(encoding="utf-8")
    self.assertIn("from examples.frozenlake import agent", source)
    self.assertIn("from examples.frozenlake import env", source)
    self.assertNotIn("class FrozenLakeAgent", source)
    self.assertNotIn("class FrozenLakeEnv", source)

  def test_dataset_is_deterministic_and_uses_reference_ranges(self):
    first = frozenlake.create_dataset(size=5, seed=123)
    second = frozenlake.create_dataset(size=5, seed=123)

    self.assertEqual(first, second)
    self.assertLen(first, 5)
    for entry in first:
      self.assertBetween(entry["seed"], 0, 99999)
      self.assertBetween(entry["size"], 2, 9)
      self.assertGreaterEqual(entry["p"], 0.6)
      self.assertLess(entry["p"], 0.85)

  def test_generated_map_is_reproducible_and_reachable(self):
    previous_max_steps = frozenlake_env.MAX_STEPS
    frozenlake_env.MAX_STEPS = 8
    self.addCleanup(setattr, frozenlake_env, "MAX_STEPS", previous_max_steps)
    first_map, first_goal = frozenlake_env.generate_random_map(
        size=5, p=0.8, seed=7
    )
    second_map, second_goal = frozenlake_env.generate_random_map(
        size=5, p=0.8, seed=7
    )

    self.assertEqual(first_map, second_map)
    self.assertEqual(first_goal, second_goal)
    self.assertTrue(
        frozenlake_env.is_valid([list(row) for row in first_map], max_size=5)
    )

  def test_environment_reaches_goal(self):
    env = frozenlake.FrozenLakeEnv(
        entry={"seed": 42, "size": 2, "p": 0.8},
        desc=["SF", "FG"],
        is_slippery=False,
        max_steps=2,
    )
    observation, _ = env.reset()
    self.assertIn("P", observation)

    _, reward, done, _ = env.step("3")
    self.assertEqual(reward, 0.0)
    self.assertFalse(done)
    observation, reward, done, _ = env.step("2")
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertIn("√", observation)

  def test_agent_parses_last_fenced_direction(self):
    agent = frozenlake.FrozenLakeAgent(use_multistep_prompt=False)
    agent.update_from_env("P _\nO G", 0.0, False)

    action = agent.update_from_model(
        "I first considered ```Up```, but the answer is ```Right```."
    )

    self.assertEqual(action.action, "3")
    self.assertEqual(agent.trajectory.steps[-1].action, "3")

  def test_prompt_item_carries_distributed_env_and_agent_config(self):
    item = frozenlake.build_prompt_item(
        entry={"seed": 1, "size": 4, "p": 0.75},
        prompt_idx=2,
        max_turns=8,
        max_response_length=2048,
        episode_timeout_secs=600,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        is_slippery=False,
        use_multistep_prompt=True,
    )

    self.assertEqual(item["prompt_id"], "frozenlake_2")
    self.assertEqual(item["max_turns"], 8)
    self.assertEqual(item["generation_kwargs"]["temperature"], 0.7)
    self.assertEqual(item["metadata"]["env_config"]["max_steps"], 8)
    self.assertEqual(
        item["metadata"]["agent_config"], {"use_multistep_prompt": True}
    )

  def test_prompt_iterator_emits_one_full_batch_per_step(self):
    items = list(
        frozenlake.iter_prompt_items(
            dataset=[{"seed": 1, "size": 2, "p": 0.8}],
            max_steps=2,
            batch_size=3,
            max_turns=4,
            max_response_length=64,
            episode_timeout_secs=30,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            is_slippery=False,
            use_multistep_prompt=True,
        )
    )
    self.assertLen(items, 6)
    self.assertLen({item["prompt_id"] for item in items}, 6)

  def test_recipe_defaults_match_reference(self):
    args = run_frozenlake_dist._parse_args([])
    self.assertEqual(args.model_id, "Qwen/Qwen3-8B")
    self.assertEqual(args.batch_size, 64)
    self.assertEqual(args.mini_batch_size, 64)
    self.assertEqual(args.num_generations, 8)
    self.assertEqual(args.max_turns, 8)
    self.assertEqual(args.epsilon, 0.003)
    self.assertEqual(args.epsilon_high, 0.005)
    self.assertEqual(args.loss_algo, "gspo-token")
    self.assertEqual(args.advantage_estimator, "rloo")

  def test_qwen3_8b_supported_by_distributed_workers(self):
    config = models._qwen3_config("Qwen3-8B")
    self.assertEqual(config.embed_dim, 4096)
    self.assertEqual(config.num_layers, 36)

  def test_launcher_uses_frozenlake_registry(self):
    launcher = (Path(frozenlake.__file__).parent / "launcher.sh").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "--registry_module=tunix.experimental.examples.frozenlake_dist.frozenlake",
        launcher,
    )
    self.assertIn("--env_name=frozenlake_env", launcher)
    self.assertIn("--agent_name=frozenlake_agent", launcher)
    self.assertEqual(launcher.count('--mini_batch_size="$MINI_BATCH_SIZE"'), 2)
    self.assertEqual(launcher.count('--num_generations="$NUM_GENERATIONS"'), 2)


if __name__ == "__main__":
  absltest.main()
