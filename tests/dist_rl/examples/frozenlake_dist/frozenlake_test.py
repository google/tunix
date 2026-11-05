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
import unittest

from absl.testing import absltest

# `gymnasium` comes with the optional `frozenlake` extra. pytest also reports
# this module-level `unittest.SkipTest` as a skip.
try:
  import gymnasium  # pylint: disable=unused-import,g-import-not-at-top
except ImportError:
  raise unittest.SkipTest("gymnasium is not installed")  # pylint: disable=raise-missing-from

# pylint: disable=g-import-not-at-top
import datasets
from tunix.dist_rl.common import registry
from tunix.dist_rl.examples.frozenlake_dist import frozenlake
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator
from tunix.models import automodel
from examples.frozenlake import agent as frozenlake_agent
from examples.frozenlake import data as frozenlake_data
from examples.frozenlake import env as frozenlake_env

# pylint: enable=g-import-not-at-top

_FROZENLAKE_YAML = Path(frozenlake.__file__).parent / "frozenlake.yaml"


def _hp(*overrides: str) -> resolve_config.DistHyperParameters:
  """Resolves `frozenlake.yaml` on top of `base_dist_config.yaml`."""
  return resolve_config.initialize(
      [f"override_config_file={_FROZENLAKE_YAML}", *overrides]
  )


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
    self.assertIn("from examples.frozenlake import data", source)
    self.assertIn("from examples.frozenlake import env", source)
    self.assertNotIn("class FrozenLakeAgent", source)
    self.assertNotIn("class FrozenLakeEnv", source)

  def test_dataset_is_deterministic_and_uses_reference_ranges(self):
    first = frozenlake.generate_entries(size=5, seed=123)
    second = frozenlake.generate_entries(size=5, seed=123)

    self.assertEqual(first, second)
    self.assertLen(first, 5)
    for entry in first:
      self.assertBetween(entry["seed"], 0, 99999)
      self.assertBetween(entry["size"], 2, 8)
      self.assertGreaterEqual(entry["p"], 0.6)
      self.assertLess(entry["p"], 0.85)

    custom_range = frozenlake.generate_entries(
        size=10, seed=123, grid_size_range=(3, 5)
    )
    for entry in custom_range:
      self.assertBetween(entry["size"], 3, 4)

  def test_dataset_generation_and_shuffle_match_reference(self):
    seeds, sizes, probabilities = frozenlake_data.generate_dataset_parameters(
        32, random_seed=42
    )
    reference = [
        frozenlake_data.get_frozenlake_dict(
            env_seed, sizes[index], probabilities[index]
        )
        for index, env_seed in enumerate(seeds)
    ]
    expected = (
        datasets.Dataset.from_list(reference).shuffle(seed=42).to_list()[:12]
    )

    actual = frozenlake.generate_entries(
        size=32, seed=42, shuffle_seed=42, limit=12
    )

    self.assertEqual(actual, expected)

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

  def test_qwen3_8b_supported_by_distributed_workers(self):
    config = automodel.call_model_config("qwen3-8b")
    self.assertEqual(config.embed_dim, 4096)
    self.assertEqual(config.num_layers, 36)

  def test_gemma4_e2b_supported_by_distributed_workers(self):
    config = automodel.call_model_config("gemma-4-e2b")
    self.assertEqual(config.embed_dim, 1536)
    self.assertEqual(config.num_layers, 35)
    # The distributed workers drive prefill/decode themselves, so the ring
    # buffer KV cache stays off; assert the default the nodes rely on.
    self.assertFalse(config.use_sliding_window_kv_cache)


class TaskRecordsTest(absltest.TestCase):
  """Tests `create_dataset` records for the orchestrator and rollout worker."""

  def test_create_dataset_wraps_the_generated_entries_in_order(self):
    for shuffle, shuffle_seed in ((True, 42), (False, None)):
      with self.subTest(shuffle=shuffle):
        records = frozenlake.create_dataset(
            split="train", shuffle=shuffle, seed=42, size=32, limit=12
        )
        entries = frozenlake.generate_entries(
            32, 42, shuffle_seed=shuffle_seed, limit=12
        )
        self.assertEqual(
            list(records),
            [
                {"prompt": "", "metadata": {"env_config": {"entry": entry}}}
                for entry in entries
            ],
        )

  def test_create_dataset_rejects_invalid_arguments(self):
    with self.assertRaisesRegex(ValueError, "'train' split"):
      frozenlake.create_dataset(split="test", shuffle=True, seed=42, size=8)
    for grid_size_range in ((2,), (2, 5, 9), (True, 9), (1, 9), (5, 5)):
      with self.subTest(grid_size_range=grid_size_range):
        with self.assertRaisesRegex(ValueError, "grid_size_range"):
          frozenlake.create_dataset(
              split="train",
              shuffle=True,
              seed=42,
              size=8,
              grid_size_range=grid_size_range,
          )

  def test_rollout_env_matches_the_previous_launcher_env(self):
    # Each `FrozenLakeEnv` sets the recipe's global `MAX_STEPS`.
    self.addCleanup(
        setattr, frozenlake_env, "MAX_STEPS", frozenlake_env.MAX_STEPS
    )
    record = frozenlake.create_dataset(
        split="train", shuffle=True, seed=42, size=32, limit=12
    )[0]
    assert record is not None  # `MapDataset.__getitem__` is typed `T | None`.
    env_config = record["metadata"]["env_config"]
    env_class = registry.ENV_REGISTRY.get(frozenlake.FROZENLAKE_ENV_NAME)
    # `RolloutManager` builds the env from `env_kwargs`, the record's
    # `env_config`, and the group fields `DistributedRLEngine` adds to it.
    env = env_class(**{
        **_hp().config_mapping("env_kwargs"),
        **env_config,
        "group_index": 0,
        "num_generations": 8,
        "policy_version": 0,
    })
    # The previous launcher's requests set the map, the max turns as
    # `max_steps`, and `is_slippery`.
    legacy_env = env_class(
        entry=env_config["entry"], max_steps=8, is_slippery=False
    )
    self.assertEqual(env.desc.tolist(), legacy_env.desc.tolist())
    self.assertEqual(env.max_steps, legacy_env.max_steps)
    self.assertEqual(env.env_kwargs, legacy_env.env_kwargs)
    self.assertEqual(env.reset(), legacy_env.reset())

  def test_orchestrator_loads_the_records_in_a_fixed_order(self):
    items = orchestrator.load_prompt_items(
        _hp("data_config.size=32", "data_config.limit=12")
    )
    item = items[0]
    assert item is not None  # `MapDataset.__getitem__` is typed `T | None`.
    # `frozenlake.yaml` shuffles the maps with seed 42.
    entries = frozenlake.generate_entries(32, 42, shuffle_seed=42, limit=12)
    self.assertEqual(
        item,
        {
            "prompt": "",
            "task_name": "frozenlake",
            "max_turns": 8,
            "metadata": {
                "env_config": {"entry": entries[0]},
                "episode_timeout": 600.0,
                "overlong_filter": False,
            },
        },
    )
    # Every epoch repeats the same order.
    self.assertEqual(
        [items[index] for index in range(12, 24)],
        [items[index] for index in range(12)],
    )

  def test_orchestrator_rejects_bad_data_config(self):
    for override, error, regex in (
        # `data_config` values reach `create_dataset` untyped.
        ("data_config.grid_size_range=[2.5,9]", ValueError, "grid_size_range"),
        ("data_config.data_dir=/tmp/data", TypeError, "data_dir"),
    ):
      with self.subTest(override=override):
        with self.assertRaisesRegex(error, regex):
          orchestrator.load_prompt_items(_hp(override))


if __name__ == "__main__":
  absltest.main()
