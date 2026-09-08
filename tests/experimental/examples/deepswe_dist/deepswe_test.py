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

from pathlib import Path
from unittest import mock

from absl.testing import absltest
from examples.deepswe import deepswe_data
from examples.deepswe import swe_env
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist
from tunix.experimental.rl.agentic import registry


class DeepSWEDistTest(absltest.TestCase):

  def test_dataset_loader_reuses_recipe_implementation(self):
    dataset = object()
    with mock.patch.object(
        deepswe_data, "create_dataset", return_value=dataset
    ) as mock_create_dataset:
      result = deepswe.load_deepswe_dataset(
          dataset_name="custom/deepswe",
          dataset_split="validation",
          cache_dir="/tmp/deepswe-cache",
          shuffle=True,
          seed=123,
      )

    self.assertIs(result, dataset)
    mock_create_dataset.assert_called_once_with(
        dataset_name="custom/deepswe",
        dataset_split="validation",
        cache_dir="/tmp/deepswe-cache",
        shuffle=True,
        seed=123,
    )

  def test_distributed_deepswe_reuses_recipe_data_agent_and_env(self):
    package_dir = Path(deepswe.__file__).parent
    self.assertFalse((package_dir / "swe_agent.py").exists())
    self.assertFalse((package_dir / "swe_env.py").exists())
    self.assertFalse((package_dir / "deepswe_data.py").exists())
    self.assertIs(deepswe.deepswe_data, deepswe_data)
    self.assertIs(deepswe.swe_env, swe_env)
    self.assertTrue(issubclass(deepswe.DeepSWEEnv, swe_env.SWEEnv))
    source = Path(deepswe.__file__).read_text(encoding="utf-8")
    self.assertIn("from examples.deepswe import swe_agent", source)

  def test_registered_components(self):
    self.assertIs(
        registry.ENV_REGISTRY.get(deepswe.DEEPSWE_ENV_NAME), deepswe.DeepSWEEnv
    )
    self.assertIs(
        registry.AGENT_REGISTRY.get(deepswe.DEEPSWE_AGENT_NAME),
        deepswe.DeepSWEAgent,
    )

  def test_build_prompt_item_carries_env_and_agent_config(self):
    item = deepswe.build_prompt_item(
        entry={
            "instance_id": "repo__issue-1",
            "problem_statement": "Fix this bug.",
        },
        prompt_idx=0,
        max_turns=3,
        max_response_length=128,
        episode_timeout_secs=300,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        step_timeout_secs=30,
        reward_timeout_secs=40,
        overlong_filter=True,
        env_backend="kubernetes",
        use_agent_sandbox=True,
        batch_size=8,
        num_generations=8,
        max_warmpool_replicas=6,
        scaffold="r2egym",
        env_verbose=True,
    )

    self.assertEqual(item["prompt_id"], "repo__issue-1__0")
    self.assertEqual(item["prompt"], "Fix this bug.")
    self.assertEqual(item["generation_kwargs"]["max_generation_steps"], 128)
    self.assertEqual(item["generation_kwargs"]["max_response_length"], 128)
    self.assertEqual(item["metadata"]["instance_id"], "repo__issue-1")
    self.assertEqual(item["metadata"]["prefix_hash"], "repo__issue-1")
    self.assertEqual(item["metadata"]["episode_timeout"], 300)
    self.assertTrue(item["metadata"]["overlong_filter"])
    env_config = item["metadata"]["env_config"]
    self.assertEqual(env_config["entry"]["instance_id"], "repo__issue-1")
    self.assertEqual(env_config["backend"], "kubernetes")
    self.assertTrue(env_config["use_agent_sandbox"])
    self.assertEqual(env_config["batch_size"], 8)
    self.assertEqual(env_config["group_size"], 8)
    self.assertEqual(env_config["max_warmpool_replicas"], 6)
    self.assertEqual(item["metadata"]["agent_config"], {"scaffold": "r2egym"})

  def test_iter_prompt_items_recycles_dataset(self):
    dataset = [
        {"instance_id": "task-1", "problem_statement": "first"},
        {"instance_id": "task-2", "problem_statement": "second"},
    ]

    items = list(
        deepswe.iter_prompt_items(
            dataset=dataset,
            max_steps=2,
            batch_size=2,
            max_turns=3,
            max_response_length=128,
            episode_timeout_secs=300,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            step_timeout_secs=30,
            reward_timeout_secs=40,
            overlong_filter=True,
            env_backend="kubernetes",
            use_agent_sandbox=True,
            num_generations=2,
            max_warmpool_replicas=None,
            scaffold="r2egym",
            env_verbose=False,
        )
    )

    self.assertEqual(
        [item["prompt_id"] for item in items],
        ["task-1__0", "task-2__1", "task-1__2", "task-2__3"],
    )
    self.assertLen({item["prompt_id"] for item in items}, 4)

  def test_sandbox_fleet_loads_full_dataset_from_env(self):
    dataset = [
        {"instance_id": "task-1", "problem_statement": "first"},
        {"instance_id": "task-2", "problem_statement": "second"},
    ]
    with mock.patch.dict(
        "os.environ",
        {
            "DATASET_NAME": "custom/deepswe",
            "DATASET_SPLIT": "validation",
            "DATASET_CACHE_DIR": "/tmp/deepswe-cache",
            "SHUFFLE": "false",
            "SEED": "123",
            "SANDBOX_MAX_CONCURRENCY": "7",
        },
    ):
      with mock.patch.object(
          deepswe, "load_deepswe_dataset", return_value=dataset
      ) as mock_load:
        with mock.patch.object(
            deepswe.swe_env,
            "_get_global_fleet",
            side_effect=RuntimeError("not initialized"),
        ):
          with mock.patch.object(
              deepswe.swe_env, "_init_global_fleet", return_value="fleet"
          ) as mock_init:
            # pylint: disable=protected-access
            fleet = deepswe._init_sandbox_fleet_from_env(
                {"instance_id": "fallback"},
                group_size=2,
                batch_size=3,
                max_warmpool_replicas=5,
            )
            # pylint: enable=protected-access

    self.assertEqual(fleet, "fleet")
    mock_load.assert_called_once_with(
        dataset_name="custom/deepswe",
        dataset_split="validation",
        dataset_path="",
        cache_dir="/tmp/deepswe-cache",
        shuffle=False,
        seed=123,
    )
    mock_init.assert_called_once_with(
        tasks=dataset,
        max_concurrency=7,
        num_generations=2,
        batch_size=3,
        max_warmpool_replicas=5,
    )

  def test_sandbox_fleet_reuses_process_global_before_loading_dataset(self):
    with mock.patch.object(
        deepswe.swe_env, "_get_global_fleet", return_value="existing"
    ):
      with mock.patch.object(deepswe, "load_deepswe_dataset") as mock_load:
        # pylint: disable=protected-access
        fleet = deepswe._init_sandbox_fleet_from_env(
            {"instance_id": "unused"},
            group_size=8,
            batch_size=8,
            max_warmpool_replicas=None,
        )
        # pylint: enable=protected-access

    self.assertEqual(fleet, "existing")
    mock_load.assert_not_called()

  def test_reference_recipe_defaults(self):
    args = run_deepswe_dist._parse_args([])  # pylint: disable=protected-access

    self.assertEqual(args.model_id, "Qwen/Qwen3-32B")
    self.assertEqual(args.batch_size, 8)
    self.assertEqual(args.mini_batch_size, 8)
    self.assertEqual(args.num_generations, 8)
    self.assertEqual(args.max_prompt_length, 4096)
    self.assertEqual(args.max_response_length, 8192)
    self.assertIsNone(args.top_p)
    self.assertIsNone(args.top_k)
    self.assertEqual(args.advantage_estimator, "rloo")
    self.assertEqual(args.loss_agg_mode, "sequence-mean-token-scale")
    self.assertFalse(args.use_agent_sandbox)
    self.assertEqual(args.env_backend, "kubernetes")
    self.assertEqual(args.episode_timeout_secs, 3 * 60 * 60)
    self.assertTrue(args.overlong_filter)
    self.assertEqual(args.weight_sync_mode.value, "none")

  def test_algorithm_uses_prompt_group_mini_batch_size(self):
    args = run_deepswe_dist._parse_args(  # pylint: disable=protected-access
        [
            "--batch_size=3",
            "--mini_batch_size=3",
            "--num_generations=2",
        ]
    )
    algo = run_deepswe_dist._build_algo(  # pylint: disable=protected-access
        args
    )

    self.assertEqual(algo.mini_batch_size, 3)
    self.assertEqual(algo.group_size, 2)

  def test_rejects_nonfunctional_or_unsupported_modes(self):
    # pylint: disable=protected-access
    fallback_args = run_deepswe_dist._parse_args(
        ["--weight_sync_mode=fallback"]
    )
    with self.assertRaisesRegex(ValueError, "protocol-only"):
      run_deepswe_dist._validate_args(  # pylint: disable=protected-access
          fallback_args
      )

    sandbox_args = run_deepswe_dist._parse_args(
        ["--use_agent_sandbox", "--scaffold=sweagent"]
    )
    with self.assertRaisesRegex(ValueError, "supports only scaffold=r2egym"):
      run_deepswe_dist._validate_args(  # pylint: disable=protected-access
          sandbox_args
      )
    # pylint: enable=protected-access

  def test_validates_full_and_mini_batch_geometry(self):
    # pylint: disable=protected-access
    indivisible_full_batch = run_deepswe_dist._parse_args(
        ["--batch_size=3", "--mini_batch_size=2"]
    )
    with self.assertRaisesRegex(ValueError, "batch_size must be divisible"):
      run_deepswe_dist._validate_args(indivisible_full_batch)

    indivisible_micro_batch = run_deepswe_dist._parse_args(
        [
            "--batch_size=4",
            "--mini_batch_size=2",
            "--num_generations=3",
            "--train_micro_batch_size=4",
        ]
    )
    with self.assertRaisesRegex(
        ValueError, "num_generations must be divisible"
    ):
      run_deepswe_dist._validate_args(indivisible_micro_batch)
    # pylint: enable=protected-access

  def test_rloo_advantages_match_reference_recipe(self):
    args = run_deepswe_dist._parse_args(  # pylint: disable=protected-access
        ["--num_generations=2"]
    )
    algo = run_deepswe_dist._build_algo(args)  # pylint: disable=protected-access

    advantages = algo.compute_advantages([1.0, 3.0, 2.0, 5.0])

    self.assertSequenceAlmostEqual(
        list(map(float, advantages)), [-2.0, 2.0, -3.0, 3.0]
    )
    model_input_fn = algo.build_gen_model_input_fn(pad_id=0, eos_id=1)
    self.assertEqual(model_input_fn.keywords["algo_config"].epsilon_high, 0.28)


if __name__ == "__main__":
  absltest.main()
