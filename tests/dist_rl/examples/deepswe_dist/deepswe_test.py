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

"""Tests for the DeepSWE distributed GRPO components and task records."""

from collections.abc import Mapping
import json
import pathlib
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

# `r2egym` comes with the optional `r2egym` extra. pytest also reports this
# module-level `unittest.SkipTest` as a skip.
try:
  import r2egym  # pylint: disable=unused-import,g-import-not-at-top
except ImportError as exc:
  raise unittest.SkipTest("r2egym is not installed") from exc

# pylint: disable=g-import-not-at-top
import datasets
import grain
import numpy as np
from tunix.cli import config as cli_config
from tunix.dist_rl.common import registry
from tunix.dist_rl.examples.deepswe_dist import deepswe
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator
from examples.deepswe import swe_agent
from examples.deepswe import swe_env

# pylint: enable=g-import-not-at-top

_DEEPSWE_YAML = pathlib.Path(deepswe.__file__).parent / "deepswe.yaml"


def _hp(*overrides: str) -> resolve_config.DistHyperParameters:
  """Resolves `deepswe.yaml` on top of `base_dist_config.yaml`."""
  return resolve_config.initialize(
      [f"override_config_file={_DEEPSWE_YAML}", *overrides]
  )


def _sample_rows(count: int) -> list[dict[str, object]]:
  """Builds deterministic R2E-Gym-style rows for hermetic on-disk tests."""
  return [
      {
          "instance_id": f"repo__{index}",
          "problem_statement": f"Fix bug {index}",
          "docker_image": f"r2egym/repo:{index}",
          "modified_files": [f"pkg/mod_{index}.py", f"tests/test_{index}.py"],
      }
      for index in range(count)
  ]


class DeepSWEDistTest(parameterized.TestCase):

  def test_registered_components(self):
    env_cls = registry.ENV_REGISTRY.get(deepswe.DEEPSWE_ENV_NAME)
    agent_cls = registry.AGENT_REGISTRY.get(deepswe.DEEPSWE_AGENT_NAME)

    self.assertIs(env_cls, deepswe.DeepSWEEnv)
    self.assertTrue(issubclass(env_cls, swe_env.SWEEnv))
    self.assertIs(agent_cls, deepswe.DeepSWEAgent)
    self.assertTrue(issubclass(agent_cls, swe_agent.SWEAgent))
    self.assertEqual(agent_cls.name, "deepswe_agent")

  def test_normalize_example_value_and_as_text(self):
    self.assertEqual(deepswe.normalize_example_value(np.array([42])), 42)
    self.assertEqual(
        deepswe.normalize_example_value(np.array([b"x", b"y"])),
        ["x", "y"],
    )
    self.assertEqual(
        deepswe.normalize_example_value(np.bytes_(b"repo__1")),
        "repo__1",
    )
    self.assertEqual(deepswe.normalize_example_value(b"patch"), "patch")
    self.assertEqual(
        deepswe.normalize_example_value([np.bytes_(b"a"), [b"b"]]),
        ["a", ["b"]],
    )
    self.assertEqual(deepswe.as_text(np.array([b"hello"])), "hello")
    self.assertEqual(deepswe.as_text(7), "7")

  def test_create_dataset_from_disk_builds_task_records(self):
    dataset_dir = self.create_tempdir().full_path
    datasets.Dataset.from_list(_sample_rows(3)).save_to_disk(dataset_dir)

    with self.assertLogs(level="INFO") as logs:
      records = deepswe.create_dataset(
          split="train",
          shuffle=False,
          seed=42,
          dataset_path=dataset_dir,
      )

    self.assertIsInstance(records, grain.MapDataset)
    self.assertLen(records, 3)
    first = records[0]
    assert first is not None
    self.assertEqual(set(first), {"prompt", "metadata"})
    self.assertEqual(first["prompt"], "Fix bug 0")

    metadata = first["metadata"]
    assert isinstance(metadata, Mapping)
    self.assertEqual(
        set(metadata),
        {"instance_id", "problem_statement", "env_config"},
    )
    self.assertEqual(metadata["instance_id"], "repo__0")
    self.assertEqual(metadata["problem_statement"], "Fix bug 0")

    env_config = metadata["env_config"]
    assert isinstance(env_config, Mapping)
    self.assertEqual(set(env_config), {"entry", "prompt_id"})
    self.assertEqual(env_config["prompt_id"], "repo__0")
    entry = env_config["entry"]
    assert isinstance(entry, Mapping)
    self.assertEqual(
        entry["modified_files"],
        json.dumps(["pkg/mod_0.py", "tests/test_0.py"]),
    )
    self.assertTrue(
        any(
            f"Loaded DeepSWE dataset: source={dataset_dir} split=train size=3."
            in message
            for message in logs.output
        )
    )

  def test_create_dataset_from_dataset_dict_shuffles_and_limits(self):
    dataset_dir = self.create_tempdir().full_path
    rows = _sample_rows(6)
    datasets.DatasetDict(
        {"train": datasets.Dataset.from_list(rows)}
    ).save_to_disk(dataset_dir)
    expected_ids = [
        row["instance_id"]
        for row in datasets.Dataset.from_list(rows)
        .shuffle(seed=17)
        .to_list()[:4]
    ]

    with self.assertLogs(level="INFO") as logs:
      records = deepswe.create_dataset(
          split="train",
          shuffle=True,
          seed=17,
          dataset_name="",
          dataset_path=dataset_dir,
          limit=4,
      )

    actual_ids = []
    for record in records:
      assert record is not None
      metadata = record["metadata"]
      assert isinstance(metadata, Mapping)
      actual_ids.append(metadata["instance_id"])
    self.assertEqual(actual_ids, expected_ids)
    self.assertTrue(
        any(
            f"Loaded DeepSWE dataset: source={dataset_dir} split=train size=4."
            in message
            for message in logs.output
        )
    )

  def test_create_dataset_fallbacks_for_instance_id_and_problem_statement(self):
    dataset_dir = self.create_tempdir().full_path
    rows = [
        {
            "instance_id": "",
            "problem_statement": "From statement",
            "prompt": "Ignored",
        },
        {
            "instance_id": "repo__1",
            "problem_statement": None,
            "prompt": "From prompt",
        },
        {
            "instance_id": "repo__2",
            "problem_statement": None,
            "prompt": None,
        },
        {
            "instance_id": None,
            "problem_statement": None,
            "prompt": None,
        },
    ]
    datasets.Dataset.from_list(rows).save_to_disk(dataset_dir)

    records = list(
        deepswe.create_dataset(
            split="train",
            shuffle=False,
            dataset_path=dataset_dir,
        )
    )

    expected = [
        ("deepswe_0", "From statement"),
        ("repo__1", "From prompt"),
        ("repo__2", "repo__2"),
        ("deepswe_3", ""),
    ]
    actual = []
    for record in records:
      assert record is not None
      metadata = record["metadata"]
      assert isinstance(metadata, Mapping)
      actual.append((metadata["instance_id"], record["prompt"]))
    self.assertEqual(actual, expected)

  def test_create_dataset_forwards_hub_arguments_and_normalizes_cache_dir(self):
    hub_source = grain.MapDataset.source([
        {
            "instance_id": np.bytes_(b"hub__0"),
            "problem_statement": "Hub issue",
            "modified_files": ["lib/core.py"],
        }
    ])
    mock_create = self.enter_context(
        mock.patch.object(
            deepswe.deepswe_data,
            "create_dataset",
            autospec=True,
            spec_set=True,
            return_value=hub_source,
        )
    )

    records = deepswe.create_dataset(
        split="validation",
        shuffle=False,
        seed=99,
        dataset_name="R2E-Gym/Custom",
        cache_dir="",
    )

    mock_create.assert_called_once_with(
        dataset_name="R2E-Gym/Custom",
        dataset_split="validation",
        cache_dir=None,
        shuffle=False,
        seed=99,
    )
    first = records[0]
    assert first is not None
    metadata = first["metadata"]
    assert isinstance(metadata, Mapping)
    self.assertEqual(metadata["instance_id"], "hub__0")
    env_config = metadata["env_config"]
    assert isinstance(env_config, Mapping)
    entry = env_config["entry"]
    assert isinstance(entry, Mapping)
    self.assertEqual(entry["modified_files"], '["lib/core.py"]')

    mock_create.reset_mock()
    deepswe.create_dataset(
        split="train",
        shuffle=True,
        seed=11,
        dataset_name="R2E-Gym/Custom",
        cache_dir="/tmp/hf_cache",
    )
    mock_create.assert_called_once_with(
        dataset_name="R2E-Gym/Custom",
        dataset_split="train",
        cache_dir="/tmp/hf_cache",
        shuffle=True,
        seed=11,
    )

  @parameterized.named_parameters(
      ("non_str_split", {"split": 123}, TypeError, "split"),
      ("empty_split", {"split": ""}, ValueError, "split"),
      ("whitespace_split", {"split": "   "}, ValueError, "split"),
      ("non_bool_shuffle", {"shuffle": 1}, TypeError, "shuffle"),
      ("bool_seed", {"seed": True}, TypeError, "seed"),
      ("non_int_seed", {"seed": "42"}, TypeError, "seed"),
      (
          "non_str_dataset_name",
          {"dataset_name": 123},
          TypeError,
          "dataset_name",
      ),
      (
          "whitespace_dataset_name",
          {"dataset_name": "  "},
          ValueError,
          "dataset_name",
      ),
      (
          "non_str_dataset_path",
          {"dataset_path": 123},
          TypeError,
          "dataset_path",
      ),
      (
          "whitespace_dataset_path",
          {"dataset_path": "  "},
          ValueError,
          "dataset_path",
      ),
      (
          "empty_source",
          {"dataset_name": "", "dataset_path": ""},
          ValueError,
          "dataset_path or dataset_name",
      ),
      ("non_str_cache_dir", {"cache_dir": 123}, TypeError, "cache_dir"),
      ("whitespace_cache_dir", {"cache_dir": "   "}, ValueError, "cache_dir"),
      ("bool_limit", {"limit": True}, TypeError, "limit"),
      ("non_int_limit", {"limit": 1.5}, TypeError, "limit"),
      ("zero_limit", {"limit": 0}, ValueError, "limit"),
      ("negative_limit", {"limit": -2}, ValueError, "limit"),
  )
  def test_create_dataset_rejects_invalid_arguments(
      self, kwargs, expected_error, error_regex
  ):
    with self.assertRaisesRegex(expected_error, error_regex):
      deepswe.create_dataset(**kwargs)

  def test_create_dataset_rejects_empty_dataset_and_non_mapping_items(self):
    mock_create = self.enter_context(
        mock.patch.object(
            deepswe.deepswe_data,
            "create_dataset",
            autospec=True,
            spec_set=True,
            return_value=grain.MapDataset.source([]),
        )
    )
    with self.assertRaisesRegex(ValueError, "empty"):
      deepswe.create_dataset()

    mock_create.return_value = grain.MapDataset.source(["not-a-mapping"])
    records = deepswe.create_dataset()
    with self.assertRaisesRegex(TypeError, "mapping"):
      _ = records[0]

  def test_orchestrator_load_prompt_items_stamps_config_and_repeats_epochs(
      self,
  ):
    dataset_dir = self.create_tempdir().full_path
    datasets.Dataset.from_list(_sample_rows(2)).save_to_disk(dataset_dir)
    hp = resolve_config.initialize([
        "data_module=tunix.dist_rl.examples.deepswe_dist.deepswe",
        "task_name=deepswe",
        f"data_config.dataset_path={dataset_dir}",
        "data_config.shuffle=true",
        "data_config.seed=7",
        "agentic_grpo_config.max_turns=5",
        "agentic_grpo_config.episode_timeout=900.0",
        "agentic_grpo_config.overlong_filter=true",
    ])

    items = orchestrator.load_prompt_items(hp)
    first = items[0]
    assert first is not None
    self.assertEqual(first["task_name"], "deepswe")
    self.assertEqual(first["max_turns"], 5)
    metadata = first["metadata"]
    assert isinstance(metadata, Mapping)
    self.assertNotIn("agent_config", metadata)
    self.assertEqual(metadata["episode_timeout"], 900.0)
    self.assertTrue(metadata["overlong_filter"])
    self.assertEqual([items[0], items[1]], [items[2], items[3]])

  # TODO(noghabi): Remove with `build_prompt_item` and `run_deepswe_dist.py`.
  def test_rollout_env_and_agent_match_legacy_build_prompt_item(self):
    dataset_dir = self.create_tempdir().full_path
    datasets.Dataset.from_list(_sample_rows(1)).save_to_disk(dataset_dir)
    record = deepswe.create_dataset(
        split="train",
        shuffle=False,
        dataset_path=dataset_dir,
    )[0]
    assert record is not None
    metadata = record["metadata"]
    assert isinstance(metadata, Mapping)
    env_config = metadata["env_config"]
    assert isinstance(env_config, Mapping)
    entry = env_config["entry"]
    assert isinstance(entry, dict)

    env_kwargs = {
        "max_steps": 3,
        "step_timeout": 1800,
        "reward_timeout": 1800,
        "backend": "kubernetes",
        "use_agent_sandbox": False,
        "scaffold": "r2egym",
        "verbose": False,
    }
    env_cls = registry.ENV_REGISTRY.get(deepswe.DEEPSWE_ENV_NAME)
    env = env_cls(**{
        **env_kwargs,
        **env_config,
        "group_index": 1,
        "num_generations": 2,
        "policy_version": 4,
    })

    legacy_item = deepswe.build_prompt_item(
        entry=entry,
        prompt_idx=0,
        max_turns=3,
        max_response_length=1024,
        temperature=0.7,
        top_p=0.95,
        top_k=None,
        step_timeout_secs=1800,
        reward_timeout_secs=1800,
        env_backend="kubernetes",
        use_agent_sandbox=False,
        scaffold="r2egym",
        env_verbose=False,
    )
    legacy_env = env_cls(**{
        **legacy_item["metadata"]["env_config"],
        "group_index": 1,
        "num_generations": 2,
        "policy_version": 4,
    })

    self.assertEqual(env.max_steps, 3)
    self.assertEqual(env.step_timeout, 1800)
    self.assertEqual(env.reward_timeout, 1800)
    self.assertEqual(env.backend, "kubernetes")
    self.assertFalse(env.use_agent_sandbox)
    self.assertEqual(env.scaffold, "r2egym")
    self.assertFalse(env.verbose)
    self.assertEqual(env.extra_kwargs["group_id"], "repo__0")
    self.assertEqual(env.extra_kwargs["pair_index"], 1)
    self.assertEqual(env.entry, legacy_env.entry)
    self.assertEqual(env.task, legacy_env.task)
    self.assertEqual(env.extra_kwargs, legacy_env.extra_kwargs)

    agent_cls = registry.AGENT_REGISTRY.get(deepswe.DEEPSWE_AGENT_NAME)
    agent = agent_cls(scaffold="r2egym")
    self.assertIsInstance(agent, deepswe.DeepSWEAgent)
    self.assertEqual(agent.name, "deepswe_agent")
    self.assertEqual(
        agent.system_prompt,
        swe_agent.get_system_prompt(scaffold="r2egym", use_fn_calling=False),
    )
    self.assertEqual(
        agent.user_prompt_template,
        swe_agent.get_user_prompt_template(
            scaffold="r2egym", use_fn_calling=False
        ),
    )

  def test_rollout_env_and_agent_from_deepswe_yaml(self):
    dataset_dir = self.create_tempdir().full_path
    datasets.Dataset.from_list(_sample_rows(1)).save_to_disk(dataset_dir)
    hp = _hp(
        f"data_config.dataset_path={dataset_dir}",
        "data_config.shuffle=false",
    )
    item = orchestrator.load_prompt_items(hp)[0]
    assert item is not None
    metadata = item["metadata"]
    assert isinstance(metadata, Mapping)
    self.assertNotIn("agent_config", metadata)
    self.assertEqual(metadata["episode_timeout"], 1800.0)
    self.assertFalse(metadata["overlong_filter"])
    env_config = metadata["env_config"]
    assert isinstance(env_config, Mapping)

    env_cls = registry.ENV_REGISTRY.get(hp.config["env_class_path"])
    env = env_cls(**{
        **hp.config_mapping("env_kwargs"),
        **env_config,
        "group_index": 1,
        "num_generations": 2,
        "policy_version": 4,
    })
    self.assertEqual(env.max_steps, 3)
    self.assertEqual(env.step_timeout, 1800)
    self.assertEqual(env.reward_timeout, 1800)
    self.assertEqual(env.backend, "kubernetes")
    self.assertFalse(env.use_agent_sandbox)
    self.assertEqual(env.scaffold, "r2egym")
    self.assertFalse(env.verbose)
    self.assertEqual(env.extra_kwargs["group_id"], "repo__0")
    self.assertEqual(env.extra_kwargs["pair_index"], 1)
    self.assertEqual(env.task["prompt_id"], "repo__0")
    self.assertEqual(env.task["group_index"], 1)
    self.assertEqual(env.task["num_generations"], 2)
    self.assertEqual(env.task["policy_version"], 4)

    agent_cls = registry.AGENT_REGISTRY.get(hp.config["agent_class_path"])
    agent = agent_cls(**hp.config_mapping("agent_kwargs"))
    self.assertIsInstance(agent, deepswe.DeepSWEAgent)
    self.assertEqual(agent.name, "deepswe_agent")
    self.assertEqual(
        agent.system_prompt,
        swe_agent.get_system_prompt(scaffold="r2egym", use_fn_calling=False),
    )
    self.assertEqual(
        agent.user_prompt_template,
        swe_agent.get_user_prompt_template(
            scaffold="r2egym", use_fn_calling=False
        ),
    )

  def test_deepswe_yaml_preserves_recipe_defaults(self):
    hp = _hp()
    self.assertEqual(
        hp.config_mapping("chat_parser_config"),
        {"type": "auto", "enable_thinking": False},
    )
    rollout_cfg = hp.config_mapping("rollout_config")
    self.assertEqual(rollout_cfg["eos_tokens"], "")
    grpo_map = hp.config_mapping("agentic_grpo_config")
    env_kwargs = hp.config_mapping("env_kwargs")
    self.assertEqual(env_kwargs["max_steps"], grpo_map["max_turns"])
    self.assertEqual(grpo_map["max_turns"], 3)
    self.assertEqual(grpo_map["max_concurrency"], 64)

    rl_train = hp.config_mapping("rl_training_config")
    optimizer = rl_train["actor_optimizer_config"]
    self.assertEqual(cli_config.create_learning_rate(optimizer), 1e-6)
    self.assertEqual((optimizer["b1"], optimizer["b2"]), (0.9, 0.999))
    self.assertEqual(optimizer["weight_decay"], 0.0)
    self.assertEqual(optimizer["chain_kwargs"], {"max_norm": 1.0})

    self.assertIsNone(grpo_map["epsilon_high"])
    grpo = orchestrator.build_grpo_config(hp)
    self.assertEqual(grpo.num_generations, 2)
    self.assertEqual(grpo.beta, 0.0)
    self.assertEqual(grpo.epsilon, 0.2)
    self.assertEqual(grpo.epsilon_high, 0.2)
    self.assertEqual(grpo.loss_algo, "grpo")
    self.assertEqual(grpo.kl_loss_mode, "kl")
    self.assertEqual(grpo.advantage_estimator, "grpo")
    self.assertIsNone(grpo.sampler_is)
    self.assertEqual(grpo.reward_mode, "env")
    self.assertTrue(grpo.use_rollout_logps)

    _, derived_rpc_timeout = orchestrator.resolve_weight_sync_timeouts(hp)
    self.assertEqual(derived_rpc_timeout, 2100.0)
    cluster = hp.config_mapping("cluster_config")
    self.assertEqual(cluster["local"]["worker_shutdown_grace_secs"], 60)

  def test_orchestrator_loads_default_deepswe_yaml_from_the_hub(self):
    hub_source = grain.MapDataset.source(_sample_rows(2))
    mock_create = self.enter_context(
        mock.patch.object(
            deepswe.deepswe_data,
            "create_dataset",
            autospec=True,
            spec_set=True,
            return_value=hub_source,
        )
    )
    items = orchestrator.load_prompt_items(_hp())
    mock_create.assert_called_once_with(
        dataset_name="R2E-Gym/R2E-Gym-Subset",
        dataset_split="train",
        cache_dir=None,
        shuffle=True,
        seed=42,
    )
    first = items[0]
    assert first is not None
    self.assertEqual(first["prompt"], "Fix bug 0")
    self.assertEqual(first["task_name"], "deepswe")
    self.assertEqual(first["max_turns"], 3)


if __name__ == "__main__":
  absltest.main()
