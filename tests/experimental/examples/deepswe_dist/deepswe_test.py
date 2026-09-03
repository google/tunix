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

from unittest import mock

from absl.testing import absltest
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.rl.agentic import registry


class DeepSWEDistTest(absltest.TestCase):

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
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        step_timeout_secs=30,
        reward_timeout_secs=40,
        env_backend="kubernetes",
        use_agent_sandbox=True,
        scaffold="r2egym",
        env_verbose=True,
    )

    self.assertEqual(item["prompt_id"], "repo__issue-1")
    self.assertEqual(item["prompt"], "Fix this bug.")
    self.assertEqual(item["generation_kwargs"]["max_generation_steps"], 128)
    env_config = item["metadata"]["env_config"]
    self.assertEqual(env_config["entry"]["instance_id"], "repo__issue-1")
    self.assertEqual(env_config["backend"], "kubernetes")
    self.assertTrue(env_config["use_agent_sandbox"])
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
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            step_timeout_secs=30,
            reward_timeout_secs=40,
            env_backend="kubernetes",
            use_agent_sandbox=True,
            scaffold="r2egym",
            env_verbose=False,
        )
    )

    self.assertEqual(
        [item["prompt_id"] for item in items],
        ["task-1", "task-2", "task-1", "task-2"],
    )

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
            deepswe.swe_env, "_init_global_fleet", return_value="fleet"
        ) as mock_init:
          fleet = deepswe._init_sandbox_fleet_from_env(  # pylint: disable=protected-access
              {"instance_id": "fallback"},
              group_size=2,
          )

    self.assertEqual(fleet, "fleet")
    mock_load.assert_called_once_with(
        dataset_name="custom/deepswe",
        dataset_split="validation",
        dataset_path="",
        cache_dir="/tmp/deepswe-cache",
        shuffle=False,
        seed=123,
    )
    mock_init.assert_called_once_with(tasks=dataset, max_concurrency=7)


if __name__ == "__main__":
  absltest.main()
