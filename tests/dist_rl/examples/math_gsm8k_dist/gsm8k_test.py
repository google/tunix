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

"""Tests for the GSM8K example: components, `gsm8k.yaml`, and task records."""

from collections.abc import Mapping, Sequence
import pathlib
from unittest import mock

from absl.testing import absltest
import numpy as np
import tensorflow_datasets as tfds
from tunix.dist_rl.common import registry
from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k
from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k_rewards
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator
from tunix.dist_rl.weight_sync import weight_sync_coordinator

_GSM8K_YAML = pathlib.Path(gsm8k.__file__).parent / "gsm8k.yaml"
_GSM8K_MODULE = "tunix.dist_rl.examples.math_gsm8k_dist.gsm8k"
_REWARDS_MODULE = "tunix.dist_rl.examples.math_gsm8k_dist.gsm8k_rewards"


def _hp(*overrides: str) -> resolve_config.DistHyperParameters:
  """Resolves `gsm8k.yaml` on top of `base_dist_config.yaml`."""
  return resolve_config.initialize(
      [f"override_config_file={_GSM8K_YAML}", *overrides]
  )


class _FakeMapDataSource:
  """Stands in for the source `tfds.data_source` returns."""

  def __init__(self, data: Sequence[Mapping[str, object]]) -> None:
    self._data = list(data)

  def __len__(self) -> int:
    return len(self._data)

  def __getitem__(self, idx: int) -> Mapping[str, object]:
    return self._data[idx]


def _tfds_examples() -> _FakeMapDataSource:
  """Two raw GSM8K examples, as `tfds.data_source` returns them."""
  return _FakeMapDataSource([
      {"question": b"Q0", "answer": b"Zero. #### 0"},
      {"question": b"Q1", "answer": b"One. #### 1"},
  ])


def _expected_record(index: int) -> dict[str, object]:
  """The task record `create_dataset` makes from `_tfds_examples()[index]`."""
  prompt = gsm8k.build_prompt(f"Q{index}")
  return {
      "prompt": prompt,
      "metadata": {
          "answer": str(index),
          "env_config": {
              # The orchestrator's trajectory log reads `prompt` here.
              "prompt": prompt,
              "question": f"Q{index}",
              "answer": str(index),
          },
      },
  }


class GSM8KTest(absltest.TestCase):

  def test_registered_components(self):
    self.assertIs(
        registry.ENV_REGISTRY.get(gsm8k.GSM8K_ENV_NAME), gsm8k.GSM8KEnv
    )
    self.assertIs(
        registry.AGENT_REGISTRY.get(gsm8k.GSM8K_AGENT_NAME), gsm8k.GSM8KAgent
    )

  def test_env_scores_formatted_correct_answer(self):
    env = gsm8k.GSM8KEnv(prompt="What is 2+2?", answer="4")
    obs, _ = env.reset()
    self.assertEqual(obs, {"prompts": "What is 2+2?"})
    next_obs, reward, done, info = env.step(
        "<reasoning>2+2=4.</reasoning><answer>\\boxed{4}</answer>"
    )
    self.assertEqual(next_obs["gold_answer"], "4")
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(info["correct"])
    self.assertTrue(info["format_correct"])

  def test_format_parity_with_qwen3_grpo_demo(self):
    """is_gsm8k_format_correct must match qwen3_grpo_demo's is_vtc_format_correct.

    Exactly one </reasoning>, exactly one <answer>..</answer>, in that order.
    The opening <reasoning> tag is not required: the prompt template opens it,
    so a completion that follows the prompt only carries the closing tag.
    """
    # A completion that follows the prompt (closing tag only) is well-formed.
    self.assertTrue(
        gsm8k.is_gsm8k_format_correct(
            "2+2=4\n</reasoning>\n<answer>\\boxed{4}</answer>"
        )
    )
    # Re-emitting the opening tag is also fine.
    self.assertTrue(
        gsm8k.is_gsm8k_format_correct(
            "<reasoning>work</reasoning><answer>\\boxed{4}</answer>"
        )
    )
    # Tags out of order, duplicated, or missing are not well-formed.
    self.assertFalse(
        gsm8k.is_gsm8k_format_correct("<answer>\\boxed{4}</answer></reasoning>")
    )
    self.assertFalse(
        gsm8k.is_gsm8k_format_correct(
            "</reasoning></reasoning><answer>\\boxed{4}</answer>"
        )
    )
    self.assertFalse(gsm8k.is_gsm8k_format_correct("The answer is \\boxed{4}"))
    # Qwen-native <think> is not the recipe's format.
    self.assertFalse(
        gsm8k.is_gsm8k_format_correct("<think>2+2=4</think>\\boxed{4}")
    )

  def test_extract_boxed_answer_requires_boxed(self):
    # Parity with examples/math_gsm8k/qwen3_grpo_demo.py: an <answer> block
    # without \boxed{} is not an answer.
    self.assertIsNone(gsm8k.extract_boxed_answer("<answer>42</answer>"))
    self.assertEqual(
        gsm8k.extract_boxed_answer("<answer>\\boxed{7}</answer>"), "7"
    )

  def test_think_and_boxed_scores_as_answer_only(self):
    # Qwen thinking-mode shape: right answer, but not the recipe's format,
    # so 0.5.
    reward, info = gsm8k.score_gsm8k_completion(
        "<think>2+2=4</think>\\boxed{4}", "4"
    )
    self.assertEqual(reward, 0.5)
    self.assertFalse(info["format_correct"])
    self.assertTrue(info["answer_correct"])

  def test_scores_formatted_wrong_answer_with_format_reward(self):
    reward, info = gsm8k.score_gsm8k_completion(
        "<reasoning>2+2=5.</reasoning><answer>\\boxed{5}</answer>", "4"
    )
    self.assertEqual(reward, 0.1)
    self.assertTrue(info["format_correct"])
    self.assertFalse(info["answer_correct"])

  def test_build_prompt_and_hash_answer(self):
    self.assertEqual(gsm8k.extract_hash_answer("reasoning #### 72"), "72")
    self.assertIn(
        "<answer>\\boxed{}</answer>",
        gsm8k.build_prompt("How many clips?"),
    )

  def test_agent_forwards_model_response_as_action(self):
    agent = gsm8k.GSM8KAgent()
    action = agent.update_from_model("The answer is 4.")
    self.assertEqual(action.action, "The answer is 4.")
    self.assertEqual(agent.name, gsm8k.GSM8K_AGENT_NAME)
    self.assertLen(agent.trajectory.steps, 1)

  def test_normalize_example_value(self):
    self.assertEqual(
        gsm8k.normalize_example_value(np.array(["test"])),
        "test",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(np.bytes_(b"bytes_val")),
        "bytes_val",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(b"raw_bytes"),
        "raw_bytes",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(np.array(["a", "b"])),
        ["a", "b"],
    )
    self.assertEqual(gsm8k.normalize_example_value("string"), "string")
    self.assertEqual(gsm8k.normalize_example_value(42), 42)

  def test_as_text(self):
    self.assertEqual(gsm8k.as_text(b"hello"), "hello")
    self.assertEqual(gsm8k.as_text(np.array(["world"])), "world")
    self.assertEqual(gsm8k.as_text(123), "123")

  def test_load_gsm8k_dataset(self):
    fake_records = [
        {
            "question": np.array([b"What is 10 + 5?"]),
            "answer": np.array([b"10 + 5 is 15 #### 15"]),
        },
        {
            "question": "What is 3 * 7?",
            "answer": "3 * 7 = 21 #### 21",
        },
    ]
    fake_source = _FakeMapDataSource(fake_records)

    with mock.patch.object(
        tfds, "data_source", return_value=fake_source
    ) as mock_ds:
      dataset = gsm8k.load_gsm8k_dataset(
          split="train",
          data_dir="/tmp/test_dir",
          shuffle=False,
          seed=123,
      )

    mock_ds.assert_called_once_with(
        "gsm8k",
        split="train",
        data_dir="/tmp/test_dir",
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )
    self.assertLen(dataset, 2)
    # `MapDataset.__getitem__` is typed `T | None`.
    first_item = dataset[0]
    assert first_item is not None
    self.assertEqual(first_item["question"], "What is 10 + 5?")
    self.assertEqual(first_item["answer"], "15")
    self.assertIn("What is 10 + 5?", first_item["prompts"])
    self.assertIn("<answer>\\boxed{}</answer>", first_item["prompts"])

    second_item = dataset[1]
    assert second_item is not None
    self.assertEqual(second_item["question"], "What is 3 * 7?")
    self.assertEqual(second_item["answer"], "21")

  def test_make_gsm8k_reward_fn(self):
    reward_fn = gsm8k.make_gsm8k_reward_fn(debug=True)

    rewards = reward_fn(
        prompts=["q1", "q2", "q3", "q4"],
        completions=[
            (
                "<reasoning>Reasoning"
                " step.</reasoning><answer>\\boxed{42}</answer>"
            ),
            (
                "<reasoning>Reasoning"
                " step.</reasoning><answer>\\boxed{100}</answer>"
            ),
            "The answer is \\boxed{42}",
            "No answer here",
        ],
        answer=["42", "42", "42", "42"],
    )
    self.assertEqual(rewards, [1.0, 0.1, 0.5, 0.0])


class ExampleConfigTest(absltest.TestCase):
  """`gsm8k.yaml` resolved on top of `base_dist_config.yaml`."""

  def test_gsm8k_yaml(self):
    hp = _hp()
    config = hp.config
    self.assertEqual(config["batch_size"], 4)
    self.assertEqual(config["rl_training_config"]["mini_batch_size"], 2)
    self.assertEqual(config["rl_training_config"]["max_steps"], 1)
    grpo = config["agentic_grpo_config"]
    self.assertEqual(grpo["num_generations"], 8)
    self.assertEqual(grpo["beta"], 0.0)
    self.assertEqual(grpo["max_turns"], 1)
    self.assertEqual(grpo["reward_mode"], "env")
    self.assertTrue(grpo["use_rollout_logps"])
    self.assertEqual(config["reward_functions"], [])
    self.assertEqual(config["chat_parser_config"]["type"], "raw")
    self.assertEqual(
        config["tokenizer_config"]["tokenizer_path"], "Qwen/Qwen3-1.7B"
    )
    self.assertEqual(
        config["actor_model_config"]["model_id"], "Qwen/Qwen3-1.7B"
    )
    self.assertEqual(config["rollout_config"]["max_prompt_length"], 1024)
    self.assertEqual(config["rollout_config"]["total_generation_steps"], 1024)
    self.assertEqual(config["data_module"], _GSM8K_MODULE)
    self.assertEqual(config["task_name"], "gsm8k")
    self.assertEqual(
        hp.config_mapping("data_config"),
        {"split": "train", "shuffle": True, "seed": 42},
    )
    self.assertEqual(config["env_class_path"], gsm8k.GSM8K_ENV_NAME)
    self.assertEqual(config["agent_class_path"], gsm8k.GSM8K_AGENT_NAME)
    optimizer = config["rl_training_config"]["actor_optimizer_config"]
    self.assertEqual(optimizer["opt_chain_type"], "clip_by_global_norm")
    self.assertEqual(optimizer["chain_kwargs"], {"max_norm": 1.0})
    self.assertEqual(optimizer["learning_rate"], 2.0e-7)
    self.assertEqual(hp.mesh_axes("actor_model_config")["tp"], 2)
    self.assertEqual(hp.mesh_axes("rollout_model_config")["tp"], 2)
    cluster = config["cluster_config"]
    self.assertEqual(cluster["target"], "local")
    self.assertEqual(
        cluster["orchestrator_main"],
        "tunix.dist_rl.launch.nodes.orchestrator.main",
    )
    self.assertEqual(cluster["registry_module"], _GSM8K_MODULE)
    self.assertTrue(cluster["artifact_root"].endswith("artifacts/math_gsm8k"))
    self.assertEqual(cluster["local"]["trainer_tpu_chips"], "0,1")
    self.assertEqual(cluster["local"]["rollout_tpu_chips"], "2,3")
    # `beta == 0.0` disables the reference inference worker.
    self.assertFalse(cluster["run_inference_node"])
    traj_store = config["trajectory_store_config"]
    self.assertFalse(traj_store["enable"])
    self.assertEqual(traj_store["backend"], "file")

  def test_weight_sync_timeouts(self):
    # Same values as `run_gsm8k_dist_grpo.py`: the coordinator's default
    # `PhaseTimeouts` and `rpc_timeout_s=1800`.
    phase_timeouts, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        _hp()
    )
    self.assertEqual(phase_timeouts, weight_sync_coordinator.PhaseTimeouts())
    self.assertEqual(rpc_timeout_s, 1800.0)

  def test_grpo_config_matches_reference_grpo_defaults(self):
    config = orchestrator.build_grpo_config(_hp())
    self.assertEqual(config.num_generations, 8)
    self.assertEqual(config.beta, 0.0)
    self.assertEqual(config.epsilon, 0.2)
    # `GRPOConfig` resolves an unset `epsilon_high` to `epsilon` (symmetric).
    self.assertEqual(config.epsilon_high, config.epsilon)
    self.assertEqual(config.kl_loss_mode, "kl")
    self.assertEqual(config.loss_algo, "grpo")
    self.assertEqual(config.advantage_estimator, "grpo")
    self.assertIsNone(config.sampler_is)
    self.assertTrue(config.use_rollout_logps)
    self.assertEqual(config.reward_mode, "env")

  def test_batch_config_and_generation_args(self):
    batch_config = orchestrator.build_batch_config(_hp(), pad_id=3, eos_id=7)
    self.assertEqual(batch_config.pad_id, 3)
    self.assertIsNone(batch_config.num_data_shards)
    self.assertEqual(batch_config.segment_alignment_boundary, 1)
    generation_args = orchestrator.build_generation_args(_hp())
    self.assertEqual(generation_args.temperature, 1.0)
    self.assertEqual(generation_args.top_k, 0)

  def test_metrics_logging_options(self):
    options = orchestrator.build_metrics_logging_options(_hp())
    self.assertEqual(options.log_dir, "/tmp/trellis_gsm8k")
    self.assertEqual(options.project_name, "trellis-gsm8k")
    self.assertEqual(options.flush_every_n_steps, 1)


class TaskRecordsTest(absltest.TestCase):
  """Tests `create_dataset` records for the orchestrator and rollout worker."""

  def test_create_dataset_records(self):
    with mock.patch.object(
        tfds, "data_source", return_value=_tfds_examples()
    ) as data_source:
      records = gsm8k.create_dataset(
          split="train[:2]", data_dir="/tmp/data", shuffle=False, seed=7
      )
    data_source.assert_called_once_with(
        "gsm8k",
        split="train[:2]",
        data_dir="/tmp/data",
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )
    self.assertLen(records, 2)
    self.assertEqual(records[1], _expected_record(1))

  def test_create_dataset_uses_the_tfds_default_data_dir(self):
    with mock.patch.object(
        tfds, "data_source", return_value=_tfds_examples()
    ) as data_source:
      records = gsm8k.create_dataset(split="train", shuffle=True, seed=42)
    self.assertIsNone(data_source.call_args.kwargs["data_dir"])
    self.assertCountEqual(
        [records[0], records[1]], [_expected_record(0), _expected_record(1)]
    )

  def test_unknown_data_config_key_fails_loud(self):
    hp = _hp("data_config.data_directory=/tmp/data")
    with self.assertRaisesRegex(TypeError, "data_directory"):
      orchestrator.load_prompt_items(hp)

  def test_orchestrator_loads_gsm8k_prompt_items(self):
    with mock.patch.object(
        tfds, "data_source", return_value=_tfds_examples()
    ) as data_source:
      prompt_items = orchestrator.load_prompt_items(_hp())
    # `gsm8k.yaml` reads the `train` split with no `data_dir`: TFDS uses its
    # default.
    self.assertEqual(data_source.call_args.kwargs["split"], "train")
    self.assertIsNone(data_source.call_args.kwargs["data_dir"])
    expected = []
    for index in range(2):
      prompt = gsm8k.build_prompt(f"Q{index}")
      expected.append({
          "prompt": prompt,
          "task_name": "gsm8k",
          "max_turns": 1,
          "metadata": {
              "answer": str(index),
              "env_config": {
                  "prompt": prompt,
                  "question": f"Q{index}",
                  "answer": str(index),
              },
              "episode_timeout": 600.0,
              "overlong_filter": False,
          },
      })
    # `gsm8k.yaml` shuffles, so the two records come in either order.
    self.assertCountEqual([prompt_items[0], prompt_items[1]], expected)
    # The records repeat (reshuffled) for as many steps as the run needs.
    self.assertCountEqual([prompt_items[2], prompt_items[3]], expected)

  def test_rollout_env_builds_from_env_config(self):
    with mock.patch.object(tfds, "data_source", return_value=_tfds_examples()):
      record = gsm8k.create_dataset(split="train", shuffle=False, seed=0)[0]
    assert record is not None
    # `Registry.get` raises `KeyError` for an unregistered name.
    env_class = registry.ENV_REGISTRY.get(gsm8k.GSM8K_ENV_NAME)
    # `RolloutManager` adds the group index and policy version.
    env = env_class(
        **record["metadata"]["env_config"], group_index=0, policy_version=0
    )
    observation, _ = env.reset()
    self.assertEqual(observation, {"prompts": gsm8k.build_prompt("Q0")})
    _, reward, done, info = env.step(
        "<reasoning>0.</reasoning><answer>\\boxed{0}</answer>"
    )
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(info["correct"])


class RewardFunctionsTest(absltest.TestCase):
  """`gsm8k_rewards`, loaded through `reward_functions`."""

  def test_reward_functions_load_only_the_gsm8k_reward(self):
    # Every public function of a `reward_functions` module is a reward, which
    # is why the reward is not in `gsm8k.py`.
    hp = resolve_config.initialize([
        "agentic_grpo_config.reward_mode=reward_fn",
        f"reward_functions=[{_REWARDS_MODULE}]",
        "tokenizer_config.tokenizer_path=/unused/tokenizer",
    ])
    reward_fns = hp.obtain_reward_fn()
    self.assertEqual(reward_fns, [gsm8k_rewards.reward_fn])
    # The name is the per-function reward metric: `rewards/train/reward_fn`,
    # as with the legacy launcher's reward function.
    self.assertEqual(reward_fns[0].__name__, "reward_fn")

  def test_gsm8k_yaml_scores_on_the_orchestrator(self):
    hp = _hp(
        "agentic_grpo_config.reward_mode=reward_fn",
        f"reward_functions=[{_REWARDS_MODULE}]",
    )
    self.assertEqual(hp.obtain_reward_fn(), [gsm8k_rewards.reward_fn])
    self.assertEqual(
        orchestrator.build_grpo_config(hp).reward_mode, "reward_fn"
    )

  def test_reward_fn_mode_needs_reward_functions(self):
    with self.assertRaisesRegex(ValueError, "reward_functions"):
      _hp("agentic_grpo_config.reward_mode=reward_fn")

  def test_reward_fn_matches_env_reward(self):
    completions = [
        "<reasoning>2+2=4.</reasoning><answer>\\boxed{4}</answer>",
        "<reasoning>2+2=5.</reasoning><answer>\\boxed{5}</answer>",
        "The answer is \\boxed{4}",
        "No answer here",
    ]
    rewards = gsm8k_rewards.reward_fn(
        prompts=["p"] * 4,
        completions=completions,
        answer=["4"] * 4,
        # The other record metadata the orchestrator passes along.
        episode_timeout=[600.0] * 4,
        overlong_filter=[False] * 4,
    )
    self.assertEqual(rewards, [1.0, 0.1, 0.5, 0.0])
    for completion, reward in zip(completions, rewards, strict=True):
      env = gsm8k.GSM8KEnv(prompts="p", answer="4")
      env.reset()
      self.assertEqual(env.step(completion)[1], reward)


if __name__ == "__main__":
  absltest.main()
