# Copyright 2026 Google LLC
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

"""Tests for token_dict."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import token_dict

_LAYERS = 3
_TOP_K = 2


def _routing(length: int, offset: int) -> np.ndarray:
  return (
      np.arange(length * _LAYERS * _TOP_K, dtype=np.int16).reshape(
          length, _LAYERS, _TOP_K
      )
      + offset
  )


def _make_trajectory(
    *, routed: bool = True, exact_token_continuity: bool = False
) -> agent_types.Trajectory:
  """Builds a finished two-turn trajectory with non-default dtypes."""
  steps = [
      agent_types.Step(
          chat_completions=[{"role": "user", "content": "q"}],
          model_response="a1",
          observation="obs1",
          reward=0.0,
          info={"turn": 0},
          assistant_tokens=np.array([11, 12, 13], dtype=np.int32),
          assistant_masks=np.array([1, 1, 0], dtype=np.int32),
          logprobs=np.array([-0.1, -0.2, 0.0], dtype=np.float32),
          env_tokens=np.array([21, 22], dtype=np.int32),
          env_masks=np.array([0, 0], dtype=np.int32),
          assistant_routed_experts=_routing(3, 0) if routed else None,
          env_routed_experts=_routing(2, 100) if routed else None,
      ),
      agent_types.Step(
          model_response="a2",
          observation="obs2",
          reward=1.5,
          done=True,
          assistant_tokens=np.array([14, 15], dtype=np.int32),
          assistant_masks=np.array([1, 1], dtype=np.int32),
          logprobs=None,
          assistant_routed_experts=_routing(2, 200) if routed else None,
      ),
  ]
  prompt_tokens = np.array([1, 2, 3, 4], dtype=np.int32)
  return agent_types.Trajectory(
      task={
          "prompts": ["q"],
          "answer": None,
          "ids": np.array([7, 8], dtype=np.int64),
          "policy_version": 4,
      },
      steps=steps,
      reward=1.5,
      status=agent_types.TrajectoryStatus.SUCCEEDED,
      env_time={"reset_latency": 0.25, "step_latency": [0.5, 0.75]},
      reward_time={"reward_latency": 0.125},
      prompt_tokens=prompt_tokens,
      prompt_length=4 if exact_token_continuity else None,
      prompt_routed_experts=_routing(4, 300) if routed else None,
  )


def _make_context(
    *, masked_out: bool = False, exact_token_continuity: bool = False
) -> token_dict.TokenDictContext:
  return token_dict.TokenDictContext(
      chat_completions=[
          {"role": "user", "content": "q"},
          {"role": "assistant", "content": "a1"},
          {"role": "user", "content": "obs1"},
          {"role": "assistant", "content": "a2"},
      ],
      policy_version=4,
      group_id=9,
      masked_out=masked_out,
      exact_token_continuity=exact_token_continuity,
  )


class TokenDictTestBase(parameterized.TestCase):

  def assertTreeEqual(self, actual: Any, expected: Any, path: str = "") -> None:
    """Asserts equal values, types, and array dtypes/shapes recursively."""
    if isinstance(expected, np.ndarray):
      self.assertIsInstance(actual, np.ndarray, path)
      self.assertEqual(actual.dtype, expected.dtype, path)
      self.assertEqual(actual.shape, expected.shape, path)
      np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
      self.assertIsInstance(actual, dict, path)
      self.assertEqual(list(actual), list(expected), path)
      for key in expected:
        self.assertTreeEqual(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
      self.assertIsInstance(actual, (list, tuple), path)
      self.assertLen(actual, len(expected), path)
      for i, (a, e) in enumerate(zip(actual, expected)):
        self.assertTreeEqual(a, e, f"{path}[{i}]")
    else:
      self.assertEqual(type(actual), type(expected), path)
      self.assertEqual(actual, expected, path)


class BuildTokenDictTest(TokenDictTestBase):

  def test_flattens_prompt_and_turns(self):
    result = token_dict.build_token_dict(_make_trajectory(), _make_context())

    self.assertEqual(
        list(result),
        [
            "conversation_text",
            "prompt_tokens",
            "conversation_tokens",
            "conversation_masks",
            "status",
            "trajectory_reward",
            "env_time",
            "reward_time",
            "old_logprobs",
            "routed_experts",
            "policy_version",
            "original_input",
            "group_id",
        ],
    )
    np.testing.assert_array_equal(
        result["conversation_tokens"], [11, 12, 13, 21, 22, 14, 15]
    )
    np.testing.assert_array_equal(
        result["conversation_masks"], [1, 1, 0, 0, 0, 1, 1]
    )
    self.assertEqual(result["old_logprobs"].shape, (7,))
    self.assertEqual(result["routed_experts"].shape, (11, _LAYERS, _TOP_K))
    np.testing.assert_array_equal(
        result["routed_experts"][:4], _routing(4, 300)
    )
    self.assertEqual(result["status"], "SUCCEEDED")
    self.assertEqual(result["group_id"], 9)

  def test_masked_out_zeroes_masks(self):
    result = token_dict.build_token_dict(
        _make_trajectory(), _make_context(masked_out=True)
    )
    self.assertFalse(np.any(result["conversation_masks"]))

  def test_reports_prompt_length_and_trajectory_id(self):
    result = token_dict.build_token_dict(
        _make_trajectory(exact_token_continuity=True),
        _make_context(exact_token_continuity=True),
        trajectory_id="abc",
    )
    self.assertEqual(result["prompt_length"], 4)
    self.assertEqual(result[token_dict.TRAJECTORY_ID_KEY], "abc")

  def test_missing_step_routing_raises(self):
    trajectory = _make_trajectory()
    trajectory.steps[1].assistant_routed_experts = None
    with self.assertRaisesRegex(ValueError, "missing assistant_routed_experts"):
      token_dict.build_token_dict(trajectory, _make_context())


class StoreRecordTest(TokenDictTestBase):

  def _stores(self):
    metadata_cls = trajectory_lib.TunixTrajectoryMetadata
    return {
        "memory": in_memory_store.InMemoryTrajectoryStore(
            metadata_cls=metadata_cls
        ),
        "file": file_store.FileTrajectoryStore(
            root_dir=self.create_tempdir().full_path,
            run_id="run",
            metadata_cls=metadata_cls,
        ),
    }

  @parameterized.product(
      backend=["memory", "file"],
      routed=[True, False],
      masked_out=[True, False],
      exact_token_continuity=[True, False],
  )
  def test_store_round_trip_rebuilds_identical_dict(
      self, backend, routed, masked_out, exact_token_continuity
  ):
    store = self._stores()[backend]
    trajectory = _make_trajectory(
        routed=routed, exact_token_continuity=exact_token_continuity
    )
    context = _make_context(
        masked_out=masked_out, exact_token_continuity=exact_token_continuity
    )
    expected = token_dict.build_token_dict(
        trajectory, context, trajectory_id="traj-1"
    )

    token_dict.write_trajectory(
        store, trajectory, context, trajectory_id="traj-1"
    )
    store.flush()
    (record,) = store.get_trajectories(["traj-1"])
    rebuilt_trajectory, rebuilt_context = token_dict.from_store_record(record)
    rebuilt = token_dict.build_token_dict(
        rebuilt_trajectory, rebuilt_context, trajectory_id="traj-1"
    )

    self.assertTreeEqual(rebuilt, expected)
    store.close()

  def test_list_prompt_tokens_round_trip(self):
    store = in_memory_store.InMemoryTrajectoryStore(
        metadata_cls=trajectory_lib.TunixTrajectoryMetadata
    )
    trajectory = _make_trajectory(routed=False)
    trajectory.prompt_tokens = [1, 2, 3, 4]
    context = _make_context()
    expected = token_dict.build_token_dict(trajectory, context)

    token_dict.write_trajectory(store, trajectory, context, trajectory_id="t")
    rebuilt = token_dict.build_token_dict(
        *token_dict.from_store_record(store.get_trajectories(["t"])[0])
    )

    self.assertTreeEqual(rebuilt, expected)

  def test_record_layout(self):
    metadata, steps = token_dict.to_store_record(
        _make_trajectory(), _make_context(), trajectory_id="t"
    )

    self.assertEqual([step.step_id for step in steps], [0, 1, 2, 3, 4])
    self.assertEqual(steps[0].source, trajectory_lib.Source.USER)
    self.assertIsInstance(steps[1], trajectory_lib.TunixAgentStep)
    self.assertIsInstance(steps[2], trajectory_lib.TunixEnvStep)
    self.assertEqual(
        metadata.extra[token_dict.TOKEN_DICT_EXTRA_KEY]["flatten_version"],
        token_dict.FLATTEN_VERSION,
    )

  def test_task_without_prompt_still_reserves_step_zero(self):
    trajectory = _make_trajectory()
    trajectory.task = {"question": "no prompts key"}
    _, steps = token_dict.to_store_record(
        trajectory, _make_context(), trajectory_id="t"
    )
    self.assertEqual([step.step_id for step in steps], [0, 1, 2, 3, 4])
    self.assertEqual(steps[0].message, "")

  def test_non_dict_task_raises(self):
    trajectory = _make_trajectory()
    trajectory.task = "a string task"
    with self.assertRaisesRegex(TypeError, "Only dict tasks"):
      token_dict.to_store_record(trajectory, _make_context(), trajectory_id="t")

  def test_flatten_version_mismatch_raises(self):
    store = in_memory_store.InMemoryTrajectoryStore(
        metadata_cls=trajectory_lib.TunixTrajectoryMetadata
    )
    metadata, steps = token_dict.to_store_record(
        _make_trajectory(), _make_context(), trajectory_id="t"
    )
    metadata.extra[token_dict.TOKEN_DICT_EXTRA_KEY]["flatten_version"] = -1
    for step in steps:
      store.add_step(step, metadata)

    with self.assertRaisesRegex(ValueError, "flatten version -1"):
      token_dict.from_store_record(store.get_trajectories(["t"])[0])

  def test_record_without_flatten_metadata_raises(self):
    store = in_memory_store.InMemoryTrajectoryStore(
        metadata_cls=trajectory_lib.TunixTrajectoryMetadata
    )
    metadata, steps = token_dict.to_store_record(
        _make_trajectory(), _make_context(), trajectory_id="t"
    )
    metadata.extra = None
    for step in steps:
      store.add_step(step, metadata)

    with self.assertRaisesRegex(ValueError, "not written by write_trajectory"):
      token_dict.from_store_record(store.get_trajectories(["t"])[0])


if __name__ == "__main__":
  absltest.main()
