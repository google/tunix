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

"""Tests for the Tunix RL extensions of the ATIF trajectory schema.

The client-agnostic extension machinery (packing into `extra`, rehydration,
`resolve_subclass`, paired trajectory classes) is covered by
`trajectory_test.py` against the `Example*` reference client; these tests cover
what is specific to the Tunix subclasses: their field sets, the prefill routing
invariants, `TunixTrajectory.add_step`, and lossless Tunix round trips.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pydantic
from tunix.dist_rl.trajectory import trajectory
from tunix.dist_rl.trajectory import trajectory_testing
from tunix.dist_rl.trajectory import tunix_trajectory
from tunix.dist_rl.trajectory import tunix_trajectory_testing


class TunixSchemaRegistrationTest(parameterized.TestCase):
  """Covers how the Tunix subclasses plug into the base ATIF extension hooks."""

  def test_tunix_models_share_the_tunix_extensions_key(self):
    for model_cls in (
        tunix_trajectory.TunixAgentStep,
        tunix_trajectory.TunixEnvStep,
        tunix_trajectory.TunixTrajectoryMetadata,
        tunix_trajectory.TunixTrajectory,
    ):
      with self.subTest(model_cls=model_cls.__name__):
        self.assertEqual(
            model_cls.EXTENSIONS_KEY, tunix_trajectory.TUNIX_EXTENSIONS_KEY
        )

  def test_pydantic_init_subclass_registers_tunix_metadata_fields(self):
    self.assertEqual(
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS[
            tunix_trajectory.TunixTrajectoryMetadata
        ],
        frozenset({
            "prompt_id",
            "group_index",
            "target_policy_versions",
            "status",
            "total_reward",
            "gamma",
            "masked_out",
            "hyperparams",
            "env_time",
            "reward_time",
        }),
    )
    self.assertNotIn(
        tunix_trajectory.TunixTrajectory,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )
    self.assertIs(
        trajectory.TrajectoryMetadata._PAIRED_TRAJECTORY_CLASSES[
            tunix_trajectory.TunixTrajectoryMetadata
        ],
        tunix_trajectory.TunixTrajectory,
    )

  def test_scalar_list_fields_populated_from_annotations(self):
    self.assertEqual(
        tunix_trajectory.TunixAgentStep.SCALAR_LIST_FIELDS,
        frozenset({"assistant_tokens", "assistant_masks", "logprobs"}),
    )
    self.assertEqual(
        tunix_trajectory.TunixEnvStep.SCALAR_LIST_FIELDS,
        frozenset({"env_tokens", "env_masks"}),
    )
    self.assertEqual(
        tunix_trajectory.TunixTrajectoryMetadata.SCALAR_LIST_FIELDS,
        frozenset({"target_policy_versions"}),
    )

  def test_resolve_subclass_resolves_tunix_models(self):
    tunix_atif_meta = (
        tunix_trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(tunix_atif_meta),
        tunix_trajectory.TunixTrajectoryMetadata,
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(
            tunix_trajectory_testing.TUNIX_TRAJECTORY_1
        ),
        tunix_trajectory.TunixTrajectoryMetadata,
    )


class TunixStepValidationTest(parameterized.TestCase):
  """Covers the Tunix-specific validators on `TunixAgentStep` / `TunixEnvStep`."""

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_class_rejects_user_source",
          step_cls=tunix_trajectory.TunixAgentStep,
          source=trajectory.Source.USER,
          expected_error=(
              "TunixAgentStep is only applicable when source is 'agent'"
          ),
      ),
      dict(
          testcase_name="env_class_rejects_agent_source",
          step_cls=tunix_trajectory.TunixEnvStep,
          source=trajectory.Source.AGENT,
          expected_error=(
              "TunixEnvStep is only applicable when source is 'system' or"
              " 'user'"
          ),
      ),
  )
  def test_step_with_mismatched_source_raises_value_error(
      self, step_cls, source, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      step_cls(step_id=0, source=source, message="mismatched")

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_start_and_num_context",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16)
          ),
          expected_error="either all None or all set",
      ),
      dict(
          testcase_name="missing_routed_experts",
          kwargs=dict(prefill_start=0, prefill_num_context=1),
          expected_error="either all None or all set",
      ),
      dict(
          testcase_name="negative_prefill_start",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16),
              prefill_start=-1,
              prefill_num_context=1,
          ),
          expected_error="prefill_start must be >= 0",
      ),
      dict(
          testcase_name="num_context_exceeds_rows",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16),
              prefill_start=0,
              prefill_num_context=3,
          ),
          expected_error="prefill_num_context must be in",
      ),
  )
  def test_agent_step_rejects_invalid_prefill_metadata(
      self, kwargs, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      tunix_trajectory.TunixAgentStep(
          step_id=0,
          source=trajectory.Source.AGENT,
          message="bad_prefill",
          **kwargs,
      )

  def test_agent_step_accepts_consistent_prefill_metadata(self):
    arr = np.arange(24, dtype=np.int16).reshape(3, 4, 2)
    step = tunix_trajectory.TunixAgentStep(
        step_id=0,
        source=trajectory.Source.AGENT,
        message="routed",
        prefill_routed_experts=arr,
        prefill_start=0,
        prefill_num_context=1,
    )

    restored = tunix_trajectory.TunixAgentStep.model_validate(
        step.model_dump(mode="json")
    )

    self.assertEqual(restored.prefill_start, 0)
    self.assertEqual(restored.prefill_num_context, 1)
    np.testing.assert_array_equal(restored.prefill_routed_experts, arr)


class TunixTrajectoryTest(tunix_trajectory_testing.TunixTrajectoryTestCase):
  """Covers `TunixTrajectory` step logging and metadata projection."""

  def test_add_step_assigns_zero_indexed_step_ids_by_source(self):
    traj = tunix_trajectory.TunixTrajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    self.assertEmpty(traj.steps)

    step1 = traj.add_step(source=trajectory.Source.USER, message="Start task")
    self.assertIsInstance(step1, tunix_trajectory.TunixEnvStep)
    self.assertEqual(step1.step_id, 0)
    self.assertLen(traj.steps, 1)
    self.assertEqual(traj.steps[0].message, "Start task")

    step2 = traj.add_step(
        source=trajectory.Source.AGENT,
        message="Working",
        reasoning_content="Logic here",
        policy_version=5,
    )
    self.assertIsInstance(step2, tunix_trajectory.TunixAgentStep)
    self.assertEqual(step2.step_id, 1)
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[1].reasoning_content, "Logic here")
    self.assertEqual(step2.policy_version, 5)

  def test_add_step_rejects_prefill_fields_for_non_agent_source(self):
    traj = tunix_trajectory.TunixTrajectory(
        session_id="s1",
        agent=trajectory.Agent(name="a", version="1"),
    )
    with self.assertRaisesRegex(
        ValueError, "only valid when source is 'agent'"
    ):
      traj.add_step(
          trajectory.Source.USER,
          "env",
          prefill_routed_experts=np.zeros((1, 2, 1), dtype=np.int16),
          prefill_start=0,
          prefill_num_context=0,
      )

  def test_get_metadata_returns_tunix_metadata_with_first_class_fields(self):
    meta = tunix_trajectory_testing.TUNIX_TRAJECTORY_1.get_metadata()

    self.assertIs(type(meta), tunix_trajectory.TunixTrajectoryMetadata)
    self.assertEqual(meta, tunix_trajectory_testing.TUNIX_METADATA_1)

  def test_create_trajectory_promotes_plain_atif_steps(self):
    meta = tunix_trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_promote",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    plain_steps = [
        trajectory.Step(
            step_id=1, source=trajectory.Source.USER, message="User prompt"
        ),
        trajectory.Step(
            step_id=2, source=trajectory.Source.AGENT, message="Agent response"
        ),
    ]

    traj = meta.create_trajectory(steps=plain_steps)

    self.assertIsInstance(traj, tunix_trajectory.TunixTrajectory)
    self.assertLen(traj.steps, 2)
    self.assertIsInstance(traj.steps[0], tunix_trajectory.TunixEnvStep)
    self.assertEqual(traj.steps[0].step_id, 0)
    self.assertIsInstance(traj.steps[1], tunix_trajectory.TunixAgentStep)
    self.assertEqual(traj.steps[1].step_id, 1)


class TunixAtifRoundTripTest(tunix_trajectory_testing.TunixTrajectoryTestCase):
  """Covers lossless projection of the Tunix fields through base ATIF."""

  def test_to_atif_step_packs_every_tunix_agent_field_into_extra(self):
    agent_step = tunix_trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={"extra": {"user_key": "val"}}
    )

    atif_agent_step = agent_step.to_atif_step()

    self.assertIs(type(atif_agent_step), trajectory.Step)
    self.assertEqual(atif_agent_step.step_id, agent_step.step_id + 1)
    self.assertEqual(
        atif_agent_step.extra,
        {
            "user_key": "val",
            tunix_trajectory.TUNIX_EXTENSIONS_KEY: {
                "assistant_tokens": [10, 20],
                "assistant_masks": [1, 1],
                "logprobs": [-0.5, -0.2],
                "policy_version": 3,
                "prefill_routed_experts": trajectory._serialize_int_3d_array(
                    [[[1, 2]], [[3, 4]], [[5, 6]]]
                ),
                "prefill_start": 0,
                "prefill_num_context": 1,
            },
        },
    )

  def test_to_atif_metadata_packs_every_tunix_metadata_field_into_extra(self):
    atif_metadata = tunix_trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()

    self.assertIs(type(atif_metadata), trajectory.TrajectoryMetadata)
    self.assertEqual(
        atif_metadata.extra,
        {
            "user_meta": "val",
            tunix_trajectory.TUNIX_EXTENSIONS_KEY: {
                "prompt_id": "p_1",
                "group_index": 2,
                "target_policy_versions": [2, 3],
                "status": "SUCCEEDED",
                "total_reward": 3.5,
                "gamma": 0.9,
                "masked_out": False,
                "hyperparams": {"temperature": 0.7},
                "env_time": {"step_0": 0.05},
                "reward_time": {"step_1": 0.02},
            },
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_step",
          step_cls=tunix_trajectory.TunixAgentStep,
          tunix_step=tunix_trajectory_testing.TUNIX_AGENT_STEP_1,
          array_field_names=(
              "assistant_tokens",
              "assistant_masks",
              "logprobs",
              "prefill_routed_experts",
          ),
      ),
      dict(
          testcase_name="env_step",
          step_cls=tunix_trajectory.TunixEnvStep,
          tunix_step=tunix_trajectory_testing.TUNIX_ENV_STEP_0,
          array_field_names=("env_tokens", "env_masks"),
      ),
  )
  def test_from_atif_step_round_trip_restores_tunix_step(
      self, step_cls, tunix_step, array_field_names
  ):
    rehydrated_step = step_cls.from_atif_step(tunix_step.to_atif_step())

    self.assertStepEqual(rehydrated_step, tunix_step)
    # `assertStepEqual` compares serialized dumps, where both ndarrays and
    # lists render as lists, so array payloads are checked element-wise here.
    for field_name in array_field_names:
      np.testing.assert_array_equal(
          getattr(rehydrated_step, field_name),
          getattr(tunix_step, field_name),
          err_msg=f"Field '{field_name}' mismatch",
      )

  def test_from_atif_trajectory_round_trip_restores_tunix_trajectory(self):
    tunix_traj = tunix_trajectory_testing.TUNIX_TRAJECTORY_1
    atif_trajectory = trajectory_testing.to_atif_trajectory(tunix_traj)

    rehydrated_trajectory = (
        tunix_trajectory.TunixTrajectory.from_atif_trajectory(atif_trajectory)
    )

    self.assertIs(type(rehydrated_trajectory), tunix_trajectory.TunixTrajectory)
    # Stores omit subagent trajectories, so the projection carries none and
    # the round trip restores every other field.
    self.assertTrajectoryEqual(
        rehydrated_trajectory,
        tunix_traj.model_copy(update={"subagent_trajectories": None}),
    )

  def test_paired_atif_trajectory_rehydrates_to_paired_tunix_trajectory(self):
    rehydrated_trajectory = (
        tunix_trajectory.TunixTrajectory.from_atif_trajectory(
            tunix_trajectory_testing.PAIRED_ATIF_TRAJECTORY
        )
    )

    self.assertTrajectoryEqual(
        rehydrated_trajectory,
        tunix_trajectory_testing.PAIRED_TUNIX_TRAJECTORY,
    )


class TunixAgentStepTokenLogprobLengthTest(parameterized.TestCase):
  """Covers the `assistant_tokens` / `logprobs` length validator."""

  def _make_step(
      self,
      *,
      assistant_tokens: np.ndarray | list[int] | None = None,
      logprobs: np.ndarray | list[float] | None = None,
  ) -> tunix_trajectory.TunixAgentStep:
    return tunix_trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="hi",
        assistant_tokens=assistant_tokens,
        logprobs=logprobs,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="matching_arrays",
          assistant_tokens=np.array([7, 8, 9], dtype=np.int32),
          logprobs=np.array([-0.5, -0.25, -0.125], dtype=np.float32),
      ),
      dict(
          testcase_name="matching_lists",
          assistant_tokens=[7, 8],
          logprobs=[-0.5, -0.25],
      ),
      dict(
          testcase_name="tokens_only",
          assistant_tokens=np.array([7, 8, 9]),
          logprobs=None,
      ),
      dict(
          testcase_name="logprobs_only",
          assistant_tokens=None,
          logprobs=np.array([-0.5]),
      ),
      dict(
          testcase_name="neither",
          assistant_tokens=None,
          logprobs=None,
      ),
  )
  def test_accepts_equal_lengths_or_missing_arrays(
      self,
      assistant_tokens: np.ndarray | list[int] | None,
      logprobs: np.ndarray | list[float] | None,
  ):
    step = self._make_step(
        assistant_tokens=assistant_tokens,
        logprobs=logprobs,
    )

    for actual, expected in (
        (step.assistant_tokens, assistant_tokens),
        (step.logprobs, logprobs),
    ):
      if expected is None:
        self.assertIsNone(actual)
      else:
        np.testing.assert_array_equal(actual, expected)

  def test_rejects_mismatched_lengths(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Length of assistant_tokens \(3\) must match length of logprobs"
        r" \(2\)",
    ):
      self._make_step(
          assistant_tokens=np.array([7, 8, 9]),
          logprobs=np.array([-0.5, -0.25]),
      )


if __name__ == "__main__":
  absltest.main()
