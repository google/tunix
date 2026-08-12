import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory

_SAMPLE_ATIF_PATH = os.path.join(
    os.path.dirname(trajectory.__file__), "testdata", "sample_atif_v1_7.json"
)


class SubagentTrajectoryRefTest(parameterized.TestCase):

  def test_subagent_trajectory_ref_valid_trajectory_id(self):
    # Valid: only trajectory_id
    ref1 = trajectory.SubagentTrajectoryRef(trajectory_id="sub-1")
    self.assertEqual(ref1.trajectory_id, "sub-1")

  def test_subagent_trajectory_ref_valid_trajectory_path(self):
    # Valid: only trajectory_path
    ref2 = trajectory.SubagentTrajectoryRef(trajectory_path="path/to/sub.json")
    self.assertEqual(ref2.trajectory_path, "path/to/sub.json")

  def test_subagent_trajectory_ref_invalid_no_id_or_path(self):
    # Invalid: neither trajectory_id nor trajectory_path
    with self.assertRaises(ValueError):
      trajectory.SubagentTrajectoryRef(session_id="session-1")


class StepTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("metrics", "metrics", trajectory.Metrics(prompt_tokens=10)),
      (
          "tool_calls",
          "tool_calls",
          [trajectory.ToolCall(tool_call_id="c1", function_name="f1")],
      ),
      ("reasoning_effort", "reasoning_effort", 1.0),
      ("model_name", "model_name", "dummy_value"),
      ("reasoning_content", "reasoning_content", "dummy_value"),
  )
  def test_validate_agent_only_fields(self, field_name, value):
    # Invalid: non-agent step containing agent-only field
    kwargs = {
        "step_id": 1,
        "source": trajectory.Source.USER,
        "message": "Hello",
        field_name: value,
    }
    with self.assertRaises(ValueError):
      trajectory.Step(**kwargs)

  @parameterized.named_parameters(
      ("metrics", "metrics", trajectory.Metrics(prompt_tokens=10)),
      ("reasoning_effort", "reasoning_effort", 1.0),
      ("reasoning_content", "reasoning_content", "dummy_value"),
      ("model_name", "model_name", "dummy_value"),
  )
  def test_validate_llm_call_count_zero_prohibits_llm_fields(
      self, field_name, value
  ):
    # Invalid: agent step with llm_call_count=0 containing LLM fields
    kwargs = {
        "step_id": 1,
        "source": trajectory.Source.AGENT,
        "message": "Deterministic action",
        "llm_call_count": 0,
        field_name: value,
    }
    with self.assertRaises(ValueError):
      trajectory.Step(**kwargs)

  def test_validate_llm_call_count_zero_allows_non_llm_fields(self):
    # Valid: agent step with llm_call_count=0 without LLM-specific fields.
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Deterministic action",
        llm_call_count=0,
    )
    self.assertEqual(step.llm_call_count, 0)


class TrajectoryTest(parameterized.TestCase):
  sample_atif_trajectory: trajectory.Trajectory

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(_SAMPLE_ATIF_PATH, "r", encoding="utf-8") as f:
      cls.sample_atif_trajectory = trajectory.Trajectory.from_json_dict(
          json.load(f)
      )

  def test_basic_serialization_and_deserialization(self):
    traj = self.sample_atif_trajectory
    self.assertEqual(traj.schema_version, "ATIF-v1.7")
    self.assertEqual(traj.session_id, "session-123")
    self.assertEqual(traj.trajectory_id, "traj-456")
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[0].message, "List directory contents")

    serialized = traj.to_json_dict()
    reloaded = trajectory.Trajectory.from_json_dict(serialized)
    store_testing.assert_trajectory_equal(self, reloaded, traj)

  def test_dynamic_step_logging(self):
    traj = trajectory.Trajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    self.assertEmpty(traj.steps)

    step1 = traj.add_step(source=trajectory.Source.USER, message="Start task")
    self.assertEqual(step1.step_id, 1)
    self.assertLen(traj.steps, 1)
    self.assertEqual(traj.steps[0].message, "Start task")

    step2 = traj.add_step(
        source=trajectory.Source.AGENT,
        message="Working",
        reasoning_content="Logic here",
    )
    self.assertEqual(step2.step_id, 2)
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[1].reasoning_content, "Logic here")

  def test_observation_and_metrics_serialization(self):
    data = {
        "schema_version": "ATIF-v1.7",
        "session_id": "test-session",
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{
            "step_id": 1,
            "source": "agent",
            "message": "Call tool",
            "tool_calls": [{
                "tool_call_id": "call-1",
                "function_name": "calculator",
                "arguments": {"expr": "2+2"},
            }],
            "observation": {
                "results": [{
                    "source_call_id": "call-1",
                    "content": "4",
                }]
            },
            "metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "cost_usd": 0.0001,
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "logprobs": [-0.1, -0.2],
                "extra": {"latency": 0.5},
            },
        }],
        "final_metrics": {
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0001,
            "total_steps": 1,
            "extra": {"overall_latency": 0.5},
        },
    }

    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertLen(traj.steps, 1)
    step = traj.steps[0]
    self.assertEqual(step.message, "Call tool")
    self.assertEqual(step.tool_calls[0].tool_call_id, "call-1")
    self.assertEqual(step.observation.results[0].content, "4")
    self.assertEqual(step.metrics.prompt_tokens, 10)
    self.assertEqual(step.metrics.extra["latency"], 0.5)
    self.assertEqual(traj.final_metrics.total_steps, 1)

    serialized = traj.to_json_dict()
    self.assertEqual(serialized["final_metrics"]["total_steps"], 1)
    self.assertEqual(
        serialized["steps"][0]["observation"]["results"][0]["content"], "4"
    )
    self.assertEqual(serialized["steps"][0]["metrics"]["prompt_tokens"], 10)

  def test_add_step_with_observation_and_metrics(self):
    traj = trajectory.Trajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    obs = trajectory.Observation(
        results=[
            trajectory.ObservationResult(
                source_call_id="call-1", content="result"
            )
        ]
    )
    metrics = trajectory.Metrics(prompt_tokens=20, completion_tokens=10)

    step = traj.add_step(
        source="agent",
        message="Running...",
        observation=obs,
        metrics=metrics,
    )

    self.assertEqual(step.step_id, 1)
    self.assertEqual(step.observation.results[0].content, "result")
    self.assertEqual(step.metrics.prompt_tokens, 20)

  def test_add_step_with_all_optional_fields(self):
    traj = trajectory.Trajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    step = traj.add_step(
        source=trajectory.Source.AGENT,
        message="Running...",
        model_name="gpt-4",
        reasoning_effort=1.5,
        is_copied_context=True,
        llm_call_count=2,
        extra={"key": "val"},
    )
    expected_data = {
        "step_id": 1,
        "source": "agent",
        "message": "Running...",
        "model_name": "gpt-4",
        "reasoning_effort": 1.5,
        "is_copied_context": True,
        "llm_call_count": 2,
        "extra": {"key": "val"},
    }
    self.assertIsNotNone(step.timestamp)
    actual_data = step.model_dump(
        exclude={"timestamp"}, exclude_none=True, mode="json"
    )
    self.assertDictEqual(actual_data, expected_data)

  def test_validate_step_ids(self):
    # Invalid: non-sequential step IDs
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "First"},
            {"step_id": 3, "source": "agent", "message": "Third"},
        ],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_unordered_steps_are_sorted(self):
    # Valid: out-of-order steps are sorted by step_id
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {"step_id": 2, "source": "agent", "message": "Second"},
            {"step_id": 1, "source": "user", "message": "First"},
        ],
    }
    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertEqual(traj.steps[0].step_id, 1)
    self.assertEqual(traj.steps[1].step_id, 2)

  def test_validate_embedded_subagent_missing_trajectory_id(self):
    # Invalid: missing trajectory_id on embedded subagent
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [{
            "agent": {"name": "sub-agent", "version": "1.0"},
            "steps": [{"step_id": 1, "source": "agent", "message": "Sub"}],
        }],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_validate_embedded_subagent_duplicate_trajectory_id(self):
    # Invalid: duplicate trajectory_id on embedded subagents
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [
            {
                "trajectory_id": "dup-id",
                "agent": {"name": "sub-1", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 1"}
                ],
            },
            {
                "trajectory_id": "dup-id",
                "agent": {"name": "sub-2", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 2"}
                ],
            },
        ],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_validate_embedded_subagent_unique_trajectory_id(self):
    # Valid: unique trajectory_id on embedded subagents
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [
            {
                "trajectory_id": "sub-1",
                "agent": {"name": "sub-1", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 1"}
                ],
            },
            {
                "trajectory_id": "sub-2",
                "agent": {"name": "sub-2", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 2"}
                ],
            },
        ],
    }
    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertLen(traj.subagent_trajectories, 2)

  def test_get_metadata(self):
    traj = self.sample_atif_trajectory
    traj.add_step(source=trajectory.Source.USER, message="Hello")

    meta = traj.get_metadata()
    self.assertIsInstance(meta, trajectory.TrajectoryMetadata)
    self.assertNotIsInstance(meta, trajectory.Trajectory)
    self.assertEqual(meta.trajectory_id, "traj-456")
    self.assertFalse(hasattr(meta, "steps"))
    self.assertFalse(hasattr(meta, "subagent_trajectories"))

  def test_step_initialization_with_rl_fields(self):
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Running bash command",
        reasoning_content="Thought process",
        assistant_tokens=np.array([10, 20]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.1, -0.2]),
        mc_return=1.5,
        extra={"custom_key": "custom_val"},
    )
    self.assertEqual(step.step_id, 1)
    self.assertEqual(step.source, trajectory.Source.AGENT)
    self.assertEqual(step.message, "Running bash command")
    self.assertEqual(step.reasoning_content, "Thought process")
    np.testing.assert_array_equal(step.assistant_tokens, np.array([10, 20]))
    np.testing.assert_array_equal(step.assistant_masks, np.array([1, 1]))
    np.testing.assert_array_equal(step.logprobs, np.array([-0.1, -0.2]))
    self.assertEqual(step.mc_return, 1.5)
    self.assertEqual(step.extra, {"custom_key": "custom_val"})
    self.assertIsNone(step.reward)
    self.assertIsNone(step.done)

  def test_step_env_fields(self):
    step = trajectory.Step(
        step_id=2,
        source=trajectory.Source.SYSTEM,
        message="Observation result",
        reward=1.0,
        done=True,
        env_tokens=np.array([100]),
        env_masks=np.array([1]),
    )
    self.assertEqual(step.reward, 1.0)
    self.assertTrue(step.done)
    np.testing.assert_array_equal(step.env_tokens, np.array([100]))
    np.testing.assert_array_equal(step.env_masks, np.array([1]))
    self.assertIsNone(step.assistant_tokens)
    self.assertIsNone(step.mc_return)

  def test_step_json_serialization_and_deserialization(self):
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Thinking",
        assistant_tokens=np.array([10, 20]),
        logprobs=np.array([-0.5, -0.3]),
        mc_return=2.0,
    )
    json_str = step.model_dump_json(exclude_none=True)
    loaded_dict = json.loads(json_str)
    self.assertEqual(loaded_dict["assistant_tokens"], [10, 20])
    self.assertEqual(loaded_dict["logprobs"], [-0.5, -0.3])
    self.assertEqual(loaded_dict["mc_return"], 2.0)

    reloaded_step = trajectory.Step.model_validate_json(json_str)
    self.assertEqual(reloaded_step.assistant_tokens, [10, 20])
    self.assertEqual(reloaded_step.logprobs, [-0.5, -0.3])
    self.assertEqual(reloaded_step.mc_return, 2.0)

  def test_step_equality(self):
    step1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 2]),
    )
    step2 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 2]),
    )
    step3 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 3]),
    )
    store_testing.assert_step_equal(self, step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())

  def test_step_equality_with_extra_numpy_arrays(self):
    step1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 4])},
        },
    )
    step2 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 4])},
        },
    )
    step3 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 5])},
        },
    )
    step4 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"arr": np.array([1, 2])},
    )
    step5 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra=None,
    )
    store_testing.assert_step_equal(self, step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())
    self.assertNotEqual(step1.model_dump(), step4.model_dump())
    self.assertNotEqual(step1.model_dump(), step5.model_dump())

  def test_step_equality_with_nested_tool_calls_and_observations(self):
    step1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 2])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    step2 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 2])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    step3 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 99])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    store_testing.assert_step_equal(self, step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())


if __name__ == "__main__":
  absltest.main()
