from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import trajectory


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

  def test_basic_serialization_and_deserialization(self):
    data = {
        "schema_version": "ATIF-v1.7",
        "session_id": "test-session",
        "trajectory_id": "root-traj",
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Hello",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Hi there!",
                "reasoning_content": "Replying back",
            },
        ],
    }
    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertEqual(traj.schema_version, "ATIF-v1.7")
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[0].message, "Hello")

    serialized = traj.to_json_dict()
    self.assertEqual(serialized["session_id"], "test-session")
    self.assertEqual(
        serialized["steps"][1]["reasoning_content"], "Replying back"
    )

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


if __name__ == "__main__":
  absltest.main()
