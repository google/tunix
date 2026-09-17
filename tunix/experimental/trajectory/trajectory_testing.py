"""Testing utilities and fixtures for trajectory tests."""

import dataclasses
import datetime
from typing import Final

from absl.testing import parameterized
import numpy as np
import pydantic
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types

TEST_TIMESTAMP: Final[datetime.datetime] = datetime.datetime(
    2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

# A single trajectory with a single step.
TRAJECTORY_ID_1: Final[str] = "traj_1001"
METADATA_1: Final[trajectory_lib.TrajectoryMetadata] = (
    trajectory_lib.TrajectoryMetadata(
        trajectory_id=TRAJECTORY_ID_1,
        agent=trajectory_lib.Agent(name="agent_v1", version="1.0"),
    )
)
STEP_1_1: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=1,
    source=trajectory_lib.Source.AGENT,
    message="Hello world",
    timestamp=TEST_TIMESTAMP,
)
TRAJECTORY_1: Final[trajectory_lib.Trajectory] = trajectory_lib.Trajectory(
    **METADATA_1.model_dump(),
    steps=[STEP_1_1],
)

# A single trajectory with two steps.
TRAJECTORY_ID_2: Final[str] = "traj_1002"
METADATA_2: Final[trajectory_lib.TrajectoryMetadata] = (
    trajectory_lib.TrajectoryMetadata(
        trajectory_id=TRAJECTORY_ID_2,
        agent=trajectory_lib.Agent(name="agent_v2", version="2.0"),
    )
)
STEP_2_1: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=1,
    source=trajectory_lib.Source.USER,
    message="First step in traj 2",
    timestamp=TEST_TIMESTAMP,
)
STEP_2_2: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=2,
    source=trajectory_lib.Source.AGENT,
    message="Second step in traj 2",
    timestamp=TEST_TIMESTAMP,
)
STEP_2_3: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=3,
    source=trajectory_lib.Source.USER,
    message="Third step in traj 2",
    timestamp=TEST_TIMESTAMP,
)
STEP_2_4: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=4,
    source=trajectory_lib.Source.AGENT,
    message="Fourth step in traj 2",
    timestamp=TEST_TIMESTAMP,
)
STEP_2_5: Final[trajectory_lib.Step] = trajectory_lib.Step(
    step_id=5,
    source=trajectory_lib.Source.AGENT,
    message="Fifth step in traj 2",
    timestamp=TEST_TIMESTAMP,
)
TRAJECTORY_2: Final[trajectory_lib.Trajectory] = trajectory_lib.Trajectory(
    **METADATA_2.model_dump(),
    steps=[STEP_2_1, STEP_2_2, STEP_2_3, STEP_2_4, STEP_2_5],
)

TUNIX_ENV_STEP_0: Final[trajectory_lib.TunixEnvStep] = (
    trajectory_lib.TunixEnvStep(
        step_id=0,
        timestamp=TEST_TIMESTAMP,
        source=trajectory_lib.Source.USER,
        message="User prompt",
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(
                    source_call_id="call_env_0",
                    content="env observation content",
                    subagent_trajectory_ref=[
                        trajectory_lib.SubagentTrajectoryRef(
                            trajectory_id="sub_traj_ref_0",
                            session_id="sub_sess_ref_0",
                            extra={"ref_key": "ref_val"},
                        )
                    ],
                    extra={"obs_res_key": "obs_res_val"},
                )
            ]
        ),
        is_copied_context=False,
        llm_call_count=1,
        extra={"env_extra_key": "env_extra_val"},
        reward=1.0,
        done=False,
        env_tokens=np.array([1, 2]),
        env_masks=np.array([1, 1]),
    )
)

TUNIX_AGENT_STEP_1: Final[trajectory_lib.TunixAgentStep] = (
    trajectory_lib.TunixAgentStep(
        step_id=1,
        timestamp=TEST_TIMESTAMP,
        source=trajectory_lib.Source.AGENT,
        model_name="gemini-2.5-pro",
        reasoning_effort=0.8,
        message="Agent turn",
        reasoning_content="Reasoning trace",
        tool_calls=[
            trajectory_lib.ToolCall(
                tool_call_id="call_1",
                function_name="search",
                arguments={"query": "tunix"},
                extra={"tc_key": "tc_val"},
            )
        ],
        observation=trajectory_lib.Observation(
            results=[
                trajectory_lib.ObservationResult(
                    source_call_id="call_1",
                    content="search result",
                    subagent_trajectory_ref=[
                        trajectory_lib.SubagentTrajectoryRef(
                            trajectory_id="sub_traj_ref_1",
                            session_id="sub_sess_ref_1",
                            extra={"ref_key": "ref_val"},
                        )
                    ],
                    extra={"obs_res_key": "obs_res_val"},
                )
            ]
        ),
        metrics=trajectory_lib.Metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            cost_usd=0.01,
            prompt_token_ids=[1, 2],
            completion_token_ids=[3, 4],
            logprobs=[-0.1, -0.2],
            extra={"metric_key": "metric_val"},
        ),
        is_copied_context=False,
        llm_call_count=1,
        extra={"user_key": "val"},
        mc_return=2.5,
        assistant_tokens=np.array([10, 20]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.5, -0.2]),
        policy_version=3,
    )
)

TUNIX_METADATA_1: Final[trajectory_lib.TunixTrajectoryMetadata] = (
    trajectory_lib.TunixTrajectoryMetadata(
        schema_version="ATIF-v1.7",
        session_id="sess_01",
        trajectory_id="t_atif",
        agent=trajectory_lib.Agent(
            name="agent_v1",
            version="1.0",
            model_name="gemini-2.5-pro",
            tool_definitions=[{"name": "search", "description": "search tool"}],
            extra={"agent_key": "agent_val"},
        ),
        notes="Metadata projection test",
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_cached_tokens=20,
            total_cost_usd=0.01,
            total_steps=2,
            extra={"final_key": "final_val"},
        ),
        continued_trajectory_ref="traj_prev_1000",
        extra={"user_meta": "val"},
        prompt_id="p_1",
        group_index=2,
        target_policy_versions=[2, 3],
        status="COMPLETED",
        total_reward=3.5,
        hyperparams={"temperature": 0.7},
        env_time={"step_0": 0.05},
        reward_time={"step_1": 0.02},
    )
)

TUNIX_SUBAGENT_TRAJECTORY_1: Final[trajectory_lib.TunixTrajectory] = (
    trajectory_lib.TunixTrajectory(
        **TUNIX_METADATA_1.model_dump(exclude={"trajectory_id"}),
        trajectory_id="sub_traj_1",
        steps=[TUNIX_ENV_STEP_0, TUNIX_AGENT_STEP_1],
        subagent_trajectories=[],
    )
)

TUNIX_TRAJECTORY_1: Final[trajectory_lib.TunixTrajectory] = (
    trajectory_lib.TunixTrajectory(
        **TUNIX_METADATA_1.model_dump(),
        steps=[TUNIX_ENV_STEP_0, TUNIX_AGENT_STEP_1],
        subagent_trajectories=[TUNIX_SUBAGENT_TRAJECTORY_1],
    )
)


def make_metadata(
    trajectory_id: str | None = TRAJECTORY_ID_1,
    agent: trajectory_lib.Agent | None = None,
    session_id: str | None = None,
) -> trajectory_lib.TrajectoryMetadata:
  """Factory creating a TrajectoryMetadata instance with sensible defaults."""
  if agent is None:
    agent = trajectory_lib.Agent(name="test_agent", version="1.0")
  return trajectory_lib.TrajectoryMetadata(
      trajectory_id=trajectory_id,
      agent=agent,
      session_id=session_id,
  )


def make_step(
    step_id: int = 1,
    source: trajectory_lib.Source = trajectory_lib.Source.AGENT,
    message: str = "test message",
    timestamp: datetime.datetime | None = TEST_TIMESTAMP,
) -> trajectory_lib.Step:
  """Factory creating a Step instance with sensible defaults."""
  return trajectory_lib.Step(
      step_id=step_id,
      source=source,
      message=message,
      timestamp=timestamp,
  )


class TrajectoryTestCase(parameterized.TestCase):
  """Base TestCase providing custom assertion methods for trajectory objects."""

  def assertStepEqual(
      self,
      actual: trajectory_lib.Step | agent_types.Step,
      expected: trajectory_lib.Step | agent_types.Step,
      msg: str | None = None,
  ) -> None:
    """Asserts that two Step instances (ATIF or Tunix) are equal."""
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, pydantic.BaseModel) and isinstance(
        expected, pydantic.BaseModel
    ):
      self.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)
    elif dataclasses.is_dataclass(actual):
      for field in dataclasses.fields(actual):
        v1 = getattr(actual, field.name)
        v2 = getattr(expected, field.name)
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
          np.testing.assert_array_equal(
              v1, v2, err_msg=msg or f"Field '{field.name}' mismatch"
          )
        else:
          self.assertEqual(v1, v2, msg=msg or f"Field '{field.name}' mismatch")
    else:
      self.assertEqual(actual, expected, msg=msg)

  def assertTrajectoryEqual(
      self,
      actual: trajectory_lib.Trajectory | trajectory_lib.TunixTrajectory,
      expected: trajectory_lib.Trajectory | trajectory_lib.TunixTrajectory,
      msg: str | None = None,
  ) -> None:
    """Asserts that two Trajectory instances (ATIF or Tunix) are equal."""
    self.assertIsInstance(
        actual, (trajectory_lib.Trajectory, trajectory_lib.TunixTrajectory)
    )
    self.assertIsInstance(
        expected, (trajectory_lib.Trajectory, trajectory_lib.TunixTrajectory)
    )
    self.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)
