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
    """Asserts that two Trajectory instances are equal, including nested steps and subagents."""
    self.assertIsInstance(
        actual, (trajectory_lib.Trajectory, trajectory_lib.TunixTrajectory)
    )
    self.assertIsInstance(
        expected, (trajectory_lib.Trajectory, trajectory_lib.TunixTrajectory)
    )
    self.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)
