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
"""Testing utilities and fixtures for trajectory tests."""

import datetime
from typing import Any, Final

from absl.testing import parameterized
from tunix.dist_rl.trajectory import trajectory as trajectory_lib


def to_atif_trajectory(
    trajectory: trajectory_lib.Trajectory[Any],
) -> trajectory_lib.Trajectory[trajectory_lib.Step]:
  """Projects a client Trajectory to the base ATIF Trajectory a store returns."""
  return trajectory_lib.Trajectory(
      **trajectory.to_atif_metadata().model_dump(),
      steps=[step.to_atif_step() for step in trajectory.steps],
  )


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
      actual: trajectory_lib.Step,
      expected: trajectory_lib.Step,
      msg: str | None = None,
  ) -> None:
    """Asserts that two Step instances (base ATIF or a subclass) are equal.

    Compares the serialized form, so NumPy array fields compare by value.

    Args:
      actual: The step under test.
      expected: The expected step; must be the same concrete type as `actual`.
      msg: Optional failure message.
    """
    self.assertEqual(type(actual), type(expected))
    self.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)

  def assertTrajectoryEqual(
      self,
      actual: trajectory_lib.Trajectory[Any],
      expected: trajectory_lib.Trajectory[Any],
      msg: str | None = None,
  ) -> None:
    """Asserts that two Trajectory instances (base ATIF or a subclass) are equal."""
    self.assertIsInstance(actual, trajectory_lib.Trajectory)
    self.assertIsInstance(expected, trajectory_lib.Trajectory)
    self.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)
