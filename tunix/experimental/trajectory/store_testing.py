"""Reusable test harness and contract tests for TrajectoryReader and TrajectoryWriter implementations."""

import abc
import datetime
from typing import Final

from absl.testing import parameterized
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib

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
TRAJECTORY_2: Final[trajectory_lib.Trajectory] = trajectory_lib.Trajectory(
    **METADATA_2.model_dump(),
    steps=[STEP_2_1, STEP_2_2],
)


class ParameterizedABCMeta(type(parameterized.TestCase), abc.ABCMeta):
  """Combined metaclass resolving conflict between parameterized.TestCase and abc.ABCMeta."""


class TrajectoryReaderTestCase(
    parameterized.TestCase, metaclass=ParameterizedABCMeta
):
  """Abstract test case defining contract tests for TrajectoryReader implementations.

  Subclasses must implement `_create_reader` to populate backend storage
  with initial test data and return a configured TrajectoryReader instance.
  """

  @abc.abstractmethod
  def _create_reader(
      self,
      initial_data: (
          list[
              tuple[
                  trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]
              ]
          ]
          | None
      ) = None,
  ) -> store.TrajectoryReader:
    """Factory method to create and populate a TrajectoryReader instance for each test."""

  def setUp(self) -> None:
    super().setUp()
    self.reader = self._create_reader(
        initial_data=[
            (METADATA_1, [STEP_1_1]),
            (METADATA_2, [STEP_2_1, STEP_2_2]),
        ],
    )

  def test_get_trajectories_metadata(self) -> None:
    """Tests that metadata for all stored trajectories is retrieved."""
    metas = self.reader.get_trajectories_metadata()
    self.assertCountEqual(metas, [METADATA_1, METADATA_2])

  def test_get_trajectories(self) -> None:
    """Tests that full trajectories are retrieved by their IDs."""
    trajs = self.reader.get_trajectories([TRAJECTORY_ID_1, TRAJECTORY_ID_2])
    self.assertCountEqual(trajs, [TRAJECTORY_1, TRAJECTORY_2])

  def test_get_trajectories_not_found(self) -> None:
    """Tests that loading a non-existent trajectory ID raises TrajectoryNotFoundError."""
    with self.assertRaises(store.TrajectoryNotFoundError):
      self.reader.get_trajectories(["non_existent_id"])


class TrajectoryWriterTestCase(
    parameterized.TestCase, metaclass=ParameterizedABCMeta
):
  """Abstract test case defining contract tests for TrajectoryWriter implementations.

  Subclasses must implement `_create_reader_and_writer` to create and return a
  tuple of (TrajectoryReader, TrajectoryWriter) for the backend under test.
  """

  @abc.abstractmethod
  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    """Factory method to create a TrajectoryReader and matching TrajectoryWriter for each test."""

  def setUp(self) -> None:
    super().setUp()
    self.reader, self.writer = self._create_reader_and_writer()

  def test_add_step(self) -> None:
    """Tests that a single step and its metadata are correctly added."""
    self.writer.add_step(STEP_1_1, METADATA_1)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [METADATA_1])

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_1])
    self.assertEqual(trajs, [TRAJECTORY_1])

  def test_add_step_multiple_steps(self) -> None:
    """Tests that sequential steps are correctly appended to a trajectory."""
    self.writer.add_step(STEP_2_1, METADATA_2)
    self.writer.add_step(STEP_2_2, METADATA_2)
    self.writer.flush()

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_2])
    self.assertEqual(trajs, [TRAJECTORY_2])

  def test_add_step_empty_trajectory_id(self) -> None:
    """Tests that logging a step with an empty trajectory ID raises ValueError."""
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="",
        agent=trajectory_lib.Agent(name="writer_agent", version="2.0"),
    )
    with self.assertRaises(ValueError):
      self.writer.add_step(STEP_1_1, meta)

  def test_add_step_multiple_trajectories(self) -> None:
    """Tests adding steps across multiple distinct trajectories."""
    self.writer.add_step(STEP_1_1, METADATA_1)
    self.writer.add_step(STEP_2_1, METADATA_2)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertLen(metas, 2)

    (traj_1,) = self.reader.get_trajectories([TRAJECTORY_ID_1])
    self.assertEqual(traj_1, TRAJECTORY_1)

    expected_traj_2_partial = trajectory_lib.Trajectory(
        **METADATA_2.model_dump(),
        steps=[STEP_2_1],
    )
    (traj_2,) = self.reader.get_trajectories([TRAJECTORY_ID_2])
    self.assertEqual(traj_2, expected_traj_2_partial)

    self.writer.add_step(STEP_2_2, METADATA_2)
    self.writer.flush()

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_1, TRAJECTORY_ID_2])
    self.assertCountEqual(trajs, [TRAJECTORY_1, TRAJECTORY_2])
