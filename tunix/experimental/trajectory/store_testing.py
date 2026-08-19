import abc
import dataclasses
import datetime
from typing import Any, Final

from absl.testing import parameterized
import numpy as np
import pydantic
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib


def assert_step_equal(
    test_case: parameterized.TestCase,
    actual: Any,
    expected: Any,
    msg: str | None = None,
) -> None:
  """Asserts that two Step instances (ATIF or Tunix) are equal."""
  test_case.assertEqual(type(actual), type(expected))
  if isinstance(actual, pydantic.BaseModel):
    test_case.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)
  elif dataclasses.is_dataclass(actual):
    for field in dataclasses.fields(actual):
      v1 = getattr(actual, field.name)
      v2 = getattr(expected, field.name)
      if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        np.testing.assert_array_equal(
            v1, v2, err_msg=msg or f"Field '{field.name}' mismatch"
        )
      else:
        test_case.assertEqual(
            v1, v2, msg=msg or f"Field '{field.name}' mismatch"
        )
  else:
    test_case.assertEqual(actual, expected, msg=msg)


def assert_trajectory_equal(
    test_case: parameterized.TestCase,
    actual: trajectory_lib.Trajectory,
    expected: trajectory_lib.Trajectory,
    msg: str | None = None,
) -> None:
  """Asserts that two Trajectory instances are equal, including nested steps and subagents."""
  test_case.assertIsInstance(actual, trajectory_lib.Trajectory)
  test_case.assertIsInstance(expected, trajectory_lib.Trajectory)
  test_case.assertEqual(actual.model_dump(), expected.model_dump(), msg=msg)

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
            (
                METADATA_2,
                [STEP_2_1, STEP_2_2, STEP_2_3, STEP_2_4, STEP_2_5],
            ),
        ],
    )

  def test_get_trajectories_metadata(self) -> None:
    """Tests that metadata for all stored trajectories is retrieved."""
    metas = self.reader.get_trajectories_metadata()
    self.assertCountEqual(metas, [METADATA_1, METADATA_2])

  def test_get_trajectories_metadata_empty(self) -> None:
    """Tests that metadata retrieval on an empty store returns an empty list."""
    empty_reader = self._create_reader(initial_data=None)
    self.assertEmpty(empty_reader.get_trajectories_metadata())

  @parameterized.named_parameters(
      ("empty_list", [], []),
      ("single_trajectory", [TRAJECTORY_ID_1], [TRAJECTORY_1]),
      (
          "multiple_trajectories",
          [TRAJECTORY_ID_1, TRAJECTORY_ID_2],
          [TRAJECTORY_1, TRAJECTORY_2],
      ),
  )
  def test_get_trajectories(
      self,
      trajectory_ids: list[str],
      expected_trajs: list[trajectory_lib.Trajectory],
  ) -> None:
    """Tests that full trajectories are retrieved by their IDs."""
    trajs = self.reader.get_trajectories(trajectory_ids)
    self.assertCountEqual(trajs, expected_trajs)

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
    self.writer.add_step(STEP_2_3, METADATA_2)
    self.writer.add_step(STEP_2_4, METADATA_2)
    self.writer.add_step(STEP_2_5, METADATA_2)
    self.writer.flush()

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_2])
    self.assertEqual(trajs, [TRAJECTORY_2])

  @parameterized.named_parameters(
      ("empty", ""),
      ("none", None),
  )
  def test_add_step_invalid_trajectory_id(
      self, trajectory_id: str | None
  ) -> None:
    """Tests that logging a step with an empty or None trajectory ID raises ValueError."""
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=trajectory_id,
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
    self.writer.add_step(STEP_2_3, METADATA_2)
    self.writer.add_step(STEP_2_4, METADATA_2)
    self.writer.add_step(STEP_2_5, METADATA_2)
    self.writer.flush()

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_1, TRAJECTORY_ID_2])
    self.assertCountEqual(trajs, [TRAJECTORY_1, TRAJECTORY_2])

  def test_flush_empty(self) -> None:
    """Tests that calling flush on an empty store does not raise an error."""
    self.writer.flush()
    self.assertEmpty(self.reader.get_trajectories_metadata())

  def test_flush_idempotent(self) -> None:
    """Tests that multiple consecutive calls to flush are safe and idempotent."""
    self.writer.add_step(STEP_1_1, METADATA_1)
    self.writer.flush()
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [METADATA_1])

    trajs = self.reader.get_trajectories([TRAJECTORY_ID_1])
    self.assertEqual(trajs, [TRAJECTORY_1])

  def test_update_metadata(self) -> None:
    """Tests updating metadata for a trajectory."""
    self.writer.add_step(STEP_1_1, METADATA_1)
    self.writer.flush()

    updated_meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=TRAJECTORY_ID_1,
        agent=METADATA_1.agent,
        notes="Updated notes",
    )
    self.writer.update_metadata(updated_meta)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [updated_meta])

    (traj_1,) = self.reader.get_trajectories([TRAJECTORY_ID_1])
    expected_traj_1 = trajectory_lib.Trajectory(
        **updated_meta.model_dump(),
        steps=[STEP_1_1],
    )
    self.assertEqual(traj_1, expected_traj_1)

  def test_update_metadata_standalone(self) -> None:
    """Tests updating metadata prior to adding any steps."""
    self.writer.update_metadata(METADATA_1)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [METADATA_1])

    (traj_1,) = self.reader.get_trajectories([TRAJECTORY_ID_1])
    expected_traj_1 = trajectory_lib.Trajectory(
        **METADATA_1.model_dump(),
        steps=[],
    )
    self.assertEqual(traj_1, expected_traj_1)

  @parameterized.named_parameters(
      ("empty", ""),
      ("none", None),
  )
  def test_update_metadata_invalid_trajectory_id(
      self, trajectory_id: str | None
  ) -> None:
    """Tests that updating metadata with an empty or None trajectory ID raises ValueError."""
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=trajectory_id,
        agent=trajectory_lib.Agent(name="writer_agent", version="2.0"),
    )
    with self.assertRaises(ValueError):
      self.writer.update_metadata(meta)
