"""Abstract test cases defining contract tests for TrajectoryStore implementations."""

import abc

from absl.testing import parameterized
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


class ParameterizedABCMeta(type(parameterized.TestCase), abc.ABCMeta):
  """Combined metaclass resolving conflict between parameterized.TestCase and abc.ABCMeta."""


class TrajectoryReaderTestCase(
    trajectory_testing.TrajectoryTestCase, metaclass=ParameterizedABCMeta
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
            (trajectory_testing.METADATA_1, [trajectory_testing.STEP_1_1]),
            (
                trajectory_testing.METADATA_2,
                [
                    trajectory_testing.STEP_2_1,
                    trajectory_testing.STEP_2_2,
                    trajectory_testing.STEP_2_3,
                    trajectory_testing.STEP_2_4,
                    trajectory_testing.STEP_2_5,
                ],
            ),
        ],
    )

  def test_get_trajectories_metadata(self) -> None:
    """Tests that metadata for all stored trajectories is retrieved."""
    metas = self.reader.get_trajectories_metadata()
    self.assertCountEqual(
        metas,
        [trajectory_testing.METADATA_1, trajectory_testing.METADATA_2],
    )

  def test_get_trajectories_metadata_empty(self) -> None:
    """Tests that metadata retrieval on an empty store returns an empty list."""
    empty_reader = self._create_reader(initial_data=None)
    self.assertEmpty(empty_reader.get_trajectories_metadata())

  @parameterized.named_parameters(
      ("empty_list", [], []),
      (
          "single_trajectory",
          [trajectory_testing.TRAJECTORY_ID_1],
          [trajectory_testing.TRAJECTORY_1],
      ),
      (
          "multiple_trajectories",
          [
              trajectory_testing.TRAJECTORY_ID_1,
              trajectory_testing.TRAJECTORY_ID_2,
          ],
          [trajectory_testing.TRAJECTORY_1, trajectory_testing.TRAJECTORY_2],
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
    trajectory_testing.TrajectoryTestCase, metaclass=ParameterizedABCMeta
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
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_add_step_multiple_steps(self) -> None:
    """Tests that sequential steps are correctly appended to a trajectory."""
    self.writer.add_step(
        trajectory_testing.STEP_2_1, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_2, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_3, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_4, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_5, trajectory_testing.METADATA_2
    )
    self.writer.flush()

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_2])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_2])

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
      self.writer.add_step(trajectory_testing.STEP_1_1, meta)

  def test_add_step_multiple_trajectories(self) -> None:
    """Tests adding steps across multiple distinct trajectories."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_1, trajectory_testing.METADATA_2
    )
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertLen(metas, 2)

    (traj_1,) = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_1]
    )
    self.assertEqual(traj_1, trajectory_testing.TRAJECTORY_1)

    expected_traj_2_partial = trajectory_lib.Trajectory(
        **trajectory_testing.METADATA_2.model_dump(),
        steps=[trajectory_testing.STEP_2_1],
    )
    (traj_2,) = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_2]
    )
    self.assertEqual(traj_2, expected_traj_2_partial)

    self.writer.add_step(
        trajectory_testing.STEP_2_2, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_3, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_4, trajectory_testing.METADATA_2
    )
    self.writer.add_step(
        trajectory_testing.STEP_2_5, trajectory_testing.METADATA_2
    )
    self.writer.flush()

    trajs = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_1, trajectory_testing.TRAJECTORY_ID_2]
    )
    self.assertCountEqual(
        trajs,
        [trajectory_testing.TRAJECTORY_1, trajectory_testing.TRAJECTORY_2],
    )

  def test_add_step_overwrite_existing_step(self) -> None:
    """Tests that logging a step with an existing step_id updates the step."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.flush()

    updated_step = trajectory_testing.STEP_1_1.model_copy(deep=True)
    updated_step.message = "updated message"
    self.writer.add_step(updated_step, trajectory_testing.METADATA_1)
    self.writer.flush()

    (traj_1,) = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_1]
    )
    self.assertLen(traj_1.steps, 1)
    self.assertEqual(traj_1.steps[0].message, "updated message")

  def test_flush_empty(self) -> None:
    """Tests that calling flush on an empty store does not raise an error."""
    self.writer.flush()
    self.assertEmpty(self.reader.get_trajectories_metadata())

  def test_flush_idempotent(self) -> None:
    """Tests that multiple consecutive calls to flush are safe and idempotent."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.flush()
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_update_metadata(self) -> None:
    """Tests updating metadata for a trajectory."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.flush()

    updated_meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_1,
        agent=trajectory_testing.METADATA_1.agent,
        notes="Updated notes",
    )
    self.writer.update_metadata(updated_meta)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [updated_meta])

    (traj_1,) = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_1]
    )
    expected_traj_1 = trajectory_lib.Trajectory(
        **updated_meta.model_dump(),
        steps=[trajectory_testing.STEP_1_1],
    )
    self.assertEqual(traj_1, expected_traj_1)

  def test_add_step_snapshots_step_and_metadata(self) -> None:
    """Tests that mutating a step or metadata after logging does not alter the store."""
    meta = trajectory_testing.METADATA_1.model_copy(deep=True)
    step = trajectory_testing.STEP_1_1.model_copy(deep=True)
    self.writer.add_step(step, meta)

    step.message = "mutated after logging"
    meta.notes = "mutated after logging"
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_update_metadata_snapshots_metadata(self) -> None:
    """Tests that mutating metadata after update_metadata does not alter the store."""
    meta = trajectory_testing.METADATA_1.model_copy(deep=True)
    self.writer.update_metadata(meta)

    meta.notes = "mutated after logging"
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

  def test_close_persists_pending_writes(self) -> None:
    """Tests that close() drains pending writes without an explicit flush()."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.close()

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_close_is_idempotent(self) -> None:
    """Tests that closing an already closed writer is a no-op."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.close()
    self.writer.close()

    trajs = self.reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_update_metadata_standalone(self) -> None:
    """Tests updating metadata prior to adding any steps."""
    self.writer.update_metadata(trajectory_testing.METADATA_1)
    self.writer.flush()

    metas = self.reader.get_trajectories_metadata()
    self.assertEqual(metas, [trajectory_testing.METADATA_1])

    (traj_1,) = self.reader.get_trajectories(
        [trajectory_testing.TRAJECTORY_ID_1]
    )
    expected_traj_1 = trajectory_lib.Trajectory(
        **trajectory_testing.METADATA_1.model_dump(),
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


