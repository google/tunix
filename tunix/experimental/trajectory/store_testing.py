"""Abstract test cases defining contract tests for TrajectoryStore implementations."""

import abc
from collections.abc import Sequence
from typing import Any, cast

from absl.testing import parameterized
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing

# `trajectory.py` uses `from __future__ import annotations`, so the field
# annotations of `Trajectory` are strings naming these symbols. Parametrizing
# it below (`Trajectory[Step]`) builds the concrete model in *this* module, so
# the names have to resolve here for `model_rebuild` to succeed.
from tunix.experimental.trajectory.trajectory import Step  # pylint: disable=g-importing-member,unused-import
from tunix.experimental.trajectory.trajectory import StepT  # pylint: disable=g-importing-member,unused-import
from tunix.experimental.trajectory.trajectory import Trajectory  # pylint: disable=g-importing-member,unused-import


class ParameterizedABCMeta(type(parameterized.TestCase), abc.ABCMeta):
  """Combined metaclass resolving conflict between parameterized.TestCase and abc.ABCMeta."""


class CustomMetadata(trajectory_lib.TrajectoryMetadata):
  """Custom metadata subclass shared by store tests."""

  EXTENSIONS_KEY = "_custom_extensions"

  custom_tag: str = ""

  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
  ) -> "CustomTrajectory":
    """Creates a CustomTrajectory, mirroring TunixTrajectoryMetadata."""
    return self._create_paired_trajectory(
        CustomTrajectory, steps, subagent_trajectories
    )


class CustomTrajectory(
    CustomMetadata, trajectory_lib.Trajectory[trajectory_lib.Step]
):
  """Trajectory paired with CustomMetadata."""

  subagent_trajectories: list["CustomTrajectory"] | None = None


CustomTrajectory.model_rebuild()


class ExtendedCustomMetadata(CustomMetadata):
  """Child metadata subclass extending CustomMetadata."""

  priority: int = 0

  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
  ) -> "ExtendedCustomTrajectory":
    """Creates an ExtendedCustomTrajectory."""
    return self._create_paired_trajectory(
        ExtendedCustomTrajectory, steps, subagent_trajectories
    )


class ExtendedCustomTrajectory(ExtendedCustomMetadata, CustomTrajectory):
  """Trajectory paired with ExtendedCustomMetadata."""

  subagent_trajectories: list["ExtendedCustomTrajectory"] | None = None


ExtendedCustomTrajectory.model_rebuild()


class OptionalOnlyMetadata(trajectory_lib.TrajectoryMetadata):
  """Metadata subclass whose only extension field defaults to None."""

  EXTENSIONS_KEY = "_optional_only_extensions"

  opt_tag: str | None = None

  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
  ) -> "OptionalOnlyTrajectory":
    """Creates an OptionalOnlyTrajectory."""
    return self._create_paired_trajectory(
        OptionalOnlyTrajectory, steps, subagent_trajectories
    )


class OptionalOnlyTrajectory(
    OptionalOnlyMetadata, trajectory_lib.Trajectory[trajectory_lib.Step]
):
  """Trajectory paired with OptionalOnlyMetadata."""

  subagent_trajectories: list["OptionalOnlyTrajectory"] | None = None


OptionalOnlyTrajectory.model_rebuild()


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

  @parameterized.named_parameters(
      (
          "all_trajectory_ids",
          None,
          [trajectory_testing.METADATA_1, trajectory_testing.METADATA_2],
      ),
      ("empty_list", [], []),
      (
          "single_trajectory",
          [trajectory_testing.TRAJECTORY_ID_1],
          [trajectory_testing.METADATA_1],
      ),
      (
          "multiple_trajectories",
          [
              trajectory_testing.TRAJECTORY_ID_1,
              trajectory_testing.TRAJECTORY_ID_2,
          ],
          [trajectory_testing.METADATA_1, trajectory_testing.METADATA_2],
      ),
      (
          "multiple_trajectories_reverse_order",
          [
              trajectory_testing.TRAJECTORY_ID_2,
              trajectory_testing.TRAJECTORY_ID_1,
          ],
          [trajectory_testing.METADATA_2, trajectory_testing.METADATA_1],
      ),
  )
  def test_get_trajectories_metadata(
      self,
      trajectory_ids: list[str] | None,
      expected_metas: list[trajectory_lib.TrajectoryMetadata],
  ) -> None:
    """Tests that metadata for trajectories is retrieved."""
    metas = self.reader.get_trajectories_metadata(trajectory_ids)
    if trajectory_ids is None:
      self.assertCountEqual(metas, expected_metas)
    else:
      self.assertEqual(metas, expected_metas)

  @parameterized.named_parameters(
      ("all_trajectory_ids", None),
      ("empty_list", []),
  )
  def test_get_trajectories_metadata_empty_store(
      self, trajectory_ids: list[str] | None
  ) -> None:
    """Tests that metadata retrieval on an empty store returns an empty list."""
    empty_reader = self._create_reader(initial_data=None)
    self.assertEmpty(empty_reader.get_trajectories_metadata(trajectory_ids))

  @parameterized.named_parameters(
      (
          "single_trajectory",
          [trajectory_testing.TRAJECTORY_ID_1],
      ),
      (
          "multiple_trajectories",
          [
              trajectory_testing.TRAJECTORY_ID_1,
              trajectory_testing.TRAJECTORY_ID_2,
          ],
      ),
  )
  def test_get_trajectories_metadata_empty_store_with_ids_raises(
      self,
      trajectory_ids: list[str],
  ) -> None:
    """Tests that querying explicit IDs on an empty store raises TrajectoryMetadataNotFoundError."""
    empty_reader = self._create_reader(initial_data=None)
    with self.assertRaisesRegex(
        store.TrajectoryMetadataNotFoundError,
        f"Trajectory metadata for ID '{trajectory_ids[0]}' not found.",
    ):
      empty_reader.get_trajectories_metadata(trajectory_ids)

  @parameterized.named_parameters(
      ("single_missing_id", ["non_existent_id"]),
      (
          "partial_missing_id",
          [trajectory_testing.TRAJECTORY_ID_1, "non_existent_id"],
      ),
  )
  def test_get_trajectories_metadata_not_found(
      self, trajectory_ids: list[str]
  ) -> None:
    """Tests that passing a non-existent trajectory ID raises TrajectoryMetadataNotFoundError."""
    with self.assertRaisesRegex(
        store.TrajectoryMetadataNotFoundError,
        "Trajectory metadata for ID 'non_existent_id' not found.",
    ):
      self.reader.get_trajectories_metadata(trajectory_ids)

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
      (
          "multiple_trajectories_reverse_order",
          [
              trajectory_testing.TRAJECTORY_ID_2,
              trajectory_testing.TRAJECTORY_ID_1,
          ],
          [trajectory_testing.TRAJECTORY_2, trajectory_testing.TRAJECTORY_1],
      ),
  )
  def test_get_trajectories(
      self,
      trajectory_ids: list[str],
      expected_trajs: list[trajectory_lib.Trajectory],
  ) -> None:
    """Tests that full trajectories are retrieved in requested ID order."""
    trajs = self.reader.get_trajectories(trajectory_ids)
    self.assertEqual(trajs, expected_trajs)

  @parameterized.named_parameters(
      ("single_missing_id", ["non_existent_id"]),
      (
          "partial_missing_id",
          [trajectory_testing.TRAJECTORY_ID_1, "non_existent_id"],
      ),
  )
  def test_get_trajectories_not_found(self, trajectory_ids: list[str]) -> None:
    """Tests that loading a non-existent trajectory ID raises TrajectoryNotFoundError."""
    with self.assertRaisesRegex(
        store.TrajectoryNotFoundError,
        "Trajectory with ID 'non_existent_id' not found.",
    ):
      self.reader.get_trajectories(trajectory_ids)


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
    """Tests that close() drains pending writes and allows subsequent reads."""
    self.writer.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    self.writer.close()

    self.assertEqual(
        self.reader.get_trajectories_metadata(),
        [trajectory_testing.METADATA_1],
    )
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

  def test_passing_trajectory_as_metadata_raises_type_error(self) -> None:
    """Verifies that passing a Trajectory instance as metadata raises TypeError."""
    for traj in (
        trajectory_testing.TRAJECTORY_1,
        trajectory_testing.TUNIX_TRAJECTORY_1,
    ):
      with self.assertRaisesRegex(
          TypeError, "Expected a TrajectoryMetadata instance"
      ):
        self.writer.update_metadata(traj)
      with self.assertRaisesRegex(
          TypeError, "Expected a TrajectoryMetadata instance"
      ):
        self.writer.add_step(trajectory_testing.STEP_1_1, traj)


class TrajectoryStoreMetadataClsTestCase(
    trajectory_testing.TrajectoryTestCase, metaclass=ParameterizedABCMeta
):
  """Abstract test case for TrajectoryStore metadata subclass handling.

  Subclasses must implement `_create_store` to return a `TrajectoryStore`
  instance.
  """

  @abc.abstractmethod
  def _create_store(self) -> store.TrajectoryStore[Any]:
    """Creates a TrajectoryStore instance."""

  def test_tunix_trajectory_with_step_zero(self) -> None:
    """Verifies storing and retrieving Tunix metadata and steps with step_id=0."""
    tunix_store = self._create_store()
    meta = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="tunix_1",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        status="RUNNING",
    )
    step0 = trajectory_lib.TunixEnvStep(
        step_id=0, source=trajectory_lib.Source.USER, message="prompt"
    )
    step1 = trajectory_lib.TunixAgentStep(
        step_id=1, source=trajectory_lib.Source.AGENT, message="response"
    )
    tunix_store.add_step(step0, meta)
    tunix_store.add_step(step1, meta)
    tunix_store.flush()

    metas = tunix_store.get_trajectories_metadata(["tunix_1"])
    self.assertLen(metas, 1)
    self.assertIsInstance(metas[0], trajectory_lib.TunixTrajectoryMetadata)
    self.assertEqual(metas[0].status, "RUNNING")

    trajs = tunix_store.get_trajectories(["tunix_1"])
    self.assertLen(trajs, 1)
    self.assertIsInstance(trajs[0], trajectory_lib.TunixTrajectory)
    self.assertEqual(trajs[0].steps[0].step_id, 0)
    self.assertEqual(trajs[0].steps[1].step_id, 1)
    self.assertIsInstance(trajs[0].steps[0], trajectory_lib.TunixEnvStep)
    self.assertIsInstance(trajs[0].steps[1], trajectory_lib.TunixAgentStep)

  def test_custom_metadata_and_trajectory_subclass(self) -> None:
    """Verifies the store supports custom metadata and trajectory types."""
    custom_store = self._create_store()
    meta = CustomMetadata(
        trajectory_id="custom_1",
        agent=trajectory_lib.Agent(name="custom_agent", version="1.0"),
        custom_tag="experiment_42",
    )
    step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.AGENT, message="custom step"
    )
    custom_store.add_step(step, meta)
    custom_store.flush()

    metas = custom_store.get_trajectories_metadata(["custom_1"])
    self.assertLen(metas, 1)
    self.assertIsInstance(metas[0], CustomMetadata)
    self.assertEqual(metas[0].custom_tag, "experiment_42")

    trajs = custom_store.get_trajectories(["custom_1"])
    self.assertLen(trajs, 1)
    self.assertIsInstance(trajs[0], CustomTrajectory)
    self.assertEqual(trajs[0].custom_tag, "experiment_42")
    self.assertLen(trajs[0].steps, 1)
    self.assertEqual(trajs[0].steps[0].message, "custom step")

  def test_hierarchical_custom_metadata_subclasses(self) -> None:
    """Verifies the store supports multi-level metadata and trajectory hierarchies."""
    ext_store = self._create_store()
    meta = ExtendedCustomMetadata(
        trajectory_id="ext_custom_1",
        agent=trajectory_lib.Agent(name="custom_agent", version="1.0"),
        custom_tag="inherited_tag",
        priority=5,
    )
    step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.AGENT, message="ext step"
    )
    ext_store.add_step(step, meta)
    ext_store.flush()

    metas = ext_store.get_trajectories_metadata(["ext_custom_1"])
    self.assertLen(metas, 1)
    self.assertIsInstance(metas[0], ExtendedCustomMetadata)
    self.assertIs(type(metas[0]), ExtendedCustomMetadata)
    self.assertEqual(metas[0].custom_tag, "inherited_tag")
    self.assertEqual(metas[0].priority, 5)

    trajs = ext_store.get_trajectories(["ext_custom_1"])
    self.assertLen(trajs, 1)
    self.assertIsInstance(trajs[0], ExtendedCustomTrajectory)
    self.assertIs(type(trajs[0]), ExtendedCustomTrajectory)
    self.assertEqual(trajs[0].custom_tag, "inherited_tag")
    self.assertEqual(trajs[0].priority, 5)
    self.assertLen(trajs[0].steps, 1)
    self.assertEqual(trajs[0].steps[0].message, "ext step")

  @parameterized.named_parameters(
      (
          "base_then_tunix",
          trajectory_testing.METADATA_1,
          trajectory_testing.TUNIX_METADATA_1,
      ),
      (
          "tunix_then_base",
          trajectory_testing.TUNIX_METADATA_1,
          trajectory_testing.METADATA_1,
      ),
      (
          "custom_then_tunix",
          CustomMetadata(
              trajectory_id="c_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
              custom_tag="tag",
          ),
          trajectory_testing.TUNIX_METADATA_1,
      ),
      (
          "parent_custom_then_child_custom",
          CustomMetadata(
              trajectory_id="c_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
              custom_tag="tag",
          ),
          ExtendedCustomMetadata(
              trajectory_id="c_2",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
              custom_tag="tag",
              priority=1,
          ),
      ),
      (
          "child_custom_then_parent_custom",
          ExtendedCustomMetadata(
              trajectory_id="c_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
              custom_tag="tag",
              priority=1,
          ),
          CustomMetadata(
              trajectory_id="c_2",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
              custom_tag="tag",
          ),
      ),
  )
  def test_mixing_metadata_types_across_writes_raises_type_error(
      self,
      first_meta: trajectory_lib.TrajectoryMetadata,
      second_meta: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Verifies a store binds on first write and rejects other metadata types."""
    store_instance = self._create_store()
    store_instance.update_metadata(first_meta)
    store_instance.flush()
    with self.assertRaisesRegex(TypeError, "is bound to metadata type"):
      store_instance.update_metadata(second_meta)
    with self.assertRaisesRegex(TypeError, "is bound to metadata type"):
      store_instance.add_step(trajectory_testing.STEP_1_1, second_meta)


class PersistentTrajectoryStoreMetadataClsTestCase(
    TrajectoryStoreMetadataClsTestCase, metaclass=ParameterizedABCMeta
):
  """Contract tests for stores that serialize metadata and read it back.

  `InMemoryTrajectoryStore` holds the written `TrajectoryMetadata` instance
  directly,
  so only backends that persist to base ATIF and rehydrate on read can open a
  second store against the same run.
  """

  def test_persistent_store_reads_subclass_and_rejects_mixed_stored_types(
      self,
  ) -> None:
    """Verifies persistent readers rehydrate subclasses and reject mixed runs."""
    tunix_writer = self._create_store()
    tunix_writer.add_step(
        trajectory_lib.TunixEnvStep(
            step_id=0, source=trajectory_lib.Source.USER, message="prompt"
        ),
        trajectory_testing.TUNIX_METADATA_1,
    )
    tunix_writer.flush()

    base_writer = self._create_store()
    base_writer.add_step(
        trajectory_testing.STEP_1_1,
        trajectory_testing.METADATA_1,
    )
    base_writer.flush()

    reader = self._create_store()
    base_id = trajectory_testing.TRAJECTORY_ID_1
    (read_meta,) = reader.get_trajectories_metadata(["t_atif"])
    self.assertEqual(read_meta, trajectory_testing.TUNIX_METADATA_1)
    (read_traj,) = reader.get_trajectories(["t_atif"])
    self.assertIsInstance(read_traj, trajectory_lib.TunixTrajectory)

    with self.assertRaisesRegex(TypeError, "is bound to metadata type"):
      reader.get_trajectories_metadata([base_id])
    with self.assertRaisesRegex(TypeError, "is bound to metadata type"):
      reader.get_trajectories([base_id])

  def test_persistent_store_round_trips_optional_only_metadata_when_none(
      self,
  ) -> None:
    """Verifies optional-only metadata subclasses round-trip when fields are None."""
    writer = self._create_store()
    meta = OptionalOnlyMetadata(
        trajectory_id="persisted_opt_none",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        opt_tag=None,
    )
    step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.USER, message="prompt"
    )
    writer.add_step(step, meta)
    writer.flush()

    reader = self._create_store()
    (read_meta,) = reader.get_trajectories_metadata(["persisted_opt_none"])
    self.assertIs(type(read_meta), OptionalOnlyMetadata)
    self.assertIsNone(read_meta.opt_tag)
    (read_traj,) = reader.get_trajectories(["persisted_opt_none"])
    self.assertIs(type(read_traj), OptionalOnlyTrajectory)


class TrajectoryStoreConfigTestCase(
    parameterized.TestCase, metaclass=ParameterizedABCMeta
):
  """Abstract test case defining contract tests for TrajectoryStore configs.

  Subclasses must implement `_create_config` to return a config dict, without
  secrets, that `TrajectoryStore.from_config` accepts for their backend.
  """

  @abc.abstractmethod
  def _create_config(self) -> dict[str, Any]:
    """Returns a secret-free config dict that selects this backend."""

  def _build_store(self, config: dict[str, Any]) -> store.TrajectoryStore:
    built = store.TrajectoryStore.from_config(config)
    if built is None:
      self.fail("from_config returned None for an enabled config.")
    self.addCleanup(built.close)
    return built

  def test_to_config_returns_the_config_it_was_built_from(self) -> None:
    config = self._create_config()

    built = self._build_store(config)

    self.assertEqual(built.to_config(), config)

  def test_from_config_round_trips_through_to_config(self) -> None:
    original = self._build_store(self._create_config())

    rebuilt = self._build_store(original.to_config())

    self.assertIsInstance(rebuilt, type(original))
    self.assertEqual(rebuilt.to_config(), original.to_config())

  @parameterized.named_parameters(
      ("base", trajectory_lib.TrajectoryMetadata),
      ("tunix", trajectory_lib.TunixTrajectoryMetadata),
      ("custom", CustomMetadata),
      ("extended_custom", ExtendedCustomMetadata),
      ("optional_only", OptionalOnlyMetadata),
  )
  def test_from_config_round_trip_preserves_metadata_type(
      self, metadata_cls: type[trajectory_lib.TrajectoryMetadata]
  ) -> None:
    """Verifies a store rebuilt from config rehydrates metadata subclasses."""
    config = self._create_config()
    original = self._build_store(config)

    rebuilt = self._build_store(original.to_config())

    self.assertEqual(rebuilt.to_config(), config)
    rebuilt.add_step(
        trajectory_lib.Step(
            step_id=1, source=trajectory_lib.Source.AGENT, message="step"
        ),
        metadata_cls(
            trajectory_id="config_meta_1",
            agent=trajectory_lib.Agent(name="a1", version="1.0"),
        ),
    )
    rebuilt.flush()
    (meta,) = rebuilt.get_trajectories_metadata(["config_meta_1"])
    self.assertIs(type(meta), metadata_cls)

  def test_tunix_trajectory_store_from_config_binds_tunix_metadata(
      self,
  ) -> None:
    """Verifies TunixTrajectoryStore.from_config binds TunixTrajectoryMetadata."""
    config = self._create_config()
    built = store.TunixTrajectoryStore.from_config(config)
    if built is None:
      self.fail("from_config returned None for an enabled config.")
    self.addCleanup(built.close)

    self.assertIs(
        built._metadata_cls,  # pylint: disable=protected-access
        trajectory_lib.TunixTrajectoryMetadata,
    )
    with self.assertRaisesRegex(TypeError, "is bound to metadata type"):
      built.update_metadata(cast(Any, trajectory_testing.METADATA_1))

  def test_to_redacted_config_without_secrets_equals_to_config(self) -> None:
    built = self._build_store(self._create_config())

    redacted_config = built.to_redacted_config()

    self.assertEqual(redacted_config, built.to_config())
