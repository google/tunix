from typing import Any

from absl.testing import absltest
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib


class InMemoryTrajectoryReaderTest(store_testing.TrajectoryReaderTestCase):
  """Contract tests for InMemoryTrajectoryStore's TrajectoryReader implementation."""

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
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    if initial_data:
      for meta, steps in initial_data:
        for step in steps:
          mem_store.add_step(step, meta)
      mem_store.flush()
    return mem_store


class InMemoryTrajectoryWriterTest(store_testing.TrajectoryWriterTestCase):
  """Contract tests for InMemoryTrajectoryStore's TrajectoryWriter implementation."""

  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    return mem_store, mem_store

  def test_update_metadata(self) -> None:
    """Verifies that updating metadata in-memory updates the stored metadata."""
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="t1",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        extra={"status": "RUNNING"},
    )
    step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.AGENT, message="m1"
    )
    mem_store.add_step(step, meta)
    meta.extra["status"] = "SUCCEEDED"
    mem_store.update_metadata(meta)
    read_meta = mem_store.get_trajectories_metadata()[0]
    self.assertEqual(read_meta.extra["status"], "SUCCEEDED")

  def test_metadata_mutation_isolation(self) -> None:
    """Verifies that mutating returned metadata does not alter internal store state."""
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="iso_1",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        notes="initial notes",
        extra={"count": 1},
    )
    step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.AGENT, message="m1"
    )
    mem_store.add_step(step, meta)

    # Caller mutates the returned metadata
    read_meta = mem_store.get_trajectories_metadata()[0]
    read_meta.notes = "mutated notes"
    read_meta.extra["count"] = 99

    # Re-reading from store should still reflect original state
    stored_meta = mem_store.get_trajectories_metadata()[0]
    self.assertEqual(stored_meta.notes, "initial notes")
    self.assertEqual(stored_meta.extra["count"], 1)


class InMemoryTrajectoryStoreMetadataClsTest(
    store_testing.TrajectoryStoreMetadataClsTestCase
):
  """Contract tests for InMemoryTrajectoryStore metadata_cls handling."""

  def _create_store(self) -> store.TrajectoryStore[Any]:
    return in_memory_store.InMemoryTrajectoryStore()


class InMemoryTrajectoryStoreConfigTest(
    store_testing.TrajectoryStoreConfigTestCase
):
  """Config contract tests for InMemoryTrajectoryStore."""

  def _create_config(self) -> dict[str, Any]:
    return {
        "enabled": True,
        "backend": "memory",
    }


if __name__ == "__main__":
  absltest.main()
