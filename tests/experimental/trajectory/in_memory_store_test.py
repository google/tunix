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

  def test_update_metadata(self):
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

  def test_tunix_trajectory_with_step_zero(self):
    mem_store = in_memory_store.InMemoryTrajectoryStore()
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
    mem_store.add_step(step0, meta)
    mem_store.add_step(step1, meta)
    trajs = mem_store.get_trajectories(["tunix_1"])
    self.assertLen(trajs, 1)
    self.assertIsInstance(trajs[0], trajectory_lib.TunixTrajectory)
    self.assertEqual(trajs[0].steps[0].step_id, 0)
    self.assertEqual(trajs[0].steps[1].step_id, 1)

  def test_metadata_mutation_isolation(self):
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


if __name__ == "__main__":
  absltest.main()
