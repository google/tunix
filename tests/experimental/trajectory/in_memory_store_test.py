from absl.testing import absltest
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib


class InMemoryTrajectoryReaderTest(store_testing.TrajectoryReaderTestCase):

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
    return mem_store


class InMemoryTrajectoryWriterTest(store_testing.TrajectoryWriterTestCase):

  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    return mem_store, mem_store


class InMemoryTrajectoryStoreOrderingTest(absltest.TestCase):

  def test_get_trajectories_sorts_steps_by_step_id(self) -> None:
    """Steps added out of order are returned sorted by step_id."""
    mem_store = in_memory_store.InMemoryTrajectoryStore()

    # Add steps in reverse step_id order.
    mem_store.add_step(store_testing.STEP_2_2, store_testing.METADATA_2)
    mem_store.add_step(store_testing.STEP_2_1, store_testing.METADATA_2)

    (traj,) = mem_store.get_trajectories([store_testing.TRAJECTORY_ID_2])

    self.assertEqual(traj, store_testing.TRAJECTORY_2)
    self.assertEqual(
        [step.step_id for step in traj.steps],
        [store_testing.STEP_2_1.step_id, store_testing.STEP_2_2.step_id],
    )


if __name__ == "__main__":
  absltest.main()
