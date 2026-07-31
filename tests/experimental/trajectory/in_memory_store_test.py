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


if __name__ == "__main__":
  absltest.main()
