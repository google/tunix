from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory.benchmarks import benchmark_lib
from tunix.experimental.trajectory.benchmarks import data_generator

_TWO_CHECKPOINTS_WORKLOAD = data_generator.WorkloadConfig(
    cumulative_trajectory_checkpoints=[3, 7],
    steps_per_trajectory=2,
    step_payload_chars=100,
)

_THREE_CHECKPOINTS_WORKLOAD = data_generator.WorkloadConfig(
    cumulative_trajectory_checkpoints=[2, 5, 10],
    steps_per_trajectory=2,
    step_payload_chars=100,
)


class BenchmarkLibTest(parameterized.TestCase):
  """Unit tests for progressive recovery benchmark engine."""

  def _verify_recovery_benchmark(
      self,
      reader: store_lib.TrajectoryReader,
      writer: store_lib.TrajectoryWriter,
      expected_type_name: str,
      workload: data_generator.WorkloadConfig,
  ) -> None:
    """Helper function to execute recovery benchmark validation for a given store."""
    report = benchmark_lib.run_recovery_benchmark(
        reader=reader,
        writer=writer,
        workload=workload,
    )

    self.assertEqual(report.workload, workload)
    self.assertEqual(report.reader_type, expected_type_name)
    self.assertEqual(report.writer_type, expected_type_name)
    self.assertLen(
        report.checkpoints, len(workload.cumulative_trajectory_checkpoints)
    )

    for i, target_count in enumerate(
        workload.cumulative_trajectory_checkpoints
    ):
      cp = report.checkpoints[i]
      self.assertEqual(cp.total_trajectories, target_count)
      self.assertTrue(cp.validation_passed)
      self.assertIsNone(cp.validation_error)
      self.assertGreater(cp.write_qps, 0)
      self.assertGreater(cp.write_mb_per_sec, 0)

  @parameterized.named_parameters(
      ("two_checkpoints", _TWO_CHECKPOINTS_WORKLOAD),
      ("three_checkpoints", _THREE_CHECKPOINTS_WORKLOAD),
  )
  def test_run_recovery_benchmark_in_memory(
      self, workload: data_generator.WorkloadConfig
  ) -> None:
    store_instance = in_memory_store.InMemoryTrajectoryStore()
    self._verify_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        expected_type_name="InMemoryTrajectoryStore",
        workload=workload,
    )

  @parameterized.named_parameters(
      ("two_checkpoints", _TWO_CHECKPOINTS_WORKLOAD),
      ("three_checkpoints", _THREE_CHECKPOINTS_WORKLOAD),
  )
  def test_run_recovery_benchmark_file_store(
      self, workload: data_generator.WorkloadConfig
  ) -> None:
    temp_dir = self.create_tempdir().full_path
    store_instance = file_store.FileTrajectoryStore(root_dir=temp_dir)
    self._verify_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        expected_type_name="FileTrajectoryStore",
        workload=workload,
    )


if __name__ == "__main__":
  absltest.main()
