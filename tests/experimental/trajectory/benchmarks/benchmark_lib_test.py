from collections.abc import Sequence
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import sql_store
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

_MULTI_WORKER_WORKLOAD = data_generator.WorkloadConfig(
    cumulative_trajectory_checkpoints=[4, 12],
    steps_per_trajectory=2,
    step_payload_chars=100,
    num_workers=4,
)

_WRITER_PER_WORKER_WORKLOAD = dataclasses.replace(
    _MULTI_WORKER_WORKLOAD, writer_per_worker=True
)


class BenchmarkLibTest(parameterized.TestCase):
  """Unit tests for progressive recovery benchmark engine."""

  def _verify_recovery_benchmark(
      self,
      reader: store_lib.TrajectoryReader,
      writer: store_lib.TrajectoryWriter | Sequence[store_lib.TrajectoryWriter],
      expected_type_name: str,
      workload: data_generator.WorkloadConfig,
  ) -> None:
    """Helper function to execute recovery benchmark validation."""
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

    prev_count = 0
    for cp, target_count in zip(
        report.checkpoints,
        workload.cumulative_trajectory_checkpoints,
        strict=True,
    ):
      self.assertEqual(cp.checkpoint_write_count, target_count - prev_count)
      self.assertEqual(cp.total_trajectories, target_count)
      self.assertTrue(cp.validation_passed)
      self.assertIsNone(cp.validation_error)
      self.assertGreater(cp.write_duration_sec, 0.0)
      self.assertGreater(cp.write_qps, 0)
      self.assertGreater(cp.write_mb_per_sec, 0)
      self.assertGreaterEqual(cp.metadata_scan_latency_ms, 0)
      self.assertGreaterEqual(cp.trajectory_load_latency_ms, 0)
      self.assertGreater(cp.enqueue_wall_sec, 0.0)
      self.assertGreaterEqual(cp.flush_wait_sec, 0.0)
      self.assertGreaterEqual(cp.cpu_sec, 0.0)
      self.assertGreater(cp.traj_write_p50_ms, 0.0)
      self.assertGreaterEqual(cp.traj_write_p95_ms, cp.traj_write_p50_ms)
      self.assertGreaterEqual(cp.traj_write_p99_ms, cp.traj_write_p95_ms)
      self.assertGreater(cp.peak_rss_mb, 0.0)
      self.assertGreater(cp.read_qps, 0)
      self.assertGreater(cp.read_mb_per_sec, 0)
      self.assertGreaterEqual(cp.avg_load_latency_per_traj_ms, 0.0)
      self.assertGreaterEqual(cp.parallelism_factor, 0.0)
      self.assertAlmostEqual(
          cp.parallelism_factor, cp.cpu_sec / cp.write_duration_sec, places=5
      )
      self.assertGreaterEqual(cp.flush_fraction, 0.0)
      self.assertLessEqual(cp.flush_fraction, 1.0)
      self.assertAlmostEqual(
          cp.flush_fraction, cp.flush_wait_sec / cp.write_duration_sec, places=5
      )
      prev_count = target_count

  @parameterized.named_parameters(
      ("two_checkpoints", _TWO_CHECKPOINTS_WORKLOAD),
      ("three_checkpoints", _THREE_CHECKPOINTS_WORKLOAD),
      ("multi_worker", _MULTI_WORKER_WORKLOAD),
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
      ("multi_worker", _MULTI_WORKER_WORKLOAD),
  )
  def test_run_recovery_benchmark_file_store(
      self, workload: data_generator.WorkloadConfig
  ) -> None:
    tmp_dir = self.create_tempdir().full_path
    store_instance = file_store.FileTrajectoryStore(root_dir=tmp_dir)
    self.addCleanup(store_instance.close)

    self._verify_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        expected_type_name="FileTrajectoryStore",
        workload=workload,
    )

  @parameterized.named_parameters(
      ("two_checkpoints", _TWO_CHECKPOINTS_WORKLOAD),
      ("three_checkpoints", _THREE_CHECKPOINTS_WORKLOAD),
      ("multi_worker", _MULTI_WORKER_WORKLOAD),
  )
  def test_run_recovery_benchmark_sql_store(
      self, workload: data_generator.WorkloadConfig
  ) -> None:
    store_instance = sql_store.SqlTrajectoryStore(
        run_id="bench_run", db_url="sqlite:///:memory:"
    )
    self.addCleanup(store_instance.close)

    self._verify_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        expected_type_name="SqlTrajectoryStore",
        workload=workload,
    )

  def test_run_recovery_benchmark_sql_store_multiple_writers(self) -> None:
    db_file = self.create_tempfile("bench.db").full_path
    db_url = f"sqlite:///{db_file}"
    writers = []
    for _ in range(4):
      writer = sql_store.SqlTrajectoryStore(run_id="bench_run", db_url=db_url)
      self.addCleanup(writer.close)
      writers.append(writer)

    self._verify_recovery_benchmark(
        reader=writers[0],
        writer=writers,
        expected_type_name="SqlTrajectoryStore",
        workload=_WRITER_PER_WORKER_WORKLOAD,
    )

  @parameterized.named_parameters(
      ("shared_writer_given_many", _MULTI_WORKER_WORKLOAD, 4),
      ("writer_per_worker_given_one", _WRITER_PER_WORKER_WORKLOAD, 1),
      ("writer_per_worker_given_too_few", _WRITER_PER_WORKER_WORKLOAD, 2),
  )
  def test_run_recovery_benchmark_writer_count_mismatch_raises_value_error(
      self, workload: data_generator.WorkloadConfig, num_writers: int
  ) -> None:
    store_instance = in_memory_store.InMemoryTrajectoryStore()
    writers = [store_instance] * num_writers

    with self.assertRaisesRegex(ValueError, r"Expected \d+ writer\(s\)"):
      benchmark_lib.run_recovery_benchmark(
          reader=store_instance, writer=writers, workload=workload
      )

  def test_recovery_benchmark_measures_write_telemetry_breakdown(self) -> None:
    store_instance = in_memory_store.InMemoryTrajectoryStore()
    report = benchmark_lib.run_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        workload=_TWO_CHECKPOINTS_WORKLOAD,
    )
    for cp in report.checkpoints:
      self.assertGreater(cp.write_duration_sec, 0.0)
      self.assertGreater(cp.enqueue_wall_sec, 0.0)
      self.assertGreaterEqual(cp.flush_wait_sec, 0.0)
      self.assertGreaterEqual(cp.cpu_sec, 0.0)
      self.assertGreaterEqual(
          cp.write_duration_sec, cp.enqueue_wall_sec + cp.flush_wait_sec - 1e-4
      )

  def test_recovery_benchmark_measures_file_store_storage_footprint(
      self,
  ) -> None:
    tmp_dir = self.create_tempdir().full_path
    store_instance = file_store.FileTrajectoryStore(root_dir=tmp_dir)
    self.addCleanup(store_instance.close)

    report = benchmark_lib.run_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        workload=_TWO_CHECKPOINTS_WORKLOAD,
    )
    for cp in report.checkpoints:
      self.assertGreater(cp.storage_bytes, 0)
      self.assertGreater(cp.storage_amplification_ratio, 0.0)

  def test_recovery_benchmark_measures_sqlite_storage_footprint(self) -> None:
    db_file = self.create_tempfile("bench.db").full_path
    db_url = f"sqlite:///{db_file}"
    store_instance = sql_store.SqlTrajectoryStore(
        run_id="bench_run", db_url=db_url
    )
    self.addCleanup(store_instance.close)

    report = benchmark_lib.run_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        workload=_TWO_CHECKPOINTS_WORKLOAD,
    )
    for cp in report.checkpoints:
      self.assertGreater(cp.storage_bytes, 0)
      self.assertGreater(cp.storage_amplification_ratio, 0.0)


if __name__ == "__main__":
  absltest.main()
