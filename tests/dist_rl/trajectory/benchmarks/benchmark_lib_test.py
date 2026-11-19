from collections.abc import Sequence
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.trajectory import file_store
from tunix.dist_rl.trajectory import in_memory_store
from tunix.dist_rl.trajectory import sql_store
from tunix.dist_rl.trajectory import store as store_lib
from tunix.dist_rl.trajectory.benchmarks import benchmark_lib
from tunix.dist_rl.trajectory.benchmarks import data_generator

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
    _MULTI_WORKER_WORKLOAD,
    writer_per_worker=True,
    metrics_updates_per_trajectory=2,
)

_METRICS_WORKLOAD = data_generator.WorkloadConfig(
    cumulative_trajectory_checkpoints=[3, 6],
    steps_per_trajectory=2,
    step_payload_chars=100,
    metrics_updates_per_trajectory=2,
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
      self.assertGreater(cp.write_duration_sec, 0)
      self.assertGreater(cp.write_qps, 0)
      self.assertGreater(cp.write_mb_per_sec, 0)
      self.assertGreaterEqual(cp.metadata_scan_latency_ms, 0)
      self.assertGreaterEqual(cp.trajectory_load_latency_ms, 0)
      prev_count = target_count

    if workload.metrics_updates_per_trajectory > 0:
      expected_extra = {
          f"metric_{i}": float(i)
          for i in range(1, workload.metrics_updates_per_trajectory + 1)
      } | {"latest_update": workload.metrics_updates_per_trajectory}
      persisted_metrics = reader.get_trajectories_metrics()
      self.assertLen(
          persisted_metrics, workload.cumulative_trajectory_checkpoints[-1]
      )
      for metrics in persisted_metrics.values():
        self.assertEqual(metrics.extra, expected_extra)

  @parameterized.named_parameters(
      ("two_checkpoints", _TWO_CHECKPOINTS_WORKLOAD),
      ("three_checkpoints", _THREE_CHECKPOINTS_WORKLOAD),
      ("multi_worker", _MULTI_WORKER_WORKLOAD),
      ("with_metrics", _METRICS_WORKLOAD),
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
      ("with_metrics", _METRICS_WORKLOAD),
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
      ("with_metrics", _METRICS_WORKLOAD),
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

  def test_run_recovery_benchmark_reports_failed_validation_on_missing_metrics(
      self,
  ) -> None:
    store_instance = in_memory_store.InMemoryTrajectoryStore()
    with mock.patch.object(
        store_instance,
        "update_metrics",
        autospec=True,
        spec_set=True,
    ):
      report = benchmark_lib.run_recovery_benchmark(
          reader=store_instance,
          writer=store_instance,
          workload=_METRICS_WORKLOAD,
      )
    self.assertFalse(report.checkpoints[0].validation_passed)
    self.assertEqual(
        report.checkpoints[0].validation_error,
        "Expected 3 trajectory metrics entries, got 0",
    )

  @parameterized.named_parameters(
      (
          "missing_metadata",
          "get_trajectories_metadata",
          "Expected 3 metadata entries, got 0",
      ),
      (
          "missing_trajectories",
          "get_trajectories",
          "Expected 3 loaded trajectories, got 0",
      ),
  )
  def test_run_recovery_benchmark_reports_failed_validation_on_read_mismatch(
      self, method_name: str, expected_error: str
  ) -> None:
    store_instance = in_memory_store.InMemoryTrajectoryStore()
    with mock.patch.object(
        store_instance,
        method_name,
        autospec=True,
        spec_set=True,
        return_value={},
    ):
      report = benchmark_lib.run_recovery_benchmark(
          reader=store_instance,
          writer=store_instance,
          workload=_TWO_CHECKPOINTS_WORKLOAD,
      )
    self.assertFalse(report.checkpoints[0].validation_passed)
    self.assertEqual(report.checkpoints[0].validation_error, expected_error)


if __name__ == "__main__":
  absltest.main()
