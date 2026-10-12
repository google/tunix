from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.trajectory.benchmarks import data_generator
from tunix.experimental.trajectory.benchmarks import run_benchmark


def _parse_cmd(cmd: str = "") -> run_benchmark.BenchmarkConfig:
  """Parses a CLI argument string into a `BenchmarkConfig`."""
  return run_benchmark.parse_flags(["test", *cmd.split()])


class RunBenchmarkCLITest(absltest.TestCase):

  def test_parse_flags_default_returns_file_store_and_default_workload(
      self,
  ) -> None:
    config = _parse_cmd()

    self.assertEqual(
        config,
        run_benchmark.BenchmarkConfig(
            workload=data_generator.WorkloadConfig(
                cumulative_trajectory_checkpoints=[100, 1000, 10000],
            ),
            store=run_benchmark.FileTrajectoryStoreConfig(
                root_dir=epath.Path("/tmp/tunix_benchmarks"),
                cleanup_after=False,
            ),
        ),
    )

  def test_parse_flags_file_store_custom_overrides_store_and_workload(
      self,
  ) -> None:
    config = _parse_cmd(
        "--store file --root_dir /tmp/custom_path --cleanup_after True"
        " --steps_per_trajectory 10 --step_payload_chars 500"
    )

    self.assertEqual(
        config.store,
        run_benchmark.FileTrajectoryStoreConfig(
            root_dir=epath.Path("/tmp/custom_path"),
            cleanup_after=True,
        ),
    )
    self.assertEqual(config.workload.steps_per_trajectory, 10)
    self.assertEqual(config.workload.step_payload_chars, 500)

  def test_parse_flags_in_memory_store_selects_in_memory_config(self) -> None:
    config = _parse_cmd("--store in_memory")

    self.assertEqual(
        config.store, run_benchmark.InMemoryTrajectoryStoreConfig()
    )

  def test_parse_flags_sql_store_default_uses_in_memory_sqlite(self) -> None:
    config = _parse_cmd("--store sql")

    self.assertEqual(
        config.store,
        run_benchmark.SqlTrajectoryStoreConfig(db_url="sqlite:///:memory:"),
    )

  def test_parse_flags_sql_store_custom_overrides_db_url(self) -> None:
    config = _parse_cmd("--store sql --db_url sqlite:////tmp/custom.db")

    self.assertEqual(
        config.store,
        run_benchmark.SqlTrajectoryStoreConfig(
            db_url="sqlite:////tmp/custom.db"
        ),
    )


class ManagedStoreTest(parameterized.TestCase):

  def test_managed_store_sql_default_builds_and_closes_in_memory_sqlite(
      self,
  ) -> None:
    config = run_benchmark.SqlTrajectoryStoreConfig()

    with run_benchmark._managed_store(config) as store_instance:
      store_config = store_instance.to_config()

    self.assertEqual(store_config["backend"], "sql")
    self.assertEqual(store_config["db_url"], "sqlite:///:memory:")
    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      store_instance.update_metadata(trajectory_testing.METADATA_1)

  def test_managed_store_sql_with_file_url_preserves_database(self) -> None:
    db_path = epath.Path(self.create_tempdir().full_path) / "user.db"
    config = run_benchmark.SqlTrajectoryStoreConfig(
        db_url=f"sqlite:///{db_path}"
    )

    with run_benchmark._managed_store(config):
      pass

    self.assertTrue(db_path.exists())

  @parameterized.named_parameters(
      ("cleanup_enabled_removes_run_dir", True, False),
      ("cleanup_disabled_preserves_run_dir", False, True),
  )
  def test_managed_store_file_respects_cleanup_after(
      self, cleanup_after: bool, expected_exists_after: bool
  ) -> None:
    root_dir = epath.Path(self.create_tempdir().full_path)
    config = run_benchmark.FileTrajectoryStoreConfig(
        root_dir=root_dir, cleanup_after=cleanup_after
    )

    with run_benchmark._managed_store(config) as store_instance:
      run_dir = root_dir / store_instance.to_config()["run_id"]
      self.assertTrue(run_dir.exists())

    self.assertEqual(run_dir.exists(), expected_exists_after)

  def test_managed_store_in_memory_yields_memory_backend(self) -> None:
    config = run_benchmark.InMemoryTrajectoryStoreConfig()

    with run_benchmark._managed_store(config) as store_instance:
      backend = store_instance.to_config()["backend"]

    self.assertEqual(backend, "memory")

  def test_managed_store_body_raises_still_closes_and_cleans_up(self) -> None:
    root_dir = epath.Path(self.create_tempdir().full_path)
    config = run_benchmark.FileTrajectoryStoreConfig(
        root_dir=root_dir, cleanup_after=True
    )

    with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
      with run_benchmark._managed_store(config) as store_instance:
        run_dir = root_dir / store_instance.to_config()["run_id"]
        self.assertTrue(run_dir.exists())
        raise RuntimeError("benchmark failed")

    self.assertFalse(run_dir.exists())
    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      store_instance.update_metadata(trajectory_testing.METADATA_1)


if __name__ == "__main__":
  absltest.main()
