from absl.testing import absltest
from etils import epath
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.trajectory.benchmarks import run_benchmark


def _parse_cmd(cmd: str = "") -> run_benchmark.BenchmarkConfig:
  """Parses a command string into a BenchmarkConfig instance.

  The absl flags parser expects argv[0] to be the binary executable name,
  so we automatically prepend 'test' to argv before parsing.

  Args:
    cmd: Optional CLI arguments string (e.g. '--store file --root_dir
      /tmp/foo').

  Returns:
    Parsed BenchmarkConfig dataclass instance.
  """
  argv = ["test"]
  if cmd:
    argv.extend(cmd.split(" "))
  # pyrefly: ignore[bad-return]
  return run_benchmark.parse_flags(argv)


def _sqlite_db_path(store_instance: store_lib.TrajectoryStore) -> epath.Path:
  """Returns the SQLite file backing `store_instance`."""
  assert isinstance(store_instance, sql_store.SqlTrajectoryStore)
  database = store_instance.engine.url.database
  assert database is not None
  return epath.Path(database)


class RunBenchmarkCLITest(absltest.TestCase):

  def test_parse_flags_default(self) -> None:
    config = _parse_cmd()
    self.assertIsInstance(config.store, run_benchmark.FileTrajectoryStoreConfig)
    self.assertEqual(config.store.root_dir, epath.Path("/tmp/tunix_benchmarks"))
    self.assertFalse(config.store.cleanup_after)
    self.assertEqual(
        config.workload.cumulative_trajectory_checkpoints, [100, 1000, 10000]
    )
    self.assertEqual(config.workload.num_workers, 1)
    self.assertFalse(config.workload.writer_per_worker)

  def test_parse_flags_file_store_custom(self) -> None:
    config = _parse_cmd(
        "--store file --root_dir /tmp/custom_path --cleanup_after True"
        " --steps_per_trajectory 10 --step_payload_chars 500 --num_workers 4"
        " --writer_per_worker True"
    )
    self.assertIsInstance(config.store, run_benchmark.FileTrajectoryStoreConfig)
    self.assertEqual(config.store.root_dir, epath.Path("/tmp/custom_path"))
    self.assertTrue(config.store.cleanup_after)
    self.assertEqual(config.workload.steps_per_trajectory, 10)
    self.assertEqual(config.workload.step_payload_chars, 500)
    self.assertEqual(config.workload.num_workers, 4)
    self.assertTrue(config.workload.writer_per_worker)

  def test_parse_flags_in_memory_store(self) -> None:
    config = _parse_cmd("--store in_memory")
    self.assertIsInstance(
        config.store, run_benchmark.InMemoryTrajectoryStoreConfig
    )

  def test_parse_flags_sql_store_default(self) -> None:
    config = _parse_cmd("--store sql")
    self.assertIsInstance(config.store, run_benchmark.SqlTrajectoryStoreConfig)
    self.assertIsNone(config.store.db_url)
    self.assertTrue(config.store.cleanup_after)

  def test_parse_flags_sql_store_custom(self) -> None:
    config = _parse_cmd(
        "--store sql --db_url sqlite:////tmp/custom.db --cleanup_after False"
    )
    self.assertIsInstance(config.store, run_benchmark.SqlTrajectoryStoreConfig)
    self.assertEqual(config.store.db_url, "sqlite:////tmp/custom.db")
    self.assertFalse(config.store.cleanup_after)


class ManagedStoreTest(absltest.TestCase):

  def test_managed_store_sql_without_url_deletes_temp_database(self) -> None:
    config = run_benchmark.SqlTrajectoryStoreConfig()

    with run_benchmark._managed_store(config) as store_instance:
      db_path = _sqlite_db_path(store_instance)
      self.assertTrue(db_path.exists())

    self.assertFalse(db_path.parent.exists())

  def test_managed_store_sql_with_user_url_preserves_database(self) -> None:
    db_path = epath.Path(self.create_tempdir().full_path) / "user.db"
    config = run_benchmark.SqlTrajectoryStoreConfig(
        db_url=f"sqlite:///{db_path}", cleanup_after=True
    )

    with run_benchmark._managed_store(config):
      pass

    self.assertTrue(db_path.exists())

  def test_managed_store_file_with_cleanup_after_removes_run_dir(self) -> None:
    root_dir = epath.Path(self.create_tempdir().full_path)
    config = run_benchmark.FileTrajectoryStoreConfig(
        root_dir=root_dir, cleanup_after=True
    )

    with run_benchmark._managed_store(config) as store_instance:
      run_dir = root_dir / store_instance.to_config()["run_id"]
      self.assertTrue(run_dir.exists())

    self.assertFalse(run_dir.exists())

  def test_managed_store_body_raises_still_closes_and_cleans_up(self) -> None:
    config = run_benchmark.SqlTrajectoryStoreConfig()

    with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
      with run_benchmark._managed_store(config) as store_instance:
        db_path = _sqlite_db_path(store_instance)
        raise RuntimeError("benchmark failed")

    self.assertFalse(db_path.parent.exists())
    with self.assertRaises(RuntimeError):
      store_instance.update_metadata(trajectory_testing.METADATA_1)

  def test_managed_writers_creates_and_closes_extra_sql_writers(self) -> None:
    config = run_benchmark.SqlTrajectoryStoreConfig()

    with run_benchmark._managed_store(config) as primary_store:
      with run_benchmark._managed_writers(
          primary_store, num_writers=3
      ) as writers:
        self.assertLen(writers, 3)
        self.assertIs(writers[0], primary_store)
        self.assertIsNot(writers[1], primary_store)
        self.assertEqual(writers[1].to_config(), primary_store.to_config())
        extra_writer = writers[1]

      with self.assertRaises(RuntimeError):
        extra_writer.update_metadata(trajectory_testing.METADATA_1)

  def test_managed_writers_in_memory_multiple_writers_raises_value_error(
      self,
  ) -> None:
    config = run_benchmark.InMemoryTrajectoryStoreConfig()

    with run_benchmark._managed_store(config) as primary_store:
      with self.assertRaisesRegex(
          ValueError, "--writer_per_worker is not supported"
      ):
        with run_benchmark._managed_writers(primary_store, num_writers=2):
          pass

  def test_managed_writers_in_memory_sqlite_multiple_writers_raises_value_error(
      self,
  ) -> None:
    config = run_benchmark.SqlTrajectoryStoreConfig(db_url="sqlite:///:memory:")

    with run_benchmark._managed_store(config) as primary_store:
      with self.assertRaisesRegex(
          ValueError, "--writer_per_worker is not supported"
      ):
        with run_benchmark._managed_writers(primary_store, num_writers=2):
          pass


if __name__ == "__main__":
  absltest.main()
