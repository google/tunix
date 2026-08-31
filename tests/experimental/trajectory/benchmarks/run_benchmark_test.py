from absl.testing import absltest
from etils import epath
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
  return run_benchmark.parse_flags(argv)


class RunBenchmarkCLITest(absltest.TestCase):

  def test_parse_flags_default(self) -> None:
    config = _parse_cmd()
    self.assertIsInstance(config.store, run_benchmark.FileTrajectoryStoreConfig)
    self.assertEqual(config.store.root_dir, epath.Path("/tmp/tunix_benchmarks"))
    self.assertFalse(config.store.cleanup_after)
    self.assertEqual(
        config.workload.cumulative_trajectory_checkpoints, [100, 1000, 10000]
    )

  def test_parse_flags_file_store_custom(self) -> None:
    config = _parse_cmd(
        "--store file --root_dir /tmp/custom_path --cleanup_after True"
        " --steps_per_trajectory 10 --step_payload_chars 500"
    )
    self.assertIsInstance(config.store, run_benchmark.FileTrajectoryStoreConfig)
    self.assertEqual(config.store.root_dir, epath.Path("/tmp/custom_path"))
    self.assertTrue(config.store.cleanup_after)
    self.assertEqual(config.workload.steps_per_trajectory, 10)
    self.assertEqual(config.workload.step_payload_chars, 500)

  def test_parse_flags_in_memory_store(self) -> None:
    config = _parse_cmd("--store in_memory")
    self.assertIsInstance(
        config.store, run_benchmark.InMemoryTrajectoryStoreConfig
    )


if __name__ == "__main__":
  absltest.main()
