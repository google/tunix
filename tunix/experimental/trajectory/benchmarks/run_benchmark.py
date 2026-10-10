"""CLI binary to execute progressive load benchmarks on Trajectory Store."""

from collections.abc import Iterator
import contextlib
import dataclasses
import tempfile
from typing import Any
import uuid

from absl import app
from etils import eapp
from etils import epath
import simple_parsing
import termcolor
from tunix.experimental.trajectory import db_engine
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory.benchmarks import benchmark_lib
from tunix.experimental.trajectory.benchmarks import data_generator


def _remove_tree(path: epath.Path) -> None:
  """Deletes a directory tree created by the benchmark, if it still exists."""
  print(termcolor.colored(f"Cleaning up temporary directory: {path}", "yellow"))
  path.rmtree(missing_ok=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class FileTrajectoryStoreConfig:
  """Configuration for FileTrajectoryStore backend."""

  root_dir: epath.Path = dataclasses.field(
      default_factory=lambda: epath.Path("/tmp/tunix_benchmarks"),
      metadata={
          "help": (
              "Root directory for FileTrajectoryStore (supports local paths"
              " and gs:// URLs)."
          )
      },
  )
  cleanup_after: bool = dataclasses.field(
      default=False,
      metadata={
          "help": "Whether to delete temporary run directory upon completion."
      },
  )

  def build_store_config(
      self, run_id: str, stack: contextlib.ExitStack
  ) -> dict[str, Any]:
    """Creates the run directory and returns the `from_config` dict.

    Args:
      run_id: Run identifier that scopes the benchmark's data.
      stack: Receives a callback deleting the run directory when `cleanup_after`
        is set.

    Returns:
      A `TrajectoryStore.from_config` dict for the file backend.
    """
    run_dir = self.root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(termcolor.colored(f"Created run directory: {run_dir}", "green"))
    if self.cleanup_after:
      stack.callback(_remove_tree, run_dir)
    return {
        "backend": "file",
        "root_dir": str(self.root_dir),
        "run_id": run_id,
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class InMemoryTrajectoryStoreConfig:
  """Configuration for InMemoryTrajectoryStore backend."""

  def build_store_config(
      self, run_id: str, stack: contextlib.ExitStack
  ) -> dict[str, Any]:
    """Returns the `from_config` dict for the in-memory backend.

    Args:
      run_id: Unused; the in-memory backend is not scoped by run.
      stack: Unused; the in-memory backend creates no artifacts.

    Returns:
      A `TrajectoryStore.from_config` dict for the in-memory backend.
    """
    del run_id, stack  # Unused.
    return {"backend": "memory"}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SqlTrajectoryStoreConfig:
  """Configuration for SqlTrajectoryStore backend."""

  db_url: str | None = dataclasses.field(
      default=None,
      metadata={
          "help": (
              "Database connection URL (e.g. 'sqlite:////tmp/db.sqlite' or"
              " 'postgresql+psycopg2://postgres@127.0.0.1:5432/postgres'). If"
              " omitted, a temporary file-backed SQLite database is created."
              " Supply Postgres passwords through the PGPASSWORD environment"
              " variable, not the URL."
          )
      },
  )
  cleanup_after: bool = dataclasses.field(
      default=True,
      metadata={
          "help": (
              "Whether to delete the temporary SQLite database created when"
              " --db_url is omitted. A user-supplied --db_url is never"
              " deleted."
          )
      },
  )

  def build_store_config(
      self, run_id: str, stack: contextlib.ExitStack
  ) -> dict[str, Any]:
    """Resolves the database URL and returns the `from_config` dict.

    Args:
      run_id: Run identifier that scopes the benchmark's data.
      stack: Receives a callback deleting the temporary SQLite database when
        `db_url` is omitted and `cleanup_after` is set.

    Returns:
      A `TrajectoryStore.from_config` dict for the SQL backend.
    """
    db_url = self.db_url
    if db_url is None:
      # Only a database created here is ever deleted; a user-supplied URL may
      # point at data that predates the benchmark.
      tmp_dir = epath.Path(tempfile.mkdtemp(prefix="tunix_sql_bench_"))
      if self.cleanup_after:
        stack.callback(_remove_tree, tmp_dir)
      db_url = f"sqlite:///{tmp_dir / 'trajectories.db'}"
    print(
        termcolor.colored(
            "Initializing SqlTrajectoryStore at"
            f" {db_engine.redact_url(db_url)} (run_id: {run_id})",
            "green",
        )
    )
    return {
        "backend": "sql",
        "db_url": db_url,
        "run_id": run_id,
    }


StoreConfig = (
    FileTrajectoryStoreConfig
    | InMemoryTrajectoryStoreConfig
    | SqlTrajectoryStoreConfig
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
  """Top-level CLI argument container for Trajectory Store benchmarks."""

  workload: data_generator.WorkloadConfig = dataclasses.field(
      default_factory=data_generator.WorkloadConfig
  )
  store: StoreConfig = simple_parsing.subgroups(
      {
          "file": FileTrajectoryStoreConfig,
          "in_memory": InMemoryTrajectoryStoreConfig,
          "sql": SqlTrajectoryStoreConfig,
      },
      default="file",
  )


parse_flags = eapp.make_flags_parser(BenchmarkConfig)


def _print_report_table(report: benchmark_lib.BenchmarkReport) -> None:
  """Prints a clean ASCII table of benchmark results to stdout with color."""
  print("\n" + "=" * 90)
  store_desc = (
      report.reader_type
      if report.reader_type == report.writer_type
      else f"Reader: {report.reader_type}, Writer: {report.writer_type}"
  )
  print(
      termcolor.colored(
          f"Tunix Trajectory Store Progressive Benchmark Report ({store_desc})",
          "cyan",
          attrs=["bold"],
      )
  )
  print("=" * 90)
  print(
      f"Config: {report.workload.steps_per_trajectory} steps/traj |"
      f" {report.workload.step_payload_chars} chars/step"
      f" (~{report.workload.step_payload_chars // 1000}KB)"
  )
  print("-" * 90)
  header = (
      f"{'Scale (Trajs)':<15} {'Write QPS':<15} {'Write MB/s':<15}"
      f" {'GetMeta (ms)':<15} {'LoadTraj (ms)':<15} {'Validation':<10}"
  )
  print(termcolor.colored(header, attrs=["bold"]))
  print("-" * 90)

  for cp in report.checkpoints:
    val_status = (
        termcolor.colored("PASSED", "green", attrs=["bold"])
        if cp.validation_passed
        else termcolor.colored("FAILED", "red", attrs=["bold"])
    )
    row = (
        f"{cp.total_trajectories:<15,d} {cp.write_qps:<15,.1f}"
        f" {cp.write_mb_per_sec:<15,.2f}"
        f" {cp.metadata_scan_latency_ms:<15,.1f}"
        f" {cp.trajectory_load_latency_ms:<15,.1f} {val_status}"
    )
    print(row)
    if not cp.validation_passed:
      print(termcolor.colored(f"  └─ Error: {cp.validation_error}", "red"))
  print("=" * 90 + "\n")


@contextlib.contextmanager
def _managed_store(
    store_config: StoreConfig,
) -> Iterator[store_lib.TrajectoryStore]:
  """Builds a store via `TrajectoryStore.from_config`, then tears it down.

  Building through `from_config` exercises the same construction path the
  orchestrator and rollout workers use. Each artifact the benchmark creates
  registers its cleanup on an `ExitStack` at creation time. Callbacks run in
  reverse order, so the store is closed (draining pending writes) before its
  files are deleted, and an error raised while closing propagates rather than
  leaving a report built on partially persisted data looking healthy.

  Args:
    store_config: Parsed CLI configuration selecting the backend.

  Yields:
    The constructed store, open for reads and writes.
  """
  run_id = f"run_{uuid.uuid4().hex[:8]}"
  with contextlib.ExitStack() as stack:
    backend_config = store_config.build_store_config(run_id, stack)
    store = store_lib.TrajectoryStore.from_config(
        {"enabled": True, **backend_config}
    )
    if store is None:
      raise ValueError(
          f"Store config for backend {backend_config['backend']!r} built no"
          " store."
      )
    # Registered last so it runs first, before any directory is deleted.
    stack.callback(store.close)
    yield store


def main(config: BenchmarkConfig) -> None:
  """Executes progressive load recovery benchmarks for trajectory stores."""
  with _managed_store(config.store) as store_instance:
    report = benchmark_lib.run_recovery_benchmark(
        reader=store_instance,
        writer=store_instance,
        workload=config.workload,
    )
    _print_report_table(report)


if __name__ == "__main__":
  eapp.better_logging()
  # pyrefly: ignore[no-matching-overload]
  app.run(main, flags_parser=parse_flags)
