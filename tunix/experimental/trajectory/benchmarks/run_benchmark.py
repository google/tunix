"""CLI binary to execute progressive load benchmarks on Trajectory Store."""

from collections.abc import Iterator
import contextlib
import dataclasses
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
      self,
      run_id: str,
      num_writers: int = 1,
  ) -> dict[str, Any]:
    """Creates the run directory and returns the `from_config` dict.

    Args:
      run_id: Run identifier that scopes the benchmark's data.
      num_writers: Number of writer instances (unused; file store supports
        multiple writer instances sharing `root_dir`).

    Returns:
      A `TrajectoryStore.from_config` dict for the file backend.
    """
    del num_writers  # Unused.
    run_dir = self.root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(termcolor.colored(f"Created run directory: {run_dir}", "green"))
    return {
        "backend": "file",
        "root_dir": str(self.root_dir),
        "run_id": run_id,
    }

  def cleanup(self, run_id: str) -> None:
    """Deletes the run directory when `cleanup_after` is enabled."""
    if self.cleanup_after:
      run_dir = self.root_dir / run_id
      print(
          termcolor.colored(
              f"Cleaning up temporary directory: {run_dir}", "yellow"
          )
      )
      run_dir.rmtree(missing_ok=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class InMemoryTrajectoryStoreConfig:
  """Configuration for InMemoryTrajectoryStore backend."""

  def build_store_config(
      self,
      run_id: str,
      num_writers: int = 1,
  ) -> dict[str, Any]:
    """Returns the `from_config` dict for the in-memory backend.

    Args:
      run_id: Unused; the in-memory backend is not scoped by run.
      num_writers: Number of writer instances. Must be <= 1 since in-memory
        instances do not share state.

    Returns:
      A `TrajectoryStore.from_config` dict for the in-memory backend.

    Raises:
      ValueError: If `num_writers > 1`.
    """
    del run_id  # Unused.
    if num_writers > 1:
      raise ValueError(
          "--writer_per_worker is not supported with --store in_memory because"
          " InMemoryTrajectoryStore instances do not share state."
      )
    return {"backend": "memory"}

  def cleanup(self, run_id: str) -> None:
    """No-op; the in-memory backend creates no artifacts."""
    del run_id  # Unused.


@dataclasses.dataclass(frozen=True, kw_only=True)
class SqlTrajectoryStoreConfig:
  """Configuration for SqlTrajectoryStore backend."""

  db_url: str = dataclasses.field(
      default="sqlite:///:memory:",
      metadata={
          "help": (
              "Database connection URL (e.g. 'sqlite:///:memory:',"
              " 'sqlite:////tmp/db.sqlite', or"
              " 'postgresql+psycopg2://postgres@127.0.0.1:5432/postgres')."
              " Supply Postgres passwords through the PGPASSWORD environment"
              " variable, not the URL."
          )
      },
  )

  def build_store_config(
      self,
      run_id: str,
      num_writers: int = 1,
  ) -> dict[str, Any]:
    """Returns the `from_config` dict for the SQL backend.

    Args:
      run_id: Run identifier that scopes the benchmark's data.
      num_writers: Number of writer instances. Must be <= 1 for in-memory SQLite
        because separate in-memory engines cannot share data.

    Returns:
      A `TrajectoryStore.from_config` dict for the SQL backend.

    Raises:
      ValueError: If `num_writers > 1` and `db_url` targets in-memory SQLite.
    """
    if (
        self.db_url.endswith("/:memory:")
        or self.db_url == "sqlite://"
        or "mode=memory" in self.db_url
    ):
      if num_writers > 1:
        raise ValueError(
            "--writer_per_worker is not supported with in-memory SQLite because"
            " each writer engine creates an isolated in-memory database."
        )
    print(
        termcolor.colored(
            "Initializing SqlTrajectoryStore at"
            f" {db_engine.redact_url(self.db_url)} (run_id: {run_id})",
            "green",
        )
    )
    return {
        "backend": "sql",
        "db_url": self.db_url,
        "run_id": run_id,
    }

  def cleanup(self, run_id: str) -> None:
    """No-op; SQL databases are not deleted by the benchmark."""
    del run_id  # Unused.


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
  table_width = 120
  print("\n" + "=" * table_width)
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
  print("=" * table_width)
  writer_mode = (
      "1 writer/worker"
      if report.workload.writer_per_worker and report.workload.num_workers > 1
      else "shared writer"
  )
  print(
      f"Config: {report.workload.steps_per_trajectory} steps/traj |"
      f" {report.workload.step_payload_chars} chars/step"
      f" (~{report.workload.step_payload_chars // 1000}KB) |"
      f" {report.workload.num_workers} worker(s) ({writer_mode})"
  )
  print("-" * table_width)

  print(
      termcolor.colored(
          "[WRITE & CONCURRENCY PROFILE]", "yellow", attrs=["bold"]
      )
  )
  w_header = (
      f"{'Scale (Trajs)':<13} {'Write QPS':<11} {'Write MB/s':<11}"
      f" {'p50 (ms)':<9} {'p95 (ms)':<9} {'p99 (ms)':<9}"
      f" {'Enqueue (s)':<12} {'Flush (s)':<10} {'CPU (s)':<9}"
      f" {'Par.Factor':<11} {'Peak RSS':<10}"
  )
  print(termcolor.colored(w_header, attrs=["bold"]))
  print("-" * table_width)
  for cp in report.checkpoints:
    par_factor_str = f"{cp.parallelism_factor:.2f}x"
    rss_str = f"{cp.peak_rss_mb:.1f} MB"
    w_row = (
        f"{cp.total_trajectories:<13,d} {cp.write_qps:<11,.1f}"
        f" {cp.write_mb_per_sec:<11,.2f} {cp.traj_write_p50_ms:<9,.1f}"
        f" {cp.traj_write_p95_ms:<9,.1f} {cp.traj_write_p99_ms:<9,.1f}"
        f" {cp.enqueue_wall_sec:<12,.3f} {cp.flush_wait_sec:<10,.3f}"
        f" {cp.cpu_sec:<9,.3f} {par_factor_str:<11} {rss_str:<10}"
    )
    print(w_row)

  print("-" * table_width)
  print(
      termcolor.colored(
          "[RECOVERY READ & STORAGE PROFILE]", "yellow", attrs=["bold"]
      )
  )
  r_header = (
      f"{'Scale (Trajs)':<13} {'Read QPS':<12} {'Read MB/s':<12}"
      f" {'Meta Scan (ms)':<15} {'Load (ms)':<11} {'Load/Traj (ms)':<15}"
      f" {'Storage (MB)':<13} {'Amp.Ratio':<10} {'Validation':<8}"
  )
  print(termcolor.colored(r_header, attrs=["bold"]))
  print("-" * table_width)
  for cp in report.checkpoints:
    val_status = (
        termcolor.colored("PASSED", "green", attrs=["bold"])
        if cp.validation_passed
        else termcolor.colored("FAILED", "red", attrs=["bold"])
    )
    storage_mb_str = (
        f"{cp.storage_bytes / (1024 * 1024):.2f}"
        if cp.storage_bytes > 0
        else "N/A"
    )
    amp_str = (
        f"{cp.storage_amplification_ratio:.2f}x"
        if cp.storage_amplification_ratio > 0
        else "N/A"
    )
    r_row = (
        f"{cp.total_trajectories:<13,d} {cp.read_qps:<12,.1f}"
        f" {cp.read_mb_per_sec:<12,.2f}"
        f" {cp.metadata_scan_latency_ms:<15,.1f}"
        f" {cp.trajectory_load_latency_ms:<11,.1f}"
        f" {cp.avg_load_latency_per_traj_ms:<15,.2f}"
        f" {storage_mb_str:<13} {amp_str:<10} {val_status}"
    )
    print(r_row)
    if not cp.validation_passed:
      print(termcolor.colored(f"  └─ Error: {cp.validation_error}", "red"))

  print("=" * table_width + "\n")


@contextlib.contextmanager
def _managed_stores(
    store_config: StoreConfig,
    num_writers: int = 1,
) -> Iterator[
    tuple[store_lib.TrajectoryStore, list[store_lib.TrajectoryStore]]
]:
  """Builds a primary reader store and `num_writers` writer instances.

  Building through `from_config` exercises the same construction path the
  orchestrator and rollout workers use. On teardown, all stores are closed
  (draining pending writes) before `store_config.cleanup(run_id)` deletes any
  run artifacts, and an error raised while closing propagates rather than
  leaving a report built on partially persisted data.

  Args:
    store_config: Parsed CLI configuration selecting the backend.
    num_writers: Number of writer store instances to construct.

  Yields:
    A tuple of (reader_store, writer_stores).
  """
  run_id = f"run_{uuid.uuid4().hex[:8]}"
  backend_config = store_config.build_store_config(
      run_id, num_writers=num_writers
  )
  try:
    full_config = {"enabled": True, **backend_config}
    with contextlib.ExitStack() as stack:
      writers = []
      for _ in range(num_writers):
        store = store_lib.TrajectoryStore.from_config(full_config)
        if store is None:
          raise ValueError(
              f"Store config for backend {backend_config['backend']!r} built no"
              " store."
          )
        stack.callback(store.close)
        writers.append(store)
      yield writers[0], writers
  finally:
    store_config.cleanup(run_id)


@contextlib.contextmanager
def _managed_store(
    store_config: StoreConfig,
) -> Iterator[store_lib.TrajectoryStore]:
  """Builds a single store via `TrajectoryStore.from_config`, then tears it down."""
  with _managed_stores(store_config, num_writers=1) as (store, _):
    yield store


def main(config: BenchmarkConfig) -> None:
  """Executes progressive load recovery benchmarks for trajectory stores."""
  num_writers = (
      config.workload.num_workers if config.workload.writer_per_worker else 1
  )
  with _managed_stores(config.store, num_writers=num_writers) as (
      reader,
      writers,
  ):
    report = benchmark_lib.run_recovery_benchmark(
        reader=reader,
        writer=writers,
        workload=config.workload,
    )
    _print_report_table(report)


if __name__ == "__main__":
  eapp.better_logging()
  # pyrefly: ignore[no-matching-overload]
  app.run(main, flags_parser=parse_flags)
