"""CLI binary to execute progressive load benchmarks on Trajectory Store."""

import dataclasses
import uuid

from absl import app
from etils import eapp
from etils import epath
from simple_parsing.helpers import subgroups as sp_subgroups
import termcolor
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class InMemoryTrajectoryStoreConfig:
  """Configuration for InMemoryTrajectoryStore backend."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
  """Top-level CLI argument container for Trajectory Store benchmarks."""

  workload: data_generator.WorkloadConfig = dataclasses.field(
      default_factory=data_generator.WorkloadConfig
  )
  store: FileTrajectoryStoreConfig | InMemoryTrajectoryStoreConfig = (
      sp_subgroups.subgroups(
          {
              "file": FileTrajectoryStoreConfig,
              "in_memory": InMemoryTrajectoryStoreConfig,
          },
          default="file",
      )
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


def main(config: BenchmarkConfig) -> None:
  run_dir = None
  match config.store:
    case FileTrajectoryStoreConfig(root_dir=target_root):
      target_root.mkdir(parents=True, exist_ok=True)
      run_id = f"run_{uuid.uuid4().hex[:8]}"
      run_dir = target_root / run_id
      run_dir.mkdir(parents=True, exist_ok=True)
      print(
          termcolor.colored(
              f"Created temporary FileTrajectoryStore run directory: {run_dir}",
              "green",
          )
      )
      store = file_store.FileTrajectoryStore(
          root_dir=target_root, run_id=run_id
      )
      reader = store
      writer = store
    case InMemoryTrajectoryStoreConfig():
      store = in_memory_store.InMemoryTrajectoryStore()
      reader = store
      writer = store
    case _:
      raise ValueError(f"Unknown store config type: {type(config.store)!r}")

  try:
    report = benchmark_lib.run_recovery_benchmark(
        reader=reader,
        writer=writer,
        workload=config.workload,
    )
    _print_report_table(report)
  finally:
    if (
        isinstance(config.store, FileTrajectoryStoreConfig)
        and config.store.cleanup_after
        and run_dir
        and run_dir.exists()
    ):
      print(
          termcolor.colored(
              f"Cleaning up temporary run directory: {run_dir}",
              "yellow",
          )
      )
      run_dir.rmtree(missing_ok=True)


if __name__ == "__main__":
  eapp.better_logging()
  app.run(main, flags_parser=parse_flags)
