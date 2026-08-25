"""Benchmark engine for progressive recovery load testing on Trajectory Store."""

from collections.abc import Iterator
import dataclasses
import time

import termcolor
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.benchmarks import data_generator


class Timer:
  """Context manager to measure block execution duration."""

  def __init__(self) -> None:
    self.start: float = 0.0
    self.duration_sec: float = 0.0

  def __enter__(self) -> "Timer":
    self.start = time.perf_counter()
    self.duration_sec = 0.0
    return self

  def __exit__(self, *args: object) -> None:
    self.duration_sec = time.perf_counter() - self.start

  @property
  def duration_ms(self) -> float:
    return self.duration_sec * 1000.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class BenchmarkCheckpointMetrics:
  """Metrics collected for a single progressive recovery checkpoint."""

  checkpoint_write_count: int = dataclasses.field(
      metadata={
          "help": (
              "Number of trajectories written since the previous checkpoint"
              " (or start of run)."
          )
      }
  )
  total_trajectories: int = dataclasses.field(
      metadata={
          "help": (
              "Cumulative count of trajectories in the store at this"
              " checkpoint."
          )
      }
  )
  write_duration_sec: float = dataclasses.field(
      metadata={"help": "Time taken to execute writer.flush() in seconds."}
  )
  write_qps: float = dataclasses.field(
      metadata={"help": "Write throughput in steps per second."}
  )
  write_mb_per_sec: float = dataclasses.field(
      metadata={"help": "Write throughput in MB per second."}
  )
  metadata_scan_latency_ms: float = dataclasses.field(
      metadata={
          "help": "Latency of get_trajectories_metadata() scan in milliseconds."
      }
  )
  trajectory_load_latency_ms: float = dataclasses.field(
      metadata={"help": "Latency of get_trajectories() load in milliseconds."}
  )
  validation_passed: bool = dataclasses.field(
      metadata={
          "help": (
              "True if retrieved metadata and trajectory counts match expected."
          )
      }
  )
  validation_error: str | None = dataclasses.field(
      default=None,
      metadata={"help": "Error message if validation failed."},
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class BenchmarkReport:
  """Overall benchmark report containing all checkpoint metrics."""

  reader_type: str
  writer_type: str
  workload: data_generator.WorkloadConfig
  checkpoints: list[BenchmarkCheckpointMetrics]


def _run_checkpoint(
    reader: store.TrajectoryReader,
    writer: store.TrajectoryWriter,
    data_iter: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    current_traj_count: int,
    target_traj_count: int,
) -> BenchmarkCheckpointMetrics:
  """Executes write, flush, read scan, and validation for a single checkpoint."""
  total_steps = 0
  total_bytes = 0
  checkpoint_write_count = target_traj_count - current_traj_count

  # 1. Write + Flush Phase using generator iterator
  print(
      f"  ├─ Writing {checkpoint_write_count:,d} trajectories"
      f" ({current_traj_count + 1:,d}..{target_traj_count:,d})..."
  )
  with Timer() as write_timer:
    for _, (metadata, steps) in zip(
        range(current_traj_count, target_traj_count), data_iter
    ):
      for step in steps:
        writer.add_step(step, metadata)
        total_steps += 1
        # We assume generated steps consist solely of string messages.
        # Check data_generator.generate_trajectories for details.
        total_bytes += len(step.message)
    writer.flush()

  write_duration = write_timer.duration_sec
  write_qps = (
      (total_steps / write_duration) if write_duration > 0 else float("inf")
  )
  write_mb_per_sec = (
      (total_bytes / (1024 * 1024 * write_duration))
      if write_duration > 0
      else float("inf")
  )

  # 2. Recovery Read Check Phase
  print(
      "  ├─ Executing recovery read validation for"
      f" {target_traj_count:,d} trajectories..."
  )
  with Timer() as meta_timer:
    metas = reader.get_trajectories_metadata()

  target_ids = [f"traj_{i:06d}" for i in range(1, target_traj_count + 1)]
  with Timer() as load_timer:
    loaded_trajs = reader.get_trajectories(target_ids)

  # 3. Validation
  validation_passed = True
  validation_error = None
  if len(metas) != target_traj_count:
    validation_passed = False
    validation_error = (
        f"Expected {target_traj_count} metadata entries, got {len(metas)}"
    )
  elif len(loaded_trajs) != target_traj_count:
    validation_passed = False
    validation_error = (
        f"Expected {target_traj_count} loaded trajectories, got"
        f" {len(loaded_trajs)}"
    )

  return BenchmarkCheckpointMetrics(
      checkpoint_write_count=checkpoint_write_count,
      total_trajectories=target_traj_count,
      write_duration_sec=write_duration,
      write_qps=write_qps,
      write_mb_per_sec=write_mb_per_sec,
      metadata_scan_latency_ms=meta_timer.duration_ms,
      trajectory_load_latency_ms=load_timer.duration_ms,
      validation_passed=validation_passed,
      validation_error=validation_error,
  )


def run_recovery_benchmark(
    reader: store.TrajectoryReader,
    writer: store.TrajectoryWriter,
    workload: data_generator.WorkloadConfig,
) -> BenchmarkReport:
  """Runs progressive recovery benchmarks across all checkpoints."""
  total_checkpoints = len(workload.cumulative_trajectory_checkpoints)
  print(
      termcolor.colored(
          "\nStarting progressive recovery benchmark across"
          f" {total_checkpoints} checkpoint(s)...",
          "cyan",
          attrs=["bold"],
      )
  )

  data_iter = data_generator.generate_trajectories(workload)
  current_traj_count = 0
  checkpoints = []

  for idx, target_traj_count in enumerate(
      workload.cumulative_trajectory_checkpoints, start=1
  ):
    print(
        termcolor.colored(
            f"[{idx}/{total_checkpoints}] Processing checkpoint target:"
            f" {target_traj_count:,d} trajectories...",
            "yellow",
            attrs=["bold"],
        )
    )
    checkpoint = _run_checkpoint(
        reader=reader,
        writer=writer,
        data_iter=data_iter,
        current_traj_count=current_traj_count,
        target_traj_count=target_traj_count,
    )
    status_color = "green" if checkpoint.validation_passed else "red"
    print(
        termcolor.colored(
            f"[{idx}/{total_checkpoints}] Checkpoint {target_traj_count:,d}"
            f" finished (Write QPS={checkpoint.write_qps:,.1f},"
            f" Write MB/s={checkpoint.write_mb_per_sec:,.2f},"
            f" Meta Scan={checkpoint.metadata_scan_latency_ms:,.1f}ms,"
            f" Load={checkpoint.trajectory_load_latency_ms:,.1f}ms)",
            status_color,
        )
    )
    checkpoints.append(checkpoint)
    current_traj_count = target_traj_count

  print(
      termcolor.colored(
          "All benchmark checkpoints completed successfully.",
          "cyan",
          attrs=["bold"],
      )
  )

  reader_type = type(reader).__name__
  writer_type = type(writer).__name__
  return BenchmarkReport(
      reader_type=reader_type,
      writer_type=writer_type,
      workload=workload,
      checkpoints=checkpoints,
  )
