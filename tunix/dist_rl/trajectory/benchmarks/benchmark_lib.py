"""Benchmark engine for progressive recovery load testing on Trajectory Store."""

from collections.abc import Iterator, Sequence
import concurrent.futures
import dataclasses
import itertools
import threading
import time

import termcolor
from tunix.dist_rl.trajectory import store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.benchmarks import data_generator


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


def _write_trajectory(
    writer: store.TrajectoryWriter,
    metadata: trajectory_lib.TrajectoryMetadata,
    steps: list[trajectory_lib.Step],
    metrics_updates: int = 0,
) -> tuple[int, int]:
  """Writes trajectory steps and optional metrics updates."""
  step_bytes = 0
  for step in steps:
    writer.add_step(step, metadata)
    # Generated steps consist solely of string messages; see
    # data_generator.generate_trajectories for details.
    step_bytes += len(step.message)
  if metrics_updates > 0:
    if metadata.trajectory_id is None:
      raise ValueError("Benchmark TrajectoryMetadata must set trajectory_id.")
    for update_idx in range(1, metrics_updates + 1):
      writer.update_metrics(
          metadata.trajectory_id,
          trajectory_lib.TrajectoryMetrics(
              extra={
                  f"metric_{update_idx}": float(update_idx),
                  "latest_update": update_idx,
              }
          ),
      )
  return len(steps), step_bytes


def _drain_trajectories(
    writer: store.TrajectoryWriter,
    trajectories: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    lock: threading.Lock,
    metrics_updates: int = 0,
) -> tuple[int, int]:
  """Writes trajectories from a shared iterator until it is exhausted.

  Each worker pulls one trajectory at a time under `lock`, so at most one
  trajectory per worker is materialized and `writer` is used by this worker
  only.

  Args:
    writer: The writer dedicated to this worker for the whole checkpoint.
    trajectories: Iterator shared by all workers of the checkpoint.
    lock: Serializes `next()` calls because generators are not thread-safe.
    metrics_updates: Number of TrajectoryMetrics updates per trajectory.

  Returns:
    The (step_count, byte_count) written by this worker.
  """
  total_steps = 0
  total_bytes = 0
  while True:
    with lock:
      item = next(trajectories, None)
    if item is None:
      return total_steps, total_bytes
    metadata, steps = item
    steps_written, bytes_written = _write_trajectory(
        writer, metadata, steps, metrics_updates=metrics_updates
    )
    total_steps += steps_written
    total_bytes += bytes_written


def _validate_checkpoint_reads(
    reader: store.TrajectoryReader,
    target_ids: Sequence[str],
    metas: Sequence[trajectory_lib.TrajectoryMetadata],
    loaded_trajs: Sequence[trajectory_lib.Trajectory],
    metrics_updates: int,
) -> str | None:
  """Validates recovered metadata, trajectories, and optional metrics."""
  target_count = len(target_ids)
  if len(metas) != target_count:
    return f"Expected {target_count} metadata entries, got {len(metas)}"
  if len(loaded_trajs) != target_count:
    return (
        f"Expected {target_count} loaded trajectories, got {len(loaded_trajs)}"
    )
  if metrics_updates > 0:
    loaded_metrics = reader.get_trajectories_metrics()
    if len(loaded_metrics) != target_count:
      return (
          f"Expected {target_count} trajectory metrics entries, got"
          f" {len(loaded_metrics)}"
      )
  return None


def _run_checkpoint(
    reader: store.TrajectoryReader,
    writers: Sequence[store.TrajectoryWriter],
    data_iter: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    current_traj_count: int,
    target_traj_count: int,
    executor: concurrent.futures.ThreadPoolExecutor,
    num_workers: int,
    metrics_updates: int = 0,
) -> BenchmarkCheckpointMetrics:
  """Executes write, flush, read scan, and validation for a single checkpoint."""
  total_steps = 0
  total_bytes = 0
  checkpoint_write_count = target_traj_count - current_traj_count
  checkpoint_trajectories = itertools.islice(data_iter, checkpoint_write_count)
  lock = threading.Lock()

  # 1. Write + Flush Phase using generator iterator
  print(
      f"  ├─ Writing {checkpoint_write_count:,d} trajectories"
      f" ({current_traj_count + 1:,d}..{target_traj_count:,d}) across"
      f" {num_workers} worker(s) ({len(writers)} writer(s))..."
  )
  with Timer() as write_timer:
    # `writers` holds either one shared writer or one writer per worker; see
    # `run_recovery_benchmark`.
    futures = [
        executor.submit(
            _drain_trajectories,
            writers[worker_idx % len(writers)],
            checkpoint_trajectories,
            lock,
            metrics_updates,
        )
        for worker_idx in range(num_workers)
    ]
    for future in futures:
      steps_written, bytes_written = future.result()
      total_steps += steps_written
      total_bytes += bytes_written
    for w in writers:
      w.flush()

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
  validation_error = _validate_checkpoint_reads(
      reader=reader,
      target_ids=target_ids,
      metas=metas,
      loaded_trajs=loaded_trajs,
      metrics_updates=metrics_updates,
  )
  validation_passed = validation_error is None

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
    writer: store.TrajectoryWriter | Sequence[store.TrajectoryWriter],
    workload: data_generator.WorkloadConfig,
) -> BenchmarkReport:
  """Runs progressive recovery benchmarks across all checkpoints.

  Args:
    reader: Store used for the recovery read and validation phases.
    writer: A single shared writer, or one writer per worker when
      `workload.writer_per_worker` is set.
    workload: Workload shape, including the worker topology.

  Returns:
    The report with one entry per checkpoint.

  Raises:
    ValueError: If the number of writers does not match the worker topology
      requested by `workload`.
  """
  writers: Sequence[store.TrajectoryWriter] = (
      (writer,) if isinstance(writer, store.TrajectoryWriter) else tuple(writer)
  )
  expected_writers = workload.num_workers if workload.writer_per_worker else 1
  if len(writers) != expected_writers:
    raise ValueError(
        f"Expected {expected_writers} writer(s) for num_workers="
        f"{workload.num_workers} and writer_per_worker="
        f"{workload.writer_per_worker}, got {len(writers)}."
    )

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

  # One pool for the whole run so thread startup and shutdown stay outside
  # every timed checkpoint section.
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=workload.num_workers
  ) as executor:
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
          writers=writers,
          data_iter=data_iter,
          current_traj_count=current_traj_count,
          target_traj_count=target_traj_count,
          executor=executor,
          num_workers=workload.num_workers,
          metrics_updates=workload.metrics_updates_per_trajectory,
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
  writer_type = type(writers[0]).__name__
  return BenchmarkReport(
      reader_type=reader_type,
      writer_type=writer_type,
      workload=workload,
      checkpoints=checkpoints,
  )
