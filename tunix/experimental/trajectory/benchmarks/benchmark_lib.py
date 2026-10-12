"""Benchmark engine for progressive recovery load testing on Trajectory Store."""

from collections.abc import Iterator, Sequence
import concurrent.futures
import dataclasses
import itertools
import os
import resource
import threading
import time

import termcolor
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import sql_store
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


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
  """Computes percentile from a sorted sequence of floats (0 to 100)."""
  if not sorted_values:
    return 0.0
  k = (len(sorted_values) - 1) * (percentile / 100.0)
  f = int(k)
  c = f + 1
  if c < len(sorted_values):
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
  return sorted_values[f]


def _directory_size_bytes(root: str | os.PathLike[str]) -> int:
  """Returns the total size in bytes of all regular files under `root`."""
  total = 0
  stack = [os.fspath(root)]
  while stack:
    current = stack.pop()
    with os.scandir(current) as entries:
      for entry in entries:
        if entry.is_dir(follow_symlinks=False):
          stack.append(entry.path)
        elif entry.is_file(follow_symlinks=False):
          total += entry.stat(follow_symlinks=False).st_size
  return total


def _measure_storage_bytes(reader: store.TrajectoryReader) -> int:
  """Calculates total local disk storage bytes occupied by the store."""
  if isinstance(reader, file_store.FileTrajectoryStore):
    root_path = reader.root_dir
    if not root_path.exists():
      return 0
    return _directory_size_bytes(str(root_path))

  if isinstance(reader, sql_store.SqlTrajectoryStore):
    if reader.engine.dialect.name != "sqlite":
      return 0
    database = reader.engine.url.database
    if (
        not database
        or database == ":memory:"
        or "mode=memory" in str(reader.engine.url)
    ):
      return 0
    db_file = str(database)
    if not os.path.exists(db_file):
      return 0
    return sum(
        os.path.getsize(path)
        for suffix in ("", "-wal", "-shm")
        if os.path.exists(path := f"{db_file}{suffix}")
    )

  return 0


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
      metadata={"help": "Total wall time taken to write and flush in seconds."}
  )
  enqueue_wall_sec: float = dataclasses.field(
      metadata={
          "help": (
              "Wall time taken to submit and complete trajectory step writes"
              " across workers in seconds."
          )
      }
  )
  flush_wait_sec: float = dataclasses.field(
      metadata={"help": "Wall time taken to flush all writers in seconds."}
  )
  cpu_sec: float = dataclasses.field(
      metadata={
          "help": (
              "Process CPU time (system + user) consumed during write and flush"
              " in seconds."
          )
      }
  )
  parallelism_factor: float = dataclasses.field(
      metadata={
          "help": (
              "Effective concurrency (cpu_sec / write_duration_sec) indicating"
              " CPU scaling vs I/O wait and GIL lock contention."
          )
      }
  )
  flush_fraction: float = dataclasses.field(
      metadata={
          "help": (
              "Fraction of write time spent waiting for flush (flush_wait_sec /"
              " write_duration_sec)."
          )
      }
  )
  write_qps: float = dataclasses.field(
      metadata={"help": "Write throughput in steps per second."}
  )
  write_mb_per_sec: float = dataclasses.field(
      metadata={"help": "Write throughput in MB per second."}
  )
  traj_write_p50_ms: float = dataclasses.field(
      metadata={"help": "Median write latency per trajectory in milliseconds."}
  )
  traj_write_p95_ms: float = dataclasses.field(
      metadata={
          "help": (
              "95th percentile write latency per trajectory in milliseconds."
          )
      }
  )
  traj_write_p99_ms: float = dataclasses.field(
      metadata={
          "help": (
              "99th percentile write latency per trajectory in milliseconds."
          )
      }
  )
  peak_rss_mb: float = dataclasses.field(
      metadata={"help": "Peak process resident set size in megabytes."}
  )
  storage_bytes: int = dataclasses.field(
      metadata={"help": "Total disk storage bytes occupied by the store."}
  )
  storage_amplification_ratio: float = dataclasses.field(
      metadata={
          "help": (
              "Ratio of on-disk storage size to uncompressed payload data size."
          )
      }
  )
  metadata_scan_latency_ms: float = dataclasses.field(
      metadata={
          "help": "Latency of get_trajectories_metadata() scan in milliseconds."
      }
  )
  trajectory_load_latency_ms: float = dataclasses.field(
      metadata={"help": "Latency of get_trajectories() load in milliseconds."}
  )
  avg_load_latency_per_traj_ms: float = dataclasses.field(
      metadata={
          "help": (
              "Average recovery load latency per trajectory in milliseconds."
          )
      }
  )
  read_mb_per_sec: float = dataclasses.field(
      metadata={"help": "Recovery read throughput in megabytes per second."}
  )
  read_qps: float = dataclasses.field(
      metadata={"help": "Recovery read throughput in steps per second."}
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _WritePhaseMetrics:
  """Internal telemetry collected during the checkpoint write + flush phase."""

  write_duration_sec: float
  enqueue_wall_sec: float
  flush_wait_sec: float
  cpu_sec: float
  parallelism_factor: float
  flush_fraction: float
  write_qps: float
  write_mb_per_sec: float
  traj_write_p50_ms: float
  traj_write_p95_ms: float
  traj_write_p99_ms: float
  bytes_written: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ReadPhaseMetrics:
  """Internal telemetry collected during the checkpoint recovery read phase."""

  metadata_scan_latency_ms: float
  trajectory_load_latency_ms: float
  avg_load_latency_per_traj_ms: float
  read_mb_per_sec: float
  read_qps: float
  validation_passed: bool
  validation_error: str | None


def _write_trajectory(
    writer: store.TrajectoryWriter,
    metadata: trajectory_lib.TrajectoryMetadata,
    steps: list[trajectory_lib.Step],
) -> tuple[int, int]:
  """Writes all trajectory steps and returns (step_count, byte_count)."""
  step_bytes = 0
  for step in steps:
    writer.add_step(step, metadata)
    # Generated steps consist solely of string messages; see
    # data_generator.generate_trajectories for details.
    step_bytes += len(step.message)
  return len(steps), step_bytes


def _drain_trajectories(
    writer: store.TrajectoryWriter,
    trajectories: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    lock: threading.Lock,
) -> tuple[int, int, list[float]]:
  """Writes trajectories from a shared iterator until it is exhausted.

  Each worker pulls one trajectory at a time under `lock`, so at most one
  trajectory per worker is materialized and `writer` is used by this worker
  only.

  Args:
    writer: The writer dedicated to this worker for the whole checkpoint.
    trajectories: Iterator shared by all workers of the checkpoint.
    lock: Serializes `next()` calls because generators are not thread-safe.

  Returns:
    The (step_count, byte_count, latencies_ms) written by this worker.
  """
  total_steps = 0
  total_bytes = 0
  latencies_ms: list[float] = []
  while True:
    with lock:
      item = next(trajectories, None)
    if item is None:
      return total_steps, total_bytes, latencies_ms
    metadata, steps = item
    start_time = time.perf_counter()
    steps_written, bytes_written = _write_trajectory(writer, metadata, steps)
    latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
    total_steps += steps_written
    total_bytes += bytes_written


def _run_write_phase(
    writers: Sequence[store.TrajectoryWriter],
    data_iter: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    current_traj_count: int,
    target_traj_count: int,
    num_workers: int,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> _WritePhaseMetrics:
  """Executes concurrent trajectory writes and writer barrier flushes."""
  checkpoint_write_count = target_traj_count - current_traj_count
  checkpoint_trajectories = itertools.islice(data_iter, checkpoint_write_count)
  lock = threading.Lock()
  total_steps = 0
  total_bytes = 0
  all_latencies_ms: list[float] = []

  print(
      f"  ├─ Writing {checkpoint_write_count:,d} trajectories"
      f" ({current_traj_count + 1:,d}..{target_traj_count:,d}) across"
      f" {num_workers} worker(s) ({len(writers)} writer(s))..."
  )
  cpu_start = time.process_time()
  with Timer() as write_timer:
    # `writers` holds either one shared writer or one writer per worker; see
    # `run_recovery_benchmark`.
    with Timer() as enqueue_timer:
      futures = [
          executor.submit(
              _drain_trajectories,
              writers[worker_idx % len(writers)],
              checkpoint_trajectories,
              lock,
          )
          for worker_idx in range(num_workers)
      ]
      for future in futures:
        steps_written, bytes_written, latencies = future.result()
        total_steps += steps_written
        total_bytes += bytes_written
        all_latencies_ms.extend(latencies)
    with Timer() as flush_timer:
      for w in writers:
        w.flush()
  cpu_sec = time.process_time() - cpu_start

  write_duration = write_timer.duration_sec
  write_qps = (
      (total_steps / write_duration) if write_duration > 0 else float("inf")
  )
  write_mb_per_sec = (
      (total_bytes / (1024 * 1024 * write_duration))
      if write_duration > 0
      else float("inf")
  )

  all_latencies_ms.sort()
  parallelism_factor = (cpu_sec / write_duration) if write_duration > 0 else 0.0
  flush_fraction = (
      (flush_timer.duration_sec / write_duration) if write_duration > 0 else 0.0
  )

  return _WritePhaseMetrics(
      write_duration_sec=write_duration,
      enqueue_wall_sec=enqueue_timer.duration_sec,
      flush_wait_sec=flush_timer.duration_sec,
      cpu_sec=cpu_sec,
      parallelism_factor=parallelism_factor,
      flush_fraction=flush_fraction,
      write_qps=write_qps,
      write_mb_per_sec=write_mb_per_sec,
      traj_write_p50_ms=_percentile(all_latencies_ms, 50.0),
      traj_write_p95_ms=_percentile(all_latencies_ms, 95.0),
      traj_write_p99_ms=_percentile(all_latencies_ms, 99.0),
      bytes_written=total_bytes,
  )


def _run_read_phase(
    reader: store.TrajectoryReader,
    target_traj_count: int,
) -> _ReadPhaseMetrics:
  """Executes metadata scan, full trajectory recovery load, and validation.

  Isolating the recovery read in a dedicated helper ensures the materialized
  `metas` and `loaded_trajs` collections are released immediately upon return,
  preventing multi-gigabyte object retention across subsequent checkpoints.

  Args:
    reader: Store used for the recovery read and validation phases.
    target_traj_count: Expected cumulative number of trajectories in the store.

  Returns:
    Computed read latencies, throughput rates, and validation status.
  """
  print(
      "  ├─ Executing recovery read validation for"
      f" {target_traj_count:,d} trajectories..."
  )
  with Timer() as meta_timer:
    metas = reader.get_trajectories_metadata()

  target_ids = [f"traj_{i:06d}" for i in range(1, target_traj_count + 1)]
  with Timer() as load_timer:
    loaded_trajs = reader.get_trajectories(target_ids)

  total_read_steps = 0
  total_read_bytes = 0
  for traj in loaded_trajs:
    total_read_steps += len(traj.steps)
    for step in traj.steps:
      total_read_bytes += len(step.message)

  load_duration = load_timer.duration_sec
  read_mb_per_sec = (
      (total_read_bytes / (1024 * 1024 * load_duration))
      if load_duration > 0
      else float("inf")
  )
  read_qps = (
      (total_read_steps / load_duration) if load_duration > 0 else float("inf")
  )
  avg_load_latency_per_traj_ms = (
      (load_timer.duration_ms / target_traj_count)
      if target_traj_count > 0
      else 0.0
  )

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

  return _ReadPhaseMetrics(
      metadata_scan_latency_ms=meta_timer.duration_ms,
      trajectory_load_latency_ms=load_timer.duration_ms,
      avg_load_latency_per_traj_ms=avg_load_latency_per_traj_ms,
      read_mb_per_sec=read_mb_per_sec,
      read_qps=read_qps,
      validation_passed=validation_passed,
      validation_error=validation_error,
  )


def _run_checkpoint(
    reader: store.TrajectoryReader,
    writers: Sequence[store.TrajectoryWriter],
    data_iter: Iterator[
        tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
    ],
    current_traj_count: int,
    target_traj_count: int,
    cumulative_payload_bytes: int,
    executor: concurrent.futures.ThreadPoolExecutor,
    workload: data_generator.WorkloadConfig,
) -> tuple[BenchmarkCheckpointMetrics, int]:
  """Executes write, flush, read scan, and validation for a single checkpoint."""
  write_metrics = _run_write_phase(
      writers=writers,
      data_iter=data_iter,
      current_traj_count=current_traj_count,
      target_traj_count=target_traj_count,
      num_workers=workload.num_workers,
      executor=executor,
  )
  updated_payload_bytes = cumulative_payload_bytes + write_metrics.bytes_written

  read_metrics = _run_read_phase(
      reader=reader,
      target_traj_count=target_traj_count,
  )

  storage_bytes = _measure_storage_bytes(reader)
  storage_amplification_ratio = (
      (storage_bytes / updated_payload_bytes)
      if updated_payload_bytes > 0 and storage_bytes > 0
      else 0.0
  )
  peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

  metrics = BenchmarkCheckpointMetrics(
      checkpoint_write_count=target_traj_count - current_traj_count,
      total_trajectories=target_traj_count,
      write_duration_sec=write_metrics.write_duration_sec,
      enqueue_wall_sec=write_metrics.enqueue_wall_sec,
      flush_wait_sec=write_metrics.flush_wait_sec,
      cpu_sec=write_metrics.cpu_sec,
      parallelism_factor=write_metrics.parallelism_factor,
      flush_fraction=write_metrics.flush_fraction,
      write_qps=write_metrics.write_qps,
      write_mb_per_sec=write_metrics.write_mb_per_sec,
      traj_write_p50_ms=write_metrics.traj_write_p50_ms,
      traj_write_p95_ms=write_metrics.traj_write_p95_ms,
      traj_write_p99_ms=write_metrics.traj_write_p99_ms,
      peak_rss_mb=peak_rss_mb,
      storage_bytes=storage_bytes,
      storage_amplification_ratio=storage_amplification_ratio,
      metadata_scan_latency_ms=read_metrics.metadata_scan_latency_ms,
      trajectory_load_latency_ms=read_metrics.trajectory_load_latency_ms,
      avg_load_latency_per_traj_ms=read_metrics.avg_load_latency_per_traj_ms,
      read_mb_per_sec=read_metrics.read_mb_per_sec,
      read_qps=read_metrics.read_qps,
      validation_passed=read_metrics.validation_passed,
      validation_error=read_metrics.validation_error,
  )
  return metrics, updated_payload_bytes


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
  cumulative_payload_bytes = 0
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
      checkpoint, cumulative_payload_bytes = _run_checkpoint(
          reader=reader,
          writers=writers,
          data_iter=data_iter,
          current_traj_count=current_traj_count,
          target_traj_count=target_traj_count,
          cumulative_payload_bytes=cumulative_payload_bytes,
          executor=executor,
          workload=workload,
      )
      status_color = "green" if checkpoint.validation_passed else "red"
      storage_str = (
          f", Storage={checkpoint.storage_bytes / (1024 * 1024):.2f}MB"
          if checkpoint.storage_bytes > 0
          else ""
      )
      print(
          termcolor.colored(
              f"[{idx}/{total_checkpoints}] Checkpoint {target_traj_count:,d}"
              f" finished (Write QPS={checkpoint.write_qps:,.1f},"
              f" Write MB/s={checkpoint.write_mb_per_sec:,.2f},"
              f" Traj p50={checkpoint.traj_write_p50_ms:,.1f}ms,"
              f" Read QPS={checkpoint.read_qps:,.1f},"
              f" Par={checkpoint.parallelism_factor:,.2f}x"
              f"{storage_str},"
              f" RSS={checkpoint.peak_rss_mb:,.1f}MB)",
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
