"""Unit tests for AsyncSqlWriter verifying SQL persistence and concurrency."""

import concurrent.futures
import threading
import time
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import async_sql_writer
from tunix.experimental.trajectory import base_writer
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import schema_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


class AsyncSqlWriterTest(parameterized.TestCase):
  """Unit tests for AsyncSqlWriter in parity with AsyncFileWriterTest."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
    schema.METADATA.create_all(self.engine)

  # ============================================================================
  # Core Writing Functionality
  # ============================================================================

  def test_write_step_non_blocking_and_flush_persists(self) -> None:
    """Verifies write_step enqueues and flush blocks until persisted."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.write_step(
        run_id="run_sync",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    writer.flush()
    writer.close()

    runs = schema_testing.fetch_all(self.engine, sa.select(schema.RUNS_TABLE))
    self.assertLen(runs, 1)
    self.assertEqual(runs[0]["run_id"], "run_sync")
    self.assertEqual(runs[0]["agent_name"], "agent_v1")

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(
        trajs[0]["trajectory_id"], trajectory_testing.TRAJECTORY_ID_1
    )
    self.assertEqual(trajs[0]["status"], "RUNNING")

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["step_id"], trajectory_testing.STEP_1_1.step_id)
    self.assertEqual(
        steps[0]["payload"]["message"], trajectory_testing.STEP_1_1.message
    )

  def test_write_step_metadata_only(self) -> None:
    """Verifies write_step enqueues and writes metadata without step payload."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.write_step(
        run_id="meta_only_run",
        metadata=trajectory_testing.METADATA_1,
        step=None,
    )
    writer.flush()
    writer.close()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(
        trajs[0]["trajectory_id"], trajectory_testing.TRAJECTORY_ID_1
    )
    self.assertEqual(trajs[0]["status"], "PENDING")

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertEmpty(steps)

  def test_sequential_fifo_order_across_steps(self) -> None:
    """Verifies that multiple steps are written sequentially in order."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_testing.METADATA_2
    steps = [
        trajectory_testing.STEP_2_1,
        trajectory_testing.STEP_2_2,
        trajectory_testing.STEP_2_3,
        trajectory_testing.STEP_2_4,
        trajectory_testing.STEP_2_5,
    ]

    for step in steps:
      writer.write_step(run_id="run_seq", metadata=meta, step=step)

    writer.flush()
    writer.close()

    saved_steps = schema_testing.fetch_all(
        self.engine,
        sa.select(schema.STEPS_TABLE).order_by(schema.STEPS_TABLE.c.step_id),
    )
    self.assertLen(saved_steps, len(steps))
    for i, step in enumerate(steps):
      self.assertEqual(saved_steps[i]["step_id"], step.step_id)
      self.assertEqual(saved_steps[i]["payload"]["message"], step.message)

  # ============================================================================
  # Snapshot-on-Enqueue Ownership
  # ============================================================================

  def test_mutating_step_after_write_step_does_not_affect_database(
      self,
  ) -> None:
    """Verifies an enqueued step is snapshotted, not shared with caller."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_testing.METADATA_1.model_copy(deep=True)
    step = trajectory_testing.STEP_1_1.model_copy(deep=True)

    block_event = threading.Event()
    worker_thread = None
    original_process_task = writer._process_task

    def blocking_process_task(task):
      block_event.wait()
      original_process_task(task)

    try:
      with mock.patch.object(
          writer, "_process_task", side_effect=blocking_process_task
      ):
        writer.write_step(run_id="copy_run", metadata=meta, step=step)
        worker_thread = writer._worker_thread

        # Mutate while task is still queued: worker has not processed it yet.
        step.message = "mutated after enqueueing"
        meta.notes = "mutated after enqueueing"

        block_event.set()
        writer.flush()
    finally:
      block_event.set()
      if worker_thread is not None:
        worker_thread.join(timeout=5.0)

    saved_steps = schema_testing.fetch_all(
        self.engine, sa.select(schema.STEPS_TABLE)
    )
    self.assertLen(saved_steps, 1)
    self.assertEqual(
        saved_steps[0]["payload"]["message"],
        trajectory_testing.STEP_1_1.message,
    )

  def test_mutating_step_after_write_step_does_not_affect_later_step(
      self,
  ) -> None:
    """Verifies a caller can reuse one mutable step object across writes."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    step = trajectory_testing.STEP_2_1.model_copy(deep=True)
    writer.write_step(
        run_id="reuse_run",
        metadata=trajectory_testing.METADATA_2,
        step=step,
    )

    # Reuse the same object for the next step, as a rollout loop might.
    step.step_id = trajectory_testing.STEP_2_2.step_id
    step.source = trajectory_testing.STEP_2_2.source
    step.message = trajectory_testing.STEP_2_2.message
    writer.write_step(
        run_id="reuse_run",
        metadata=trajectory_testing.METADATA_2,
        step=step,
    )
    writer.flush()
    writer.close()

    saved_steps = schema_testing.fetch_all(
        self.engine,
        sa.select(schema.STEPS_TABLE).order_by(schema.STEPS_TABLE.c.step_id),
    )
    self.assertLen(saved_steps, 2)
    self.assertEqual(
        saved_steps[0]["step_id"], trajectory_testing.STEP_2_1.step_id
    )
    self.assertEqual(
        saved_steps[0]["payload"]["message"],
        trajectory_testing.STEP_2_1.message,
    )
    self.assertEqual(
        saved_steps[1]["step_id"], trajectory_testing.STEP_2_2.step_id
    )
    self.assertEqual(
        saved_steps[1]["payload"]["message"],
        trajectory_testing.STEP_2_2.message,
    )

  # ============================================================================
  # Lazy Worker Thread Initialization
  # ============================================================================

  def test_lazy_worker_thread_initialization(self) -> None:
    """Verifies worker is not started in __init__ and starts on write_step."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.assertIsNone(writer._worker_thread)

    writer.write_step(run_id="test_run", metadata=trajectory_testing.METADATA_1)
    self.assertIsNotNone(writer._worker_thread)
    self.assertEqual(writer._worker_thread.name, "AsyncSqlWriterWorker")
    self.assertTrue(writer._worker_thread.is_alive())
    writer.flush()
    writer.close()

  def test_lazy_worker_thread_initialization_concurrent(self) -> None:
    """Verifies concurrent writes safely start exactly one worker thread."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.assertIsNone(writer._worker_thread)

    num_threads = 8
    num_steps_per_thread = 4

    def write_worker(thread_idx: int) -> None:
      traj_id = f"lazy_init_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps_per_thread + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        writer.write_step(run_id="concurrent_init", metadata=meta, step=step)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [executor.submit(write_worker, i) for i in range(num_threads)]
      for f in futures:
        f.result()

    writer.flush()
    self.assertIsNotNone(writer._worker_thread)
    self.assertTrue(writer._worker_thread.is_alive())
    writer.close()
    self.assertFalse(writer._worker_thread.is_alive())

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, num_threads * num_steps_per_thread)

  # ============================================================================
  # Optimizations & Caching
  # ============================================================================

  def test_run_caching_behavior(self) -> None:
    """Verifies parent run row is inserted once and cached in memory."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta1 = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_1",
        agent=trajectory_lib.Agent(name="agent_1", version="1.0"),
    )
    meta2 = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_2",
        agent=trajectory_lib.Agent(name="agent_1", version="1.0"),
    )

    writer.write_step(run_id="cached_run", metadata=meta1)
    writer.flush()
    self.assertIn("cached_run", writer._registered_run_ids)

    writer.write_step(run_id="cached_run", metadata=meta2)
    writer.flush()
    writer.close()

    runs = schema_testing.fetch_all(self.engine, sa.select(schema.RUNS_TABLE))
    self.assertLen(runs, 1)

  # ============================================================================
  # Relational Upserts & Status Handling
  # ============================================================================

  def test_conflict_upsert_updates_existing_record(self) -> None:
    """Verifies on_conflict_do_update updates mutable fields idempotently."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_up",
        agent=trajectory_lib.Agent(name="agent_u", version="1.0"),
        status="PENDING",
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="original"
    )

    writer.write_step(run_id="conflict_run", metadata=meta, step=step)
    writer.flush()

    meta_updated = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_up",
        agent=trajectory_lib.Agent(name="agent_u", version="1.0"),
        status="COMPLETED",
    )
    step_updated = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="updated"
    )

    writer.write_step(
        run_id="conflict_run", metadata=meta_updated, step=step_updated
    )
    writer.flush()
    writer.close()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["status"], "COMPLETED")

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["payload"]["message"], "updated")

  def test_stepless_metadata_update_preserves_existing_status(self) -> None:
    """Verifies stepless metadata update preserves existing database status."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    # Initial write with a step defaults status to RUNNING.
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_preserve",
        agent=trajectory_lib.Agent(name="agent_p", version="1.0"),
        notes="initial notes",
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="step 0"
    )
    writer.write_step(run_id="status_preserve_run", metadata=meta, step=step)
    writer.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["status"], "RUNNING")

    # Subsequent update with step=None and no explicit status updates notes
    # while preserving RUNNING.
    meta_notes = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_preserve",
        agent=trajectory_lib.Agent(name="agent_p", version="1.0"),
        notes="updated notes",
    )
    writer.write_step(
        run_id="status_preserve_run", metadata=meta_notes, step=None
    )
    writer.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["status"], "RUNNING")
    self.assertEqual(trajs[0]["trajectory_metadata"]["notes"], "updated notes")

    # Explicit terminal status transitions status to COMPLETED.
    meta_completed = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_preserve",
        agent=trajectory_lib.Agent(name="agent_p", version="1.0"),
        status="COMPLETED",
    )
    writer.write_step(
        run_id="status_preserve_run", metadata=meta_completed, step=None
    )
    writer.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["status"], "COMPLETED")

    # Further stepless update with base TrajectoryMetadata preserves COMPLETED.
    meta_post = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_preserve",
        agent=trajectory_lib.Agent(name="agent_p", version="1.0"),
        notes="post-completion notes",
    )
    writer.write_step(
        run_id="status_preserve_run", metadata=meta_post, step=None
    )
    writer.flush()
    writer.close()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["status"], "COMPLETED")
    self.assertEqual(
        trajs[0]["trajectory_metadata"]["notes"], "post-completion notes"
    )

  def test_resolve_status_resolution_order(self) -> None:
    """Verifies the precedence matrix in _resolve_status."""
    # 1. Explicit status takes precedence.
    meta_explicit = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="t1",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        status="COMPLETED",
    )
    self.assertEqual(
        async_sql_writer._resolve_status(meta_explicit, has_step=True),
        "COMPLETED",
    )

    # 2. final_metrics presence implies COMPLETED for standard ATIF.
    meta_final = trajectory_lib.TrajectoryMetadata(
        trajectory_id="t2",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        final_metrics=trajectory_lib.FinalMetrics(
            total_reward=1.0, step_count=5
        ),
    )
    self.assertEqual(
        async_sql_writer._resolve_status(meta_final, has_step=False),
        "COMPLETED",
    )

    # 3. has_step transitions to RUNNING when no terminal status given.
    meta_running = trajectory_lib.TrajectoryMetadata(
        trajectory_id="t3",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
    )
    self.assertEqual(
        async_sql_writer._resolve_status(meta_running, has_step=True),
        "RUNNING",
    )

    # 4. Stepless update without status returns None to preserve existing
    # database status.
    self.assertIsNone(
        async_sql_writer._resolve_status(meta_running, has_step=False)
    )

    # 5. Metadata extra["status"] is ignored; does not affect resolution.
    meta_extra = trajectory_lib.TrajectoryMetadata(
        trajectory_id="t4",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
        extra={"status": "FAILED"},
    )
    self.assertIsNone(
        async_sql_writer._resolve_status(meta_extra, has_step=False)
    )

  def test_write_step_persists_specified_run_id(self) -> None:
    """Verifies caller-specified run_id is stored with whitespace stripped."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="t1",
        session_id="ignored_session_run",
        agent=trajectory_lib.Agent(name="a1", version="1.0"),
    )
    writer.write_step(run_id="  explicit_run  ", metadata=meta)
    writer.flush()
    writer.close()

    trajs = schema_testing.fetch_all(
        self.engine, schema.TRAJECTORIES_TABLE.select()
    )
    self.assertEqual(trajs[0]["run_id"], "explicit_run")

  # ============================================================================
  # Barrier Synchronization
  # ============================================================================

  def test_flush_idempotent_and_empty(self) -> None:
    """Verifies flush on empty writer is safe and flushes are idempotent."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.flush()
    writer.flush()

    writer.write_step(
        run_id="flush_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    writer.flush()
    writer.flush()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    writer.close()

  # ============================================================================
  # Error Handling & Best-Effort Resilience
  # ============================================================================

  def test_error_handling_suppresses_exceptions_and_logs(self) -> None:
    """Verifies DB errors are logged and suppressed without raising on flush."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta_bad = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_err",
        agent=trajectory_lib.Agent(name="agent_err", version="1.0"),
    )

    with mock.patch.object(
        self.engine, "begin", side_effect=RuntimeError("Simulated DB error")
    ):
      writer.write_step(run_id="err_run", metadata=meta_bad)
      # Flush must not hang or raise despite task failure.
      writer.flush()

    # Verify worker remains alive and processes subsequent valid writes.
    meta_good = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_ok",
        agent=trajectory_lib.Agent(name="agent_ok", version="1.0"),
    )
    writer.write_step(run_id="err_run", metadata=meta_good)
    writer.flush()
    writer.close()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["trajectory_id"], "traj_ok")

  def test_subsequent_writes_continue_after_error(self) -> None:
    """Verifies worker continues processing subsequent writes after error."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    original_begin = self.engine.begin

    call_count = 0

    def failing_begin():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise RuntimeError("Simulated step 1 DB failure")
      return original_begin()

    with mock.patch.object(self.engine, "begin", side_effect=failing_begin):
      writer.write_step(
          run_id="r1",
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )
      writer.write_step(
          run_id="r2",
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_1,
      )
      writer.flush()
      writer.close()

    # Step 2 was written successfully despite step 1 failing.
    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(
        steps[0]["payload"]["message"], trajectory_testing.STEP_2_1.message
    )

  def test_log_task_error_without_step(self) -> None:
    """Verifies _log_task_error formats log message properly without step."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    task_no_step = base_writer.WriteTask(
        run_id="r1",
        metadata=trajectory_lib.TrajectoryMetadata(
            trajectory_id="t1",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        ),
    )
    with mock.patch.object(async_sql_writer.logging, "exception") as mock_log:
      writer._log_task_error(task_no_step)
      mock_log.assert_called_once()
      self.assertEqual(mock_log.call_args[0][1], "metadata")
      self.assertEqual(mock_log.call_args[0][2], "r1")
      self.assertEqual(mock_log.call_args[0][3], "t1")

  def test_log_task_error_with_step(self) -> None:
    """Verifies _log_task_error formats log message properly with step."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    task_with_step = base_writer.WriteTask(
        run_id="r2",
        metadata=trajectory_lib.TrajectoryMetadata(
            trajectory_id="t2",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        ),
        step=trajectory_lib.Step(
            step_id=42,
            source=trajectory_lib.Source.AGENT,
            message="err",
        ),
    )
    with mock.patch.object(async_sql_writer.logging, "exception") as mock_log:
      writer._log_task_error(task_with_step)
      mock_log.assert_called_once()
      self.assertEqual(mock_log.call_args[0][1], "step 42")
      self.assertEqual(mock_log.call_args[0][2], "r2")
      self.assertEqual(mock_log.call_args[0][3], "t2")

  # ============================================================================
  # Concurrency
  # ============================================================================

  def test_concurrent_writes(self) -> None:
    """Verifies concurrent writes from multiple threads across trajectories."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    num_threads = 4
    num_steps_per_thread = 5

    def worker_thread(thread_idx: int) -> None:
      traj_id = f"concurrent_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps_per_thread + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        writer.write_step(run_id="concurrent_run", metadata=meta, step=step)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
      for f in futures:
        f.result()

    writer.flush()
    writer.close()

    for thread_idx in range(num_threads):
      traj_id = f"concurrent_traj_{thread_idx}"
      trajs = schema_testing.fetch_all(
          self.engine,
          sa.select(schema.TRAJECTORIES_TABLE).where(
              schema.TRAJECTORIES_TABLE.c.trajectory_id == traj_id
          ),
      )
      self.assertLen(trajs, 1)
      steps = schema_testing.fetch_all(
          self.engine,
          sa.select(schema.STEPS_TABLE).where(
              schema.STEPS_TABLE.c.trajectory_id == traj_id
          ),
      )
      self.assertLen(steps, num_steps_per_thread)

  # ============================================================================
  # Input & Dialect Validation
  # ============================================================================

  def test_subclasses_base_async_writer(self) -> None:
    """Verifies AsyncSqlWriter inherits from BaseAsyncWriter."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.assertIsInstance(writer, base_writer.BaseAsyncWriter)
    writer.close()

  def test_postgresql_dialect_initialization(self) -> None:
    """Verifies AsyncSqlWriter binds postgresql.insert for Postgres dialect."""
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = "postgresql"
    writer = async_sql_writer.AsyncSqlWriter(engine=mock_engine)
    self.assertIs(writer._insert_fn, async_sql_writer.postgresql.insert)

  def test_unsupported_dialect_raises_value_error(self) -> None:
    """Verifies unsupported dialects raise a descriptive ValueError."""
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = "oracle"
    with self.assertRaisesRegex(ValueError, r"Unsupported database dialect"):
      async_sql_writer.AsyncSqlWriter(engine=mock_engine)

  def test_write_step_with_empty_trajectory_id_raises_value_error(self) -> None:
    """Verifies write_step raises ValueError for empty trajectory_id."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="   ",
        agent=trajectory_lib.Agent(name="agent_i", version="1.0"),
    )
    with self.assertRaisesRegex(
        ValueError, r"trajectory_id must be a non-empty string"
    ):
      writer.write_step(run_id="test_run", metadata=meta)
    writer.close()

  @parameterized.named_parameters(
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_write_step_with_empty_run_id_raises_value_error(
      self, invalid_run_id: str
  ) -> None:
    """Verifies write_step raises ValueError for empty or whitespace run_id."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_valid",
        agent=trajectory_lib.Agent(name="agent_i", version="1.0"),
    )
    with self.assertRaisesRegex(
        ValueError, r"run_id must be a non-empty string"
    ):
      writer.write_step(run_id=invalid_run_id, metadata=meta)
    writer.close()

  # ============================================================================
  # Shutdown & Destructor Teardown
  # ============================================================================

  def test_close_shuts_down_worker_and_prevents_further_writes(self) -> None:
    """Verifies close drains items, stops worker, and rejects future writes."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.write_step(
        run_id="close_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    writer.close()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)

    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      writer.write_step(
          run_id="close_run",
          metadata=trajectory_testing.METADATA_1,
          step=trajectory_testing.STEP_1_1,
      )

    # Calling close again is safe and idempotent.
    writer.close()

  def test_close_concurrent_with_writes(self) -> None:
    """Verifies writes during close are either persisted or cleanly rejected."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    num_writers = 8
    num_steps = 10
    accepted_steps: list[int] = []
    lock = threading.Lock()

    def write_worker(thread_idx: int) -> None:
      traj_id = f"concurrent_close_traj_{thread_idx}"
      meta = trajectory_testing.METADATA_1.model_copy(
          update={"trajectory_id": traj_id}
      )
      for step_id in range(1, num_steps + 1):
        step = trajectory_testing.STEP_1_1.model_copy(
            update={"step_id": step_id}
        )
        try:
          writer.write_step(run_id="close_run", metadata=meta, step=step)
          with lock:
            accepted_steps.append(step_id)
        except RuntimeError as e:
          if "Cannot write to a closed" in str(e):
            break
          raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_writers + 1
    ) as executor:
      write_futures = [
          executor.submit(write_worker, i) for i in range(num_writers)
      ]
      time.sleep(0.005)
      close_future = executor.submit(writer.close)

      for f in write_futures:
        f.result()
      close_future.result()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertEqual(len(steps), len(accepted_steps))

  def test_multiple_concurrent_close_calls_safe(self) -> None:
    """Verifies multiple threads calling close() terminate cleanly."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.write_step(
        run_id="close_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      futures = [executor.submit(writer.close) for _ in range(5)]
      for f in futures:
        f.result()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)

  def test_close_timeout_logs_warning(self) -> None:
    """Verifies close() logs warning if worker does not stop in timeout."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    block_event = threading.Event()
    worker_thread = None

    def blocking_process_task(task):
      del task
      block_event.wait()

    try:
      with mock.patch.object(
          writer, "_process_task", side_effect=blocking_process_task
      ):
        # Enqueue task 1: worker thread starts and blocks on task 1.
        writer.write_step(
            run_id="t1",
            metadata=trajectory_testing.METADATA_1,
            step=trajectory_testing.STEP_1_1,
        )
        time.sleep(0.05)

        # Enqueue task 2: stays in queue while worker is blocked on task 1.
        writer.write_step(
            run_id="t2",
            metadata=trajectory_testing.METADATA_2,
            step=trajectory_testing.STEP_2_1,
        )

        worker_thread = writer._worker_thread
        self.assertIsNotNone(worker_thread)

        with mock.patch.object(
            worker_thread, "is_alive", return_value=True
        ), mock.patch.object(logging, "warning", autospec=True) as mock_warning:
          writer.close(timeout=0.01)
          mock_warning.assert_called_once()
          call_args = mock_warning.call_args[0]
          self.assertIn("did not finish within timeout", call_args[0])
          self.assertIn(
              "Discarded remaining tasks for trajectory IDs", call_args[0]
          )
          self.assertEqual(call_args[1], "AsyncSqlWriter")
          self.assertEqual(call_args[3], [trajectory_testing.TRAJECTORY_ID_2])
    finally:
      block_event.set()
      if worker_thread is not None:
        worker_thread.join(timeout=5.0)

  def test_destructor_on_unstarted_writer(self) -> None:
    """Verifies __del__ on unstarted writer executes cleanly without errors."""
    unstarted_writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.assertIsNone(unstarted_writer._worker_thread)
    unstarted_writer.__del__()
    self.assertTrue(unstarted_writer._closed)

  def test_destructor_closes_worker_gracefully(self) -> None:
    """Verifies that __del__ closes the worker thread and flushes tasks."""
    fresh_writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    fresh_writer.write_step(
        run_id="del_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    worker_thread = fresh_writer._worker_thread
    self.assertIsNotNone(worker_thread)
    self.assertTrue(worker_thread.is_alive())

    fresh_writer.__del__()
    self.assertFalse(worker_thread.is_alive())
    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)


class AsyncSqlWriterShutdownHookTest(parameterized.TestCase):
  """Tests the atexit hook that drains writers still live at process exit."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
    schema.METADATA.create_all(self.engine)

  def test_live_writer_is_registered_and_unregistered_on_close(self) -> None:
    """Verifies writers track their liveness for the shutdown hook."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.assertIn(writer, base_writer._LIVE_WRITERS)

    writer.close()
    self.assertNotIn(writer, base_writer._LIVE_WRITERS)

  def test_pending_writes_persisted_by_shutdown_hook(self) -> None:
    """Verifies queued steps reach DB when hook runs, without flush()."""
    writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    writer.write_step(
        run_id="hook_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    base_writer._close_live_writers()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertTrue(writer._closed)

  def test_shutdown_hook_suppresses_close_errors(self) -> None:
    """Verifies one failing writer neither propagates nor blocks the others."""
    failing_writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    healthy_writer = async_sql_writer.AsyncSqlWriter(engine=self.engine)
    self.addCleanup(base_writer._LIVE_WRITERS.discard, failing_writer)

    healthy_writer.write_step(
        run_id="healthy_run",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )

    with mock.patch.object(
        failing_writer, "close", side_effect=RuntimeError("close failed")
    ):
      with mock.patch.object(logging, "exception") as mock_log_exception:
        base_writer._close_live_writers()

    mock_log_exception.assert_called_once()
    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertTrue(healthy_writer._closed)


if __name__ == "__main__":
  absltest.main()
