"""Asynchronous SQL writer for Trajectory Store using SQLAlchemy."""

from collections.abc import Callable
import datetime
from typing import Any

from absl import logging
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from tunix.experimental.trajectory import base_writer
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import sql_serialization
from tunix.experimental.trajectory import trajectory as trajectory_lib


def _resolve_status(
    metadata: trajectory_lib.TrajectoryMetadata,
    has_step: bool,
) -> str | None:
  """Resolves the trajectory lifecycle status from metadata or step presence.

  Resolution Order:
    1. Explicit top-level status attribute (e.g.
       TunixTrajectoryMetadata.status).
    2. COMPLETED if metadata.final_metrics is present (standard ATIF
       completion).
    3. RUNNING if a step is attached.
    4. None (indicates no status change; preserves existing database status).

  Args:
    metadata: TrajectoryMetadata instance.
    has_step: Whether a step is attached to the current write task.

  Returns:
    Status string or None if status should remain unchanged.
  """
  if isinstance(metadata, trajectory_lib.TunixTrajectoryMetadata):
    if metadata.status and metadata.status.strip():
      return metadata.status.strip()

  if metadata.final_metrics is not None:
    return schema.Status.COMPLETED

  if has_step:
    return schema.Status.RUNNING

  return None


def _validate_non_empty_str(value: str | None, name: str) -> str:
  """Validates that a string is non-empty and returns its stripped content.

  Args:
    value: String value to validate.
    name: Field name used in the error message.

  Returns:
    The stripped non-empty string.

  Raises:
    ValueError: If value is None, empty, or composed solely of whitespace.
  """
  stripped = value.strip() if value else ""
  if not stripped:
    raise ValueError(f"{name} must be a non-empty string.")
  return stripped


class AsyncSqlWriter(base_writer.BaseAsyncWriter):
  """Asynchronously writes trajectory metadata and step records to database.

  Architectural Decisions & Invariants:
    1. Single Dedicated Background Worker Thread:
       Inherits worker lifecycle, task queueing, barrier synchronization
       (`flush`), and `atexit` draining from `BaseAsyncWriter`.
    2. Synchronous SQLAlchemy Engine Execution:
       Executes synchronous SQLAlchemy statements on the background thread,
       avoiding asynchronous event loop complexities (`aiosqlite`/`greenlet`).
    3. Run Record Caching:
       Caches verified `run_id`s in memory (`_registered_run_ids`) to avoid
       redundant run table inserts on every step.
    4. Best-Effort Fault Tolerance:
       Persistence errors (transient database disconnections, lock timeouts)
       are logged with full traceback via `_log_task_error` and suppressed to
       prevent crashing training loops.
  """

  def __init__(self, engine: sa.Engine):
    """Initializes AsyncSqlWriter without starting the background worker.

    Args:
      engine: SQLAlchemy Engine instance connected to database.
    """
    super().__init__()
    # Synchronous SQLAlchemy engine providing database connectivity and dialect.
    self._engine = engine
    # Set of run IDs registered in the runs table to avoid redundant inserts.
    self._registered_run_ids: set[str] = set()

    # Dialect-specific insert statement constructor for ON CONFLICT DO UPDATE.
    self._insert_fn: Callable[..., Any]
    if engine.dialect.name == "postgresql":
      self._insert_fn = postgresql.insert
    elif engine.dialect.name == "sqlite":
      self._insert_fn = sqlite.insert
    else:
      raise ValueError(
          f"Unsupported database dialect: {engine.dialect.name}. "
          "Supported dialects are 'postgresql' and 'sqlite'."
      )

  def write_step(
      self,
      run_id: str,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Enqueues a trajectory metadata and/or step write operation.

    Args:
      run_id: Run identifier associated with written trajectory and steps.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside metadata.

    Raises:
      ValueError: If run_id is empty or whitespace, or if trajectory_id is
        missing or whitespace.
      RuntimeError: If the writer has already been closed.
    """
    clean_run_id = _validate_non_empty_str(run_id, "run_id")
    _validate_non_empty_str(metadata.trajectory_id, "trajectory_id")

    task = base_writer.WriteTask(
        run_id=clean_run_id,
        metadata=metadata.model_copy(deep=True),
        step=step.model_copy(deep=True) if step is not None else None,
    )
    self._enqueue(task, "AsyncSqlWriterWorker")

  def _process_task(self, task: base_writer.WriteTask) -> None:
    """Processes a single write task by upserting to SQL database.

    Args:
      task: Container holding run_id, metadata, and optional step.
    """
    run_id = _validate_non_empty_str(task.run_id, "run_id")
    traj_id = _validate_non_empty_str(
        task.metadata.trajectory_id, "trajectory_id"
    )

    now_dt = datetime.datetime.now(datetime.timezone.utc)

    # Check whether the run row has already been registered in the database.
    # We cache registered run_ids in memory to eliminate redundant database
    # roundtrips on every step write during RL rollouts.
    should_insert_run = run_id not in self._registered_run_ids
    with self._engine.begin() as conn:
      if should_insert_run:
        agent_name = (
            getattr(getattr(task.metadata, "agent", None), "name", None)
            or "unknown"
        )
        # Ensure the parent run entry exists to satisfy the foreign key
        # constraint in trajectories. on_conflict_do_nothing safely handles
        # concurrent insertions across distributed training workers or
        # pre-existing run registrations without overwriting metadata.
        run_insert_statement = (
            self._insert_fn(schema.RUNS_TABLE)
            .values(
                run_id=run_id,
                agent_name=agent_name,
                status=schema.Status.PENDING,
                created_at=now_dt,
                config={},
            )
            .on_conflict_do_nothing()
        )
        conn.execute(run_insert_statement)

      metadata_payload = sql_serialization.serialize_metadata(task.metadata)

      # Determine target status: uses explicit status, final_metrics
      # completion, or RUNNING if a step is attached. If updating stepless
      # metadata without status, target_status is None, preserving the existing
      # database status during updates.
      target_status = _resolve_status(
          task.metadata, has_step=task.step is not None
      )

      trajectory_update_set: dict[str, Any] = {
          "updated_at": now_dt,
          "trajectory_metadata": metadata_payload,
      }
      if target_status is not None:
        trajectory_update_set["status"] = target_status

      # Upsert trajectory row. on_conflict_do_update updates mutable lifecycle
      # fields (status, updated_at, and full metadata payload) while preserving
      # the current status during step-less metadata updates without status.
      trajectory_upsert_statement = (
          self._insert_fn(schema.TRAJECTORIES_TABLE)
          .values(
              run_id=run_id,
              trajectory_id=traj_id,
              status=target_status or schema.Status.PENDING,
              created_at=now_dt,
              updated_at=now_dt,
              trajectory_metadata=metadata_payload,
          )
          .on_conflict_do_update(
              index_elements=["run_id", "trajectory_id"],
              set_=trajectory_update_set,
          )
      )
      conn.execute(trajectory_upsert_statement)

      if task.step is not None:
        step_payload, step_time = sql_serialization.serialize_step(task.step)
        # Upsert step row. on_conflict_do_update ensures idempotent writes
        # if a step is retried or modified during rollout processing.
        step_upsert_statement = (
            self._insert_fn(schema.STEPS_TABLE)
            .values(
                run_id=run_id,
                trajectory_id=traj_id,
                step_id=task.step.step_id,
                payload=step_payload,
                created_at=step_time,
            )
            .on_conflict_do_update(
                index_elements=["run_id", "trajectory_id", "step_id"],
                set_={
                    "payload": step_payload,
                    "created_at": step_time,
                },
            )
        )
        conn.execute(step_upsert_statement)
    self._registered_run_ids.add(run_id)

  def _log_task_error(self, task: base_writer.WriteTask) -> None:
    """Logs detailed task error with trajectory and run context.

    Args:
      task: The write task that failed to process.
    """
    step_info = (
        f"step {task.step.step_id}" if task.step is not None else "metadata"
    )
    logging.exception(
        "Failed to write trajectory %s (run_id=%s, trajectory_id=%s) to"
        " database.",
        step_info,
        task.run_id,
        task.metadata.trajectory_id,
    )
