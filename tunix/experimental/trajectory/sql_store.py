"""SQL-backed implementation for Trajectory Store."""

import collections
from collections.abc import Callable
import datetime
import threading
import types
from typing import Any, Final, Self

from absl import logging
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from tunix.experimental.trajectory import async_writer
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib

POSTGRESQL_DIALECT: Final[str] = "postgresql"
SQLITE_DIALECT: Final[str] = "sqlite"

# Maximum number of trajectory metadata entries retained per run in the worker's
# bounded LRU cache to skip redundant trajectory table upserts across multi-step
# RL rollouts. In Tunix GRPO/Agentic RL (`agentic_rl_learner.py`), up to
# `full_batch_size * num_generations` trajectories are inflight per global step.
# With default `batch_size=128` (`train_deepscaler_nb.py`) and
# `num_generations=8` (`train_deepswe_nb.py`, `train_frozenlake.py`,
# `run_gsm8k_dist_grpo.py`), one rollout step runs 128 * 8 = 1,024 trajectories;
# 4,096 provides 4x headroom across overlapping steps (~4.5 MB RAM).
_MAX_CACHED_TRAJECTORIES: Final[int] = 4_096

# Process-local lock serializing concurrent schema initialization across threads
# within a single process (e.g. shared SQLite StaticPool engines).
_SCHEMA_INIT_LOCK: Final[threading.Lock] = threading.Lock()

# Deterministic 64-bit signed integer key for PostgreSQL `pg_advisory_xact_lock`
# during cold-start schema DDL initialization (derived from SHA-256 of
# `"TRAJECTORY_STORE"`).
_POSTGRES_SCHEMA_INIT_LOCK_ID: Final[int] = 0x568C418D_F1971EA0


def _to_utc_timestamp(dt: datetime.datetime | None) -> datetime.datetime:
  """Normalizes a datetime to a timezone-aware UTC timestamp.

  Args:
    dt: Datetime to normalize, or None. If naive, assumes UTC time.

  Returns:
    A timezone-aware UTC datetime. Defaults to the current UTC time if dt is
    None.
  """
  if dt is None:
    return datetime.datetime.now(datetime.timezone.utc)
  if dt.tzinfo is not None:
    return dt.astimezone(datetime.timezone.utc)
  return dt.replace(tzinfo=datetime.timezone.utc)


def _resolve_status(
    metadata: trajectory_lib.TrajectoryMetadata,
) -> schema.Status:
  """Returns the status of the trajectory as stated by the caller.

  Only the caller knows what state a trajectory is in, so the status is read
  from `metadata.get_extensions()` and normalized to `schema.Status`. The
  absence of a valid status returns `schema.Status.UNKNOWN`.

  Args:
    metadata: TrajectoryMetadata instance.

  Returns:
    The stated `schema.Status`, or `schema.Status.UNKNOWN` if absent/invalid.
  """
  raw_status = metadata.get_extensions().get("status")
  if isinstance(raw_status, str):
    try:
      return schema.Status(raw_status.strip().upper())
    except ValueError:
      pass
  return schema.Status.UNKNOWN


class _AsyncSqlWriter(async_writer.AsyncWriter[async_writer.WriteTask]):
  """Asynchronously writes trajectory metadata and step records to database.

  Architectural Decisions & Invariants:
    1. Single Dedicated Background Worker Thread:
       Inherits worker lifecycle, task queueing, barrier synchronization
       (`flush`), and `atexit` draining from `AsyncWriter`.
    2. Synchronous SQLAlchemy Engine Execution:
       Executes synchronous SQLAlchemy statements on the background thread,
       avoiding asynchronous event loop complexities (`aiosqlite`/`greenlet`).
    3. Unified Run & Trajectory Metadata Caching:
       Caches registered `run_id`s and a bounded post-commit `OrderedDict` LRU
       cache of committed `metadata_payload` dicts in a single nested mapping
       (`_trajectories_by_run: dict[str, OrderedDict[str, dict[str, Any]]]`),
       evicting the oldest 50% LRU chunk when capacity is reached to eliminate
       redundant database roundtrips and PostgreSQL MVCC/index bloat.
    4. Best-Effort Fault Tolerance:
       Persistence errors (transient database disconnections, lock timeouts)
       are logged with full traceback via `_log_task_error` and suppressed to
       prevent crashing training loops.
    5. Caller-Owned Status:
       Trajectory status is only ever reported by the caller. This writer never
       infers one; a row it creates starts at UNKNOWN, and a write that states
       no status leaves the stored value untouched.
  """

  def __init__(
      self,
      engine: sa.Engine,
      max_cached_trajectories: int = _MAX_CACHED_TRAJECTORIES,
  ):
    """Initializes _AsyncSqlWriter without starting the background worker.

    Args:
      engine: SQLAlchemy Engine instance connected to database.
      max_cached_trajectories: Maximum number of trajectory metadata entries
        retained per run in the in-memory LRU cache to skip redundant trajectory
        upserts. Set to 0 to disable client-side trajectory metadata caching.

    Raises:
      ValueError: If the engine uses an unsupported database dialect, or if
        max_cached_trajectories is negative.
    """
    super().__init__()
    if max_cached_trajectories < 0:
      raise ValueError("max_cached_trajectories must be non-negative.")
    # Synchronous SQLAlchemy engine providing database connectivity and dialect.
    self._engine = engine
    # Unified post-commit cache mapping each registered run_id to a bounded
    # OrderedDict of {trajectory_id: metadata_payload}. Outer keys track
    # registered runs; inner OrderedDict entries skip repeat trajectory upserts.
    self._max_cached_trajectories = max_cached_trajectories
    self._trajectories_by_run: dict[
        str, collections.OrderedDict[str, dict[str, Any]]
    ] = {}

    # Dialect-specific insert statement constructor for ON CONFLICT DO UPDATE.
    self._insert_fn: Callable[..., Any]
    if engine.dialect.name == POSTGRESQL_DIALECT:
      self._insert_fn = postgresql.insert
    elif engine.dialect.name == SQLITE_DIALECT:
      self._insert_fn = sqlite.insert
    else:
      raise ValueError(
          f"Unsupported database dialect: {engine.dialect.name}. "
          f"Supported dialects are {POSTGRESQL_DIALECT!r} and"
          f" {SQLITE_DIALECT!r}."
      )

  def enqueue_write(
      self,
      run_id: str,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Enqueues a trajectory metadata and/or step write operation.

    Identifiers are validated on the calling thread so malformed writes fail
    fast with actionable errors instead of surfacing later as suppressed worker
    errors. They are persisted exactly as supplied. `metadata` and `step` are
    deep copied by `WriteTask`.

    Args:
      run_id: Run identifier associated with written trajectory and steps.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside metadata.

    Raises:
      ValueError: If run_id is empty or whitespace, if trajectory_id is
        missing or whitespace, or if step has no step_id.
      RuntimeError: If the writer has already been closed.
    """
    if not run_id.strip():
      raise ValueError("run_id must be a non-empty string.")
    if metadata.trajectory_id is None or not metadata.trajectory_id.strip():
      raise ValueError("trajectory_id must be a non-empty string.")
    if step is not None and step.step_id is None:
      raise ValueError("Step must have a non-empty step_id.")

    self._enqueue(
        async_writer.WriteTask(run_id=run_id, metadata=metadata, step=step)
    )

  def _cache_committed_state(
      self,
      run_id: str,
      traj_id: str,
      metadata_payload: dict[str, Any],
  ) -> None:
    """Caches a committed run ID and trajectory metadata payload.

    Registers `run_id` in `_trajectories_by_run` and updates `traj_id` as the
    most recently used entry. When inserting a new `traj_id` at capacity,
    evicts the oldest 50% of least recently used entries in a single batch.

    Args:
      run_id: Committed run identifier.
      traj_id: Committed trajectory identifier.
      metadata_payload: Serialized JSON-compatible trajectory metadata dict.
    """
    metadata_by_traj = self._trajectories_by_run.setdefault(
        run_id, collections.OrderedDict()
    )
    if self._max_cached_trajectories <= 0:
      return

    if (
        traj_id not in metadata_by_traj
        and len(metadata_by_traj) >= self._max_cached_trajectories
    ):
      for _ in range(max(1, len(metadata_by_traj) // 2)):
        metadata_by_traj.popitem(last=False)

    metadata_by_traj[traj_id] = metadata_payload
    metadata_by_traj.move_to_end(traj_id)

  def _process_task(self, task: async_writer.WriteTask) -> None:
    """Processes a single write task by upserting to SQL database.

    Every statement for a task runs inside one transaction, so a trajectory row
    and its step land atomically. Run IDs and trajectory metadata states are
    cached only after that transaction commits, so a rolled back write is
    retried on the next task without violating foreign-key constraints.

    Database errors propagate to the worker loop, which logs them via
    `_log_task_error` and continues with the next task.

    Args:
      task: Container holding run_id, metadata, and optional step.
    """
    run_id = task.run_id
    assert run_id is not None
    traj_id = task.trajectory_id
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    metadata_payload = task.metadata.model_dump(mode="json", exclude_none=True)

    with self._engine.begin() as conn:
      self._ensure_run_exists(conn, run_id, task.metadata, now_dt)
      self._upsert_trajectory(
          conn, run_id, traj_id, task.metadata, metadata_payload, now_dt
      )
      if task.step is not None:
        self._upsert_step(conn, run_id, traj_id, task.step)

    self._cache_committed_state(run_id, traj_id, metadata_payload)

  def _ensure_run_exists(
      self,
      conn: sa.Connection,
      run_id: str,
      metadata: trajectory_lib.TrajectoryMetadata,
      now_dt: datetime.datetime,
  ) -> None:
    """Inserts the parent run row if it is not already known to exist.

    Registered run IDs are cached as outer keys in `_trajectories_by_run` to
    eliminate a database roundtrip on every step write during RL rollouts.
    `on_conflict_do_nothing` safely handles concurrent insertions across
    distributed training workers and pre-existing run registrations without
    overwriting their metadata.

    Args:
      conn: Open connection participating in the task's transaction.
      run_id: Validated non-empty run identifier.
      metadata: Trajectory metadata supplying the owning agent name.
      now_dt: Single timestamp shared by every row written for this task.
    """
    if run_id in self._trajectories_by_run:
      return

    run_insert_statement = (
        self._insert_fn(schema.RUNS_TABLE)
        .values(
            run_id=run_id,
            agent_name=metadata.agent.name,
            status=schema.Status.PENDING,
            created_at=now_dt,
            config={},
        )
        .on_conflict_do_nothing()
    )
    conn.execute(run_insert_statement)

  def _upsert_trajectory(
      self,
      conn: sa.Connection,
      run_id: str,
      traj_id: str,
      metadata: trajectory_lib.TrajectoryMetadata,
      metadata_payload: dict[str, Any],
      now_dt: datetime.datetime,
  ) -> None:
    """Upserts the trajectory row if its metadata has changed.

    Skips the database roundtrip entirely when `metadata_payload` matches the
    cached payload in `_trajectories_by_run`. On a cache miss,
    `on_conflict_do_update` refreshes `trajectory_metadata` and `updated_at`
    (plus `status` when provided) only when the incoming `trajectory_metadata`
    differs from the stored row.

    Args:
      conn: Open connection participating in the task's transaction.
      run_id: Validated non-empty run identifier.
      traj_id: Validated non-empty trajectory identifier.
      metadata: Trajectory metadata instance used to resolve status.
      metadata_payload: Serialized JSON-compatible trajectory metadata dict.
      now_dt: Single timestamp shared by every row written for this task.
    """
    if (
        self._trajectories_by_run.get(run_id, {}).get(traj_id)
        == metadata_payload
    ):
      return

    status = _resolve_status(metadata)

    # Columns to overwrite in an existing row.
    columns_to_overwrite: dict[str, Any] = {
        "updated_at": now_dt,
        "trajectory_metadata": metadata_payload,
    }
    if status != schema.Status.UNKNOWN:
      columns_to_overwrite["status"] = status

    trajectory_upsert_statement = (
        self._insert_fn(schema.TRAJECTORIES_TABLE)
        .values(
            run_id=run_id,
            trajectory_id=traj_id,
            status=status,
            created_at=now_dt,
            updated_at=now_dt,
            trajectory_metadata=metadata_payload,
        )
        .on_conflict_do_update(
            index_elements=["run_id", "trajectory_id"],
            set_=columns_to_overwrite,
            where=schema.TRAJECTORIES_TABLE.c.trajectory_metadata.is_distinct_from(
                metadata_payload
            ),
        )
    )
    conn.execute(trajectory_upsert_statement)

  def _upsert_step(
      self,
      conn: sa.Connection,
      run_id: str,
      traj_id: str,
      step: trajectory_lib.Step,
  ) -> None:
    """Upserts the step row attached to the trajectory.

    `on_conflict_do_update` replaces `payload` and `created_at` when the same
    step id is written again.

    Args:
      conn: Open connection participating in the task's transaction.
      run_id: Validated non-empty run identifier.
      traj_id: Validated non-empty trajectory identifier.
      step: The step to write.
    """
    step_payload = step.model_dump(mode="json", exclude_none=True)
    step_time = _to_utc_timestamp(step.timestamp)

    step_upsert_statement = (
        self._insert_fn(schema.STEPS_TABLE)
        .values(
            run_id=run_id,
            trajectory_id=traj_id,
            step_id=step.step_id,
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

  def _log_task_error(self, task: async_writer.WriteTask) -> None:
    """Logs detailed task error with trajectory and run context.

    Args:
      task: The write task that failed to process.
    """
    step_info = (
        f"step {task.step.step_id}" if task.step is not None else "metadata"
    )
    logging.exception(
        "%s failed to write trajectory %s (run_id=%s, trajectory_id=%s) to"
        " database.",
        self._thread_name,
        step_info,
        task.run_id,
        task.trajectory_id,
    )


class SqlTrajectoryStore(store.TrajectoryWriter):
  """SQL-backed implementation of TrajectoryWriter.

  `SqlTrajectoryStore` manages the persistence of reinforcement learning (RL)
  agent rollouts and step trajectories into relational database backends (e.g.
  SQLite, PostgreSQL) using SQLAlchemy.

  Architectural Separation of Responsibilities:
    `SqlTrajectoryStore` acts as a lightweight frontend responsible for:
    1. Schema initialization (`_initialize_schema`) and run scoping.
    2. Synchronous frontend input validation on the calling thread.
    3. Forwarding step write tasks, metadata updates, and flush barriers to
       `_AsyncSqlWriter`.

    All asynchronous queuing, background worker thread lifecycle, error
    suppression for rollout resilience, and database transactions are handled
    by `_AsyncSqlWriter`.
  """

  def __init__(
      self,
      engine: sa.Engine,
      run_id: str,
      auto_init: bool = True,
  ) -> None:
    """Initializes SqlTrajectoryStore.

    Args:
      engine: Configured SQLAlchemy Engine providing database connectivity. The
        caller retains ownership of `engine` and is responsible for calling
        `engine.dispose()` when the connection pool is no longer needed.
      run_id: Run identifier used to scope trajectories and steps. Lazily
        registered in `RUNS_TABLE` on the first write task once
        `TrajectoryMetadata` (e.g. `agent_name`) is provided.
      auto_init: If True, automatically creates database tables and indexes on
        startup via `_initialize_schema`.

    Raises:
      ValueError: If run_id is empty, None, or whitespace, or if the engine uses
        an unsupported database dialect.
    """
    if not run_id or not run_id.strip():
      raise ValueError("SqlTrajectoryStore requires a non-empty run_id.")
    self._engine = engine
    self._run_id = run_id.strip()
    self._writer = _AsyncSqlWriter(engine=engine)

    if auto_init:
      self._initialize_schema()

  def _has_all_schema_tables(self, conn: sa.Connection) -> bool:
    """Returns True if all Trajectory Store tables exist in the database."""
    existing_tables = set(sa.inspect(conn).get_table_names())
    return schema.METADATA.tables.keys() <= existing_tables

  def _acquire_schema_init_lock(self, conn: sa.Connection) -> None:
    """Acquires a dialect-specific transaction lock before running schema DDL."""
    dialect_name = self._engine.dialect.name
    if dialect_name == POSTGRESQL_DIALECT:
      conn.execute(
          sa.select(
              sa.func.pg_advisory_xact_lock(_POSTGRES_SCHEMA_INIT_LOCK_ID)
          )
      )
    elif dialect_name == SQLITE_DIALECT:
      conn.exec_driver_sql("BEGIN IMMEDIATE")

  def _initialize_schema(self) -> None:
    """Creates database tables and indexes safely under multi-worker concurrency.

    Implements a two-phase initialization protocol to prevent concurrent DDL
    race conditions when multiple distributed rollout workers start
    simultaneously:

    1. Read-Only Fast Path: Checks whether all required tables (`runs`,
       `trajectories`, `steps`) already exist. On warm databases, returns
       immediately without acquiring write locks or opening a write transaction.
    2. Serialized Transactional DDL: On cold databases, acquires a dialect-level
       transaction lock (`pg_advisory_xact_lock` on PostgreSQL or
       `BEGIN IMMEDIATE` on SQLite) inside `engine.begin()` and delegates table
       creation to `schema.METADATA.create_all(conn, checkfirst=True)`, which
       re-verifies table existence inside the lock before releasing it on
       commit.
    """
    with _SCHEMA_INIT_LOCK:
      with self._engine.connect() as conn:
        if self._has_all_schema_tables(conn):
          return

      with self._engine.begin() as conn:
        self._acquire_schema_init_lock(conn)
        schema.METADATA.create_all(conn, checkfirst=True)

  @property
  def engine(self) -> sa.Engine:
    """Returns the underlying SQLAlchemy engine."""
    return self._engine

  @property
  def run_id(self) -> str:
    """Returns the configured run identifier."""
    return self._run_id

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Asynchronously logs a turn step and its trajectory metadata.

    Validates input parameters on the calling thread so invalid IDs fail fast
    with actionable errors, then delegates asynchronous queuing and non-blocking
    database persistence to `_AsyncSqlWriter`.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    self._writer.enqueue_write(
        run_id=self._run_id, metadata=metadata, step=step
    )

  def update_metadata(
      self,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Updates or creates trajectory metadata asynchronously.

    Validates input parameters on the calling thread so invalid IDs fail fast
    with actionable errors, then delegates asynchronous queuing and non-blocking
    database persistence to `_AsyncSqlWriter`.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    self._writer.enqueue_write(
        run_id=self._run_id, metadata=metadata, step=None
    )

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage.

    Users do not need to call `flush()` in normal usage; it is primarily for
    testing.

    Delegates directly to `_AsyncSqlWriter.flush()` to provide strict barrier
    synchronization.
    """
    self._writer.flush()

  def close(self) -> None:
    """Flushes pending writes and shuts down the background writer thread.

    Calling `close()` is optional: the underlying `_AsyncSqlWriter` also drains
    itself at interpreter exit. It is worth calling explicitly for a store that
    becomes garbage well before the process ends, so its worker thread is
    released promptly. Closing is idempotent, but the store must not be written
    to afterwards; reads remain available. Does not dispose `self._engine` so
    shared engines remain usable; callers are responsible for calling
    `engine.dispose()`.
    """
    self._writer.close()

  def __enter__(self) -> Self:
    """Returns this store, for use as a context manager."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> None:
    """Closes the store on exiting the context manager."""
    del exc_type, exc_value, traceback
    self.close()
