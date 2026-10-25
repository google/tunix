"""SQL-backed implementation for Trajectory Store."""

import collections
from collections.abc import Callable, Mapping, Sequence
import datetime
import json
import threading
from typing import Any, ClassVar, Final, TypeVar

from absl import logging
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from tunix.dist_rl.trajectory import async_writer
from tunix.dist_rl.trajectory import base_store as store
from tunix.dist_rl.trajectory import db_engine
from tunix.dist_rl.trajectory import schema
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

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


def _sanitize_nul_bytes(obj: Any) -> Any:
  """Recursively strips \x00 NUL characters from string keys and values."""
  if isinstance(obj, str):
    return obj.replace("\x00", "")
  if isinstance(obj, dict):
    return {
        k.replace("\x00", "")
        if isinstance(k, str)
        else k: _sanitize_nul_bytes(v)
        for k, v in obj.items()
    }
  if isinstance(obj, list):
    return [_sanitize_nul_bytes(v) for v in obj]
  return obj


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
) -> str:
  """Returns the status of the trajectory as stated by the caller.

  Only the caller knows what state a trajectory is in, so the status is read
  from `metadata.get_extensions()` and written as-is when non-empty. The
  absence of a valid status returns `schema.Status.UNKNOWN`.

  Args:
    metadata: TrajectoryMetadata instance.

  Returns:
    The stated status string, or `schema.Status.UNKNOWN` if absent or blank.
  """
  raw_status = metadata.get_extensions().get("status")
  if isinstance(raw_status, str) and raw_status.strip():
    return raw_status.strip()
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
      run_id: str | None = None,
      max_cached_trajectories: int = _MAX_CACHED_TRAJECTORIES,
      step_array_filter: async_writer.StepArrayFilter | None = None,
  ):
    """Initializes _AsyncSqlWriter without starting the background worker.

    Args:
      engine: SQLAlchemy Engine instance connected to database.
      run_id: Optional default run identifier used for metrics writes.
      max_cached_trajectories: Maximum number of trajectory metadata entries
        retained per run in the in-memory LRU cache to skip redundant trajectory
        upserts. Set to 0 to disable client-side trajectory metadata caching.
      step_array_filter: Optional configuration for stripping per-token array
        fields before enqueuing or writing steps.

    Raises:
      ValueError: If the engine uses an unsupported database dialect, or if
        max_cached_trajectories is negative.
    """
    super().__init__(step_array_filter=step_array_filter)
    if max_cached_trajectories < 0:
      raise ValueError("max_cached_trajectories must be non-negative.")
    # Synchronous SQLAlchemy engine providing database connectivity and dialect.
    self._engine = engine
    self._run_id = run_id
    # Unified post-commit cache mapping each registered run_id to a bounded
    # OrderedDict of {trajectory_id: metadata_payload}. Outer keys track
    # registered runs; inner OrderedDict entries skip repeat trajectory upserts.
    self._max_cached_trajectories = max_cached_trajectories
    self._trajectories_by_run: dict[
        str, collections.OrderedDict[str, dict[str, Any]]
    ] = {}
    self._runs_without_metadata: set[str] = set()

    # Dialect-specific insert statement constructor for ON CONFLICT DO UPDATE.
    self._insert_fn: Callable[..., Any]
    if engine.dialect.name == db_engine.Dialect.POSTGRESQL:
      self._insert_fn = postgresql.insert
    elif engine.dialect.name == db_engine.Dialect.SQLITE:
      self._insert_fn = sqlite.insert
    else:
      raise ValueError(
          f"Unsupported database dialect: {engine.dialect.name}. "
          f"Supported dialects are {', '.join(db_engine.Dialect)}."
      )

  def _create_metrics_task(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> async_writer.WriteTask:
    """Factory creating a WriteTask instance for metrics updates."""
    return async_writer.WriteTask(
        run_id=self._run_id,
        trajectory_id=trajectory_id,
        metrics=metrics,
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
        async_writer.WriteTask(
            run_id=run_id,
            metadata=metadata,
            step=step,
            step_array_filter=self._step_array_filter,
        )
    )

  def enqueue_metrics_write(
      self,
      run_id: str,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Enqueues a trajectory metrics write operation.

    Identifiers and types are validated on the calling thread so malformed
    writes fail fast with actionable errors.

    Args:
      run_id: Run identifier associated with the trajectory.
      trajectory_id: Trajectory identifier.
      metrics: TrajectoryMetrics object containing metrics to persist.

    Raises:
      ValueError: If run_id or trajectory_id is empty or whitespace.
      TypeError: If metrics is not an instance of TrajectoryMetrics.
      RuntimeError: If the writer has already been closed.
    """
    if not run_id or not run_id.strip():
      raise ValueError("run_id must be a non-empty string.")
    if not trajectory_id or not trajectory_id.strip():
      raise ValueError("trajectory_id must be a non-empty string.")
    if not isinstance(metrics, trajectory_lib.TrajectoryMetrics):
      raise TypeError(
          "metrics must be an instance of TrajectoryMetrics, got"
          f" {type(metrics)!r}."
      )

    self._enqueue(
        async_writer.WriteTask(
            run_id=run_id.strip(),
            trajectory_id=trajectory_id.strip(),
            metrics=metrics,
        )
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
    self._runs_without_metadata.discard(run_id)
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
      task: Container holding run_id, metadata, metrics, and optional step.
    """
    traj_id = task.trajectory_id
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    if task.metrics is not None:
      run_id = task.run_id
      assert run_id is not None
      with self._engine.begin() as conn:
        self._ensure_run_exists(conn, run_id, task.metadata, now_dt)
        self._upsert_metrics(conn, run_id, traj_id, task.metrics, now_dt)
      if run_id not in self._trajectories_by_run:
        self._runs_without_metadata.add(run_id)
      return

    if task.metadata is not None:
      run_id = task.run_id
      assert run_id is not None
      metadata_payload = task.metadata.model_dump(
          mode="json", exclude_none=True
      )

      with self._engine.begin() as conn:
        self._ensure_run_exists(conn, run_id, task.metadata, now_dt)
        self._upsert_trajectory(
            conn, run_id, traj_id, task.metadata, metadata_payload, now_dt
        )
        if task.step is not None:
          self._upsert_step(conn, run_id, traj_id, task.step)

      self._cache_committed_state(run_id, traj_id, metadata_payload)
    elif task.step is not None:
      run_id = task.run_id
      assert run_id is not None
      with self._engine.begin() as conn:
        self._upsert_step(conn, run_id, traj_id, task.step)

  def _ensure_run_exists(
      self,
      conn: sa.Connection,
      run_id: str,
      metadata: trajectory_lib.TrajectoryMetadata | None,
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
      metadata: Optional trajectory metadata supplying the owning agent name.
      now_dt: Single timestamp shared by every row written for this task.
    """
    if run_id in self._trajectories_by_run:
      return
    if metadata is None and run_id in self._runs_without_metadata:
      return

    agent_name = metadata.agent.name if metadata is not None else ""
    run_insert = self._insert_fn(schema.RUNS_TABLE).values(
        run_id=run_id,
        agent_name=agent_name,
        status=schema.Status.PENDING,
        created_at=now_dt,
        config={},
    )
    if metadata is not None and metadata.agent.name:
      run_insert_statement = run_insert.on_conflict_do_update(
          index_elements=["run_id"],
          set_={"agent_name": metadata.agent.name},
          where=(schema.RUNS_TABLE.c.agent_name == ""),
      )
    else:
      run_insert_statement = run_insert.on_conflict_do_nothing(
          index_elements=["run_id"]
      )
    conn.execute(run_insert_statement)

  def _upsert_metrics(
      self,
      conn: sa.Connection,
      run_id: str,
      traj_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
      now_dt: datetime.datetime,
  ) -> None:
    """Upserts trajectory metrics with shallow-merge semantics."""
    sanitized_payload = _sanitize_nul_bytes(metrics.model_dump(mode="json"))

    if self._engine.dialect.name == db_engine.Dialect.POSTGRESQL:
      pg_insert_fn = postgresql.insert(schema.TRAJECTORY_METRICS_TABLE)
      upsert_statement = pg_insert_fn.values(
          run_id=run_id,
          trajectory_id=traj_id,
          payload=sanitized_payload,
          created_at=now_dt,
          updated_at=now_dt,
      ).on_conflict_do_update(
          index_elements=["run_id", "trajectory_id"],
          set_={
              "updated_at": now_dt,
              "payload": sa.text(
                  "CASE"
                  " WHEN (trajectory_metrics.payload->>'extra') IS NULL"
                  " AND (EXCLUDED.payload->>'extra') IS NULL"
                  " THEN EXCLUDED.payload"
                  " WHEN (EXCLUDED.payload->>'extra') IS NULL"
                  " THEN jsonb_set(EXCLUDED.payload, '{extra}',"
                  " trajectory_metrics.payload->'extra')"
                  " ELSE jsonb_set(EXCLUDED.payload, '{extra}',"
                  " COALESCE(NULLIF(trajectory_metrics.payload->'extra',"
                  " 'null'::jsonb), '{}'::jsonb) ||"
                  " (EXCLUDED.payload->'extra'))"
                  " END"
              ),
          },
      )
      conn.execute(upsert_statement)
    else:
      existing_row = conn.execute(
          sa.select(schema.TRAJECTORY_METRICS_TABLE.c.payload).where(
              schema.TRAJECTORY_METRICS_TABLE.c.run_id == run_id,
              schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id == traj_id,
          )
      ).scalar_one_or_none()

      if existing_row is not None:
        existing_payload = (
            json.loads(existing_row)
            if isinstance(existing_row, str)
            else existing_row
        )
        existing_extra = existing_payload.get("extra")
        incoming_extra = sanitized_payload.get("extra")
        if incoming_extra is None:
          merged_extra = existing_extra
        elif existing_extra is None:
          merged_extra = incoming_extra
        else:
          merged_extra = {
              **existing_extra,
              **incoming_extra,
          }
        merged_payload = {**sanitized_payload, "extra": merged_extra}
        conn.execute(
            sa.update(schema.TRAJECTORY_METRICS_TABLE)
            .where(
                schema.TRAJECTORY_METRICS_TABLE.c.run_id == run_id,
                schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id == traj_id,
            )
            .values(
                payload=merged_payload,
                updated_at=now_dt,
            )
        )
      else:
        conn.execute(
            self._insert_fn(schema.TRAJECTORY_METRICS_TABLE).values(
                run_id=run_id,
                trajectory_id=traj_id,
                payload=sanitized_payload,
                created_at=now_dt,
                updated_at=now_dt,
            )
        )

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
    if task.step is not None:
      task_info = f"step {task.step.step_id}"
    elif task.metrics is not None:
      task_info = "metrics"
    else:
      task_info = "metadata"
    logging.exception(
        "%s failed to write trajectory %s (run_id=%s, trajectory_id=%s) to"
        " database.",
        self._thread_name,
        task_info,
        task.run_id,
        task.trajectory_id,
    )


class SqlTrajectoryStore(store.TrajectoryStore[MetadataT]):
  """SQL-backed implementation of TrajectoryReader and TrajectoryWriter.

  `SqlTrajectoryStore` manages the persistence and retrieval of reinforcement
  learning (RL) agent rollouts and step trajectories in relational database
  backends (e.g. SQLite, PostgreSQL) using SQLAlchemy.

  Architectural Separation of Responsibilities:
    `SqlTrajectoryStore` acts as a lightweight frontend responsible for:
    1. Schema initialization (`_initialize_schema`) and run scoping.
    2. Synchronous frontend input validation on the calling thread.
    3. Forwarding step write tasks, metadata updates, and flush barriers to
       `_AsyncSqlWriter`.
    4. Synchronous read queries (`get_trajectories()` and
       `get_trajectories_metadata()`) via `self._engine.connect()`.

    The store owns an engine handle, which wraps the engine. The handle is
    acquired from `db_engine` on construction and released on `close()`;
    `db_engine` manages disposing the engine once its handles are released.

    All asynchronous queuing, background worker thread lifecycle, error
    suppression for rollout resilience, and database transactions are handled
    by `_AsyncSqlWriter`.
  """

  BACKEND: ClassVar[str] = "sql"

  def __init__(
      self,
      *,
      run_id: str,
      db_url: str,
      auto_init: bool = True,
      step_array_filter: async_writer.StepArrayFilter | None = None,
  ) -> None:
    """Initializes SqlTrajectoryStore.

    Args:
      run_id: Run identifier used to scope trajectories and steps. Lazily
        registered in `RUNS_TABLE` on the first write task once
        `TrajectoryMetadata` (e.g. `agent_name`) is provided.
      db_url: Database connection URL, e.g. 'sqlite:///traj.db' or
        'postgresql+psycopg2://user@host/db'. Callers resolve any credentials
        (e.g. from a secret manager) before passing it. A PostgreSQL URL may
        omit the password; libpq then reads `PGPASSWORD` or `~/.pgpass`.
      auto_init: If True, automatically creates database tables and indexes on
        startup via `_initialize_schema`.
      step_array_filter: Optional configuration for stripping per-token array
        fields before enqueuing or writing steps.

    Raises:
      ValueError: If `run_id` or `db_url` is empty, None, or whitespace, or if
        `db_url` uses an unsupported database dialect.
    """
    super().__init__()
    if not run_id or not run_id.strip():
      raise ValueError("SqlTrajectoryStore requires a non-empty run_id.")
    if not db_url or not db_url.strip():
      raise ValueError("SqlTrajectoryStore requires a non-empty db_url.")

    self._db_url = db_url
    self._run_id = run_id.strip()
    self._engine_handle = db_engine.acquire_engine(
        db_engine.EngineConfig(url=db_url)
    )
    self._engine = self._engine_handle.engine
    try:
      self._writer = _AsyncSqlWriter(
          engine=self._engine,
          run_id=self._run_id,
          step_array_filter=step_array_filter,
      )
    except Exception:
      self._engine_handle.release()
      raise
    if auto_init:
      try:
        self._initialize_schema()
      except Exception:
        # Also closes the writer, so it is not left in the interpreter-exit
        # drain with a disposed engine.
        self.close()
        raise

  @classmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
  ) -> "SqlTrajectoryStore[MetadataT]":
    """Builds a SQL-backed store from `config`.

    Args:
      config: Requires "db_url" and "run_id".

    Returns:
      A new SqlTrajectoryStore.

    Raises:
      ValueError: If "db_url" or "run_id" is missing or empty.
    """
    return cls(
        run_id=config.get("run_id", ""),
        db_url=config.get("db_url", ""),
    )

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    The returned "db_url" is the literal URL this store was built with,
    including any password it embeds. Do not log it; log
    `to_redacted_config()` instead.
    """
    return {
        "backend": self.BACKEND,
        "db_url": self._db_url,
        "run_id": self._run_id,
    }

  def to_redacted_config(self) -> dict[str, Any]:
    """Returns `to_config()` with any password in "db_url" masked."""
    config = self.to_config()
    config["db_url"] = db_engine.redact_url(self._db_url)
    return config

  def _has_all_schema_tables(self, conn: sa.Connection) -> bool:
    """Returns True if all Trajectory Store tables exist in the database."""
    existing_tables = set(sa.inspect(conn).get_table_names())
    return schema.METADATA.tables.keys() <= existing_tables

  def _acquire_schema_init_lock(self, conn: sa.Connection) -> None:
    """Acquires a dialect-specific transaction lock before running schema DDL."""
    dialect_name = self._engine.dialect.name
    if dialect_name == db_engine.Dialect.POSTGRESQL:
      conn.execute(
          sa.select(
              sa.func.pg_advisory_xact_lock(_POSTGRES_SCHEMA_INIT_LOCK_ID)
          )
      )
    elif dialect_name == db_engine.Dialect.SQLITE:
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
       `BEGIN IMMEDIATE` on SQLite) inside `engine.begin()`, re-checks table
       existence in case another worker initialized the schema while waiting on
       the lock, and delegates table creation to
       `schema.METADATA.create_all(conn, checkfirst=True)` before releasing the
       lock on commit.
    """
    with _SCHEMA_INIT_LOCK:
      with self._engine.connect() as conn:
        if self._has_all_schema_tables(conn):
          return

      with self._engine.begin() as conn:
        self._acquire_schema_init_lock(conn)
        if self._has_all_schema_tables(conn):
          return
        schema.METADATA.create_all(conn, checkfirst=True)

  @property
  def engine(self) -> sa.Engine:
    """Returns the store's SQLAlchemy engine."""
    return self._engine

  @property
  def run_id(self) -> str:
    """Returns the configured run identifier."""
    return self._run_id

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: MetadataT,
  ) -> None:
    """Asynchronously logs a turn step and its trajectory metadata.

    Validates input parameters on the calling thread so invalid IDs fail fast
    with actionable errors, then delegates asynchronous queuing and non-blocking
    database persistence to `_AsyncSqlWriter`.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
      ValueError: If metadata.trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    self._validate_and_bind_metadata(metadata)
    self._writer.enqueue_write(
        run_id=self._run_id, metadata=metadata, step=step
    )

  def update_metadata(
      self,
      metadata: MetadataT,
  ) -> None:
    """Updates or creates trajectory metadata asynchronously.

    Validates input parameters on the calling thread so invalid IDs fail fast
    with actionable errors, then delegates asynchronous queuing and non-blocking
    database persistence to `_AsyncSqlWriter`.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
      ValueError: If metadata.trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    self._validate_and_bind_metadata(metadata)
    self._writer.enqueue_write(
        run_id=self._run_id, metadata=metadata, step=None
    )

  def update_metrics(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Updates or creates trajectory metrics asynchronously.

    Validates input parameters on the calling thread so invalid inputs fail
    fast with actionable errors, then delegates asynchronous queuing and
    non-blocking database persistence to `_AsyncSqlWriter`.

    Args:
      trajectory_id: Unique identifier for the trajectory.
      metrics: TrajectoryMetrics containing the metrics data to record.

    Raises:
      TypeError: If metrics is not a TrajectoryMetrics instance.
      ValueError: If trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    if not trajectory_id or not trajectory_id.strip():
      raise ValueError("SqlTrajectoryStore requires a non-empty trajectory_id.")
    if not isinstance(metrics, trajectory_lib.TrajectoryMetrics):
      raise TypeError(
          "metrics must be an instance of TrajectoryMetrics, got"
          f" {type(metrics)!r}."
      )
    self._writer.enqueue_metrics_write(
        run_id=self._run_id,
        trajectory_id=trajectory_id.strip(),
        metrics=metrics,
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
    """Flushes pending writes, stops the writer thread, and releases the engine.

    Calling `close()` is optional because the writer also drains at interpreter
    exit. Call it explicitly to free the writer thread for a store that is
    discarded long before the process ends.

    Closing is idempotent. Do not write to the store afterwards. After closing,
    reads still work on file SQLite or PostgreSQL, since the data lives outside
    the connection pool. With in-memory SQLite the data is lost, because
    closing currently disposes the only connection holding it.
    """
    try:
      self._writer.close()
    finally:
      self._engine_handle.release()

  def get_trajectories_metadata(
      self, trajectory_ids: Sequence[str] | None = None
  ) -> list[MetadataT]:
    """Retrieves metadata for trajectories in the run.

    Args:
      trajectory_ids: Optional sequence of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.

    Returns:
      A list of TrajectoryMetadata objects for the requested trajectories.

    Raises:
      store.TrajectoryMetadataNotFoundError: If any requested trajectory ID does
        not exist.
    """
    # Query all trajectory metadata in the run when no ID filter is provided.
    if trajectory_ids is None:
      statement = (
          sa.select(schema.TRAJECTORIES_TABLE.c.trajectory_metadata)
          .where(schema.TRAJECTORIES_TABLE.c.run_id == self._run_id)
          .order_by(
              schema.TRAJECTORIES_TABLE.c.created_at,
              schema.TRAJECTORIES_TABLE.c.trajectory_id,
          )
      )
      with self._engine.connect() as conn:
        metadata_rows = conn.execute(statement)
        return [
            self._load_metadata(row) for row in metadata_rows.scalars()
        ]

    # Query only the requested trajectory IDs when a sequence is provided.
    if not trajectory_ids:
      return []

    statement = sa.select(
        schema.TRAJECTORIES_TABLE.c.trajectory_id,
        schema.TRAJECTORIES_TABLE.c.trajectory_metadata,
    ).where(
        schema.TRAJECTORIES_TABLE.c.run_id == self._run_id,
        schema.TRAJECTORIES_TABLE.c.trajectory_id.in_(trajectory_ids),
    )
    with self._engine.connect() as conn:
      metadata_rows = conn.execute(statement)
      metadata_by_trajectory_id = {
          row.trajectory_id: self._load_metadata(row.trajectory_metadata)
          for row in metadata_rows
      }
    metadata_list = []
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metadata_by_trajectory_id:
        raise store.TrajectoryMetadataNotFoundError(trajectory_id)
      metadata_list.append(metadata_by_trajectory_id[trajectory_id])
    return metadata_list

  def get_trajectories(
      self, trajectory_ids: Sequence[str]
  ) -> list[trajectory_lib.Trajectory[Any]]:
    """Retrieves full trajectories for a sequence of trajectory IDs.

    Args:
      trajectory_ids: Sequence of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects corresponding to the requested IDs.

    Raises:
      store.TrajectoryNotFoundError: If any requested trajectory ID does not
        exist.
    """
    if not trajectory_ids:
      return []

    metadata_statement = sa.select(
        schema.TRAJECTORIES_TABLE.c.trajectory_id,
        schema.TRAJECTORIES_TABLE.c.trajectory_metadata,
    ).where(
        schema.TRAJECTORIES_TABLE.c.run_id == self._run_id,
        schema.TRAJECTORIES_TABLE.c.trajectory_id.in_(trajectory_ids),
    )
    steps_statement = (
        sa.select(
            schema.STEPS_TABLE.c.trajectory_id,
            schema.STEPS_TABLE.c.payload,
        )
        .where(
            schema.STEPS_TABLE.c.run_id == self._run_id,
            schema.STEPS_TABLE.c.trajectory_id.in_(trajectory_ids),
        )
        .order_by(
            schema.STEPS_TABLE.c.trajectory_id,
            schema.STEPS_TABLE.c.step_id,
        )
    )

    with self._engine.connect() as conn:
      metadata_rows = conn.execute(metadata_statement)
      metadata_by_trajectory_id = {
          trajectory_id: payload for trajectory_id, payload in metadata_rows
      }
      step_rows = conn.execute(steps_statement)
      steps_by_trajectory_id: dict[str, list[dict[str, Any]]] = (
          collections.defaultdict(list)
      )
      for trajectory_id, payload in step_rows:
        steps_by_trajectory_id[trajectory_id].append(payload)

    trajectories: list[trajectory_lib.Trajectory[Any]] = []
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metadata_by_trajectory_id:
        raise store.TrajectoryNotFoundError(trajectory_id)
      meta = self._load_metadata(metadata_by_trajectory_id[trajectory_id])
      steps = [
          trajectory_lib.Step.model_validate(step_payload)
          for step_payload in steps_by_trajectory_id.get(trajectory_id, [])
      ]
      trajectories.append(meta.create_trajectory(steps=steps))

    return trajectories

  def get_trajectories_metrics(
      self, trajectory_ids: Sequence[str] | None = None
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    """Retrieves metrics for trajectories in the run.

    Args:
      trajectory_ids: Optional sequence of unique trajectory identifiers. If
        specified, only metrics for these IDs is returned. If None, metrics for
        all trajectories in the run with recorded metrics is returned.

    Returns:
      A dictionary mapping trajectory IDs to TrajectoryMetrics objects for the
      requested trajectories, in the order of `trajectory_ids` if specified.

    Raises:
      store.TrajectoryMetricsNotFoundError: If any requested trajectory ID does
        not have recorded metrics.
    """
    if trajectory_ids is None:
      statement = (
          sa.select(
              schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id,
              schema.TRAJECTORY_METRICS_TABLE.c.payload,
          )
          .where(schema.TRAJECTORY_METRICS_TABLE.c.run_id == self._run_id)
          .order_by(
              schema.TRAJECTORY_METRICS_TABLE.c.created_at,
              schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id,
          )
      )
      with self._engine.connect() as conn:
        rows = conn.execute(statement)
        return {
            row.trajectory_id: self._load_metrics(row.payload) for row in rows
        }

    if not trajectory_ids:
      return {}

    statement = sa.select(
        schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id,
        schema.TRAJECTORY_METRICS_TABLE.c.payload,
    ).where(
        schema.TRAJECTORY_METRICS_TABLE.c.run_id == self._run_id,
        schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id.in_(trajectory_ids),
    )
    with self._engine.connect() as conn:
      rows = conn.execute(statement)
      metrics_by_id = {
          row.trajectory_id: self._load_metrics(row.payload) for row in rows
      }

    missing_ids = [tid for tid in trajectory_ids if tid not in metrics_by_id]
    if missing_ids:
      raise store.TrajectoryMetricsNotFoundError(missing_ids)

    return {tid: metrics_by_id[tid] for tid in trajectory_ids}

  def _load_metrics(
      self, payload: dict[str, Any] | str
  ) -> trajectory_lib.TrajectoryMetrics:
    """Rehydrates a stored trajectory metrics payload."""
    if isinstance(payload, str):
      payload = json.loads(payload)
    return trajectory_lib.TrajectoryMetrics.model_validate(payload)

  def _load_metadata(self, payload: dict[str, Any]) -> MetadataT:
    """Rehydrates a stored base-ATIF metadata payload."""
    base_meta = trajectory_lib.TrajectoryMetadata.model_validate(payload)
    return self._rehydrate_metadata(base_meta)
