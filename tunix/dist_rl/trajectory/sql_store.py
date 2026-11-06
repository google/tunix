"""SQL-backed implementation for Trajectory Store."""

import collections
from collections.abc import Callable, Mapping, Sequence
import datetime
import threading
from typing import Any, ClassVar, Final, TypeVar

from absl import logging
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from tunix.dist_rl.trajectory import async_writer
from tunix.dist_rl.trajectory import base_store as store
from tunix.dist_rl.trajectory import db_engine
from tunix.dist_rl.trajectory import pg_wire
from tunix.dist_rl.trajectory import schema
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

# Maximum number of trajectory metadata entries retained per run in the worker's
# bounded LRU cache to skip redundant trajectory table upserts across multi-step
# RL rollouts. In Tunix GRPO/Agentic RL (`agentic_rl_learner.py`), up to
# `full_batch_size * num_generations` trajectories are inflight per global step.
# With default `batch_size=128` (`train_deepscaler_nb.py`) and
# `num_generations=8` (`train_deepswe_nb.py`, `train_frozenlake.py`,
# `math_gsm8k_dist/gsm8k.yaml`), one rollout step runs 128 * 8 = 1,024
# trajectories; 4,096 provides 4x headroom across overlapping steps (~4.5 MB
# RAM).
_MAX_CACHED_TRAJECTORIES: Final[int] = 4_096

# Process-local lock serializing concurrent schema initialization across threads
# within a single process (e.g. shared SQLite StaticPool engines).
_SCHEMA_INIT_LOCK: Final[threading.Lock] = threading.Lock()

# Deterministic 64-bit signed integer key for PostgreSQL `pg_advisory_xact_lock`
# during cold-start schema DDL initialization (derived from SHA-256 of
# `"TRAJECTORY_STORE"`).
_POSTGRES_SCHEMA_INIT_LOCK_ID: Final[int] = 0x568C418D_F1971EA0


def _validate_non_empty_str(value: str | None, field_name: str) -> str:
  """Validates that `value` is a non-empty, non-whitespace string."""
  if not value or not value.strip():
    raise ValueError(f"{field_name} must be a non-empty string.")
  return value


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


def _build_postgres_metrics_merge_payload(
    metrics_insert: postgresql.Insert,
) -> sa.ColumnElement[Any]:
  """Returns a PostgreSQL JSONB expression that shallow-merges `extra`."""
  stored_extra = schema.TRAJECTORY_METRICS_TABLE.c.payload["extra"]
  incoming_extra = metrics_insert.excluded.payload["extra"]
  merged_extra = sa.case(
      (
          sa.func.jsonb_typeof(stored_extra) == "object",
          stored_extra.concat(incoming_extra),
      ),
      else_=incoming_extra,
  )
  return sa.func.jsonb_set(
      metrics_insert.excluded.payload,
      sa.literal_column("'{extra}'"),
      merged_extra,
  )


class _AsyncSqlWriter(async_writer.AsyncWriter[async_writer.WriteTask]):
  """Asynchronously writes trajectory metadata, step, and metrics records.

  Architectural Decisions & Invariants:
    1. Single Dedicated Background Worker Thread:
       Inherits worker lifecycle, task queueing, barrier synchronization
       (`flush`), and `atexit` draining from `AsyncWriter`.
    2. Synchronous SQLAlchemy Engine Execution:
       Executes synchronous SQLAlchemy statements on the background thread,
       avoiding asynchronous event loop complexities (`aiosqlite`/`greenlet`).
    3. Post-Commit Run & Trajectory Metadata Caching:
       Caches committed `run_id -> agent_name` registrations in
       `_run_agent_by_id` and a bounded post-commit `OrderedDict` LRU cache of
       committed `metadata_payload` dicts in `_trajectories_by_run`, evicting
       the oldest 50% LRU chunk when capacity is reached to eliminate redundant
       database roundtrips and PostgreSQL MVCC/index bloat.
    4. Unified 3-Stage Transactional Pipeline (Single Task & Multi-Row Batch):
       Both single-task steady-state writes (`_process_task`) and contention
       batches (`_process_batch`) execute `_execute_batch_transaction`:
         a. In-memory normalization and cache filtering per table.
         b. Foreign-key-ordered multi-row `INSERT ... ON CONFLICT` statements
            inside a single transaction (`runs` -> `trajectories` -> `steps`
            and `trajectory_metrics`).
         c. Post-commit cache updates once the transaction commits.
       Non-batchable bulk errors fall back to `super()._process_batch` to
       isolate the failing task and persist the remaining valid tasks.
    5. Best-Effort Fault Tolerance:
       Persistence errors (transient database disconnections, lock timeouts)
       are logged with full traceback via `_log_task_error` (or at the batch
       level when a bulk contention error aborts a batch) and suppressed to
       prevent crashing training loops.
    6. Caller-Owned Status:
       Trajectory status is only ever reported by the caller. This writer never
       infers one; a row it creates starts at UNKNOWN, and a write that states
       no status leaves the stored value untouched.
  """

  def __init__(
      self,
      engine: sa.Engine,
      max_cached_trajectories: int = _MAX_CACHED_TRAJECTORIES,
      batching: async_writer.BatchConfig | None = None,
      copy_step: bool = False,
  ):
    """Initializes _AsyncSqlWriter without starting the background worker.

    Args:
      engine: SQLAlchemy Engine instance connected to database.
      max_cached_trajectories: Maximum number of trajectory metadata entries
        retained per run in the in-memory LRU cache to skip redundant trajectory
        upserts. Set to 0 to disable client-side trajectory metadata caching.
      batching: Optional BatchConfig governing how queued tasks are grouped into
        multi-row SQL transactions.
      copy_step: If True, deep copies step objects before enqueuing. Defaults to
        False (zero-copy step handover) to avoid synchronous GIL contention.

    Raises:
      ValueError: If the engine uses an unsupported database dialect, or if
        max_cached_trajectories is negative.
    """
    super().__init__(batching=batching, copy_step=copy_step)
    if max_cached_trajectories < 0:
      raise ValueError("max_cached_trajectories must be non-negative.")
    # Synchronous SQLAlchemy engine providing database connectivity and dialect.
    self._engine = engine
    # Post-commit cache mapping committed run_id -> agent_name (empty until
    # non-empty metadata arrives) to skip repeat `runs` upserts.
    self._run_agent_by_id: dict[str, str] = {}
    # Bounded post-commit LRU cache of {trajectory_id: metadata_payload} per
    # run to skip repeat `trajectories` upserts.
    self._max_cached_trajectories = max_cached_trajectories
    self._trajectories_by_run: dict[
        str, collections.OrderedDict[str, dict[str, Any]]
    ] = {}

    # Dialect-specific insert statement constructor for ON CONFLICT DO UPDATE.
    insert_fn_by_dialect: Mapping[str, Callable[..., Any]] = {
        db_engine.Dialect.POSTGRESQL: postgresql.insert,
        db_engine.Dialect.SQLITE: sqlite.insert,
    }
    insert_fn = insert_fn_by_dialect.get(engine.dialect.name)
    if insert_fn is None:
      raise ValueError(
          f"Unsupported database dialect: {engine.dialect.name}. "
          f"Supported dialects are {', '.join(db_engine.Dialect)}."
      )
    self._insert_fn: Callable[..., Any] = insert_fn

  def enqueue_write(
      self,
      run_id: str,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Enqueues a trajectory metadata and/or step write operation.

    Identifiers are validated on the calling thread so malformed writes fail
    fast with actionable errors instead of surfacing later as suppressed worker
    errors. They are persisted exactly as supplied. `metadata` is always deep
    copied by `WriteTask`; `step` is handed over with zero copy by default and
    deep copied only when `copy_step` is True.

    Args:
      run_id: Run identifier associated with written trajectory and steps.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside metadata.

    Raises:
      ValueError: If run_id is empty or whitespace, if trajectory_id is
        missing or whitespace, or if step has no step_id.
      RuntimeError: If the writer has already been closed.
    """
    _validate_non_empty_str(run_id, "run_id")
    _validate_non_empty_str(metadata.trajectory_id, "trajectory_id")
    if step is not None and step.step_id is None:
      raise ValueError("Step must have a non-empty step_id.")

    self._enqueue(
        async_writer.WriteTask(
            run_id=run_id,
            metadata=metadata,
            step=step,
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
    _validate_non_empty_str(run_id, "run_id")
    _validate_non_empty_str(trajectory_id, "trajectory_id")
    if not isinstance(metrics, trajectory_lib.TrajectoryMetrics):
      raise TypeError(
          f"Expected TrajectoryMetrics instance, got {type(metrics).__name__}."
      )

    self._enqueue(
        async_writer.WriteTask(
            run_id=run_id,
            trajectory_id=trajectory_id,
            metrics=metrics,
        )
    )

  def _cache_committed_state(
      self,
      run_id: str,
      traj_id: str,
      metadata_payload: dict[str, Any],
  ) -> None:
    """Caches a committed trajectory metadata payload in the per-run LRU cache.

    Updates `traj_id` as the most recently used entry for `run_id`. When
    inserting a new `traj_id` at capacity, evicts the oldest 50% of least
    recently used entries in a single batch.

    Args:
      run_id: Committed run identifier.
      traj_id: Committed trajectory identifier.
      metadata_payload: Serialized JSON-compatible trajectory metadata dict.
    """
    if self._max_cached_trajectories <= 0:
      return
    metadata_by_traj = self._trajectories_by_run.setdefault(
        run_id, collections.OrderedDict()
    )

    if (
        traj_id not in metadata_by_traj
        and len(metadata_by_traj) >= self._max_cached_trajectories
    ):
      for _ in range(max(1, len(metadata_by_traj) // 2)):
        metadata_by_traj.popitem(last=False)

    metadata_by_traj[traj_id] = metadata_payload
    metadata_by_traj.move_to_end(traj_id)

  def _should_batch_on_error(self, exc: BaseException) -> bool:
    """Returns True for database lock contention or pool timeout errors."""
    return isinstance(
        exc,
        (
            sa.exc.OperationalError,
            sa.exc.TimeoutError,
        ),
    )

  def _process_batch(self, tasks: Sequence[async_writer.WriteTask]) -> None:
    """Upserts a batch in one transaction, falling back per task on data error."""
    if len(tasks) <= 1:
      super()._process_batch(tasks)
      return

    try:
      self._execute_batch_transaction(tasks)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if self._should_batch_on_error(exc):
        raise
      logging.warning(
          "%s bulk transaction failed for %d tasks; falling back to per-task"
          " replay.",
          self._thread_name,
          len(tasks),
          exc_info=True,
      )
    else:
      self._decrease_batch_wait()
      return

    super()._process_batch(tasks)

  def _process_task(self, task: async_writer.WriteTask) -> None:
    """Persists a single write task through the unified transaction pipeline."""
    self._execute_batch_transaction((task,))

  def _collect_run_agents_to_upsert(
      self, tasks: Sequence[async_writer.WriteTask]
  ) -> dict[str, str]:
    """Validates `run_id` and returns `run_id -> agent_name` updates."""
    agent_name_by_run_id: dict[str, str] = {}
    for task in tasks:
      run_id = task.run_id
      if not run_id:
        raise ValueError("WriteTask requires run_id.")
      if task.metadata is not None:
        agent_name_by_run_id[run_id] = (
            agent_name_by_run_id.get(run_id) or task.metadata.agent.name
        )
      elif task.metrics is not None:
        agent_name_by_run_id.setdefault(run_id, "")

    return {
        run_id: agent_name
        for run_id, agent_name in agent_name_by_run_id.items()
        if (cached_agent := self._run_agent_by_id.get(run_id)) is None
        or (not cached_agent and agent_name)
    }

  def _collect_trajectory_updates(
      self,
      tasks: Sequence[async_writer.WriteTask],
      trajectory_batches: Sequence[
          async_writer.TrajectoryBatch[async_writer.WriteTask]
      ],
      now_dt: datetime.datetime,
  ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Returns deduplicated trajectory rows and post-commit cache entries."""
    latest_status_by_key: dict[tuple[str, str], str] = {}
    for task in tasks:
      if (
          task.run_id
          and task.metadata is not None
          and (status := _resolve_status(task.metadata))
          != schema.Status.UNKNOWN
      ):
        latest_status_by_key[(task.run_id, task.trajectory_id)] = status

    trajectory_rows: list[dict[str, Any]] = []
    committed_metadata_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in trajectory_batches:
      meta_task = batch.latest_metadata_task
      if (
          batch.run_id is None
          or meta_task is None
          or meta_task.metadata is None
      ):
        continue
      traj_key = (batch.run_id, batch.trajectory_id)
      metadata_payload = meta_task.metadata.model_dump(
          mode="json", exclude_none=True
      )
      committed_metadata_by_key[traj_key] = metadata_payload
      resolved_status = latest_status_by_key.get(
          traj_key, schema.Status.UNKNOWN
      )
      has_earlier_batch_status = resolved_status != _resolve_status(
          meta_task.metadata
      )
      cached_payload = self._trajectories_by_run.get(batch.run_id, {}).get(
          batch.trajectory_id
      )
      if cached_payload != metadata_payload or has_earlier_batch_status:
        trajectory_rows.append({
            "run_id": batch.run_id,
            "trajectory_id": batch.trajectory_id,
            "status": resolved_status,
            "created_at": now_dt,
            "updated_at": now_dt,
            "trajectory_metadata": metadata_payload,
        })
    return trajectory_rows, committed_metadata_by_key

  def _collect_step_and_metrics_rows(
      self,
      trajectory_batches: Sequence[
          async_writer.TrajectoryBatch[async_writer.WriteTask]
      ],
      now_dt: datetime.datetime,
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns deduplicated step and metrics row dicts across trajectories."""
    step_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for batch in trajectory_batches:
      if batch.run_id is None:
        continue
      latest_step_by_id = {
          step_task.step.step_id: step_task.step
          for step_task in batch.step_tasks
          if step_task.step is not None and step_task.step.step_id is not None
      }
      for step in latest_step_by_id.values():
        step_rows.append({
            "run_id": batch.run_id,
            "trajectory_id": batch.trajectory_id,
            "step_id": step.step_id,
            "step": step,
            "created_at": _to_utc_timestamp(step.timestamp),
        })
      merged_metrics_task = batch.merged_queued_metrics_task
      if (
          merged_metrics_task is not None
          and merged_metrics_task.metrics is not None
      ):
        metrics_rows.append({
            "run_id": batch.run_id,
            "trajectory_id": batch.trajectory_id,
            "payload": merged_metrics_task.metrics.model_dump(mode="json"),
            "created_at": now_dt,
            "updated_at": now_dt,
        })
    return step_rows, metrics_rows

  def _execute_batch_transaction(
      self, tasks: Sequence[async_writer.WriteTask]
  ) -> None:
    """Persists deduplicated run, trajectory, step, and metrics rows atomically."""
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    run_agent_by_id = self._collect_run_agents_to_upsert(tasks)
    run_rows = [
        {
            "run_id": run_id,
            "agent_name": agent_name,
            "status": schema.Status.PENDING,
            "created_at": now_dt,
            "config": {},
        }
        for run_id, agent_name in run_agent_by_id.items()
    ]
    trajectory_batches = async_writer.group_tasks_by_trajectory(tasks)
    trajectory_rows, committed_metadata_by_key = (
        self._collect_trajectory_updates(tasks, trajectory_batches, now_dt)
    )
    step_rows, metrics_rows = self._collect_step_and_metrics_rows(
        trajectory_batches, now_dt
    )

    if any((run_rows, trajectory_rows, step_rows, metrics_rows)):
      with self._engine.begin() as conn:
        self._upsert_run_rows(conn, run_rows)
        self._upsert_trajectory_rows(conn, trajectory_rows)
        self._upsert_step_rows(conn, step_rows)
        self._upsert_metrics_rows(conn, metrics_rows, now_dt)

    self._run_agent_by_id.update(run_agent_by_id)
    for (
        run_id,
        traj_id,
    ), metadata_payload in committed_metadata_by_key.items():
      self._cache_committed_state(run_id, traj_id, metadata_payload)

  def _upsert_run_rows(
      self, conn: sa.Connection, run_rows: Sequence[dict[str, Any]]
  ) -> None:
    """Inserts parent run rows or backfills empty `agent_name` values."""
    if not run_rows:
      return
    run_insert = self._insert_fn(schema.RUNS_TABLE).values(run_rows)
    if any(row["agent_name"] for row in run_rows):
      stored_agent = schema.RUNS_TABLE.c.agent_name
      incoming_agent = run_insert.excluded.agent_name
      run_upsert = run_insert.on_conflict_do_update(
          index_elements=["run_id"],
          set_={"agent_name": incoming_agent},
          where=sa.and_(
              stored_agent == sa.literal(""),
              incoming_agent != sa.literal(""),
          ),
      )
    else:
      run_upsert = run_insert.on_conflict_do_nothing(index_elements=["run_id"])
    conn.execute(run_upsert)

  def _upsert_trajectory_rows(
      self, conn: sa.Connection, trajectory_rows: Sequence[dict[str, Any]]
  ) -> None:
    """Executes a multi-row trajectory upsert in the open transaction."""
    if not trajectory_rows:
      return
    trajectory_insert = self._insert_fn(schema.TRAJECTORIES_TABLE).values(
        trajectory_rows
    )
    stored_metadata = schema.TRAJECTORIES_TABLE.c.trajectory_metadata
    stored_status = schema.TRAJECTORIES_TABLE.c.status
    incoming = trajectory_insert.excluded
    has_incoming_status = incoming.status != schema.Status.UNKNOWN
    conn.execute(
        trajectory_insert.on_conflict_do_update(
            index_elements=["run_id", "trajectory_id"],
            set_={
                "updated_at": incoming.updated_at,
                "trajectory_metadata": incoming.trajectory_metadata,
                "status": sa.case(
                    (has_incoming_status, incoming.status),
                    else_=stored_status,
                ),
            },
            where=sa.or_(
                stored_metadata.is_distinct_from(incoming.trajectory_metadata),
                sa.and_(has_incoming_status, stored_status != incoming.status),
            ),
        )
    )

  def _upsert_step_rows(
      self, conn: sa.Connection, step_rows: Sequence[dict[str, Any]]
  ) -> None:
    """Executes step upserts via pg_wire or a multi-row SQLAlchemy upsert."""
    if not step_rows:
      return
    first_row = step_rows[0]
    if pg_wire.try_upsert_step(
        conn,
        run_id=first_row["run_id"],
        trajectory_id=first_row["trajectory_id"],
        step=first_row["step"],
        created_at=first_row["created_at"],
    ):
      for row in step_rows[1:]:
        if not pg_wire.try_upsert_step(
            conn,
            run_id=row["run_id"],
            trajectory_id=row["trajectory_id"],
            step=row["step"],
            created_at=row["created_at"],
        ):
          raise sa.exc.DatabaseError(
              "pg_wire.try_upsert_step",
              None,
              RuntimeError("pg_wire became unavailable mid-batch."),
          )
      return

    serialized_rows = [
        {
            "run_id": row["run_id"],
            "trajectory_id": row["trajectory_id"],
            "step_id": row["step_id"],
            "payload": row["step"].model_dump(mode="json", exclude_none=True),
            "created_at": row["created_at"],
        }
        for row in step_rows
    ]
    step_insert = self._insert_fn(schema.STEPS_TABLE).values(serialized_rows)
    incoming = step_insert.excluded
    conn.execute(
        step_insert.on_conflict_do_update(
            index_elements=["run_id", "trajectory_id", "step_id"],
            set_={
                "payload": incoming.payload,
                "created_at": incoming.created_at,
            },
        )
    )

  def _upsert_metrics_rows(
      self,
      conn: sa.Connection,
      metrics_rows: Sequence[dict[str, Any]],
      now_dt: datetime.datetime,
  ) -> None:
    """Upserts trajectory metrics rows with shallow top-level `extra` merge."""
    insert_only_rows = [
        row for row in metrics_rows if row["payload"]["extra"] is None
    ]
    if insert_only_rows:
      conn.execute(
          self._insert_fn(schema.TRAJECTORY_METRICS_TABLE)
          .values(insert_only_rows)
          .on_conflict_do_nothing(index_elements=["run_id", "trajectory_id"])
      )

    merge_rows = [
        row for row in metrics_rows if row["payload"]["extra"] is not None
    ]
    if not merge_rows:
      return

    if self._engine.dialect.name == db_engine.Dialect.POSTGRESQL:
      metrics_insert = postgresql.insert(
          schema.TRAJECTORY_METRICS_TABLE
      ).values(merge_rows)
      stored_payload = schema.TRAJECTORY_METRICS_TABLE.c.payload
      merged_payload = _build_postgres_metrics_merge_payload(metrics_insert)
      conn.execute(
          metrics_insert.on_conflict_do_update(
              index_elements=["run_id", "trajectory_id"],
              set_={
                  "updated_at": now_dt,
                  "payload": merged_payload,
              },
              where=stored_payload.is_distinct_from(merged_payload),
          )
      )
      return

    for row in merge_rows:
      self._upsert_sqlite_metrics_row(conn, row, now_dt)

  def _upsert_sqlite_metrics_row(
      self,
      conn: sa.Connection,
      row: Mapping[str, Any],
      now_dt: datetime.datetime,
  ) -> None:
    """Upserts one SQLite trajectory metrics row with non-None `extra`."""
    insert_result = conn.execute(
        self._insert_fn(schema.TRAJECTORY_METRICS_TABLE)
        .values(**row)
        .on_conflict_do_nothing(index_elements=["run_id", "trajectory_id"])
    )
    if insert_result.rowcount:
      return

    run_id = row["run_id"]
    traj_id = row["trajectory_id"]
    metrics_payload = row["payload"]
    existing_payload = conn.execute(
        sa.select(schema.TRAJECTORY_METRICS_TABLE.c.payload).where(
            schema.TRAJECTORY_METRICS_TABLE.c.run_id == run_id,
            schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id == traj_id,
        )
    ).scalar_one()
    existing_extra = existing_payload["extra"]
    merged_extra = {**(existing_extra or {}), **metrics_payload["extra"]}
    if merged_extra == existing_extra:
      return
    conn.execute(
        sa.update(schema.TRAJECTORY_METRICS_TABLE)
        .where(
            schema.TRAJECTORY_METRICS_TABLE.c.run_id == run_id,
            schema.TRAJECTORY_METRICS_TABLE.c.trajectory_id == traj_id,
        )
        .values(
            payload={**metrics_payload, "extra": merged_extra},
            updated_at=now_dt,
        )
    )

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
    3. Forwarding step, metadata, and metrics write tasks and flush barriers to
       `_AsyncSqlWriter`.
    4. Synchronous read queries (`get_trajectories()`,
       `get_trajectories_metadata()`, and `get_trajectories_metrics()`) via
       `self._engine.connect()`.

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
      batching: async_writer.BatchConfig | None = None,
      copy_step: bool = False,
  ) -> None:
    """Initializes SqlTrajectoryStore.

    Args:
      run_id: Run identifier used to scope trajectories, steps, and metrics.
        Lazily registered in `RUNS_TABLE` on the first write task (with
        `agent_name` populated or backfilled once `TrajectoryMetadata` is
        provided).
      db_url: Database connection URL, e.g. 'sqlite:///traj.db' or
        'postgresql+psycopg2://user@host/db'. Callers resolve any credentials
        (e.g. from a secret manager) before passing it. A PostgreSQL URL may
        omit the password; libpq then reads `PGPASSWORD` or `~/.pgpass`.
      auto_init: If True, automatically creates database tables and indexes on
        startup via `_initialize_schema`.
      batching: Optional BatchConfig governing how queued tasks are grouped into
        multi-row SQL transactions.
      copy_step: If True, deep copies step objects before enqueuing to database.
        Defaults to False (zero-copy step handover) to avoid synchronous GIL
        contention on rollout workers.

    Raises:
      ValueError: If `run_id` or `db_url` is empty, None, or whitespace, or if
        `db_url` uses an unsupported database dialect.
    """
    super().__init__()
    self._run_id = _validate_non_empty_str(run_id, "run_id").strip()
    self._db_url = _validate_non_empty_str(db_url, "db_url")
    self._engine_handle = db_engine.acquire_engine(
        db_engine.EngineConfig(url=db_url)
    )
    self._engine = self._engine_handle.engine
    try:
      self._writer = _AsyncSqlWriter(
          engine=self._engine, batching=batching, copy_step=copy_step
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

  @property
  def batching(self) -> async_writer.BatchConfig:
    """Returns the active BatchConfig for this store."""
    return self._writer.batching

  @classmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
  ) -> "SqlTrajectoryStore[MetadataT]":
    """Builds a SQL-backed store from `config`.

    Args:
      config: Requires "db_url" and "run_id", with optional boolean "copy_step"
        (defaults to False).

    Returns:
      A new SqlTrajectoryStore.

    Raises:
      ValueError: If "db_url" or "run_id" is missing or empty.
    """
    return cls(
        run_id=config.get("run_id", ""),
        db_url=config.get("db_url", ""),
        copy_step=bool(config.get("copy_step", False)),
    )

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    The returned "db_url" is the literal URL this store was built with,
    including any password it embeds. Do not log it; log
    `to_redacted_config()` instead. Writer-local flags such as `auto_init` and
    `copy_step` are omitted so `to_config()` strictly represents shared storage
    identity across processes.
    """
    return {
        "backend": self.BACKEND,
        "db_url": self._db_url,
        "run_id": self._run_id,
    }

  @property
  def copy_step(self) -> bool:
    """Returns True if step objects are deep copied before writing."""
    return self._writer.copy_step

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
       `trajectories`, `trajectory_metrics`, `steps`) already exist. On warm
       databases, returns immediately without acquiring write locks or opening a
       write transaction.
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

    `metadata` is always snapshotted at call time. By default
    (`copy_step=False`), ownership of `step` transfers to the background writer,
    so callers must not mutate `step` or its nested containers after calling
    `add_step()`. Pass `copy_step=True` if the caller mutates `step` afterwards.

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

    `metadata` is always snapshotted at call time, so callers may keep mutating
    it afterwards without affecting what was logged.

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

    Top-level keys in `metrics.extra` are shallow-merged onto any existing
    `extra` dictionary persisted for `trajectory_id`. `metrics` is snapshotted
    on the calling thread before enqueueing.

    Args:
      trajectory_id: Unique identifier for the trajectory.
      metrics: TrajectoryMetrics containing the metrics data to record.

    Raises:
      TypeError: If metrics is not a TrajectoryMetrics instance.
      ValueError: If trajectory_id is empty, None, or whitespace.
      RuntimeError: If the store has already been closed.
    """
    self._writer.enqueue_metrics_write(
        run_id=self._run_id,
        trajectory_id=trajectory_id,
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
        return [self._load_metadata(row) for row in metadata_rows.scalars()]

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
    with self._engine.connect() as conn:
      metadata_rows = conn.execute(metadata_statement)
      metadata_by_trajectory_id = {
          trajectory_id: payload for trajectory_id, payload in metadata_rows
      }
      pg_step_rows = pg_wire.try_fetch_steps(
          conn, run_id=self._run_id, trajectory_ids=trajectory_ids
      )
      raw_steps_by_trajectory_id: dict[str, list[str | Mapping[str, Any]]] = (
          collections.defaultdict(list)
      )
      if pg_step_rows is not None:
        for trajectory_id, payload_json, _ in pg_step_rows:
          raw_steps_by_trajectory_id[trajectory_id].append(payload_json)
      else:
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
        for trajectory_id, payload in conn.execute(steps_statement):
          raw_steps_by_trajectory_id[trajectory_id].append(payload)

    validate_step: Callable[[Any], trajectory_lib.Step] = (
        trajectory_lib.Step.model_validate_json
        if pg_step_rows is not None
        else trajectory_lib.Step.model_validate
    )
    trajectories: list[trajectory_lib.Trajectory[Any]] = []
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metadata_by_trajectory_id:
        raise store.TrajectoryNotFoundError(trajectory_id)
      meta = self._load_metadata(metadata_by_trajectory_id[trajectory_id])
      steps = [
          validate_step(raw_step)
          for raw_step in raw_steps_by_trajectory_id.get(trajectory_id, [])
      ]
      trajectories.append(meta.create_trajectory(steps=steps))

    return trajectories

  def get_trajectories_metrics(
      self, trajectory_ids: Sequence[str] | None = None
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    """Retrieves metrics for trajectories in the run.

    Args:
      trajectory_ids: Optional sequence of unique trajectory identifiers. If
        specified, only metrics for these IDs are returned. If None, metrics for
        all trajectories in the run with recorded metrics are returned.

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
        return {
            row.trajectory_id: trajectory_lib.TrajectoryMetrics.model_validate(
                row.payload
            )
            for row in conn.execute(statement)
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
      metrics_by_id = {
          row.trajectory_id: trajectory_lib.TrajectoryMetrics.model_validate(
              row.payload
          )
          for row in conn.execute(statement)
      }

    result: dict[str, trajectory_lib.TrajectoryMetrics] = {}
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metrics_by_id:
        raise store.TrajectoryMetricsNotFoundError(trajectory_id)
      result[trajectory_id] = metrics_by_id[trajectory_id]
    return result

  def _load_metadata(self, payload: dict[str, Any]) -> MetadataT:
    """Rehydrates a stored base-ATIF metadata payload."""
    base_meta = trajectory_lib.TrajectoryMetadata.model_validate(payload)
    return self._rehydrate_metadata(base_meta)
