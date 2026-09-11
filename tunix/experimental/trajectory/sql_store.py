"""SQL-backed implementation of TrajectoryStore using synchronous SQLAlchemy."""

import types
from typing import Self

import sqlalchemy as sa
from tunix.experimental.trajectory import async_sql_writer
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib


class SqlTrajectoryStore(store.TrajectoryWriter):
  """SQL-backed implementation of TrajectoryWriter.

  `SqlTrajectoryStore` manages the persistence of reinforcement learning (RL)
  agent rollouts and step trajectories into relational database backends (e.g.
  SQLite, PostgreSQL) using SQLAlchemy.

  Key Responsibilities & Operational Invariants:
    1. Relational Data Hierarchy:
       Maintains relational integrity across three primary tables:
       - `runs`: Top-level execution runs identified by `run_id`, recording
         overall status, agent information, and configuration.
       - `trajectories`: Individual rollout trajectories identified by
         `(run_id, trajectory_id)`, tracking status, timestamps, and metadata.
       - `steps`: Sequential turn records identified by
         `(run_id, trajectory_id, step_id)`, storing observations, actions, and
         step-level payloads.

    2. Run Scoping & Dynamic Identifier Resolution:
       Trajectories and steps are scoped under a specific `run_id`. The active
       identifier is resolved dynamically via `metadata.session_id` when
       provided, falling back to the configured instance-level `run_id`.

    3. Non-blocking Asynchronous Writes:
       Step additions and metadata updates are offloaded onto a background
       worker thread, allowing training and rollout loops to proceed without
       blocking on database I/O.

    4. Idempotent Upserts:
       Employs dialect-specific conflict resolution (`ON CONFLICT DO UPDATE` /
       `ON CONFLICT DO NOTHING`) to ensure step updates, retries, and status
       transitions are written idempotently without primary key collisions.

    5. Barrier Synchronization & Lifecycle Management:
       `flush()` provides strict barrier synchronization by blocking until all
       enqueued operations are committed. `close()` and context manager exit
       guarantee graceful draining of pending writes before worker shutdown.
  """

  def __init__(
      self,
      engine: sa.Engine,
      run_id: str = "default",
      auto_init: bool = True,
  ):
    """Initializes SqlTrajectoryStore.

    Args:
      engine: Configured SQLAlchemy Engine providing database connectivity.
      run_id: Fallback run identifier used to scope trajectories and steps if
        not explicitly provided on trajectory metadata.
      auto_init: If True, automatically creates database tables and indexes on
        startup via `schema.METADATA.create_all`.
    """
    self._engine = engine
    self._run_id = run_id.strip() if run_id and run_id.strip() else "default"
    self._writer = async_sql_writer.AsyncSqlWriter(engine=engine)

    if auto_init:
      schema.METADATA.create_all(self._engine)

  @property
  def engine(self) -> sa.Engine:
    """Returns the underlying SQLAlchemy engine."""
    return self._engine

  @property
  def run_id(self) -> str:
    """Returns the default run identifier."""
    return self._run_id

  def resolve_run_id(self, metadata: trajectory_lib.TrajectoryMetadata) -> str:
    """Resolves the run identifier for a given metadata object.

    Args:
      metadata: TrajectoryMetadata containing session or run metadata.

    Returns:
      Resolved run identifier string.
    """
    if metadata.session_id and metadata.session_id.strip():
      return metadata.session_id.strip()
    return self._run_id

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Asynchronously logs a turn step and its trajectory metadata.

    Resolves the effective run identifier, validates input parameters, and
    delegates asynchronous queueing and non-blocking database persistence to
    the underlying `AsyncSqlWriter`.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata is None, or metadata.trajectory_id is empty or
        whitespace.
      RuntimeError: If the store has already been closed.
    """
    self.update_metadata(metadata, step=step)

  def update_metadata(
      self,
      metadata: trajectory_lib.TrajectoryMetadata,
      step: trajectory_lib.Step | None = None,
  ) -> None:
    """Updates (or creates) trajectory metadata asynchronously.

    Optionally writes an associated step.

    Resolves the effective run identifier, validates input parameters, and
    delegates asynchronous queueing and non-blocking database persistence to
    the underlying `AsyncSqlWriter`.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
      step: Optional Step object to write alongside the metadata.

    Raises:
      ValueError: If metadata is None, or metadata.trajectory_id is empty or
        whitespace.
      RuntimeError: If the store has already been closed.
    """
    if metadata is None:
      raise ValueError("TrajectoryMetadata cannot be None.")
    run_id = self.resolve_run_id(metadata)
    self._writer.write_step(run_id=run_id, metadata=metadata, step=step)

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage.

    Users do not need to call `flush()` in normal usage; it is primarily for
    testing.

    Delegates directly to `AsyncSqlWriter.flush()` to provide strict barrier
    synchronization.
    """
    self._writer.flush()

  def close(self) -> None:
    """Flushes pending writes and shuts down the background writer thread.

    Calling `close()` is optional: the underlying `AsyncSqlWriter` also drains
    itself at interpreter exit. It is worth calling explicitly for a store that
    becomes garbage well before the process ends, so its worker thread is
    released promptly. Closing is idempotent, but the store must not be written
    to afterwards; reads remain available.
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
