"""Database engine management for SQL Trajectory Store."""

from collections.abc import Callable
import dataclasses
import enum
import json
import sqlite3
import threading
from typing import Any

import sqlalchemy as sa


class Dialect(enum.StrEnum):
  """SQL dialects supported by the SQL Trajectory Store.

  Values are the lowercased member names, which must match SQLAlchemy dialect
  names (`sa.URL.get_backend_name()` and `sa.Engine.dialect.name`).
  """

  POSTGRESQL = enum.auto()
  SQLITE = enum.auto()


def _serialize_json(value: Any) -> str:
  """Serializes JSON payloads and strips NUL escapes rejected by PostgreSQL JSONB."""
  return json.dumps(value).replace("\\u0000", "")


def _set_sqlite_pragmas(
    dbapi_connection: sqlite3.Connection,
    unused_connection_record: sa.pool.ConnectionPoolEntry,
) -> None:
  """Configures PRAGMA settings shared by every SQLite connection."""
  cursor = dbapi_connection.cursor()
  try:
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
  finally:
    cursor.close()


def _set_sqlite_file_pragmas(
    dbapi_connection: sqlite3.Connection,
    unused_connection_record: sa.pool.ConnectionPoolEntry,
) -> None:
  """Configures journaling PRAGMAs that only apply to file-backed SQLite."""
  cursor = dbapi_connection.cursor()
  try:
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
  finally:
    cursor.close()


def _is_in_memory_sqlite(parsed_url: sa.URL) -> bool:
  """Returns True if the SQLite URL targets an in-memory database."""
  return (
      parsed_url.database in (":memory:", "", None)
      or parsed_url.query.get("mode") == "memory"
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineConfig:
  """Settings for building a SQL Trajectory Store engine.

  Attributes:
    url: Database connection URL (e.g. 'sqlite:///:memory:' or
      'postgresql+psycopg2://user:pass@host/db').
    echo: If True, log SQL queries emitted by the engine.
    pool_size: Number of persistent connections kept open in the pool. Defaults
      to 1 per process to prevent connection exhaustion across distributed
      workers.
    max_overflow: Maximum transient connections allowed beyond `pool_size`.
      Defaults to 1 because a store needs at most two connections at once: one
      for its background writer thread and one for reads on the caller thread.
    pool_timeout_s: Seconds to wait for a free pooled connection before raising,
      so an exhausted pool fails fast instead of hanging.
  """

  url: str
  echo: bool = False
  pool_size: int = 1
  max_overflow: int = 1
  pool_timeout_s: float = 10.0


def create_trajectory_engine(config: EngineConfig) -> sa.Engine:
  """Creates and configures a synchronous SQLAlchemy engine.

  Args:
    config: Engine settings, including the database URL and pool sizing.

  Returns:
    Configured Engine instance.

  Raises:
    ValueError: If `config.url` is empty/whitespace or specifies an unsupported
      database dialect.
  """
  url = config.url
  if not url or not url.strip():
    raise ValueError("Database URL must be a non-empty string.")

  try:
    parsed_url = sa.make_url(url)
  except sa.exc.ArgumentError as exc:
    raise ValueError("Database URL could not be parsed.") from exc
  backend = parsed_url.get_backend_name()

  if backend == Dialect.POSTGRESQL:
    return sa.create_engine(
        url,
        echo=config.echo,
        json_serializer=_serialize_json,
        pool_pre_ping=True,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout_s,
    )

  if backend == Dialect.SQLITE:
    is_in_memory = _is_in_memory_sqlite(parsed_url)
    if is_in_memory:
      engine = sa.create_engine(
          url,
          echo=config.echo,
          json_serializer=_serialize_json,
          connect_args={"check_same_thread": False},
          poolclass=sa.pool.StaticPool,
      )
    else:
      engine = sa.create_engine(
          url,
          echo=config.echo,
          json_serializer=_serialize_json,
          pool_pre_ping=True,
          pool_size=config.pool_size,
          max_overflow=config.max_overflow,
          pool_timeout=config.pool_timeout_s,
          connect_args={"check_same_thread": False},
      )
    sa.event.listen(engine, "connect", _set_sqlite_pragmas)
    if not is_in_memory:
      sa.event.listen(engine, "connect", _set_sqlite_file_pragmas)
    return engine

  raise ValueError(
      f"Unsupported database dialect: {backend}. Supported dialects are"
      f" {', '.join(Dialect)}."
  )


class EngineHandle:
  """A borrowed engine plus the obligation to give it back.

  Stores hold a handle instead of an engine so they never decide how an engine
  is torn down. Today every handle owns a private engine; a future per-process
  cache can hand out shared engines behind the same interface.
  """

  def __init__(self, engine: sa.Engine, release_fn: Callable[[], None]):
    self._engine = engine
    self._release_fn = release_fn
    self._lock = threading.Lock()
    self._released = False

  @property
  def engine(self) -> sa.Engine:
    """Returns the engine behind this handle."""
    return self._engine

  def release(self) -> None:
    """Returns the engine to its owner. Safe to call more than once."""
    with self._lock:
      if self._released:
        return
      self._released = True
    self._release_fn()


def acquire_engine(config: EngineConfig) -> EngineHandle:
  """Returns a handle to an engine configured by `config`.

  Callers must call `EngineHandle.release()` exactly when done and must not
  dispose or mutate the engine themselves. That contract lets this function
  later return a refcounted, per-process shared engine without changing
  callers.

  Args:
    config: Engine settings, including the database URL and pool sizing.

  Returns:
    A handle to a newly created engine; releasing it disposes the engine.

  Raises:
    ValueError: If `config.url` is empty/whitespace or specifies an unsupported
      database dialect.
  """
  engine = create_trajectory_engine(config)
  return EngineHandle(engine=engine, release_fn=engine.dispose)


def redact_url(url: str) -> str:
  """Returns `url` with any password masked, for safe logging and printing.

  Args:
    url: Database connection URL, possibly embedding credentials.

  Returns:
    The URL rendered with its password replaced by `***`.
  """
  return sa.make_url(url).render_as_string(hide_password=True)
