"""Database engine management for SQL Trajectory Store."""

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import os
import re
import sqlite3
import threading
from typing import Any, Final

import sqlalchemy as sa

_NUL_ESCAPE_RE: Final[re.Pattern[str]] = re.compile(r"(\\\\)+|\\u0000")


def _serialize_json(value: Any) -> str:
  """Serializes JSON payloads and strips NUL escapes rejected by PostgreSQL JSONB."""
  dumped = json.dumps(value)
  if "\\u0000" not in dumped:
    return dumped
  return _NUL_ESCAPE_RE.sub(
      lambda m: "" if m.group(0) == "\\u0000" else m.group(0),
      dumped,
  )


class Dialect(enum.StrEnum):
  """SQL dialects supported by the SQL Trajectory Store.

  Values are the lowercased member names, which must match SQLAlchemy dialect
  names (`sa.URL.get_backend_name()` and `sa.Engine.dialect.name`).
  """

  POSTGRESQL = enum.auto()
  SQLITE = enum.auto()


class PoolKind(enum.StrEnum):
  """How an engine pools its database connections.

  Running behind an external pooler on PostgreSQL (PgBouncer, or Cloud SQL
  managed connection pooling) keeps a large fleet of worker processes under the
  server's `max_connections`. Use `NULL` there, so the external pooler alone
  decides how many server connections are open. Transaction pooling mode is
  compatible with the SQL Trajectory Store:

  - Every read and write runs in a short transaction, and the store sets no
    session state (no `SET`, `LISTEN`, temporary tables, or session locks).
  - Schema initialization serializes with `pg_advisory_xact_lock`, which is
    released when its transaction ends, not when the session ends.
  - The psycopg2 driver interpolates parameters on the client and never creates
    server-side prepared statements. (psycopg 3 does, after 5 executions of a
    query; with it, pass `prepare_threshold=None` in the connect args or use a
    PgBouncer version that supports prepared statements in transaction mode.)

  Attributes:
    QUEUE: SQLAlchemy's default `QueuePool`, sized by `EngineConfig.pool_size`
      and `EngineConfig.max_overflow`, with a liveness ping on checkout.
    NULL: `NullPool`, which opens a connection on each checkout and closes it on
      return. Use behind an external pooler, which makes connecting cheap.
  """

  QUEUE = enum.auto()
  NULL = enum.auto()


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
    pool_kind: How the engine pools connections; see `PoolKind`. A string value
      (e.g. "null" from an untyped config) is converted to `PoolKind`. With
      `PoolKind.NULL` the sizing fields below are ignored.
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
  pool_kind: PoolKind = PoolKind.QUEUE
  pool_size: int = 1
  max_overflow: int = 1
  pool_timeout_s: float = 10.0

  def __post_init__(self) -> None:
    """Normalizes `pool_kind` to `PoolKind` and rejects unknown values.

    Raises:
      ValueError: If `pool_kind` is not a `PoolKind` value.
    """
    try:
      pool_kind = PoolKind(self.pool_kind)
    except ValueError:
      raise ValueError(
          f"Unsupported pool_kind {self.pool_kind!r}; expected one of"
          f" {', '.join(repr(kind.value) for kind in PoolKind)}."
      ) from None
    object.__setattr__(self, "pool_kind", pool_kind)


def _pool_kwargs(config: EngineConfig) -> dict[str, Any]:
  """Returns the `sa.create_engine` pooling arguments for `config`.

  Args:
    config: Engine settings whose `pool_kind` selects the pool class.

  Raises:
    ValueError: If `config.pool_kind` is not a known `PoolKind`.
  """
  if config.pool_kind == PoolKind.NULL:
    # No pre-ping: every checkout opens a fresh connection, so there is no
    # stale pooled connection to detect.
    return {"poolclass": sa.pool.NullPool}
  if config.pool_kind == PoolKind.QUEUE:
    return {
        "pool_pre_ping": True,
        "pool_size": config.pool_size,
        "max_overflow": config.max_overflow,
        "pool_timeout": config.pool_timeout_s,
    }
  raise ValueError(f"Unsupported pool_kind {config.pool_kind!r}.")


def create_trajectory_engine(config: EngineConfig) -> sa.Engine:
  """Creates and configures a synchronous SQLAlchemy engine.

  Args:
    config: Engine settings, including the database URL and pool sizing.

  Returns:
    Configured Engine instance.

  Raises:
    ValueError: If `config.url` is empty/whitespace or specifies an unsupported
      database dialect, or if `config.pool_kind` is `PoolKind.NULL` for an
      in-memory SQLite URL.
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
        **_pool_kwargs(config),
    )

  if backend == Dialect.SQLITE:
    is_in_memory = _is_in_memory_sqlite(parsed_url)
    if is_in_memory:
      if config.pool_kind == PoolKind.NULL:
        # The database lives only as long as its one connection, so closing
        # each connection on return would discard it.
        raise ValueError(
            "PoolKind.NULL is not supported for in-memory SQLite, which keeps"
            " its data in a single shared connection."
        )
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
          connect_args={"check_same_thread": False},
          **_pool_kwargs(config),
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
  is torn down. A handle either owns a private engine or holds one reference to
  an engine shared within the process (see `acquire_engine`).
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


@dataclasses.dataclass(kw_only=True)
class _SharedEngine:
  """A process-shared engine and the number of handles holding it."""

  engine: sa.Engine
  refcount: int = 0


# Key of a shared engine: the owning process id plus its config. Keying on the
# pid means a forked child never reuses an engine (and its open sockets)
# inherited from its parent; it builds its own on first acquire.
_SharedEngineKey = tuple[int, EngineConfig]

# Guards `_SHARED_ENGINES` and every `_SharedEngine.refcount`.
_shared_engines_lock = threading.Lock()
_SHARED_ENGINES: dict[_SharedEngineKey, _SharedEngine] = {}


def _reset_shared_engines_in_child() -> None:
  """Resets the shared-engine registry in a forked child.

  Replaces the lock, since a fork while another thread holds it would leave the
  child with a lock that is never released. Also drops the parent's entries, so
  the child doesn't keep them alive if it never releases inherited handles. An
  inherited handle can still be released afterwards: `_release_shared_engine`
  only removes an entry that is still registered.
  """
  global _shared_engines_lock
  _shared_engines_lock = threading.Lock()
  _SHARED_ENGINES.clear()


if hasattr(os, "register_at_fork"):  # POSIX only.
  os.register_at_fork(after_in_child=_reset_shared_engines_in_child)


def _release_shared_engine(key: _SharedEngineKey, entry: _SharedEngine) -> None:
  """Drops one reference to `entry` and disposes it after the last one."""
  with _shared_engines_lock:
    entry.refcount -= 1
    if entry.refcount > 0:
      return
    if _SHARED_ENGINES.get(key) is entry:
      del _SHARED_ENGINES[key]
  owner_pid, _ = key
  # A child releasing an engine inherited from its parent must not close the
  # parent's sockets; it only drops its references to them.
  entry.engine.dispose(close=owner_pid == os.getpid())


def _acquire_shared_engine(config: EngineConfig) -> EngineHandle:
  """Returns a handle to this process's shared engine for `config`."""
  key = (os.getpid(), config)
  with _shared_engines_lock:
    entry = _SHARED_ENGINES.get(key)
    if entry is None:
      # Built under the lock so concurrent first acquires create one engine.
      # Creating an engine does not connect to the database, so this is fast.
      entry = _SharedEngine(engine=create_trajectory_engine(config))
      _SHARED_ENGINES[key] = entry
    entry.refcount += 1
  return EngineHandle(
      engine=entry.engine,
      release_fn=functools.partial(_release_shared_engine, key, entry),
  )


def acquire_engine(
    config: EngineConfig, *, shared: bool = False
) -> EngineHandle:
  """Returns a handle to an engine configured by `config`.

  Callers must call `EngineHandle.release()` exactly when done and must not
  dispose or mutate the engine themselves, since with `shared=True` other
  handles may be using the same engine.

  Args:
    config: Engine settings, including the database URL and pool sizing.
    shared: If False (the default), the handle owns a new private engine. If
      True, all handles acquired in this process with an equal `config` share
      one engine and its connection pool: the first acquire builds it and the
      last release disposes it. A forked child gets its own engine. With
      in-memory SQLite, sharing also means sharing the database itself.

  Returns:
    A handle to the engine; releasing the last handle to an engine disposes it.

  Raises:
    ValueError: If `config.url` is empty/whitespace or specifies an unsupported
      database dialect.
  """
  if shared:
    return _acquire_shared_engine(config)
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
