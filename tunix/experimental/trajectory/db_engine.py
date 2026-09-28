"""Database engine management for SQL Trajectory Store."""

import sqlite3
from typing import Final

import sqlalchemy as sa

POSTGRESQL_DIALECT: Final[str] = "postgresql"
SQLITE_DIALECT: Final[str] = "sqlite"


def _set_sqlite_pragmas(
    dbapi_connection: sqlite3.Connection,
    unused_connection_record: sa.pool.ConnectionPoolEntry,
) -> None:
  """Configures PRAGMA settings on SQLite connections."""
  cursor = dbapi_connection.cursor()
  try:
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
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


def create_trajectory_engine(
    url: str,
    echo: bool = False,
    pool_size: int = 1,
    max_overflow: int = 2,
) -> sa.Engine:
  """Creates and configures a synchronous SQLAlchemy engine.

  Args:
    url: Database connection URL (e.g. 'sqlite:///:memory:' or
      'postgresql+psycopg2://user:pass@host/db').
    echo: If True, log SQL queries emitted by the engine.
    pool_size: Number of persistent connections kept open in the pool. Defaults
      to 1 per rollout worker to prevent connection exhaustion across
      distributed workers.
    max_overflow: Maximum transient overflow connections allowed beyond
      `pool_size` during concurrent reads/writes.

  Returns:
    Configured Engine instance.

  Raises:
    ValueError: If `url` is empty/whitespace or specifies an unsupported
      database dialect.
  """
  if not url or not url.strip():
    raise ValueError("Database URL must be a non-empty string.")

  parsed_url = sa.make_url(url)
  backend = parsed_url.get_backend_name()

  if backend == POSTGRESQL_DIALECT:
    return sa.create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )

  if backend == SQLITE_DIALECT:
    if _is_in_memory_sqlite(parsed_url):
      engine = sa.create_engine(
          url,
          echo=echo,
          connect_args={"check_same_thread": False},
          poolclass=sa.pool.StaticPool,
      )
    else:
      engine = sa.create_engine(
          url,
          echo=echo,
          pool_pre_ping=True,
          pool_size=pool_size,
          max_overflow=max_overflow,
          connect_args={"check_same_thread": False},
      )
    sa.event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine

  raise ValueError(
      f"Unsupported database dialect: {backend}. Supported dialects are"
      f" {POSTGRESQL_DIALECT} and {SQLITE_DIALECT}."
  )
