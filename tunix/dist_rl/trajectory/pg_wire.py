"""PostgreSQL libpq wire execution helpers for Trajectory Store."""

from collections.abc import Sequence
import dataclasses
import datetime
import sys
from typing import Final, NoReturn

import numpy as np
import sqlalchemy as sa
from tunix.dist_rl.trajectory import db_engine
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

try:
  # pylint: disable=g-import-not-at-top
  from tunix.dist_rl.trajectory import _sql_pg_ext
except ImportError:
  _sql_pg_ext = None


_PG_OPERATIONAL_SQLSTATE_PREFIXES: Final[tuple[str, ...]] = (
    "08",  # connection_exception
    "40",  # transaction_rollback (serialization_failure, deadlock)
    "53",  # insufficient_resources (disk_full, out_of_memory)
    "55P03",  # lock_not_available
    "57",  # operator_intervention (query_canceled, admin_shutdown)
    "58",  # system_error (io_error)
)

_PG_DISCONNECT_SQLSTATE_PREFIXES: Final[tuple[str, ...]] = (
    "08000",  # connection_exception
    "08003",  # connection_does_not_exist
    "08006",  # connection_failure
    "08001",  # sqlclient_unable_to_establish_sqlconnection
    "08004",  # sqlserver_rejected_establishment_of_sqlconnection
    "57P01",  # admin_shutdown
    "57P02",  # crash_shutdown
    "57P03",  # cannot_connect_now
)

_MAX_PG_FETCH_IDS: Final[int] = 30_000


@dataclasses.dataclass(frozen=True, slots=True)
class PgWireTarget:
  """Resolved PostgreSQL libpq connection address and library path.

  Attributes:
    pgconn_ptr: Integer address of the underlying libpq PGconn struct.
    lib_path: Filesystem path to the loaded psycopg2 C extension library.
  """

  pgconn_ptr: int
  lib_path: str


def _get_psycopg_lib_path() -> str:
  """Returns filesystem path to the loaded psycopg2 C extension library."""
  mod = sys.modules.get("psycopg2._psycopg")
  if mod is None:
    return ""
  try:
    lib_file = mod.__file__
  except AttributeError:
    return ""
  if isinstance(lib_file, str):
    return lib_file
  return ""


def _get_pgconn_ptr(conn: sa.Connection) -> int | None:
  """Extracts a live libpq PGconn pointer integer from a connection."""
  if conn.dialect.name != db_engine.Dialect.POSTGRESQL:
    return None
  try:
    dbapi_conn = conn.connection.dbapi_connection
  except AttributeError:
    dbapi_conn = conn.connection
  if dbapi_conn is None:
    return None
  try:
    pgconn_ptr = dbapi_conn.pgconn_ptr
  except AttributeError:
    return None
  if isinstance(pgconn_ptr, int) and pgconn_ptr > 0:
    return pgconn_ptr
  return None


def resolve_pg_wire_target(conn: sa.Connection) -> PgWireTarget | None:
  """Resolves the raw pgconn pointer and psycopg2 library path.

  Args:
    conn: Active SQLAlchemy connection.

  Returns:
    A PgWireTarget if the dialect is PostgreSQL, _sql_pg_ext is loaded, and
    pgconn_ptr is valid, or None otherwise.
  """
  if _sql_pg_ext is None:
    return None
  pgconn_ptr = _get_pgconn_ptr(conn)
  if pgconn_ptr is None:
    return None
  lib_path = _get_psycopg_lib_path()
  return PgWireTarget(pgconn_ptr=pgconn_ptr, lib_path=lib_path)


def _extract_sqlstate(message: str) -> str:
  """Extracts the SQLSTATE code from a formatted libpq error string."""
  prefix = "[SQLSTATE:"
  if not message.startswith(prefix):
    return ""
  end_idx = message.find("]", len(prefix))
  if end_idx == -1:
    return ""
  return message[len(prefix) : end_idx]


def _raise_sqlalchemy_pg_error(
    conn: sa.Connection, statement: str, exc: RuntimeError
) -> NoReturn:
  """Maps a C++ _sql_pg_ext RuntimeError to a SQLAlchemy DBAPIError."""
  sqlstate = _extract_sqlstate(str(exc))
  invalidated = sqlstate.startswith(_PG_DISCONNECT_SQLSTATE_PREFIXES)
  if invalidated:
    conn.invalidate(exc)
  if sqlstate.startswith(_PG_OPERATIONAL_SQLSTATE_PREFIXES):
    raise sa.exc.OperationalError(
        statement, None, exc, connection_invalidated=invalidated
    ) from exc
  if sqlstate.startswith("23"):
    raise sa.exc.IntegrityError(statement, None, exc) from exc
  raise sa.exc.DatabaseError(statement, None, exc) from exc


def try_upsert_step(
    conn: sa.Connection,
    *,
    run_id: str,
    trajectory_id: str,
    step: trajectory_lib.Step,
    created_at: datetime.datetime,
    binary_data: bytes | None = None,
    binary_col: str = "",
) -> bool:
  """Tries to upsert a step row using direct libpq wire execution.

  Args:
    conn: Active SQLAlchemy connection participating in the transaction.
    run_id: Non-empty run identifier.
    trajectory_id: Non-empty trajectory identifier.
    step: Step payload to persist.
    created_at: UTC timestamp when the step was captured.
    binary_data: Optional raw binary payload to write.
    binary_col: Column name receiving binary_data (e.g. "routed_experts").

  Returns:
    True if the wire extension executed the upsert, or False if wire execution
    is unavailable or unsupported for this connection.
  """
  if _sql_pg_ext is None:
    return False
  target = resolve_pg_wire_target(conn)
  if target is None or not _sql_pg_ext.can_execute_pg_binary(target.lib_path):
    return False

  step_payload_json = _sql_pg_ext.strip_json_nul_bytes(
      step.model_dump_json(exclude_none=True)
  )
  step_time_iso = created_at.isoformat()
  try:
    _sql_pg_ext.pg_upsert_step(
        pgconn_ptr=target.pgconn_ptr,
        run_id=run_id,
        trajectory_id=trajectory_id,
        step_id=step.step_id,
        payload_json=step_payload_json,
        created_at_iso=step_time_iso,
        psycopg_lib_path=target.lib_path,
        binary_data=binary_data,
        binary_col=binary_col,
    )
    return True
  except RuntimeError as exc:
    _raise_sqlalchemy_pg_error(conn, "pg_upsert_step", exc)


def try_fetch_steps(
    conn: sa.Connection,
    *,
    run_id: str,
    trajectory_ids: Sequence[str],
    binary_col: str = "",
) -> list[tuple[str, str, np.ndarray | None]] | None:
  """Tries to fetch step rows using direct libpq wire execution.

  Args:
    conn: Active SQLAlchemy connection.
    run_id: Non-empty run identifier.
    trajectory_ids: Trajectory identifiers whose steps should be retrieved.
    binary_col: Column name receiving binary payload (e.g. "routed_experts").

  Returns:
    A list of (trajectory_id, payload_json, binary_data) tuples if wire
    execution succeeded, or None if wire execution is unavailable or the ID
    count exceeds the single-query parameter threshold.
  """
  if _sql_pg_ext is None:
    return None
  if len(trajectory_ids) > _MAX_PG_FETCH_IDS:
    return None
  target = resolve_pg_wire_target(conn)
  if target is None or not _sql_pg_ext.can_fetch_pg_binary(target.lib_path):
    return None
  if not trajectory_ids:
    return []
  try:
    return _sql_pg_ext.pg_fetch_steps(
        pgconn_ptr=target.pgconn_ptr,
        run_id=run_id,
        trajectory_ids=list(trajectory_ids),
        binary_col=binary_col,
        psycopg_lib_path=target.lib_path,
    )
  except RuntimeError as exc:
    _raise_sqlalchemy_pg_error(conn, "pg_fetch_steps", exc)
