"""Tests for PostgreSQL libpq wire execution in pg_wire."""

from concurrent import futures
import datetime
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import sqlalchemy as sa
from tunix.dist_rl.trajectory import db_engine
from tunix.dist_rl.trajectory import pg_wire
from tunix.dist_rl.trajectory import trajectory_testing


def _make_pg_connection_mock(pgconn_ptr: int = 12345) -> mock.MagicMock:
  mock_conn = mock.MagicMock()
  mock_conn.dialect.name = db_engine.Dialect.POSTGRESQL
  mock_driver = mock.MagicMock()
  mock_driver.pgconn_ptr = pgconn_ptr
  mock_conn.connection.dbapi_connection = mock_driver
  return mock_conn


class PgWireTest(parameterized.TestCase):
  """Unit tests for pg_wire helper functions and error translations."""

  def setUp(self) -> None:
    super().setUp()
    if pg_wire._sql_pg_ext is None:
      self.skipTest("C++ _sql_pg_ext extension unavailable.")

  def test_strip_json_nul_bytes_with_nul_sequences_strips_all_nuls(
      self,
  ) -> None:
    raw = '{"key": "hello\x00world\\u0000test"}'

    cleaned = pg_wire._sql_pg_ext.strip_json_nul_bytes(raw)

    self.assertEqual(cleaned, '{"key": "helloworldtest"}')

  def test_strip_json_nul_bytes_with_clean_input_returns_same_str(
      self,
  ) -> None:
    clean = '{"key": "clean_value"}'

    result = pg_wire._sql_pg_ext.strip_json_nul_bytes(clean)

    self.assertIs(result, clean)

  def test_strip_json_nul_bytes_with_even_backslashes_preserves_literal(
      self,
  ) -> None:
    literal_backslashes = r'{"key": "\\u0000"}'

    result = pg_wire._sql_pg_ext.strip_json_nul_bytes(literal_backslashes)

    self.assertIs(result, literal_backslashes)

  def test_strip_json_nul_bytes_with_odd_backslashes_strips_escaped_nul(
      self,
  ) -> None:
    escaped_nul = r'{"key": "\\\u0000"}'

    result = pg_wire._sql_pg_ext.strip_json_nul_bytes(escaped_nul)

    self.assertEqual(result, r'{"key": "\\"}')

  @parameterized.named_parameters(
      (
          "null_pointer",
          0,
          "run_1",
          "traj_1",
          0,
          "{}",
          "2026-11-06T00:00:00Z",
          None,
          "",
          ValueError,
          "pgconn_ptr",
      ),
      (
          "negative_step_id",
          12345,
          "run_1",
          "traj_1",
          -1,
          "{}",
          "2026-11-06T00:00:00Z",
          None,
          "",
          ValueError,
          "step_id",
      ),
      (
          "step_id_overflow",
          12345,
          "run_1",
          "traj_1",
          2147483648,
          "{}",
          "2026-11-06T00:00:00Z",
          None,
          "",
          ValueError,
          "step_id exceeds PostgreSQL integer range",
      ),
      (
          "invalid_binary_col",
          12345,
          "run_1",
          "traj_1",
          0,
          "{}",
          "2026-11-06T00:00:00Z",
          b"data",
          "invalid;col",
          ValueError,
          "Invalid binary_col identifier",
      ),
      (
          "missing_binary_col",
          12345,
          "run_1",
          "traj_1",
          0,
          "{}",
          "2026-11-06T00:00:00Z",
          b"data",
          "",
          ValueError,
          "binary_col must be non-empty",
      ),
  )
  def test_pg_upsert_step_input_validation_raises_error(
      self,
      pgconn_ptr: int,
      run_id: str,
      trajectory_id: str,
      step_id: int,
      payload_json: str,
      created_at_iso: str,
      binary_data: bytes | None,
      binary_col: str,
      expected_exc: type[Exception],
      expected_regex: str,
  ) -> None:
    with self.assertRaisesRegex(expected_exc, expected_regex):
      pg_wire._sql_pg_ext.pg_upsert_step(
          pgconn_ptr=pgconn_ptr,
          run_id=run_id,
          trajectory_id=trajectory_id,
          step_id=step_id,
          payload_json=payload_json,
          created_at_iso=created_at_iso,
          binary_data=binary_data,
          binary_col=binary_col,
      )

  def test_pg_fetch_steps_ids_exceeding_limit_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "exceeds PostgreSQL parameter limit"
    ):
      pg_wire._sql_pg_ext.pg_fetch_steps(
          pgconn_ptr=12345,
          run_id="run_1",
          trajectory_ids=["t"] * 65535,
      )

  @parameterized.named_parameters(
      (
          "disconnect_error_08006",
          "[SQLSTATE:08006] connection severed",
          r"08006",
      ),
      (
          "admin_shutdown_57p01",
          "[SQLSTATE:57P01] terminating connection",
          r"57P01",
      ),
  )
  def test_raise_sqlalchemy_pg_error_on_disconnect_invalidates_connection(
      self,
      error_message: str,
      expected_regex: str,
  ) -> None:
    mock_conn = mock.MagicMock(spec=sa.Connection)
    exc = RuntimeError(error_message)

    with self.assertRaisesRegex(sa.exc.OperationalError, expected_regex) as ctx:
      pg_wire._raise_sqlalchemy_pg_error(mock_conn, "test_stmt", exc)

    self.assertTrue(ctx.exception.connection_invalidated)
    mock_conn.invalidate.assert_called_once_with(exc)

  @parameterized.named_parameters(
      (
          "integrity_error_23505",
          "[SQLSTATE:23505] unique violation",
          sa.exc.IntegrityError,
          r"23505",
      ),
      (
          "deadlock_error_40001",
          "[SQLSTATE:40001] serialization failure",
          sa.exc.OperationalError,
          r"40001",
      ),
      (
          "lock_not_available_55P03",
          "[SQLSTATE:55P03] lock not available",
          sa.exc.OperationalError,
          r"55P03",
      ),
      (
          "database_error_XX000",
          "[SQLSTATE:XX000] internal error",
          sa.exc.DatabaseError,
          r"XX000",
      ),
  )
  def test_raise_sqlalchemy_pg_error_on_non_disconnect_preserves_connection(
      self,
      error_message: str,
      expected_exc: type[Exception],
      expected_regex: str,
  ) -> None:
    mock_conn = mock.MagicMock(spec=sa.Connection)
    exc = RuntimeError(error_message)

    with self.assertRaisesRegex(expected_exc, expected_regex):
      pg_wire._raise_sqlalchemy_pg_error(mock_conn, "test_stmt", exc)

    mock_conn.invalidate.assert_not_called()

  @parameterized.named_parameters(
      ("sqlite_dialect", db_engine.Dialect.SQLITE, 54321, None),
      ("valid_postgresql_ptr", db_engine.Dialect.POSTGRESQL, 54321, 54321),
      (
          "non_int_postgresql_ptr",
          db_engine.Dialect.POSTGRESQL,
          "invalid_ptr",
          None,
      ),
      ("zero_postgresql_ptr", db_engine.Dialect.POSTGRESQL, 0, None),
  )
  def test_get_pgconn_ptr_returns_valid_address_or_none(
      self,
      dialect_name: str,
      raw_ptr: object,
      expected_ptr: int | None,
  ) -> None:
    mock_conn = mock.MagicMock()
    mock_conn.dialect.name = dialect_name
    mock_conn.connection.dbapi_connection.pgconn_ptr = raw_ptr

    actual_ptr = pg_wire._get_pgconn_ptr(mock_conn)

    self.assertEqual(actual_ptr, expected_ptr)

  def test_get_psycopg_lib_path_returns_string(self) -> None:
    path = pg_wire._get_psycopg_lib_path()
    self.assertIsInstance(path, str)

  def test_try_upsert_step_when_non_postgres_returns_false(self) -> None:
    mock_conn = mock.MagicMock()
    mock_conn.dialect.name = db_engine.Dialect.SQLITE
    result = pg_wire.try_upsert_step(
        mock_conn,
        run_id="run_1",
        trajectory_id="traj_1",
        step=trajectory_testing.STEP_1_1,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    self.assertFalse(result)

  def test_try_upsert_step_when_binary_unsupported_returns_false(self) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    with mock.patch.object(
        pg_wire._sql_pg_ext, "can_execute_pg_binary", return_value=False
    ):
      result = pg_wire.try_upsert_step(
          mock_conn,
          run_id="run_1",
          trajectory_id="traj_1",
          step=trajectory_testing.STEP_1_1,
          created_at=datetime.datetime.now(datetime.timezone.utc),
      )
    self.assertFalse(result)

  def test_try_upsert_step_when_valid_connection_returns_true(self) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    with (
        mock.patch.object(
            pg_wire._sql_pg_ext, "can_execute_pg_binary", return_value=True
        ),
        mock.patch.object(pg_wire._sql_pg_ext, "pg_upsert_step") as mock_upsert,
    ):
      result = pg_wire.try_upsert_step(
          mock_conn,
          run_id="run_1",
          trajectory_id="traj_1",
          step=trajectory_testing.STEP_1_1,
          created_at=datetime.datetime.now(datetime.timezone.utc),
      )
    self.assertTrue(result)
    mock_upsert.assert_called_once()

  def test_try_upsert_step_when_runtime_error_raises_integrity_error(
      self,
  ) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    with (
        mock.patch.object(
            pg_wire._sql_pg_ext, "can_execute_pg_binary", return_value=True
        ),
        mock.patch.object(
            pg_wire._sql_pg_ext,
            "pg_upsert_step",
            side_effect=RuntimeError("[SQLSTATE:23505] unique constraint"),
        ),
    ):
      with self.assertRaisesRegex(sa.exc.IntegrityError, r"23505"):
        pg_wire.try_upsert_step(
            mock_conn,
            run_id="run_1",
            trajectory_id="traj_1",
            step=trajectory_testing.STEP_1_1,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

  def test_try_fetch_steps_when_non_postgres_returns_none(self) -> None:
    mock_conn = mock.MagicMock()
    mock_conn.dialect.name = db_engine.Dialect.SQLITE
    result = pg_wire.try_fetch_steps(
        mock_conn, run_id="run_1", trajectory_ids=["traj_1"]
    )
    self.assertIsNone(result)

  def test_try_fetch_steps_when_fetch_unsupported_returns_none(self) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    with mock.patch.object(
        pg_wire._sql_pg_ext, "can_fetch_pg_binary", return_value=False
    ):
      result = pg_wire.try_fetch_steps(
          mock_conn, run_id="run_1", trajectory_ids=["traj_1"]
      )
    self.assertIsNone(result)

  def test_try_fetch_steps_when_exceeding_max_ids_returns_none(
      self,
  ) -> None:
    mock_conn = _make_pg_connection_mock(12345)

    result = pg_wire.try_fetch_steps(
        mock_conn,
        run_id="run_1",
        trajectory_ids=[f"t_{i}" for i in range(30_001)],
    )

    self.assertIsNone(result)

  def test_try_fetch_steps_when_valid_connection_returns_rows(self) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    fake_array = np.array([1, 2, 3, 4], dtype=np.uint8)
    expected_rows = [("traj_1", '{"step_id": 0}', fake_array)]
    with (
        mock.patch.object(
            pg_wire._sql_pg_ext, "can_fetch_pg_binary", return_value=True
        ),
        mock.patch.object(
            pg_wire._sql_pg_ext, "pg_fetch_steps", return_value=expected_rows
        ),
    ):
      rows = pg_wire.try_fetch_steps(
          mock_conn,
          run_id="run_1",
          trajectory_ids=["traj_1"],
          binary_col="routed_experts",
      )
    self.assertEqual(rows, expected_rows)

  def test_try_fetch_steps_when_runtime_error_raises_operational_error(
      self,
  ) -> None:
    mock_conn = _make_pg_connection_mock(12345)
    with (
        mock.patch.object(
            pg_wire._sql_pg_ext, "can_fetch_pg_binary", return_value=True
        ),
        mock.patch.object(
            pg_wire._sql_pg_ext,
            "pg_fetch_steps",
            side_effect=RuntimeError("[SQLSTATE:08006] connection severed"),
        ),
    ):
      with self.assertRaisesRegex(sa.exc.OperationalError, r"08006"):
        pg_wire.try_fetch_steps(
            mock_conn, run_id="run_1", trajectory_ids=["traj_1"]
        )


class PgWireLibpqAbiTest(parameterized.TestCase):
  """ABI and wire execution tests using hermetic fake_libpq shared libraries."""

  def setUp(self) -> None:
    super().setUp()
    if pg_wire._sql_pg_ext is None:
      self.skipTest("C++ _sql_pg_ext extension unavailable.")

    testing_dir = os.path.join(os.path.dirname(pg_wire.__file__), "testing")
    self.fake_pq_path = os.path.join(testing_dir, "libfake_pq.so")
    self.fake_pq_copy_path = os.path.join(testing_dir, "libfake_pq_copy.so")
    if not os.path.exists(self.fake_pq_path) or not os.path.exists(
        self.fake_pq_copy_path
    ):
      if "TEST_SRCDIR" in os.environ:
        self.fail(
            f"Hermetic fake libpq shared libraries missing in {testing_dir}"
        )
      self.skipTest("Hermetic fake_libpq shared libraries not built.")

  def test_can_execute_and_fetch_pg_binary_with_fake_pq_returns_true(
      self,
  ) -> None:
    self.assertTrue(
        pg_wire._sql_pg_ext.can_execute_pg_binary(self.fake_pq_path)
    )
    self.assertTrue(pg_wire._sql_pg_ext.can_fetch_pg_binary(self.fake_pq_path))

  def test_can_execute_and_fetch_pg_binary_with_fake_pq_copy_checks_binary_fallback(
      self,
  ) -> None:
    # COPY binary fallback handles upserts even without PQexecParams.
    self.assertTrue(
        pg_wire._sql_pg_ext.can_execute_pg_binary(self.fake_pq_copy_path)
    )
    # Fetch requires PQexecParams, so it returns False for libfake_pq_copy.so.
    self.assertFalse(
        pg_wire._sql_pg_ext.can_fetch_pg_binary(self.fake_pq_copy_path)
    )

  def test_try_upsert_step_with_fake_libpq_executes_wire_upsert(
      self,
  ) -> None:
    mock_conn = _make_pg_connection_mock(12345)

    with mock.patch.object(
        pg_wire, "_get_psycopg_lib_path", return_value=self.fake_pq_path
    ):
      result = pg_wire.try_upsert_step(
          mock_conn,
          run_id="run_1",
          trajectory_id="traj_1",
          step=trajectory_testing.STEP_1_1,
          created_at=datetime.datetime.now(datetime.timezone.utc),
      )
    self.assertTrue(result)

  def test_try_fetch_steps_with_fake_libpq_returns_rows(self) -> None:
    mock_conn = _make_pg_connection_mock(12345)

    with mock.patch.object(
        pg_wire, "_get_psycopg_lib_path", return_value=self.fake_pq_path
    ):
      rows = pg_wire.try_fetch_steps(
          mock_conn,
          run_id="run_1",
          trajectory_ids=["traj_1"],
          binary_col="routed_experts",
      )

    self.assertIsNotNone(rows)
    self.assertLen(rows, 1)
    tid, payload, binary_data = rows[0]
    self.assertEqual(tid, "traj_1")
    self.assertIn('"step_id": 1', payload)
    self.assertIsInstance(binary_data, np.ndarray)
    self.assertEqual(binary_data.tobytes(), b"\x01\x02\x03\x04")

  def test_concurrent_extension_calls_across_threads_succeed(self) -> None:
    def run_worker_task(task_idx: int) -> tuple[str, bytes]:
      lib_path = (
          self.fake_pq_path if task_idx % 2 == 0 else self.fake_pq_copy_path
      )
      cleaned_json = pg_wire._sql_pg_ext.strip_json_nul_bytes(
          f'{{"step_id": {task_idx}, "msg": "ok\\u0000\x00"}}'
      )
      pg_wire._sql_pg_ext.pg_upsert_step(
          pgconn_ptr=12345,
          run_id=f"run_{task_idx}",
          trajectory_id=f"traj_{task_idx}",
          step_id=task_idx,
          payload_json=cleaned_json,
          created_at_iso="2026-11-06T00:00:00Z",
          binary_data=b"\x01\x02\x03\x04",
          binary_col="routed_experts",
          psycopg_lib_path=lib_path,
      )
      rows = pg_wire._sql_pg_ext.pg_fetch_steps(
          pgconn_ptr=12345,
          run_id=f"run_{task_idx}",
          trajectory_ids=[f"traj_{task_idx}"],
          binary_col="routed_experts",
          psycopg_lib_path=self.fake_pq_path,
      )
      _, _, binary_arr = rows[0]
      return cleaned_json, binary_arr.tobytes()

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
      results = list(executor.map(run_worker_task, range(32)))

    self.assertLen(results, 32)
    for task_idx, (cleaned_json, raw_bytes) in enumerate(results):
      self.assertEqual(cleaned_json, f'{{"step_id": {task_idx}, "msg": "ok"}}')
      self.assertEqual(raw_bytes, b"\x01\x02\x03\x04")

  def test_pg_upsert_step_5_column_pqexecparams_success(self) -> None:
    pg_wire._sql_pg_ext.pg_upsert_step(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_id="traj_1",
        step_id=0,
        payload_json='{"step_id": 0}',
        created_at_iso="2026-11-06T00:00:00Z",
        psycopg_lib_path=self.fake_pq_path,
    )

  def test_pg_upsert_step_6_column_binary_pqexecparams_success(self) -> None:
    pg_wire._sql_pg_ext.pg_upsert_step(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_id="traj_1",
        step_id=0,
        payload_json='{"step_id": 0}',
        created_at_iso="2026-11-06T00:00:00Z",
        binary_data=b"\x01\x02\x03\x04",
        binary_col="routed_experts",
        psycopg_lib_path=self.fake_pq_path,
    )

  def test_pg_upsert_step_5_column_copy_binary_fallback_success(self) -> None:
    pg_wire._sql_pg_ext.pg_upsert_step(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_id="traj_1",
        step_id=0,
        payload_json='{"step_id": 0}',
        created_at_iso="2026-11-06T00:00:00Z",
        psycopg_lib_path=self.fake_pq_copy_path,
    )

  def test_pg_upsert_step_6_column_copy_binary_fallback_success(self) -> None:
    pg_wire._sql_pg_ext.pg_upsert_step(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_id="traj_1",
        step_id=0,
        payload_json='{"step_id": 0}',
        created_at_iso="2026-11-06T00:00:00Z",
        binary_data=b"\x01\x02\x03\x04",
        binary_col="routed_experts",
        psycopg_lib_path=self.fake_pq_copy_path,
    )

  def test_pg_upsert_step_failure_raises_runtime_error(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "SQLSTATE:23505"):
      pg_wire._sql_pg_ext.pg_upsert_step(
          pgconn_ptr=12345,
          run_id="fail_upsert",
          trajectory_id="traj_1",
          step_id=0,
          payload_json='{"step_id": 0}',
          created_at_iso="2026-11-06T00:00:00Z",
          psycopg_lib_path=self.fake_pq_path,
      )

  def test_pg_fetch_steps_5_column_success(self) -> None:
    rows = pg_wire._sql_pg_ext.pg_fetch_steps(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_ids=["traj_1"],
        psycopg_lib_path=self.fake_pq_path,
    )
    self.assertLen(rows, 1)
    tid, payload, binary_data = rows[0]
    self.assertEqual(tid, "traj_1")
    self.assertIn('"step_id": 1', payload)
    self.assertIsNone(binary_data)

  def test_pg_fetch_steps_6_column_binary_success(self) -> None:
    rows = pg_wire._sql_pg_ext.pg_fetch_steps(
        pgconn_ptr=12345,
        run_id="run_1",
        trajectory_ids=["traj_1"],
        binary_col="routed_experts",
        psycopg_lib_path=self.fake_pq_path,
    )
    self.assertLen(rows, 1)
    tid, payload, binary_data = rows[0]
    self.assertEqual(tid, "traj_1")
    self.assertIn('"step_id": 1', payload)
    self.assertIsInstance(binary_data, np.ndarray)
    self.assertEqual(binary_data.dtype, np.uint8)
    self.assertEqual(binary_data.tobytes(), b"\x01\x02\x03\x04")

  def test_pg_fetch_steps_failure_raises_runtime_error(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "SQLSTATE:08006"):
      pg_wire._sql_pg_ext.pg_fetch_steps(
          pgconn_ptr=12345,
          run_id="fail_fetch",
          trajectory_ids=["traj_1"],
          psycopg_lib_path=self.fake_pq_path,
      )


if __name__ == "__main__":
  absltest.main()
