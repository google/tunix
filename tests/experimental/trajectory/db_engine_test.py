"""Unit tests for db_engine."""

import os
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import db_engine


def _query_pragma(engine: sa.Engine, pragma: str) -> Any:
  """Executes a PRAGMA query and returns the scalar value."""
  with engine.connect() as conn:
    result = conn.execute(sa.text(f"PRAGMA {pragma};"))
    return result.scalar()


class DbEngineTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("memory_keyword", "sqlite:///:memory:"),
      ("empty_path", "sqlite:///"),
      ("uri_mode_memory", "sqlite:///file:memdb1?cache=shared&mode=memory"),
  )
  def test_create_trajectory_engine_in_memory_sqlite_uses_static_pool(
      self, url: str
  ) -> None:
    engine = db_engine.create_trajectory_engine(url)
    self.addCleanup(engine.dispose)

    self.assertEqual(engine.dialect.name, db_engine.SQLITE_DIALECT)
    self.assertIsInstance(engine.pool, sa.pool.StaticPool)

  @parameterized.named_parameters(
      ("foreign_keys", "foreign_keys", 1),
      ("busy_timeout", "busy_timeout", 5000),
  )
  def test_create_trajectory_engine_sqlite_sets_default_pragmas(
      self, pragma: str, expected_value: int
  ) -> None:
    engine = db_engine.create_trajectory_engine("sqlite:///:memory:")
    self.addCleanup(engine.dispose)

    self.assertEqual(_query_pragma(engine, pragma), expected_value)

  def test_create_trajectory_engine_file_sqlite_enforces_wal(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "test_wal.db")
    engine = db_engine.create_trajectory_engine(f"sqlite:///{db_path}")
    self.addCleanup(engine.dispose)

    journal_val = _query_pragma(engine, "journal_mode")
    self.assertEqual(str(journal_val).lower(), "wal")

  def test_create_trajectory_engine_file_sqlite_configures_pooling(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "test_pool.db")
    engine = db_engine.create_trajectory_engine(
        f"sqlite:///{db_path}", pool_size=5, max_overflow=2
    )
    self.addCleanup(engine.dispose)

    pool = cast(sa.pool.QueuePool, engine.pool)
    self.assertIsInstance(pool, sa.pool.QueuePool)
    self.assertEqual(pool.size(), 5)
    self.assertEqual(pool._max_overflow, 2)

  def test_create_trajectory_engine_postgresql_configures_engine(
      self,
  ) -> None:
    url = "postgresql+psycopg2://user:pass@localhost:5432/testdb"
    with mock.patch.object(
        db_engine.sa, "create_engine", autospec=True, spec_set=True
    ) as mock_create:
      db_engine.create_trajectory_engine(url, pool_size=10, max_overflow=5)
      mock_create.assert_called_once_with(
          url,
          echo=False,
          pool_pre_ping=True,
          pool_size=10,
          max_overflow=5,
      )

  @parameterized.named_parameters(
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_create_trajectory_engine_empty_url_raises_value_error(
      self, url: str
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, r"Database URL must be a non-empty string"
    ):
      db_engine.create_trajectory_engine(url)

  def test_create_trajectory_engine_unsupported_dialect_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, r"Unsupported database dialect"):
      db_engine.create_trajectory_engine("mysql://user:pass@localhost/testdb")


if __name__ == "__main__":
  absltest.main()
