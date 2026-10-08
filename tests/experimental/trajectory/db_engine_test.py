"""Unit tests for db_engine."""

import os
from typing import Any
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
    engine = db_engine.create_trajectory_engine(db_engine.EngineConfig(url=url))
    self.addCleanup(engine.dispose)

    self.assertEqual(engine.dialect.name, db_engine.Dialect.SQLITE)
    self.assertIsInstance(engine.pool, sa.pool.StaticPool)

  @parameterized.named_parameters(
      ("foreign_keys", "foreign_keys", 1),
      ("busy_timeout", "busy_timeout", 5000),
  )
  def test_create_trajectory_engine_sqlite_sets_default_pragmas(
      self, pragma: str, expected_value: int
  ) -> None:
    engine = db_engine.create_trajectory_engine(
        db_engine.EngineConfig(url="sqlite:///:memory:")
    )
    self.addCleanup(engine.dispose)

    self.assertEqual(_query_pragma(engine, pragma), expected_value)

  def test_create_trajectory_engine_file_sqlite_enforces_wal(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "test_wal.db")
    engine = db_engine.create_trajectory_engine(
        db_engine.EngineConfig(url=f"sqlite:///{db_path}")
    )
    self.addCleanup(engine.dispose)

    journal_val = _query_pragma(engine, "journal_mode")
    self.assertEqual(str(journal_val).lower(), "wal")

  def test_create_trajectory_engine_in_memory_sqlite_skips_file_pragmas(
      self,
  ) -> None:
    engine = db_engine.create_trajectory_engine(
        db_engine.EngineConfig(url="sqlite:///:memory:")
    )
    self.addCleanup(engine.dispose)

    self.assertTrue(
        sa.event.contains(engine, "connect", db_engine._set_sqlite_pragmas)
    )
    self.assertFalse(
        sa.event.contains(engine, "connect", db_engine._set_sqlite_file_pragmas)
    )

  def test_create_trajectory_engine_file_sqlite_configures_pooling(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "test_pool.db")
    url = f"sqlite:///{db_path}"
    real_create_engine = sa.create_engine

    with mock.patch.object(
        db_engine.sa,
        "create_engine",
        autospec=True,
        side_effect=real_create_engine,
    ) as mock_create:
      engine = db_engine.create_trajectory_engine(
          db_engine.EngineConfig(url=url, pool_size=5, max_overflow=2)
      )
    self.addCleanup(engine.dispose)

    _, kwargs = mock_create.call_args
    self.assertEqual(kwargs["pool_size"], 5)
    self.assertEqual(kwargs["max_overflow"], 2)

  def test_create_trajectory_engine_postgresql_configures_engine(
      self,
  ) -> None:
    url = "postgresql+psycopg2://user:pass@localhost:5432/testdb"
    with mock.patch.object(
        db_engine.sa, "create_engine", autospec=True, spec_set=True
    ) as mock_create:
      db_engine.create_trajectory_engine(
          db_engine.EngineConfig(
              url=url, pool_size=10, max_overflow=5, pool_timeout_s=3.0
          )
      )
      mock_create.assert_called_once_with(
          url,
          echo=False,
          pool_pre_ping=True,
          pool_size=10,
          max_overflow=5,
          pool_timeout=3.0,
      )

  def test_engine_config_defaults_match_single_writer_budget(self) -> None:
    config = db_engine.EngineConfig(url="sqlite:///:memory:")

    self.assertEqual(config.pool_size, 1)
    self.assertEqual(config.max_overflow, 1)
    self.assertEqual(config.pool_timeout_s, 10.0)

  def test_engine_config_equal_values_hash_equal(self) -> None:
    config_a = db_engine.EngineConfig(url="sqlite:///a.db", pool_size=2)
    config_b = db_engine.EngineConfig(url="sqlite:///a.db", pool_size=2)

    self.assertEqual(config_a, config_b)
    self.assertEqual(hash(config_a), hash(config_b))

  def test_acquire_engine_valid_config_returns_handle_to_configured_engine(
      self,
  ) -> None:
    config = db_engine.EngineConfig(url="sqlite:///:memory:")

    handle = db_engine.acquire_engine(config)
    self.addCleanup(handle.release)

    self.assertEqual(handle.engine.dialect.name, db_engine.Dialect.SQLITE)
    self.assertEqual(_query_pragma(handle.engine, "foreign_keys"), 1)

  def test_acquire_engine_same_config_returns_distinct_engines(self) -> None:
    # Documents current behaviour, one engine per handle. A shared, refcounted
    # engine per config will replace this test.
    config = db_engine.EngineConfig(url="sqlite:///:memory:")

    handle_a = db_engine.acquire_engine(config)
    self.addCleanup(handle_a.release)
    handle_b = db_engine.acquire_engine(config)
    self.addCleanup(handle_b.release)

    self.assertIsNot(handle_a.engine, handle_b.engine)

  def test_release_called_twice_invokes_release_fn_once(self) -> None:
    release_fn = mock.Mock()
    handle = db_engine.EngineHandle(
        engine=mock.create_autospec(sa.Engine, instance=True),
        release_fn=release_fn,
    )

    handle.release()
    handle.release()

    release_fn.assert_called_once_with()

  def test_acquire_engine_release_disposes_engine(self) -> None:
    mock_engine = mock.create_autospec(sa.Engine, instance=True)
    with mock.patch.object(
        db_engine,
        "create_trajectory_engine",
        autospec=True,
        return_value=mock_engine,
    ):
      handle = db_engine.acquire_engine(
          db_engine.EngineConfig(url="sqlite:///:memory:")
      )

    handle.release()

    mock_engine.dispose.assert_called_once_with()

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
      db_engine.create_trajectory_engine(db_engine.EngineConfig(url=url))

  def test_create_trajectory_engine_unsupported_dialect_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, r"Unsupported database dialect"):
      db_engine.create_trajectory_engine(
          db_engine.EngineConfig(url="mysql://user:pass@localhost/testdb")
      )

  def test_create_trajectory_engine_malformed_url_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, r"Database URL could not be parsed"
    ):
      db_engine.create_trajectory_engine(
          db_engine.EngineConfig(url="not-a-url")
      )

  @parameterized.named_parameters(
      (
          "postgres_password",
          "postgresql+psycopg2://postgres:s3cret@10.0.0.5:5432/bench",
          "postgresql+psycopg2://postgres:***@10.0.0.5:5432/bench",
      ),
      (
          "no_password",
          "postgresql+psycopg2://postgres@127.0.0.1:5432/bench",
          "postgresql+psycopg2://postgres@127.0.0.1:5432/bench",
      ),
      ("sqlite_file", "sqlite:////tmp/bench.db", "sqlite:////tmp/bench.db"),
  )
  def test_redact_url_masks_password_only(
      self, url: str, expected: str
  ) -> None:
    self.assertEqual(db_engine.redact_url(url), expected)


if __name__ == "__main__":
  absltest.main()
