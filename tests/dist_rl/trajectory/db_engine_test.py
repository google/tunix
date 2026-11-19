"""Unit tests for db_engine."""

from concurrent import futures
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.dist_rl.trajectory import db_engine


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
          json_serializer=db_engine._serialize_json,
          pool_pre_ping=True,
          pool_size=10,
          max_overflow=5,
          pool_timeout=3.0,
      )

  def test_serialize_json_strips_null_byte_escapes(self) -> None:
    engine = db_engine.create_trajectory_engine(
        db_engine.EngineConfig(url="sqlite:///:memory:")
    )
    self.addCleanup(engine.dispose)
    metadata = sa.MetaData()
    items = sa.Table(
        "items",
        metadata,
        sa.Column("item_id", sa.String, primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
      conn.execute(
          items.insert(),
          {
              "item_id": "row1",
              "payload": {
                  "stdout": "+ \x01\x00\x00\x00Pill",
                  "nested": ["ok", "a\x00b", 42],
                  "literal_escaped_backslash": r"\u0000",
                  "backslash_then_nul": "\\\x00",
                  "long_backslash_run": "\\" * 16_000,
              },
          },
      )
    with engine.connect() as conn:
      saved = conn.execute(
          sa.select(items.c.payload).where(items.c.item_id == "row1")
      ).scalar_one()

    self.assertEqual(
        saved,
        {
            "stdout": "+ \x01Pill",
            "nested": ["ok", "ab", 42],
            "literal_escaped_backslash": r"\u0000",
            "backslash_then_nul": "\\",
            "long_backslash_run": "\\" * 16_000,
        },
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
    # Unshared (default) handles each own a private engine.
    config = db_engine.EngineConfig(url="sqlite:///:memory:")

    handle_a = db_engine.acquire_engine(config)
    self.addCleanup(handle_a.release)
    handle_b = db_engine.acquire_engine(config)
    self.addCleanup(handle_b.release)

    self.assertIsNot(handle_a.engine, handle_b.engine)

  def test_acquire_engine_shared_same_config_returns_same_engine(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")

    handle_a = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(handle_a.release)
    handle_b = db_engine.acquire_engine(
        db_engine.EngineConfig(url=f"sqlite:///{db_path}"), shared=True
    )
    self.addCleanup(handle_b.release)

    self.assertIs(handle_a.engine, handle_b.engine)

  def test_acquire_engine_shared_different_config_returns_distinct_engines(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    url = f"sqlite:///{db_path}"

    handle_a = db_engine.acquire_engine(
        db_engine.EngineConfig(url=url), shared=True
    )
    self.addCleanup(handle_a.release)
    handle_b = db_engine.acquire_engine(
        db_engine.EngineConfig(url=url, pool_size=2), shared=True
    )
    self.addCleanup(handle_b.release)

    self.assertIsNot(handle_a.engine, handle_b.engine)

  def test_acquire_engine_shared_and_unshared_return_distinct_engines(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")

    shared_handle = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(shared_handle.release)
    private_handle = db_engine.acquire_engine(config)
    self.addCleanup(private_handle.release)

    self.assertIsNot(shared_handle.engine, private_handle.engine)

  def test_acquire_engine_shared_engine_outlives_released_handle(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")
    writer = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(writer.release)
    reader = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(reader.release)

    with writer.engine.begin() as conn:
      conn.execute(sa.text("CREATE TABLE items (item_id TEXT PRIMARY KEY)"))
      conn.execute(sa.text("INSERT INTO items VALUES ('row1')"))
    writer.release()

    with reader.engine.connect() as conn:
      rows = conn.execute(sa.text("SELECT item_id FROM items")).scalars().all()
    self.assertEqual(rows, ["row1"])

  def test_acquire_engine_shared_disposes_only_on_last_release(self) -> None:
    mock_engine = mock.create_autospec(sa.Engine, instance=True)
    config = db_engine.EngineConfig(url="sqlite:///:memory:")
    with mock.patch.object(
        db_engine,
        "create_trajectory_engine",
        autospec=True,
        return_value=mock_engine,
    ) as mock_create:
      handle_a = db_engine.acquire_engine(config, shared=True)
      self.addCleanup(handle_a.release)
      handle_b = db_engine.acquire_engine(config, shared=True)
      self.addCleanup(handle_b.release)

    handle_a.release()
    handle_a.release()  # A repeated release must not drop handle_b's ref.
    mock_engine.dispose.assert_not_called()
    handle_b.release()

    mock_create.assert_called_once_with(config)
    mock_engine.dispose.assert_called_once_with(close=True)

  def test_acquire_engine_shared_after_last_release_builds_new_engine(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")
    first = db_engine.acquire_engine(config, shared=True)
    first_engine = first.engine
    first.release()

    second = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(second.release)

    self.assertIsNot(second.engine, first_engine)

  def test_acquire_engine_shared_new_pid_builds_new_engine(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")
    parent_handle = db_engine.acquire_engine(config, shared=True)
    self.addCleanup(parent_handle.release)

    # Simulates acquiring in a forked child, which has a new pid.
    with mock.patch.object(
        db_engine.os, "getpid", autospec=True, return_value=os.getpid() + 1
    ):
      child_handle = db_engine.acquire_engine(config, shared=True)
      self.assertIsNot(child_handle.engine, parent_handle.engine)
      child_handle.release()

  def test_release_shared_engine_in_new_pid_keeps_parent_connections(
      self,
  ) -> None:
    mock_engine = mock.create_autospec(sa.Engine, instance=True)
    with mock.patch.object(
        db_engine,
        "create_trajectory_engine",
        autospec=True,
        return_value=mock_engine,
    ):
      handle = db_engine.acquire_engine(
          db_engine.EngineConfig(url="sqlite:///:memory:"), shared=True
      )
      self.addCleanup(handle.release)

    # Simulates a forked child releasing a handle inherited from its parent.
    with mock.patch.object(
        db_engine.os, "getpid", autospec=True, return_value=os.getpid() + 1
    ):
      handle.release()

    mock_engine.dispose.assert_called_once_with(close=False)

  def test_reset_in_child_clears_registry_and_inherited_release_still_works(
      self,
  ) -> None:
    mock_engine = mock.create_autospec(sa.Engine, instance=True)
    config = db_engine.EngineConfig(url="sqlite:///:memory:")
    with mock.patch.object(
        db_engine,
        "create_trajectory_engine",
        autospec=True,
        return_value=mock_engine,
    ):
      handle = db_engine.acquire_engine(config, shared=True)
      self.addCleanup(handle.release)

    # Simulates the fork hook running in a child with a new pid.
    with mock.patch.object(
        db_engine.os, "getpid", autospec=True, return_value=os.getpid() + 1
    ):
      db_engine._reset_shared_engines_in_child()
      self.assertEmpty(db_engine._SHARED_ENGINES)
      handle.release()

    mock_engine.dispose.assert_called_once_with(close=False)

  def test_acquire_engine_shared_concurrent_acquires_build_one_engine(
      self,
  ) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "shared.db")
    config = db_engine.EngineConfig(url=f"sqlite:///{db_path}")
    real_create = db_engine.create_trajectory_engine

    with mock.patch.object(
        db_engine,
        "create_trajectory_engine",
        autospec=True,
        side_effect=real_create,
    ) as mock_create:
      with futures.ThreadPoolExecutor(max_workers=8) as executor:
        handles = list(
            executor.map(
                lambda _: db_engine.acquire_engine(config, shared=True),
                range(16),
            )
        )
    for handle in handles:
      self.addCleanup(handle.release)

    mock_create.assert_called_once_with(config)
    self.assertLen({id(handle.engine) for handle in handles}, 1)

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
