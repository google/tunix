"""Abstract test cases defining contract tests for Trajectory Store schemas."""

import abc
import datetime
import sqlite3
from typing import Any

from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import schema


class ParameterizedABCMeta(type(parameterized.TestCase), abc.ABCMeta):
  """Combined metaclass resolving parameterized.TestCase and abc.ABCMeta."""


def _set_sqlite_pragma(
    dbapi_connection: sqlite3.Connection,
    connection_record: sa.pool.ConnectionPoolEntry,
) -> None:
  """Enables foreign key constraint enforcement on SQLite connections."""
  del connection_record
  cursor = dbapi_connection.cursor()
  cursor.execute("PRAGMA foreign_keys=ON")
  cursor.close()


def create_sqlite_memory_engine(shared_pool: bool = False) -> sa.Engine:
  """Creates an in-memory SQLite engine with foreign key enforcement enabled.

  Args:
    shared_pool: If True, uses StaticPool and check_same_thread=False so that
      the in-memory database is shared across multiple concurrent threads (e.g.
      for asynchronous background worker threads).

  Returns:
    A configured SQLAlchemy Engine.
  """
  kwargs: dict[str, Any] = {}
  if shared_pool:
    kwargs["poolclass"] = sa.pool.StaticPool
    kwargs["connect_args"] = {"check_same_thread": False}
  engine = sa.create_engine("sqlite:///:memory:", **kwargs)
  sa.event.listen(engine, "connect", _set_sqlite_pragma)
  return engine


def fetch_all(
    engine: sa.Engine, statement: sa.sql.Executable
) -> list[dict[str, Any]]:
  """Executes a statement and returns all rows as dictionary mappings."""
  with engine.connect() as conn:
    result = conn.execute(statement)
    return [dict(row) for row in result.mappings().all()]


class SchemaTestCase(parameterized.TestCase, metaclass=ParameterizedABCMeta):
  """Abstract contract test case for SQL Trajectory Store database engines.

  Subclasses must implement `create_engine()` to provide a dialect-specific
  engine (e.g. SQLite, PostgreSQL).
  """

  @abc.abstractmethod
  def create_engine(self) -> sa.Engine:
    """Factory method to provide a database engine for the test suite."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = self.create_engine()
    schema.METADATA.create_all(self.engine)

  def tearDown(self) -> None:
    schema.METADATA.drop_all(self.engine)
    self.engine.dispose()
    super().tearDown()

  def test_insert_and_defaults(self) -> None:
    """Verifies inserting runs, trajectories, and steps with default values."""
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="run_alpha",
              agent_name="agent_test",
              environment_name="env_test",
          )
      )
      conn.execute(
          schema.TRAJECTORIES_TABLE.insert().values(
              run_id="run_alpha",
              trajectory_id="traj_001",
          )
      )
      conn.execute(
          schema.STEPS_TABLE.insert().values(
              run_id="run_alpha",
              trajectory_id="traj_001",
              step_id=0,
              payload={"reward": 1.0, "observation": "initial"},
          )
      )

      run_row = (
          conn.execute(
              sa.select(schema.RUNS_TABLE).where(
                  schema.RUNS_TABLE.c.run_id == "run_alpha"
              )
          )
          .mappings()
          .one()
      )
      self.assertEqual(run_row["status"], schema.Status.PENDING)
      self.assertEqual(run_row["config"], {})

      traj_row = (
          conn.execute(
              sa.select(schema.TRAJECTORIES_TABLE).where(
                  (schema.TRAJECTORIES_TABLE.c.run_id == "run_alpha")
                  & (schema.TRAJECTORIES_TABLE.c.trajectory_id == "traj_001")
              )
          )
          .mappings()
          .one()
      )
      self.assertEqual(traj_row["status"], schema.Status.PENDING)
      self.assertEqual(traj_row["trajectory_metadata"], {})

      step_row = (
          conn.execute(
              sa.select(schema.STEPS_TABLE).where(
                  (schema.STEPS_TABLE.c.run_id == "run_alpha")
                  & (schema.STEPS_TABLE.c.trajectory_id == "traj_001")
                  & (schema.STEPS_TABLE.c.step_id == 0)
              )
          )
          .mappings()
          .one()
      )
      self.assertEqual(
          step_row["payload"],
          {"reward": 1.0, "observation": "initial"},
      )

  def test_delete_run_cascades_to_trajectories_and_steps(self) -> None:
    """Verifies that deleting a run cascades to trajectories and steps."""
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="run_alpha",
              agent_name="agent_test",
              environment_name="env_test",
          )
      )
      conn.execute(
          schema.TRAJECTORIES_TABLE.insert().values(
              run_id="run_alpha",
              trajectory_id="traj_001",
          )
      )
      conn.execute(
          schema.STEPS_TABLE.insert().values(
              run_id="run_alpha",
              trajectory_id="traj_001",
              step_id=0,
              payload={"reward": 1.0, "observation": "initial"},
          )
      )

      conn.execute(
          schema.RUNS_TABLE.delete().where(
              schema.RUNS_TABLE.c.run_id == "run_alpha"
          )
      )

      self.assertEqual(
          conn.execute(
              sa.select(sa.func.count()).select_from(schema.TRAJECTORIES_TABLE)
          ).scalar(),
          0,
      )
      self.assertEqual(
          conn.execute(
              sa.select(sa.func.count()).select_from(schema.STEPS_TABLE)
          ).scalar(),
          0,
      )

  def test_delete_trajectory_cascades_to_steps(self) -> None:
    """Verifies that deleting a trajectory cascades to associated steps."""
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="run_beta",
              agent_name="agent_test",
          )
      )
      conn.execute(
          schema.TRAJECTORIES_TABLE.insert().values(
              run_id="run_beta",
              trajectory_id="traj_002",
          )
      )
      conn.execute(
          schema.STEPS_TABLE.insert().values(
              run_id="run_beta",
              trajectory_id="traj_002",
              step_id=0,
              payload={"obs": "step0"},
          )
      )
      conn.execute(
          schema.STEPS_TABLE.insert().values(
              run_id="run_beta",
              trajectory_id="traj_002",
              step_id=1,
              payload={"obs": "step1"},
          )
      )

      # Delete trajectory only (keeping run_beta intact).
      conn.execute(
          schema.TRAJECTORIES_TABLE.delete().where(
              (schema.TRAJECTORIES_TABLE.c.run_id == "run_beta")
              & (schema.TRAJECTORIES_TABLE.c.trajectory_id == "traj_002")
          )
      )

      # Associated steps and trajectory are deleted; parent run remains.
      self.assertEqual(
          conn.execute(
              sa.select(sa.func.count()).select_from(schema.STEPS_TABLE)
          ).scalar(),
          0,
      )
      self.assertEqual(
          conn.execute(
              sa.select(sa.func.count()).select_from(schema.TRAJECTORIES_TABLE)
          ).scalar(),
          0,
      )
      self.assertEqual(
          conn.execute(
              sa.select(sa.func.count()).select_from(schema.RUNS_TABLE)
          ).scalar(),
          1,
      )

  def test_duplicate_primary_key_raises_integrity_error(self) -> None:
    """Verifies inserting duplicate primary keys raises IntegrityError."""
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="run_alpha",
              agent_name="agent_test",
          )
      )

    with self.assertRaisesRegex(
        sa.exc.IntegrityError, r"(?i)(unique|duplicate)"
    ):
      with self.engine.begin() as conn:
        conn.execute(
            schema.RUNS_TABLE.insert().values(
                run_id="run_alpha",
                agent_name="agent_test_duplicate",
            )
        )

  def test_foreign_key_insert_violation_raises_integrity_error(self) -> None:
    """Verifies inserting rows referencing nonexistent parents raises error."""
    with self.assertRaisesRegex(sa.exc.IntegrityError, r"(?i)foreign key"):
      with self.engine.begin() as conn:
        conn.execute(
            schema.TRAJECTORIES_TABLE.insert().values(
                run_id="nonexistent_run",
                trajectory_id="traj_001",
            )
        )

    with self.assertRaisesRegex(sa.exc.IntegrityError, r"(?i)foreign key"):
      with self.engine.begin() as conn:
        conn.execute(
            schema.STEPS_TABLE.insert().values(
                run_id="nonexistent_run",
                trajectory_id="nonexistent_traj",
                step_id=0,
                payload={"obs": "step0"},
            )
        )

  def test_non_nullable_column_raises_integrity_error(self) -> None:
    """Verifies omitting required non-nullable columns raises IntegrityError."""
    with self.assertRaisesRegex(
        sa.exc.IntegrityError, r"(?i)(not null|violates not-null)"
    ):
      with self.engine.begin() as conn:
        # agent_name is nullable=False without a server default.
        conn.execute(
            schema.RUNS_TABLE.insert().values(
                run_id="run_missing_agent",
            )
        )

  def test_update_trajectory_updates_timestamp(self) -> None:
    """Verifies that updating a trajectory updates its updated_at column."""
    past_timestamp = datetime.datetime(
        2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="run_alpha",
              agent_name="agent_test",
          )
      )
      conn.execute(
          schema.TRAJECTORIES_TABLE.insert().values(
              run_id="run_alpha",
              trajectory_id="traj_001",
              updated_at=past_timestamp,
          )
      )
      conn.execute(
          schema.TRAJECTORIES_TABLE.update()
          .where(
              (schema.TRAJECTORIES_TABLE.c.run_id == "run_alpha")
              & (schema.TRAJECTORIES_TABLE.c.trajectory_id == "traj_001")
          )
          .values(status=schema.Status.COMPLETED)
      )
      updated_row = (
          conn.execute(
              sa.select(schema.TRAJECTORIES_TABLE).where(
                  (schema.TRAJECTORIES_TABLE.c.run_id == "run_alpha")
                  & (schema.TRAJECTORIES_TABLE.c.trajectory_id == "traj_001")
              )
          )
          .mappings()
          .one()
      )
      self.assertEqual(updated_row["status"], schema.Status.COMPLETED)
      self.assertIsNotNone(updated_row["updated_at"])
      updated_at = updated_row["updated_at"]
      if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)
      self.assertGreater(updated_at, past_timestamp)
