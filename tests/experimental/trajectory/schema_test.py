"""Tests for SQL Trajectory Store relational database schema."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import schema_testing


@dataclasses.dataclass(frozen=True)
class _DialectSpec:
  url: str
  dialect: sa.Dialect
  expected_json: str
  expected_timestamp: str


_DIALECT_SPECS = (
    (
        "postgresql",
        _DialectSpec(
            url="postgresql://",
            dialect=postgresql.dialect(),
            expected_json="JSONB",
            expected_timestamp="TIMESTAMP WITH TIME ZONE",
        ),
    ),
    (
        "sqlite",
        _DialectSpec(
            url="sqlite://",
            dialect=sqlite.dialect(),
            expected_json="JSON",
            expected_timestamp="DATETIME",
        ),
    ),
)


class SchemaCompilationTest(parameterized.TestCase):
  """Verifies offline DDL and index compilation for PostgreSQL and SQLite."""

  @parameterized.named_parameters(*_DIALECT_SPECS)
  def test_ddl_compilation(self, spec: _DialectSpec) -> None:
    """Verifies DDL compilation across PostgreSQL and SQLite dialects."""
    runs_ddl = str(
        sa.schema.CreateTable(schema.RUNS_TABLE).compile(dialect=spec.dialect)
    )
    self.assertIn("CREATE TABLE runs", runs_ddl)
    self.assertIn("run_id VARCHAR(128) NOT NULL", runs_ddl)
    self.assertIn("agent_name VARCHAR(128) NOT NULL", runs_ddl)
    self.assertIn(spec.expected_timestamp, runs_ddl)
    self.assertIn(spec.expected_json, runs_ddl)
    self.assertIn("CONSTRAINT pk_runs PRIMARY KEY (run_id)", runs_ddl)

    traj_ddl = str(
        sa.schema.CreateTable(schema.TRAJECTORIES_TABLE).compile(
            dialect=spec.dialect
        )
    )
    self.assertIn("CREATE TABLE trajectories", traj_ddl)
    self.assertIn("run_id VARCHAR(128) NOT NULL", traj_ddl)
    self.assertIn("trajectory_id VARCHAR(128) NOT NULL", traj_ddl)
    self.assertIn("trajectory_metadata", traj_ddl)
    self.assertIn(spec.expected_json, traj_ddl)
    self.assertIn(spec.expected_timestamp, traj_ddl)
    self.assertIn(
        "CONSTRAINT pk_trajectories PRIMARY KEY (run_id, trajectory_id)",
        traj_ddl,
    )
    self.assertIn(
        "CONSTRAINT fk_trajectories_runs FOREIGN KEY(run_id) REFERENCES runs"
        " (run_id) ON DELETE CASCADE",
        traj_ddl,
    )

    steps_ddl = str(
        sa.schema.CreateTable(schema.STEPS_TABLE).compile(dialect=spec.dialect)
    )
    self.assertIn("CREATE TABLE steps", steps_ddl)
    self.assertIn("run_id VARCHAR(128) NOT NULL", steps_ddl)
    self.assertIn("trajectory_id VARCHAR(128) NOT NULL", steps_ddl)
    self.assertIn("step_id INTEGER NOT NULL", steps_ddl)
    self.assertIn(spec.expected_json, steps_ddl)
    self.assertIn(spec.expected_timestamp, steps_ddl)
    self.assertIn(
        "CONSTRAINT pk_steps PRIMARY KEY (run_id, trajectory_id, step_id)",
        steps_ddl,
    )
    self.assertIn(
        "CONSTRAINT fk_steps_trajectories FOREIGN KEY(run_id, trajectory_id)"
        " REFERENCES trajectories (run_id, trajectory_id) ON DELETE CASCADE",
        steps_ddl,
    )

  @parameterized.named_parameters(*_DIALECT_SPECS)
  def test_secondary_indexes_compilation(self, spec: _DialectSpec) -> None:
    all_indexes = []
    for table in schema.METADATA.tables.values():
      all_indexes.extend(table.indexes)
    self.assertLen(all_indexes, 4)
    for idx in all_indexes:
      idx_ddl = str(sa.schema.CreateIndex(idx).compile(dialect=spec.dialect))
      self.assertIn("CREATE INDEX", idx_ddl)

  @parameterized.named_parameters(*_DIALECT_SPECS)
  def test_metadata_create_all_and_drop_all(self, spec: _DialectSpec) -> None:
    """Verifies full DDL create and drop sequences compile for dialect."""
    ddl_statements = []

    def dump(sql, *multiparams, **params):
      del multiparams, params
      ddl_statements.append(str(sql.compile(dialect=spec.dialect)))

    mock_engine = sa.create_mock_engine(spec.url, dump)
    schema.METADATA.create_all(mock_engine, checkfirst=False)

    full_create_script = "\n".join(ddl_statements)
    self.assertIn("CREATE TABLE runs", full_create_script)
    self.assertIn("CREATE TABLE trajectories", full_create_script)
    self.assertIn("CREATE TABLE steps", full_create_script)
    self.assertIn("CREATE INDEX idx_runs_status", full_create_script)
    self.assertIn("CREATE INDEX idx_trajectories_status", full_create_script)
    self.assertIn(
        "CREATE INDEX idx_trajectories_created_at", full_create_script
    )
    self.assertIn(
        "CREATE INDEX idx_trajectories_updated_at", full_create_script
    )

    ddl_statements.clear()
    schema.METADATA.drop_all(mock_engine, checkfirst=False)

    full_drop_script = "\n".join(ddl_statements)
    self.assertIn("DROP TABLE steps", full_drop_script)
    self.assertIn("DROP TABLE trajectories", full_drop_script)
    self.assertIn("DROP TABLE runs", full_drop_script)


class SqliteSchemaTest(schema_testing.SchemaTestCase):
  """Hermetic SQLite execution tests for SQL Trajectory Store schema."""

  def create_engine(self) -> sa.Engine:
    return schema_testing.create_sqlite_memory_engine()


if __name__ == "__main__":
  absltest.main()
