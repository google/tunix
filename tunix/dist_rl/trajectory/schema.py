"""Relational database schema definitions for SQL Trajectory Store.

This module defines the relational schema (tables, constraints, and indexes)
for persisting agent rollouts and evaluation trajectories across both
PostgreSQL (Cloud SQL production) and SQLite (local testing and CI/CD).

Dialect-Specific Notes:
  - JSON Storage: Columns use SQLAlchemy's JSON type with a PostgreSQL JSONB
    variant (`sa.JSON().with_variant(postgresql.JSONB, "postgresql")`). On
    PostgreSQL, this provides binary JSON storage and indexability; on SQLite,
    it is stored as serialized JSON text.
  - Timestamps: DateTime columns use `DateTime(timezone=True)`. On PostgreSQL,
    this compiles to `TIMESTAMPTZ`. On SQLite, timestamps are serialized as
    ISO-8601 strings.
  - Timestamp Updates: `trajectories.updated_at` specifies
    `onupdate=sa.func.now()`. This is a client-side SQLAlchemy Core construct
    evaluated during SQLAlchemy-generated `update()` statements. Direct updates
    outside SQLAlchemy on PostgreSQL require a database trigger.
  - Table Partitioning: `steps` specifies
    `postgresql_partition_by="HASH (run_id)"` with `STEPS_HASH_PARTITIONS` (16)
    fixed modulus child partitions (`steps_p00` .. `steps_p15`) registered via
    PostgreSQL `after_create` DDL hooks on `STEPS_TABLE`. On PostgreSQL, this
    bounds per-partition B-tree index size and enables single-partition pruning
    on `WHERE run_id = ...` queries without unbounded partition growth; on
    SQLite, partitioning is skipped and `steps` is created as a standard table.
"""

import enum
from typing import Final

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


class Status(enum.StrEnum):
  """Lifecycle status for runs and trajectories."""

  @staticmethod
  def _generate_next_value_(
      name: str, start: int, count: int, last_values: list[str]
  ) -> str:
    del start, count, last_values
    return name

  UNKNOWN = enum.auto()
  PENDING = enum.auto()
  RUNNING = enum.auto()
  COMPLETED = enum.auto()
  FAILED = enum.auto()
  CANCELLED = enum.auto()
  TIMEOUT = enum.auto()


# Built-in SQLAlchemy JSON type: compiles to JSONB on PostgreSQL and JSON
# on SQLite.
_JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB, "postgresql")

METADATA = sa.MetaData()

RUNS_TABLE = sa.Table(
    "runs",
    METADATA,
    sa.Column("run_id", sa.String(128), nullable=False),
    sa.Column("agent_name", sa.String(128), nullable=False),
    sa.Column("environment_name", sa.String(128), nullable=True),
    sa.Column(
        "status",
        sa.String(32),
        nullable=False,
        default=Status.UNKNOWN,
        server_default=sa.text(f"'{Status.UNKNOWN}'"),
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column(
        "config",
        _JSON_TYPE,
        nullable=False,
        default=dict,
        server_default=sa.text("'{}'"),
    ),
    sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
    # Secondary index for macro run status lookups.
    sa.Index("idx_runs_status", "status"),
)

TRAJECTORIES_TABLE = sa.Table(
    "trajectories",
    METADATA,
    sa.Column("run_id", sa.String(128), nullable=False),
    sa.Column("trajectory_id", sa.String(128), nullable=False),
    sa.Column(
        "status",
        sa.String(32),
        nullable=False,
        default=Status.UNKNOWN,
        server_default=sa.text(f"'{Status.UNKNOWN}'"),
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
    ),
    sa.Column(
        "trajectory_metadata",
        _JSON_TYPE,
        nullable=False,
        default=dict,
        server_default=sa.text("'{}'"),
    ),
    sa.PrimaryKeyConstraint("run_id", "trajectory_id", name="pk_trajectories"),
    sa.ForeignKeyConstraint(
        ["run_id"],
        ["runs.run_id"],
        name="fk_trajectories_runs",
        ondelete="CASCADE",
    ),
    # Secondary index for timestamp range queries and recovery sweeps.
    sa.Index("idx_trajectories_updated_at", "run_id", "updated_at"),
    sa.Index("idx_trajectories_status", "run_id", "status"),
    sa.Index("idx_trajectories_created_at", "run_id", "created_at"),
)

# 16 (2^4) fixed hash buckets reduce per-partition row count and B-tree index
# size by 16x and allow parallel autovacuum across buckets while keeping
# unpruned operations (`drop_all`, `pg_dump`) at 2 * 16 + 1 = 33 relation locks,
# safely below PostgreSQL's default `max_locks_per_transaction = 64`.
STEPS_HASH_PARTITIONS: Final[int] = 16

STEPS_TABLE = sa.Table(
    "steps",
    METADATA,
    sa.Column("run_id", sa.String(128), nullable=False),
    sa.Column("trajectory_id", sa.String(128), nullable=False),
    sa.Column("step_id", sa.Integer(), nullable=False),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("payload", _JSON_TYPE, nullable=False),
    sa.PrimaryKeyConstraint(
        "run_id", "trajectory_id", "step_id", name="pk_steps"
    ),
    sa.ForeignKeyConstraint(
        ["run_id", "trajectory_id"],
        ["trajectories.run_id", "trajectories.trajectory_id"],
        name="fk_steps_trajectories",
        ondelete="CASCADE",
    ),
    postgresql_partition_by="HASH (run_id)",
)


def _register_steps_hash_partitions() -> None:
  """Registers PostgreSQL `after_create` DDL hooks for `steps` hash buckets."""
  for bucket_idx in range(STEPS_HASH_PARTITIONS):
    sa.event.listen(
        STEPS_TABLE,
        "after_create",
        sa.DDL(
            f"CREATE TABLE IF NOT EXISTS steps_p{bucket_idx:02d} "
            "PARTITION OF steps "
            f"FOR VALUES WITH (MODULUS {STEPS_HASH_PARTITIONS}, "
            f"REMAINDER {bucket_idx})"
        ).execute_if(dialect="postgresql"),
    )


_register_steps_hash_partitions()
