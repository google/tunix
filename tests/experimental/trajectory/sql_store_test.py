"""Unit tests for the SQL-backed trajectory store writer."""

from concurrent import futures
import os
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import schema_testing
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


class SqlStoreWriterTest(parameterized.TestCase):
  """Tests for the SQL behavior of _AsyncSqlWriter."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
    self.addCleanup(self.engine.dispose)
    schema.METADATA.create_all(self.engine)
    self.executed_sql: list[str] = []
    sa.event.listen(
        self.engine,
        "before_cursor_execute",
        self._record_sql_statement,
    )

  def _record_sql_statement(
      self,
      conn: Any,
      cursor: Any,
      statement: str,
      parameters: Any,
      context: Any,
      executemany: bool,
  ) -> None:
    del conn, cursor, parameters, context, executemany
    self.executed_sql.append(statement)

  def _count_sql(self, prefix: str) -> int:
    """Returns the number of executed SQL statements starting with `prefix`."""
    return sum(stmt.startswith(prefix) for stmt in self.executed_sql)

  def _create_writer(
      self,
      max_cached_trajectories: int = sql_store._MAX_CACHED_TRAJECTORIES,
  ) -> sql_store._AsyncSqlWriter:
    """Returns a writer bound to the test engine, closed on teardown."""
    writer = sql_store._AsyncSqlWriter(
        engine=self.engine,
        max_cached_trajectories=max_cached_trajectories,
    )
    self.addCleanup(writer.close)
    return writer

  def _fetch_runs(self) -> list[dict[str, Any]]:
    """Returns every row in the runs table."""
    return schema_testing.fetch_all(self.engine, sa.select(schema.RUNS_TABLE))

  def _fetch_trajectories(self) -> list[dict[str, Any]]:
    """Returns every row in the trajectories table."""
    return schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )

  def _fetch_steps(self) -> list[dict[str, Any]]:
    """Returns every row in the steps table, ordered by step id."""
    return schema_testing.fetch_all(
        self.engine,
        sa.select(schema.STEPS_TABLE).order_by(schema.STEPS_TABLE.c.step_id),
    )

  def test_init_with_postgresql_engine_binds_postgresql_insert(self) -> None:
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = sql_store.POSTGRESQL_DIALECT

    writer = sql_store._AsyncSqlWriter(engine=mock_engine)

    self.assertIs(writer._insert_fn, sql_store.postgresql.insert)

  def test_init_with_sqlite_engine_binds_sqlite_insert(self) -> None:
    writer = self._create_writer()

    self.assertIs(writer._insert_fn, sql_store.sqlite.insert)

  def test_init_with_unsupported_dialect_raises_value_error(self) -> None:
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = "oracle"

    with self.assertRaisesRegex(ValueError, r"Unsupported database dialect"):
      sql_store._AsyncSqlWriter(engine=mock_engine)

  def test_init_with_negative_max_cached_trajectories_raises_value_error(
      self,
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, r"max_cached_trajectories must be non-negative"
    ):
      sql_store._AsyncSqlWriter(engine=self.engine, max_cached_trajectories=-1)

  def test_resolve_status_with_stated_status_returns_schema_status(
      self,
  ) -> None:
    atif_metadata = trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()
    self.assertEqual(
        sql_store._resolve_status(atif_metadata),
        schema.Status.COMPLETED,
    )

  @parameterized.named_parameters(
      ("lowercase", "completed", schema.Status.COMPLETED),
      ("uppercase_with_whitespace", "  FAILED  ", schema.Status.FAILED),
      ("unrecognized_string", "not_a_status", schema.Status.UNKNOWN),
      ("non_string", 123, schema.Status.UNKNOWN),
  )
  def test_resolve_status_normalizes_case_and_ignores_invalid_values(
      self, raw_status: Any, expected: schema.Status
  ) -> None:
    metadata = trajectory_testing.TUNIX_METADATA_1.model_copy(
        update={"status": raw_status}
    ).to_atif_metadata()
    self.assertEqual(sql_store._resolve_status(metadata), expected)

  def test_resolve_status_with_final_metrics_returns_unknown(self) -> None:
    metadata = trajectory_testing.METADATA_1.model_copy(
        update={"final_metrics": trajectory_lib.FinalMetrics(total_steps=5)}
    )
    self.assertEqual(sql_store._resolve_status(metadata), schema.Status.UNKNOWN)

  def test_resolve_status_without_stated_status_returns_unknown(self) -> None:
    self.assertEqual(
        sql_store._resolve_status(trajectory_testing.METADATA_1),
        schema.Status.UNKNOWN,
    )

  @parameterized.named_parameters(
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_enqueue_write_with_blank_run_id_raises_value_error(
      self, blank_run_id: str
  ) -> None:
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(trajectory_id="traj_valid")

    with self.assertRaisesRegex(
        ValueError, r"run_id must be a non-empty string"
    ):
      writer.enqueue_write(run_id=blank_run_id, metadata=metadata)

  @parameterized.named_parameters(
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_enqueue_write_with_blank_trajectory_id_raises_value_error(
      self, blank_trajectory_id: str
  ) -> None:
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(
        trajectory_id=blank_trajectory_id
    )

    with self.assertRaisesRegex(
        ValueError, r"trajectory_id must be a non-empty string"
    ):
      writer.enqueue_write(run_id="run_valid", metadata=metadata)

  def test_enqueue_write_with_none_step_id_raises_value_error(self) -> None:
    writer = self._create_writer()
    step_without_id = trajectory_testing.STEP_1_1.model_copy(
        update={"step_id": None}
    )

    with self.assertRaisesRegex(
        ValueError, r"Step must have a non-empty step_id"
    ):
      writer.enqueue_write(
          run_id="run_valid",
          metadata=trajectory_testing.METADATA_1,
          step=step_without_id,
      )

  def test_enqueue_write_with_step_persists_run_trajectory_and_step_rows(
      self,
  ) -> None:
    writer = self._create_writer()

    writer.enqueue_write(
        run_id="run_sync",
        metadata=trajectory_testing.METADATA_1,
        step=trajectory_testing.STEP_1_1,
    )
    writer.flush()

    runs = self._fetch_runs()
    self.assertLen(runs, 1)
    self.assertEqual(runs[0]["run_id"], "run_sync")

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(
        trajectories[0]["trajectory_id"], trajectory_testing.TRAJECTORY_ID_1
    )
    self.assertEqual(trajectories[0]["status"], schema.Status.UNKNOWN)

    steps = self._fetch_steps()
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["step_id"], trajectory_testing.STEP_1_1.step_id)
    self.assertEqual(
        steps[0]["payload"]["message"], trajectory_testing.STEP_1_1.message
    )

  def test_enqueue_write_without_step_persists_trajectory_and_no_step_row(
      self,
  ) -> None:
    writer = self._create_writer()

    writer.enqueue_write(
        run_id="meta_only_run",
        metadata=trajectory_testing.METADATA_1,
        step=None,
    )
    writer.flush()

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(
        trajectories[0]["trajectory_id"], trajectory_testing.TRAJECTORY_ID_1
    )
    self.assertEqual(trajectories[0]["status"], schema.Status.UNKNOWN)
    self.assertEmpty(self._fetch_steps())

  def test_enqueue_write_populates_run_row_columns(self) -> None:
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(
        trajectory_id="traj_run_cols",
        agent=trajectory_lib.Agent(name="agent_named", version="1.0"),
    )

    writer.enqueue_write(run_id="run_cols", metadata=metadata)
    writer.flush()

    runs = self._fetch_runs()
    self.assertLen(runs, 1)
    self.assertEqual(runs[0]["agent_name"], "agent_named")
    self.assertEqual(runs[0]["status"], schema.Status.PENDING)
    self.assertEqual(runs[0]["config"], {})

  def test_enqueue_write_ignores_session_id_when_persisting_run_id(
      self,
  ) -> None:
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(
        trajectory_id="traj_verbatim", session_id="ignored_session_run"
    )

    writer.enqueue_write(run_id="explicit_run", metadata=metadata)
    writer.flush()

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["run_id"], "explicit_run")

  def test_enqueue_write_with_multiple_steps_persists_one_row_per_step(
      self,
  ) -> None:
    writer = self._create_writer()
    steps = [
        trajectory_testing.STEP_2_1,
        trajectory_testing.STEP_2_2,
        trajectory_testing.STEP_2_3,
        trajectory_testing.STEP_2_4,
        trajectory_testing.STEP_2_5,
    ]

    for step in steps:
      writer.enqueue_write(
          run_id="run_seq", metadata=trajectory_testing.METADATA_2, step=step
      )
    writer.flush()

    saved_steps = self._fetch_steps()
    self.assertLen(saved_steps, len(steps))
    for saved_step, step in zip(saved_steps, steps):
      self.assertEqual(saved_step["step_id"], step.step_id)
      self.assertEqual(saved_step["payload"]["message"], step.message)

  def test_enqueue_write_for_run_registered_elsewhere_preserves_stored_row(
      self,
  ) -> None:
    with self.engine.begin() as conn:
      conn.execute(
          schema.RUNS_TABLE.insert().values(
              run_id="shared_run",
              agent_name="registered_by_another_worker",
              status=schema.Status.RUNNING,
          )
      )
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(
        trajectory_id="traj_shared",
        agent=trajectory_lib.Agent(name="late_worker", version="1.0"),
    )

    writer.enqueue_write(run_id="shared_run", metadata=metadata)
    writer.flush()

    runs = self._fetch_runs()
    self.assertLen(runs, 1)
    self.assertEqual(runs[0]["agent_name"], "registered_by_another_worker")
    self.assertEqual(runs[0]["status"], schema.Status.RUNNING)

  def test_enqueue_write_for_existing_trajectory_overwrites_metadata(
      self,
  ) -> None:
    writer = self._create_writer()
    original = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_up",
        agent=trajectory_testing.METADATA_1.agent,
        status="PENDING",
    )
    updated = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_up",
        agent=trajectory_testing.METADATA_1.agent,
        status="COMPLETED",
    )

    writer.enqueue_write(run_id="conflict_run", metadata=original)
    writer.flush()
    writer.enqueue_write(run_id="conflict_run", metadata=updated)
    writer.flush()

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["status"], "COMPLETED")

  def test_enqueue_write_for_existing_trajectory_preserves_created_at(
      self,
  ) -> None:
    writer = self._create_writer()
    metadata = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_created",
        agent=trajectory_testing.METADATA_1.agent,
        notes="first",
    )

    writer.enqueue_write(run_id="created_run", metadata=metadata)
    writer.flush()
    created_at = self._fetch_trajectories()[0]["created_at"]

    writer.enqueue_write(
        run_id="created_run",
        metadata=metadata.model_copy(update={"notes": "second"}),
    )
    writer.flush()

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["created_at"], created_at)
    self.assertEqual(trajectories[0]["trajectory_metadata"]["notes"], "second")

  @parameterized.named_parameters(
      ("metadata_only", None),
      ("with_step", trajectory_testing.STEP_1_1),
  )
  def test_enqueue_write_without_stated_status_preserves_stored_status(
      self, step: trajectory_lib.Step | None
  ) -> None:
    writer = self._create_writer()
    writer.enqueue_write(
        run_id="preserve_run",
        metadata=trajectory_lib.TunixTrajectoryMetadata(
            trajectory_id="traj_preserve",
            agent=trajectory_testing.METADATA_1.agent,
            status="COMPLETED",
        ),
    )
    writer.flush()

    writer.enqueue_write(
        run_id="preserve_run",
        metadata=trajectory_lib.TrajectoryMetadata(
            trajectory_id="traj_preserve",
            agent=trajectory_testing.METADATA_1.agent,
            notes="new notes",
        ),
        step=step,
    )
    writer.flush()

    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["status"], "COMPLETED")
    self.assertEqual(
        trajectories[0]["trajectory_metadata"]["notes"], "new notes"
    )

  def test_enqueue_write_for_existing_step_replaces_stored_payload(
      self,
  ) -> None:
    writer = self._create_writer()
    metadata = trajectory_testing.make_metadata(trajectory_id="traj_revise")
    original = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="original"
    )
    revised = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="revised"
    )

    writer.enqueue_write(run_id="revise_run", metadata=metadata, step=original)
    writer.flush()
    writer.enqueue_write(run_id="revise_run", metadata=metadata, step=revised)
    writer.flush()

    steps = self._fetch_steps()
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["payload"]["message"], "revised")

  def test_enqueue_write_for_known_run_skips_repeat_run_insert(self) -> None:
    writer = self._create_writer()
    first = trajectory_testing.make_metadata(trajectory_id="traj_1")
    second = trajectory_testing.make_metadata(trajectory_id="traj_2")

    writer.enqueue_write(run_id="cached_run", metadata=first)
    writer.enqueue_write(run_id="cached_run", metadata=second)
    writer.flush()

    self.assertEqual(self._count_sql("INSERT INTO runs"), 1)
    self.assertLen(self._fetch_runs(), 1)

  def test_enqueue_write_with_unchanged_metadata_skips_repeat_trajectory_upsert(
      self,
  ) -> None:
    writer = self._create_writer()
    running_metadata = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id=trajectory_testing.TRAJECTORY_ID_2,
        agent=trajectory_testing.METADATA_2.agent,
        status="RUNNING",
    )
    completed_metadata = running_metadata.model_copy(
        update={"status": "COMPLETED"}
    )

    writer.enqueue_write(
        run_id="run_cached_meta",
        metadata=running_metadata,
        step=trajectory_testing.STEP_2_1,
    )
    writer.enqueue_write(
        run_id="run_cached_meta",
        metadata=running_metadata,
        step=trajectory_testing.STEP_2_2,
    )
    writer.enqueue_write(
        run_id="run_cached_meta",
        metadata=completed_metadata,
        step=trajectory_testing.STEP_2_3,
    )
    writer.flush()

    # Step 1 inserts the trajectory, Step 2 skips it, and Step 3 updates it.
    self.assertEqual(self._count_sql("INSERT INTO trajectories"), 2)
    self.assertEqual(self._count_sql("INSERT INTO steps"), 3)
    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["status"], "COMPLETED")

  def test_enqueue_write_after_step_failure_retries_trajectory_upsert_on_next_step(
      self,
  ) -> None:
    writer = self._create_writer()
    original_upsert_step = writer._upsert_step
    call_count = 0

    def fail_on_first_step(*args: Any, **kwargs: Any) -> None:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise sa.exc.OperationalError("simulated deadlock", None, Exception())
      original_upsert_step(*args, **kwargs)

    with mock.patch.object(
        writer, "_upsert_step", side_effect=fail_on_first_step
    ):
      writer.enqueue_write(
          run_id="run_rollback",
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_1,
      )
      writer.enqueue_write(
          run_id="run_rollback",
          metadata=trajectory_testing.METADATA_2,
          step=trajectory_testing.STEP_2_2,
      )
      writer.flush()

    self.assertEqual(self._count_sql("INSERT INTO runs"), 2)
    self.assertEqual(self._count_sql("INSERT INTO trajectories"), 2)
    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(
        trajectories[0]["trajectory_id"], trajectory_testing.TRAJECTORY_ID_2
    )
    steps = self._fetch_steps()
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["step_id"], trajectory_testing.STEP_2_2.step_id)

  def test_enqueue_write_evicts_least_recently_used_chunk_and_retains_run_id(
      self,
  ) -> None:
    writer = self._create_writer(max_cached_trajectories=4)
    meta_1 = trajectory_testing.make_metadata(trajectory_id="traj_lru_1")
    meta_2 = trajectory_testing.make_metadata(trajectory_id="traj_lru_2")
    meta_3 = trajectory_testing.make_metadata(trajectory_id="traj_lru_3")
    meta_4 = trajectory_testing.make_metadata(trajectory_id="traj_lru_4")
    meta_5 = trajectory_testing.make_metadata(trajectory_id="traj_lru_5")

    writer.enqueue_write(run_id="run_lru", metadata=meta_1)
    writer.enqueue_write(run_id="run_lru", metadata=meta_2)
    writer.enqueue_write(run_id="run_lru", metadata=meta_3)
    writer.enqueue_write(run_id="run_lru", metadata=meta_4)
    # Touch traj_lru_1 and traj_lru_2 so traj_lru_3 and traj_lru_4 become the
    # oldest 50% LRU chunk.
    writer.enqueue_write(run_id="run_lru", metadata=meta_1)
    writer.enqueue_write(run_id="run_lru", metadata=meta_2)
    # Writing a 5th trajectory triggers 50% chunk eviction (traj_lru_3 and
    # traj_lru_4 are evicted; traj_lru_1, traj_lru_2, and traj_lru_5 remain).
    writer.enqueue_write(run_id="run_lru", metadata=meta_5)
    # Touch retained traj_lru_1 (cache hit -> no new SQL upsert) and evicted
    # traj_lru_3 (cache miss -> 1 new SQL upsert).
    writer.enqueue_write(run_id="run_lru", metadata=meta_1)
    writer.enqueue_write(run_id="run_lru", metadata=meta_3)
    writer.flush()

    self.assertEqual(
        list(writer._trajectories_by_run["run_lru"].keys()),
        ["traj_lru_2", "traj_lru_5", "traj_lru_1", "traj_lru_3"],
    )
    self.assertEqual(self._count_sql("INSERT INTO runs"), 1)
    # 5 initial trajectory inserts + 1 re-upsert for evicted traj_lru_3 = 6.
    self.assertEqual(self._count_sql("INSERT INTO trajectories"), 6)

  def test_enqueue_write_with_disabled_client_cache_preserves_updated_at_when_unchanged(
      self,
  ) -> None:
    writer = self._create_writer(max_cached_trajectories=0)
    metadata = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_sql_where",
        agent=trajectory_testing.METADATA_1.agent,
        status="RUNNING",
    )

    writer.enqueue_write(
        run_id="run_sql_where",
        metadata=metadata,
        step=trajectory_testing.STEP_1_1,
    )
    writer.flush()
    initial_updated_at = self._fetch_trajectories()[0]["updated_at"]

    writer.enqueue_write(
        run_id="run_sql_where",
        metadata=metadata,
        step=trajectory_testing.STEP_2_2,
    )
    writer.flush()

    self.assertEqual(self._count_sql("INSERT INTO runs"), 1)
    self.assertEqual(self._count_sql("INSERT INTO trajectories"), 2)
    trajectories = self._fetch_trajectories()
    self.assertLen(trajectories, 1)
    self.assertEqual(trajectories[0]["updated_at"], initial_updated_at)
    self.assertLen(self._fetch_steps(), 2)


class _SqlTestReader(store.TrajectoryReader):
  """Minimal reader used to verify SqlTrajectoryStore writes against contract tests."""

  def __init__(self, engine: sa.Engine, run_id: str) -> None:
    self._engine = engine
    self._run_id = run_id

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    query = sa.select(
        schema.TRAJECTORIES_TABLE.c.trajectory_id,
        schema.TRAJECTORIES_TABLE.c.trajectory_metadata,
    ).where(schema.TRAJECTORIES_TABLE.c.run_id == self._run_id)
    rows = schema_testing.fetch_all(self._engine, query)
    metadata_by_id = {
        row["trajectory_id"]: trajectory_lib.TrajectoryMetadata.model_validate(
            row["trajectory_metadata"]
        )
        for row in rows
    }
    if trajectory_ids is None:
      return list(metadata_by_id.values())
    result = []
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metadata_by_id:
        raise store.TrajectoryMetadataNotFoundError(trajectory_id)
      result.append(metadata_by_id[trajectory_id])
    return result

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory]:
    metadata_by_id = {
        meta.trajectory_id: meta for meta in self.get_trajectories_metadata()
    }
    step_rows = schema_testing.fetch_all(
        self._engine,
        sa.select(
            schema.STEPS_TABLE.c.trajectory_id,
            schema.STEPS_TABLE.c.payload,
        )
        .where(schema.STEPS_TABLE.c.run_id == self._run_id)
        .order_by(schema.STEPS_TABLE.c.step_id),
    )
    steps_by_trajectory_id: dict[str, list[trajectory_lib.Step]] = {}
    for row in step_rows:
      steps_by_trajectory_id.setdefault(row["trajectory_id"], []).append(
          trajectory_lib.Step.model_validate(row["payload"])
      )
    result = []
    for trajectory_id in trajectory_ids:
      if trajectory_id not in metadata_by_id:
        raise store.TrajectoryNotFoundError(trajectory_id)
      meta = metadata_by_id[trajectory_id]
      steps = steps_by_trajectory_id.get(trajectory_id, [])
      result.append(trajectory_lib.Trajectory(**meta.model_dump(), steps=steps))
    return result


class SqlTrajectoryWriterTest(store_testing.TrajectoryWriterTestCase):
  """Contract tests for SqlTrajectoryStore's TrajectoryWriter implementation."""

  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
    self.addCleanup(engine.dispose)
    run_id = "test_contract_run"
    writer = sql_store.SqlTrajectoryStore(engine=engine, run_id=run_id)
    self.addCleanup(writer.close)
    reader = _SqlTestReader(engine=engine, run_id=run_id)
    return reader, writer


class SqlTrajectoryStoreTest(parameterized.TestCase):
  """Unit tests for SqlTrajectoryStore initialization and lifecycle behavior."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
    self.addCleanup(self.engine.dispose)

  def _create_store(
      self, run_id: str = "default_run"
  ) -> sql_store.SqlTrajectoryStore:
    store_inst = sql_store.SqlTrajectoryStore(engine=self.engine, run_id=run_id)
    self.addCleanup(store_inst.close)
    return store_inst

  @parameterized.named_parameters(
      ("none", None),
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_init_with_missing_or_blank_run_id_raises_value_error(
      self, invalid_run_id: Any
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, r"SqlTrajectoryStore requires a non-empty run_id"
    ):
      sql_store.SqlTrajectoryStore(engine=self.engine, run_id=invalid_run_id)

  def test_init_sets_engine_and_run_id(self) -> None:
    store_inst = sql_store.SqlTrajectoryStore(
        engine=self.engine, run_id="  my_run  "
    )
    self.addCleanup(store_inst.close)

    self.assertEqual(store_inst.run_id, "my_run")
    self.assertIs(store_inst.engine, self.engine)

  def test_init_with_auto_init_false_does_not_create_tables(self) -> None:
    uninitialized_engine = schema_testing.create_sqlite_memory_engine(
        shared_pool=True
    )
    self.addCleanup(uninitialized_engine.dispose)
    store_inst = sql_store.SqlTrajectoryStore(
        engine=uninitialized_engine, run_id="no_init_run", auto_init=False
    )
    self.addCleanup(store_inst.close)

    inspector = sa.inspect(uninitialized_engine)
    self.assertEmpty(inspector.get_table_names())

  def test_init_on_postgresql_acquires_advisory_lock_when_cold(
      self,
  ) -> None:
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = sql_store.POSTGRESQL_DIALECT
    mock_read_conn = mock.MagicMock()
    mock_write_conn = mock.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_read_conn
    mock_engine.begin.return_value.__enter__.return_value = mock_write_conn
    mock_inspector = mock.MagicMock()
    mock_inspector.get_table_names.return_value = []

    with (
        mock.patch.object(
            sa, "inspect", return_value=mock_inspector
        ) as mock_inspect,
        mock.patch.object(schema.METADATA, "create_all") as mock_create_all,
    ):
      with sql_store.SqlTrajectoryStore(
          engine=mock_engine, run_id="test_run", auto_init=True
      ):
        pass

    mock_inspect.assert_called_once_with(mock_read_conn)
    mock_write_conn.execute.assert_called_once()
    executed_stmt = str(
        mock_write_conn.execute.call_args.args[0].compile(
            compile_kwargs={"literal_binds": True}
        )
    )
    self.assertIn(
        f"pg_advisory_xact_lock({sql_store._POSTGRES_SCHEMA_INIT_LOCK_ID})",
        executed_stmt,
    )
    mock_create_all.assert_called_once_with(mock_write_conn, checkfirst=True)

  def test_init_when_tables_exist_skips_advisory_lock_and_ddl(
      self,
  ) -> None:
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = sql_store.POSTGRESQL_DIALECT
    mock_read_conn = mock.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_read_conn
    mock_inspector = mock.MagicMock()
    mock_inspector.get_table_names.return_value = list(
        schema.METADATA.tables.keys()
    )

    with (
        mock.patch.object(sa, "inspect", return_value=mock_inspector),
        mock.patch.object(schema.METADATA, "create_all") as mock_create_all,
    ):
      with sql_store.SqlTrajectoryStore(
          engine=mock_engine, run_id="test_run", auto_init=True
      ):
        pass

    mock_engine.begin.assert_not_called()
    mock_create_all.assert_not_called()

  def test_init_concurrent_workers_with_auto_init_true_succeeds(self) -> None:
    num_workers = 32
    barrier = threading.Barrier(num_workers)
    db_path = os.path.join(
        self.create_tempdir().full_path, "concurrent_init.db"
    )

    def _init_worker(worker_idx: int) -> None:
      worker_engine = schema_testing.create_sqlite_file_engine(db_path)
      try:
        barrier.wait(timeout=5.0)
        with sql_store.SqlTrajectoryStore(
            engine=worker_engine,
            run_id=f"concurrent_run_{worker_idx}",
            auto_init=True,
        ) as worker_store:
          worker_store.add_step(
              trajectory_testing.STEP_1_1,
              trajectory_testing.make_metadata(
                  trajectory_id=f"traj_{worker_idx}"
              ),
          )
      finally:
        worker_engine.dispose()

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
      worker_futures = [
          executor.submit(_init_worker, idx) for idx in range(num_workers)
      ]
      for future in futures.as_completed(worker_futures):
        future.result()

    verify_engine = schema_testing.create_sqlite_file_engine(db_path)
    self.addCleanup(verify_engine.dispose)
    inspector = sa.inspect(verify_engine)
    self.assertContainsSubset(
        schema.METADATA.tables.keys(),
        set(inspector.get_table_names()),
    )
    for table in (
        schema.RUNS_TABLE,
        schema.TRAJECTORIES_TABLE,
        schema.STEPS_TABLE,
    ):
      self.assertLen(
          schema_testing.fetch_all(verify_engine, sa.select(table)),
          num_workers,
      )

  @parameterized.named_parameters(
      ("slash", "traj/001"),
      ("colons_uuid", "urn:uuid:1234-5678"),
      ("dots", "sim.attempt.001"),
  )
  def test_valid_arbitrary_trajectory_id_succeeds(self, traj_id: str) -> None:
    store_inst = self._create_store(run_id="arb_run")
    meta = trajectory_testing.make_metadata(trajectory_id=traj_id)

    store_inst.add_step(trajectory_testing.STEP_1_1, meta)
    store_inst.flush()

    reader = _SqlTestReader(engine=self.engine, run_id="arb_run")
    trajs = reader.get_trajectories([traj_id])
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0].trajectory_id, traj_id)

  def test_write_to_closed_store_raises_runtime_error(self) -> None:
    store_inst = self._create_store()
    store_inst.close()

    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      store_inst.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

    with self.assertRaisesRegex(RuntimeError, r"Cannot write to a closed"):
      store_inst.update_metadata(trajectory_testing.METADATA_1)

  def test_context_manager_lifecycle_drains_writes_on_exit(self) -> None:
    with sql_store.SqlTrajectoryStore(
        engine=self.engine, run_id="ctx_run"
    ) as store_ctx:
      store_ctx.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

    ctx_reader = _SqlTestReader(engine=self.engine, run_id="ctx_run")
    trajs = ctx_reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])

  def test_file_based_sqlite_persistence(self) -> None:
    db_path = os.path.join(self.create_tempdir().full_path, "test.db")
    engine = schema_testing.create_sqlite_file_engine(db_path)
    self.addCleanup(engine.dispose)

    with sql_store.SqlTrajectoryStore(
        engine=engine, run_id="persisted_run"
    ) as store_ctx:
      store_ctx.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

    read_engine = schema_testing.create_sqlite_file_engine(db_path)
    self.addCleanup(read_engine.dispose)
    read_reader = _SqlTestReader(engine=read_engine, run_id="persisted_run")
    trajs = read_reader.get_trajectories([trajectory_testing.TRAJECTORY_ID_1])
    self.assertEqual(trajs, [trajectory_testing.TRAJECTORY_1])


if __name__ == "__main__":
  absltest.main()
