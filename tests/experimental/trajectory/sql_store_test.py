"""Unit tests for the SQL-backed trajectory store writer."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import schema_testing
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


class SqlStoreWriterTest(parameterized.TestCase):
  """Tests for the SQL behavior of _AsyncSqlWriter."""

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)
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
    mock_engine.dialect.name = "postgresql"

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

  def test_enqueue_write_for_known_run_skips_repeat_run_insert(self) -> None:
    writer = self._create_writer()
    first = trajectory_testing.make_metadata(trajectory_id="traj_1")
    second = trajectory_testing.make_metadata(trajectory_id="traj_2")

    writer.enqueue_write(run_id="cached_run", metadata=first)
    writer.enqueue_write(run_id="cached_run", metadata=second)
    writer.flush()

    self.assertEqual(self._count_sql("INSERT INTO runs"), 1)
    self.assertLen(self._fetch_runs(), 1)

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


if __name__ == "__main__":
  absltest.main()
