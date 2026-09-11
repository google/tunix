"""Unit tests for SqlTrajectoryStore (Writer implementation)."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import sqlalchemy as sa
from tunix.experimental.trajectory import base_writer
from tunix.experimental.trajectory import schema
from tunix.experimental.trajectory import schema_testing
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib


class SqlTrajectoryWriterTest(parameterized.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.engine = schema_testing.create_sqlite_memory_engine(shared_pool=True)

  def _create_store(
      self, run_id: str = "default_run"
  ) -> sql_store.SqlTrajectoryStore:
    store_inst = sql_store.SqlTrajectoryStore(engine=self.engine, run_id=run_id)
    self.addCleanup(store_inst.close)
    return store_inst

  def test_implements_trajectory_writer_and_composes_async_sql_writer(
      self,
  ) -> None:
    store_inst = self._create_store(run_id="inherit_run")
    self.assertIsInstance(store_inst, store.TrajectoryWriter)
    self.assertIsInstance(store_inst._writer, base_writer.BaseAsyncWriter)

  def test_add_step_and_flush_persists_records(self) -> None:
    store_inst = self._create_store(run_id="run_alpha")
    meta = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_001",
        prompt_id="prompt_42",
        agent=trajectory_lib.Agent(name="agent_x", version="1.0"),
        status="RUNNING",
    )
    step0 = trajectory_lib.TunixEnvStep(
        step_id=0,
        source=trajectory_lib.Source.USER,
        message="user prompt",
        reward=1.0,
    )
    step1 = trajectory_lib.TunixAgentStep(
        step_id=1,
        source=trajectory_lib.Source.AGENT,
        message="agent reply",
        mc_return=0.85,
    )

    store_inst.add_step(step0, meta)
    store_inst.add_step(step1, meta)
    store_inst.flush()

    runs = schema_testing.fetch_all(self.engine, sa.select(schema.RUNS_TABLE))
    self.assertLen(runs, 1)
    self.assertEqual(runs[0]["run_id"], "run_alpha")
    self.assertEqual(runs[0]["status"], "PENDING")

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["trajectory_id"], "traj_001")
    self.assertEqual(trajs[0]["status"], "RUNNING")
    self.assertEqual(
        trajs[0]["trajectory_metadata"]["_metadata_type"],
        "TunixTrajectoryMetadata",
    )
    self.assertEqual(trajs[0]["trajectory_metadata"]["prompt_id"], "prompt_42")

    steps = schema_testing.fetch_all(
        self.engine,
        sa.select(schema.STEPS_TABLE).order_by(schema.STEPS_TABLE.c.step_id),
    )
    self.assertLen(steps, 2)
    self.assertEqual(steps[0]["step_id"], 0)
    self.assertEqual(steps[0]["payload"]["_step_type"], "TunixEnvStep")
    self.assertEqual(steps[0]["payload"]["reward"], 1.0)
    self.assertEqual(steps[1]["step_id"], 1)
    self.assertEqual(steps[1]["payload"]["_step_type"], "TunixAgentStep")
    self.assertEqual(steps[1]["payload"]["mc_return"], 0.85)

  def test_update_metadata_without_step(self) -> None:
    store_inst = self._create_store(run_id="meta_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_meta_only",
        agent=trajectory_lib.Agent(name="agent_m", version="1.0"),
    )

    store_inst.update_metadata(meta)
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["trajectory_id"], "traj_meta_only")
    self.assertEqual(trajs[0]["status"], "PENDING")

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertEmpty(steps)

  def test_upsert_existing_step_overwrites_payload(self) -> None:
    store_inst = self._create_store(run_id="upsert_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_up",
        agent=trajectory_lib.Agent(name="agent_u", version="1.0"),
    )
    step_v1 = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.AGENT, message="original"
    )
    step_v2 = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.AGENT, message="updated"
    )

    store_inst.add_step(step_v1, meta)
    store_inst.flush()
    store_inst.add_step(step_v2, meta)
    store_inst.flush()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["payload"]["message"], "updated")

  def test_flush_empty(self) -> None:
    store_inst = self._create_store()
    store_inst.flush()
    runs = schema_testing.fetch_all(self.engine, sa.select(schema.RUNS_TABLE))
    self.assertEmpty(runs)
    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertEmpty(trajs)

  def test_flush_idempotent(self) -> None:
    store_inst = self._create_store(run_id="run_idem")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_idem",
        agent=trajectory_lib.Agent(name="agent_idem", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )
    store_inst.add_step(step, meta)
    store_inst.flush()
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)

  def test_close_is_idempotent(self) -> None:
    store_inst = self._create_store(run_id="run_close_idem")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_close_idem",
        agent=trajectory_lib.Agent(name="agent_ci", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )
    store_inst.add_step(step, meta)
    store_inst.close()
    store_inst.close()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)

  def test_add_step_snapshots_step_and_metadata(self) -> None:
    store_inst = self._create_store(run_id="run_snap")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_snap",
        agent=trajectory_lib.Agent(name="agent_snap", version="1.0"),
        notes="original note",
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="original msg"
    )
    store_inst.add_step(step, meta)

    # Mutate objects in-place after scheduling write.
    step.message = "mutated msg"
    meta.notes = "mutated note"
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertEqual(trajs[0]["trajectory_metadata"]["notes"], "original note")

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertEqual(steps[0]["payload"]["message"], "original msg")

  def test_update_metadata_snapshots_metadata(self) -> None:
    store_inst = self._create_store(run_id="run_meta_snap")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_meta_snap",
        agent=trajectory_lib.Agent(name="agent_snap", version="1.0"),
        notes="original note",
    )
    store_inst.update_metadata(meta)

    # Mutate metadata in-place after scheduling write.
    meta.notes = "mutated note"
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertEqual(trajs[0]["trajectory_metadata"]["notes"], "original note")

  def test_add_step_multiple_trajectories(self) -> None:
    store_inst = self._create_store(run_id="run_multi")
    meta1 = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_1",
        agent=trajectory_lib.Agent(name="agent_1", version="1.0"),
    )
    meta2 = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_2",
        agent=trajectory_lib.Agent(name="agent_2", version="1.0"),
    )
    step1_0 = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="traj1 step0"
    )
    step2_0 = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="traj2 step0"
    )
    step1_1 = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.AGENT, message="traj1 step1"
    )

    store_inst.add_step(step1_0, meta1)
    store_inst.add_step(step2_0, meta2)
    store_inst.add_step(step1_1, meta1)
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine,
        sa.select(schema.TRAJECTORIES_TABLE).order_by(
            schema.TRAJECTORIES_TABLE.c.trajectory_id
        ),
    )
    self.assertLen(trajs, 2)
    self.assertEqual(trajs[0]["trajectory_id"], "traj_1")
    self.assertEqual(trajs[1]["trajectory_id"], "traj_2")

    steps = schema_testing.fetch_all(
        self.engine,
        sa.select(schema.STEPS_TABLE).order_by(
            schema.STEPS_TABLE.c.trajectory_id, schema.STEPS_TABLE.c.step_id
        ),
    )
    self.assertLen(steps, 3)
    self.assertEqual(steps[0]["trajectory_id"], "traj_1")
    self.assertEqual(steps[0]["step_id"], 0)
    self.assertEqual(steps[0]["payload"]["message"], "traj1 step0")

    self.assertEqual(steps[1]["trajectory_id"], "traj_1")
    self.assertEqual(steps[1]["step_id"], 1)
    self.assertEqual(steps[1]["payload"]["message"], "traj1 step1")

    self.assertEqual(steps[2]["trajectory_id"], "traj_2")
    self.assertEqual(steps[2]["step_id"], 0)
    self.assertEqual(steps[2]["payload"]["message"], "traj2 step0")

  @parameterized.named_parameters(
      ("empty", ""),
      ("whitespace", "   "),
  )
  def test_invalid_trajectory_id_raises_value_error(self, traj_id: str) -> None:
    store_inst = self._create_store(run_id="val_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=traj_id,
        agent=trajectory_lib.Agent(name="agent_v", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )

    with self.assertRaisesRegex(
        ValueError, "trajectory_id must be a non-empty string"
    ):
      store_inst.add_step(step, meta)

    with self.assertRaisesRegex(
        ValueError, "trajectory_id must be a non-empty string"
    ):
      store_inst.update_metadata(meta)

  @parameterized.named_parameters(
      ("slash", "traj/001"),
      ("colons_uuid", "urn:uuid:1234-5678"),
      ("dots", "sim.attempt.001"),
  )
  def test_valid_arbitrary_trajectory_id_succeeds(self, traj_id: str) -> None:
    store_inst = self._create_store(run_id="arb_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id=traj_id,
        agent=trajectory_lib.Agent(name="agent_v", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )
    store_inst.add_step(step, meta)
    store_inst.flush()

    trajs = schema_testing.fetch_all(
        self.engine, sa.select(schema.TRAJECTORIES_TABLE)
    )
    self.assertLen(trajs, 1)
    self.assertEqual(trajs[0]["trajectory_id"], traj_id)

  def test_write_to_closed_store_raises_runtime_error(self) -> None:
    store_inst = self._create_store(run_id="close_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_closed",
        agent=trajectory_lib.Agent(name="agent_c", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )

    store_inst.close()

    with self.assertRaisesRegex(RuntimeError, "Cannot write to a closed"):
      store_inst.add_step(step, meta)

    with self.assertRaisesRegex(RuntimeError, "Cannot write to a closed"):
      store_inst.update_metadata(meta)

  def test_close_drains_unflushed_writes(self) -> None:
    store_inst = self._create_store(run_id="drain_run")
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_drain",
        agent=trajectory_lib.Agent(name="agent_d", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="drain msg"
    )

    store_inst.add_step(step, meta)
    store_inst.close()

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["payload"]["message"], "drain msg")

  def test_none_metadata_raises_value_error(self) -> None:
    store_inst = self._create_store()
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="msg"
    )
    with self.assertRaisesRegex(
        ValueError, r"TrajectoryMetadata cannot be None"
    ):
      store_inst.add_step(step, None)  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(
        ValueError, r"TrajectoryMetadata cannot be None"
    ):
      store_inst.update_metadata(None)  # pytype: disable=wrong-arg-types

  def test_run_id_resolution_and_whitespace_fallback(self) -> None:
    store_inst = self._create_store(run_id="default_run")

    # 1. Explicit session_id takes priority.
    meta_session = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_sess",
        session_id="custom_sess",
        agent=trajectory_lib.Agent(name="agent_a", version="1.0"),
    )
    self.assertEqual(store_inst.resolve_run_id(meta_session), "custom_sess")

    # 2. Metadata extra["run_id"] is ignored; falls back to store run_id.
    meta_extra = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_extra",
        agent=trajectory_lib.Agent(name="agent_b", version="1.0"),
        extra={"run_id": "extra_run"},
    )
    self.assertEqual(store_inst.resolve_run_id(meta_extra), "default_run")

    # 3. Whitespace session_id falls back to default.
    meta_ws_sess = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_ws_sess",
        session_id="   ",
        agent=trajectory_lib.Agent(name="agent_c", version="1.0"),
    )
    self.assertEqual(store_inst.resolve_run_id(meta_ws_sess), "default_run")

    # 4. Store initialized without run_id defaults to "default".
    store_no_run_id = sql_store.SqlTrajectoryStore(engine=self.engine)
    self.addCleanup(store_no_run_id.close)
    self.assertEqual(store_no_run_id.run_id, "default")
    self.assertEqual(store_no_run_id.resolve_run_id(meta_ws_sess), "default")

  def test_context_manager_lifecycle(self) -> None:
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_ctx",
        agent=trajectory_lib.Agent(name="agent_ctx", version="1.0"),
    )
    step = trajectory_lib.Step(
        step_id=0, source=trajectory_lib.Source.USER, message="ctx msg"
    )

    with sql_store.SqlTrajectoryStore(
        engine=self.engine, run_id="ctx_run"
    ) as store_ctx:
      store_ctx.add_step(step, meta)

    steps = schema_testing.fetch_all(self.engine, sa.select(schema.STEPS_TABLE))
    self.assertLen(steps, 1)
    self.assertEqual(steps[0]["payload"]["message"], "ctx msg")

  def test_file_based_sqlite_persistence(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      db_path = os.path.join(tmp_dir, "test.db")
      url = f"sqlite:///{db_path}"
      engine = sa.create_engine(url, connect_args={"check_same_thread": False})

      with sql_store.SqlTrajectoryStore(
          engine=engine, run_id="persisted_run"
      ) as store_ctx:
        meta = trajectory_lib.TrajectoryMetadata(
            trajectory_id="traj_persist",
            agent=trajectory_lib.Agent(name="agent_p", version="1.0"),
        )
        step = trajectory_lib.Step(
            step_id=1,
            source=trajectory_lib.Source.AGENT,
            message="persist step",
        )
        store_ctx.add_step(step, meta)

      read_engine = sa.create_engine(
          url, connect_args={"check_same_thread": False}
      )
      steps = schema_testing.fetch_all(
          read_engine, sa.select(schema.STEPS_TABLE)
      )
      self.assertLen(steps, 1)
      self.assertEqual(steps[0]["payload"]["message"], "persist step")

  def test_unsupported_dialect_raises_value_error(self) -> None:
    mock_engine = mock.MagicMock()
    mock_engine.dialect.name = "oracle"
    with self.assertRaisesRegex(ValueError, r"Unsupported database dialect"):
      sql_store.SqlTrajectoryStore(engine=mock_engine)


if __name__ == "__main__":
  absltest.main()
