"""Unit tests for SQL Trajectory Store serialization utilities."""

import datetime

from absl.testing import absltest
from tunix.experimental.trajectory import sql_serialization
from tunix.experimental.trajectory import trajectory as trajectory_lib


class SqlSerializationTest(absltest.TestCase):

  def test_trajectory_metadata_serialization_round_trip(self) -> None:
    meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_001",
        session_id="session_abc",
        agent=trajectory_lib.Agent(name="eval_agent", version="1.0"),
        notes="standard atif run",
    )

    payload = sql_serialization.serialize_metadata(meta)
    restored = sql_serialization.deserialize_metadata(payload)

    self.assertIsInstance(restored, trajectory_lib.TrajectoryMetadata)
    self.assertNotIsInstance(restored, trajectory_lib.TunixTrajectoryMetadata)
    self.assertEqual(restored.trajectory_id, "traj_001")
    self.assertEqual(restored.session_id, "session_abc")
    self.assertEqual(restored.agent.name, "eval_agent")
    self.assertEqual(restored.notes, "standard atif run")

  def test_tunix_trajectory_metadata_serialization_round_trip(self) -> None:
    meta = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="tunix_002",
        prompt_id="prompt_99",
        group_index=2,
        target_policy_versions=[1, 2, 3],
        status="RUNNING",
        agent=trajectory_lib.Agent(name="rl_agent", version="2.0"),
    )

    payload = sql_serialization.serialize_metadata(meta)
    restored = sql_serialization.deserialize_metadata(payload)

    self.assertIsInstance(restored, trajectory_lib.TunixTrajectoryMetadata)
    self.assertEqual(restored.trajectory_id, "tunix_002")
    self.assertEqual(restored.prompt_id, "prompt_99")
    self.assertEqual(restored.group_index, 2)
    self.assertEqual(restored.target_policy_versions, [1, 2, 3])
    self.assertEqual(restored.status, "RUNNING")
    self.assertEqual(restored.agent.name, "rl_agent")

  def test_serialize_step(self) -> None:
    dt = datetime.datetime(2026, 9, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
    step = trajectory_lib.Step(
        step_id=1,
        source=trajectory_lib.Source.USER,
        message="hello",
        timestamp=dt,
    )
    payload, ts = sql_serialization.serialize_step(step)

    self.assertEqual(ts, dt)
    self.assertEqual(payload["_step_type"], "Step")
    self.assertEqual(payload["step_id"], 1)
    self.assertEqual(payload["message"], "hello")

    step_without_ts = trajectory_lib.Step(
        step_id=2,
        source=trajectory_lib.Source.USER,
        message="no timestamp",
        timestamp=None,
    )
    _, fallback_ts = sql_serialization.serialize_step(step_without_ts)
    self.assertEqual(fallback_ts.tzinfo, datetime.timezone.utc)

  def test_deserialize_step_polymorphic_classes(self) -> None:
    base_step = trajectory_lib.Step(
        step_id=1, source=trajectory_lib.Source.USER, message="user msg"
    )
    env_step = trajectory_lib.TunixEnvStep(
        step_id=2,
        source=trajectory_lib.Source.USER,
        message="env msg",
        reward=1.0,
        done=False,
    )
    agent_step = trajectory_lib.TunixAgentStep(
        step_id=3,
        source=trajectory_lib.Source.AGENT,
        message="agent reply",
        mc_return=0.9,
    )

    payload_base, _ = sql_serialization.serialize_step(base_step)
    payload_env, _ = sql_serialization.serialize_step(env_step)
    payload_agent, _ = sql_serialization.serialize_step(agent_step)

    restored_base = sql_serialization.deserialize_step(payload_base)
    restored_env = sql_serialization.deserialize_step(payload_env)
    restored_agent = sql_serialization.deserialize_step(payload_agent)

    self.assertIsInstance(restored_base, trajectory_lib.Step)
    self.assertIsInstance(restored_env, trajectory_lib.TunixEnvStep)
    self.assertIsInstance(restored_agent, trajectory_lib.TunixAgentStep)
    self.assertEqual(restored_env.reward, 1.0)
    self.assertEqual(restored_agent.mc_return, 0.9)

  def test_deserialize_unknown_type_falls_back_to_base_class(self) -> None:
    step_payload = {
        "_step_type": "UnknownCustomStep",
        "step_id": 5,
        "source": "user",
        "message": "fallback test",
    }
    meta_payload = {
        "_metadata_type": "UnknownCustomMeta",
        "trajectory_id": "traj_unknown",
        "agent": {"name": "agent_u", "version": "1.0"},
    }

    restored_step = sql_serialization.deserialize_step(step_payload)
    restored_meta = sql_serialization.deserialize_metadata(meta_payload)

    self.assertIs(type(restored_step), trajectory_lib.Step)
    self.assertIs(type(restored_meta), trajectory_lib.TrajectoryMetadata)

  def test_build_trajectory_sorts_steps_and_preserves_tunix_type(self) -> None:
    meta = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="tunix_sorted",
        prompt_id="prompt_sort",
        agent=trajectory_lib.Agent(name="agent_s", version="1.0"),
    )
    step2 = trajectory_lib.TunixEnvStep(
        step_id=2,
        source=trajectory_lib.Source.USER,
        message="msg 2",
        reward=0.5,
    )
    step0 = trajectory_lib.TunixEnvStep(
        step_id=0,
        source=trajectory_lib.Source.USER,
        message="msg 0",
        reward=1.0,
    )
    step1 = trajectory_lib.TunixAgentStep(
        step_id=1, source=trajectory_lib.Source.AGENT, message="step 1"
    )

    traj = sql_serialization.build_trajectory(meta, [step2, step0, step1])

    self.assertIsInstance(traj, trajectory_lib.TunixTrajectory)
    self.assertEqual(traj.prompt_id, "prompt_sort")
    self.assertEqual([s.step_id for s in traj.steps], [0, 1, 2])


if __name__ == "__main__":
  absltest.main()
