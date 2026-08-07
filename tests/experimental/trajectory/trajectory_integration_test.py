"""Integration tests for ATIF trajectory schema and store implementations."""

import concurrent.futures
import datetime
import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory

_SAMPLE_ATIF_PATH = os.path.join(
    os.path.dirname(trajectory.__file__), "testdata", "sample_atif_v1_7.json"
)

_TEST_DATETIME = datetime.datetime(
    2026, 8, 7, 10, 0, 0, tzinfo=datetime.timezone.utc
)


class TrajectoryIntegrationTest(parameterized.TestCase):
  """Integration tests covering end-to-end flows for trajectories and stores."""

  def setUp(self) -> None:
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))

  def test_sample_atif_json_roundtrip_through_stores(self) -> None:
    """Verifies that sample ATIF JSON can be loaded and round-tripped through both stores."""
    with open(_SAMPLE_ATIF_PATH, "r", encoding="utf-8") as f:
      original_traj = trajectory.Trajectory.from_json_dict(json.load(f))

    metadata = original_traj.get_metadata()

    # 1. Test InMemoryTrajectoryStore roundtrip
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    for step in original_traj.steps:
      mem_store.add_step(step, metadata)

    mem_metas = mem_store.get_trajectories_metadata()
    self.assertEqual(mem_metas, [metadata])

    (mem_reloaded_traj,) = mem_store.get_trajectories([metadata.trajectory_id])
    self.assertEqual(mem_reloaded_traj, original_traj)

    # 2. Test FileTrajectoryStore roundtrip
    f_store = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="sample_roundtrip_run"
    )
    for step in original_traj.steps:
      f_store.add_step(step, metadata)

    file_metas = f_store.get_trajectories_metadata()
    self.assertEqual(file_metas, [metadata])

    (file_reloaded_traj,) = f_store.get_trajectories([metadata.trajectory_id])
    self.assertEqual(file_reloaded_traj, original_traj)

  def test_end_to_end_agent_subagent_lifecycle(self) -> None:
    """Simulates a full agent-subagent execution lifecycle with trajectory stores."""
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    f_store = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="agent_lifecycle_run"
    )

    # Define subagent metadata and trajectory
    subagent_id = "sub_traj_001"
    sub_metadata = trajectory.TrajectoryMetadata(
        trajectory_id=subagent_id,
        session_id="session-sub-1",
        agent=trajectory.Agent(
            name="code_editor_subagent",
            version="1.0.0",
            model_name="gemini-1.5-flash",
        ),
    )

    sub_step_1 = trajectory.Step(
        step_id=1,
        timestamp=_TEST_DATETIME,
        source=trajectory.Source.USER,
        message="Refactor file.py to remove unused imports",
    )
    sub_step_2 = trajectory.Step(
        step_id=2,
        timestamp=_TEST_DATETIME,
        source=trajectory.Source.AGENT,
        model_name="gemini-1.5-flash",
        reasoning_effort=0.8,
        message="Removed 2 unused imports in file.py",
        reasoning_content="Analyzed AST and stripped unused imports.",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call_edit_1",
                function_name="edit_file",
                arguments={"file": "file.py", "action": "remove_unused"},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call_edit_1",
                    content="Successfully edited file.py",
                )
            ]
        ),
        metrics=trajectory.Metrics(
            prompt_tokens=50, completion_tokens=15, cost_usd=0.0001
        ),
        llm_call_count=1,
    )

    # Save subagent steps
    for s in [sub_step_1, sub_step_2]:
      mem_store.add_step(s, sub_metadata)
      f_store.add_step(s, sub_metadata)

    # Define main agent metadata
    main_id = "main_traj_100"
    main_metadata = trajectory.TrajectoryMetadata(
        trajectory_id=main_id,
        session_id="session-main-1",
        agent=trajectory.Agent(
            name="orchestrator_agent",
            version="2.0.0",
            model_name="gemini-1.5-pro",
        ),
        final_metrics=trajectory.FinalMetrics(
            total_prompt_tokens=200,
            total_completion_tokens=50,
            total_cost_usd=0.001,
            total_steps=2,
        ),
    )

    main_step_1 = trajectory.Step(
        step_id=1,
        timestamp=_TEST_DATETIME,
        source=trajectory.Source.USER,
        message="Please cleanup unused imports in project.",
    )

    # Main agent delegates to subagent and receives observation with subagent_trajectory_ref
    subagent_ref = trajectory.SubagentTrajectoryRef(
        trajectory_id=subagent_id,
        session_id="session-sub-1",
        extra={"status": "completed"},
    )
    main_step_2 = trajectory.Step(
        step_id=2,
        timestamp=_TEST_DATETIME,
        source=trajectory.Source.AGENT,
        model_name="gemini-1.5-pro",
        reasoning_effort=1.0,
        message="Delegated cleanup to subagent and completed task.",
        reasoning_content="Delegated file refactoring to code_editor_subagent.",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call_delegate_1",
                function_name="run_subagent",
                arguments={"subagent_name": "code_editor_subagent"},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call_delegate_1",
                    content="Subagent completed refactoring.",
                    subagent_trajectory_ref=[subagent_ref],
                )
            ]
        ),
        metrics=trajectory.Metrics(
            prompt_tokens=150, completion_tokens=35, cost_usd=0.0009
        ),
        llm_call_count=1,
    )

    # Save main agent steps
    for s in [main_step_1, main_step_2]:
      mem_store.add_step(s, main_metadata)
      f_store.add_step(s, main_metadata)

    # Verify retrieval from FileTrajectoryStore
    retrieved_trajs = f_store.get_trajectories([main_id, subagent_id])
    self.assertLen(retrieved_trajs, 2)

    traj_dict = {t.trajectory_id: t for t in retrieved_trajs}
    self.assertIn(main_id, traj_dict)
    self.assertIn(subagent_id, traj_dict)

    main_t = traj_dict[main_id]
    self.assertLen(main_t.steps, 2)
    self.assertEqual(main_t.final_metrics.total_steps, 2)
    obs_ref = main_t.steps[1].observation.results[0].subagent_trajectory_ref[0]
    self.assertEqual(obs_ref.trajectory_id, subagent_id)

    sub_t = traj_dict[subagent_id]
    self.assertLen(sub_t.steps, 2)
    self.assertEqual(sub_t.steps[1].tool_calls[0].function_name, "edit_file")

  def test_store_interoperability_and_migration(self) -> None:
    """Tests writing data to InMemoryTrajectoryStore and migrating to FileTrajectoryStore."""
    source_store = in_memory_store.InMemoryTrajectoryStore()

    meta_a = trajectory.TrajectoryMetadata(
        trajectory_id="traj_a",
        agent=trajectory.Agent(name="agent_a", version="1.0"),
    )
    step_a1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.USER,
        message="Query A",
        timestamp=_TEST_DATETIME,
    )
    step_a2 = trajectory.Step(
        step_id=2,
        source=trajectory.Source.AGENT,
        message="Response A",
        timestamp=_TEST_DATETIME,
    )
    source_store.add_step(step_a1, meta_a)
    source_store.add_step(step_a2, meta_a)

    meta_b = trajectory.TrajectoryMetadata(
        trajectory_id="traj_b",
        agent=trajectory.Agent(name="agent_b", version="2.0"),
    )
    step_b1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.USER,
        message="Query B",
        timestamp=_TEST_DATETIME,
    )
    source_store.add_step(step_b1, meta_b)

    # Migrate from source_store to destination FileTrajectoryStore
    target_store = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="migration_run"
    )

    all_metas = source_store.get_trajectories_metadata()
    self.assertLen(all_metas, 2)

    for meta in all_metas:
      (traj,) = source_store.get_trajectories([meta.trajectory_id])
      for step in traj.steps:
        target_store.add_step(step, meta)

    # Verify target_store contains exact same metadata and trajectories
    target_metas = target_store.get_trajectories_metadata()
    self.assertCountEqual(target_metas, all_metas)

    target_trajs = target_store.get_trajectories(["traj_a", "traj_b"])
    source_trajs = source_store.get_trajectories(["traj_a", "traj_b"])
    self.assertCountEqual(target_trajs, source_trajs)

  def test_concurrent_multi_threaded_logging(self) -> None:
    """Tests concurrent writes across multiple threads to FileTrajectoryStore and InMemoryTrajectoryStore."""
    mem_store = in_memory_store.InMemoryTrajectoryStore()
    f_store = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="concurrent_run"
    )

    num_threads = 8
    steps_per_thread = 5

    def log_trajectory(thread_idx: int) -> str:
      traj_id = f"concurrent_traj_{thread_idx}"
      metadata = trajectory.TrajectoryMetadata(
          trajectory_id=traj_id,
          agent=trajectory.Agent(name=f"worker_{thread_idx}", version="1.0"),
      )
      for step_idx in range(1, steps_per_thread + 1):
        step = trajectory.Step(
            step_id=step_idx,
            source=(
                trajectory.Source.USER
                if step_idx % 2 == 1
                else trajectory.Source.AGENT
            ),
            message=f"Message {step_idx} from thread {thread_idx}",
            timestamp=_TEST_DATETIME,
        )
        mem_store.add_step(step, metadata)
        f_store.add_step(step, metadata)
      return traj_id

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
      futures = [
          executor.submit(log_trajectory, i) for i in range(num_threads)
      ]
      traj_ids = [f.result() for f in concurrent.futures.as_completed(futures)]

    self.assertLen(traj_ids, num_threads)

    # Verify both stores captured all metadata and steps
    mem_metas = mem_store.get_trajectories_metadata()
    file_metas = f_store.get_trajectories_metadata()
    self.assertLen(mem_metas, num_threads)
    self.assertLen(file_metas, num_threads)

    mem_trajs = mem_store.get_trajectories(traj_ids)
    file_trajs = f_store.get_trajectories(traj_ids)

    self.assertLen(mem_trajs, num_threads)
    self.assertLen(file_trajs, num_threads)

    for traj in file_trajs:
      self.assertLen(traj.steps, steps_per_thread)
      for i, step in enumerate(traj.steps, start=1):
        self.assertEqual(step.step_id, i)

  def test_store_recovery_and_persistence_across_instances(self) -> None:
    """Simulates process restart by initializing a new FileTrajectoryStore instance on existing directory."""
    run_id = "persistent_run_42"

    # Instance 1: write initial steps
    store_instance_1 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    meta_1 = trajectory.TrajectoryMetadata(
        trajectory_id="traj_p1",
        agent=trajectory.Agent(name="agent_1", version="1.0"),
    )
    step_1_1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.USER,
        message="Initial prompt",
        timestamp=_TEST_DATETIME,
    )
    store_instance_1.add_step(step_1_1, meta_1)

    # Instance 2: new process reading and appending to same run_id
    store_instance_2 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )

    metas_2 = store_instance_2.get_trajectories_metadata()
    self.assertEqual(metas_2, [meta_1])

    step_1_2 = trajectory.Step(
        step_id=2,
        source=trajectory.Source.AGENT,
        message="Followup response",
        timestamp=_TEST_DATETIME,
    )
    store_instance_2.add_step(step_1_2, meta_1)

    # Instance 3: verify complete recovered state
    store_instance_3 = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id=run_id
    )
    (recovered_traj,) = store_instance_3.get_trajectories(["traj_p1"])
    self.assertLen(recovered_traj.steps, 2)
    self.assertEqual(recovered_traj.steps[0].message, "Initial prompt")
    self.assertEqual(recovered_traj.steps[1].message, "Followup response")

  def test_embedded_subagent_trajectories_serialization(self) -> None:
    """Verifies trajectories containing embedded subagent_trajectories serialize and deserialize."""
    sub_traj = trajectory.Trajectory(
        trajectory_id="embedded_sub_1",
        agent=trajectory.Agent(name="sub_agent", version="1.0"),
        steps=[
            trajectory.Step(
                step_id=1,
                source=trajectory.Source.AGENT,
                message="Subagent action",
                timestamp=_TEST_DATETIME,
            )
        ],
    )

    parent_traj = trajectory.Trajectory(
        trajectory_id="parent_traj_1",
        agent=trajectory.Agent(name="parent_agent", version="2.0"),
        steps=[
            trajectory.Step(
                step_id=1,
                source=trajectory.Source.USER,
                message="Parent request",
                timestamp=_TEST_DATETIME,
            )
        ],
        subagent_trajectories=[sub_traj],
    )

    json_dict = parent_traj.to_json_dict()
    reconstructed = trajectory.Trajectory.from_json_dict(json_dict)

    self.assertEqual(reconstructed, parent_traj)
    self.assertLen(reconstructed.subagent_trajectories, 1)
    self.assertEqual(
        reconstructed.subagent_trajectories[0].trajectory_id, "embedded_sub_1"
    )

  def test_store_error_handling_and_boundary_conditions(self) -> None:
    """Verifies error handling for missing trajectories, invalid metadata, and unsupported characters."""
    f_store = file_store.FileTrajectoryStore(
        root_dir=self.tmp_dir, run_id="error_run"
    )

    # 1. Non-existent trajectory ID
    with self.assertRaises(store.TrajectoryNotFoundError):
      f_store.get_trajectories(["non_existent_traj"])

    # 2. Add step with empty trajectory ID
    invalid_meta_empty = trajectory.TrajectoryMetadata(
        trajectory_id="", agent=trajectory.Agent(name="agent", version="1.0")
    )
    dummy_step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.USER,
        message="test",
        timestamp=_TEST_DATETIME,
    )
    with self.assertRaises(ValueError):
      f_store.add_step(dummy_step, invalid_meta_empty)

    # 3. Add step with invalid trajectory ID characters
    invalid_meta_chars = trajectory.TrajectoryMetadata(
        trajectory_id="invalid/traj/id",
        agent=trajectory.Agent(name="agent", version="1.0"),
    )
    with self.assertRaises(ValueError):
      f_store.add_step(dummy_step, invalid_meta_chars)


if __name__ == "__main__":
  absltest.main()
