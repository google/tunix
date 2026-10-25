# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for export_trajectories_df."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from tunix.dist_rl.trajectory import export
from tunix.dist_rl.trajectory import file_store
from tunix.dist_rl.trajectory import in_memory_store
from tunix.dist_rl.trajectory import sql_store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


class ExportTrajectoriesDfTest(parameterized.TestCase):

  def test_export_trajectories_df_empty(self):
    store = in_memory_store.InMemoryTrajectoryStore()
    df = export.export_trajectories_df(store)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertTrue(df.empty)

  def test_export_trajectories_df_left_join_with_and_without_metrics(self):
    store = in_memory_store.InMemoryTrajectoryStore()

    # Trajectory 1: Complete with metrics
    meta1 = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_1",
        session_id="session_1",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
        prompt_id="prompt_42",
        group_index=0,
        status="SUCCEEDED",
        total_reward=1.0,
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_steps=3,
        ),
        extra={"custom_tag": "tag_a"},
    )
    store.update_metadata(meta1)
    metrics1 = trajectory_lib.TrajectoryMetrics(
        extra={
            "global_step": 10,
            "advantage": 0.85,
            "trajectory_reward": 1.0,
            "staleness": 0,
        },
    )
    store.update_metrics("traj_1", metrics1)

    # Trajectory 2: In-flight / dropped rollout without metrics
    meta2 = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_2",
        session_id="session_1",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
        prompt_id="prompt_42",
        group_index=1,
        status="FAILED",
        total_reward=0.0,
        final_metrics=trajectory_lib.FinalMetrics(
            total_prompt_tokens=80,
            total_completion_tokens=20,
            total_steps=1,
        ),
    )
    store.update_metadata(meta2)

    df = export.export_trajectories_df(store)

    self.assertEqual(len(df), 2)
    self.assertIn("trajectory_id", df.columns)
    self.assertIn("global_step", df.columns)
    self.assertIn("advantage", df.columns)
    self.assertIn("total_prompt_tokens", df.columns)
    self.assertIn("prompt_id", df.columns)

    # Row 0 (traj_1)
    row1 = df[df["trajectory_id"] == "traj_1"].iloc[0]
    self.assertEqual(row1["prompt_id"], "prompt_42")
    self.assertEqual(row1["group_index"], 0)
    self.assertEqual(row1["status"], "SUCCEEDED")
    self.assertEqual(row1["total_prompt_tokens"], 100)
    self.assertEqual(row1["total_completion_tokens"], 50)
    self.assertEqual(row1["total_steps"], 3)
    self.assertEqual(row1["global_step"], 10)
    self.assertAlmostEqual(row1["advantage"], 0.85)
    self.assertEqual(row1["custom_tag"], "tag_a")

    # Row 1 (traj_2) - Left join preserves row with NaN / None metric values
    row2 = df[df["trajectory_id"] == "traj_2"].iloc[0]
    self.assertEqual(row2["prompt_id"], "prompt_42")
    self.assertEqual(row2["group_index"], 1)
    self.assertEqual(row2["status"], "FAILED")
    self.assertEqual(row2["total_prompt_tokens"], 80)
    self.assertTrue(pd.isna(row2["global_step"]))
    self.assertTrue(pd.isna(row2["advantage"]))

  def test_export_trajectories_df_filter_by_trajectory_ids(self):
    store = in_memory_store.InMemoryTrajectoryStore()
    meta1 = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_1",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
        prompt_id="p1",
    )
    meta2 = trajectory_lib.TunixTrajectoryMetadata(
        trajectory_id="traj_2",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
        prompt_id="p2",
    )
    store.update_metadata(meta1)
    store.update_metadata(meta2)

    df = export.export_trajectories_df(store, trajectory_ids=["traj_2"])
    self.assertEqual(len(df), 1)
    self.assertEqual(df.iloc[0]["trajectory_id"], "traj_2")
    self.assertEqual(df.iloc[0]["prompt_id"], "p2")

  def test_export_trajectories_metadata_df_alias(self):
    self.assertIs(
        export.export_trajectories_metadata_df, export.export_trajectories_df
    )

  def test_export_trajectories_df_with_base_atif_rehydration(self):
    store = in_memory_store.InMemoryTrajectoryStore()
    # Base TrajectoryMetadata with packed _tunix_extensions
    base_meta = trajectory_lib.TrajectoryMetadata(
        trajectory_id="traj_atif",
        agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
        extra={
            trajectory_lib.TUNIX_EXTENSIONS_KEY: {
                "prompt_id": "prompt_atif_123",
                "group_index": 3,
                "status": "SUCCEEDED",
                "masked_out": False,
            },
            "user_key": "user_val",
        },
    )
    store.update_metadata(base_meta)

    df = export.export_trajectories_df(store)
    self.assertEqual(len(df), 1)
    row = df.iloc[0]
    self.assertEqual(row["trajectory_id"], "traj_atif")
    self.assertEqual(row["prompt_id"], "prompt_atif_123")
    self.assertEqual(row["group_index"], 3)
    self.assertEqual(row["status"], "SUCCEEDED")
    self.assertEqual(row["user_key"], "user_val")
    self.assertNotIn(trajectory_lib.TUNIX_EXTENSIONS_KEY, df.columns)

  def test_export_trajectories_df_across_file_and_sql_stores(self):
    temp_dir = self.create_tempdir().full_path
    f_store = file_store.FileTrajectoryStore(root_dir=temp_dir, run_id="run_f")
    try:
      meta = trajectory_lib.TunixTrajectoryMetadata(
          trajectory_id="traj_f",
          agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
          prompt_id="p_f",
      )
      f_store.update_metadata(meta)
      f_store.update_metrics(
          "traj_f",
          trajectory_lib.TrajectoryMetrics(extra={"metric_x": 42}),
      )
      f_store.flush()

      df_f = export.export_trajectories_df(f_store)
      self.assertEqual(len(df_f), 1)
      self.assertEqual(df_f.iloc[0]["trajectory_id"], "traj_f")
      self.assertEqual(df_f.iloc[0]["prompt_id"], "p_f")
      self.assertEqual(df_f.iloc[0]["metric_x"], 42)
    finally:
      f_store.close()

    db_path = f"sqlite:///{self.create_tempdir().full_path}/test.db"
    s_store = sql_store.SqlTrajectoryStore(run_id="run_s", db_url=db_path)
    try:
      meta = trajectory_lib.TunixTrajectoryMetadata(
          trajectory_id="traj_s",
          agent=trajectory_lib.Agent(name="test_agent", version="1.0.0"),
          prompt_id="p_s",
      )
      s_store.update_metadata(meta)
      s_store.update_metrics(
          "traj_s",
          trajectory_lib.TrajectoryMetrics(extra={"metric_y": 99}),
      )
      s_store.flush()

      df_s = export.export_trajectories_df(s_store)
      self.assertEqual(len(df_s), 1)
      self.assertEqual(df_s.iloc[0]["trajectory_id"], "traj_s")
      self.assertEqual(df_s.iloc[0]["prompt_id"], "p_s")
      self.assertEqual(df_s.iloc[0]["metric_y"], 99)
    finally:
      s_store.close()


if __name__ == "__main__":
  absltest.main()
