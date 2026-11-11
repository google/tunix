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

import os
from typing import Any, ClassVar
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
from tunix.dist_rl.trajectory import base_store
from tunix.dist_rl.trajectory import export
from tunix.dist_rl.trajectory import file_store
from tunix.dist_rl.trajectory import in_memory_store
from tunix.dist_rl.trajectory import sql_store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

_AGENT = trajectory_lib.Agent(name="agent", version="1.0")


class _MetricsFieldMetadata(trajectory_lib.TrajectoryMetadata):
  """Metadata whose `metrics` field collides with the `metrics.*` columns.

  Subclass registration is process-global, so this uses a unique extensions key
  that no other metadata in this test module packs into `extra`.
  """

  EXTENSIONS_KEY: ClassVar[str] = "export_test_metrics_field"

  metrics: dict[str, Any] | None = None


def _metadata(
    trajectory_id: str, **kwargs: Any
) -> trajectory_lib.TrajectoryMetadata:
  return trajectory_lib.TrajectoryMetadata(
      trajectory_id=trajectory_id, agent=_AGENT, **kwargs
  )


class ExportTrajectoriesDfTest(parameterized.TestCase):

  def _create_store(self, backend: str) -> base_store.TrajectoryStore[Any]:
    tmp_dir = self.create_tempdir().full_path
    match backend:
      case "memory":
        return in_memory_store.InMemoryTrajectoryStore()
      case "file":
        trajectory_store = file_store.FileTrajectoryStore(root_dir=tmp_dir)
      case "sql":
        trajectory_store = sql_store.SqlTrajectoryStore(
            run_id="export_test",
            db_url=f"sqlite:///{os.path.join(tmp_dir, 'store.db')}",
        )
      case _:
        raise ValueError(f"Unknown backend {backend!r}.")
    self.addCleanup(trajectory_store.close)
    return trajectory_store

  def _populate(
      self,
      trajectory_store: base_store.TrajectoryStore[Any],
      metadata: list[trajectory_lib.TrajectoryMetadata],
      metrics: dict[str, trajectory_lib.TrajectoryMetrics],
  ) -> None:
    for meta in metadata:
      trajectory_store.update_metadata(meta)
    for trajectory_id, trajectory_metrics in metrics.items():
      trajectory_store.update_metrics(trajectory_id, trajectory_metrics)
    trajectory_store.flush()

  @parameterized.parameters("memory", "file", "sql")
  def test_empty_store_returns_empty_df_with_id_column(
      self, backend: str
  ) -> None:
    df = export.export_trajectories_df(self._create_store(backend))

    self.assertEmpty(df)
    self.assertEqual(list(df.columns), ["trajectory_id"])

  @parameterized.parameters("memory", "file", "sql")
  def test_left_joins_metadata_and_metrics(self, backend: str) -> None:
    trajectory_store = self._create_store(backend)
    self._populate(
        trajectory_store,
        metadata=[
            _metadata(
                "t1",
                final_metrics=trajectory_lib.FinalMetrics(
                    total_steps=3, extra={"accuracy": 0.5}
                ),
                extra={"task": "math"},
            ),
            _metadata("t2", extra={"task": "code"}),
        ],
        metrics={
            "t1": trajectory_lib.TrajectoryMetrics(
                extra={"reward": 1.0, "timing": {"env_s": 2.5}}
            ),
        },
    )

    df = export.export_trajectories_df(trajectory_store).set_index(
        "trajectory_id"
    )

    self.assertCountEqual(df.index, ["t1", "t2"])
    self.assertEqual(df.loc["t1", "agent.name"], "agent")
    self.assertEqual(df.loc["t1", "final_metrics.total_steps"], 3)
    self.assertEqual(df.loc["t1", "final_metrics.extra.accuracy"], 0.5)
    self.assertEqual(df.loc["t1", "extra.task"], "math")
    self.assertEqual(df.loc["t2", "extra.task"], "code")
    self.assertEqual(df.loc["t1", "metrics.reward"], 1.0)
    self.assertEqual(df.loc["t1", "metrics.timing.env_s"], 2.5)
    # t2 has no TrajectoryMetrics, so its row is kept with null metrics.
    self.assertTrue(pd.isna(df.loc["t2", "metrics.reward"]))
    self.assertTrue(pd.isna(df.loc["t2", "metrics.timing.env_s"]))
    self.assertTrue(pd.isna(df.loc["t2", "final_metrics.total_steps"]))

  def test_preserves_metadata_order(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("b"), _metadata("a"), _metadata("c")],
        metrics={
            "c": trajectory_lib.TrajectoryMetrics(extra={"reward": 3.0}),
            "a": trajectory_lib.TrajectoryMetrics(extra={"reward": 1.0}),
        },
    )

    df = export.export_trajectories_df(trajectory_store)

    self.assertEqual(df["trajectory_id"].tolist(), ["b", "a", "c"])
    self.assertEqual(df.columns[0], "trajectory_id")
    pd.testing.assert_series_equal(
        df["metrics.reward"],
        pd.Series([float("nan"), 1.0, 3.0], name="metrics.reward"),
    )

  @parameterized.parameters("memory", "file", "sql")
  def test_trajectory_ids_filter_applies_to_metadata_and_metrics(
      self, backend: str
  ) -> None:
    trajectory_store = self._create_store(backend)
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1"), _metadata("t2"), _metadata("t3")],
        metrics={
            "t1": trajectory_lib.TrajectoryMetrics(extra={"reward": 1.0}),
            "t3": trajectory_lib.TrajectoryMetrics(extra={"loss": 0.1}),
        },
    )

    # "t2" has no metrics, which must not raise.
    df = export.export_trajectories_df(
        trajectory_store, trajectory_ids=["t2", "t1"]
    )

    self.assertEqual(df["trajectory_id"].tolist(), ["t2", "t1"])
    self.assertNotIn("metrics.loss", df.columns)
    self.assertTrue(pd.isna(df["metrics.reward"].iloc[0]))
    self.assertEqual(df["metrics.reward"].iloc[1], 1.0)

  def test_trajectory_ids_with_all_metrics_present(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1"), _metadata("t2")],
        metrics={
            "t1": trajectory_lib.TrajectoryMetrics(extra={"reward": 1.0}),
            "t2": trajectory_lib.TrajectoryMetrics(extra={"reward": 2.0}),
        },
    )

    df = export.export_trajectories_df(trajectory_store, trajectory_ids=["t2"])

    self.assertEqual(df["trajectory_id"].tolist(), ["t2"])
    self.assertEqual(df["metrics.reward"].tolist(), [2.0])

  def test_missing_metadata_for_requested_id_raises(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(trajectory_store, metadata=[_metadata("t1")], metrics={})

    with self.assertRaises(base_store.TrajectoryMetadataNotFoundError):
      export.export_trajectories_df(trajectory_store, trajectory_ids=["nope"])

  @parameterized.parameters("memory", "file", "sql")
  def test_metrics_without_metadata_add_no_columns(self, backend: str) -> None:
    trajectory_store = self._create_store(backend)
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1"), _metadata("t2")],
        metrics={
            "t1": trajectory_lib.TrajectoryMetrics(extra={"reward": 1.0}),
            "orphan": trajectory_lib.TrajectoryMetrics(extra={"orphan_key": 9}),
        },
    )

    df = export.export_trajectories_df(trajectory_store)

    self.assertEqual(df["trajectory_id"].tolist(), ["t1", "t2"])
    self.assertNotIn("metrics.orphan_key", df.columns)
    self.assertEqual(
        [c for c in df.columns if c.startswith("metrics.")], ["metrics.reward"]
    )

  def test_only_orphan_metrics_add_no_metrics_columns(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1")],
        metrics={
            "orphan": trajectory_lib.TrajectoryMetrics(extra={"reward": 9.0}),
        },
    )

    df = export.export_trajectories_df(trajectory_store)

    self.assertEqual(df["trajectory_id"].tolist(), ["t1"])
    self.assertFalse(any(c.startswith("metrics") for c in df.columns))

  def test_single_trajectory_id_without_metrics_skips_full_scan(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1"), _metadata("t2")],
        metrics={"t2": trajectory_lib.TrajectoryMetrics(extra={"reward": 2.0})},
    )

    with mock.patch.object(
        trajectory_store,
        "get_trajectories_metrics",
        wraps=trajectory_store.get_trajectories_metrics,
    ) as get_metrics:
      df = export.export_trajectories_df(
          trajectory_store, trajectory_ids=["t1"]
      )

    self.assertEqual(df["trajectory_id"].tolist(), ["t1"])
    self.assertFalse(any(c.startswith("metrics") for c in df.columns))
    get_metrics.assert_called_once_with(["t1"])

  @parameterized.named_parameters(
      dict(
          testcase_name="metadata_extra",
          metadata_extra={"timing.env_s": 1.0, "timing": {"env_s": 2.5}},
          metrics_extra=None,
      ),
      dict(
          testcase_name="metrics_extra",
          metadata_extra=None,
          metrics_extra={"timing.env_s": 1.0, "timing": {"env_s": 2.5}},
      ),
  )
  def test_duplicate_flattened_keys_raise(
      self,
      metadata_extra: dict[str, Any] | None,
      metrics_extra: dict[str, Any] | None,
  ) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1", extra=metadata_extra)],
        metrics={"t1": trajectory_lib.TrajectoryMetrics(extra=metrics_extra)},
    )

    with self.assertRaisesRegex(ValueError, "Duplicate flattened key"):
      export.export_trajectories_df(trajectory_store)

  def test_metadata_metrics_field_collision_raises(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[
            _MetricsFieldMetadata(
                trajectory_id="t1", agent=_AGENT, metrics={"reward": 1.0}
            )
        ],
        metrics={},
    )

    with self.assertRaisesRegex(ValueError, "reserved 'metrics' column prefix"):
      export.export_trajectories_df(trajectory_store)

  def test_no_metrics_at_all_has_no_metrics_columns(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(trajectory_store, metadata=[_metadata("t1")], metrics={})

    df = export.export_trajectories_df(trajectory_store)

    self.assertEqual(df["trajectory_id"].tolist(), ["t1"])
    self.assertFalse(any(c.startswith("metrics") for c in df.columns))

  def test_same_key_in_metadata_extra_and_metrics_does_not_collide(
      self,
  ) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[
            _metadata(
                "t1",
                final_metrics=trajectory_lib.FinalMetrics(
                    extra={"reward": 1.0}
                ),
                extra={"reward": 2.0},
            )
        ],
        metrics={"t1": trajectory_lib.TrajectoryMetrics(extra={"reward": 3.0})},
    )

    df = export.export_trajectories_df(trajectory_store)

    self.assertTrue(df.columns.is_unique)
    self.assertEqual(df["final_metrics.extra.reward"].iloc[0], 1.0)
    self.assertEqual(df["extra.reward"].iloc[0], 2.0)
    self.assertEqual(df["metrics.reward"].iloc[0], 3.0)

  def test_drops_null_placeholder_for_partially_set_dicts(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore()
    self._populate(
        trajectory_store,
        metadata=[_metadata("t1", extra={"task": "math"}), _metadata("t2")],
        metrics={},
    )

    df = export.export_trajectories_df(trajectory_store)

    self.assertIn("extra.task", df.columns)
    self.assertNotIn("extra", df.columns)
    # Unset scalar fields remain as (null) columns for a stable layout.
    self.assertIn("notes", df.columns)

  def test_flattens_metadata_subclass_fields(self) -> None:
    trajectory_store = in_memory_store.InMemoryTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]()
    self._populate(
        trajectory_store,
        metadata=[
            trajectory_lib.TunixTrajectoryMetadata(
                trajectory_id="t1",
                agent=_AGENT,
                status="SUCCEEDED",
                total_reward=0.75,
                hyperparams={"temperature": 0.7},
            )
        ],
        metrics={"t1": trajectory_lib.TrajectoryMetrics(extra={"kl": 0.01})},
    )

    df = export.export_trajectories_df(trajectory_store)

    row = df.iloc[0]
    self.assertEqual(row["status"], "SUCCEEDED")
    self.assertEqual(row["total_reward"], 0.75)
    self.assertEqual(row["hyperparams.temperature"], 0.7)
    self.assertEqual(row["metrics.kl"], 0.01)


if __name__ == "__main__":
  absltest.main()
