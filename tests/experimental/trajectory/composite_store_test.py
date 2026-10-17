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

"""Unit and contract tests for CompositeTrajectoryStore."""

from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from tunix.experimental.trajectory import composite_store
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import store_testing
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing


def _create_file_and_sql_composite(
    test_case: absltest.TestCase,
    *,
    root_dir: epath.Path | None = None,
    run_id: str = "test_composite_run",
) -> tuple[
    composite_store.CompositeTrajectoryStore[Any],
    file_store.FileTrajectoryStore[Any],
    sql_store.SqlTrajectoryStore[Any],
]:
  """Creates a CompositeTrajectoryStore backed by File and SQLite stores."""
  if root_dir is None:
    root_dir = epath.Path(test_case.create_tempdir().full_path)
  file_child = file_store.FileTrajectoryStore(
      root_dir=root_dir / "files",
      run_id=run_id,
  )
  sql_child = sql_store.SqlTrajectoryStore(
      run_id=run_id,
      db_url=f"sqlite:///{root_dir / 'traj.db'}",
  )
  comp = composite_store.CompositeTrajectoryStore(
      stores=[file_child, sql_child]
  )
  test_case.addCleanup(comp.close)
  return comp, file_child, sql_child


class CompositeTrajectoryReaderTest(store_testing.TrajectoryReaderTestCase):
  """Contract tests for CompositeTrajectoryStore's TrajectoryReader interface."""

  # More trajectories than FileTrajectoryStore read threads so concurrent reads
  # on the primary child store finish out of request order.
  NUM_MANY_TRAJECTORIES = 3 * file_store._MAX_READ_WORKERS

  def _create_reader(
      self,
      initial_data: (
          list[
              tuple[
                  trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]
              ]
          ]
          | None
      ) = None,
  ) -> store.TrajectoryReader:
    comp, _, _ = _create_file_and_sql_composite(self, run_id="reader_run")
    if initial_data:
      for meta, steps in initial_data:
        if not steps:
          comp.update_metadata(meta)
        for step in steps:
          comp.add_step(step, meta)
      comp.flush()
    return comp


class CompositeTrajectoryWriterTest(store_testing.TrajectoryWriterTestCase):
  """Contract tests for CompositeTrajectoryStore's TrajectoryWriter interface."""

  def _create_reader_and_writer(
      self,
  ) -> tuple[store.TrajectoryReader, store.TrajectoryWriter]:
    comp, _, _ = _create_file_and_sql_composite(self, run_id="writer_run")
    return comp, comp


class CompositeTrajectoryStoreMetadataClsTest(
    store_testing.PersistentTrajectoryStoreMetadataClsTestCase
):
  """Contract tests for metadata subclass persistence and rehydration."""

  def setUp(self) -> None:
    super().setUp()
    self._shared_dir = epath.Path(self.create_tempdir().full_path)

  def _create_store(
      self,
  ) -> composite_store.CompositeTrajectoryStore[Any]:
    comp, _, _ = _create_file_and_sql_composite(
        self, root_dir=self._shared_dir, run_id="meta_cls_run"
    )
    return comp


class CompositeTrajectoryStoreConfigTest(
    store_testing.TrajectoryStoreConfigTestCase
):
  """Contract and unit tests for CompositeTrajectoryStore configuration."""

  def setUp(self) -> None:
    super().setUp()
    self._tmp_dir = epath.Path(self.create_tempdir().full_path)

  def _create_config(self) -> dict[str, Any]:
    return {
        "enabled": True,
        "backend": "composite",
        "stores": [
            {
                "enabled": True,
                "backend": "file",
                "root_dir": str(self._tmp_dir / "files"),
                "run_id": "cfg_run",
            },
            {
                "enabled": True,
                "backend": "sql",
                "db_url": f"sqlite:///{self._tmp_dir / 'cfg.db'}",
                "run_id": "cfg_run",
            },
        ],
    }

  def test_to_redacted_config_delegates_redaction_to_child_stores(self) -> None:
    file_child = file_store.FileTrajectoryStore(
        root_dir=self._tmp_dir / "files",
        run_id="redact_run",
    )
    self.addCleanup(file_child.close)
    mock_sql_child = mock.create_autospec(
        sql_store.SqlTrajectoryStore, instance=True
    )
    mock_sql_child.to_config.return_value = {
        "enabled": True,
        "backend": "sql",
        "db_url": "postgresql+psycopg2://postgres:secret-pw@10.0.0.5:5432/db",
        "run_id": "redact_run",
    }
    mock_sql_child.to_redacted_config.return_value = {
        "enabled": True,
        "backend": "sql",
        "db_url": "postgresql+psycopg2://postgres:***@10.0.0.5:5432/db",
        "run_id": "redact_run",
    }
    comp = composite_store.CompositeTrajectoryStore(
        stores=[file_child, mock_sql_child]
    )

    self.assertEqual(
        comp.to_config()["stores"][1]["db_url"],
        "postgresql+psycopg2://postgres:secret-pw@10.0.0.5:5432/db",
    )
    self.assertEqual(
        comp.to_redacted_config()["stores"][1]["db_url"],
        "postgresql+psycopg2://postgres:***@10.0.0.5:5432/db",
    )

  def test_from_config_skips_disabled_child_stores(self) -> None:
    config = self._create_config()
    config["stores"][1]["enabled"] = False
    config["stores"][1]["db_url"] = ""

    built = self._build_store(config)

    self.assertIsInstance(built, composite_store.CompositeTrajectoryStore)
    assert isinstance(built, composite_store.CompositeTrajectoryStore)
    self.assertLen(built.stores, 1)
    self.assertIsInstance(built.stores[0], file_store.FileTrajectoryStore)

  @parameterized.named_parameters(
      (
          "missing_stores",
          {"enabled": True, "backend": "composite"},
          "non-empty 'stores' sequence",
      ),
      (
          "empty_stores",
          {"enabled": True, "backend": "composite", "stores": []},
          "non-empty 'stores' sequence",
      ),
      (
          "string_stores",
          {"enabled": True, "backend": "composite", "stores": "file"},
          "non-empty 'stores' sequence",
      ),
      (
          "non_mapping_child",
          {"enabled": True, "backend": "composite", "stores": ["file"]},
          "configuration mapping",
      ),
      (
          "all_children_disabled",
          {
              "enabled": True,
              "backend": "composite",
              "stores": [{"enabled": False, "backend": "memory"}],
          },
          "at least one enabled child store",
      ),
  )
  def test_from_config_invalid_stores_raises_value_error(
      self, bad_config: dict[str, Any], error_regex: str
  ) -> None:
    with self.assertRaisesRegex(ValueError, error_regex):
      store.TrajectoryStore.from_config(bad_config)

  def test_from_config_closes_earlier_children_when_later_child_fails(
      self,
  ) -> None:
    bad_config = self._create_config()
    bad_config["stores"][1]["db_url"] = ""

    with mock.patch.object(
        file_store.FileTrajectoryStore,
        "close",
        autospec=True,
        wraps=file_store.FileTrajectoryStore.close,
    ) as mock_file_close:
      with self.assertRaisesRegex(ValueError, "non-empty db_url"):
        store.TrajectoryStore.from_config(bad_config)

    mock_file_close.assert_called_once()


class CompositeTrajectoryStoreFanOutTest(trajectory_testing.TrajectoryTestCase):
  """Tests verifying multi-backend fan-out, zero-duplication, and isolation."""

  @parameterized.named_parameters(
      ("empty_stores", [], ValueError, "at least one child store"),
      ("non_store_element", ["not_a_store"], TypeError, "TrajectoryStore"),
  )
  def test_init_with_invalid_stores_raises_error(
      self,
      invalid_stores: list[Any],
      expected_error: type[Exception],
      error_regex: str,
  ) -> None:
    with self.assertRaisesRegex(expected_error, error_regex):
      composite_store.CompositeTrajectoryStore(stores=invalid_stores)

  @parameterized.named_parameters(
      (
          "conflicting_sibling_children",
          lambda: composite_store.CompositeTrajectoryStore(
              stores=cast(
                  Any,
                  [
                      in_memory_store.InMemoryTrajectoryStore[
                          trajectory_lib.TunixTrajectoryMetadata
                      ](),
                      in_memory_store.InMemoryTrajectoryStore[
                          trajectory_lib.TrajectoryMetadata
                      ](),
                  ],
              )
          ),
          trajectory_testing.TUNIX_METADATA_1,
          (
              "bound to metadata type TrajectoryMetadata, got"
              " TunixTrajectoryMetadata"
          ),
      ),
      (
          "subscripted_composite_with_conflicting_child",
          lambda: composite_store.CompositeTrajectoryStore[
              trajectory_lib.TrajectoryMetadata
          ](
              stores=cast(
                  Any,
                  [
                      in_memory_store.InMemoryTrajectoryStore[
                          trajectory_lib.TunixTrajectoryMetadata
                      ]()
                  ],
              )
          ),
          trajectory_testing.METADATA_1,
          (
              "bound to metadata type TunixTrajectoryMetadata, got"
              " TrajectoryMetadata"
          ),
      ),
  )
  def test_write_with_conflicting_bound_metadata_types_raises_type_error(
      self,
      store_factory: Any,
      metadata: trajectory_lib.TrajectoryMetadata,
      error_regex: str,
  ) -> None:
    comp = store_factory()
    self.addCleanup(comp.close)

    with self.assertRaisesRegex(TypeError, error_regex):
      comp.update_metadata(metadata)

  def test_fan_out_persists_tunix_trajectory_across_all_child_stores(
      self,
  ) -> None:
    _, file_child, sql_child = _create_file_and_sql_composite(self)
    mem_child = in_memory_store.InMemoryTrajectoryStore()
    comp = composite_store.CompositeTrajectoryStore(
        stores=[mem_child, file_child, sql_child]
    )
    self.addCleanup(comp.close)
    initial_meta = trajectory_testing.TUNIX_METADATA_1.model_copy(
        update={"status": "RUNNING"}
    )

    for step in trajectory_testing.PAIRED_TUNIX_TRAJECTORY.steps:
      comp.add_step(step, initial_meta)
    comp.update_metadata(
        trajectory_testing.PAIRED_TUNIX_TRAJECTORY.get_metadata()
    )
    comp.flush()

    traj_id = trajectory_testing.PAIRED_TUNIX_TRAJECTORY.trajectory_id
    assert traj_id is not None
    for reader in (comp, mem_child, file_child, sql_child):
      (recovered,) = reader.get_trajectories([traj_id])
      self.assertTrajectoryEqual(
          recovered, trajectory_testing.PAIRED_TUNIX_TRAJECTORY
      )

  def test_fan_out_shares_immutable_scalars_while_isolating_containers(
      self,
  ) -> None:
    child_a = in_memory_store.InMemoryTrajectoryStore()
    child_b = in_memory_store.InMemoryTrajectoryStore()
    comp = composite_store.CompositeTrajectoryStore(stores=[child_a, child_b])
    self.addCleanup(comp.close)

    caller_extra = {"mutable_list": [1, 2, 3]}
    step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={
            "step_id": 0,
            "assistant_tokens": [101, 102, 103],
            "extra": caller_extra,
        }
    )
    comp.add_step(step, trajectory_testing.TUNIX_METADATA_1)

    # Mutating caller_extra after add_step must not affect either child store.
    caller_extra["mutable_list"].append(999)

    step_a = child_a.get_trajectories(["t_atif"])[0].steps[0]
    step_b = child_b.get_trajectories(["t_atif"])[0].steps[0]
    assert isinstance(step_a, trajectory_lib.TunixAgentStep)
    assert isinstance(step_b, trajectory_lib.TunixAgentStep)

    # Immutable scalar payloads (strings, ints) are shared by reference.
    self.assertIs(step_a.message, step_b.message)
    # Mutable containers are isolated across child stores and from the caller.
    self.assertIsNot(step_a.extra, step_b.extra)
    self.assertIsNot(step_a.assistant_tokens, step_b.assistant_tokens)
    self.assertEqual(step_a.extra, {"mutable_list": [1, 2, 3]})
    self.assertEqual(step_b.extra, {"mutable_list": [1, 2, 3]})

  def test_nested_composite_stores_fan_out_and_round_trip_config(self) -> None:
    inner_comp, file_child, sql_child = _create_file_and_sql_composite(
        self, run_id="nested_run"
    )
    mem_child = in_memory_store.InMemoryTrajectoryStore()
    outer_comp = composite_store.CompositeTrajectoryStore(
        stores=[mem_child, inner_comp]
    )
    self.addCleanup(outer_comp.close)

    outer_comp.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    outer_comp.flush()

    traj_id = trajectory_testing.TRAJECTORY_ID_1
    for reader in (outer_comp, mem_child, file_child, sql_child):
      self.assertEqual(
          reader.get_trajectories([traj_id]), [trajectory_testing.TRAJECTORY_1]
      )
    self.assertEqual(
        outer_comp.to_config()["stores"][1], inner_comp.to_config()
    )

  def test_close_attempts_all_children_when_first_child_raises(self) -> None:
    child_1 = in_memory_store.InMemoryTrajectoryStore()
    child_2 = in_memory_store.InMemoryTrajectoryStore()
    comp = composite_store.CompositeTrajectoryStore(stores=[child_1, child_2])

    with (
        mock.patch.object(
            child_1, "close", side_effect=RuntimeError("child 1 close error")
        ),
        mock.patch.object(child_2, "close") as mock_child_2_close,
    ):
      with self.assertRaisesRegex(RuntimeError, "child 1 close error"):
        comp.close()

    mock_child_2_close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
