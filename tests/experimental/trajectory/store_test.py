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

"""Tests for TrajectoryStore.from_config dispatch."""

import tempfile
from typing import Generic, TypeVar

from absl.testing import absltest
from etils import epath
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import sql_store
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib


class BackendRegistrationTest(absltest.TestCase):
  """Tests which classes `__init_subclass__` puts in the backend registry."""

  def test_subclass_without_own_backend_leaves_its_base_registered(self):
    # `BACKEND` is inherited like any other class attribute, so this subclass
    # is still *visible* as "file". Registration has to key off what a class
    # declares, not what it can see: a subclass that never asked to be a
    # backend would otherwise take over "file" for the rest of the process,
    # and `from_config` would hand every caller this class instead.
    class SubclassWithoutBackend(
        file_store.FileTrajectoryStore[trajectory_lib.TunixTrajectoryMetadata]
    ):
      pass

    self.assertEqual(SubclassWithoutBackend.BACKEND, "file")
    self.assertIs(
        store_lib.TrajectoryStore._REGISTRY["file"],
        file_store.FileTrajectoryStore,
    )

  def test_subclass_with_own_backend_registers_under_its_own_key(self):
    class SubclassWithBackend(file_store.FileTrajectoryStore):
      BACKEND = "file_registration_test"

    try:
      self.assertIs(
          store_lib.TrajectoryStore._REGISTRY["file_registration_test"],
          SubclassWithBackend,
      )
      self.assertIs(
          store_lib.TrajectoryStore._REGISTRY["file"],
          file_store.FileTrajectoryStore,
      )
    finally:
      store_lib.TrajectoryStore._REGISTRY.pop("file_registration_test", None)


class FromConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))

  def _file_config(self, **overrides):
    config = {
        "enabled": True,
        "backend": "file",
        "root_dir": str(self.tmp_dir),
        "run_id": "run_1",
    }
    config.update(overrides)
    return config

  def _sql_config(self, **overrides):
    config = {
        "enabled": True,
        "backend": "sql",
        "db_url": f"sqlite:///{self.tmp_dir / 'test.db'}",
        "run_id": "run_1",
    }
    config.update(overrides)
    return config

  def test_none_config_disables_the_store(self):
    self.assertIsNone(store_lib.TrajectoryStore.from_config(None))

  def test_missing_enabled_key_disables_the_store(self):
    self.assertIsNone(
        store_lib.TrajectoryStore.from_config({"backend": "memory"})
    )

  def test_enabled_false_disables_the_store(self):
    # The rest of the config stays valid and unused, so a run can be turned
    # off without unsetting its root_dir and run_id.
    self.assertIsNone(
        store_lib.TrajectoryStore.from_config(self._file_config(enabled=False))
    )

  def test_file_backend_is_scoped_by_run_id(self):
    store = store_lib.TrajectoryStore.from_config(self._file_config())
    self.assertIsInstance(store, file_store.FileTrajectoryStore)
    self.assertEqual(store.root_dir, self.tmp_dir / "run_1")
    store.close()

  def test_memory_backend_needs_no_other_keys(self):
    store = store_lib.TrajectoryStore.from_config(
        {"enabled": True, "backend": "memory"}
    )
    self.assertIsInstance(store, in_memory_store.InMemoryTrajectoryStore)

  def test_sql_backend_is_scoped_by_run_id(self):
    store = store_lib.TrajectoryStore.from_config(self._sql_config())
    self.assertIsInstance(store, sql_store.SqlTrajectoryStore)
    self.assertEqual(store.run_id, "run_1")
    store.close()

  def test_unknown_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown Trajectory Store"):
      store_lib.TrajectoryStore.from_config(
          {"enabled": True, "backend": "sqlite"}
      )

  def test_missing_backend_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown Trajectory Store"):
      store_lib.TrajectoryStore.from_config({"enabled": True})

  def test_file_backend_without_root_dir_raises(self):
    with self.assertRaisesRegex(ValueError, "'root_dir'"):
      store_lib.TrajectoryStore.from_config(self._file_config(root_dir=""))

  def test_file_backend_without_run_id_raises(self):
    # Optional on FileTrajectoryStore.__init__, required here: a configured
    # run is shared across processes and restarts, and run_id is what scopes
    # their common directory.
    with self.assertRaisesRegex(ValueError, "'run_id'"):
      store_lib.TrajectoryStore.from_config(self._file_config(run_id=None))

  def test_file_backend_rejects_a_run_id_that_is_not_a_path_segment(self):
    with self.assertRaisesRegex(ValueError, "unsupported characters"):
      store_lib.TrajectoryStore.from_config(self._file_config(run_id="a/b"))

  def test_sql_backend_without_db_url_raises(self):
    with self.assertRaisesRegex(ValueError, "db_url"):
      store_lib.TrajectoryStore.from_config(self._sql_config(db_url=""))

  def test_sql_backend_without_run_id_raises(self):
    with self.assertRaisesRegex(ValueError, "run_id"):
      store_lib.TrajectoryStore.from_config(self._sql_config(run_id=None))

  def test_two_calls_build_two_independent_stores(self):
    # Nothing here is a singleton: the guard against a second store (and, for
    # the file backend, a second writer thread) is that each process calls
    # this once and holds the result.
    config = self._file_config()
    first = store_lib.TrajectoryStore.from_config(config)
    second = store_lib.TrajectoryStore.from_config(config)
    self.assertIsNot(first, second)
    self.assertEqual(first.root_dir, second.root_dir)
    first.close()
    second.close()

  def test_custom_subclass_registers_automatically(self):
    class CustomStore(store_lib.TrajectoryStore):
      BACKEND = "custom_test_backend"

      @classmethod
      def _from_config(cls, config):
        return cls()

      def to_config(self):
        return {
            "enabled": True,
            "backend": self.BACKEND,
        }

      def get_trajectories_metadata(self, trajectory_ids=None):
        return []

      def get_trajectories(self, trajectory_ids):
        return []

      def add_step(self, step, metadata):
        pass

      def update_metadata(self, metadata):
        pass

      def flush(self):
        pass

      def close(self):
        pass

    try:
      store = store_lib.TrajectoryStore.from_config({
          "enabled": True,
          "backend": "custom_test_backend",
      })
      self.assertIsInstance(store, CustomStore)
    finally:
      store_lib.TrajectoryStore._REGISTRY.pop("custom_test_backend", None)

  def test_tunix_trajectory_store_from_config_binds_tunix_metadata(self):
    store = store_lib.TunixTrajectoryStore.from_config(self._file_config())
    self.assertIsInstance(store, file_store.FileTrajectoryStore)
    self.assertIs(
        store._metadata_cls,  # pylint: disable=protected-access
        trajectory_lib.TunixTrajectoryMetadata,
    )
    with self.assertRaisesRegex(
        TypeError,
        "FileTrajectoryStore is bound to metadata type"
        " TunixTrajectoryMetadata, got TrajectoryMetadata",
    ):
      store.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )
    store.close()


class GenericTypeParametersTest(absltest.TestCase):
  """Tests that TrajectoryStore and backends preserve generic type parameters without Generic."""

  def test_classes_inherit_parameters_without_explicit_generic(self) -> None:
    """Verifies that Python typing automatically discovers MetadataT."""
    self.assertLen(store_lib.TrajectoryStore.__parameters__, 1)
    self.assertLen(in_memory_store.InMemoryTrajectoryStore.__parameters__, 1)
    self.assertLen(file_store.FileTrajectoryStore.__parameters__, 1)
    self.assertLen(sql_store.SqlTrajectoryStore.__parameters__, 1)

    self.assertEqual(
        store_lib.TrajectoryStore.__parameters__,
        (store_lib.MetadataT,),
    )
    self.assertEqual(
        in_memory_store.InMemoryTrajectoryStore.__parameters__,
        (in_memory_store.MetadataT,),
    )
    self.assertEqual(
        file_store.FileTrajectoryStore.__parameters__,
        (file_store.MetadataT,),
    )
    self.assertEqual(
        sql_store.SqlTrajectoryStore.__parameters__,
        (sql_store.MetadataT,),
    )

  def test_classes_are_runtime_subscriptable(self) -> None:
    """Verifies that classes are subscriptable with concrete models at runtime."""
    subscripted_store = store_lib.TrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]
    subscripted_mem = in_memory_store.InMemoryTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]
    subscripted_file = file_store.FileTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]
    subscripted_sql = sql_store.SqlTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]
    self.assertIsNotNone(subscripted_store)
    self.assertIsNotNone(subscripted_mem)
    self.assertIsNotNone(subscripted_file)
    self.assertIsNotNone(subscripted_sql)

  def test_instances_satisfy_protocols_and_abc(self) -> None:
    """Verifies protocol and ABC conformance on instantiated instances."""
    mem = in_memory_store.InMemoryTrajectoryStore()
    self.assertIsInstance(mem, store_lib.TrajectoryStore)
    self.assertIsInstance(mem, store_lib.TrajectoryReader)
    self.assertIsInstance(mem, store_lib.TrajectoryWriter)

    tmp_dir = self.create_tempdir().full_path
    f_store = file_store.FileTrajectoryStore(root_dir=tmp_dir)
    self.assertIsInstance(f_store, store_lib.TrajectoryStore)
    self.assertIsInstance(f_store, store_lib.TrajectoryReader)
    self.assertIsInstance(f_store, store_lib.TrajectoryWriter)
    f_store.close()

    sql_s = sql_store.SqlTrajectoryStore(
        run_id="run_1",
        db_url=f"sqlite:///{tmp_dir}/store.db",
    )
    self.assertIsInstance(sql_s, store_lib.TrajectoryStore)
    self.assertIsInstance(sql_s, store_lib.TrajectoryReader)
    self.assertIsInstance(sql_s, store_lib.TrajectoryWriter)
    sql_s.close()

  def test_subclassing_closes_type_parameters(self) -> None:
    """Verifies that concrete subclassing binds and closes type parameters."""

    class ConcreteFileStore(
        file_store.FileTrajectoryStore[trajectory_lib.TunixTrajectoryMetadata]
    ):
      pass

    self.assertEmpty(ConcreteFileStore.__parameters__)
    tmp_dir = self.create_tempdir().full_path
    c_store = ConcreteFileStore(root_dir=tmp_dir)
    self.assertIsInstance(c_store, store_lib.TrajectoryStore)
    with self.assertRaisesRegex(
        TypeError,
        "ConcreteFileStore is bound to metadata type TunixTrajectoryMetadata,"
        " got TrajectoryMetadata",
    ):
      c_store.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )
    c_store.close()

  def test_subscripted_store_binds_metadata_type_on_first_write(self) -> None:
    """Verifies that Store[MetadataT]() rejects mismatched metadata on first write."""
    mem = in_memory_store.InMemoryTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ]()
    with self.assertRaisesRegex(
        TypeError,
        "InMemoryTrajectoryStore is bound to metadata type"
        " TunixTrajectoryMetadata, got TrajectoryMetadata",
    ):
      mem.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )

  def test_subscripted_store_binds_metadata_type_on_read(self) -> None:
    """Verifies that Store[MetadataT]() rejects mismatched metadata on read."""
    tmp_dir = self.create_tempdir().full_path
    writer = file_store.FileTrajectoryStore(root_dir=tmp_dir)
    writer.add_step(
        trajectory_lib.Step(
            step_id=1, source=trajectory_lib.Source.AGENT, message="step"
        ),
        trajectory_lib.TrajectoryMetadata(
            trajectory_id="t_1",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        ),
    )
    writer.close()

    reader = file_store.FileTrajectoryStore[
        trajectory_lib.TunixTrajectoryMetadata
    ](root_dir=tmp_dir)
    try:
      with self.assertRaisesRegex(
          TypeError,
          "FileTrajectoryStore is bound to metadata type"
          " TunixTrajectoryMetadata, got TrajectoryMetadata",
      ):
        reader.get_trajectories_metadata(["t_1"])
      with self.assertRaisesRegex(
          TypeError,
          "FileTrajectoryStore is bound to metadata type"
          " TunixTrajectoryMetadata, got TrajectoryMetadata",
      ):
        reader.get_trajectories(["t_1"])
    finally:
      reader.close()

  def test_multi_level_store_subclass_inherits_bound_metadata_cls(self) -> None:
    """Verifies that a subclass of a bound store subclass inherits _metadata_cls."""

    class BaseTunixStore(
        in_memory_store.InMemoryTrajectoryStore[
            trajectory_lib.TunixTrajectoryMetadata
        ]
    ):
      pass

    class DerivedTunixStore(BaseTunixStore):
      pass

    derived = DerivedTunixStore()
    derived.update_metadata(
        trajectory_lib.TunixTrajectoryMetadata(
            trajectory_id="t_1",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        )
    )
    with self.assertRaisesRegex(
        TypeError,
        "DerivedTunixStore is bound to metadata type TunixTrajectoryMetadata,"
        " got TrajectoryMetadata",
    ):
      derived.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_2",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )

  def test_multiple_inheritance_with_unrelated_generic_base(self) -> None:
    """Verifies unrelated generic bases do not shadow the store's MetadataT."""
    other_t = TypeVar("other_t")

    class UnrelatedGenericMixin(Generic[other_t]):
      pass

    class MixedStore(
        UnrelatedGenericMixin[trajectory_lib.TrajectoryMetadata],
        in_memory_store.InMemoryTrajectoryStore[
            trajectory_lib.TunixTrajectoryMetadata
        ],
    ):
      pass

    self.assertIs(
        MixedStore._metadata_cls,
        trajectory_lib.TunixTrajectoryMetadata,
    )
    store = MixedStore()
    with self.assertRaisesRegex(
        TypeError,
        "MixedStore is bound to metadata type TunixTrajectoryMetadata,"
        " got TrajectoryMetadata",
    ):
      store.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )

  def test_multiple_inheritance_with_multi_parameter_generic_store(
      self,
  ) -> None:
    """Verifies _extract_metadata_cls resolves MetadataT across multiple type params."""
    aux_t = TypeVar("aux_t")
    meta_t = TypeVar("meta_t", bound=trajectory_lib.TrajectoryMetadata)

    class MultiParamStore(
        Generic[aux_t, meta_t],
        in_memory_store.InMemoryTrajectoryStore[meta_t],
    ):
      pass

    class ConcreteMultiParamStore(
        MultiParamStore[int, trajectory_lib.TunixTrajectoryMetadata]
    ):
      pass

    self.assertIs(
        ConcreteMultiParamStore._metadata_cls,
        trajectory_lib.TunixTrajectoryMetadata,
    )

    alias_instance = MultiParamStore[
        str, trajectory_lib.TunixTrajectoryMetadata
    ]()
    with self.assertRaisesRegex(
        TypeError,
        "MultiParamStore is bound to metadata type TunixTrajectoryMetadata,"
        " got TrajectoryMetadata",
    ):
      alias_instance.update_metadata(
          trajectory_lib.TrajectoryMetadata(
              trajectory_id="t_1",
              agent=trajectory_lib.Agent(name="a", version="1.0"),
          )
      )

  def test_multiple_inheritance_combining_backend_and_tunix_store(self) -> None:
    """Verifies multiple inheritance with a non-alias bound store base."""

    class TunixInMemoryStore(
        in_memory_store.InMemoryTrajectoryStore,
        store_lib.TunixTrajectoryStore,
    ):
      pass

    self.assertIs(
        TunixInMemoryStore._metadata_cls,
        trajectory_lib.TunixTrajectoryMetadata,
    )

  def test_multiple_inheritance_conflicting_metadata_types_raises(self) -> None:
    """Verifies conflicting TrajectoryMetadata bases raise TypeError."""
    with self.assertRaisesRegex(
        TypeError,
        "Conflicting TrajectoryMetadata types in ConflictingStore:"
        " TrajectoryMetadata, TunixTrajectoryMetadata",
    ):

      class ConflictingStore(  # pylint: disable=unused-variable
          in_memory_store.InMemoryTrajectoryStore[
              trajectory_lib.TrajectoryMetadata
          ],
          store_lib.TunixTrajectoryStore,
      ):
        pass

  def test_separate_unparameterized_store_instances_bind_independently(
      self,
  ) -> None:
    """Verifies instance-level metadata binding does not mutate class defaults."""
    store_a = in_memory_store.InMemoryTrajectoryStore()
    store_b = in_memory_store.InMemoryTrajectoryStore()

    store_a.update_metadata(
        trajectory_lib.TunixTrajectoryMetadata(
            trajectory_id="t_1",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        )
    )
    store_b.update_metadata(
        trajectory_lib.TrajectoryMetadata(
            trajectory_id="t_2",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
        )
    )
    self.assertIsNone(in_memory_store.InMemoryTrajectoryStore._metadata_cls)
    self.assertLen(store_a.get_trajectories_metadata(), 1)
    self.assertLen(store_b.get_trajectories_metadata(), 1)

  def test_bound_store_rehydrates_its_bound_metadata_type_on_field_tie(
      self,
  ) -> None:
    """Verifies a bound store resolves its own MetadataT when two subclasses share field names."""

    class DuplicateMetaA(trajectory_lib.TrajectoryMetadata):
      EXTENSIONS_KEY = "_dup_extensions"
      shared_metric: int = 1

    class DuplicateMetaB(trajectory_lib.TrajectoryMetadata):
      EXTENSIONS_KEY = "_dup_extensions"
      shared_metric: int = 2

    self.assertIn(
        DuplicateMetaA,
        trajectory_lib.TrajectoryMetadata._SUBCLASS_FIELDS,  # pylint: disable=protected-access
    )
    tmp_dir = self.create_tempdir().full_path
    writer = file_store.FileTrajectoryStore[DuplicateMetaB](root_dir=tmp_dir)
    writer.update_metadata(
        DuplicateMetaB(
            trajectory_id="t_tie",
            agent=trajectory_lib.Agent(name="a", version="1.0"),
            shared_metric=7,
        )
    )
    writer.close()

    reader = file_store.FileTrajectoryStore[DuplicateMetaB](root_dir=tmp_dir)
    try:
      metas = reader.get_trajectories_metadata(["t_tie"])
      self.assertLen(metas, 1)
      self.assertIs(type(metas[0]), DuplicateMetaB)
      self.assertEqual(metas[0].shared_metric, 7)
    finally:
      reader.close()


class ExceptionsTest(absltest.TestCase):

  def test_trajectory_not_found_error(self):
    err = store_lib.TrajectoryNotFoundError("t_123")
    self.assertEqual(err.trajectory_id, "t_123")
    self.assertIn("t_123", str(err))

  def test_trajectory_metadata_not_found_error(self):
    err = store_lib.TrajectoryMetadataNotFoundError("t_456")
    self.assertEqual(err.trajectory_id, "t_456")
    self.assertIn("t_456", str(err))


if __name__ == "__main__":
  absltest.main()
