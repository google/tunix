# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the transport-neutral weight sync contract."""

from __future__ import annotations

import inspect

from absl.testing import absltest

from tunix.experimental.orchestrator import weight_sync


def _tensor(**overrides) -> weight_sync.TensorMetadata:
  fields = dict(
      name="w",
      shape=(8, 4),
      mesh_shape=(1, 2),
      layout=(1, 0),
      item_size=4,
      sharding_spec=("", "tp"),
  )
  fields.update(overrides)
  return weight_sync.TensorMetadata(**fields)


class WorkUnitIdTest(absltest.TestCase):

  def test_preserves_the_complete_transport_neutral_identity(self):
    unit = weight_sync.WorkUnitId(
        job_name="trainer",
        job_replica_id="host-2",
        data_name="model.layers.0.mlp.weight",
        data_replica_idx=3,
    )

    self.assertEqual(unit.job_name, "trainer")
    self.assertEqual(unit.job_replica_id, "host-2")
    self.assertEqual(unit.data_name, "model.layers.0.mlp.weight")
    self.assertEqual(unit.data_replica_idx, 3)

  def test_rejects_an_empty_job_or_negative_replica(self):
    with self.assertRaisesRegex(ValueError, "job_name"):
      weight_sync.WorkUnitId(job_name="")
    with self.assertRaisesRegex(ValueError, "non-negative"):
      weight_sync.WorkUnitId(job_name="trainer", data_replica_idx=-1)


class TensorMetadataTest(absltest.TestCase):

  def test_accepts_partial_layout_and_replicated_dimensions(self):
    tensor = _tensor(layout=(-1, 0))

    self.assertEqual(tensor.layout, (-1, 0))
    self.assertEqual(tensor.sharding_spec, ("", "tp"))

  def test_rejects_an_empty_name_or_invalid_shape(self):
    with self.assertRaisesRegex(ValueError, "name"):
      _tensor(name="")
    for shape in ((), (8, 0), (-1, 4)):
      with self.subTest(shape=shape):
        with self.assertRaisesRegex(ValueError, "invalid shape"):
          _tensor(shape=shape)

  def test_rejects_invalid_mesh_or_layout_rank(self):
    with self.assertRaisesRegex(ValueError, "mesh_shape"):
      _tensor(mesh_shape=(2,))
    with self.assertRaisesRegex(ValueError, "positive dimensions"):
      _tensor(mesh_shape=(1, 0))
    with self.assertRaisesRegex(ValueError, "layout"):
      _tensor(layout=(0,))

  def test_rejects_invalid_item_size_or_layer_index(self):
    with self.assertRaisesRegex(ValueError, "item_size"):
      _tensor(item_size=0)
    with self.assertRaisesRegex(ValueError, "layer_idx"):
      _tensor(layer_idx=-1)

  def test_rejects_invalid_sharding_spec(self):
    with self.assertRaisesRegex(ValueError, "sharding_spec"):
      _tensor(sharding_spec=("tp",))
    with self.assertRaisesRegex(ValueError, "may not shard two"):
      _tensor(sharding_spec=("tp", "tp"))


class NeutralContractTest(absltest.TestCase):

  def test_work_unit_metadata_carries_a_multi_tensor_manifest(self):
    tensors = (_tensor(name="w0"), _tensor(name="w1", layer_idx=1))
    metadata = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId("trainer"),
        variables=tensors,
        mesh_shape=(1, 2),
        mesh_axes=("fsdp", "tp"),
    )

    self.assertEqual(metadata.variables, tensors)
    self.assertEqual(metadata.mesh_axes, ("fsdp", "tp"))

  def test_handler_boundary_exposes_no_raiden_types(self):
    with self.assertRaises(TypeError):
      weight_sync.WeightSyncHandler()
    for method_name in ("register_work_unit", "transfer"):
      signature = inspect.signature(
          getattr(weight_sync.WeightSyncHandler, method_name)
      )
      self.assertNotIn("Raiden", str(signature))
    self.assertFalse(hasattr(weight_sync.TensorMetadata, "to_proto"))

  def test_module_does_not_expose_the_raiden_implementation(self):
    self.assertFalse(hasattr(weight_sync, "RaidenHandler"))
    self.assertFalse(hasattr(weight_sync, "RaidenTransferOptions"))
    self.assertFalse(hasattr(weight_sync, "raiden_controller"))


if __name__ == "__main__":
  absltest.main()
