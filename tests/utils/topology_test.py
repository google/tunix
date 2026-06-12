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

from absl.testing import absltest
from tunix.utils import topology


class TopologyTest(absltest.TestCase):

  def test_normalize_device_kind_recognizes_supported_families(self):
    self.assertEqual(topology._normalize_device_kind("TPU v7"), "v7x")
    self.assertEqual(topology._normalize_device_kind("TPU v6e"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v6 lite"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v5e"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5 lite"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5p"), "v5p")
    self.assertEqual(topology._normalize_device_kind("TPU v4"), "v4")
    self.assertIsNone(topology._normalize_device_kind("gpu"))

  def test_best_topology_shapes_for_chip_count_returns_unique_edge_shape(self):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=2),
        [(2, 4)],
    )
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=3),
        [(2, 4, 1)],
    )

  def test_best_topology_shapes_for_chip_count_returns_empty_for_unsupported_edge_count(
      self,
  ):
    # 3 chips is not a supported edge shape, so there is no candidate.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 3),
        [],
    )

  def test_best_topology_shapes_for_chip_count_prefers_most_cubical_fish_shape(
      self,
  ):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 256),
        [(4, 8, 8)],
    )

  def test_best_topology_shapes_for_chip_count_supports_single_host_fish_subslice(
      self,
  ):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7",
            2,
            available_chip_shape=(4, 4, 4),
        ),
        [(2, 1, 1)],
    )

  def test_best_topology_shapes_for_chip_count_returns_unique_fish_sub_cube(
      self,
  ):
    # Below the first full cube each volume maps to exactly one sub-cube shape.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 16),
        [(2, 2, 4)],
    )
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 32),
        [(2, 4, 4)],
    )

  def test_best_topology_shapes_for_chip_count_fish_sub_cube_respects_available_shape(
      self,
  ):
    # The 16-chip sub-cube (2, 2, 4) cannot fit when z is limited to 1.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7", 16, available_chip_shape=(2, 2, 1)
        ),
        [],
    )

  def test_best_topology_shapes_for_chip_count_filters_by_available_shape(self):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v6e",
            8,
            chip_rank=2,
            available_chip_shape=(1, 8),
        ),
        [],
    )

  def test_best_topology_shapes_for_chip_count_derives_shape_within_remaining_region(
      self,
  ):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7",
            576,
            available_chip_shape=(4, 12, 16),
        ),
        [(4, 12, 12)],
    )

  def test_best_topology_shapes_for_chip_count_rejects_non_cube_multiple(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by 64 chips"):
      topology.best_topology_shapes_for_chip_count("TPU v7", 96)

  def test_best_fish_cube_shape_returns_single_cube(self):
    self.assertEqual(topology._best_fish_cube_shape(64, None), (4, 4, 4))

  def test_best_fish_cube_shape_prefers_most_cubical_arrangement(self):
    # 256 chips = 4 cubes; (4, 8, 8) is more cubical than (4, 4, 16).
    self.assertEqual(topology._best_fish_cube_shape(256, None), (4, 8, 8))

  def test_best_fish_cube_shape_respects_available_shape(self):
    self.assertEqual(
        topology._best_fish_cube_shape(576, (4, 12, 16)), (4, 12, 12)
    )

  def test_best_fish_cube_shape_returns_none_below_one_cube(self):
    self.assertIsNone(topology._best_fish_cube_shape(32, None))

  def test_best_fish_cube_shape_returns_none_when_no_arrangement_fits(self):
    # 128 chips = 2 cubes, but a 4x4x4 bound leaves room for only one cube.
    self.assertIsNone(topology._best_fish_cube_shape(128, (4, 4, 4)))

  def test_best_fish_cube_shape_raises_on_non_cube_multiple(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by 64 chips"):
      topology._best_fish_cube_shape(96, None)

  def test_device_family_resolves_from_first_device_kind(self):
    class FakeDevice:

      def __init__(self, device_kind):
        self.device_kind = device_kind

    self.assertEqual(topology._device_family([FakeDevice("TPU v6e")]), "v6e")
    self.assertEqual(topology._device_family([FakeDevice("TPU v7")]), "v7x")

  def test_device_family_returns_none_without_device_kind(self):
    class FakeDevice:
      pass

    self.assertIsNone(topology._device_family([]))
    self.assertIsNone(topology._device_family([FakeDevice()]))

  def test_resolve_family_accepts_raw_kind_and_family_key(self):
    self.assertEqual(topology._resolve_family("TPU v5 lite"), "v5e")
    self.assertEqual(topology._resolve_family("v5e"), "v5e")  # already a key
    self.assertIsNone(topology._resolve_family("gpu"))

  def test_device_attr_calls_callable_attributes(self):
    class FakeDevice:
      coords = (1, 2, 3)

      def device_kind(self):  # exposed lazily as a method
        return "TPU v7"

    device = FakeDevice()
    self.assertEqual(topology._device_attr(device, "device_kind"), "TPU v7")
    self.assertEqual(topology._device_attr(device, "coords"), (1, 2, 3))
    self.assertIsNone(topology._device_attr(device, "missing"))
    self.assertEqual(topology._device_attr(device, "missing", default=5), 5)

  def test_pathways_device_host_attr_parses_int_and_list_values(self):
    class FakeDevice:

      def __repr__(self):
        return "device(0,TPU,coords=[1,2,3],logical_task=11,slice=3)"

    device = FakeDevice()
    self.assertEqual(
        topology._pathways_device_host_attr(device, "logical_task"), 11
    )
    self.assertEqual(
        topology._pathways_device_host_attr(device, "coords"), [1, 2, 3]
    )

  def test_pathways_device_host_attr_returns_none_when_absent(self):
    class FakeDevice:

      def __repr__(self):
        return "device(0,TPU,slice=3)"

    self.assertIsNone(
        topology._pathways_device_host_attr(FakeDevice(), "logical_task")
    )

  def test_pathways_device_host_attr_anchors_on_delimiter(self):
    # A field that merely ends with the attr name must not match.
    class FakeDevice:

      def __repr__(self):
        return "device(0,TPU,xlogical_task=99,logical_task=7)"

    self.assertEqual(
        topology._pathways_device_host_attr(FakeDevice(), "logical_task"), 7
    )


if __name__ == "__main__":
  absltest.main()
