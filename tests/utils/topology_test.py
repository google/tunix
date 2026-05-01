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
    self.assertEqual(topology._normalize_device_kind("TPU v7"), "tpu7x")
    self.assertEqual(topology._normalize_device_kind("TPU v6e"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v6 lite"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v5e"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5 lite"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5p"), "v5p")
    self.assertEqual(topology._normalize_device_kind("TPU v4"), "v4")
    self.assertIsNone(topology._normalize_device_kind("gpu"))

  def test_infer_chips_per_host_bounds_returns_none_for_empty_devices(self):
    self.assertIsNone(topology.infer_chips_per_host_bounds([]))

  def test_infer_chips_per_host_bounds_returns_none_for_missing_device_kind(self):
    class FakeDevice:
      pass

    self.assertIsNone(topology.infer_chips_per_host_bounds([FakeDevice()]))

  def test_infer_chips_per_host_bounds_uses_single_host_shapes(self):
    class FakeDevice:

      def __init__(self, device_kind, coords=None):
        self.device_kind = device_kind
        self.coords = coords

    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v5e", (0, 0, 0))]),
        (1, 1, 1),
    )
    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v6e", (0, 0))]),
        (1, 1, 1),
    )
    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v6e", (0, 0, 0))]),
        (1, 1, 1),
    )
    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v7"), FakeDevice("TPU v7")]),
        (1, 1, 1),
    )

  def test_infer_chips_per_host_bounds_uses_multi_host_shape_otherwise(self):
    class FakeDevice:

      def __init__(self, device_kind, coords=None):
        self.device_kind = device_kind
        self.coords = coords

    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v7") for _ in range(4)]),
        (2, 2, 1),
    )
    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v4") for _ in range(8)]),
        (2, 2, 1),
    )
    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice("TPU v5e", (0, 0)), FakeDevice("TPU v5e", (0, 1))]),
        (2, 2, 1),
    )

  def test_infer_chips_per_host_bounds_prefers_runtime_host_shape(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.device_kind = "TPU v5 lite"
        self.coords = coords
        self.process_index = process_index

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((1, 0, 0), 0),
        FakeDevice((0, 1, 0), 0),
        FakeDevice((1, 1, 0), 0),
        FakeDevice((0, 2, 0), 0),
        FakeDevice((1, 2, 0), 0),
        FakeDevice((0, 3, 0), 0),
        FakeDevice((1, 3, 0), 0),
    ]

    self.assertEqual(
        topology.infer_chips_per_host_bounds(fake_devices),
        (2, 4, 1),
    )

  def test_infer_chips_per_host_bounds_handles_callable_device_kind(self):
    class FakeDevice:

      def device_kind(self):
        return "TPU v7"

    self.assertEqual(
        topology.infer_chips_per_host_bounds([FakeDevice() for _ in range(128)]),
        (2, 2, 1),
    )

  def test_best_topology_shapes_for_chip_count_returns_unique_edge_shape(self):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=2),
        [(2, 4)],
    )
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=3),
        [(2, 4, 1)],
    )

  def test_best_topology_shapes_for_chip_count_prefers_most_cubical_fish_shape(self):
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 256),
        [(4, 8, 8)],
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

  def test_best_topology_shapes_for_chip_count_derives_shape_within_remaining_region(self):
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


if __name__ == "__main__":
  absltest.main()
