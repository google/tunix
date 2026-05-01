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

from unittest import mock

from absl.testing import absltest
import jax
from tunix.utils import mesh


class MeshUtilsTest(absltest.TestCase):

  def test_device_host_key_prefers_slice_and_process_metadata(self):
    class FakeDevice:

      def __init__(self):
        self.slice_index = 4
        self.process_index = 7

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 7))

  def test_device_host_key_uses_task_id_without_slice_index(self):
    class FakeDevice:

      def __init__(self):
        self.task_id = 9

    self.assertEqual(mesh.device_host_key(FakeDevice()), (None, 9))

  def test_device_host_key_prefers_task_id_over_process_index(self):
    class FakeDevice:

      def __init__(self):
        self.slice_index = 4
        self.process_index = 0
        self.task_id = 9

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 9))

  def test_device_host_key_returns_none_without_task_metadata(self):
    class FakeDevice:
      pass

    self.assertIsNone(mesh.device_host_key(FakeDevice()))

  def test_device_slice_id_uses_slice_index_only(self):
    class SliceIndexDevice:

      def __init__(self):
        self.slice_index = 4

    self.assertEqual(mesh.device_slice_id(SliceIndexDevice()), 4)
    self.assertIsNone(mesh.device_slice_id(object()))

  def test_group_devices_by_slice_preserves_first_seen_order(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index):
        self.id = device_id
        self.slice_index = slice_index

    grouped = mesh.group_devices_by_slice([
        FakeDevice(0, 2),
        FakeDevice(1, 2),
        FakeDevice(2, 1),
        FakeDevice(3, 1),
    ])

    self.assertEqual([[device.id for device in group] for group in grouped], [[0, 1], [2, 3]])

  def test_group_devices_by_slice_treats_missing_metadata_as_one_slice(self):
    class FakeDevice:

      def __init__(self, device_id):
        self.id = device_id

    grouped = mesh.group_devices_by_slice([
        FakeDevice(0),
        FakeDevice(1),
    ])

    self.assertEqual([[device.id for device in group] for group in grouped], [[0, 1]])

  def test_candidate_uses_whole_chips_requires_all_cores(self):
    class FakeDevice:

      def __init__(self, device_id, coords, core_on_chip):
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip

    topology = mesh.get_coord_topology([
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (0, 0, 0), 1),
        FakeDevice(2, (1, 0, 0), 0),
        FakeDevice(3, (1, 0, 0), 1),
    ])

    self.assertFalse(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0, 0), (1, 0, 0, 0)],
        )
    )
    self.assertTrue(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 1)],
        )
    )

  def test_candidate_uses_whole_chips_ignores_plain_coords_without_core_axis(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    topology = mesh.get_coord_topology([
        FakeDevice(0, (0, 0, 0)),
        FakeDevice(1, (0, 0, 1)),
    ])

    self.assertTrue(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0)],
        )
    )

  def test_satisfies_host_bound_shape_rejects_ragged_coords(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    host_devices = [
        FakeDevice(0, (1, 1, 0)),
        FakeDevice(1, (1, 0, 1)),
        FakeDevice(2, (0, 1, 1)),
        FakeDevice(3, (1, 1, 1)),
    ]

    self.assertFalse(
        mesh._satisfies_host_bound_shape(
            host_devices,
            (2, 2, 1),
            4,
        )
    )

  def test_get_coord_topology_builds_bounding_box(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (2, 1, 0)),
        FakeDevice(1, (3, 1, 0)),
        FakeDevice(2, (2, 2, 0)),
        FakeDevice(3, (3, 2, 0)),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertIsNotNone(topology)
    self.assertEqual(topology.num_dims, 3)
    self.assertEqual(topology.max_shape, (2, 2, 1))
    self.assertEqual(topology.all_coords, ((2, 1, 0), (3, 1, 0), (2, 2, 0), (3, 2, 0)))

  def test_get_coord_topology_rejects_duplicate_coords(self):
    class FakeDevice:

      def __init__(self, coords):
        self.coords = coords

    fake_devices = [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 0))]

    self.assertIsNone(mesh.get_coord_topology(fake_devices))

  def test_get_coord_topology_uses_core_on_chip_to_disambiguate_devices(self):
    class FakeDevice:

      def __init__(self, coords, core_on_chip):
        self.coords = coords
        self.core_on_chip = core_on_chip

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((0, 0, 0), 1),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertIsNotNone(topology)
    self.assertEqual(topology.all_coords, ((0, 0, 0, 0), (0, 0, 0, 1)))
    self.assertTrue(topology.has_core_on_chip_dimension)

  def test_get_coord_topology_marks_plain_coords_without_core_axis(self):
    class FakeDevice:

      def __init__(self, coords):
        self.coords = coords

    topology = mesh.get_coord_topology([FakeDevice((0, 0, 0)), FakeDevice((0, 0, 1))])

    self.assertIsNotNone(topology)
    self.assertFalse(topology.has_core_on_chip_dimension)

  def test_get_coord_topology_rejects_empty_device_list(self):
    self.assertIsNone(mesh.get_coord_topology([]))

  def test_get_coord_topology_rejects_mismatched_coord_dimensions(self):
    class FakeDevice:

      def __init__(self, coords):
        self.coords = coords

    fake_devices = [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 0, 1))]

    self.assertIsNone(mesh.get_coord_topology(fake_devices))

  def test_summarize_devices_for_logging_includes_id_coords_and_host(self):
    class FakeDevice:

      def __init__(self, device_id, coords, process_index, slice_index):
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.slice_index = slice_index

    self.assertEqual(
        mesh.summarize_devices_for_logging([FakeDevice(11, (1, 2, 0), 5, 6)]),
        [{"id": 11, "coords": (1, 2, 0), "host": (6, 5)}],
    )

  def test_group_devices_by_host_groups_equal_sized_hosts(self):
    class FakeDevice:

      def __init__(self, device_id, process_index):
        self.id = device_id
        self.process_index = process_index

    grouped = mesh.group_devices_by_host([
        FakeDevice(0, 0),
        FakeDevice(1, 0),
        FakeDevice(2, 1),
        FakeDevice(3, 1),
    ])

    self.assertEqual([[device.id for device in group] for group in grouped], [[0, 1], [2, 3]])

  def test_group_devices_by_host_sorts_by_slice_then_host_id(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, process_index):
        self.id = device_id
        self.slice_index = slice_index
        self.process_index = process_index

    grouped = mesh.group_devices_by_host([
        FakeDevice(3, 1, 1),
        FakeDevice(2, 1, 0),
        FakeDevice(1, 0, 1),
        FakeDevice(0, 0, 0),
    ])

    self.assertEqual([[device.id for device in group] for group in grouped], [[0], [1], [2], [3]])

  def test_allocate_named_mesh_device_slices_ignores_host_groups_when_coord_allocation_fails(self):
    class FakeDevice:

      def __init__(self, device_id, task_id, coords, core_on_chip):
        self.id = device_id
        self.process_index = 0
        self.task_id = task_id
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for task_id, z in ((0, 0), (1, 1)):
      for x in range(2):
        for y in range(2):
          for core_on_chip in (0, 1):
            fake_devices.append(
                FakeDevice(device_id, task_id, (x, y, z), core_on_chip)
            )
            device_id += 1

    with self.assertRaisesRegex(
        ValueError,
        "coord-based allocation could not construct a valid box",
    ):
      with mock.patch.object(mesh, "_allocate_devices_by_coords", return_value=(None, None)):
        mesh.allocate_named_mesh_device_slices(
            [("actor", 8)],
            devices=fake_devices,
        )

  def test_group_devices_by_host_returns_none_without_host_metadata(self):
    class FakeDevice:
      pass

    self.assertIsNone(mesh.group_devices_by_host([FakeDevice()]))

  def test_group_devices_by_host_returns_none_for_inconsistent_host_sizes(self):
    class FakeDevice:

      def __init__(self, device_id, process_index):
        self.id = device_id
        self.process_index = process_index

    self.assertIsNone(
        mesh.group_devices_by_host([
            FakeDevice(0, 0),
            FakeDevice(1, 0),
            FakeDevice(2, 1),
        ])
    )

  def test_divisors_returns_sorted_unique_factors(self):
    self.assertEqual(mesh._divisors(12), [1, 2, 3, 4, 6, 12])

  def test_enumerate_box_shapes_returns_shapes_with_requested_volume(self):
    self.assertEqual(
        mesh._enumerate_box_shapes(4, (4, 2, 2)),
        [(1, 2, 2), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    )

  def test_coord_box_score_prefers_more_compact_shapes(self):
    compact_score = mesh._coord_box_score((0, 0, 0), (2, 2, 1))
    stretched_score = mesh._coord_box_score((0, 0, 0), (1, 4, 1))

    self.assertLess(compact_score, stretched_score)

  def test_split_coord_region_returns_z_y_x_remainders(self):
    region = mesh.CoordRegion((0, 0, 0), (16, 16, 16))

    self.assertEqual(
      mesh._split_coord_region(region, (0, 0, 0), (4, 4, 8)),
        (
            mesh.CoordRegion((0, 0, 8), (4, 4, 8)),
            mesh.CoordRegion((0, 4, 0), (4, 12, 16)),
            mesh.CoordRegion((4, 0, 0), (12, 16, 16)),
        ),
    )

  def test_split_coord_region_accounts_for_non_origin_allocated_start(self):
    region = mesh.CoordRegion((0, 0, 0), (16, 16, 16))

    self.assertEqual(
        mesh._split_coord_region(region, (4, 4, 4), (4, 4, 8)),
        (
            mesh.CoordRegion((4, 4, 0), (4, 4, 4)),
            mesh.CoordRegion((4, 4, 12), (4, 4, 4)),
            mesh.CoordRegion((4, 0, 0), (4, 4, 16)),
            mesh.CoordRegion((4, 8, 0), (4, 8, 16)),
            mesh.CoordRegion((0, 0, 0), (4, 16, 16)),
            mesh.CoordRegion((8, 0, 0), (8, 16, 16)),
        ),
    )

  def test_device_mesh_coords_appends_core_on_chip_when_present(self):
    class FakeDevice:

      def __init__(self):
        self.coords = (1, 2, 0)
        self.core_on_chip = 1

    self.assertEqual(
        mesh.device_mesh_coords(FakeDevice()),
        (1, 2, 0, 1),
    )

  def test_device_mesh_coords_returns_none_without_coords(self):
    class FakeDevice:
      pass

    self.assertIsNone(mesh.device_mesh_coords(FakeDevice()))

  def test_known_host_mesh_shape_returns_none_for_unknown_device_family(self):
    class FakeDevice:

      def __init__(self):
        self.coords = (0, 0, 0)
        self.device_kind = "unknown"

    self.assertIsNone(mesh.known_host_mesh_shape([FakeDevice()]))

  def test_known_host_mesh_shape_returns_none_when_coord_rank_mismatches_bounds(self):
    class FakeDevice:

      def __init__(self):
        self.coords = (0, 0)
        self.device_kind = "TPU v7"

    fake_devices = [FakeDevice() for _ in range(128)]

    self.assertIsNone(mesh.known_host_mesh_shape(fake_devices))

  def test_resolve_per_host_mesh_shape_returns_inferred_shape(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((1, 0, 0), 0),
        FakeDevice((0, 1, 0), 0),
        FakeDevice((1, 1, 0), 0),
        FakeDevice((2, 0, 0), 1),
        FakeDevice((3, 0, 0), 1),
        FakeDevice((2, 1, 0), 1),
        FakeDevice((3, 1, 0), 1),
    ]

    self.assertEqual(mesh.resolve_per_host_mesh_shape(fake_devices), (2, 2, 1))

  def test_known_host_mesh_shape_uses_static_topology_metadata(self):
    class FakeDevice:

      def __init__(self):
        self.coords = (0, 0, 0)
        self.device_kind = "TPU v7"

    fake_devices = [FakeDevice() for _ in range(128)]

    self.assertEqual(
        mesh.known_host_mesh_shape(fake_devices),
        (2, 2, 1),
    )

  def test_known_host_mesh_shape_uses_single_host_bounds_for_tpu7x_2(self):
    class FakeDevice:

      def __init__(self, coords):
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 0))]

    self.assertEqual(
        mesh.known_host_mesh_shape(fake_devices),
        (1, 1, 1),
    )

  def test_known_host_mesh_shape_appends_core_dimension_when_present(self):
    class FakeDevice:

      def __init__(self, coords, core_on_chip):
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.device_kind = "TPU v7"

    fake_devices = []
    for x in range(4):
      for y in range(4):
        for z in range(4):
          for core_on_chip in (0, 1):
            fake_devices.append(FakeDevice((x, y, z), core_on_chip))

    self.assertEqual(
        mesh.known_host_mesh_shape(fake_devices),
        (2, 2, 1, 2),
    )

  def test_known_host_mesh_shape_handles_edge_family_2d_coords(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index
        self.device_kind = "TPU v6e"

    fake_devices = [
        FakeDevice((0, 0), 0),
        FakeDevice((1, 0), 0),
        FakeDevice((0, 1), 0),
        FakeDevice((1, 1), 0),
    ]

    self.assertEqual(mesh.known_host_mesh_shape(fake_devices), (2, 2, 1))

  def test_known_host_mesh_shape_handles_edge_family_3d_coords_with_core_axis(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index
        self.core_on_chip = 0
        self.device_kind = "TPU v5 lite"

    fake_devices = []
    for y in range(4):
      for x in range(2):
        fake_devices.append(FakeDevice((x, y, 0), 0))

    self.assertEqual(mesh.known_host_mesh_shape(fake_devices), (2, 4, 1, 1))

  def test_allocate_devices_by_coords_supports_edge_family_3d_coords_with_trailing_singleton(self):
    class FakeDevice:

      def __init__(self, device_id, coords, process_index):
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.core_on_chip = 0
        self.device_kind = "TPU v5 lite"

    fake_devices = []
    device_id = 0
    for y in range(4):
      for x in range(4):
        fake_devices.append(FakeDevice(device_id, (x, y, 0), 0))
        device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 8)

    self.assertEqual([device.id for device in allocated], [0, 1, 4, 5, 8, 9, 12, 13])

  def test_resolve_per_host_mesh_shape_raises_on_mismatch(self):
    class FakeDevice:

      def __init__(self, device_id, coords, process_index):
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (1, 0, 0), 0),
        FakeDevice(2, (2, 0, 0), 0),
        FakeDevice(3, (3, 0, 0), 0),
        FakeDevice(4, (0, 0, 1), 1),
        FakeDevice(5, (1, 0, 1), 1),
        FakeDevice(6, (2, 0, 1), 1),
        FakeDevice(7, (3, 0, 1), 1),
    ]

    with self.assertRaisesRegex(ValueError, "Observed host devices do not match known host bounds"):
      mesh.resolve_per_host_mesh_shape(fake_devices)

  def test_allocate_named_mesh_device_slices_prefers_coord_boxes(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (0, 0, 0, 0)),
        FakeDevice(1, (0, 0, 0, 1)),
        FakeDevice(2, (1, 0, 0, 0)),
        FakeDevice(3, (1, 0, 0, 1)),
        FakeDevice(4, (2, 0, 0, 0)),
        FakeDevice(5, (2, 0, 0, 1)),
        FakeDevice(6, (3, 0, 0, 0)),
        FakeDevice(7, (3, 0, 0, 1)),
        FakeDevice(8, (0, 1, 0, 0)),
        FakeDevice(9, (0, 1, 0, 1)),
        FakeDevice(10, (1, 1, 0, 0)),
        FakeDevice(11, (1, 1, 0, 1)),
        FakeDevice(12, (2, 1, 0, 0)),
        FakeDevice(13, (2, 1, 0, 1)),
        FakeDevice(14, (3, 1, 0, 0)),
        FakeDevice(15, (3, 1, 0, 1)),
    ]

    allocated = mesh.allocate_named_mesh_device_slices(
        [("actor", 8)],
        devices=fake_devices,
    )

    self.assertEqual(
        [device.id for device in allocated["actor"]],
      [0, 1, 8, 9, 2, 3, 10, 11],
    )

  def test_allocate_devices_by_coords_uses_core_on_chip_dimension(self):
    class FakeDevice:

      def __init__(self, device_id, coords, core_on_chip):
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(4):
      for y in range(4):
        for z in range(2):
          for core_on_chip in (0, 1):
            fake_devices.append(FakeDevice(device_id, (x, y, z), core_on_chip))
            device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 8)

    self.assertEqual(
        [device.id for device in allocated],
        [0, 1, 4, 5, 16, 17, 20, 21],
    )

  def test_allocate_devices_by_coords_returns_none_without_coord_topology(self):
    class FakeDevice:

      def __init__(self, process_index):
        self.process_index = process_index

    self.assertEqual(
      mesh._allocate_devices_by_coords([FakeDevice(0), FakeDevice(0)], 2),
      (None, None),
    )

  def test_allocate_devices_by_coords_returns_best_contiguous_box(self):
    class FakeDevice:

      def __init__(self, device_id, coords, process_index):
        self.id = device_id
        self.coords = coords
        self.process_index = process_index

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (1, 0, 0), 0),
        FakeDevice(2, (0, 1, 0), 0),
        FakeDevice(3, (1, 1, 0), 0),
        FakeDevice(4, (2, 0, 0), 1),
        FakeDevice(5, (3, 0, 0), 1),
        FakeDevice(6, (2, 1, 0), 1),
        FakeDevice(7, (3, 1, 0), 1),
    ]

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 4)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_by_coords_prefers_more_cubical_supported_shape(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(4):
      for y in range(8):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 256)

    allocated_coords = [device.coords for device in allocated]
    mins = tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3))
    maxs = tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3))

    self.assertLen(allocated, 256)
    self.assertEqual(mins, (0, 0, 0))
    self.assertEqual(maxs, (3, 7, 7))

  def test_allocate_devices_tracks_remaining_coord_regions(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    allocated, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )

    allocated_coords = [device.coords for device in allocated]

    self.assertEqual(
        next_state.remaining_coord_regions_by_slice,
        {
            None: (
                mesh.CoordRegion((0, 0, 8), (4, 4, 8)),
                mesh.CoordRegion((0, 4, 0), (4, 12, 16)),
                mesh.CoordRegion((4, 0, 0), (12, 16, 16)),
            )
        },
    )
    self.assertEqual(
        (
            tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3)),
            tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3)),
        ),
        ((0, 0, 0), (3, 3, 7)),
    )

  def test_allocate_devices_prefers_smallest_remaining_coord_region_first(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )
    allocated = mesh.allocate_devices(128, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3)),
            tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3)),
        ),
        ((0, 0, 8), (3, 3, 15)),
    )

  def test_allocate_devices_matches_required_count_to_smallest_fitting_remaining_region(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )
    allocated = mesh.allocate_devices(576, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3)),
            tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3)),
        ),
        ((0, 4, 0), (3, 15, 11)),
    )

  def test_allocate_devices_performance_policy_prefers_more_cubical_shape(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        allocation_policy="PERFORMANCE",
        return_state=True,
    )
    allocated = mesh.allocate_devices(512, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3)),
            tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3)),
        ),
        ((4, 0, 0), (11, 7, 7)),
    )

  def test_allocate_devices_compact_policy_prefers_smallest_fitting_region(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        allocation_policy="COMPACT",
        return_state=True,
    )
    allocated = mesh.allocate_devices(512, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(min(coords[dim] for coords in allocated_coords) for dim in range(3)),
            tuple(max(coords[dim] for coords in allocated_coords) for dim in range(3)),
        ),
        ((0, 4, 0), (3, 11, 15)),
    )

  def test_allocate_devices_rejects_mismatched_policy_for_existing_state(self):
    fake_devices = [object(), object()]

    with mock.patch.object(mesh, "_allocate_devices_by_coords", return_value=([fake_devices[0]], None)):
      _, state = mesh.allocate_devices(
          1,
          devices=fake_devices,
          allocation_policy="COMPACT",
          return_state=True,
      )

    with self.assertRaisesRegex(
        ValueError,
        "allocation_policy must match allocation_state.allocation_policy",
    ):
      mesh.allocate_devices(
          1,
          allocation_state=state,
          allocation_policy="PERFORMANCE",
      )

  def test_allocate_devices_allocates_single_mesh(self):
    fake_devices = [object(), object()]

    with mock.patch.object(
        mesh,
        "_allocate_devices_by_coords",
        return_value=(fake_devices, None),
    ) as allocate_mock:
      allocated = mesh.allocate_devices(2, devices=fake_devices)

    allocate_mock.assert_called_once_with(
        fake_devices,
        2,
      None,
      "COMPACT",
    )
    self.assertIs(allocated, fake_devices)

  def test_allocate_devices_prefers_coords_over_host_groups(self):
    class FakeDevice:

      def __init__(self, device_id, coords, process_index):
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (1, 0, 0), 0),
        FakeDevice(2, (0, 1, 0), 0),
        FakeDevice(3, (1, 1, 0), 0),
        FakeDevice(4, (2, 0, 0), 1),
        FakeDevice(5, (3, 0, 0), 1),
        FakeDevice(6, (2, 1, 0), 1),
        FakeDevice(7, (3, 1, 0), 1),
    ]

    allocated = mesh.allocate_devices(4, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 2, 1, 3])

  def test_allocate_devices_returns_updated_state_for_incremental_use(self):
    fake_devices = [object(), object(), object()]

    with mock.patch.object(
        mesh,
        "_allocate_devices_by_coords",
        side_effect=[(fake_devices[:1], None), (fake_devices[1:], None)],
    ):
      assigned_devices, next_state = mesh.allocate_devices(
          1,
          devices=fake_devices,
          return_state=True,
      )
      remaining_devices = list(next_state.remaining_devices)
      assigned_devices_2, final_state = mesh.allocate_devices(
          2,
          allocation_state=next_state,
          return_state=True,
      )

    self.assertEqual(assigned_devices, fake_devices[:1])
    self.assertEqual(remaining_devices, fake_devices[1:])
    self.assertEqual(assigned_devices_2, fake_devices[1:])
    self.assertEqual(list(final_state.remaining_devices), [])
    self.assertEqual(final_state.used_device_count, 3)

  def test_allocate_devices_raises_when_incremental_state_is_exhausted(self):
    allocation_state = mesh.DeviceAllocationState(
        remaining_devices=(),
        full_devices_per_host=0,
        host_bound_shape=None,
        total_device_count=0,
    )

    with self.assertRaisesRegex(ValueError, "but only 0 remain available"):
      mesh.allocate_devices(1, allocation_state=allocation_state)

  def test_allocate_devices_rejects_devices_and_state_together(self):
    fake_devices = [object()]
    allocation_state = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        full_devices_per_host=0,
        host_bound_shape=None,
        total_device_count=1,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Pass either devices or allocation_state to allocate_devices, not both",
    ):
      mesh.allocate_devices(
          1,
          devices=fake_devices,
          allocation_state=allocation_state,
      )

  def test_allocate_devices_prefers_single_slice_before_cross_slice(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, coords):
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
      FakeDevice(1, 0, (1, 0, 0)),
      FakeDevice(2, 0, (2, 0, 0)),
      FakeDevice(3, 0, (3, 0, 0)),
      FakeDevice(4, 1, (4, 0, 0)),
      FakeDevice(5, 1, (5, 0, 0)),
      FakeDevice(6, 1, (6, 0, 0)),
      FakeDevice(7, 1, (7, 0, 0)),
    ]

    allocated = mesh.allocate_devices(4, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_prefers_single_slice_before_other_slices(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, process_index, coords):
        self.id = device_id
        self.slice_index = slice_index
        self.process_index = process_index
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = [
      FakeDevice(0, 0, 0, (0, 0, 0)),
      FakeDevice(1, 0, 0, (1, 0, 0)),
      FakeDevice(2, 0, 0, (0, 1, 0)),
      FakeDevice(3, 0, 0, (1, 1, 0)),
      FakeDevice(4, 1, 1, (2, 0, 0)),
      FakeDevice(5, 1, 1, (3, 0, 0)),
      FakeDevice(6, 1, 1, (2, 1, 0)),
      FakeDevice(7, 1, 1, (3, 1, 0)),
    ]

    allocated = mesh.allocate_devices(2, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1])

  def test_allocate_devices_raises_when_cross_slice_request_needs_partial_slice(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, coords):
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (2, 0, 0)),
        FakeDevice(2, 0, (4, 0, 0)),
        FakeDevice(3, 0, (6, 0, 0)),
        FakeDevice(4, 1, (1, 0, 0)),
        FakeDevice(5, 1, (3, 0, 0)),
        FakeDevice(6, 1, (5, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "cross-slice allocation only supports whole slices",
    ):
      mesh.allocate_devices(6, devices=fake_devices)

  def test_allocate_devices_consumes_whole_slices_in_order_when_cross_slice(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, coords):
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (2, 0, 0)),
        FakeDevice(2, 0, (4, 0, 0)),
        FakeDevice(3, 0, (6, 0, 0)),
        FakeDevice(4, 1, (1, 0, 0)),
        FakeDevice(5, 1, (3, 0, 0)),
        FakeDevice(6, 1, (5, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    allocated = mesh.allocate_devices(8, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3, 4, 5, 6, 7])

  def test_allocate_devices_skips_partial_slice_for_cross_slice_request(self):
    class FakeDevice:

      def __init__(self, device_id, slice_index, coords):
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
      FakeDevice(1, 0, (1, 0, 0)),
      FakeDevice(2, 0, (2, 0, 0)),
      FakeDevice(3, 0, (3, 0, 0)),
      FakeDevice(4, 1, (4, 0, 0)),
      FakeDevice(5, 1, (5, 0, 0)),
      FakeDevice(6, 1, (6, 0, 0)),
      FakeDevice(7, 1, (7, 0, 0)),
        FakeDevice(8, 2, (8, 0, 0)),
      FakeDevice(9, 2, (9, 0, 0)),
      FakeDevice(10, 2, (10, 0, 0)),
      FakeDevice(11, 2, (11, 0, 0)),
    ]

    assigned_devices, next_state = mesh.allocate_devices(
        2,
        devices=fake_devices,
        return_state=True,
    )
    self.assertEqual([device.id for device in assigned_devices], [0, 1])

    allocated = mesh.allocate_devices(8, allocation_state=next_state)

    self.assertEqual([device.id for device in allocated], [4, 5, 6, 7, 8, 9, 10, 11])

  def test_allocate_named_mesh_device_slices_calls_allocate_devices_in_loop(self):
    fake_devices = [object(), object(), object()]
    state_0 = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        full_devices_per_host=0,
        host_bound_shape=None,
        total_device_count=3,
        used_device_count=0,
    )
    state_1 = mesh.DeviceAllocationState(
      remaining_devices=tuple(fake_devices[1:]),
      full_devices_per_host=0,
      host_bound_shape=None,
      total_device_count=3,
      used_device_count=1,
    )
    state_2 = mesh.DeviceAllocationState(
      remaining_devices=(),
      full_devices_per_host=0,
      host_bound_shape=None,
      total_device_count=3,
      used_device_count=3,
    )

    with mock.patch.object(
      mesh,
      "allocate_devices",
      side_effect=[
          ([fake_devices[0]], state_1),
          ([fake_devices[1], fake_devices[2]], state_2),
      ],
    ) as allocate_mock, mock.patch.object(
      mesh,
      "_create_device_allocation_state",
        return_value=state_0,
    ) as state_mock, mock.patch.object(
      mesh.logging,
      "warning",
    ) as warning_mock:
      allocated = mesh.allocate_named_mesh_device_slices(
        [("mesh1", 1), ("mesh2", 2)],
        devices=fake_devices,
      )

    state_mock.assert_called_once_with(
      fake_devices,
      allocation_policy="COMPACT",
    )
    self.assertEqual(allocate_mock.call_count, 2)
    self.assertEqual(
      allocate_mock.call_args_list,
      [
        mock.call(
          1,
          mesh_name="mesh1",
            allocation_state=state_0,
          return_state=True,
        ),
        mock.call(
          2,
          mesh_name="mesh2",
            allocation_state=state_1,
          return_state=True,
        ),
      ],
    )
    warning_mock.assert_not_called()
    self.assertEqual(
      allocated,
      {"mesh1": [fake_devices[0]], "mesh2": [fake_devices[1], fake_devices[2]]},
    )

  @mock.patch.object(jax, "device_count")
  def test_create_mesh_uses_jax_make_mesh_without_assigned_devices(
      self, mock_device_count_fn
  ):
    mock_device_count_fn.return_value = 4
    expected_mesh = object()

    with mock.patch.object(jax, "make_mesh", return_value=expected_mesh) as make_mesh_mock:
      created_mesh = mesh.create_mesh((2, 2), ("x", "y"))

    make_mesh_mock.assert_called_once_with(
      (2, 2),
      ("x", "y"),
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    self.assertIs(created_mesh, expected_mesh)

  def test_create_mesh_uses_assigned_devices(self):
    assigned_devices = ["d0", "d1", "d2", "d3"]

    class FakeMesh:

      def __init__(self, devices, axis_names, axis_types=None):
        self.devices = devices
        self.axis_names = axis_names
        self.axis_types = axis_types

    with mock.patch.object(jax.sharding, "Mesh", side_effect=FakeMesh):
      created_mesh = mesh.create_mesh(
          (2, 2),
          ("x", "y"),
          devices=assigned_devices,
      )

    self.assertEqual(created_mesh.devices.shape, (2, 2))
    self.assertEqual(
        created_mesh.devices.flatten().tolist(),
        assigned_devices,
    )
    self.assertEqual(created_mesh.axis_names, ("x", "y"))

  def test_allocate_named_mesh_device_slices_uses_jax_devices_by_default(self):
    class FakeDevice:

      def __init__(self, device_id):
        self.id = device_id
        self.coords = (device_id, 0, 0)

    fake_devices = [FakeDevice(0), FakeDevice(1)]

    with mock.patch.object(mesh.jax, "devices", return_value=fake_devices):
      allocated = mesh.allocate_named_mesh_device_slices([("trainer", 2)])

    self.assertEqual([device.id for device in allocated["trainer"]], [0, 1])

  def test_allocate_named_mesh_device_slices_raises_when_coord_allocation_fails(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 0, (0, 1, 0)),
        FakeDevice(3, 0, (1, 1, 0)),
        FakeDevice(4, 1, (0, 0, 1)),
        FakeDevice(5, 1, (1, 0, 1)),
        FakeDevice(6, 1, (0, 1, 1)),
        FakeDevice(7, 1, (1, 1, 1)),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "coord-based allocation could not construct a valid box",
    ):
      with mock.patch.object(mesh, "_allocate_devices_by_coords", return_value=(None, None)):
        mesh.allocate_named_mesh_device_slices(
            [("trainer", 4), ("rollout", 4)],
            devices=fake_devices,
        )

  def test_allocate_named_mesh_device_slices_raises_when_not_enough_devices(self):
    class FakeDevice:

      def __init__(self, device_id):
        self.id = device_id

    fake_devices = [FakeDevice(0), FakeDevice(1)]

    with self.assertRaisesRegex(ValueError, "but only 2 remain available"):
      mesh.allocate_named_mesh_device_slices(
          [("trainer", 3)],
          devices=fake_devices,
      )


if __name__ == "__main__":
  absltest.main()
