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

  def test_device_attr_calls_callable_attributes(self):
    class FakeDevice:

      def coords(self):
        return (1, 2, 3)

    self.assertEqual(mesh.device_attr(FakeDevice(), "coords"), (1, 2, 3))
    self.assertIsNone(mesh.device_attr(FakeDevice(), "missing"))

  def test_device_host_key_prefers_slice_and_process_metadata(self):
    class FakeDevice:

      def __init__(self):
        self.slice_index = 4
        self.process_index = 7

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 7))

  def test_device_host_key_falls_back_to_slice_and_task_id(self):
    class FakeDevice:

      def __init__(self):
        self.slice = 3
        self.task_id = 9

    self.assertEqual(mesh.device_host_key(FakeDevice()), (3, 9))

  def test_device_host_key_prefers_logical_task_over_process_index(self):
    class FakeDevice:

      def __init__(self):
        self.slice_index = 4
        self.process_index = 0
        self.logical_task = 7

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 7))

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

  def test_device_slice_id_prefers_slice_index_then_slice(self):
    class SliceIndexDevice:

      def __init__(self):
        self.slice_index = 4

    class SliceDevice:

      def __init__(self):
        self.slice = 7

    self.assertEqual(mesh.device_slice_id(SliceIndexDevice()), 4)
    self.assertEqual(mesh.device_slice_id(SliceDevice()), 7)
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

  def test_find_candidate_coord_boxes_finds_contiguous_boxes(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (0, 0, 0)),
        FakeDevice(1, (1, 0, 0)),
        FakeDevice(2, (0, 1, 0)),
        FakeDevice(3, (1, 1, 0)),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertEqual(
        mesh.find_candidate_coord_boxes(topology, 4),
        [
            (
                (0, 0, 0),
                (2, 2, 1),
                ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
            )
        ],
    )

  def test_find_candidate_coord_boxes_skips_missing_coords(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (0, 0, 0)),
        FakeDevice(1, (1, 0, 0)),
        FakeDevice(2, (1, 1, 0)),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertEqual(mesh.find_candidate_coord_boxes(topology, 4), [])

  def test_find_candidate_coord_boxes_can_return_multiple_candidates(self):
    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (0, 0, 0)),
        FakeDevice(1, (1, 0, 0)),
        FakeDevice(2, (2, 0, 0)),
        FakeDevice(3, (3, 0, 0)),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertEqual(
        mesh.find_candidate_coord_boxes(topology, 2),
        [
            ((0, 0, 0), (2, 1, 1), ((0, 0, 0), (1, 0, 0))),
            ((1, 0, 0), (2, 1, 1), ((1, 0, 0), (2, 0, 0))),
            ((2, 0, 0), (2, 1, 1), ((2, 0, 0), (3, 0, 0))),
        ],
    )

  def test_find_candidate_coord_boxes_rejects_split_chip_candidates(self):
    class FakeDevice:

      def __init__(self, device_id, coords, core_on_chip):
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (0, 0, 0), 1),
        FakeDevice(2, (1, 0, 0), 0),
        FakeDevice(3, (1, 0, 0), 1),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertEqual(
      mesh.find_candidate_coord_boxes(topology, 1),
        [],
    )

  def test_find_host_aligned_candidate_coord_boxes_respects_exact_host_shape(self):
    class FakeDevice:

      def __init__(self, device_id, coords, core_on_chip):
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip

    fake_devices = []
    device_id = 0
    for x in range(4):
      for y in range(4):
        for z in range(2):
          for core_on_chip in (0, 1):
            fake_devices.append(FakeDevice(device_id, (x, y, z), core_on_chip))
            device_id += 1

    topology = mesh.get_coord_topology(fake_devices)

    candidate_boxes = mesh.find_host_aligned_candidate_coord_boxes(
      topology, 8, (2, 2, 1, 2)
    )

    self.assertLen(candidate_boxes, 8)
    self.assertContainsSubset(
      [
        (
          (0, 0, 0, 0),
          (2, 2, 1, 2),
          (
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
          ),
        ),
        (
          (0, 0, 1, 0),
          (2, 2, 1, 2),
          (
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (1, 0, 1, 0),
            (1, 0, 1, 1),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
          ),
        ),
      ],
      candidate_boxes,
    )

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

  def test_allocate_named_mesh_device_slices_uses_logical_task_host_groups(self):
    class FakeDevice:

      def __init__(self, device_id, logical_task, coords):
        self.id = device_id
        self.process_index = 0
        self.logical_task = logical_task
        self.coords = coords

    fake_devices = []
    for device_id in range(16):
      host_index = device_id // 2
      fake_devices.append(
          FakeDevice(
              device_id,
              device_id % 2,
              (host_index % 2, (host_index // 2) % 2, host_index // 4),
          )
      )

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("actor", 8)],
        devices=fake_devices,
      )

    self.assertEqual(
        [device.id for device in allocated["actor"]],
        [0, 2, 4, 6, 8, 10, 12, 14],
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

  def test_host_mesh_shape_infers_consistent_per_host_shape(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index

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

    self.assertEqual(mesh.host_mesh_shape(fake_devices), (2, 2, 1))

  def test_host_mesh_shape_returns_none_for_sparse_host_box(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((1, 0, 0), 0),
        FakeDevice((1, 1, 0), 0),
    ]

    self.assertIsNone(mesh.host_mesh_shape(fake_devices))

  def test_host_mesh_shape_returns_none_for_inconsistent_host_shapes(self):
    class FakeDevice:

      def __init__(self, coords, process_index):
        self.coords = coords
        self.process_index = process_index

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((1, 0, 0), 0),
        FakeDevice((0, 1, 0), 0),
        FakeDevice((1, 1, 0), 0),
        FakeDevice((2, 0, 0), 1),
        FakeDevice((3, 0, 0), 1),
    ]

    self.assertIsNone(mesh.host_mesh_shape(fake_devices))

  def test_divisors_returns_sorted_unique_factors(self):
    self.assertEqual(mesh._divisors(12), [1, 2, 3, 4, 6, 12])

  def test_enumerate_box_shapes_returns_shapes_with_requested_volume(self):
    self.assertEqual(
        mesh._enumerate_box_shapes(4, (4, 2, 2)),
        [(1, 2, 2), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    )

  def test_coord_box_score_prefers_host_aligned_boxes(self):
    aligned_score = mesh._coord_box_score((0, 0, 0), (2, 2, 1), (2, 2, 1))
    unaligned_score = mesh._coord_box_score((1, 0, 0), (2, 2, 1), (2, 2, 1))

    self.assertLess(aligned_score, unaligned_score)

  def test_select_best_candidate_coords_prefers_host_aligned_box(self):
    candidate_boxes = [
        ((1, 0, 0), (2, 2, 1), ((1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0))),
        ((0, 0, 0), (2, 2, 1), ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))),
    ]

    self.assertEqual(
        mesh.select_best_candidate_coords(candidate_boxes, (2, 2, 1)),
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
    )

  def test_select_best_candidate_coords_prefers_chip_host_aligned_box_with_core_dimension(self):
    candidate_boxes = [
        (
            (0, 0, 0, 0),
            (1, 2, 2, 2),
            (
                (0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 1, 0, 0),
                (0, 1, 0, 1),
                (0, 0, 1, 0),
                (0, 0, 1, 1),
                (0, 1, 1, 0),
                (0, 1, 1, 1),
            ),
        ),
        (
            (0, 0, 0, 0),
            (2, 2, 1, 2),
            (
                (0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 1, 0, 0),
                (0, 1, 0, 1),
                (1, 0, 0, 0),
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (1, 1, 0, 1),
            ),
        ),
    ]

    self.assertEqual(
        mesh.select_best_candidate_coords(candidate_boxes, (2, 2, 1, 2)),
        [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
        ],
    )

  def test_select_best_candidate_coords_prefers_more_compact_shape(self):
    candidate_boxes = [
        ((0, 0, 0), (1, 4, 1), ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0))),
        ((0, 0, 0), (2, 2, 1), ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))),
    ]

    self.assertEqual(
        mesh.select_best_candidate_coords(candidate_boxes, None),
        [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
    )

  def test_select_best_candidate_coords_uses_start_as_tiebreaker(self):
    candidate_boxes = [
        ((2, 0, 0), (2, 1, 1), ((2, 0, 0), (3, 0, 0))),
        ((0, 0, 0), (2, 1, 1), ((0, 0, 0), (1, 0, 0))),
    ]

    self.assertEqual(
        mesh.select_best_candidate_coords(candidate_boxes, None),
        [(0, 0, 0), (1, 0, 0)],
    )

  def test_select_best_candidate_coords_returns_none_without_candidates(self):
    self.assertIsNone(mesh.select_best_candidate_coords([], (2, 2, 1)))

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

  def test_resolve_per_host_mesh_shape_raises_on_mismatch(self):
    class FakeDevice:

      def __init__(self, device_id, coords, logical_task):
        self.id = device_id
        self.coords = coords
        self.logical_task = logical_task
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

    with self.assertRaisesRegex(ValueError, "does not match known host bounds"):
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
        [0, 1, 2, 3, 8, 9, 10, 11],
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

    allocated = mesh.allocate_devices_by_coords(fake_devices, 8)

    self.assertEqual(
        [device.id for device in allocated],
        [0, 1, 4, 5, 16, 17, 20, 21],
    )

  def test_allocate_devices_by_coords_returns_none_without_coord_topology(self):
    class FakeDevice:

      def __init__(self, process_index):
        self.process_index = process_index

    self.assertIsNone(
        mesh.allocate_devices_by_coords([FakeDevice(0), FakeDevice(0)], 2)
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

    allocated = mesh.allocate_devices_by_coords(fake_devices, 4)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_allocates_single_mesh(self):
    fake_devices = [object(), object()]

    with mock.patch.object(
        mesh,
        "allocate_devices_by_coords",
        return_value=fake_devices,
    ) as allocate_mock:
      allocated = mesh.allocate_devices(2, devices=fake_devices)

    allocate_mock.assert_called_once_with(
        fake_devices,
        2,
    )
    self.assertIs(allocated, fake_devices)

  def test_allocate_devices_returns_updated_state_for_incremental_use(self):
    fake_devices = [object(), object(), object()]

    with mock.patch.object(
        mesh,
        "allocate_devices_by_coords",
        side_effect=[fake_devices[:1], fake_devices[1:]],
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

  def test_allocate_devices_rejects_devices_and_state_together(self):
    fake_devices = [object()]
    allocation_state = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        remaining_host_groups=None,
        full_devices_per_host=0,
        host_bound_shape=None,
        host_bound_device_count=None,
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
        FakeDevice(1, 0, (2, 0, 0)),
        FakeDevice(2, 0, (4, 0, 0)),
        FakeDevice(3, 0, (6, 0, 0)),
        FakeDevice(4, 1, (1, 0, 0)),
        FakeDevice(5, 1, (3, 0, 0)),
        FakeDevice(6, 1, (5, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    allocated = mesh.allocate_devices(4, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_spills_to_next_slice_in_order(self):
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

    allocated = mesh.allocate_devices(6, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3, 4, 5])

  def test_allocate_named_mesh_device_slices_calls_allocate_devices_in_loop(self):
    fake_devices = [object(), object(), object()]
    state_0 = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        remaining_host_groups=None,
        full_devices_per_host=0,
        host_bound_shape=None,
        host_bound_device_count=None,
        total_device_count=3,
        used_device_count=0,
    )
    state_1 = mesh.DeviceAllocationState(
      remaining_devices=tuple(fake_devices[1:]),
      remaining_host_groups=None,
      full_devices_per_host=0,
      host_bound_shape=None,
      host_bound_device_count=None,
      total_device_count=3,
      used_device_count=1,
    )
    state_2 = mesh.DeviceAllocationState(
      remaining_devices=(),
      remaining_host_groups=None,
      full_devices_per_host=0,
      host_bound_shape=None,
      host_bound_device_count=None,
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

    state_mock.assert_called_once_with(fake_devices)
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

    fake_devices = [FakeDevice(0), FakeDevice(1)]

    with mock.patch.object(mesh.jax, "devices", return_value=fake_devices):
      allocated = mesh.allocate_named_mesh_device_slices([("trainer", 2)])

    self.assertEqual([device.id for device in allocated["trainer"]], [0, 1])

  def test_allocate_named_mesh_device_slices_uses_whole_hosts(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 1, (0, 0, 1)),
        FakeDevice(3, 1, (1, 0, 1)),
    ]

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("trainer", 2), ("rollout", 2)],
        devices=fake_devices,
      )

    self.assertEqual([device.id for device in allocated["trainer"]], [0, 1])
    self.assertEqual([device.id for device in allocated["rollout"]], [2, 3])

  def test_allocate_named_mesh_device_slices_allows_multiple_single_host_subslices(self):
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
        FakeDevice(4, (0, 2, 0), 0),
        FakeDevice(5, (1, 2, 0), 0),
        FakeDevice(6, (0, 3, 0), 0),
        FakeDevice(7, (1, 3, 0), 0),
    ]

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("actor", 2), ("reference", 2), ("rollout", 2)],
        devices=fake_devices,
      )

    self.assertEqual([device.id for device in allocated["actor"]], [0, 1])
    self.assertEqual([device.id for device in allocated["reference"]], [2, 3])
    self.assertEqual([device.id for device in allocated["rollout"]], [4, 5])

  def test_allocate_named_mesh_device_slices_reuses_partial_host_leftovers(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords

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

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("mesh1", 3), ("mesh2", 2), ("mesh3", 2), ("mesh4", 1)],
        devices=fake_devices,
      )

    self.assertEqual([device.id for device in allocated["mesh1"]], [0, 1, 2])
    self.assertEqual([device.id for device in allocated["mesh2"]], [4, 5])
    self.assertEqual([device.id for device in allocated["mesh3"]], [6, 7])
    self.assertEqual([device.id for device in allocated["mesh4"]], [3])

  def test_allocate_named_mesh_device_slices_reuses_valid_leftover_host_group(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 0, (0, 1, 0)),
        FakeDevice(3, 0, (1, 1, 0)),
        FakeDevice(4, 0, (0, 0, 1)),
        FakeDevice(5, 0, (1, 0, 1)),
        FakeDevice(6, 0, (0, 1, 1)),
        FakeDevice(7, 0, (1, 1, 1)),
        FakeDevice(8, 1, (0, 0, 2)),
        FakeDevice(9, 1, (1, 0, 2)),
        FakeDevice(10, 1, (0, 1, 2)),
        FakeDevice(11, 1, (1, 1, 2)),
        FakeDevice(12, 1, (0, 0, 3)),
        FakeDevice(13, 1, (1, 0, 3)),
        FakeDevice(14, 1, (0, 1, 3)),
        FakeDevice(15, 1, (1, 1, 3)),
    ]

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("mesh1", 6), ("mesh2", 2)],
        devices=fake_devices,
      )

    self.assertEqual([device.id for device in allocated["mesh1"]], [0, 1, 2, 3, 4, 5])
    self.assertEqual([device.id for device in allocated["mesh2"]], [6, 7])

  def test_allocate_named_mesh_device_slices_allocates_full_host_then_remainder(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 1, (0, 0, 1)),
        FakeDevice(3, 1, (1, 0, 1)),
    ]

    with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
      allocated = mesh.allocate_named_mesh_device_slices(
        [("trainer", 3)],
        devices=fake_devices,
      )

    self.assertEqual([device.id for device in allocated["trainer"]], [0, 1, 2])

  def test_allocate_named_mesh_device_slices_raises_without_host_bound_metadata(self):
    class FakeDevice:

      def __init__(self, device_id, process_index):
        self.id = device_id
        self.process_index = process_index

    fake_devices = [
        FakeDevice(0, 0),
        FakeDevice(1, 0),
        FakeDevice(2, 1),
        FakeDevice(3, 1),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "Host-group allocation requires an inferable host-bound shape and device count",
    ):
      with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
        mesh.allocate_named_mesh_device_slices(
            [("trainer", 2)],
            devices=fake_devices,
        )

  def test_allocate_named_mesh_device_slices_raises_when_not_enough_hosts(self):
    class FakeDevice:

      def __init__(self, device_id, process_index, coords):
        self.id = device_id
        self.process_index = process_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 1, (0, 0, 1)),
        FakeDevice(3, 1, (1, 0, 1)),
    ]

    with self.assertRaisesRegex(ValueError, "but only 2 are available"):
      with mock.patch.object(mesh, "allocate_devices_by_coords", return_value=None):
        mesh.allocate_named_mesh_device_slices(
            [("trainer", 6)],
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
