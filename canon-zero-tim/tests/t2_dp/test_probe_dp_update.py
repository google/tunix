"""CPU-only topology controls for the P32 DP update probe."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np

import probe_dp_update as target


@dataclass(frozen=True)
class FakeDevice:
  id: int


def _devices(count: int):
  return [FakeDevice(index) for index in range(count)]


class TopologyMeshTest(unittest.TestCase):

  def test_full_slice_topology_mesh_is_accepted(self):
    devices = _devices(8)
    arranged = np.asarray(devices, dtype=object).reshape(2, 4)
    with mock.patch.object(
        target.mesh_utils, "create_device_mesh", return_value=arranged
    ) as create:
      mesh = target._topology_mesh(devices, 2, 4)
    self.assertEqual(mesh.devices.shape, (2, 4))
    create.assert_called_once_with(
        (2, 4), devices, allow_split_physical_axes=True
    )

  def test_duplicate_device_is_rejected_without_reshape_fallback(self):
    devices = _devices(8)
    arranged = np.asarray(devices[:-1] + [devices[0]], dtype=object).reshape(2, 4)
    with mock.patch.object(
        target.mesh_utils, "create_device_mesh", return_value=arranged
    ):
      with self.assertRaisesRegex(RuntimeError, "repeats"):
        target._topology_mesh(devices, 2, 4)

  def test_unsupported_topology_api_is_rejected(self):
    devices = _devices(8)
    with mock.patch.object(
        target.mesh_utils,
        "create_device_mesh",
        side_effect=TypeError("synthetic"),
    ):
      with self.assertRaisesRegex(RuntimeError, "refusing a logical reshape"):
        target._topology_mesh(devices, 2, 4)


if __name__ == "__main__":
  unittest.main()
