"""CPU-only positive and negative controls for the full-slice way-count probe."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import probe_waycount as target


def _devices(count: int):
    return [types.SimpleNamespace(id=index, coords=(index, 0, 0)) for index in range(count)]


class WayCountContractTest(unittest.TestCase):
    def test_default_widths_are_bounded_and_divisible(self):
        self.assertEqual(target._default_widths(64), [2, 4, 8])
        self.assertEqual(target._default_widths(8), [2, 4, 8])
        self.assertEqual(target._default_widths(2), [2])
        self.assertEqual(target._default_widths(3), [])

    def test_schedule_rejects_invalid_and_duplicate_values(self):
        target._validate_schedule(64, [2, 4, 8], [8, 15])
        with self.assertRaisesRegex(ValueError, "exactly divide"):
            target._validate_schedule(64, [3], [8])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            target._validate_schedule(64, [1], [8])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            target._parse_int_list("2,2", name="widths")

    def test_attestation_accepts_exact_full_slice(self):
        devices = _devices(8)
        built = np.asarray(devices, dtype=object).reshape(2, 4)
        self.assertEqual(
            target._attest_full_slice(built, devices, 4),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )

    def test_attestation_rejects_duplicate_and_missing_devices(self):
        devices = _devices(8)
        duplicate = np.asarray(devices[:-1] + [devices[0]], dtype=object).reshape(2, 4)
        with self.assertRaisesRegex(ValueError, "repeats"):
            target._attest_full_slice(duplicate, devices, 4)

        foreign = _devices(8)
        foreign[-1] = types.SimpleNamespace(id=99, coords=(99, 0, 0))
        built = np.asarray(foreign, dtype=object).reshape(2, 4)
        with self.assertRaisesRegex(ValueError, "not full-slice"):
            target._attest_full_slice(built, devices, 4)

    def test_mesh_builder_uses_full_shape_and_no_prefix(self):
        devices = _devices(8)
        built = np.asarray(devices, dtype=object).reshape(2, 4)
        with mock.patch.object(
            target.mesh_utils, "create_device_mesh", return_value=built
        ) as create:
            actual = target._create_full_slice_mesh(devices, 4)
        self.assertEqual(actual.shape, (2, 4))
        create.assert_called_once_with(
            (2, 4), devices, allow_split_physical_axes=True
        )

    def test_metrics_and_measurement_count_have_negative_controls(self):
        left = np.asarray([1.0, 2.0], dtype=np.float32)
        same = left.copy()
        changed = np.asarray([1.0, 3.0], dtype=np.float32)
        self.assertEqual(target._differing_bytes(left, same), 0)
        self.assertGreater(target._differing_bytes(left, changed), 0)
        self.assertEqual(target._error_metrics(left, same), (0.0, 0.0, 0.0))
        rel_l2, one_minus_cos, max_abs = target._error_metrics(left, changed)
        self.assertGreater(rel_l2, 0.0)
        self.assertGreater(one_minus_cos, 0.0)
        self.assertEqual(max_abs, 1.0)
        self.assertEqual(target._expected_measurements([2, 4, 8], [8, 15]), 18)
        self.assertTrue(target._measurements_complete(18, 18))
        self.assertFalse(target._measurements_complete(17, 18))


if __name__ == "__main__":
    unittest.main()
