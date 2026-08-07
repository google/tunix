"""CPU-only controls for the Pathways Mosaic compatibility gate."""

from __future__ import annotations

import unittest

import probe_mosaic_compat as target


class MosaicCompatibilityContractTest(unittest.TestCase):
    def test_version_mismatch_is_extracted(self):
        message = "Unsupported version: expected <= 13 but got 15"
        self.assertEqual(target._version_mismatch(message), (13, 15))
        self.assertIn("server_max=13", target._compact_error(RuntimeError(message)))
        self.assertIn("client_module=15", target._compact_error(RuntimeError(message)))

    def test_unknown_error_is_compact_and_single_line(self):
        actual = target._compact_error(RuntimeError("first line\nsecond line"))
        self.assertEqual(actual, "first line")

    def test_admission_requires_exact_shape_and_finite_values(self):
        self.assertTrue(target._admitted((8, 4096), 4096, True))
        self.assertFalse(target._admitted((7, 4096), 4096, True))
        self.assertFalse(target._admitted((8, 4096), 4096, False))

    def test_version_contract_requires_matching_client_and_release(self):
        versions = {"jax": "0.10.2", "jaxlib": "0.10.2"}
        self.assertEqual(
            target._version_contract(
                versions, "0.10.2", "20260730-jax_0.10.2"
            ),
            (True, "ok"),
        )
        self.assertEqual(
            target._version_contract(
                {"jax": "0.10.2", "jaxlib": "0.9.1"},
                "0.10.2",
                "20260730-jax_0.10.2",
            ),
            (False, "client-version-contract"),
        )
        self.assertEqual(
            target._version_contract(versions, "0.10.2", "20260730-jax_0.9.1"),
            (False, "pathways-release-contract"),
        )


if __name__ == "__main__":
    unittest.main()
