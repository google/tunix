"""CPU-only fail-stop controls for the unified T1 runner."""

from __future__ import annotations

import types
import unittest

import unified_runner as target


class UnifiedRunnerTest(unittest.TestCase):
    def _run(self, probes, modules, *, num_devices=64):
        imported = []
        output = []
        errors = []

        def importer(name):
            imported.append(name)
            value = modules[name]
            if isinstance(value, BaseException):
                raise value
            return value

        return_code = target._run_probe_sequence(
            probes,
            num_devices=num_devices,
            importer=importer,
            emit=output.append,
            emit_error=errors.append,
            print_traceback=lambda: None,
        )
        return return_code, imported, output, errors

    def test_success_preserves_declared_order(self):
        probes = (
            target.Probe("P2", "mesh", "p2"),
            target.Probe("P3", "bucket", "p3"),
            target.Probe("P1", "numeric", "p1"),
        )
        modules = {
            name: types.SimpleNamespace(main=lambda: 0) for name in ("p1", "p2", "p3")
        }
        rc, imported, output, errors = self._run(probes, modules)
        self.assertEqual(rc, 0)
        self.assertEqual(imported, ["p2", "p3", "p1"])
        self.assertFalse(any("SKIP_TAINTED" in line for line in output))
        self.assertEqual(errors, [])

    def test_nonzero_stops_and_taints_every_later_probe(self):
        probes = (
            target.Probe("P2", "mesh", "p2"),
            target.Probe("P1", "numeric", "p1"),
            target.Probe("H1", "legacy", "h1", False),
            target.Probe("H2", "legacy", "h2", False),
        )
        modules = {
            "p2": types.SimpleNamespace(main=lambda: 0),
            "p1": types.SimpleNamespace(main=lambda: 7),
            "h1": types.SimpleNamespace(),
            "h2": types.SimpleNamespace(),
        }
        rc, imported, output, errors = self._run(probes, modules)
        self.assertEqual(rc, 1)
        self.assertEqual(imported, ["p2", "p1"])
        self.assertIn(
            "[t1.unified] SKIP_TAINTED after=P1 skipped=H1,H2", output
        )
        self.assertEqual(errors, ["[t1.unified] FAIL probe=P1 exit=7"])

    def test_exception_stops_before_any_later_import(self):
        probes = (
            target.Probe("P1", "numeric", "p1"),
            target.Probe("H2", "legacy", "h2", False),
        )
        rc, imported, output, errors = self._run(
            probes,
            {"p1": RuntimeError("synthetic"), "h2": types.SimpleNamespace()},
        )
        self.assertEqual(rc, 1)
        self.assertEqual(imported, ["p1"])
        self.assertIn("[t1.unified] SKIP_TAINTED after=P1 skipped=H2", output)
        self.assertEqual(
            errors, ["[t1.unified] FAIL probe=P1 exception=RuntimeError"]
        )

    def test_subset_legacy_probe_is_not_applicable_on_full_slice(self):
        probes = (
            target.Probe("H1", "legacy", "h1", False, max_devices=4),
            target.Probe("H2", "portable", "h2", False),
        )
        rc, imported, output, errors = self._run(
            probes, {"h1": types.SimpleNamespace(), "h2": types.SimpleNamespace()}
        )
        self.assertEqual(rc, 0)
        self.assertEqual(imported, ["h2"])
        self.assertTrue(any("SKIP_NOT_APPLICABLE probe=H1" in line for line in output))
        self.assertEqual(errors, [])

    def test_overlay_missing_attribute_is_rejected(self):
        output = []
        errors = []
        ok = target._verify_overlay(
            importer=lambda _name: types.SimpleNamespace(),
            emit=output.append,
            emit_error=errors.append,
        )
        self.assertFalse(ok)
        self.assertEqual(output, [])
        self.assertEqual(len(errors), len(target.OVERLAY_CHECKS))


if __name__ == "__main__":
    unittest.main()
