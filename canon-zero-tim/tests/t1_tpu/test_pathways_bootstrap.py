"""CPU-only positive and negative controls for the T1 Pathways bootstrap."""

import types
import unittest

from pathways_bootstrap import initialize_pathways, pathways_required


class PathwaysBootstrapTest(unittest.TestCase):
    def test_required_detection(self):
        self.assertTrue(pathways_required({"JAX_PLATFORMS": "proxy,cpu"}))
        self.assertTrue(pathways_required({"PATHWAYS_HEAD": "coordinator:29000"}))
        self.assertFalse(pathways_required({"JAX_PLATFORMS": "cpu"}))
        self.assertFalse(
            pathways_required(
                {"JAX_PLATFORMS": "proxy", "CANON_REQUIRE_PATHWAYS": "0"}
            )
        )
        with self.assertRaises(ValueError):
            pathways_required({"CANON_REQUIRE_PATHWAYS": "sometimes"})

    def test_proxy_missing_module_is_rejected(self):
        markers = []

        def missing(_name):
            raise ModuleNotFoundError("synthetic")

        with self.assertRaises(RuntimeError):
            initialize_pathways(
                environ={"JAX_PLATFORMS": "proxy"},
                argv=[],
                importer=missing,
                emit=markers.append,
            )
        self.assertEqual(
            markers,
            ["[T1.PATHWAYS] required=1 initialized=0 status=import-ModuleNotFoundError"],
        )

    def test_proxy_initialization_failure_is_rejected(self):
        markers = []
        module = types.SimpleNamespace(initialize=lambda: (_ for _ in ()).throw(OSError()))
        with self.assertRaises(RuntimeError):
            initialize_pathways(
                environ={"PATHWAYS_HEAD": "coordinator:29000"},
                argv=[],
                importer=lambda _name: module,
                emit=markers.append,
            )
        self.assertEqual(
            markers,
            ["[T1.PATHWAYS] required=1 initialized=0 status=initialize-OSError"],
        )

    def test_direct_attached_missing_module_is_allowed(self):
        markers = []
        args = []

        def missing(_name):
            raise ModuleNotFoundError("synthetic")

        self.assertFalse(
            initialize_pathways(
                environ={"JAX_PLATFORMS": "cpu"},
                argv=args,
                importer=missing,
                emit=markers.append,
            )
        )
        self.assertEqual(len(args), 2)
        self.assertEqual(
            markers,
            ["[T1.PATHWAYS] required=0 initialized=0 status=import-ModuleNotFoundError"],
        )

    def test_success_is_marked_and_flag_is_idempotent(self):
        markers = []
        calls = []
        args = []
        env = {"JAX_PLATFORMS": "proxy"}
        module = types.SimpleNamespace(initialize=lambda: calls.append("initialize"))
        self.assertTrue(
            initialize_pathways(
                environ=env,
                argv=args,
                importer=lambda _name: module,
                emit=markers.append,
            )
        )
        self.assertEqual(calls, ["initialize"])
        self.assertEqual(
            args.count("--FLAGS_pathways_enforce_subset_devices_form_subslice=false"), 1
        )
        self.assertEqual(
            args.count("--pathways_enforce_subset_devices_form_subslice=false"), 1
        )
        self.assertEqual(
            env["FLAGS_pathways_enforce_subset_devices_form_subslice"], "false"
        )
        self.assertEqual(
            env["PATHWAYS_ENFORCE_SUBSET_DEVICES_FORM_SUBSLICE"], "false"
        )
        self.assertEqual(
            markers, ["[T1.PATHWAYS] required=1 initialized=1 status=ok"]
        )


if __name__ == "__main__":
    unittest.main()
