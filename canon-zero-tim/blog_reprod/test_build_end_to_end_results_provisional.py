#!/usr/bin/env python3
"""Host tests for the drawing library imported by the measured Figure 4 builder.

Run from this directory:
    python -m unittest test_build_end_to_end_results_provisional -v
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "build_end_to_end_results_provisional.py"
SPEC = importlib.util.spec_from_file_location("build_end_to_end_results_provisional", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProvisionalFigureTest(unittest.TestCase):
    def _history(
        self,
        root: Path,
        name: str,
        rewards: list[float],
        logp_diff: list[float],
    ) -> Path:
        path = root / name
        fields = ["_step", MODULE.REWARD_KEY, MODULE.LOGP_DIFF_KEY]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step, (reward, diff) in enumerate(zip(rewards, logp_diff)):
                writer.writerow(
                    {
                        "_step": step,
                        MODULE.REWARD_KEY: reward,
                        MODULE.LOGP_DIFF_KEY: diff,
                    }
                )
        return path

    def _load(self, root: Path, *, zero_values, assume_zero: bool):
        native = self._history(root, "native.csv", [0.2, 0.3, 0.4], [0.02] * 3)
        importance_sampling = self._history(
            root, "is.csv", [0.2, 0.35, 0.45], [0.015] * 3
        )
        zero_tim = self._history(root, "zero.csv", [0.2, 0.4, 0.5], zero_values)
        return MODULE.load_workload(
            title="Five-turn FrozenLake",
            subtitle="Up to 5 turns",
            native=native,
            importance_sampling=importance_sampling,
            zero_tim=zero_tim,
            axis_max=100,
            assume_certified_zero=assume_zero,
        )

    def test_observed_mode_preserves_nonzero_candidate_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._load(Path(tmp), zero_values=[0.0, 0.001, 0.0], assume_zero=False)
            self.assertEqual(
                [point.value for point in run.logp_diff["zero_tim"]],
                [0.0, 0.001, 0.0],
            )

    def test_assumed_zero_is_explicit_and_affects_only_zero_tim_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._load(Path(tmp), zero_values=[0.0, 0.001, 0.0], assume_zero=True)
            self.assertTrue(all(point.value == 0 for point in run.logp_diff["zero_tim"]))
            self.assertTrue(any(point.value > 0 for point in run.logp_diff["native"]))
            self.assertEqual(run.solve_rate["zero_tim"][-1].value, 0.5)
            svg = MODULE.render_svg(run, assumed_zero=True)
            ET.fromstring(svg)
            self.assertIn("do not publish before strict receipt replacement", svg)
            self.assertNotIn("PROVISIONAL DIAGNOSTIC", svg)

    def test_short_zero_tim_run_stops_all_curves_without_shrinking_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = self._history(root, "native.csv", [0.1] * 8, [0.02] * 8)
            importance_sampling = self._history(root, "is.csv", [0.2] * 8, [0.01] * 8)
            zero_tim = self._history(root, "zero.csv", [0.3] * 3, [0.0] * 3)
            run = MODULE.load_workload(
                title="Fifteen-turn FrozenLake",
                subtitle="Up to 15 turns",
                native=native,
                importance_sampling=importance_sampling,
                zero_tim=zero_tim,
                axis_max=100,
                assume_certified_zero=True,
            )
            self.assertEqual(run.axis_max, 100)
            self.assertEqual(run.observed_updates, 3)
            self.assertTrue(all(len(points) == 3 for points in run.solve_rate.values()))

    def test_negative_mean_absolute_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "is negative"):
                self._load(Path(tmp), zero_values=[0.0, -0.1, 0.0], assume_zero=False)


if __name__ == "__main__":
    unittest.main()
