"""Fail-closed data checks for the measured, three-arm blog figure.

Run from this directory:
    python -m unittest test_end_to_end_results_measured -v
"""

import csv
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import build_end_to_end_results_measured as figure
from build_end_to_end_results_provisional import Point, moving_average


def history(*, count=200, mutation=None):
    rows = [{"_step": str(i), figure.REWARD: "0.5", figure.MEAN: "0.0",
             figure.MAXIMUM: "0.0", figure.BYTES: "0.0"} for i in range(count)]
    if mutation:
        mutation(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=figure.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


class MeasuredFigureTests(unittest.TestCase):
    def test_preserves_exact_source_strings(self):
        data = history(mutation=lambda rows: rows[42].update({figure.MEAN: "0.012345678912345"}))
        selected, _ = figure.select_rows(data, "standard")
        self.assertEqual(selected[42][figure.MEAN], "0.012345678912345")
        self.assertEqual(len(selected), 200)

    def test_window_is_first_200_not_all_available(self):
        selected, _ = figure.select_rows(history(count=216), "zero_tim")
        self.assertEqual(selected[-1]["_step"], "199")

    def test_missing_last_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete steps"):
            figure.select_rows(history(count=199), "zero_tim")

    def test_missing_internal_measurement_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            figure.select_rows(history(mutation=lambda rows: rows[10].update({figure.MEAN: ""})), "zero_tim")

    def test_duplicate_or_out_of_order_steps_rejected(self):
        for duplicate in (True, False):
            def mutate(rows):
                if duplicate:
                    rows[10]["_step"] = "9"
                else:
                    rows[10], rows[11] = rows[11], rows[10]
            with self.assertRaisesRegex(ValueError, "unique, ordered"):
                figure.select_rows(history(mutation=mutate), "standard")

    def test_nonzero_zero_tim_metrics_rejected_without_rounding(self):
        for key in (figure.MEAN, figure.MAXIMUM, figure.BYTES):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "measured nonzero"):
                figure.select_rows(history(mutation=lambda rows: rows[5].update({key: "1e-30"})), "zero_tim")

    def test_nonfinite_and_negative_values_rejected(self):
        for value in ("nan", "inf", "-0.001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                figure.select_rows(history(mutation=lambda rows: rows[0].update({figure.MEAN: value})), "standard")

    def test_standard_cannot_be_historical_rollout_denominator(self):
        config = {**figure.COMMON, "sampler_is": "none", "eval_every_n_steps": 50}
        with self.assertRaisesRegex(ValueError, "trainer-old"):
            figure.validate_config(config, "standard")
        config["old_logps_source"] = "trainer"
        figure.validate_config(config, "standard")

    def test_tis_correction_and_geometry_checked(self):
        config = {**figure.COMMON, "sampler_is": "none", "eval_every_n_steps": 50}
        with self.assertRaisesRegex(ValueError, "wrong sampler correction"):
            figure.validate_config(config, "importance_sampling")
        config["sampler_is"] = "token"
        config["mesh_dp"] = 16
        with self.assertRaisesRegex(ValueError, "mesh_dp"):
            figure.validate_config(config, "importance_sampling")

    def test_smoothing_is_trailing_not_centered(self):
        points = tuple(Point(i + 1, float(i)) for i in range(12))
        smooth = moving_average(points)
        self.assertEqual(smooth[0].value, 0)
        self.assertEqual(smooth[9].value, 4.5)
        self.assertEqual(smooth[10].value, 5.5)

    def test_axis_rejects_out_of_range_raw_values(self):
        rows, _ = figure.select_rows(history(), "standard")
        rows[0][figure.REWARD] = "0.2"
        with self.assertRaisesRegex(ValueError, "axis would omit"):
            figure.make_workload({arm: rows for arm in figure.RUNS})

    def test_cached_export_validation_and_endpoint_values(self):
        selected, manifest = figure.load_exported_data(figure.ROOT / "data")
        run = figure.make_workload(selected)
        endpoints = {entry["arm"]: entry for entry in figure.smoothed_endpoints(run)}
        expected = {"standard": (0.6265625, "62.7%"),
                    "importance_sampling": (0.66875, "66.9%"),
                    "zero_tim": (0.882421875, "88.2%")}
        for arm, (value, label) in expected.items():
            self.assertAlmostEqual(endpoints[arm]["value"], value)
            self.assertEqual(endpoints[arm]["text"], label)
            self.assertEqual(endpoints[arm]["step"], 200)
            self.assertEqual(endpoints[arm]["label_y"], endpoints[arm]["y"])
        svg = ET.fromstring(figure.render_svg(run, manifest))
        groups = svg.findall("{http://www.w3.org/2000/svg}g[@class='endpoint']")
        self.assertEqual(len(groups), 3)
        for group in groups:
            self.assertEqual(group.attrib["data-step"], "200")
            self.assertEqual([child.tag for child in group], ["{http://www.w3.org/2000/svg}text"])
            self.assertEqual(group[0].attrib["x"], "1212")
            self.assertEqual(group[0].text, expected[group.attrib["data-arm"]][1])
        self.assertEqual(svg.attrib["viewBox"], "0 0 1320 430")
        self.assertIn(".endpoint text { font-size: 12px; font-weight: 400; }",
                      svg.find("{http://www.w3.org/2000/svg}style").text)

    def test_endpoint_uses_average_not_last_raw_point(self):
        rows, _ = figure.select_rows(history(), "standard")
        rows[-1][figure.REWARD] = "0.9"
        run = figure.make_workload({arm: rows for arm in figure.RUNS})
        for entry in figure.smoothed_endpoints(run):
            self.assertAlmostEqual(entry["value"], .54)
            self.assertEqual(entry["text"], "54.0%")

    def test_endpoint_labels_do_not_overlap(self):
        rows, _ = figure.select_rows(history(), "standard")
        run = figure.make_workload({arm: rows for arm in figure.RUNS})
        endpoints = figure.smoothed_endpoints(run)
        for previous, current in zip(endpoints, endpoints[1:]):
            self.assertGreaterEqual(current["label_y"] - previous["label_y"], 12)
        self.assertEqual(len({entry["y"] for entry in endpoints}), 1)

    def test_modified_cached_csv_is_rejected(self):
        data = figure.ROOT / "data"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "manifest.json").write_bytes((data / "manifest.json").read_bytes())
            (temporary / "standard.csv").write_bytes(history())
            with self.assertRaisesRegex(ValueError, "CSV hash mismatch"):
                figure.load_exported_data(temporary)


if __name__ == "__main__":
    unittest.main()
