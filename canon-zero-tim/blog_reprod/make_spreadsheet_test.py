#!/usr/bin/env python3
"""Read figure4.xlsx back and prove it still says exactly what data/ says.

Regenerates the workbook into a temporary file, reopens it with openpyxl and
checks: sheet names and order; every raw curve cell is the CSV's own string;
recipe.differs is TRUE for exactly the three keys that separate the arms; the
trailing averages end on the manifest's last_ten_mean.

Run from this directory (needs openpyxl and PyYAML):
    python -m unittest make_spreadsheet_test -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

import build_end_to_end_results_measured as figure
import make_spreadsheet

DIFFERING_KEYS = {"old_logps_source", "sampler_is", "eval_every_n_steps"}
RAW_METRICS = {"solve_ratio": figure.REWARD, "logp_diff_mean": figure.MEAN}


def assert_close(case, actual, expected, label):
    """A cell keeps the manifest value to spreadsheet float precision, not bit-exactly."""
    case.assertIsInstance(actual, (int, float), label)
    case.assertAlmostEqual(actual, expected, delta=abs(expected) * 1e-15, msg=label)


def csv_rows(arm: str) -> list[dict]:
    with (make_spreadsheet.DATA / f"{arm}.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["_step"] != "" and 0 <= int(row["_step"]) < figure.STEPS]
    assert len(rows) == figure.STEPS, f"{arm}: {len(rows)} rows"
    return rows


class SpreadsheetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        path = make_spreadsheet.build(Path(cls.directory.name) / "figure4.xlsx")
        cls.book = load_workbook(path)
        cls.manifest = json.loads(
            (make_spreadsheet.DATA / "manifest.json").read_text(encoding="utf-8"))
        cls.rows = {arm: csv_rows(arm) for arm, _ in make_spreadsheet.ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_sheet_names_and_order(self):
        self.assertEqual(self.book.sheetnames, list(make_spreadsheet.SHEETS))

    def test_curves_wide_raw_cells_are_the_csv_strings(self):
        sheet = self.book["curves_wide"]
        header = [cell.value for cell in sheet[1]]
        self.assertEqual(sheet.max_row, figure.STEPS + 1)
        for arm, label in make_spreadsheet.ARMS:
            for metric, key in RAW_METRICS.items():
                column = header.index(f"{label} {metric}") + 1
                for index, row in enumerate(self.rows[arm]):
                    value = sheet.cell(row=index + 2, column=column).value
                    self.assertIsInstance(value, str)
                    self.assertEqual(value, row[key])
            column = header.index(f"{label} solve_trail10") + 1
            last = sheet.cell(row=figure.STEPS + 1, column=column).value
            assert_close(self, last,
                         self.manifest["runs"][arm]["metrics"][figure.REWARD]["last_ten_mean"],
                         f"{label} solve_trail10 endpoint")
        for index in range(figure.STEPS):
            self.assertEqual(sheet.cell(row=index + 2, column=1).value, index + 1)

    def test_curves_long_raw_cells_are_the_csv_strings(self):
        sheet = self.book["curves_long"]
        self.assertEqual([cell.value for cell in sheet[1]], ["arm", "step", "metric", "value"])
        self.assertEqual(sheet.max_row, 3 * 3 * figure.STEPS + 1)
        seen = 0
        for arm_name, step, metric, value in sheet.iter_rows(min_row=2, values_only=True):
            arm = next(key for key, label in make_spreadsheet.ARMS if label == arm_name)
            if metric in RAW_METRICS:
                self.assertIsInstance(value, str)
                self.assertEqual(value, self.rows[arm][step - 1][RAW_METRICS[metric]])
                seen += 1
        self.assertEqual(seen, 2 * 3 * figure.STEPS)

    def test_recipe_marks_exactly_the_three_differing_keys(self):
        sheet = self.book["recipe"]
        self.assertEqual([cell.value for cell in sheet[1]],
                         ["config_key"] + [label for _, label in make_spreadsheet.ARMS]
                         + ["differs"])
        config = self.manifest["runs"]["standard"]["config"]
        differing, keys = set(), []
        for key, *values, differs in sheet.iter_rows(min_row=2, values_only=True):
            keys.append(key)
            self.assertIn(differs, (True, False))
            if differs:
                differing.add(key)
            else:
                self.assertEqual(len(set(map(str, values))), 1, key)
        self.assertEqual(keys, list(config))
        self.assertEqual(differing, DIFFERING_KEYS)

    def test_provenance_and_endpoints_come_from_the_manifest(self):
        provenance = {row[0]: row for row in
                      self.book["provenance"].iter_rows(min_row=2, values_only=True)}
        endpoints = {(row[0], row[1]): row[2:] for row in
                     self.book["endpoints"].iter_rows(min_row=2, values_only=True)}
        for arm, label in make_spreadsheet.ARMS:
            entry = self.manifest["runs"][arm]
            row = provenance[label]
            self.assertEqual(row[1], entry["run_id"])
            self.assertEqual(row[2], make_spreadsheet.WANDB_PROJECT)
            self.assertTrue(entry["run_id"].endswith(row[3]))
            self.assertEqual(row[5], entry["source_history_sha256"])
            self.assertEqual(row[7], entry["source_config_sha256"])
            self.assertEqual(row[9], entry["plotted_csv_sha256"])
            self.assertEqual((row[10], row[11]),
                             (entry["source_csv_lines"][0], entry["source_csv_lines"][-1]))
            for metric, stats in entry["metrics"].items():
                cells = endpoints[(label, metric)]
                for cell, name in zip(cells, ("last_ten_mean", "minimum", "maximum")):
                    assert_close(self, cell, stats[name], f"{label} {metric} {name}")


if __name__ == "__main__":
    unittest.main()
