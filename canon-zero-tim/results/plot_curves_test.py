#!/usr/bin/env python3
"""Tests for plot_curves.py.

    python3 -m unittest plot_curves_test

Pins the three trailing-ten endpoints Figure 4 prints, proves the checked-in
curves.svg and summary.csv are what the script writes today, and renders a
synthetic two-arm thirty-step workload so the drawing code is not tied to the
two-hundred-update window Figure 4 happens to use.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_curves  # noqa: E402

WORKLOAD = "frozenlake-short-horizon"
# The endpoint labels Figure 4 prints, at W&B _step 199 -- policy update 200.
FIGURE_UPDATE = 200
ENDPOINTS = {"standard": 0.6265625, "tis": 0.66875, "zero_tim": 0.882421875}


def as_written(value):
  """The %.16g rule the spreadsheet test uses, so a float compares as printed."""
  return float("%.16g" % value)


class PlotCurvesTest(unittest.TestCase):

  def test_the_three_figure_four_endpoints(self):
    runs = {run.arm: run for run in plot_curves.load_workload(HERE / WORKLOAD)}
    self.assertEqual(set(runs), set(ENDPOINTS))
    for arm, want in ENDPOINTS.items():
      with self.subTest(arm=arm):
        point = next(p for p in runs[arm].trail if p.update == FIGURE_UPDATE)
        self.assertEqual(as_written(point.value), as_written(want))

  def test_the_checked_in_outputs_are_what_the_script_writes(self):
    tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, tmp, True)
    done = subprocess.run([sys.executable, str(HERE / "plot_curves.py"),
                           "--workload", WORKLOAD, "--out", str(tmp)],
                          capture_output=True, text=True)
    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
    self.assertIn(f"PLOT_CURVES PASS workload={WORKLOAD} runs=3", done.stdout)
    for name in ("curves.svg", "summary.csv"):
      with self.subTest(name=name):
        self.assertEqual((tmp / name).read_bytes(), (HERE / WORKLOAD / name).read_bytes())

  def test_a_synthetic_two_arm_thirty_step_workload_renders(self):
    tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, tmp, True)
    for index, arm in enumerate(("standard", "zero_tim")):
      run = tmp / arm / f"abc{index}_synthetic"
      run.mkdir(parents=True)
      rows = [f"_step,{plot_curves.SOLVE},{plot_curves.LOGP}"]
      rows += [f"{step},{(step + index) / 40:.6f},{index * step / 1000:.6f}"
               for step in range(30)]
      (run / "history.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    runs = plot_curves.load_workload(tmp)
    self.assertEqual([run.steps for run in runs], [30, 30])
    ElementTree.fromstring(plot_curves.render_svg("synthetic", runs))
    plot_curves.write_summary(runs, tmp / "summary.csv")
    lines = (tmp / "summary.csv").read_text(encoding="utf-8").splitlines()
    self.assertEqual(len(lines), 3)
    self.assertTrue(lines[1].startswith("standard,abc0_synthetic,30,"))


if __name__ == "__main__":
  unittest.main()
