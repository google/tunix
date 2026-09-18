#!/usr/bin/env python3
"""Tests for check_results.py, including its two negative controls.

    python3 -m unittest check_results_test

The checked-in tree is only read.  The negative controls stage a one-run
synthetic results/ in a temp directory and run a copy of the script there, so
the fault under test is the only difference from a passing tree.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE / "check_results.py"
HEADER = "workload\tarm\trun_id\twandb\texecuted_sha\tsteps\tgrade\tdate\tnote"
ROW = "w\ta\tr\tentity/project/r\t1234abcd\t3\tcomplete\t2026-09-18\tsynthetic"
FILES = ("history.csv", "config.yaml", "summary.json")


class CheckResultsTest(unittest.TestCase):

  def stage(self, files=FILES, rows=(ROW,)):
    tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, tmp, True)
    run = tmp / "w" / "a" / "r"
    run.mkdir(parents=True)
    for name in files:
      (run / name).write_text("_step\n0\n1\n2\n" if name == "history.csv" else "{}\n")
    shutil.copy2(SCRIPT, tmp / SCRIPT.name)
    (tmp / "RUNS.tsv").write_text("\n".join((HEADER,) + tuple(rows)) + "\n")
    return subprocess.run([sys.executable, str(tmp / SCRIPT.name)],
                          capture_output=True, text=True)

  def test_the_checked_in_tree_passes(self):
    done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
    self.assertIn("RESULTS_CHECK PASS runs=6", done.stdout)

  def test_the_staged_tree_passes(self):
    done = self.stage()
    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
    self.assertIn("RESULTS_CHECK PASS runs=1", done.stdout)

  def test_a_run_missing_a_file_fails(self):
    done = self.stage(files=("history.csv", "config.yaml"))
    self.assertEqual(done.returncode, 1, done.stdout)
    self.assertIn("FAIL w/a/r: missing file summary.json", done.stdout)
    self.assertIn("RESULTS_CHECK FAIL failures=1", done.stdout)

  def test_a_row_without_a_directory_fails(self):
    done = self.stage(rows=(ROW, ROW.replace("\tr\t", "\tghost\t", 1)))
    self.assertEqual(done.returncode, 1, done.stdout)
    self.assertIn("FAIL w/a/ghost: RUNS.tsv row has no directory", done.stdout)
    self.assertIn("RESULTS_CHECK FAIL failures=1", done.stdout)


if __name__ == "__main__":
  unittest.main()
