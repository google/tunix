#!/usr/bin/env python3
"""Byte-for-byte test for cut_plotted_columns.py.

python3 -m unittest cut_plotted_columns_test

Runs the script on each archived W&B history export and asserts the SHA-256 of
its output equals the shipped `data/<arm>.csv`.  `BLOG_REPROD_DIR` overrides
where `runs/` and `data/` are read from; by default it is this file's directory
when that directory holds `data/`, otherwise the checked-out worktree.
"""
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
_FALLBACK = pathlib.Path(
    "/mnt/disks/tunix-data/worktrees/zero_tim_repro_20260917/canon-zero-tim/blog_reprod")
ROOT = pathlib.Path(
    os.environ.get("BLOG_REPROD_DIR")
    or (_HERE if (_HERE / "data").is_dir() else _FALLBACK))
SCRIPT = _HERE / "cut_plotted_columns.py"

# arm -> (archived run id, sha256 of the shipped data/<arm>.csv)
ARMS = {
    "standard": (
        "jff877lt_canon-p57-fl-stan-r01-567c96d5",
        "09eb10ee12467c93007da3a221be5c55c52e0f56eb6e34f5b49e513700d70cec"),
    "importance_sampling": (
        "8zjz4li7_canon-p57-fl-is-i45g-ccbcf572",
        "e61d8ce7ab297069a2c8fe3a6025262f3efd560d14c2d59c56658bcefeff96e9"),
    "zero_tim": (
        "tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9",
        "ea1af6c35612753b422f19fdaeb24361fb6d7c55d4cad39719bbba2bb06ff670"),
}


def sha256(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run_script(history_csv, out_csv):
  return subprocess.run([sys.executable, str(SCRIPT), str(history_csv), str(out_csv)],
                        capture_output=True, text=True)


class CutPlottedColumnsTest(unittest.TestCase):

  def test_shipped_hashes_are_the_hashes_on_disk(self):
    for arm, (_, want) in ARMS.items():
      with self.subTest(arm=arm):
        self.assertEqual(sha256(ROOT / "data" / f"{arm}.csv"), want)

  def test_each_arm_reproduces_its_shipped_csv_byte_for_byte(self):
    for arm, (run_id, want) in ARMS.items():
      with self.subTest(arm=arm):
        with tempfile.TemporaryDirectory() as tmp:
          out = pathlib.Path(tmp) / f"{arm}.csv"
          done = run_script(ROOT / "runs" / run_id / "history.csv", out)
          self.assertEqual(done.returncode, 0, done.stderr)
          self.assertEqual(sha256(out), want)

  def test_a_truncated_history_does_not_reproduce(self):
    run_id, want = ARMS["standard"]
    history = (ROOT / "runs" / run_id / "history.csv").read_bytes().split(b"\n")
    with tempfile.TemporaryDirectory() as tmp:
      short = pathlib.Path(tmp) / "history.csv"
      short.write_bytes(b"\n".join(history[:101]))  # header + 100 rows
      out = pathlib.Path(tmp) / "standard.csv"
      done = run_script(short, out)
      self.assertEqual(done.returncode, 0, done.stderr)
      self.assertNotEqual(sha256(out), want)

  def test_wrong_argument_count_exits_two(self):
    done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
  unittest.main()
