#!/usr/bin/env python3
"""Byte-for-byte test for make_manifest.py.

python3 -m unittest make_manifest_test

Every case stages a throwaway copy of the package - `blog_reprod/data/` plus the
sibling `results/<workload>/` runs_layout.py points at - in a temp directory and
runs a copy of the script there, so the tree's own `data/manifest.json` is never
rewritten.  `BLOG_REPROD_DIR` overrides where the inputs are read from; by
default it is this file's directory when that directory holds `data/`, otherwise
the checked-out worktree.
"""
import json
import os
import pathlib
import shutil
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
SCRIPT = _HERE / "make_manifest.py"
sys.path.insert(0, str(ROOT))
import runs_layout  # noqa: E402

ARCHIVE_COMMIT = "a7255cfc4b15a29ed9cabcd67a54cac416cd8b74"
STANDARD_RUN = "jff877lt_canon-p57-fl-stan-r01-567c96d5"
# The only fields the archive layout and a regeneration from runs/ disagree on.
PROVENANCE_FIELDS = {
    f"runs.{arm}.{field}"
    for arm in ("standard", "importance_sampling", "zero_tim")
    for field in ("source_history", "source_config")
}


def leaves(value, path=""):
  """Flattens a manifest into {dotted path: scalar} so diffs name a field."""
  if isinstance(value, dict):
    flat = {}
    for key, item in value.items():
      flat.update(leaves(item, f"{path}.{key}" if path else key))
    return flat
  return {path: value}


class MakeManifestTest(unittest.TestCase):

  def setUp(self):
    self.tmp = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.tmp, True)
    self.package = self.tmp / "blog_reprod"
    self.runs = self.tmp / "results" / runs_layout.WORKLOAD
    shutil.copytree(runs_layout.RUNS_DIR, self.runs)
    shutil.copytree(ROOT / "data", self.package / "data")
    for name in (SCRIPT.name, "runs_layout.py"):
      shutil.copy2(ROOT / name, self.package / name)
    self.shipped = (ROOT / "data" / "manifest.json").read_bytes()

  def run_script(self, *args):
    return subprocess.run([sys.executable, str(self.package / SCRIPT.name), *args],
                          capture_output=True, text=True)

  def regenerate(self, *args):
    done = self.run_script("--source-commit", ARCHIVE_COMMIT, *args)
    self.assertEqual(done.returncode, 0, done.stderr)
    return (self.package / "data" / "manifest.json").read_bytes()

  def test_regenerates_the_shipped_manifest_byte_for_byte(self):
    self.assertEqual(self.regenerate("--source-prefix", "wandb_exports"), self.shipped)

  def test_default_prefix_differs_only_in_the_six_provenance_strings(self):
    shipped, rebuilt = leaves(json.loads(self.shipped)), leaves(json.loads(self.regenerate()))
    self.assertEqual(set(shipped), set(rebuilt))
    differing = {field for field, value in shipped.items() if rebuilt[field] != value}
    self.assertEqual(differing, PROVENANCE_FIELDS)
    for field in differing:
      prefix = runs_layout.source_prefix(field.split(".")[1])
      self.assertEqual(rebuilt[field], shipped[field].replace("wandb_exports", prefix, 1))

  def test_refuses_a_source_commit_that_is_not_a_full_hash(self):
    done = self.run_script("--source-commit", "deadbeef")
    self.assertEqual(done.returncode, 1)
    self.assertIn("--source-commit must be a 40-character commit hash", done.stderr)

  def test_refuses_a_history_that_is_missing_steps(self):
    history = self.runs / "standard" / STANDARD_RUN / "history.csv"
    history.write_bytes(b"\n".join(history.read_bytes().split(b"\n")[:151]))
    done = self.run_script("--source-commit", ARCHIVE_COMMIT)
    self.assertEqual(done.returncode, 1)
    self.assertIn("standard: needs one observation per step 0..199", done.stderr)

  def test_refuses_a_csv_that_is_not_the_cut_of_its_history(self):
    shutil.copy2(self.package / "data" / "zero_tim.csv",
                 self.package / "data" / "standard.csv")
    done = self.run_script("--source-commit", ARCHIVE_COMMIT)
    self.assertEqual(done.returncode, 1)
    self.assertIn("standard: data/standard.csv is not the cut of that history.csv", done.stderr)


if __name__ == "__main__":
  unittest.main()
