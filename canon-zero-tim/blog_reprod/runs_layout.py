#!/usr/bin/env python3
"""The one place that knows where the three Figure 4 exports live.

They moved out of blog_reprod/runs/ into the tree's single home for raw W&B
exports, canon-zero-tim/results/<workload>/<arm>/<run-id>/, where results/
carries its own README, RUNS.tsv row and checker for each run.  blog_reprod/
stays the frozen Figure 4 package and just reads them from there; this module
also holds the only mapping from the manifest's arm names to the arm directory
names results/ uses.  Standard library only.
"""
from pathlib import Path

WORKLOAD = "frozenlake-short-horizon"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_DIR = RESULTS_DIR / WORKLOAD
# manifest arm name -> results/<workload>/ subdirectory
ARM_DIRS = {"standard": "standard", "importance_sampling": "tis", "zero_tim": "zero_tim"}


def run_dir(arm, run_id):
  """The directory holding one arm's history.csv, config.yaml and summary.json."""
  return RUNS_DIR / ARM_DIRS[arm] / run_id


def source_prefix(arm):
  """That directory as data/manifest.json records it, relative to canon-zero-tim/."""
  return f"results/{WORKLOAD}/{ARM_DIRS[arm]}"
