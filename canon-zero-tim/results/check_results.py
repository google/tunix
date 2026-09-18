#!/usr/bin/env python3
"""Check that results/ and results/RUNS.tsv describe the same runs.

    python3 canon-zero-tim/results/check_results.py

Every results/<workload>/<arm>/<run-id>/ must hold exactly history.csv,
config.yaml and summary.json; its history.csv must carry one _step per row,
contiguous from 0; and it must have a RUNS.tsv row whose steps column equals
that row count and whose grade is one of the four allowed.  Every RUNS.tsv row
must have its directory.  Prints RESULTS_CHECK PASS runs=<n>, or one FAIL line
per problem and exits 1.  Standard library only.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = {"history.csv", "config.yaml", "summary.json"}
COLUMNS = ["workload", "arm", "run_id", "wandb", "executed_sha", "steps", "grade",
           "date", "note"]
GRADES = ("complete", "incomplete", "legacy-code", "receipts-only")


def subdirs(path):
  """The directories directly under `path`, sorted, skipping . and _ names."""
  return sorted(p for p in path.iterdir() if p.is_dir() and p.name[0] not in "._")


def run_dirs(root):
  """Yields (workload, arm, run directory) for every results/<w>/<a>/<r>/."""
  for workload in subdirs(root):
    for arm in subdirs(workload):
      for run in subdirs(arm):
        yield workload.name, arm.name, run


def step_count(history, fail, where):
  """Returns the history's row count, or None when its _step column is unusable."""
  with history.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if "_step" not in (reader.fieldnames or []):
      fail(f"{where}: history.csv has no _step column")
      return None
    raw = [row.get("_step") or "" for row in reader]
  steps = [int(value) for value in raw if value.isdigit()]
  if not raw or steps != list(range(len(raw))):
    fail(f"{where}: history.csv _step must be one integer per row, from 0 upwards")
    return None
  return len(raw)


def main():
  failures = []
  fail = failures.append
  with (ROOT / "RUNS.tsv").open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != COLUMNS:
      print(f"FAIL RUNS.tsv: header {reader.fieldnames} is not {COLUMNS}")
      print("RESULTS_CHECK FAIL failures=1")
      return 1
    rows = {(r["workload"], r["arm"], r["run_id"]): r for r in reader}
  found = set()
  for workload, arm, run in run_dirs(ROOT):
    where = f"{workload}/{arm}/{run.name}"
    found.add((workload, arm, run.name))
    names = {p.name for p in run.iterdir()}
    for name in sorted(REQUIRED - names) + sorted(names - REQUIRED):
      fail(f"{where}: {'missing' if name in REQUIRED else 'unknown'} file {name}")
    row = rows.get((workload, arm, run.name))
    if row is None:
      fail(f"{where}: no RUNS.tsv row")
    elif row["grade"] not in GRADES:
      fail(f"{where}: grade {row['grade']!r} is not one of {GRADES}")
    count = None if REQUIRED - names else step_count(run / "history.csv", fail, where)
    if row is not None and count is not None and row["steps"] != str(count):
      fail(f"{where}: RUNS.tsv says steps={row['steps']}, history.csv has {count} rows")
  for key in sorted(set(rows) - found):
    fail("/".join(key) + ": RUNS.tsv row has no directory")
  for message in failures:
    print(f"FAIL {message}")
  if failures:
    print(f"RESULTS_CHECK FAIL failures={len(failures)}")
    return 1
  print(f"RESULTS_CHECK PASS runs={len(found)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
