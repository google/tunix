#!/usr/bin/env python3
"""Re-export one Weights & Biases run into the three files vendored under runs/.

The credentials come from the environment only; this script never takes an API
key as an argument and never prints one:

    export WANDB_API_KEY=...          # optional: export WANDB_ENTITY=<team>
    python3 canon-zero-tim/blog_reprod/export_wandb.py \
      --project zero-tim-p57-frozenlake-tim \
      --run jff877lt \
      --out canon-zero-tim/blog_reprod/runs/jff877lt

It writes <out>/history.csv (one row per history row, columns in first-seen
order, ascending _step), <out>/config.yaml (run.config without the _wandb
block) and <out>/summary.json.  Cell values that are not strings are written
with json.dumps, so ints, floats and bools keep their JSON text and a missing
key is an empty cell.  Line endings are LF; the exports already in runs/ came
from an older tool that wrote CRLF, so compare the parsed cells (see
compare_history.py), not file hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import yaml


def make_api(timeout: int = 60):
  """Return a wandb.Api client.  Tests replace this hook with a fake."""
  import wandb  # imported here so --help works without the package installed

  return wandb.Api(timeout=timeout)


def cell(value) -> str:
  """Render one history value the way the vendored exports hold it."""
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  return json.dumps(value)


def _step_key(row):
  try:
    return (0, float(row.get("_step")))
  except (TypeError, ValueError):
    return (1, 0.0)  # rows without a usable _step keep scan order, at the end


def history_table(run):
  """Return (columns in first-seen order, rows sorted by ascending _step)."""
  columns: list[str] = []
  seen: set[str] = set()
  rows: list[dict] = []
  for row in run.scan_history():
    for key in row:
      if key not in seen:
        seen.add(key)
        columns.append(key)
    rows.append(dict(row))
  rows.sort(key=_step_key)  # stable: equal steps keep their scan order
  return columns, rows


def write_history(run, out: Path) -> tuple[int, int]:
  columns, rows = history_table(run)
  with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
      writer.writerow([cell(row[key]) if key in row else "" for key in columns])
  return len(rows), len(columns)


def write_config(run, out: Path) -> int:
  config = {
      key: value for key, value in dict(run.config).items() if key != "_wandb"
  }
  (out / "config.yaml").write_text(
      yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
  )
  return len(config)


def write_summary(run, out: Path) -> int:
  summary = dict(run.summary)
  (out / "summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
  )
  return len(summary)


def run_path(project: str, run: str, entity: str) -> str:
  return f"{entity}/{project}/{run}" if entity else f"{project}/{run}"


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(
      description="Export one W&B run to history.csv / config.yaml / summary.json"
  )
  parser.add_argument("--project", required=True)
  parser.add_argument("--run", required=True, help="W&B run id, e.g. jff877lt")
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument(
      "--entity", default=None, help="defaults to $WANDB_ENTITY, then to the key's"
  )
  args = parser.parse_args(argv)

  if not os.environ.get("WANDB_API_KEY"):
    print(
        "WANDB_API_KEY is not set.  Export it before running this script; it is "
        "never accepted as a command-line argument.",
        file=sys.stderr,
    )
    return 2

  entity = args.entity or os.environ.get("WANDB_ENTITY") or ""
  path = run_path(args.project, args.run, entity)
  run = make_api().run(path)
  args.out.mkdir(parents=True, exist_ok=True)
  rows, columns = write_history(run, args.out)
  config_keys = write_config(run, args.out)
  summary_keys = write_summary(run, args.out)
  print(
      f"WANDB_EXPORT_PASS run={path} history_rows={rows} history_columns={columns} "
      f"config_keys={config_keys} summary_keys={summary_keys} out={args.out}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
