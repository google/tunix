#!/usr/bin/env python3
"""Compare two W&B history exports cell by cell.

    python3 canon-zero-tim/blog_reprod/compare_history.py <new.csv> <archived.csv>

Passes when both files hold the same set of _step values and the same set of
columns, and every cell that is non-empty in both is equal as a float (or, when
it does not parse as a float, as a string).  A cell that is empty on one side
and not on the other is a difference.  Byte-for-byte equality is not the test:
the vendored exports were written with CRLF and a different column order is
allowed as long as the column set matches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def load(path: Path) -> dict[str, dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  table = {}
  for row in rows:
    step = (row.get("_step") or "").strip()
    if not step:
      continue
    key = str(int(float(step)))
    if key in table:
      raise SystemExit(f"{path}: duplicate _step {key}")
    table[key] = row
  return table


def numeric(value: str):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("new", type=Path)
  parser.add_argument("archived", type=Path)
  parser.add_argument("--max-differences", type=int, default=5)
  args = parser.parse_args(argv)

  left, right = load(args.new), load(args.archived)
  differences: list[str] = []
  only_left = sorted(set(left) - set(right), key=int)
  only_right = sorted(set(right) - set(left), key=int)
  for step in only_left:
    differences.append(f"step {step}: only in {args.new}")
  for step in only_right:
    differences.append(f"step {step}: only in {args.archived}")

  columns_left = {key for row in left.values() for key in row}
  columns_right = {key for row in right.values() for key in row}
  for column in sorted(columns_left ^ columns_right):
    side = args.new if column in columns_left else args.archived
    differences.append(f"column {column!r}: only in {side}")

  compared = 0
  for step in sorted(set(left) & set(right), key=int):
    for column in sorted(columns_left & columns_right):
      a = (left[step].get(column) or "").strip()
      b = (right[step].get(column) or "").strip()
      if a == "" and b == "":
        continue
      compared += 1
      fa, fb = numeric(a), numeric(b)
      equal = (fa == fb) if (fa is not None and fb is not None) else (a == b)
      if not equal:
        differences.append(f"step {step} column {column!r}: {a!r} != {b!r}")

  for line in differences[: args.max_differences]:
    print(line)
  print(
      f"COMPARE_HISTORY {'PASS' if not differences else 'FAIL'} "
      f"steps={len(left)}/{len(right)} columns={len(columns_left)}/{len(columns_right)} "
      f"cells_compared={compared} differences={len(differences)}"
  )
  return 1 if differences else 0


if __name__ == "__main__":
  sys.exit(main())
