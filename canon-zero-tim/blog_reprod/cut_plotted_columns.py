#!/usr/bin/env python3
"""Cut the five plotted columns out of a W&B history export.

python3 canon-zero-tim/blog_reprod/cut_plotted_columns.py \
  canon-zero-tim/blog_reprod/runs/<run-id>/history.csv \
  canon-zero-tim/blog_reprod/data/<arm>.csv

Keeps the rows whose `_step` is 0..199, sorted by `_step`, and writes exactly
the five plotted columns in plotting order with LF line endings.  A column the
arm never logged stays empty.  Rerunning it on the archived exports rewrites
the three shipped `data/*.csv` byte for byte.
"""
import csv
import sys

FIELDS = [
    "_step",
    "rewards/train/solve_ratio",
    "sampler_trainer/train/logp_diff_mean",
    "sampler_trainer/train/logp_diff_max",
    "canonical/train/alignment_max_differing_bytes",
]
FIRST_STEP = 0
STOP_STEP = 200


def cut(history_csv, out_csv):
  """Writes the plotted columns of `history_csv` to `out_csv`; returns rows."""
  with open(history_csv, newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["_step"] and FIRST_STEP <= int(float(r["_step"])) < STOP_STEP]
  rows.sort(key=lambda r: int(float(r["_step"])))
  with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  return len(rows)


def main(argv):
  if len(argv) != 3:
    sys.stderr.write(__doc__)
    return 2
  cut(argv[1], argv[2])
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
