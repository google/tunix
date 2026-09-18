#!/usr/bin/env python3
"""Rebuild data/manifest.json from the exports in runs/ and the CSVs in data/.

    python3 canon-zero-tim/blog_reprod/make_manifest.py --source-commit <sha40> \
      --standard <run-id> --tis <run-id> --zero <run-id>

Every arm needs one training observation per step 0..199 in runs/<run-id>/history.csv,
and its data/<arm>.csv must be the cut of that history.  --source-commit is the commit
the runs were launched from; --source-prefix only changes the recorded provenance
strings source_history / source_config — every consumer reads runs/<run-id>/.
Standard library only.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STEPS = 200
FIELDS = ("_step", "rewards/train/solve_ratio", "sampler_trainer/train/logp_diff_mean",
          "sampler_trainer/train/logp_diff_max",
          "canonical/train/alignment_max_differing_bytes")
ARMS = (("standard", "--standard", "jff877lt_canon-p57-fl-stan-r01-567c96d5"),
        ("importance_sampling", "--tis", "8zjz4li7_canon-p57-fl-is-i45g-ccbcf572"),
        ("zero_tim", "--zero", "tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9"))
CONFIG_KEYS = (
    "model_id loss_algo advantage_estimator batch_size mini_batch_size num_generations seed "
    "mesh_dp mesh_tp env_max_steps max_prompt_length max_response_length num_steps "
    "old_logps_source sampler_is eval_every_n_steps learning_rate epsilon epsilon_high "
    "loss_agg_mode temperature top_p top_k train_trajectory_micro_batch_size").split()


def read_config(blob):
  """Returns the manifest's config subset from a flat `key: value` config.yaml.

  Values are ints, floats or plain strings: none of CONFIG_KEYS is a boolean, a
  null or a quoted string in any arm.
  """
  values = {}
  for line in blob.decode("utf-8").splitlines():
    key, separator, raw = line.partition(": ")
    if not separator or line[:1].isspace() or line.startswith("-"):
      continue  # a nested block; no key of ours needs one
    for cast in (int, float, str):
      try:
        values[key] = cast(raw.strip())
        break
      except ValueError:
        pass
  return {key: values.get(key) for key in CONFIG_KEYS}


def entry(arm, run_id, prefix):
  """Returns one arm's manifest entry, hashing the three files it pins."""
  run = ROOT / "runs" / run_id
  history, config = (run / "history.csv").read_bytes(), (run / "config.yaml").read_bytes()
  plotted = (ROOT / "data" / (arm + ".csv")).read_bytes()
  selected, lines = [], []
  for line, row in enumerate(csv.DictReader(io.StringIO(history.decode("utf-8"))), 2):
    if not row.get("_step") or not 0 <= int(float(row["_step"])) < STEPS:
      continue
    selected.append({key: row.get(key, "") for key in FIELDS})
    lines.append(line)
  if [int(float(row["_step"])) for row in selected] != list(range(STEPS)):
    raise SystemExit(f"FAIL: {arm}: needs one observation per step 0..{STEPS - 1}")
  buffer = io.StringIO(newline="")
  writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
  writer.writeheader()
  writer.writerows(selected)
  if buffer.getvalue().encode("utf-8") != plotted:
    raise SystemExit(f"FAIL: {arm}: data/{arm}.csv is not the cut of that history.csv")
  metrics = {}
  for key in FIELDS[1:]:
    values = [float(row[key]) for row in selected if row[key] != ""]
    if values:
      metrics[key] = {"count": len(values), "minimum": min(values),
                      "maximum": max(values), "last_ten_mean": sum(values[-10:]) / 10}
  return {"run_id": run_id, "source_history": f"{prefix}/{run_id}/history.csv",
          "source_history_sha256": hashlib.sha256(history).hexdigest(),
          "source_config": f"{prefix}/{run_id}/config.yaml",
          "source_config_sha256": hashlib.sha256(config).hexdigest(),
          "plotted_csv": arm + ".csv",
          "plotted_csv_sha256": hashlib.sha256(plotted).hexdigest(),
          "config": read_config(config), "source_csv_lines": lines, "metrics": metrics}


def main(argv=None):
  parser = argparse.ArgumentParser(description="Rebuild data/manifest.json.")
  parser.add_argument("--source-commit", required=True,
                      help="the 40-character commit the runs were launched from")
  for arm, flag, default in ARMS:
    parser.add_argument(flag, dest=arm, default=default, help=f"{arm} W&B run id")
  parser.add_argument("--source-prefix", default="runs",
                      help="directory recorded in source_history / source_config")
  args = parser.parse_args(argv)
  commit = args.source_commit
  if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
    raise SystemExit("FAIL: --source-commit must be a 40-character commit hash")
  manifest = {
      "schema_version": 1, "source_commit": commit,
      "evidence_grade": "analysis-grade full-resolution training telemetry",
      "signed_full_run_certification": False,
      "window": {"source_step_first": 0, "source_step_last": STEPS - 1,
                 "display_step_first": 1, "display_step_last": STEPS, "count_per_arm": STEPS},
      "transformations": {"logprob_mean": "none", "solve_rate": "raw plus trailing mean",
                          "trailing_window": 10, "initial_window": "available points only"},
      "runs": {arm: entry(arm, getattr(args, arm), args.source_prefix) for arm, _, _ in ARMS},
  }
  (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
  print(f"PASS: wrote data/manifest.json (source_commit {commit}; 3 arms; {STEPS} rows each)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
