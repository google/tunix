#!/usr/bin/env python3
"""P58 64split target-accounting regression tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/workloads/deepswe-comparison/scripts"
    / "account_64split_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("p58_64split_accounting", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 64split accounting tool")
accounting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accounting)


def _write_jsonl(path: Path, records) -> None:
  path.write_text(
      "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
      encoding="utf-8",
  )


def _fixture(root: Path) -> dict[str, Path]:
  classification = root / "classification.json"
  classification.write_text(json.dumps({
      "verdict": "PASS",
      "topology": "64split",
      "stage": "three-update",
      "arm": "zero",
  }), encoding="utf-8")
  debug = root / "debug"
  debug.mkdir()
  metrics = []
  for step in range(3):
    metrics.append({
        "step": step,
        "optimizer_step": step,
        "trajectories": 128,
        "prompt_groups": 8,
        "trajectory_solve_ratio": 0.125,
        "all_solved_prompt_groups": 0,
        "all_failed_prompt_groups": 2,
        "mixed_prompt_groups": 6,
        "incomplete_prompt_groups": 0,
        "compact_filtered_trajectories": 0,
        "timing": {
            "batch_elapsed_seconds": 100.0 + step,
            "stage_seconds": {
                "sandbox_start": {
                    "p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0,
                },
            },
            "group_completion_seconds": list(range(1, 9)),
        },
    })
  _write_jsonl(debug / "batch_metrics.jsonl", metrics)
  updates = root / "updates.jsonl"
  _write_jsonl(updates, [
      {
          "commits": 1,
          "train_steps_after": step + 1,
          "verdict": "PASS",
          "gradient_finite": True,
          "optimizer_placement": "device-resident",
          "elapsed_seconds": 20.0 + step,
          "hbm_after_commit": [{
              "peak_bytes_in_use": 60 * 1024**3,
              "bytes_limit": 100 * 1024**3,
          }],
      }
      for step in range(3)
  ])
  run_log = root / "run.log"
  lines = [
      "[P58.36.BATCH] FULL_CONSUMER_PASS",
      "[ENGINE_STEP_LOG] on full=200",
      "[ENGINE_STEP] n=1 t=1.0",
  ]
  for step in range(3):
    lines.append("[P58.36.BATCH] DEADLINE_START")
    lines.extend("[P58.36.GROUP] COMPLETE" for _ in range(8))
    lines.append(f"[PERF] step={step} stage=weight_sync seconds={3 + step}")
    lines.append(
        f"[PERF] stage=segmented_value_and_grad seconds={10 + step}"
    )
    lines.append("[PERF] stage=optimizer_transaction seconds=2.0")
  run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return {
      "classification": classification,
      "run_log": run_log,
      "debug": debug,
      "updates": updates,
  }


class P5864SplitAccountingTest(unittest.TestCase):

  def test_postflight_runs_accounting_only_for_exact_pilot_identity(self):
    postflight = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "64split:three-update:zero",
        postflight,
    )
    self.assertIn(
        "scripts/account_64split_pilot.py",
        postflight,
    )
    self.assertIn("[P58.64SPLIT.ACCOUNT] PASS", postflight)

  def test_complete_pilot_passes_and_reports_split(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      report = accounting.account(
          classification_path=paths["classification"],
          run_log_path=paths["run_log"],
          debug_dir=paths["debug"],
          update_report_path=paths["updates"],
          reference_update_report_path=paths["updates"],
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(report["observed_commits"], 3)
      self.assertEqual(report["timing"]["rollout_R32_seconds"]["p50"], 101.0)
      self.assertEqual(report["timing"]["trainer_U32_seconds"]["p50"], 21.0)
      self.assertEqual(report["timing"]["weight_sync_S_seconds"]["p50"], 4.0)
      self.assertEqual(
          report["trainer_hbm"]["peak_fraction_of_minimum_limit"], 0.6
      )
      self.assertTrue(report["reference_update_report"]["bytewise_equal"])

  def test_missing_group_receipt_is_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      text = paths["run_log"].read_text(encoding="utf-8")
      paths["run_log"].write_text(
          text.replace("[P58.36.GROUP] COMPLETE\n", "", 1),
          encoding="utf-8",
      )
      report = accounting.account(
          classification_path=paths["classification"],
          run_log_path=paths["run_log"],
          debug_dir=paths["debug"],
          update_report_path=paths["updates"],
      )
      self.assertEqual(report["verdict"], "INCONCLUSIVE")
      self.assertIn("shared_deadline_receipts", report["failed"])

  def test_missing_trainer_hbm_receipt_is_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      updates = accounting._read_jsonl(paths["updates"])
      for update in updates:
        update.pop("hbm_after_commit", None)
      _write_jsonl(paths["updates"], updates)
      report = accounting.account(
          classification_path=paths["classification"],
          run_log_path=paths["run_log"],
          debug_dir=paths["debug"],
          update_report_path=paths["updates"],
      )
      self.assertEqual(report["verdict"], "INCONCLUSIVE")
      self.assertIn("trainer_hbm_receipt_present", report["failed"])
      self.assertIn(
          "trainer_hbm_peak_headroom_at_least_8gib", report["failed"]
      )

  def test_one_complete_unconsumed_prefetch_batch_is_reported(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = _fixture(Path(directory))
      with paths["run_log"].open("a", encoding="utf-8") as target:
        target.write("[P58.36.BATCH] DEADLINE_START\n")
        target.writelines("[P58.36.GROUP] COMPLETE\n" for _ in range(8))
      report = accounting.account(
          classification_path=paths["classification"],
          run_log_path=paths["run_log"],
          debug_dir=paths["debug"],
          update_report_path=paths["updates"],
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(report["unconsumed_prefetch_batches"], 1)


if __name__ == "__main__":
  unittest.main()
