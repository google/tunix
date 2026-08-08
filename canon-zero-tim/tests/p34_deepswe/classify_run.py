#!/usr/bin/env python3
"""Fail-closed classifier for one P34 DeepSWE promotion stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_STAGE_UPDATES = {
    "backward-no-commit": 1,
    "one-update": 1,
    "three-update": 3,
    "full": 1000,
}


def _json_records(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise ValueError(f"missing evidence file: {path}")
  records = []
  for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not isinstance(value, dict):
      raise ValueError(f"non-object evidence at {path}:{line_number}")
    records.append(value)
  if not records:
    raise ValueError(f"empty evidence file: {path}")
  return records


def classify(
    *, log_text: str, alignment: list[dict[str, Any]], updates: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
  """Returns the complete verdict without synthesizing missing evidence."""
  if stage not in _STAGE_UPDATES:
    raise ValueError(f"unknown P34 stage: {stage!r}")
  expected_updates = _STAGE_UPDATES[stage]
  expected_alignment = expected_updates * 4
  expected_commits = 0 if stage == "backward-no-commit" else expected_updates
  checks = {
      "attempt_zero": log_text.count(
          "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)"
      ) == 1,
      "pathways_once": log_text.count(
          "[P34.PATHWAYS] initialized_once=1 before_jax=1"
      ) == 1,
      "cli_exact": log_text.count("[P34.CLI] PASS") == 1,
      "source_exact": log_text.count("[sync] provenance ok") == 1,
      "whitelist_exact": log_text.count("[env] P34 whitelist SHA256 OK:") == 1,
      "topology_exact": log_text.count("[P34.TOPOLOGY] PASS") == 1,
      "dataset_filtered": log_text.count("[P34.DATASET] GOLD_FILTER_PASS") == 1,
      "r2e_bounded": log_text.count(
          "[P34.R2E] BOUNDED_KUBERNETES_PATCH_PASS"
      ) == 1,
      "wandb_online": log_text.count(
          "[CANON_P34_WANDB] ONLINE_RUN_PASS"
      ) == 1,
      "fixed_ar_executed": "CANON_FIXED_AR=1 fixed-order tree" in log_text,
      "fixed_embed_executed": (
          "CANON_FIXED_AR_EMBED=1 fixed-order embed gather" in log_text
      ),
      "logprob_m_executed": "CANON_LOGPROB_M on" in log_text,
      "alignment_count": len(alignment) == expected_alignment,
      "alignment_pass": all(record.get("verdict") == "PASS" for record in alignment),
      "four_boundaries_exact": all(
          all(
              boundary.get("differing_bytes") == 0
              for boundary in record.get("boundaries", {}).values()
          )
          and record.get("exact", {}).get("w_all_exactly_1") is True
          and record.get("exact", {}).get("r_all_exactly_1") is True
          and record.get("exact", {}).get("wr_all_exactly_1") is True
          and record.get("clip_hits") == 0
          and record.get("tis_hits") == 0
          for record in alignment
      ),
      "update_count": len(updates) == expected_updates,
      "update_pass": all(record.get("verdict") == "PASS" for record in updates),
      "commit_count": sum(int(record.get("commits", -1)) for record in updates)
      == expected_commits,
      "gradient_health": all(
          bool(record.get("gradient_activity"))
          and any(bool(value) for value in record["gradient_activity"])
          and record.get("gradient_finite") is True
          for record in updates
      ),
      "fixed_dp_transaction": all(
          record.get("dp_replicas_exact") is True
          and record.get("dp_reduction_transactions") == 4
          and record.get("dp_reduction_rounds_per_transaction") == 8
          and record.get("dp_rank_pullbacks_per_transaction") == 16
          for record in updates
      ),
      "optimizer_host_roundtrip": all(
          record.get("optimizer_memory_kinds_before") == ["pinned_host"]
          and (
              stage == "backward-no-commit"
              or record.get("optimizer_memory_kinds_after") == ["pinned_host"]
          )
          for record in updates
      ),
  }
  if stage != "backward-no-commit":
    checks["weight_sync_count"] = log_text.count(
        "[P28.G6] weight_sync_committed count=1"
    ) == expected_updates
    steps = [int(record.get("train_steps_after", -1)) for record in updates]
    checks["monotonic_update_steps"] = steps == sorted(set(steps))
  else:
    checks["gradient_deterministic_repeat"] = all(
        record.get("gradient_deterministic") is True for record in updates
    )
  failed = sorted(name for name, passed in checks.items() if not passed)
  return {
      "schema": "canon.p34.deepswe.run.v1",
      "stage": stage,
      "verdict": "PASS" if not failed else "FAIL",
      "expected_updates": expected_updates,
      "expected_alignment_records": expected_alignment,
      "checks": checks,
      "failed": failed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", required=True, choices=tuple(_STAGE_UPDATES))
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      log_text=args.run_log.read_text(errors="replace"),
      alignment=_json_records(args.alignment_report),
      updates=_json_records(args.update_report),
      stage=args.stage,
  )
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True), flush=True)
  if report["verdict"] != "PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
