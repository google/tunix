#!/usr/bin/env python3
"""Reproduce the P38 mismatch-to-incident join from a flattened evidence bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


def _load_classifier(script_dir: Path):
  path = script_dir / "classify_p38_serving_capture.py"
  spec = importlib.util.spec_from_file_location("p38_capture_classifier", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load classifier: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _value_record(capsule: dict[str, Any], join: dict[str, Any]) -> dict[str, Any]:
  arrays = capsule["arrays"]
  rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  matches = np.flatnonzero(rows == int(join["source_row"]))
  if matches.size != 1:
    raise RuntimeError(
        f"source row {join['source_row']} has {matches.size} capsule matches"
    )
  capsule_index = int(matches[0])
  position = int(join["completion_position"])
  a = float(np.asarray(arrays["s_decode"])[capsule_index, position])
  b = float(np.asarray(arrays["s_prefill"])[capsule_index, position])
  c = float(np.asarray(arrays["t_old"])[capsule_index, position])
  return {
      "s_decode": a,
      "s_prefill": b,
      "t_old": c,
      "abs_a_b": abs(a - b),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evidence-dir", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--expected-target-call", type=int, default=None)
  args = parser.parse_args()

  evidence = args.evidence_dir.resolve()
  ledger = evidence / "incident-ledger.jsonl"
  if not ledger.is_file():
    ledger = evidence / "p38_incident_ledger.jsonl"
  capsules = sorted(evidence.glob("p38_frozenlake_mismatch_capsule.round-*.npz"))
  if not ledger.is_file() or not capsules:
    raise SystemExit("evidence bundle lacks incident ledger or round capsules")

  classifier = _load_classifier(Path(__file__).resolve().parent)
  with tempfile.TemporaryDirectory(prefix="p38-single-active-audit-") as tmp:
    normalized = Path(tmp)
    shutil.copyfile(ledger, normalized / "p38_incident_ledger.jsonl")
    entries = classifier._load_incident_ledger(
        normalized, "standard", 1400, 3072
    )

  joins: list[dict[str, Any]] = []
  round_summaries = []
  for capsule_path in capsules:
    capsule = classifier._load_mismatch_capsule(capsule_path)
    round_joins, missing = classifier._join_incident_to_capsule(entries, capsule)
    if missing:
      raise SystemExit(
          f"capsule {capsule_path.name} has missing incident joins: {missing[:8]}"
      )
    for join in round_joins:
      join = {**join, **_value_record(capsule, join)}
      joins.append(join)
    round_summaries.append({
        "diagnostic_round": int(capsule["metadata"]["diagnostic_round"]),
        "capsule": capsule_path.name,
        "mismatch_elements": len(round_joins),
    })

  single_active = [
      join for join in joins
      if int(join["scheduled_request_count"]) == 1
      and join["single_active_token_ids_present"] is True
  ]
  if not single_active:
    raise SystemExit("no naturally single-active mismatch joined")
  targets = single_active
  if args.expected_target_call is not None:
    targets = [
        join for join in single_active
        if int(join["call_index"]) == args.expected_target_call
    ]
    if len(targets) != 1:
      raise SystemExit(
          f"expected exactly one target call {args.expected_target_call}; got {len(targets)}"
      )

  geometry_signatures = {
      json.dumps(entry.get("compile_geometry"), sort_keys=True)
      for entry in entries
      if entry.get("compile_geometry") is not None
  }
  raw_record_count = sum(1 for line in ledger.read_text().splitlines() if line)
  report = {
      "schema": "p38-single-active-audit-v1",
      "status": "PASS",
      "claim_level": "exact-host-join-not-kv-content",
      "evidence_dir": str(evidence),
      "ledger": ledger.name,
      "ledger_records": raw_record_count,
      "ledger_request_entries": len(entries),
      "last_call_index": max(int(entry["call_index"]) for entry in entries),
      "rounds": round_summaries,
      "joined_mismatch_elements": len(joins),
      "single_active_mismatch_elements": len(single_active),
      "fixed_m_geometry_signatures": len(geometry_signatures),
      "target": targets[0] if len(targets) == 1 else targets,
      "claim_ceiling": [
          "The join proves the exact historical decode call and fixed-M geometry.",
          "It does not contain live KV bytes or prove clean-KV equality.",
          "A DP1 one-host replay is E0-lite, not production-executable identity.",
      ],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
