#!/usr/bin/env python3
"""Classify one frozen P58 coarse seam-localization round."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
_M15_PATH = ROOT / (
    "canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/"
    "classify_m15_apc_wide_seam.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "classify_m15_apc_wide_seam_for_p58", _M15_PATH
)
if _SPEC is None or _SPEC.loader is None:
  raise RuntimeError("cannot import the bounded P38 seam classifier")
_M15 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M15)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def classify(
    *, directory: Path, alignment_report: Path, capsules: list[Path]
) -> dict[str, Any]:
  """Returns a P58-scoped report for one already staged round."""
  underlying = _M15.classify(
      directory=directory,
      alignment_report=alignment_report,
      capsules=capsules,
      mode="layer",
      arm="on",
      replay_ledger=None,
      expected_layer=None,
      require_first_action=False,
  )
  _require(underlying.get("status") == "PASS", "seam classifier did not pass")
  alignment = underlying.get("alignment", {})
  coverage = underlying.get("coverage", {})
  signatures = underlying.get("first_difference_signatures", [])
  _require(
      int(alignment.get("a_b_differing_bytes", 0)) > 0,
      "P58 localization round is not A-B red",
  )
  _require(
      int(alignment.get("b_c_differing_bytes", -1)) == 0,
      "P58 localization round changed the B-C control",
  )
  _require(
      int(coverage.get("standard_joinable_red_points", 0)) > 0,
      "P58 localization round has no exact standard-path red join",
  )
  _require(signatures, "P58 localization round has no first-red signature")
  return {
      "schema": "canon.p58.coarse-seam-round-classification.v1",
      "verdict": "PASS",
      "outcome": "COARSE_FIRST_RED_INTERVAL",
      "diagnostic_round": int(underlying["diagnostic_round"]),
      "observer_mode": "layer",
      "alignment": alignment,
      "coverage": coverage,
      "seam_inventory": underlying.get("seam_inventory"),
      "tail_inventory": underlying.get("tail_inventory"),
      "first_difference_signatures": signatures,
      "mixed_first_difference_signatures": bool(
          underlying.get("mixed_first_difference_signatures")
      ),
      "selected_layer": underlying.get("selected_layer"),
      "last_exact_boundary": underlying.get("last_exact_boundary"),
      "first_red_boundary": underlying.get("first_red_boundary"),
      "source_interval": underlying.get("source_interval"),
      "anchors": underlying.get("anchors", []),
      "backward": 0,
      "optimizer_commits": 0,
      "next_action": underlying.get("next_action"),
      "claim_ceiling": (
          "This frozen P58 round localizes exact standard-path red anchors "
          "inside the configured position window. Continue-decode actions "
          "outside that path remain unobserved, and integer fingerprint "
          "equality is not full-tensor byte equality."
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--alignment-report", required=True, type=Path)
  parser.add_argument("--capsule", action="append", default=[], type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {args.output}")
  report = classify(
      directory=args.directory,
      alignment_report=args.alignment_report,
      capsules=args.capsule,
  )
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P58_COARSE_SEAM_ROUND_CLASSIFICATION "
      f"round={report['diagnostic_round']} verdict={report['verdict']} "
      f"selected_layer={report['selected_layer']} backward=0 "
      "optimizer_commits=0",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
