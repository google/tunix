#!/usr/bin/env python3
"""Validate the committed D3e return before rendering the E0 KV probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class AdmissionError(RuntimeError):
  """Raised when the immutable D3e facts do not admit E0 preparation."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise AdmissionError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def review(evidence: Path) -> dict[str, Any]:
  manifest = evidence / "SHA256SUMS"
  _require(manifest.is_file(), "D3e SHA256SUMS is absent")
  expected_names = {
      "D36_OFFLINE_REVIEW.json",
      "D36_RECLASSIFICATION.json",
      "REMOTE_MULTIROUND_SUMMARY.json",
  }
  entries = {}
  for line in manifest.read_text(encoding="ascii").splitlines():
    parts = line.split()
    _require(len(parts) == 2, "D3e manifest line is invalid")
    digest, name = parts
    _require(name in expected_names, f"unexpected D3e manifest member: {name}")
    _require(name not in entries, f"duplicate D3e manifest member: {name}")
    path = evidence / name
    _require(path.is_file(), f"D3e manifest member is absent: {name}")
    _require(_sha256(path) == digest, f"D3e manifest member drifted: {name}")
    entries[name] = digest
  _require(set(entries) == expected_names, "D3e manifest inventory drifted")

  offline = _load(evidence / "D36_OFFLINE_REVIEW.json")
  classification = _load(evidence / "D36_RECLASSIFICATION.json")
  remote = _load(evidence / "REMOTE_MULTIROUND_SUMMARY.json")
  _require(
      offline.get("status") == "FIRST_RED_LOCALIZED"
      and offline.get("reclassification_gate") == "FIRST_RED_LOCALIZED"
      and offline.get("decision_scope") == "COMPLETION_POSITION_ZERO"
      and offline.get("source_request_binding_statuses") == [
          "UNIQUE_FUTURE_PREFIX_BINDING"
      ],
      "D3e offline decision does not admit E0",
  )
  _require(
      offline.get("target_executed") is False
      and offline.get("remote_mutation") is False
      and offline.get("numerical_repair_authorized") is False
      and offline.get("pinned_exact_image_required") is True,
      "D3e claim boundary drifted",
  )
  _require(
      classification.get("status") == "PASS"
      and classification.get("gate") == "FIRST_RED_LOCALIZED"
      and classification.get("decision_scope") == "COMPLETION_POSITION_ZERO"
      and classification.get("observer_mode") == "full"
      and classification.get("selected_layer") == 0,
      "D3e classification identity drifted",
  )
  alignment = classification.get("alignment", {})
  _require(
      alignment == {
          "a_b_differing_bytes": 207,
          "a_b_differing_elements": 95,
          "b_c_differing_bytes": 0,
          "n_action": 119150,
      },
      "D3e numerical ledger drifted",
  )
  _require(
      classification.get("last_exact_boundary") == {
          "checkpoint": "k_post_rope", "layer": 0
      }
      and classification.get("first_red_boundary", {}).get("checkpoint") ==
      "rpa_output"
      and classification.get("first_red_boundary", {}).get("layer") == 0,
      "D3e tensor boundary drifted",
  )
  anchors = classification.get("anchors", [])
  _require(len(anchors) == 1, "D3e canonical-action anchor count drifted")
  anchor = anchors[0]
  binding = anchor.get("source_request_binding", {})
  target_sha = anchor.get("token_prefix_sha256")
  _require(
      anchor.get("source_row") == 217
      and anchor.get("completion_position") == 0
      and anchor.get("source_position") == 1225
      and anchor.get("a", {}).get("call_index") == 83
      and anchor.get("a", {}).get("record_geometry", {}).get(
          "layer_fingerprint_shape"
      ) == [2048, 1, 15, 8]
      and anchor.get("a", {}).get("record_geometry", {}).get(
          "final_fingerprint_shape"
      ) == [2048, 8],
      "D3e anchor geometry drifted",
  )
  _require(
      binding.get("status") == "UNIQUE_FUTURE_PREFIX_BINDING"
      and binding.get("anchor_prefix_tokens") == 1226
      and binding.get("required_disambiguation_prefix_tokens") == 1227
      and binding.get("selected_proof_prefix_tokens") == 1300
      and len(binding.get("candidates", [])) == 8
      and isinstance(target_sha, str)
      and len(target_sha) == 64,
      "D3e source-request binding drifted",
  )
  receipt_key = (
      f"A:{anchor['a']['call_index']}:"
      f"{binding['selected_request_id']}"
  )
  receipt = classification.get("replay_ledger_receipts", {}).get(
      receipt_key, {}
  )
  _require(
      receipt.get("block_size") == 16
      and receipt.get("logical_blocks_before") == 76
      and receipt.get("logical_blocks_after") == 77
      and receipt.get("num_computed_tokens") == 1216
      and receipt.get("scheduled_tokens") == 10
      and receipt.get("num_tokens") == 1226
      and len(receipt.get("physical_pages", [])) == 77,
      "D3e page receipt drifted",
  )
  _require(
      remote.get("status") == "PARTIAL_ROUNDS_RECOVERED"
      and remote.get("source_commit") == offline.get("runtime_source_commit"),
      "D3e remote partial-evidence boundary drifted",
  )
  return {
      "schema": "m15-attempt18-e0-admission-v1",
      "status": "E0_PREPARATION_ADMITTED",
      "runtime_source_commit": offline["runtime_source_commit"],
      "analysis_commit": offline["analysis_commit"],
      "d3e_gate": offline["reclassification_gate"],
      "decision_scope": offline["decision_scope"],
      "last_exact_boundary": classification["last_exact_boundary"],
      "first_red_boundary": {
          "checkpoint": classification["first_red_boundary"]["checkpoint"],
          "layer": classification["first_red_boundary"]["layer"],
      },
      "shape_ledger": {
          "layer": [2048, 1, 15, 8],
          "final": [2048, 8],
      },
      "target_prefix": {
          "tokens": binding["anchor_prefix_tokens"],
          "sha256": target_sha,
          "aliases": len(binding["candidates"]),
          "block_size": receipt["block_size"],
          "logical_pages": receipt["logical_blocks_after"],
          "observer_layer": 0,
          "observer_page_bound": 96,
      },
      "alignment": alignment,
      "evidence_manifest_sha256": _sha256(manifest),
      "evidence_members": entries,
      "pinned_exact_image_required": True,
      "launch_authorized": False,
      "numerical_repair_authorized": False,
      "target_executed": False,
      "remote_mutation": False,
      "claim_ceiling": (
          "This admits source preparation only. A fresh pinned exact-image "
          "gate and separate target-launch approval remain mandatory."
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evidence", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  report = review(args.evidence)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(
      "M15_E0_ADMISSION_PASS "
      f"gate={report['d3e_gate']} layer=0 "
      f"prefix_tokens={report['target_prefix']['tokens']} "
      f"aliases={report['target_prefix']['aliases']} "
      "launch_authorized=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
