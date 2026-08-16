#!/usr/bin/env python3
"""Verify a returned P38 reduction bundle and reproduce its classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import classify_p38_seam as seam


class BundleAuditError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise BundleAuditError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  _require(path.is_file(), f"required bundle file is absent: {path.name}")
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"bundle JSON is not an object: {path.name}")
  return value


def _verify_sha_inventory(root: Path) -> int:
  manifest = root / "SHA256SUMS"
  _require(manifest.is_file(), "bundle SHA256SUMS is absent")
  expected = {}
  for line_number, raw in enumerate(
      manifest.read_text(encoding="utf-8").splitlines(), start=1
  ):
    parts = raw.split(maxsplit=1)
    _require(len(parts) == 2 and len(parts[0]) == 64,
             f"invalid bundle SHA line {line_number}")
    relative = parts[1].lstrip("*")
    path = Path(relative)
    _require(relative and not path.is_absolute() and ".." not in path.parts,
             f"unsafe bundle SHA path: {relative}")
    _require(relative not in expected, f"duplicate bundle SHA path: {relative}")
    expected[relative] = parts[0]
  actual = {
      path.relative_to(root).as_posix()
      for path in root.rglob("*")
      if path.is_file() and path != manifest
  }
  _require(set(expected) == actual,
           "bundle file inventory differs from SHA256SUMS")
  for relative, digest in expected.items():
    _require(_sha256(root / relative) == digest,
             f"bundle SHA failed: {relative}")
  return len(expected)


def audit(root: Path) -> dict[str, Any]:
  root = root.resolve()
  _require(root.is_dir(), f"bundle directory is absent: {root}")
  file_count = _verify_sha_inventory(root)
  manifest = _load_json(root / "REDUCTION_MANIFEST.json")
  verdict = _load_json(root / "verdict.json")
  ambiguity = _load_json(root / "AMBIGUITY_AUDIT.json")
  snapshot = _load_json(root / "SNAPSHOT_SELECTION.json")
  _require(manifest.get("schema") == "p38-seam-reduction-v2",
           "bundle is not a v2 seam reduction")
  _require(verdict.get("schema") == "p38-seam-reduction-verdict-v2",
           "bundle verdict schema drifted")
  _require(ambiguity.get("schema") == "p38-seam-ambiguity-audit-v1",
           "bundle ambiguity-audit schema drifted")
  _require(snapshot.get("schema") == "p38-live-snapshot-selection-v1",
           "bundle snapshot-selection schema drifted")
  _require(snapshot.get("selection_complete") is True,
           "bundle source snapshot was not admitted")
  _require(
      manifest.get("source_gcs_uri", "").rstrip("/")
      == snapshot.get("selected_source_gcs_uri", "").rstrip("/"),
      "bundle source URI differs from snapshot selection",
  )
  _require(manifest.get("capsule_rounds") == snapshot.get(
      "selected_capsule_rounds"),
      "bundle capsule rounds differ from snapshot selection")
  red_points = int(manifest.get("red_points", -1))
  required = int(manifest.get("required_arm_keys", -1))
  matched = int(manifest.get("matched_arm_keys", -1))
  _require(red_points > 0 and required == 2 * red_points,
           "bundle red-point/key totals are invalid")
  _require(
      ambiguity.get("required_arm_keys") == required
      and ambiguity.get("selection_complete")
      == manifest.get("selection_complete"),
      "bundle ambiguity totals differ from reduction manifest",
  )
  _require(verdict.get("red_points") == red_points,
           "bundle verdict red-point total drifted")
  _require(verdict.get("selection_complete") == manifest.get(
      "selection_complete"), "bundle verdict selection state drifted")
  _require(verdict.get("run_contract_complete") == manifest.get(
      "run_contract_complete"), "bundle verdict run-contract state drifted")

  classification = None
  if manifest.get("selection_complete") is True:
    _require(matched == required, "complete reduction did not match every arm key")
    _require(not manifest.get("unmatched_keys")
             and not manifest.get("ambiguous_keys"),
             "complete reduction retains unresolved keys")
    records = root / str(manifest.get("records_directory", "records"))
    capsules = [root / str(item["path"]) for item in manifest.get("capsules", ())]
    classification = seam.classify(
        records,
        capsules,
        str(manifest.get("observer_mode")),
        reduction_manifest=root / "REDUCTION_MANIFEST.json",
    )
    committed = _load_json(root / "classification.json")
    _require(classification == committed,
             "official classifier output differs from bundled classification")
    expected_verdict = (
        "PASS" if manifest.get("run_contract_complete")
        else "INCONCLUSIVE_PARTIAL_RUN")
  else:
    _require(matched < required, "incomplete reduction reports full key coverage")
    _require(bool(manifest.get("unmatched_keys"))
             or bool(manifest.get("ambiguous_keys")),
             "incomplete reduction has no recorded blocker")
    _require(not (root / "classification.json").exists(),
             "incomplete reduction must not contain a classification")
    expected_verdict = "INCONCLUSIVE_REDUCTION_JOIN"
  _require(verdict.get("verdict") == expected_verdict,
           "bundle verdict is inconsistent with reduction state")
  _require(
      verdict.get("classification")
      == (classification.get("classification") if classification else None),
      "bundle verdict classification drifted",
  )
  return {
      "schema": "p38-seam-reduction-bundle-audit-v1",
      "bundle_integrity": "PASS",
      "scientific_verdict": expected_verdict,
      "source_snapshot": snapshot.get("selected_snapshot"),
      "capsule_rounds": manifest.get("capsule_rounds"),
      "red_points": red_points,
      "matched_arm_keys": matched,
      "required_arm_keys": required,
      "equivalent_alias_keys": len(manifest.get("equivalent_alias_keys", ())),
      "payload_conflict_keys": len(manifest.get("ambiguous_keys", ())),
      "unmatched_keys": len(manifest.get("unmatched_keys", ())),
      "classification": (
          classification.get("classification") if classification else None),
      "sha_verified_files": file_count,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bundle-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    report = audit(args.bundle_dir)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  except (BundleAuditError, seam.SeamError, OSError, ValueError) as error:
    print(f"[P38.REDUCE.AUDIT] REFUSING: {error}", file=sys.stderr)
    return 2
  print(
      "[P38.REDUCE.AUDIT] PASS "
      f"scientific_verdict={report['scientific_verdict']} "
      f"red_points={report['red_points']} "
      f"matched_arm_keys={report['matched_arm_keys']} "
      f"sha_verified_files={report['sha_verified_files']}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
