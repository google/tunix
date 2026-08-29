#!/usr/bin/env python3
"""Reclassify the sealed Attempt-17 treatment bundle without target access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Any


class M15Attempt17ReviewError(RuntimeError):
  pass


SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = SCRIPT_DIR / "classify_m15_apc_wide_seam.py"
_CLASSIFIER_SPEC = importlib.util.spec_from_file_location(
    "classify_m15_apc_wide_seam_for_attempt17", CLASSIFIER_PATH
)
assert _CLASSIFIER_SPEC and _CLASSIFIER_SPEC.loader
CLASSIFIER = importlib.util.module_from_spec(_CLASSIFIER_SPEC)
_CLASSIFIER_SPEC.loader.exec_module(CLASSIFIER)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise M15Attempt17ReviewError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_member_name(name: str) -> bool:
  value = PurePosixPath(name)
  return bool(name) and not value.is_absolute() and ".." not in value.parts


def _extract_verified_bundle(bundle: Path, destination: Path) -> None:
  _require(bundle.is_file() and bundle.stat().st_size > 0,
           f"Attempt-17 bundle is absent or empty: {bundle}")
  with tarfile.open(bundle, mode="r:*") as archive:
    members = archive.getmembers()
    names = [item.name for item in members]
    _require(len(names) == len(set(names)), "Attempt-17 bundle has duplicate members")
    _require(all(_safe_member_name(name) for name in names),
             "Attempt-17 bundle has an unsafe member name")
    _require(all(item.isfile() for item in members),
             "Attempt-17 bundle contains a non-regular member")
    destination.mkdir(mode=0o700)
    for item in members:
      source = archive.extractfile(item)
      _require(source is not None, f"Attempt-17 bundle member is unreadable: {item.name}")
      target = destination / PurePosixPath(item.name)
      target.parent.mkdir(parents=True, exist_ok=True)
      with target.open("xb") as stream:
        shutil.copyfileobj(source, stream, length=1024 * 1024)

  manifest_path = destination / "SHA256SUMS"
  _require(manifest_path.is_file(), "Attempt-17 bundle lacks internal SHA256SUMS")
  manifest: dict[str, str] = {}
  for line in manifest_path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(
        separator == "  " and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and _safe_member_name(name) and name not in manifest,
        "Attempt-17 bundle manifest is invalid",
    )
    manifest[name] = digest
  extracted = {
      str(path.relative_to(destination))
      for path in destination.rglob("*")
      if path.is_file() and path != manifest_path
  }
  _require(set(manifest) == extracted,
           "Attempt-17 bundle members differ from its internal manifest")
  for name, digest in manifest.items():
    _require(_sha256(destination / name) == digest,
             f"Attempt-17 bundle member changed: {name}")


def review(
    *,
    bundle: Path,
    expected_classification: Path,
    source_commit: str,
    analysis_commit: str,
    output: Path,
    scratch_parent: Path,
    core_summary: Path | None = None,
) -> dict[str, Any]:
  _require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
           "Attempt-17 source commit must be a full lowercase SHA")
  _require(re.fullmatch(r"[0-9a-f]{40}", analysis_commit) is not None,
           "Attempt-17 analysis commit must be a full lowercase SHA")
  _require(expected_classification.is_file(),
           "committed Attempt-17 classification is absent")
  _require(scratch_parent.is_dir(), "Attempt-17 scratch parent is absent")
  _require(not output.exists(), f"refusing to overwrite Attempt-17 review: {output}")
  core = None
  if core_summary is not None:
    _require(core_summary.is_file(), "Attempt-17 remote multiround summary is absent")
    core = json.loads(core_summary.read_text(encoding="utf-8"))
    _require(
        core.get("schema") == "m15-apc-multiround-small-return-v1"
        and core.get("source_commit") == source_commit,
        "Attempt-17 remote multiround summary identity drifted",
    )
    on_rounds = core.get("arms", {}).get("on", {}).get("rounds", ())
    _require(
        len(on_rounds) == 3
        and on_rounds[0].get("status") == "SEALED"
        and on_rounds[0].get("classification")
        == "M15_INTERNAL_FIRST_RED_CANDIDATE_SET",
        "Attempt-17 remote treatment round 0 is not the sealed candidate set",
    )
  with tempfile.TemporaryDirectory(
      prefix="m15-attempt17-review.", dir=scratch_parent
  ) as scratch_text:
    extracted = Path(scratch_text) / "bundle"
    _extract_verified_bundle(bundle, extracted)
    bundled_classification = extracted / "classification.json"
    _require(
        bundled_classification.read_bytes() == expected_classification.read_bytes(),
        "remote Attempt-17 classification differs from the committed receipt",
    )
    original = json.loads(bundled_classification.read_text(encoding="utf-8"))
    _require(
        original.get("classification") == "M15_INTERNAL_FIRST_RED_CANDIDATE_SET"
        and original.get("gate") == "FIRST_RED_CANDIDATE_SET",
        "Attempt-17 committed treatment is not the expected candidate set",
    )
    expected_layer = int(original.get("expected_layer", -1))
    _require(0 <= expected_layer < 36,
             "Attempt-17 committed treatment has an invalid expected layer")
    capsules = sorted((extracted / "capsules").glob("capsule-*.npz"))
    _require(capsules, "Attempt-17 bundle lacks a mismatch capsule")
    reclassification = CLASSIFIER.classify(
        directory=extracted / "records",
        alignment_report=extracted / "pre-alignment.jsonl",
        capsules=capsules,
        mode="full",
        arm="on",
        replay_ledger=extracted / "m15-replay-envelope.jsonl",
        expected_layer=expected_layer,
        require_first_action=True,
    )

  gate = str(reclassification.get("gate"))
  _require(gate in ("FIRST_RED_LOCALIZED", "FIRST_RED_CANDIDATE_SET"),
           f"Attempt-17 offline reclassification returned an invalid gate: {gate}")
  status = (
      "FIRST_RED_LOCALIZED"
      if gate == "FIRST_RED_LOCALIZED" else "FIRST_RED_CANDIDATE_SET_PRESERVED"
  )
  binding_statuses = sorted({
      str(anchor.get("source_request_binding", {}).get("status", "ABSENT"))
      for anchor in reclassification.get("anchors", ())
  })
  output.mkdir(parents=True, mode=0o700)
  classification_output = output / "D36_RECLASSIFICATION.json"
  classification_output.write_text(
      json.dumps(reclassification, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
  )
  summary = {
      "schema": "m15-attempt17-d36-offline-review-v1",
      "status": status,
      "runtime_source_commit": source_commit,
      "analysis_commit": analysis_commit,
      "bundle_sha256": _sha256(bundle),
      "committed_classification_sha256": _sha256(expected_classification),
      "classifier_source_sha256": _sha256(CLASSIFIER_PATH),
      "original_classification": original["classification"],
      "original_gate": original["gate"],
      "reclassification": reclassification["classification"],
      "reclassification_gate": gate,
      "decision_scope": reclassification.get("decision_scope"),
      "all_join_mixed_first_difference_signatures": reclassification.get(
          "all_join_mixed_first_difference_signatures"
      ),
      "source_request_binding_statuses": binding_statuses,
      "target_executed": False,
      "remote_mutation": False,
      "numerical_repair_authorized": False,
      "pinned_exact_image_required": gate == "FIRST_RED_LOCALIZED",
      "remote_core_status": core.get("status") if core is not None else None,
      "remote_core_summary_sha256": (
          _sha256(core_summary) if core_summary is not None else None
      ),
      "claim_ceiling": (
          "This is an offline reclassification of the immutable Attempt-17 "
          "treatment bundle. It changes no runtime or numerical path and is "
          "not a fresh target execution. A localized completion-position-zero "
          "interval still requires independent shape/coordinate review and "
          "the pinned exact-image gate before any Phase-E discussion."
      ),
  }
  summary_path = output / "D36_OFFLINE_REVIEW.json"
  summary_path.write_text(
      json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  names = ["D36_OFFLINE_REVIEW.json", "D36_RECLASSIFICATION.json"]
  if core_summary is not None:
    remote_summary_name = "REMOTE_MULTIROUND_SUMMARY.json"
    (output / remote_summary_name).write_bytes(core_summary.read_bytes())
    names.append(remote_summary_name)
  (output / "SHA256SUMS").write_text(
      "".join(f"{_sha256(output / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bundle", type=Path, required=True)
  parser.add_argument("--expected-classification", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--analysis-commit", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--scratch-parent", type=Path, default=Path("/tmp"))
  parser.add_argument("--core-summary", type=Path)
  args = parser.parse_args()
  result = review(
      bundle=args.bundle,
      expected_classification=args.expected_classification,
      source_commit=args.source_commit,
      analysis_commit=args.analysis_commit,
      output=args.output,
      scratch_parent=args.scratch_parent,
      core_summary=args.core_summary,
  )
  print(
      "M15_D36_OFFLINE_REVIEW_COMPLETE "
      f"status={result['status']} "
      f"gate={result['reclassification_gate']} "
      f"output={args.output}"
  )


if __name__ == "__main__":
  main()
