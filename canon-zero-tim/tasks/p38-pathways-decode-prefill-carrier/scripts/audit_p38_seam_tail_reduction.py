#!/usr/bin/env python3
"""Independently audit a P38s18r2 immutable-round seam/tail reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import classify_p38_seam as seam
import reduce_p38_seam_evidence as base
import reduce_p38_seam_tail_evidence as reducer


class SeamTailBundleAuditError(RuntimeError):
  pass


_LEGACY_CLASSIFIER_COMPATIBILITY = {
    (
        "1cd458a09d792d558b3d107689643be377ab7cfb6d3fe13a434cd6163b210c1a",
        "08bc4ed3e0e8651a58aced44749163ed598ca4d395b73e0cd110bc0647c808ce",
    ),
}


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SeamTailBundleAuditError(message)


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
  expected: dict[str, str] = {}
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


def _verify_inventory(
    root: Path,
    directory_name: str,
    entries: Any,
    prefixes: tuple[str, ...],
) -> None:
  _require(isinstance(entries, list) and entries,
           f"{directory_name} manifest inventory is empty")
  expected = set()
  for entry in entries:
    _require(isinstance(entry, dict),
             f"{directory_name} manifest entry is invalid")
    relative = Path(str(entry.get("path", "")))
    _require(
        len(relative.parts) == 2
        and relative.parts[0] == directory_name
        and relative.name.startswith(prefixes)
        and relative.suffix in (".json", ".npz"),
        f"invalid {directory_name} manifest path: {relative}",
    )
    target = (root / relative).resolve()
    _require(target.parent == (root / directory_name).resolve(),
             f"{directory_name} manifest path escaped: {relative}")
    _require(target.is_file(), f"bundle file is absent: {relative}")
    _require(_sha256(target) == entry.get("sha256"),
             f"bundle file SHA differs from manifest: {relative}")
    _require(target.stat().st_size == int(entry.get("bytes", -1)),
             f"bundle file size differs from manifest: {relative}")
    expected.add(relative.name)
  actual = {
      path.name for path in (root / directory_name).iterdir()
      if path.is_file() and path.name.startswith(prefixes)
      and path.suffix in (".json", ".npz")
  }
  _require(actual == expected,
           f"{directory_name} file inventory differs from manifest")


def _expected_required(
    root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[tuple[int, bytes, str]]]:
  capsule_entries = manifest.get("capsules")
  _require(isinstance(capsule_entries, list) and len(capsule_entries) == 1,
           "bundle must contain exactly one immutable-round capsule")
  capsule_entry = capsule_entries[0]
  capsule = root / str(capsule_entry.get("path", ""))
  _require(capsule.is_file(), "bundle mismatch capsule is absent")
  _require(_sha256(capsule) == capsule_entry.get("sha256"),
           "bundle mismatch capsule SHA drifted")
  points = seam._red_points([capsule])
  required = {
      base._key(point["diagnostic_round"], point["token_prefix_sha256"], arm)
      for point in points for arm in ("A", "B")
  }
  return points, required


def audit(root: Path) -> dict[str, Any]:
  root = root.resolve()
  _require(root.is_dir(), f"bundle directory is absent: {root}")
  file_count = _verify_sha_inventory(root)
  manifest = _load_json(root / "REDUCTION_MANIFEST.json")
  target_identity_required = (
      manifest.get("tail_target_identity_required") is True)
  ambiguity = _load_json(root / "AMBIGUITY_AUDIT.json")
  selection = _load_json(root / "SNAPSHOT_SELECTION.json")
  verdict = _load_json(root / "verdict.json")

  _require(manifest.get("schema") == "p38-seam-reduction-v2",
           "bundle is not a v2 seam reduction")
  _require(manifest.get("require_tail") is True,
           "bundle did not require terminal-tail evidence")
  _require(ambiguity.get("schema") == "p38-seam-tail-ambiguity-audit-v1",
           "bundle seam/tail ambiguity schema drifted")
  _require(selection.get("schema") == "p38-immutable-round-selection-v1",
           "bundle immutable-round selection schema drifted")
  _require(verdict.get("schema") == "p38-seam-tail-reduction-verdict-v1",
           "bundle seam/tail verdict schema drifted")
  source_uri = str(manifest.get("source_gcs_uri", "")).rstrip("/")
  _require(source_uri.startswith("gs://") and source_uri,
           "bundle source GCS URI is empty or invalid")
  _require(
      source_uri == str(selection.get("selected_source_gcs_uri", "")).rstrip("/"),
      "bundle source URI differs from immutable-round selection",
  )
  _require(selection.get("selection_complete") is True,
           "immutable source round was not selected")
  source_complete = _load_json(root / "SOURCE_ROUND_COMPLETE.json")
  source_inventory = _load_json(root / "SOURCE_ROUND_INVENTORY.json")
  _require(
      source_complete.get("schema") == "canon-p38-round-completion-v1"
      and source_complete.get("status") == "sealed-and-verified"
      and source_complete.get("source_commit") == manifest.get("source_commit")
      and source_complete.get("manifest_sha256")
      == _sha256(root / "SOURCE_SHA256SUMS"),
      "bundle source completion contract drifted",
  )
  _require(
      source_inventory.get("schema") == "canon-p38-round-stage-v1"
      and source_inventory.get("diagnostic_round")
      == source_complete.get("diagnostic_round"),
      "bundle source inventory contract drifted",
  )
  _require(
      manifest.get("source_snapshot_manifest_sha256")
      == selection.get("source_manifest_sha256")
      == _sha256(root / "SOURCE_SHA256SUMS"),
      "bundle source-manifest provenance drifted",
  )
  _require(
      manifest.get("object_listing_sha256")
      == selection.get("listing_sha256")
      == _sha256(root / "OBJECT_LISTING.txt"),
      "bundle object-listing provenance drifted",
  )
  _require(manifest.get("source_round_complete_sha256")
           == _sha256(root / "SOURCE_ROUND_COMPLETE.json"),
           "bundle source completion provenance drifted")
  _require(manifest.get("source_round_inventory_sha256")
           == _sha256(root / "SOURCE_ROUND_INVENTORY.json"),
           "bundle source inventory provenance drifted")
  analysis_commit = (root / "analysis_source_commit.txt").read_text(
      encoding="utf-8").strip()
  _require(re.fullmatch(r"[0-9a-f]{40}", analysis_commit) is not None
           and analysis_commit == manifest.get("analysis_source_commit"),
           "bundle analysis source commit drifted")
  classifier_line = (root / "classifier_source.sha256").read_text(
      encoding="utf-8").strip().split(maxsplit=1)
  _require(len(classifier_line) == 2
           and classifier_line[1] == "classify_p38_seam.py"
           and classifier_line[0]
           == manifest.get("classifier_source_sha256"),
           "bundle classifier source SHA drifted")
  active_classifier_sha = _sha256(Path(seam.__file__).resolve())
  if target_identity_required:
    _require(classifier_line[0] == active_classifier_sha,
             "auditor classifier source differs from the bundle classifier")
  else:
    _require(
        classifier_line[0] == active_classifier_sha
        or (classifier_line[0], active_classifier_sha)
        in _LEGACY_CLASSIFIER_COMPATIBILITY,
        "auditor classifier source differs from the legacy-compatible set",
    )
  source_manifest_entries = base._manifest_entries(
      root / "SOURCE_SHA256SUMS")
  source_names = {relative for _, relative in source_manifest_entries}
  listing_names = reducer._listing_names(
      root / "OBJECT_LISTING.txt", source_uri)
  _require(
      listing_names == source_names | {"ROUND_COMPLETE.json", "SHA256SUMS"},
      "bundled source listing differs from bundled source manifest",
  )
  _require(
      len(source_manifest_entries) == manifest.get("source_snapshot_files")
      and len(listing_names) == manifest.get("source_object_count"),
      "bundled source inventory totals drifted",
  )
  _require(
      source_inventory.get("seam_records") == manifest.get("source_seam_records")
      and source_inventory.get("tail_records")
      == manifest.get("source_tail_records"),
      "bundled source observer totals drifted",
  )

  _verify_inventory(
      root, "records", manifest.get("record_files"), ("p38_seam_",))
  _verify_inventory(
      root, "records", manifest.get("tail_record_files"), ("p38_tail_",))
  _verify_inventory(
      root, "candidates", manifest.get("candidate_record_files"),
      ("p38_seam_", "p38_tail_"),
  )
  points, required = _expected_required(root, manifest)
  _require(len(points) == int(manifest.get("red_points", -1))
           and len(required) == int(manifest.get("required_arm_keys", -1)),
           "bundle capsule red-point totals drifted")

  candidate_dir = root / "candidates"
  seam_matches, _, _ = base._scan_records(
      candidate_dir, str(manifest.get("observer_mode")), required)
  tail_matches, _, _ = reducer._scan_tail_records(candidate_dir, required)
  seam_resolution = reducer._resolve_matches(seam_matches)
  required_tail = (
      reducer._required_tail_keys(points, required)
      if target_identity_required else None)
  tail_resolution = (
      reducer._resolve_tail_matches(tail_matches, required_tail)
      if required_tail is not None
      else reducer._resolve_matches(tail_matches)
  )
  _require(manifest.get("join_entries") == seam_resolution["join_entries"],
           "seam alias decisions differ when independently reproduced")
  _require(manifest.get("tail_join_entries")
           == tail_resolution["join_entries"],
           "tail alias decisions differ when independently reproduced")
  _require(ambiguity.get("seam") == seam_resolution,
           "seam ambiguity audit differs when independently reproduced")
  _require(ambiguity.get("tail") == tail_resolution,
           "tail ambiguity audit differs when independently reproduced")
  if target_identity_required:
    _require(manifest.get("required_tail_keys") == len(required_tail),
             "target-aware required-tail total drifted")
    _require(
        manifest.get("tail_target_mismatch_candidates")
        == tail_resolution["target_mismatch_candidates"],
        "tail target-mismatch audit differs when independently reproduced",
    )
  combined_unmatched = [
      {"observer": "seam", **entry}
      for entry in seam_resolution["unmatched_keys"]
  ] + [
      {"observer": "tail", **entry}
      for entry in tail_resolution["unmatched_keys"]
  ]
  combined_conflicts = [
      {"observer": "seam", **entry}
      for entry in seam_resolution["payload_conflict_keys"]
  ] + [
      {"observer": "tail", **entry}
      for entry in tail_resolution["payload_conflict_keys"]
  ]
  _require(
      ambiguity.get("unmatched_keys") == combined_unmatched
      and manifest.get("unmatched_keys") == combined_unmatched,
      "combined seam/tail missing-key audit drifted",
  )
  _require(
      ambiguity.get("payload_conflict_keys") == combined_conflicts
      and manifest.get("ambiguous_keys") == combined_conflicts,
      "combined seam/tail conflict audit drifted",
  )
  _require(
      manifest.get("selected_record_indices")
      == seam_resolution["selected_record_indices"]
      and manifest.get("selected_tail_record_indices")
      == tail_resolution["selected_record_indices"],
      "bundle selected record indices drifted",
  )
  selection_complete = (
      seam_resolution["selection_complete"]
      and tail_resolution["selection_complete"])
  _require(selection_complete == manifest.get("selection_complete")
           == verdict.get("selection_complete"),
           "bundle reduction selection state drifted")
  _require(seam_resolution["matched_keys"]
           == manifest.get("matched_seam_keys"),
           "bundle matched seam-key total drifted")
  _require(tail_resolution["matched_keys"]
           == manifest.get("matched_tail_keys"),
           "bundle matched tail-key total drifted")

  classifier_rc = int((root / "classifier.rc").read_text(
      encoding="utf-8").strip())
  _require(classifier_rc == int(verdict.get("classifier_rc", -1)),
           "bundle classifier rc drifted")
  classification = None
  if selection_complete:
    capsules = [root / str(item["path"]) for item in manifest["capsules"]]
    try:
      classification = seam.classify(
          root / "records",
          capsules,
          str(manifest.get("observer_mode")),
          reduction_manifest=root / "REDUCTION_MANIFEST.json",
          require_tail=True,
      )
    except seam.SeamError as error:
      _require(classifier_rc == 2,
               "official classifier failed but bundle rc is not failure")
      _require(not (root / "classification.json").exists(),
               "failed classifier bundle contains a classification")
      _require((root / "classifier.stdout").read_text(encoding="utf-8") == "",
               "failed classifier bundle stdout is not empty")
      _require(
          (root / "classifier.stderr").read_text(encoding="utf-8")
          == f"{type(error).__name__}: {error}\n",
          "official classifier failure differs when reproduced",
      )
      expected_verdict = "INCONCLUSIVE_REMOTE_CLASSIFICATION"
    else:
      _require(classifier_rc == 0,
               "official classifier succeeded but bundle rc is not zero")
      committed = _load_json(root / "classification.json")
      _require(classification == committed,
               "official classifier output differs when reproduced")
      _require(
          (root / "classifier.stdout").read_text(encoding="utf-8")
          == json.dumps(classification, sort_keys=True) + "\n"
          and (root / "classifier.stderr").read_text(encoding="utf-8") == "",
          "successful classifier stdout/stderr differs from its result",
      )
      expected_verdict = (
          "PASS" if manifest.get("run_contract_complete")
          else "INCONCLUSIVE_PARTIAL_RUN")
  else:
    _require(classifier_rc == 4,
             "incomplete join did not retain classifier-not-run rc")
    _require(not (root / "classification.json").exists(),
             "incomplete join contains a classification")
    _require((root / "classifier.stdout").read_text(encoding="utf-8") == "",
             "incomplete join classifier stdout is not empty")
    _require(
        (root / "classifier.stderr").read_text(encoding="utf-8")
        == "classifier not run: seam/tail reduction join incomplete\n",
        "incomplete join classifier receipt drifted",
    )
    expected_verdict = "INCONCLUSIVE_REDUCTION_JOIN"
  _require(verdict.get("verdict") == expected_verdict,
           "bundle scientific verdict is inconsistent with reproduced state")
  _require(
      verdict.get("classification")
      == (classification.get("classification") if classification else None),
      "bundle verdict classification drifted",
  )
  _require(verdict.get("joined_red_points")
           == (classification.get("joined_red_points") if classification else 0),
           "bundle joined-red-point total drifted")

  report = {
      "schema": "p38-seam-tail-reduction-bundle-audit-v1",
      "bundle_integrity": "PASS",
      "scientific_verdict": expected_verdict,
      "source_gcs_uri": source_uri,
      "analysis_source_commit": analysis_commit,
      "red_points": len(points),
      "required_arm_keys": len(required),
      "matched_seam_keys": seam_resolution["matched_keys"],
      "matched_tail_keys": tail_resolution["matched_keys"],
      "seam_equivalent_alias_keys": len(
          seam_resolution["equivalent_alias_keys"]),
      "tail_equivalent_alias_keys": len(
          tail_resolution["equivalent_alias_keys"]),
      "seam_payload_conflict_keys": len(
          seam_resolution["payload_conflict_keys"]),
      "tail_payload_conflict_keys": len(
          tail_resolution["payload_conflict_keys"]),
      "classification": (
          classification.get("classification") if classification else None),
      "joined_red_points": (
          classification.get("joined_red_points") if classification else 0),
      "classifier_rc": classifier_rc,
      "sha_verified_files": file_count,
  }
  if target_identity_required:
    report["tail_target_mismatch_rows"] = sum(
        entry["candidate_count"]
        for entry in tail_resolution["target_mismatch_candidates"])
  return report


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bundle-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    report = audit(args.bundle_dir)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
  except (SeamTailBundleAuditError, reducer.SeamTailReductionError,
          base.ReductionError, seam.SeamError, OSError, ValueError) as error:
    print(f"[P38.SEAM_TAIL.AUDIT] REFUSING: {error}", file=sys.stderr)
    return 2
  print(
      "[P38.SEAM_TAIL.AUDIT] PASS "
      f"scientific_verdict={report['scientific_verdict']} "
      f"red_points={report['red_points']} "
      f"matched_seam_keys={report['matched_seam_keys']} "
      f"matched_tail_keys={report['matched_tail_keys']} "
      f"classifier_rc={report['classifier_rc']} "
      f"sha_verified_files={report['sha_verified_files']}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
