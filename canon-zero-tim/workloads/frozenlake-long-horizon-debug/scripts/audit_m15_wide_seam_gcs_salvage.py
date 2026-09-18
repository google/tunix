#!/usr/bin/env python3
"""Audit small M15 wide-seam artifacts downloaded from two GCS attempt roots.

The downloader deliberately leaves the token-bearing compact tar outside the
return package.  This analyzer verifies it in place and emits only the
classifier JSON, hashes, marker inventory, and a mechanical next decision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any


class SalvageError(RuntimeError):
  pass


_CLASSIFICATION_NAMES = (
    "seam-classification.json",
    "p38_seam.classification.json",
)
_MARKERS = ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json")
_SOURCE_RE = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SalvageError(message)


def _sha256_bytes(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
  return _sha256_bytes(path.read_bytes())


def _json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file(), f"{label} is absent: {path}")
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"{label} is not a JSON object")
  return value


def _manifest(path: Path) -> dict[str, str]:
  result: dict[str, str] = {}
  if not path.is_file():
    return result
  for line in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(separator == "  ", "root SHA256SUMS has an invalid row")
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             f"root SHA256SUMS has an invalid digest for {name}")
    _require(name and "/" not in name and name not in result,
             f"root SHA256SUMS has an unsafe or duplicate member: {name}")
    result[name] = digest
  return result


def _classification(root: Path, expected_arm: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
  candidates = [root / name for name in _CLASSIFICATION_NAMES if (root / name).is_file()]
  if not candidates:
    return None, {"present": False, "candidate_names": []}
  payloads = [path.read_bytes() for path in candidates]
  _require(all(payload == payloads[0] for payload in payloads[1:]),
           f"{expected_arm} classification aliases disagree byte-for-byte")
  value = json.loads(payloads[0])
  _require(
      isinstance(value, dict)
      and value.get("schema") == "m15-apc-wide-seam-classification-v1"
      and value.get("status") == "PASS"
      and value.get("arm") == expected_arm,
      f"{expected_arm} classification contract is invalid",
  )
  return value, {
      "present": True,
      "candidate_names": [path.name for path in candidates],
      "sha256": _sha256_bytes(payloads[0]),
      "bytes": len(payloads[0]),
  }


def _verify_bundle(
    path: Path, classification: dict[str, Any] | None
) -> dict[str, Any]:
  if not path.is_file():
    return {"present": False}
  payload = path.read_bytes()
  with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
    members = archive.getmembers()
    names = [member.name for member in members]
    for member in members:
      name = PurePosixPath(member.name)
      _require(
          not name.is_absolute()
          and ".." not in name.parts
          and member.isfile(),
          f"unsafe compact bundle member: {member.name}",
      )
    _require(len(names) == len(set(names)), "compact bundle has duplicate members")
    _require("SHA256SUMS" in names, "compact bundle lacks SHA256SUMS")
    manifest_payload = archive.extractfile("SHA256SUMS").read()
    checked = 0
    for line in manifest_payload.decode("ascii").splitlines():
      digest, separator, name = line.partition("  ")
      _require(separator == "  " and re.fullmatch(r"[0-9a-f]{64}", digest),
               f"compact bundle manifest row is invalid: {line}")
      _require(name in names and name != "SHA256SUMS",
               f"compact bundle manifest member is absent: {name}")
      member_payload = archive.extractfile(name).read()
      _require(_sha256_bytes(member_payload) == digest,
               f"compact bundle member SHA drifted: {name}")
      checked += 1
    _require(checked > 0, "compact bundle manifest is empty")
    receipt = json.loads(archive.extractfile("RECEIPT.json").read())
    _require(receipt.get("schema") == "m15-apc-wide-seam-bundle-v1",
             "compact bundle receipt schema drifted")
    if classification is not None:
      embedded = json.loads(archive.extractfile("classification.json").read())
      _require(embedded == classification,
               "compact bundle classification differs from returned classifier")
  return {
      "present": True,
      "sha256": _sha256_bytes(payload),
      "bytes": len(payload),
      "logical_files": len(names),
      "manifest_entries": checked,
      "classification": receipt.get("classification"),
      "observer_mode": receipt.get("observer_mode"),
      "arm": receipt.get("arm"),
      "internal_sha256_pass": True,
  }


def _audit_arm(root: Path, arm: str, output: Path) -> dict[str, Any]:
  manifest = _manifest(root / "SHA256SUMS")
  marker_status: dict[str, Any] = {}
  for name in _MARKERS:
    path = root / name
    if not path.is_file():
      marker_status[name] = {"present": False}
      continue
    value = _json(path, f"{arm} {name}")
    marker_status[name] = {
        "present": True,
        "sha256": _sha256(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "source_commit": value.get("source_commit"),
    }

  classification, classification_receipt = _classification(root, arm)
  if classification is not None:
    canonical = root / "seam-classification.json"
    manifest_binding = (
        canonical.is_file()
        and manifest.get(canonical.name) == _sha256(canonical)
    )
    classification_receipt["root_manifest_binding"] = manifest_binding
    destination = output / f"{arm}.classification.json"
    destination.write_text(
        json.dumps(classification, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
  bundle = _verify_bundle(root / "m15_wide_seam_bundle.tar", classification)
  evidence_bound = bool(
      classification_receipt.get("root_manifest_binding")
      or bundle.get("internal_sha256_pass")
  )
  inventory = []
  inventory_path = root / "remote-inventory.txt"
  if inventory_path.is_file():
    inventory = inventory_path.read_text(encoding="utf-8").splitlines()
  return {
      "arm": arm,
      "classification": classification_receipt,
      "classification_value": classification,
      "bundle": bundle,
      "evidence_bound": evidence_bound,
      "markers": marker_status,
      "root_manifest_present": bool(manifest),
      "root_manifest_entries": len(manifest),
      "remote_inventory": inventory,
  }


def _decision(
    off: dict[str, Any], on: dict[str, Any], source_conflicts: list[dict[str, str]]
) -> tuple[str, str]:
  if source_conflicts:
    return "SOURCE_MISMATCH", "verify the runtime source identity before using either classifier"
  off_value = off["classification_value"]
  on_value = on["classification_value"]
  if off_value is None:
    return "INCOMPLETE", "return a valid APC-off wide-seam classification"
  if not off["evidence_bound"]:
    return "INCOMPLETE", "bind the APC-off classifier to a root or compact-bundle manifest"
  if off_value.get("classification") != "M15_OBSERVER_CONTROL_EXACT":
    return "CONTROL_RED", "stop; the observed APC-off control is not exact"
  if on_value is None:
    return "INCOMPLETE", "return a valid APC-on wide-seam classification"
  if not on["evidence_bound"]:
    return "INCOMPLETE", "bind the APC-on classifier to a root or compact-bundle manifest"
  classification = str(on_value.get("classification"))
  if classification == "M15_LAYER_FIRST_RED_LOCALIZED":
    selected = on_value.get("selected_layer")
    _require(isinstance(selected, int) and 0 <= selected < 36,
             "layer classification lacks a valid selected_layer")
    return "LAYER_SELECTED", f"render full observer only at layer {selected}"
  if classification == "M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED":
    return "TAIL_SELECTED", "localize the reported terminal-tail interval"
  if classification == "M15_OBSERVER_TREATMENT_EXACT":
    return "TREATMENT_EXACT", "record one exact target observation; do not claim a repair"
  if classification == "M15_INTERNAL_FIRST_RED_LOCALIZED":
    return "FIRST_RED_LOCALIZED", "open the minimal Phase-E repair review"
  return "REVIEW_REQUIRED", f"review unregistered classification {classification}"


def audit(*, receipt_path: Path, off_root: Path, on_root: Path, output: Path) -> dict[str, Any]:
  _require(not output.exists(), f"refusing to overwrite salvage output: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale salvage partial exists: {partial}")
  receipt = _json(receipt_path, "Attempt-9 receipt")
  source_commit = str(receipt.get("source_commit", ""))
  _require(_SOURCE_RE.fullmatch(source_commit) is not None,
           "Attempt-9 receipt source commit is not a full SHA")
  partial.mkdir(parents=True)
  try:
    off = _audit_arm(off_root, "off", partial)
    on = _audit_arm(on_root, "on", partial)
    source_conflicts = []
    for arm_record in (off, on):
      for marker_name, marker in arm_record["markers"].items():
        marker_source = marker.get("source_commit")
        if marker_source not in (None, "unknown", source_commit):
          source_conflicts.append({
              "arm": arm_record["arm"],
              "marker": marker_name,
              "receipt_source_commit": source_commit,
              "marker_source_commit": str(marker_source),
          })
    status, next_action = _decision(off, on, source_conflicts)
    for arm in (off, on):
      arm.pop("classification_value", None)
    result = {
        "schema": "m15-apc-wide-seam-gcs-salvage-v1",
        "status": status,
        "attempt": int(receipt.get("attempt", -1)),
        "campaign_root": receipt.get("campaign_root"),
        "source_commit": source_commit,
        "receipt_sha256": _sha256(receipt_path),
        "source_commit_conflicts": source_conflicts,
        "off": off,
        "on": on,
        "next_action": next_action,
        "claim_ceiling": (
            "This return audits already-persisted small evidence only; it does "
            "not rerun TPU inference or repair APC numerics."
        ),
    }
    summary = partial / "SALVAGE_SUMMARY.json"
    summary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                       encoding="utf-8")
    packaging = partial / "PACKAGING.txt"
    packaging.write_text(
        "M15 wide-seam GCS salvage return\n"
        f"status={status}\n"
        f"next_action={next_action}\n"
        "token_bearing_bundle_returned=0\n"
        "remote_state_mutated=0\n",
        encoding="utf-8",
    )
    names = sorted(path.name for path in partial.iterdir() if path.is_file())
    manifest = partial / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{_sha256(partial / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    partial.replace(output)
    return result
  except Exception:
    shutil.rmtree(partial, ignore_errors=True)
    raise


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--receipt", type=Path, required=True)
  parser.add_argument("--off-root", type=Path, required=True)
  parser.add_argument("--on-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    result = audit(
        receipt_path=args.receipt,
        off_root=args.off_root,
        on_root=args.on_root,
        output=args.output,
    )
  except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError, SalvageError) as exc:
    raise SystemExit(f"M15_WIDE_SEAM_GCS_SALVAGE_RED {exc}") from exc
  print(
      "M15_WIDE_SEAM_GCS_SALVAGE_COMPLETE "
      f"status={result['status']} output={args.output}"
  )


if __name__ == "__main__":
  main()
