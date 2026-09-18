#!/usr/bin/env python3
"""Build a read-only operator-return locator from the submitted d33 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml


class RecoveryContractError(RuntimeError):
  pass


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CAMPAIGN = "v1-apc-m15-attempt14-d33"
_MEMBERS = {
    "INCIDENT_REPORT.md",
    "p38_seam.classification_off.json",
    "p38_seam.classification_on.json",
    "receipt.json",
}
_GCS_ROOT = re.compile(
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    r"(?P<jobset>[a-z0-9-]+)/attempt-0"
)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise RecoveryContractError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_json(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON is not an object: {path.name}")
  return value


def _verify_submitted_evidence(root: Path) -> tuple[dict, str]:
  manifest_path = root / "SHA256SUMS"
  _require(manifest_path.is_file(), "submitted SHA256SUMS is absent")
  rows: dict[str, str] = {}
  for line in manifest_path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(
        separator == "  "
        and _SHA256.fullmatch(digest) is not None
        and Path(name).name == name
        and name not in rows,
        f"invalid submitted manifest row: {line!r}",
    )
    rows[name] = digest
  actual = {
      path.name for path in root.iterdir()
      if path.is_file() and path.name != "SHA256SUMS"
  }
  _require(set(rows) == _MEMBERS, "submitted manifest membership drifted")
  _require(actual == _MEMBERS, "submitted evidence directory membership drifted")
  for name, digest in rows.items():
    _require(_sha256(root / name) == digest, f"submitted SHA failed: {name}")
  return _load_json(root / "receipt.json"), _sha256(manifest_path)


def _arm_locator(receipt: dict, arm: str, source: str) -> tuple[str, str]:
  key = "control_arm_off" if arm == "off" else "treatment_arm_on"
  value = receipt.get(key)
  _require(isinstance(value, dict), f"submitted {arm} arm is absent")
  jobset = str(value.get("jobset_name", ""))
  expected = f"canon-v1-apc-m15-{arm}-d33-{source[:8]}"
  _require(jobset == expected, f"submitted {arm} JobSet identity drifted")
  uri = str(value.get("gcs_source_uri", ""))
  match = _GCS_ROOT.fullmatch(uri)
  _require(match is not None and match.group("jobset") == jobset,
           f"submitted {arm} GCS locator drifted")
  return jobset, uri


def build(evidence: Path, output: Path) -> dict:
  _require(evidence.is_dir(), f"submitted evidence is absent: {evidence}")
  _require(not output.exists(), f"refusing to overwrite recovery contract: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale partial recovery contract exists: {partial}")
  receipt, manifest_sha = _verify_submitted_evidence(evidence)
  _require(receipt.get("attempt") == 14, "submitted attempt is not 14")
  _require(receipt.get("campaign_root") == _CAMPAIGN,
           "submitted campaign identity drifted")
  source = str(receipt.get("source_commit", ""))
  _require(_SHA40.fullmatch(source) is not None, "submitted source is invalid")
  arms: dict[str, dict[str, str]] = {}
  for arm in ("off", "on"):
    jobset, uri = _arm_locator(receipt, arm, source)
    arms[arm] = {"jobset": jobset, "gcs_root": uri}
  recovery = {
      "schema": "m15-apc-attempt14-recovery-input-v1",
      "status": "LOCATOR_ONLY",
      "campaign_root": _CAMPAIGN,
      "source_commit": source,
      "submitted_manifest_sha256": manifest_sha,
      "submitted_receipt_sha256": _sha256(evidence / "receipt.json"),
      "jobsets": {arm: arms[arm]["jobset"] for arm in ("off", "on")},
      "claim_ceiling": (
          "This receipt locates the immutable d33 operator artifacts only. "
          "Numerical claims come exclusively from the recovered official classifiers."
      ),
  }
  partial.mkdir(parents=True)
  try:
    for arm in ("off", "on"):
      env = [
          {"name": "CANON_APC_M15_TARGET_DEBUG", "value": arm},
          {"name": "CANON_EXPECT_COMMIT", "value": source},
          {"name": "CANON_P38_GCS_PREFIX", "value": arms[arm]["gcs_root"]},
          {"name": "CANON_P38_DIAGNOSTIC_ROUNDS", "value": "3"},
          {"name": "CANON_P38_SEAM_OBSERVER", "value": "full"},
          {"name": "CANON_P38_SEAM_LAYER", "value": "0"},
          {"name": "CANON_P33_RUN_STAGE", "value": "backward-no-commit"},
          {"name": "CANON_P33_NO_COMMIT", "value": "1"},
      ]
      document = {
          "metadata": {"name": arms[arm]["jobset"]},
          "spec": {"replicatedJobs": [{"template": {"spec": {"template": {
              "spec": {"containers": [{"env": env}]}
          }}}}]},
      }
      (partial / f"jobset-v1-apc-m15-{arm}-full.yaml").write_text(
          yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
      )
    (partial / "RECOVERY_INPUT_RECEIPT.json").write_text(
        json.dumps(recovery, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(output)
  except Exception:
    for path in partial.glob("*"):
      path.unlink()
    partial.rmdir()
    raise
  return recovery


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evidence", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = build(args.evidence, args.output)
  except (OSError, ValueError, json.JSONDecodeError, RecoveryContractError) as exc:
    raise SystemExit(f"M15_D33_RECOVERY_CONTRACT_RED {exc}") from exc
  print(
      "M15_D33_RECOVERY_CONTRACT_PASS "
      f"source={result['source_commit']} jobsets=2 output={args.output}"
  )


if __name__ == "__main__":
  main()
