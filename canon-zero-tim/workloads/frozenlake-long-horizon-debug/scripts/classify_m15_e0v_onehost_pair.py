#!/usr/bin/env python3
"""Classify a matched E0v exact-TiTO one-host APC pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


_ARM_FILES = {
    "RUN_CONTRACT.json",
    "alignment.classification.json",
    "diagnostic_round",
    "pre_alignment.jsonl",
    "raw.log",
    "source.diff",
    "tito.classification.json",
}


class OnehostPairError(RuntimeError):
  """Raised when one-host evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise OnehostPairError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict:
  _require(path.is_file(), f"missing JSON artifact: {path.name}")
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON artifact is not an object: {path.name}")
  return value


def _verify_arm_manifest(root: Path) -> str:
  manifest = root / "SHA256SUMS"
  _require(manifest.is_file(), f"missing arm manifest: {root.name}")
  observed = {}
  for line in manifest.read_text(encoding="ascii").splitlines():
    parts = line.split("  ", 1)
    _require(len(parts) == 2, f"malformed manifest line in {root.name}")
    digest, name = parts
    _require(
        len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        f"malformed digest in {root.name}",
    )
    _require(
        name not in observed and name == Path(name).name,
        f"unsafe or duplicate manifest name in {root.name}: {name}",
    )
    observed[name] = digest
  _require(set(observed) == _ARM_FILES, f"arm artifact inventory drifted: {root.name}")
  for name, digest in observed.items():
    _require(_sha256(root / name) == digest, f"arm artifact hash drifted: {root.name}/{name}")
  return _sha256(manifest)


def _classify_arm(root: Path, arm: str) -> dict:
  manifest_sha = _verify_arm_manifest(root)
  contract = _load_json(root / "RUN_CONTRACT.json")
  alignment = _load_json(root / "alignment.classification.json")
  tito = _load_json(root / "tito.classification.json")
  raw_sha = _sha256(root / "raw.log")

  _require(contract.get("schema") == "m15-e0v-tito-onehost-arm-v1", "arm contract schema drifted")
  _require(contract.get("arm") == arm, f"arm contract identity drifted: {arm}")
  _require(contract.get("apc") == (1 if arm == "on" else 0), f"APC selector drifted: {arm}")
  _require(contract.get("topology") == "DP1xTP4", "one-host topology drifted")
  _require(contract.get("rounds") == 3, "one-host round count drifted")
  _require(contract.get("docker_exit") == 42, "one-host controlled exit drifted")
  _require(contract.get("backward") == 0 and contract.get("optimizer_commits") == 0,
           "one-host execution escaped zero-backward scope")
  _require(contract.get("m15_token_continuity") == "exact", "one-host TiTO selector drifted")

  _require(
      alignment.get("schema") == "m15-e0v-tito-onehost-arm-classification-v1",
      f"alignment classification schema drifted: {arm}",
  )
  expected_statuses = (
      {"CONTROL_GREEN"}
      if arm == "off"
      else {"TREATMENT_EXACT", "TREATMENT_RED"}
  )
  _require(alignment.get("status") in expected_statuses,
           f"alignment classification failed: {arm}")
  _require(alignment.get("records") == 3, f"alignment round count drifted: {arm}")
  _require(alignment.get("diagnostic_rounds") == [0, 1, 2], f"alignment rounds drifted: {arm}")
  a_b = alignment.get("a_b_differing_bytes")
  _require(
      isinstance(a_b, list) and len(a_b) == 3
      and all(isinstance(value, int) and value >= 0 for value in a_b),
      f"A-B counters are malformed: {arm}",
  )
  if arm == "off":
    _require(a_b == [0, 0, 0], "APC-off control A-B is red")
  elif alignment.get("status") == "TREATMENT_EXACT":
    _require(a_b == [0, 0, 0], "treatment exact status contradicts A-B counters")
  else:
    _require(any(value > 0 for value in a_b), "treatment red status has no A-B difference")
  _require(alignment.get("b_c_differing_bytes") == [0, 0, 0], f"B-C is red: {arm}")
  b_receipts = alignment.get("b_full_reset_receipt_counts")
  _require(
      isinstance(b_receipts, list) and len(b_receipts) == 3
      and all(int(value) > 0 for value in b_receipts),
      f"B full-reset receipt coverage drifted: {arm}",
  )
  _require(alignment.get("raw_sha256") == raw_sha, f"alignment raw binding drifted: {arm}")
  if arm == "on":
    _require(float(alignment.get("max_prefix_cache_hit_rate_percent") or 0.0) > 0.0,
             "APC-on one-host arm observed no cache hit")

  _require(tito.get("status") == "PASS", f"TiTO postflight failed: {arm}")
  _require(tito.get("scope") == "onehost" and tito.get("arm") == arm,
           f"TiTO identity drifted: {arm}")
  _require(tito.get("topology") == "DP1xTP4", f"TiTO topology drifted: {arm}")
  _require(tito.get("diagnostic_rounds") == 3, f"TiTO rounds drifted: {arm}")
  counts = tito.get("round_receipt_counts")
  _require(isinstance(counts, list) and len(counts) == 3 and all(int(value) > 0 for value in counts),
           f"TiTO receipt coverage drifted: {arm}")
  _require(tito.get("different_or_malformed_receipts") == 0, f"TiTO mismatch observed: {arm}")
  _require(tito.get("run_log_sha256") == raw_sha, f"TiTO raw binding drifted: {arm}")

  return {
      "arm": arm,
      "manifest_sha256": manifest_sha,
      "raw_sha256": raw_sha,
      "source_commit": contract.get("source_commit"),
      "source_diff_sha256": contract.get("source_diff_sha256"),
      "image_id": contract.get("image_id"),
      "a_b_differing_bytes": a_b,
      "b_c_differing_bytes": alignment["b_c_differing_bytes"],
      "b_full_reset_receipt_counts": b_receipts,
      "max_prefix_cache_hit_rate_percent": alignment.get(
          "max_prefix_cache_hit_rate_percent"
      ),
      "round_receipt_counts": counts,
      "total_exact_equal_receipts": tito.get("total_exact_equal_receipts"),
      "contract": contract,
  }


def classify(root: Path) -> dict:
  _require(root.is_dir(), "one-host pair directory is absent")
  arms = [_classify_arm(root / arm, arm) for arm in ("off", "on")]
  off, on = arms
  for field in ("source_commit", "source_diff_sha256", "image_id"):
    _require(off[field] and off[field] == on[field], f"matched-pair {field} drifted")
  normalized = []
  for row in arms:
    contract = dict(row["contract"])
    contract["arm"] = "<ARM>"
    contract["apc"] = "<APC>"
    contract.pop("docker_exit", None)
    contract.pop("elapsed_seconds", None)
    normalized.append(contract)
  _require(normalized[0] == normalized[1], "one-host arms differ beyond APC treatment")
  for row in arms:
    row.pop("contract")
  treatment_exact = on["a_b_differing_bytes"] == [0, 0, 0]
  return {
      "schema": "m15-e0v-tito-onehost-pair-v1",
      "status": (
          "ONEHOST_PAIR_EXACT" if treatment_exact else "ONEHOST_RED_REPRODUCED"
      ),
      "topology": "DP1xTP4",
      "rounds_per_arm": 3,
      "source_commit": off["source_commit"],
      "source_diff_sha256": off["source_diff_sha256"],
      "image_id": off["image_id"],
      "arms": arms,
      "control_a_b_zero": True,
      "treatment_a_b_zero": treatment_exact,
      "both_b_c_zero": True,
      "tito_exact_both_arms": True,
      "backward": 0,
      "optimizer_commits": 0,
      "historical_1226_prefix_reused": False,
      "target_executed": False,
      "target_pass": False,
      "numerical_repair_authorized": False,
      "claim": "one-host carrier only; DP8xTP8 target remains unexecuted",
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  try:
    report = classify(args.root)
  except (OSError, UnicodeError, json.JSONDecodeError, OnehostPairError) as error:
    print(f"[M15.E0V.ONEHOST] INCONCLUSIVE {error}")
    return 2
  args.output.write_text(
      json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(
      f"[M15.E0V.ONEHOST] {report['status']} topology=DP1xTP4 arms=2 "
      "rounds=3/3,3/3 B-C=0/0 tito_exact=1 "
      "backward=0 optimizer_commits=0 target_executed=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
