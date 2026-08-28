#!/usr/bin/env python3
"""Review a returned Attempt-13 inventory without querying GCS again.

This is a transport and contract review, not a numerical classifier. It keeps
the immutable receipt's ``seam_records`` separate from the physical shard
completion ``record_pairs`` and makes their difference explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


SOURCE_COMMIT = "7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
RECEIPT_SHA256 = "d1941c2de85050a5652bc5c6e809987f6bf72b996aa817371b08b43870835f95"
ARMS = {
    "off": {"field": "control_arm_off", "shards": 77},
    "on": {"field": "treatment_arm_on", "shards": 70},
}
INVENTORY_MEMBERS = {
    "D32_INVENTORY.json",
    "PACKAGING.txt",
    "off.objects.txt",
    "off.shard-completions.jsonl",
    "on.objects.txt",
    "on.shard-completions.jsonl",
}
SHARD_MEMBERS = ("SHARD_ARCHIVE.tar", "SHA256SUMS", "SHARD_COMPLETE.json")
SHA_RE = re.compile(r"[0-9a-f]{64}")


class Attempt13ReviewError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt13ReviewError(message)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as error:
    raise Attempt13ReviewError(f"{label} is invalid") from error
  _require(isinstance(value, dict), f"{label} is not an object")
  return value


def _manifest(path: Path) -> dict[str, str]:
  rows = {}
  for raw in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = raw.partition("  ")
    _require(
        separator == "  " and SHA_RE.fullmatch(digest) is not None
        and name and "/" not in name and name not in rows,
        "inventory manifest contains an unsafe row",
    )
    rows[name] = digest
  return rows


def _review_arm(
    *,
    arm: str,
    root: Path,
    inventory: dict[str, Any],
    receipt_arm: dict[str, Any],
) -> dict[str, Any]:
  expected_shards = int(ARMS[arm]["shards"])
  objects = (root / f"{arm}.objects.txt").read_text(encoding="utf-8").splitlines()
  _require(objects == sorted(set(objects)), f"{arm} object inventory is not unique")
  _require(not any("://" in row or row.startswith("/") for row in objects),
           f"{arm} object inventory exposes a remote or absolute path")
  expected_objects = {"PREFLIGHT.json"}
  for sequence in range(expected_shards):
    expected_objects.update(
        f"wide/shards/{sequence:06d}/{name}" for name in SHARD_MEMBERS
    )
  _require(set(objects) == expected_objects,
           f"{arm} object inventory is not exactly PREFLIGHT plus shard triples")

  completions = []
  for raw in (root / f"{arm}.shard-completions.jsonl").read_text(
      encoding="utf-8"
  ).splitlines():
    value = json.loads(raw)
    _require(isinstance(value, dict), f"{arm} completion row is not an object")
    completions.append(value)
  _require(
      [int(value.get("sequence", -1)) for value in completions]
      == list(range(expected_shards)),
      f"{arm} completion sequences are not contiguous",
  )
  for value in completions:
    _require(
        int(value.get("record_pairs", -1)) > 0
        and int(value.get("payload_bytes", -1)) > 0
        and SHA_RE.fullmatch(str(value.get("manifest_sha256", ""))) is not None
        and SHA_RE.fullmatch(str(value.get("archive_sha256", ""))) is not None
        and SHA_RE.fullmatch(str(value.get("completion_sha256", ""))) is not None,
        f"{arm} completion counts or hashes are invalid",
    )
  observed = sum(int(value["record_pairs"]) for value in completions)
  receipt_seam = int(receipt_arm.get("seam_records", -1))
  _require(receipt_seam > 0, f"{arm} immutable receipt seam count is invalid")
  recorded = int(
      inventory.get("shard_record_pairs", inventory.get("record_pairs", -1))
  )
  _require(recorded == observed, f"{arm} inventory record-pair sum drifted")
  _require(
      inventory.get("status") == "PASS"
      and inventory.get("jobset") == receipt_arm.get("jobset_name")
      and int(inventory.get("flat_shards", -1)) == expected_shards
      and int(inventory.get("object_count", -1)) == len(objects)
      and int(inventory.get("live_objects", -1)) == 0
      and int(inventory.get("wide_round_objects", -1)) == 0
      and inventory.get("live_absence_proven") is True
      and int(inventory.get("query", {}).get("exit_code", -1)) == 0
      and int(inventory.get("query", {}).get("stderr_bytes", -1)) == 0,
      f"{arm} inventory did not prove a successful no-live listing",
  )
  return {
      "shards": expected_shards,
      "object_count": len(objects),
      "live_objects": 0,
      "observed_shard_record_pairs": observed,
      "receipt_classifier_seam_records": receipt_seam,
      "record_count_delta": observed - receipt_seam,
      "record_count_relation": "EQUAL" if observed == receipt_seam else "DRIFT",
  }


def review(*, inventory_root: Path, receipt_path: Path) -> dict[str, Any]:
  _require(inventory_root.is_dir(), "inventory root is absent")
  manifest = _manifest(inventory_root / "SHA256SUMS")
  _require(set(manifest) == INVENTORY_MEMBERS,
           "inventory manifest membership drifted")
  _require(
      set(path.name for path in inventory_root.iterdir())
      == INVENTORY_MEMBERS | {"SHA256SUMS"},
      "inventory directory contains missing or extra files",
  )
  for name, digest in manifest.items():
    _require(_sha(inventory_root / name) == digest,
             f"inventory member failed SHA: {name}")
  _require(_sha(receipt_path) == RECEIPT_SHA256,
           "Attempt-13 immutable receipt SHA drifted")
  receipt = _json(receipt_path, "Attempt-13 receipt")
  source = _json(inventory_root / "D32_INVENTORY.json", "D32 inventory")
  _require(
      source.get("schema") == "m15-attempt13-flat-gcs-inventory-v1"
      and source.get("status") == "PASS"
      and source.get("source_commit") == SOURCE_COMMIT
      and source.get("receipt_sha256") == RECEIPT_SHA256
      and source.get("remote_state_mutated") is False
      and source.get("official_classifier_replay") == "NOT_PERFORMED"
      and source.get("numerical_repair_authorized") is False,
      "D32 inventory top-level contract drifted",
  )
  arms = {}
  for arm, contract in ARMS.items():
    receipt_arm = receipt.get(contract["field"])
    inventory_arm = source.get("arms", {}).get(arm)
    _require(isinstance(receipt_arm, dict) and isinstance(inventory_arm, dict),
             f"{arm} inventory or receipt arm is absent")
    arms[arm] = _review_arm(
        arm=arm,
        root=inventory_root,
        inventory=inventory_arm,
        receipt_arm=receipt_arm,
    )
  count_drift = any(value["record_count_relation"] == "DRIFT"
                    for value in arms.values())
  return {
      "schema": "m15-attempt13-inventory-review-v1",
      "status": "PASS",
      "decision": (
          "D32_LIVE_ABSENT_WITH_COUNT_DRIFT"
          if count_drift else "D32_LIVE_ABSENT_COUNTS_MATCH"
      ),
      "attempt": 13,
      "source_commit": SOURCE_COMMIT,
      "source_inventory_manifest_sha256": _sha(inventory_root / "SHA256SUMS"),
      "receipt_sha256": RECEIPT_SHA256,
      "inventory_transport_status": "PASS",
      "live_absence_status": "CONFIRMED",
      "count_contract_status": "DRIFT" if count_drift else "MATCH",
      "arms": arms,
      "historical_official_replay_possible": False,
      "d33_preparation_eligible": True,
      "d33_launch_authorized": False,
      "numerical_repair_authorized": False,
      "claim_ceiling": (
          "The returned object inventory is complete and both registered roots "
          "lack live replay inputs. The shard and classifier counts use "
          "unreconciled metrics, so Attempt 13 is not an official numerical "
          "replay and cannot authorize a repair."
      ),
  }


def _write_return(output: Path, result: dict[str, Any]) -> None:
  _require(not output.exists(), f"refusing to overwrite output: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale partial output exists: {partial}")
  partial.mkdir(parents=True)
  try:
    (partial / "D32_REVIEW.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (partial / "PACKAGING.txt").write_text(
        "M15 Attempt-13 offline inventory review\n"
        f"decision={result['decision']}\n"
        f"count_contract_status={result['count_contract_status']}\n"
        "d33_preparation_eligible=1\n"
        "d33_launch_authorized=0\n"
        "numerical_repair_authorized=0\n",
        encoding="utf-8",
    )
    names = ("D32_REVIEW.json", "PACKAGING.txt")
    (partial / "SHA256SUMS").write_text(
        "".join(f"{_sha(partial / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    partial.replace(output)
  except Exception:
    shutil.rmtree(partial, ignore_errors=True)
    raise


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--inventory", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  task_dir = Path(__file__).resolve().parents[1]
  receipt = task_dir / (
      "evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
  )
  result = review(inventory_root=args.inventory, receipt_path=receipt)
  _write_return(args.output, result)
  print(
      f"M15_ATTEMPT13_REVIEW_PASS decision={result['decision']} "
      f"count_contract_status={result['count_contract_status']} "
      "d33_preparation_eligible=1 d33_launch_authorized=0 "
      "numerical_repair_authorized=0"
  )
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (Attempt13ReviewError, json.JSONDecodeError, OSError) as error:
    print(f"M15_ATTEMPT13_REVIEW_RED {error}")
    raise SystemExit(2) from error
