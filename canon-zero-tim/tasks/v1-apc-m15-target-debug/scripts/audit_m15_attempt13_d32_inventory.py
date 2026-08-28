#!/usr/bin/env python3
"""Produce a small, self-hashed inventory of Attempt 13's registered roots.

The audit is read-only. It recursively lists both arms, validates the complete
flat-shard object geometry, downloads only the small SHARD_COMPLETE receipts,
and distinguishes a successful listing with no ``live/`` objects from a query
failure. Remote object names are normalized to root-relative paths before they
enter the return package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


SOURCE_COMMIT = "7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
RECEIPT_SHA256 = "d1941c2de85050a5652bc5c6e809987f6bf72b996aa817371b08b43870835f95"
ARM_CONTRACTS = {
    "off": {
        "field": "control_arm_off",
        "jobset": "canon-v1-apc-m15-off-d32-7d30f382",
        "shards": 77,
        "record_pairs": 2445,
    },
    "on": {
        "field": "treatment_arm_on",
        "jobset": "canon-v1-apc-m15-on-d32-7d30f382",
        "shards": 70,
        "record_pairs": 2188,
    },
}
_SHARD_MEMBERS = {"SHARD_ARCHIVE.tar", "SHA256SUMS", "SHARD_COMPLETE.json"}
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class Attempt13InventoryError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt13InventoryError(message)


def _sha_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
  return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")


class StorageClient:
  """Small command adapter whose receipts never expose a remote URI."""

  def __init__(self) -> None:
    gcloud = shutil.which("gcloud")
    gsutil = shutil.which("gsutil")
    if gcloud:
      self.tool = "gcloud-storage"
      self.binary = gcloud
    elif gsutil:
      self.tool = "gsutil"
      self.binary = gsutil
    else:
      raise Attempt13InventoryError("gcloud or gsutil is required")

  @staticmethod
  def _receipt(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = result.stdout.encode("utf-8")
    stderr = result.stderr.encode("utf-8")
    return {
        "exit_code": int(result.returncode),
        "stdout_bytes": len(stdout),
        "stdout_sha256": _sha_bytes(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": _sha_bytes(stderr),
    }

  def list_recursive(self, root: str) -> tuple[dict[str, Any], list[str]]:
    if self.tool == "gcloud-storage":
      command = [self.binary, "storage", "ls", "--recursive", root + "/**"]
    else:
      command = [self.binary, "-q", "ls", "-r", root + "/**"]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    receipt = {"tool": self.tool, **self._receipt(result)}
    return receipt, result.stdout.splitlines()

  def copy(self, uri: str, destination: Path) -> dict[str, Any]:
    if self.tool == "gcloud-storage":
      command = [self.binary, "storage", "cp", uri, str(destination)]
    else:
      command = [self.binary, "-q", "cp", uri, str(destination)]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    return {"tool": self.tool, **self._receipt(result)}


def _relative_rows(root: str, rows: list[str]) -> list[str]:
  prefix = root.rstrip("/") + "/"
  normalized = []
  for raw in rows:
    row = raw.strip()
    if not row:
      continue
    # gsutil recursive listings may include directory headings.
    if row.endswith(":"):
      continue
    _require(row.startswith(prefix), "recursive listing returned a foreign root")
    relative = row[len(prefix):]
    _require(relative and not relative.startswith("/") and ".." not in Path(relative).parts,
             "recursive listing returned an unsafe relative path")
    normalized.append(relative)
  _require(len(normalized) == len(set(normalized)),
           "recursive listing contains duplicate objects")
  return sorted(normalized)


def _validate_completion(
    path: Path, *, sequence: int, expected_source: str
) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as error:
    raise Attempt13InventoryError(
        f"shard completion {sequence:06d} is invalid"
    ) from error
  _require(
      value.get("schema") == "m15-wide-observer-shard-completion-v1"
      and value.get("status") == "sealed-uploaded-verified"
      and int(value.get("sequence", -1)) == sequence
      and int(value.get("diagnostic_round", -1)) == 0
      and value.get("expected_source_commit") == expected_source
      and value.get("runtime_source_commit") == expected_source,
      f"shard completion {sequence:06d} contract drifted",
  )
  record_pairs = int(value.get("record_pairs", -1))
  payload_bytes = int(value.get("payload_bytes", -1))
  manifest_sha = str(value.get("manifest_sha256", ""))
  archive_sha = str(value.get("archive_sha256", ""))
  _require(
      record_pairs > 0 and payload_bytes > 0
      and _SHA_RE.fullmatch(manifest_sha) is not None
      and _SHA_RE.fullmatch(archive_sha) is not None,
      f"shard completion {sequence:06d} counts or hashes drifted",
  )
  return {
      "sequence": sequence,
      "record_pairs": record_pairs,
      "payload_bytes": payload_bytes,
      "manifest_sha256": manifest_sha,
      "archive_sha256": archive_sha,
      "completion_sha256": _sha(path),
  }


def _audit_arm(
    *,
    arm: str,
    root: str,
    contract: dict[str, Any],
    client: Any,
    scratch: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
  query, raw_rows = client.list_recursive(root)
  base = {
      "arm": arm,
      "jobset": contract["jobset"],
      "root_identity_sha256": _sha_bytes(root.encode("utf-8")),
      "query": query,
  }
  if int(query.get("exit_code", -1)) != 0:
    return {
        **base,
        "status": "RED",
        "failure": "RECURSIVE_LIST_QUERY_FAILED",
        "live_absence_proven": False,
    }, {}
  try:
    rows = _relative_rows(root, raw_rows)
    _require("PREFLIGHT.json" in rows, f"{arm} root lacks PREFLIGHT.json")
    expected_shards = int(contract["shards"])
    expected_sequences = list(range(expected_shards))
    member_map: dict[int, set[str]] = {}
    for row in rows:
      match = re.fullmatch(
          r"wide/shards/([0-9]{6})/(SHARD_ARCHIVE\.tar|SHA256SUMS|SHARD_COMPLETE\.json)",
          row,
      )
      if match:
        member_map.setdefault(int(match.group(1)), set()).add(match.group(2))
    _require(sorted(member_map) == expected_sequences,
             f"{arm} flat-shard sequences are incomplete")
    _require(all(members == _SHARD_MEMBERS for members in member_map.values()),
             f"{arm} flat-shard object triples are incomplete")

    arm_scratch = scratch / arm
    arm_scratch.mkdir()
    completions = []
    for sequence in expected_sequences:
      destination = arm_scratch / f"{sequence:06d}.json"
      transfer = client.copy(
          root + f"/wide/shards/{sequence:06d}/SHARD_COMPLETE.json",
          destination,
      )
      _require(int(transfer.get("exit_code", -1)) == 0 and destination.is_file(),
               f"{arm} shard completion {sequence:06d} download failed")
      completions.append(_validate_completion(
          destination, sequence=sequence, expected_source=SOURCE_COMMIT
      ))
    record_pairs = sum(item["record_pairs"] for item in completions)
    payload_bytes = sum(item["payload_bytes"] for item in completions)
    _require(record_pairs == int(contract["record_pairs"]),
             f"{arm} flat-shard record-pair total drifted")
    live_objects = [row for row in rows if row.startswith("live/")]
    round_objects = [row for row in rows if row.startswith("wide/rounds/")]
    objects_text = ("\n".join(rows) + "\n").encode("utf-8")
    completions_text = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in completions
    ).encode("utf-8")
    return {
        **base,
        "status": "PASS",
        "object_count": len(rows),
        "object_inventory_sha256": _sha_bytes(objects_text),
        "flat_shards": expected_shards,
        "flat_shard_required_objects": expected_shards * 3,
        "record_pairs": record_pairs,
        "payload_bytes": payload_bytes,
        "completion_receipts_sha256": _sha_bytes(completions_text),
        "live_objects": len(live_objects),
        "wide_round_objects": len(round_objects),
        "live_absence_proven": not live_objects,
    }, {
        f"{arm}.objects.txt": objects_text,
        f"{arm}.shard-completions.jsonl": completions_text,
    }
  except Attempt13InventoryError as error:
    return {
        **base,
        "status": "RED",
        "failure": str(error),
        "live_absence_proven": False,
    }, {}


def audit(
    *,
    receipt_path: Path,
    output: Path,
    scratch: Path,
    client: Any,
) -> dict[str, Any]:
  _require(not output.exists(), f"refusing to overwrite output: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale partial output exists: {partial}")
  _require(_sha(receipt_path) == RECEIPT_SHA256,
           "Attempt-13 receipt SHA drifted")
  receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  _require(receipt.get("attempt") == 13
           and receipt.get("source_commit") == SOURCE_COMMIT,
           "Attempt-13 receipt identity drifted")
  partial.mkdir(parents=True)
  scratch.mkdir(parents=True, exist_ok=True)
  arms = {}
  files: dict[str, bytes] = {}
  for arm, contract in ARM_CONTRACTS.items():
    value = receipt.get(contract["field"])
    _require(isinstance(value, dict)
             and value.get("jobset_name") == contract["jobset"],
             f"Attempt-13 {arm} JobSet identity drifted")
    root = str(value.get("gcs_source_uri", ""))
    _require(root.startswith("gs://") and root.endswith("/attempt-0"),
             f"Attempt-13 {arm} registered root drifted")
    arms[arm], arm_files = _audit_arm(
        arm=arm, root=root, contract=contract, client=client, scratch=scratch
    )
    files.update(arm_files)

  all_pass = all(value["status"] == "PASS" for value in arms.values())
  both_live_absent = all_pass and all(
      value["live_absence_proven"] for value in arms.values()
  )
  any_live = all_pass and any(value["live_objects"] for value in arms.values())
  if not all_pass:
    decision = "D32_INVENTORY_AUDIT_RED"
  elif both_live_absent:
    decision = "D32_LIVE_ABSENT_CONFIRMED"
  elif any_live:
    decision = "D32_LIVE_PRESENT_REPLAY_SHOULD_CONTINUE"
  else:
    decision = "D32_INVENTORY_INCONSISTENT"
  result = {
      "schema": "m15-attempt13-flat-gcs-inventory-v1",
      "status": "PASS" if all_pass else "RED",
      "attempt": 13,
      "source_commit": SOURCE_COMMIT,
      "receipt_sha256": RECEIPT_SHA256,
      "decision": decision,
      "arms": arms,
      "remote_state_mutated": False,
      "official_classifier_replay": "NOT_PERFORMED",
      "numerical_repair_authorized": False,
      "claim_ceiling": (
          "This inventory proves registered object presence or absence and "
          "validates flat-shard completion receipts. It does not verify shard "
          "archive payload bytes or replay the numerical classifier."
      ),
  }
  for name, payload in files.items():
    (partial / name).write_bytes(payload)
  _write_json(partial / "D32_INVENTORY.json", result)
  (partial / "PACKAGING.txt").write_text(
      "M15 Attempt-13 registered-root inventory\n"
      f"decision={decision}\n"
      f"status={result['status']}\n"
      "remote_state_mutated=0\n"
      "official_classifier_replay=NOT_PERFORMED\n"
      "numerical_repair_authorized=0\n",
      encoding="utf-8",
  )
  names = sorted(path.name for path in partial.iterdir())
  (partial / "SHA256SUMS").write_text(
      "".join(f"{_sha(partial / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  partial.replace(output)
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--scratch-parent", type=Path, default=Path("/tmp"))
  args = parser.parse_args()
  _require(args.scratch_parent.is_dir(), "scratch parent is absent")
  task_dir = Path(__file__).resolve().parents[1]
  receipt = task_dir / (
      "evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
  )
  scratch = args.scratch_parent / (args.output.name + ".inventory-scratch")
  partial = args.output.with_name(args.output.name + ".partial")
  _require(not scratch.exists(), f"scratch path already exists: {scratch}")
  try:
    result = audit(
        receipt_path=receipt,
        output=args.output,
        scratch=scratch,
        client=StorageClient(),
    )
  finally:
    shutil.rmtree(scratch, ignore_errors=True)
    if not args.output.exists():
      shutil.rmtree(partial, ignore_errors=True)
  marker = "PASS" if result["status"] == "PASS" else "RED"
  print(
      f"M15_ATTEMPT13_INVENTORY_{marker} decision={result['decision']} "
      "remote_state_mutated=0 numerical_repair_authorized=0"
  )
  return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (Attempt13InventoryError, json.JSONDecodeError, OSError) as error:
    print(f"M15_ATTEMPT13_INVENTORY_RED {error}")
    raise SystemExit(2) from error
