#!/usr/bin/env python3
"""Produce a verified flat-shard content audit of Attempt 14 (d33).

This tool derives exact object roots and JobSet identities from
RECOVERY_INPUT_RECEIPT.json. It downloads and verifies SHARD_COMPLETE.json
and SHA256SUMS for all 88 off and 74 on sequences, validates manifest hashes,
checks diagnostic_round metadata, computes round histograms, and emits a
formal machine decision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


SCHEMA_VERSION = "m15-apc-attempt14-flat-shard-audit-v1"
EXPECTED_RECEIPT_SCHEMA = "m15-apc-attempt14-recovery-input-v1"
EXPECTED_SOURCE = "003276a3fe2a0ceeaa95a7d940550dab627b8324"
EXPECTED_CAMPAIGN = "v1-apc-m15-attempt14-d33"
EXPECTED_SHARD_COUNTS = {"off": 88, "on": 74}
REQUIRED_TRIPLE_MEMBERS = {"SHARD_ARCHIVE.tar", "SHA256SUMS", "SHARD_COMPLETE.json"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Attempt14FlatShardAuditError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt14FlatShardAuditError(message)


def _sha_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
  return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")


def _sanitize_text(text: str) -> str:
  sanitized = re.sub(r"gs://[a-zA-Z0-9_\-\.\/]+", "<SANITIZED_GCS_URI>", text)
  sanitized = re.sub(r"(token|secret|password|bearer|auth)=[^\s]+", r"\1=<REDACTED>", sanitized, flags=re.IGNORECASE)
  return sanitized.strip()


class StorageClient:
  """Storage client supporting fast parallel downloads and command logging."""

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
      raise Attempt14FlatShardAuditError("gcloud or gsutil is required")

  @staticmethod
  def _receipt(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = result.stdout.encode("utf-8")
    stderr = result.stderr.encode("utf-8")
    raw_stderr = result.stderr
    is_not_found = (
        result.returncode != 0
        and any(nf in raw_stderr for nf in ("NotFoundException", "404", "matched no objects", "does not exist", "NoSuchKey"))
    )
    outcome = "PASS" if result.returncode == 0 else ("NOT_FOUND" if is_not_found else "QUERY_FAILED")
    return {
        "exit_code": int(result.returncode),
        "outcome": outcome,
        "stdout_bytes": len(stdout),
        "stdout_sha256": _sha_bytes(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": _sha_bytes(stderr),
        "sanitized_stderr": _sanitize_text(raw_stderr)[:500] if result.returncode != 0 else "",
    }

  def list_recursive(self, root: str) -> tuple[dict[str, Any], list[str]]:
    if self.tool == "gcloud-storage":
      command = [self.binary, "storage", "ls", "--recursive", root.rstrip("/") + "/**"]
    else:
      command = [self.binary, "-q", "ls", "-r", root.rstrip("/") + "/**"]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    receipt = {"tool": self.tool, **self._receipt(result)}
    rows = result.stdout.splitlines() if result.returncode == 0 else []
    return receipt, rows

  def copy_file(self, uri: str, destination: Path) -> dict[str, Any]:
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
    if not row or row.endswith(":"):
      continue
    _require(row.startswith(prefix), f"listing returned foreign prefix: {row}")
    rel = row[len(prefix):]
    _require(rel and not rel.startswith("/") and ".." not in Path(rel).parts,
             f"unsafe relative path: {rel}")
    normalized.append(rel)
  _require(len(normalized) == len(set(normalized)), "duplicate objects in listing")
  return sorted(normalized)


def _validate_single_shard(
    *,
    sequence: int,
    complete_path: Path,
    sums_path: Path,
    expected_source: str,
) -> dict[str, Any]:
  try:
    complete_data = json.loads(complete_path.read_text(encoding="utf-8"))
  except Exception as err:
    raise Attempt14FlatShardAuditError(f"shard {sequence:06d} completion invalid json") from err

  _require(
      complete_data.get("schema") == "m15-wide-observer-shard-completion-v1",
      f"shard {sequence:06d} invalid schema: {complete_data.get('schema')}"
  )
  _require(
      complete_data.get("status") == "sealed-uploaded-verified",
      f"shard {sequence:06d} invalid status: {complete_data.get('status')}"
  )
  _require(
      int(complete_data.get("sequence", -1)) == sequence,
      f"shard {sequence:06d} sequence mismatch: {complete_data.get('sequence')}"
  )
  _require(
      complete_data.get("expected_source_commit") == expected_source
      and complete_data.get("runtime_source_commit") == expected_source,
      f"shard {sequence:06d} source commit mismatch"
  )

  manifest_sha = complete_data.get("manifest_sha256")
  _require(bool(_SHA256_RE.fullmatch(str(manifest_sha))), f"shard {sequence:06d} malformed manifest_sha256")

  archive_sha = complete_data.get("archive_sha256")
  _require(bool(_SHA256_RE.fullmatch(str(archive_sha))), f"shard {sequence:06d} malformed archive_sha256")

  sums_text = sums_path.read_text(encoding="utf-8")
  calc_manifest_sha = _sha_bytes(sums_text.encode("utf-8"))
  _require(
      manifest_sha == calc_manifest_sha,
      f"shard {sequence:06d} manifest SHA mismatch: {manifest_sha} != {calc_manifest_sha}"
  )

  record_pairs = int(complete_data.get("record_pairs", -1))
  payload_bytes = int(complete_data.get("payload_bytes", -1))
  _require(record_pairs > 0, f"shard {sequence:06d} record_pairs <= 0")
  _require(payload_bytes > 0, f"shard {sequence:06d} payload_bytes <= 0")

  raw_round = complete_data.get("diagnostic_round")
  _require(raw_round is not None, f"shard {sequence:06d} missing diagnostic_round")
  try:
    diagnostic_round = int(raw_round)
  except (ValueError, TypeError):
    raise Attempt14FlatShardAuditError(f"shard {sequence:06d} non-integer diagnostic_round: {raw_round}")
  _require(diagnostic_round >= 0, f"shard {sequence:06d} negative diagnostic_round: {diagnostic_round}")

  return {
      "sequence": sequence,
      "diagnostic_round": diagnostic_round,
      "record_pairs": record_pairs,
      "payload_bytes": payload_bytes,
      "manifest_sha256": manifest_sha,
      "archive_sha256": archive_sha,
      "complete_sha256": _sha(complete_path),
      "sums_sha256": calc_manifest_sha,
  }


def audit_flat_shards(
    *,
    recovery_receipt_path: Path,
    output_dir: Path,
    storage_client: Any = None,
    max_workers: int = 16,
) -> dict[str, Any]:
  _require(recovery_receipt_path.is_file(), f"missing recovery receipt: {recovery_receipt_path}")
  receipt_content = json.loads(recovery_receipt_path.read_text(encoding="utf-8"))

  _require(
      receipt_content.get("schema") == EXPECTED_RECEIPT_SCHEMA,
      f"unexpected receipt schema: {receipt_content.get('schema')}"
  )
  _require(
      receipt_content.get("source_commit") == EXPECTED_SOURCE,
      f"unexpected source commit: {receipt_content.get('source_commit')}"
  )
  _require(
      receipt_content.get("campaign_root") == EXPECTED_CAMPAIGN,
      f"unexpected campaign root: {receipt_content.get('campaign_root')}"
  )

  jobsets = receipt_content.get("jobsets", {})
  _require("off" in jobsets and "on" in jobsets, "missing off or on jobset definitions in receipt")

  client = storage_client or StorageClient()

  scratch_output = output_dir.with_name(output_dir.name + ".tmp")
  if scratch_output.exists():
    shutil.rmtree(scratch_output)
  scratch_output.mkdir(parents=True, exist_ok=True)

  arm_summaries = {}
  query_receipts = {}
  all_rounds_observed = set()

  for arm in ("off", "on"):
    jobset_name = jobsets[arm]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset_name}/attempt-0"
    list_receipt, raw_rows = client.list_recursive(gcs_root)
    query_receipts[arm] = {"list_query": list_receipt}

    _require(list_receipt["outcome"] == "PASS", f"{arm} GCS recursive list failed")

    rel_paths = _relative_rows(gcs_root, raw_rows)
    expected_shards = EXPECTED_SHARD_COUNTS[arm]

    shard_triples: dict[int, set[str]] = {}
    for path_str in rel_paths:
      match = re.fullmatch(r"wide/shards/([0-9]{6})/(SHARD_ARCHIVE\.tar|SHA256SUMS|SHARD_COMPLETE\.json)", path_str)
      if match:
        seq = int(match.group(1))
        member = match.group(2)
        shard_triples.setdefault(seq, set()).add(member)

    expected_sequences = list(range(expected_shards))
    _require(
        sorted(shard_triples.keys()) == expected_sequences,
        f"{arm} sequences not contiguous 0..{expected_shards - 1}: observed {sorted(shard_triples.keys())[:5]}...{sorted(shard_triples.keys())[-5:]}"
    )

    for seq in expected_sequences:
      members = shard_triples.get(seq, set())
      _require(
          members == REQUIRED_TRIPLE_MEMBERS,
          f"{arm} shard {seq:06d} incomplete triple members: observed {members}"
      )

    arm_dir = scratch_output / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Download SHARD_COMPLETE.json and SHA256SUMS for each shard
    def _fetch_shard_files(seq: int) -> tuple[int, Path, Path]:
      seq_dir = arm_dir / f"{seq:06d}"
      seq_dir.mkdir(parents=True, exist_ok=True)
      complete_dest = seq_dir / "SHARD_COMPLETE.json"
      sums_dest = seq_dir / "SHA256SUMS"

      complete_uri = f"{gcs_root}/wide/shards/{seq:06d}/SHARD_COMPLETE.json"
      sums_uri = f"{gcs_root}/wide/shards/{seq:06d}/SHA256SUMS"

      r1 = client.copy_file(complete_uri, complete_dest)
      _require(r1.get("outcome") == "PASS" and complete_dest.is_file(), f"{arm} shard {seq:06d} copy SHARD_COMPLETE failed")

      r2 = client.copy_file(sums_uri, sums_dest)
      _require(r2.get("outcome") == "PASS" and sums_dest.is_file(), f"{arm} shard {seq:06d} copy SHA256SUMS failed")

      return seq, complete_dest, sums_dest

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      fetched = list(pool.map(_fetch_shard_files, expected_sequences))

    shards_receipts = []
    rounds_histogram: dict[str, int] = {}
    total_record_pairs = 0
    total_payload_bytes = 0

    for seq, comp_path, sums_p in sorted(fetched, key=lambda x: x[0]):
      record = _validate_single_shard(
          sequence=seq,
          complete_path=comp_path,
          sums_path=sums_p,
          expected_source=EXPECTED_SOURCE,
      )
      shards_receipts.append(record)
      r_str = str(record["diagnostic_round"])
      rounds_histogram[r_str] = rounds_histogram.get(r_str, 0) + 1
      all_rounds_observed.add(record["diagnostic_round"])
      total_record_pairs += record["record_pairs"]
      total_payload_bytes += record["payload_bytes"]

    arm_summaries[arm] = {
        "jobset": jobset_name,
        "total_shards": len(shards_receipts),
        "total_record_pairs": total_record_pairs,
        "total_payload_bytes": total_payload_bytes,
        "rounds_histogram": rounds_histogram,
        "shards": shards_receipts,
    }

    _write_json(scratch_output / f"{arm}.shards.json", {
        "arm": arm,
        "jobset": jobset_name,
        "total_shards": len(shards_receipts),
        "total_record_pairs": total_record_pairs,
        "total_payload_bytes": total_payload_bytes,
        "rounds_histogram": rounds_histogram,
        "shards": shards_receipts,
    })

  # Machine Decision
  if all_rounds_observed == {0, 1, 2}:
    decision = "D33_FLAT_SHARDS_THREE_ROUNDS_VERIFIED"
  elif all_rounds_observed == {0}:
    decision = "D33_FLAT_SHARDS_ROUND0_ONLY"
  elif not all_rounds_observed:
    decision = "D33_FLAT_SHARDS_METADATA_INSUFFICIENT"
  else:
    decision = "D33_FLAT_SHARDS_METADATA_INSUFFICIENT"

  summary = {
      "schema": SCHEMA_VERSION,
      "source_commit": EXPECTED_SOURCE,
      "campaign_root": EXPECTED_CAMPAIGN,
      "decision": decision,
      "all_rounds_observed": sorted(list(all_rounds_observed)),
      "arms": {
          arm: {
              "jobset": arm_summaries[arm]["jobset"],
              "total_shards": arm_summaries[arm]["total_shards"],
              "total_record_pairs": arm_summaries[arm]["total_record_pairs"],
              "total_payload_bytes": arm_summaries[arm]["total_payload_bytes"],
              "rounds_histogram": arm_summaries[arm]["rounds_histogram"],
          }
          for arm in ("off", "on")
      },
  }

  _write_json(scratch_output / "FLAT_SHARD_AUDIT_SUMMARY.json", summary)
  _write_json(scratch_output / "QUERY_RECEIPTS.json", query_receipts)
  _write_json(scratch_output / "RECOVERY_INPUT_RECEIPT.json", receipt_content)

  # Generate SHA256SUMS for top-level output artifacts
  files_to_sum = [
      "FLAT_SHARD_AUDIT_SUMMARY.json",
      "QUERY_RECEIPTS.json",
      "RECOVERY_INPUT_RECEIPT.json",
      "off.shards.json",
      "on.shards.json",
  ]
  sums_lines = []
  for name in sorted(files_to_sum):
    p = scratch_output / name
    sums_lines.append(f"{_sha(p)}  {name}")
  (scratch_output / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

  if output_dir.exists():
    shutil.rmtree(output_dir)
  scratch_output.rename(output_dir)
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recovery-receipt", required=True, type=Path, help="Path to RECOVERY_INPUT_RECEIPT.json")
  parser.add_argument("--output-dir", required=True, type=Path, help="Target directory for output evidence package")
  parser.add_argument("--max-workers", type=int, default=16, help="Max concurrency for GCS downloads")
  args = parser.parse_args()

  summary = audit_flat_shards(
      recovery_receipt_path=args.recovery_receipt,
      output_dir=args.output_dir,
      max_workers=args.max_workers,
  )
  print(
      f"AUDIT_M15_ATTEMPT14_D33_FLAT_SHARDS decision={summary['decision']} "
      f"rounds={summary['all_rounds_observed']} "
      f"off_shards={summary['arms']['off']['total_shards']} "
      f"on_shards={summary['arms']['on']['total_shards']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
