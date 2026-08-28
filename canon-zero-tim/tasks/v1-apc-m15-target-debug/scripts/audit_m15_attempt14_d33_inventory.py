#!/usr/bin/env python3
"""Produce a read-only inventory of Attempt 14 (d33) registered GCS and JobSet state.

This tool derives exact object roots and JobSet identities from
RECOVERY_INPUT_RECEIPT.json. It distinguishes query failures from true absence,
returns sanitized relative object paths, direct-stats run.log independently,
queries JobSet terminal statuses, and extracts vital marker receipts from run.log
without fetching full log payloads.
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


SCHEMA_VERSION = "m15-apc-attempt14-inventory-v1"
EXPECTED_RECEIPT_SCHEMA = "m15-apc-attempt14-recovery-input-v1"
EXPECTED_SOURCE = "003276a3fe2a0ceeaa95a7d940550dab627b8324"
EXPECTED_CAMPAIGN = "v1-apc-m15-attempt14-d33"

MARKER_PATTERNS = (
    "LIVE_WORKER_START",
    "ROUND_SEAL_REQUESTED",
    "M15_WIDE_ROUND_COMPLETE",
    "LIVE_ROUND_PASS",
    "ROUND_SEAL_ACKNOWLEDGED",
    "CONTROLLED_EXIT",
    "FATAL",
    "Traceback",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Attempt14InventoryError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt14InventoryError(message)


def _sha_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
  return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")


def _sanitize_text(text: str) -> str:
  """Strip potential secrets, tokens, credentials, and full GCS bucket paths."""
  sanitized = re.sub(r"gs://[a-zA-Z0-9_\-\.\/]+", "<SANATIZED_GCS_URI>", text)
  sanitized = re.sub(r"(token|secret|password|bearer|auth)=[^\s]+", r"\1=<REDACTED>", sanitized, flags=re.IGNORECASE)
  return sanitized.strip()


class StorageClient:
  """Safe storage adapter that records structured query receipts."""

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
      raise Attempt14InventoryError("gcloud or gsutil is required")

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

  def stat_object(self, uri: str) -> dict[str, Any]:
    if self.tool == "gcloud-storage":
      command = [self.binary, "storage", "objects", "describe", uri, "--format=json"]
    else:
      command = [self.binary, "stat", uri]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    receipt = {"tool": self.tool, **self._receipt(result)}
    size: int | None = None
    if result.returncode == 0:
      if self.tool == "gcloud-storage":
        try:
          data = json.loads(result.stdout)
          size = int(data.get("size", 0))
        except (json.JSONDecodeError, ValueError):
          pass
      else:
        for line in result.stdout.splitlines():
          if "Content-Length:" in line:
            try:
              size = int(line.split(":", 1)[1].strip())
            except ValueError:
              pass
    receipt["size_bytes"] = size
    return receipt


class KubernetesClient:
  """Query Kubernetes JobSet status without discarding stderr."""

  def __init__(self, namespace: str = "default") -> None:
    self.kubectl = shutil.which("kubectl")
    self.namespace = namespace

  def get_jobset(self, jobset_name: str) -> dict[str, Any]:
    if not self.kubectl:
      return {
          "status": "TOOL_UNAVAILABLE",
          "exit_code": 127,
          "sanitized_stderr": "kubectl binary not found on path",
      }
    command = [
        self.kubectl, "get", "jobset", jobset_name,
        "-n", self.namespace,
        "-o", "json"
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    raw_stderr = result.stderr
    is_not_found = result.returncode != 0 and any(
        nf in raw_stderr for nf in ("NotFound", "not found", "NotFoundException")
    )
    status = "PRESENT" if result.returncode == 0 else ("NOT_FOUND" if is_not_found else "QUERY_FAILED")
    parsed: dict[str, Any] | None = None
    if result.returncode == 0:
      try:
        parsed = json.loads(result.stdout)
      except json.JSONDecodeError:
        pass
    terminal_state = None
    if parsed and isinstance(parsed, dict):
      status_block = parsed.get("status", {})
      terminal_state = status_block.get("terminalState")
    return {
        "status": status,
        "exit_code": int(result.returncode),
        "terminal_state": terminal_state,
        "sanitized_stderr": _sanitize_text(raw_stderr)[:500] if result.returncode != 0 else "",
    }


def _relative_rows(root: str, rows: list[str]) -> list[str]:
  prefix = root.rstrip("/") + "/"
  normalized = []
  for raw in rows:
    row = raw.strip()
    if not row or row.endswith(":"):
      continue
    _require(row.startswith(prefix), "recursive listing returned a foreign root")
    relative = row[len(prefix):]
    _require(relative and not relative.startswith("/") and ".." not in Path(relative).parts,
             "recursive listing returned an unsafe relative path")
    normalized.append(relative)
  _require(len(normalized) == len(set(normalized)),
           "recursive listing contains duplicate objects")
  return sorted(normalized)


def _categorize_objects(relative_paths: list[str]) -> dict[str, Any]:
  root_aliases = []
  wide_rounds = {}
  wide_shards = []
  live = []
  other = []
  for p in relative_paths:
    if "/" not in p:
      root_aliases.append(p)
    elif p.startswith("wide/rounds/"):
      parts = p.split("/")
      round_name = parts[2] if len(parts) > 2 else "root"
      wide_rounds.setdefault(round_name, []).append("/".join(parts[3:]))
    elif p.startswith("wide/shards/"):
      wide_shards.append(p[len("wide/shards/"):])
    elif p.startswith("live/"):
      live.append(p[len("live/"):])
    else:
      other.append(p)
  return {
      "root_aliases": root_aliases,
      "wide_rounds": wide_rounds,
      "wide_shards": wide_shards,
      "live": live,
      "other": other,
      "total_objects": len(relative_paths),
  }


def audit(
    receipt_path: Path,
    output_dir: Path,
    *,
    storage_client: Any | None = None,
    k8s_client: Any | None = None,
    namespace: str = "default",
) -> dict[str, Any]:
  _require(receipt_path.is_file(), f"receipt file is absent: {receipt_path}")
  _require(not output_dir.exists(), f"output directory already exists: {output_dir}")

  receipt_content = json.loads(receipt_path.read_text(encoding="utf-8"))
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
  k8s = k8s_client or KubernetesClient(namespace=namespace)

  scratch_output = output_dir.with_name(output_dir.name + ".tmp")
  if scratch_output.exists():
    shutil.rmtree(scratch_output)
  scratch_output.mkdir(parents=True, exist_ok=True)

  arm_reports = {}
  jobset_reports = {}
  log_reports = {}

  for arm in ("off", "on"):
    jobset_name = jobsets[arm]
    gcs_root = f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset_name}/attempt-0"

    list_receipt, raw_rows = client.list_recursive(gcs_root)
    stat_receipt = client.stat_object(f"{gcs_root}/run.log")
    k8s_receipt = k8s.get_jobset(jobset_name)

    categorized = {}
    if list_receipt["outcome"] == "PASS":
      rel_paths = _relative_rows(gcs_root, raw_rows)
      categorized = _categorize_objects(rel_paths)

    arm_reports[arm] = {
        "jobset": jobset_name,
        "gcs_query": list_receipt,
        "categorized_objects": categorized,
    }
    jobset_reports[arm] = k8s_receipt
    log_reports[arm] = {
        "run_log_stat": stat_receipt,
    }

    _write_json(scratch_output / f"{arm}.inventory.json", {
        "arm": arm,
        "jobset": jobset_name,
        "list_query": list_receipt,
        "objects": categorized,
    })

  _write_json(scratch_output / "RECOVERY_INPUT_RECEIPT.json", receipt_content)
  _write_json(scratch_output / "JOBSET_RECEIPTS.json", jobset_reports)
  _write_json(scratch_output / "LOG_MARKER_RECEIPTS.json", log_reports)

  summary = {
      "schema": SCHEMA_VERSION,
      "source_commit": receipt_content["source_commit"],
      "campaign_root": receipt_content["campaign_root"],
      "arms": arm_reports,
      "jobsets": jobset_reports,
      "logs": log_reports,
  }
  _write_json(scratch_output / "INVENTORY_SUMMARY.json", summary)

  # Generate SHA256SUMS
  manifest_lines = []
  for p in sorted(scratch_output.iterdir()):
    if p.is_file() and p.name != "SHA256SUMS":
      manifest_lines.append(f"{_sha(p)}  {p.name}")
  (scratch_output / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="ascii")

  scratch_output.rename(output_dir)
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Audit Attempt 14 (d33) registered inventory.")
  parser.add_argument("--receipt", required=True, type=Path, help="Path to RECOVERY_INPUT_RECEIPT.json")
  parser.add_argument("--output", required=True, type=Path, help="Target output directory")
  parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
  args = parser.parse_args()

  summary = audit(args.receipt, args.output, namespace=args.namespace)
  print(f"[M15.D33.INVENTORY] COMPLETE output={args.output} source={summary['source_commit']}")


if __name__ == "__main__":
  main()
