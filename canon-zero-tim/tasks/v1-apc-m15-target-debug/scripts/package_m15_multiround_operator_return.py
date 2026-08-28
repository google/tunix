#!/usr/bin/env python3
"""Seal the numerical and operator receipts for one M15 multiround pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml


class OperatorReturnError(RuntimeError):
  pass


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GCS_ROOT = re.compile(
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    r"[a-z0-9-]+/attempt-0"
)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise OperatorReturnError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON is not an object: {path}")
  return value


def _manifest(path: Path) -> dict[str, str]:
  rows: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(
        separator == "  "
        and _SHA256.fullmatch(digest) is not None
        and Path(name).name == name
        and name != "SHA256SUMS"
        and name not in rows,
        f"invalid manifest row: {line!r}",
    )
    rows[name] = digest
  return rows


def _verify_core(root: Path) -> dict[str, Any]:
  _require(root.is_dir(), f"core return is absent: {root}")
  manifest = _manifest(root / "SHA256SUMS")
  actual = {
      path.name for path in root.iterdir()
      if path.is_file() and path.name != "SHA256SUMS"
  }
  _require(set(manifest) == actual, "core return manifest membership drifted")
  for name, digest in manifest.items():
    _require(_sha256(root / name) == digest, f"core return SHA failed: {name}")
  summary = _json(root / "MULTIROUND_SUMMARY.json")
  _require(
      summary.get("schema") == "m15-apc-multiround-small-return-v1",
      "core return schema drifted",
  )
  return summary


def _container_env(document: dict[str, Any]) -> dict[str, str]:
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]["containers"][0]
  return {
      str(row["name"]): str(row["value"])
      for row in container["env"] if "value" in row
  }


def _render_contract(root: Path) -> dict[str, dict[str, str]]:
  paths = sorted(root.glob("jobset-v1-apc-m15-*-*.yaml"))
  _require(len(paths) == 2, "render directory must contain exactly two M15 YAMLs")
  result: dict[str, dict[str, str]] = {}
  source = ""
  for path in paths:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = _container_env(document)
    arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
    current_source = env.get("CANON_EXPECT_COMMIT", "")
    jobset = str(document.get("metadata", {}).get("name", ""))
    gcs_root = env.get("CANON_P38_GCS_PREFIX", "")
    _require(arm in ("off", "on") and arm not in result, "invalid rendered arm")
    _require(_SHA40.fullmatch(current_source) is not None, "invalid rendered source")
    _require(_GCS_ROOT.fullmatch(gcs_root) is not None, "invalid rendered GCS root")
    _require(env.get("CANON_P38_DIAGNOSTIC_ROUNDS") == "3", "round count drifted")
    _require(env.get("CANON_P38_SEAM_OBSERVER") == "full", "observer drifted")
    _require(env.get("CANON_P38_SEAM_LAYER") == "0", "seam layer drifted")
    _require(env.get("CANON_P33_RUN_STAGE") == "backward-no-commit", "stage drifted")
    _require(env.get("CANON_P33_NO_COMMIT") == "1", "no-commit gate drifted")
    if source:
      _require(current_source == source, "paired source commits differ")
    source = current_source
    result[arm] = {
        "jobset": jobset,
        "source_commit": current_source,
    }
  _require(set(result) == {"off", "on"}, "rendered pair is incomplete")
  return result


def _recovery_input(
    root: Path, contract: dict[str, dict[str, str]]
) -> dict[str, Any] | None:
  path = root / "RECOVERY_INPUT_RECEIPT.json"
  if not path.exists():
    return None
  receipt = _json(path)
  _require(
      receipt.get("schema") == "m15-apc-attempt14-recovery-input-v1"
      and receipt.get("status") == "LOCATOR_ONLY",
      "recovery input schema/status drifted",
  )
  _require(
      receipt.get("source_commit") == contract["off"]["source_commit"],
      "recovery input source drifted",
  )
  _require(
      receipt.get("jobsets")
      == {arm: contract[arm]["jobset"] for arm in ("off", "on")},
      "recovery input JobSets drifted",
  )
  for field in ("submitted_manifest_sha256", "submitted_receipt_sha256"):
    _require(
        _SHA256.fullmatch(str(receipt.get(field, ""))) is not None,
        f"recovery input {field} drifted",
    )
  return receipt


def _load_receipts(
    root: Path,
    *,
    schema: str,
    contract: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
  result: dict[str, dict[str, Any]] = {}
  for arm in ("off", "on"):
    receipt = _json(root / f"{arm}.json")
    _require(receipt.get("schema") == schema, f"{arm} receipt schema drifted")
    _require(receipt.get("arm") == arm, f"{arm} receipt arm drifted")
    _require(
        receipt.get("jobset") == contract[arm]["jobset"],
        f"{arm} receipt JobSet drifted",
    )
    _require(
        receipt.get("source_commit") == contract[arm]["source_commit"],
        f"{arm} receipt source drifted",
    )
    result[arm] = receipt
  return result


def package(
    *,
    render_dir: Path,
    core_return: Path,
    jobset_receipts: Path,
    raw_log_receipts: Path,
    output: Path,
) -> dict[str, Any]:
  _require(not output.exists(), f"refusing to overwrite return: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale partial return exists: {partial}")
  contract = _render_contract(render_dir)
  recovery_input = _recovery_input(render_dir, contract)
  core = _verify_core(core_return)
  source = contract["off"]["source_commit"]
  _require(core.get("source_commit") == source, "core/render source mismatch")
  jobsets = _load_receipts(
      jobset_receipts,
      schema="m15-apc-jobset-status-v1",
      contract=contract,
  )
  raw_logs = _load_receipts(
      raw_log_receipts,
      schema="m15-apc-raw-log-receipt-v1",
      contract=contract,
  )
  for arm, receipt in raw_logs.items():
    if receipt.get("status") == "PRESENT":
      _require(
          receipt.get("object_identity")
          == f"{contract[arm]['jobset']}/attempt-0/run.log"
          and _SHA256.fullmatch(str(receipt.get("sha256", ""))) is not None
          and isinstance(receipt.get("bytes"), int)
          and int(receipt["bytes"]) > 0,
          f"{arm} raw-log receipt drifted",
      )
  jobsets_terminal = all(
      receipt.get("query_status") == "PASS"
      and receipt.get("terminal_condition") in ("Completed", "Failed")
      for receipt in jobsets.values()
  )
  raw_logs_present = all(
      receipt.get("status") == "PRESENT" for receipt in raw_logs.values()
  )
  core_status = str(core.get("status", "UNKNOWN"))
  status = (
      core_status
      if jobsets_terminal and raw_logs_present
      else f"{core_status}_OPERATOR_RECEIPTS_INCOMPLETE"
  )
  summary = {
      "schema": "m15-apc-multiround-operator-return-v1",
      "status": status,
      "core_status": core_status,
      "source_commit": source,
      "jobsets_terminal": jobsets_terminal,
      "raw_logs_present": raw_logs_present,
      "jobsets": jobsets,
      "raw_logs": raw_logs,
      "recovery_input_bound": recovery_input is not None,
      "claim_ceiling": (
          "Numerical status comes only from MULTIROUND_SUMMARY.json. JobSet and "
          "raw-log receipts establish operator completeness, not numerical equality."
      ),
  }
  partial.mkdir(parents=True)
  try:
    for path in core_return.iterdir():
      if path.is_file() and path.name != "SHA256SUMS":
        shutil.copyfile(path, partial / path.name)
    (partial / "JOBSET_STATUS.json").write_text(
        json.dumps(jobsets, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (partial / "RAW_LOG_RECEIPTS.json").write_text(
        json.dumps(raw_logs, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (partial / "OPERATOR_RETURN_SUMMARY.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if recovery_input is not None:
      (partial / "RECOVERY_INPUT_RECEIPT.json").write_text(
          json.dumps(recovery_input, sort_keys=True, indent=2) + "\n",
          encoding="utf-8",
      )
    (partial / "OPERATOR_PACKAGING.txt").write_text(
        "M15 multiround operator return\n"
        f"status={status}\n"
        f"core_status={core_status}\n"
        f"jobsets_terminal={int(jobsets_terminal)}\n"
        f"raw_logs_present={int(raw_logs_present)}\n"
        f"recovery_input_bound={int(recovery_input is not None)}\n"
        "raw_log_payload_returned=0\n"
        "token_bearing_bundle_returned=0\n"
        "remote_state_mutated=0\n",
        encoding="utf-8",
    )
    names = sorted(path.name for path in partial.iterdir() if path.is_file())
    (partial / "SHA256SUMS").write_text(
        "".join(f"{_sha256(partial / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    partial.replace(output)
  except Exception:
    shutil.rmtree(partial, ignore_errors=True)
    raise
  return summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--render-dir", required=True, type=Path)
  parser.add_argument("--core-return", required=True, type=Path)
  parser.add_argument("--jobset-receipts", required=True, type=Path)
  parser.add_argument("--raw-log-receipts", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = package(
        render_dir=args.render_dir,
        core_return=args.core_return,
        jobset_receipts=args.jobset_receipts,
        raw_log_receipts=args.raw_log_receipts,
        output=args.output,
    )
  except (OSError, ValueError, json.JSONDecodeError, OperatorReturnError) as exc:
    raise SystemExit(f"M15_OPERATOR_RETURN_RED {exc}") from exc
  print(
      "M15_OPERATOR_RETURN_COMPLETE "
      f"status={result['status']} output={args.output}"
  )


if __name__ == "__main__":
  main()
