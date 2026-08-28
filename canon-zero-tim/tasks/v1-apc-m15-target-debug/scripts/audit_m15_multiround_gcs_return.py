#!/usr/bin/env python3
"""Audit small per-round M15 evidence without downloading token-bearing tars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


class MultiRoundAuditError(RuntimeError):
  pass


_ROUND_FILES = {
    "ROUND_INPUT_RECEIPT.json",
    "p38_seam.classification.json",
    "WIDE_SHA256SUMS",
    "WIDE_ROUND_COMPLETE.json",
}
_MANIFEST_MEMBERS = {
    "ROUND_INPUT_RECEIPT.json",
    "p38_seam.classification.json",
    "m15_wide_seam_bundle.tar",
}
_CLASSIFICATIONS = {
    "off": {"M15_OBSERVER_CONTROL_EXACT"},
    "on": {
        "M15_OBSERVER_TREATMENT_EXACT",
        "M15_LAYER_FIRST_RED_LOCALIZED",
        "M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED",
        "M15_INTERNAL_FIRST_RED_LOCALIZED",
    },
}
_STAGE_SPECS = (
    (10, "assemble"),
    (20, "classify"),
    (30, "package"),
    (35, "local-export"),
    (40, "manifest"),
    (50, "upload"),
    (60, "remote-verify"),
    (70, "completion"),
)
_STAGE_STATUSES = ("STARTED", "PASS", "FAIL")


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise MultiRoundAuditError(message)


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
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and name not in rows,
        f"invalid wide-round manifest row: {line!r}",
    )
    rows[name] = digest
  _require(set(rows) == _MANIFEST_MEMBERS,
           f"wide-round manifest membership drifted: {sorted(rows)}")
  return rows


def _inventory(path: Path) -> dict[str, bool]:
  rows: dict[str, bool] = {}
  if not path.is_file():
    return rows
  for line in path.read_text(encoding="utf-8").splitlines():
    name, separator, status = line.rpartition(" ")
    _require(separator == " " and status in ("present", "absent"),
             f"invalid remote inventory row: {line!r}")
    rows[name] = status == "present"
  return rows


def _audit_stages(
    directory: Path,
    *,
    arm: str,
    round_index: int,
    source_commit: str,
    output: Path,
) -> dict[str, Any]:
  stage_dir = directory / "stages"
  receipts: list[dict[str, Any]] = []
  files: dict[tuple[int, str, str], Path] = {}
  expected_names = {
      f"STAGE_{ordinal}_{stage}_{status}.json"
      for ordinal, stage in _STAGE_SPECS
      for status in _STAGE_STATUSES
  }
  if stage_dir.is_dir():
    unexpected = {
        path.name for path in stage_dir.iterdir()
        if path.is_file() and path.name not in expected_names
    }
    _require(not unexpected,
             f"{arm} round {round_index} has unexpected stage receipts: "
             f"{sorted(unexpected)}")
    for ordinal, stage in _STAGE_SPECS:
      for status in _STAGE_STATUSES:
        path = stage_dir / f"STAGE_{ordinal}_{stage}_{status}.json"
        if path.is_file():
          files[(ordinal, stage, status)] = path

  inventory = _inventory(directory / "remote-inventory.txt")
  for ordinal, stage in _STAGE_SPECS:
    for status in _STAGE_STATUSES:
      name = f"STAGE_{ordinal}_{stage}_{status}.json"
      inventory_key = f"stages/{name}"
      if inventory_key in inventory:
        _require(
            inventory[inventory_key]
            == ((ordinal, stage, status) in files),
            f"{arm} round {round_index} stage inventory drifted: {name}",
        )

  if not files:
    return {
        "status": "UNINSTRUMENTED",
        "receipt_count": 0,
        "last_completed_stage": None,
        "active_stage": None,
        "failure_stage": None,
        "failure_exit_code": None,
        "receipts": [],
    }

  terminal_seen = False
  last_completed: str | None = None
  active_stage: str | None = None
  failure_stage: str | None = None
  failure_exit_code: int | None = None
  for ordinal, stage in _STAGE_SPECS:
    statuses = {
        status for status in _STAGE_STATUSES
        if (ordinal, stage, status) in files
    }
    if not statuses:
      terminal_seen = True
      continue
    _require(not terminal_seen,
             f"{arm} round {round_index} has a stage after a pipeline gap: {stage}")
    _require("STARTED" in statuses,
             f"{arm} round {round_index} {stage} lacks STARTED")
    _require(not ({"PASS", "FAIL"} <= statuses),
             f"{arm} round {round_index} {stage} has PASS and FAIL")
    for status in _STAGE_STATUSES:
      path = files.get((ordinal, stage, status))
      if path is None:
        continue
      value = _json(path)
      exit_code = value.get("exit_code")
      _require(
          value.get("schema") == "m15-wide-round-stage-v1"
          and int(value.get("diagnostic_round", -1)) == round_index
          and value.get("runtime_source_commit") == source_commit
          and value.get("stage") == stage
          and value.get("status") == status
          and type(exit_code) is int
          and (
              (status in ("STARTED", "PASS") and exit_code == 0)
              or (status == "FAIL" and exit_code > 0)
          ),
          f"{arm} round {round_index} {stage}/{status} receipt drifted",
      )
      destination_name = (
          f"{arm}.round-{round_index:06d}.stage-"
          f"{ordinal}-{stage}-{status}.json"
      )
      destination = output / destination_name
      destination.write_bytes(path.read_bytes())
      receipts.append({
          "stage": stage,
          "ordinal": ordinal,
          "status": status,
          "exit_code": exit_code,
          "sha256": _sha256(path),
          "returned_file": destination_name,
      })
    if "PASS" in statuses:
      last_completed = stage
      continue
    active_stage = stage
    terminal_seen = True
    if "FAIL" in statuses:
      failure_stage = stage
      failure_exit_code = int(
          _json(files[(ordinal, stage, "FAIL")])["exit_code"]
      )

  if failure_stage is not None:
    status = "FAILED"
  elif active_stage is not None:
    status = "STARTED_ONLY"
  elif last_completed == _STAGE_SPECS[-1][1]:
    status = "PIPELINE_COMPLETE"
  else:
    status = "PROGRESS_ONLY"
  return {
      "status": status,
      "receipt_count": len(receipts),
      "last_completed_stage": last_completed,
      "active_stage": active_stage,
      "failure_stage": failure_stage,
      "failure_exit_code": failure_exit_code,
      "receipts": receipts,
  }


def _audit_round(
    directory: Path,
    *,
    arm: str,
    round_index: int,
    source_commit: str,
    output: Path,
) -> dict[str, Any]:
  inventory = _inventory(directory / "remote-inventory.txt")
  stage_state = _audit_stages(
      directory,
      arm=arm,
      round_index=round_index,
      source_commit=source_commit,
      output=output,
  )
  present = {name for name in _ROUND_FILES if (directory / name).is_file()}
  if not present and not any(inventory.values()):
    return {
        "diagnostic_round": round_index,
        "status": "ABSENT",
        "stage_state": stage_state,
    }
  if present != _ROUND_FILES and stage_state["receipt_count"]:
    return {
        "diagnostic_round": round_index,
        "status": "UNSEALED",
        "partial_round_files": sorted(present),
        "stage_state": stage_state,
    }
  _require(present == _ROUND_FILES,
           f"{arm} round {round_index} small evidence is partial: {sorted(present)}")
  _require(inventory.get("m15_wide_seam_bundle.tar") is True,
           f"{arm} round {round_index} compact bundle is absent remotely")
  manifest_path = directory / "WIDE_SHA256SUMS"
  manifest = _manifest(manifest_path)
  for name in ("ROUND_INPUT_RECEIPT.json", "p38_seam.classification.json"):
    _require(_sha256(directory / name) == manifest[name],
             f"{arm} round {round_index} {name} failed SHA")

  receipt = _json(directory / "ROUND_INPUT_RECEIPT.json")
  classification = _json(directory / "p38_seam.classification.json")
  completion = _json(directory / "WIDE_ROUND_COMPLETE.json")
  _require(
      receipt.get("schema") == "m15-wide-sealed-input-v1"
      and receipt.get("status") == "PASS"
      and int(receipt.get("diagnostic_round", -1)) == round_index
      and receipt.get("expected_source_commit") == source_commit
      and receipt.get("runtime_source_commit") == source_commit
      and int(receipt.get("record_pairs", 0)) > 0
      and int(receipt.get("replay_records", 0)) > 0,
      f"{arm} round {round_index} input receipt drifted",
  )
  classification_name = str(classification.get("classification", ""))
  boundary = classification.get("alignment", {})
  _require(
      classification.get("schema") == "m15-apc-wide-seam-classification-v1"
      and classification.get("status") == "PASS"
      and classification.get("arm") == arm
      and int(classification.get("diagnostic_round", -1)) == round_index
      and classification_name in _CLASSIFICATIONS[arm]
      and int(boundary.get("b_c_differing_bytes", -1)) == 0,
      f"{arm} round {round_index} classifier contract drifted",
  )
  ab_bytes = int(boundary.get("a_b_differing_bytes", -1))
  if arm == "off" or classification_name.endswith("_EXACT"):
    _require(ab_bytes == 0,
             f"{arm} round {round_index} exact classifier has A-B bytes")
  else:
    _require(ab_bytes > 0,
             f"{arm} round {round_index} red classifier has no A-B bytes")
  _require(
      completion.get("schema") == "m15-wide-round-completion-v1"
      and completion.get("status") == "classified-and-uploaded"
      and int(completion.get("diagnostic_round", -1)) == round_index
      and completion.get("expected_source_commit") == source_commit
      and completion.get("runtime_source_commit") == source_commit
      and completion.get("classification") == classification_name
      and completion.get("manifest_sha256") == _sha256(manifest_path)
      and int(completion.get("record_pairs", -1))
      == int(receipt["record_pairs"])
      and completion.get("shards") == receipt.get("shards"),
      f"{arm} round {round_index} completion receipt drifted",
  )
  _require(
      stage_state["status"] in ("UNINSTRUMENTED", "PIPELINE_COMPLETE"),
      f"{arm} round {round_index} is sealed but its stage pipeline is incomplete",
  )
  destination = output / f"{arm}.round-{round_index:06d}.classification.json"
  destination.write_text(
      json.dumps(classification, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
  )
  return {
      "diagnostic_round": round_index,
      "status": "SEALED",
      "classification": classification_name,
      "a_b_differing_bytes": ab_bytes,
      "b_c_differing_bytes": 0,
      "record_pairs": int(receipt["record_pairs"]),
      "replay_records": int(receipt["replay_records"]),
      "manifest_sha256": _sha256(manifest_path),
      "bundle_sha256": manifest["m15_wide_seam_bundle.tar"],
      "bundle_downloaded": False,
      "stage_state": stage_state,
  }


def _root_state(root: Path, source_commit: str) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for name in ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json"):
    path = root / name
    if not path.is_file():
      result[name] = {"present": False}
      continue
    value = _json(path)
    marker_source = value.get("source_commit")
    _require(marker_source in (None, "unknown", source_commit),
             f"root marker source drifted: {name}")
    result[name] = {
        "present": True,
        "sha256": _sha256(path),
        "status": value.get("status"),
        "source_commit": marker_source,
    }
  return result


def audit(
    *,
    source_commit: str,
    rounds: int,
    off_root: Path,
    on_root: Path,
    output: Path,
) -> dict[str, Any]:
  _require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
           "source commit must be a full lowercase SHA")
  _require(rounds == 3, "M15 wide return requires exactly three rounds")
  _require(not output.exists(), f"refusing to overwrite return: {output}")
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"stale partial return exists: {partial}")
  partial.mkdir(parents=True)
  try:
    arms: dict[str, Any] = {}
    for arm, root in (("off", off_root), ("on", on_root)):
      round_results = [
          _audit_round(
              root / f"round-{round_index:06d}",
              arm=arm,
              round_index=round_index,
              source_commit=source_commit,
              output=partial,
          )
          for round_index in range(rounds)
      ]
      arms[arm] = {
          "sealed_rounds": sum(row["status"] == "SEALED" for row in round_results),
          "rounds": round_results,
          "root_markers": _root_state(root / "root", source_commit),
      }
    sealed = [arms[arm]["sealed_rounds"] for arm in ("off", "on")]
    stage_states = [
        row["stage_state"]["status"]
        for arm in ("off", "on")
        for row in arms[arm]["rounds"]
    ]
    stage_receipt_count = sum(
        int(row["stage_state"]["receipt_count"])
        for arm in ("off", "on")
        for row in arms[arm]["rounds"]
    )
    stage_failure_count = stage_states.count("FAILED")
    root_complete = all(
        arms[arm]["root_markers"]["COLLECTED.json"]["present"]
        and arms[arm]["root_markers"]["COMPLETE.json"]["present"]
        for arm in ("off", "on")
    )
    if sealed == [rounds, rounds] and root_complete:
      status = "COMPLETE"
      next_action = "review all six sealed classifiers and choose the next numerical phase"
    elif sealed == [rounds, rounds]:
      status = "ROUNDS_RECOVERED_ROOT_INCOMPLETE"
      next_action = "use sealed per-round evidence; root run remains analysis-grade"
    elif any(sealed):
      status = "PARTIAL_ROUNDS_RECOVERED"
      next_action = "use recovered rounds; do not claim paired target completion"
    elif "FAILED" in stage_states:
      status = "ROUND_STAGE_FAILURE_IDENTIFIED"
      next_action = (
          "repair the reported failing stage; do not relaunch until its "
          "focused negative and positive controls pass"
      )
    elif any(value != "UNINSTRUMENTED" for value in stage_states):
      status = "ROUND_STAGE_PROGRESS_ONLY"
      next_action = (
          "inspect the last completed/active stage and terminal worker log "
          "before another target launch"
      )
    else:
      status = "NO_DURABLE_ROUND"
      next_action = "inspect worker/upload failure before another target launch"
    result = {
        "schema": "m15-apc-multiround-small-return-v1",
        "status": status,
        "source_commit": source_commit,
        "expected_rounds_per_arm": rounds,
        "arms": arms,
        "next_action": next_action,
        "claim_ceiling": (
            "Only SEALED rows establish independently durable per-round "
            "classifiers. Stage receipts locate persistence progress/failure "
            "but carry no numerical equality claim. Root COLLECTED/COMPLETE "
            "are still required for a full signed target PASS."
        ),
    }
    (partial / "MULTIROUND_SUMMARY.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (partial / "PACKAGING.txt").write_text(
        "M15 three-round small evidence return\n"
        f"status={status}\n"
        f"off_sealed_rounds={sealed[0]}\n"
        f"on_sealed_rounds={sealed[1]}\n"
        f"stage_receipts={stage_receipt_count}\n"
        f"stage_failures={stage_failure_count}\n"
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
    return result
  except Exception:
    shutil.rmtree(partial, ignore_errors=True)
    raise


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--rounds", required=True, type=int)
  parser.add_argument("--off-root", required=True, type=Path)
  parser.add_argument("--on-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = audit(
        source_commit=args.source_commit,
        rounds=args.rounds,
        off_root=args.off_root,
        on_root=args.on_root,
        output=args.output,
    )
  except (OSError, ValueError, json.JSONDecodeError, MultiRoundAuditError) as exc:
    raise SystemExit(f"M15_MULTIROUND_RETURN_RED {exc}") from exc
  print(f"M15_MULTIROUND_RETURN_COMPLETE status={result['status']} output={args.output}")


if __name__ == "__main__":
  main()
