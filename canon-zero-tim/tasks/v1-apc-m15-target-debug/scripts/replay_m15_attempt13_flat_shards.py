#!/usr/bin/env python3
"""Replay Attempt 13 from one verified flat-shard union per arm.

This module never reads GCS directly. The shell wrapper supplies extracted,
hash-verified ``wide/shards`` members plus one verified live snapshot containing
the host-only alignment, replay-ledger, and mismatch-capsule inputs. Raw
token-bearing inputs remain in scratch; only small classifier receipts leave it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from assemble_m15_wide_round import assemble
from classify_m15_apc_wide_seam import classify
from package_m15_apc_wide_seam import package
from stage_m15_wide_shard import M15WideShardError


SOURCE_COMMIT = "7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
EXPECTED_LAYER = 0
EXPECTED_SHARDS = {"off": 77, "on": 70}
EXPECTED_RECORD_PAIRS = {"off": 2474, "on": 2087}
EXPECTED_ALIGNMENT = {
    # The exact-arm classifier intentionally omits a redundant element count.
    "off": {"a_b_differing_bytes": 0},
    "on": {"a_b_differing_bytes": 239, "a_b_differing_elements": 114},
}


class Attempt13FlatReplayError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt13FlatReplayError(message)


def _json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"{label} is absent: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as error:
    raise Attempt13FlatReplayError(f"{label} is invalid: {path}") from error
  _require(isinstance(value, dict), f"{label} is not an object: {path}")
  return value


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
  _require(path.is_file() and path.stat().st_size > 0,
           f"manifest is absent: {path}")
  rows: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(separator == "  " and len(digest) == 64 and name,
             f"invalid manifest row: {line!r}")
    _require("/" not in name and name not in rows,
             f"unsafe or duplicate manifest member: {name!r}")
    rows[name] = digest
  return rows


def _validate_shards(
    shard_root: Path,
    *,
    expected_shards: int,
    expected_pairs: int,
    source_commit: str,
) -> dict[str, Any]:
  _require(shard_root.is_dir(), f"flat shard root is absent: {shard_root}")
  directories = sorted(path for path in shard_root.iterdir() if path.is_dir())
  expected_names = [f"{index:06d}" for index in range(expected_shards)]
  _require([path.name for path in directories] == expected_names,
           "flat shard sequences are incomplete, duplicated, or non-contiguous")
  total_pairs = 0
  total_payload_bytes = 0
  members: set[str] = set()
  receipts = []
  for sequence, directory in enumerate(directories):
    completion = _json(directory / "SHARD_COMPLETE.json", "shard completion")
    inventory = _json(directory / "SHARD_INVENTORY.json", "shard inventory")
    rows = _manifest(directory / "SHA256SUMS")
    _require(
        completion.get("schema") == "m15-wide-observer-shard-completion-v1"
        and completion.get("status") == "sealed-uploaded-verified"
        and int(completion.get("sequence", -1)) == sequence
        and int(completion.get("diagnostic_round", -1)) == 0
        and completion.get("expected_source_commit") == source_commit
        and completion.get("runtime_source_commit") == source_commit,
        f"shard completion contract drifted: {directory.name}",
    )
    _require(
        inventory.get("schema") == "m15-wide-observer-shard-v1"
        and int(inventory.get("sequence", -1)) == sequence
        and int(inventory.get("diagnostic_round", -1)) == 0,
        f"shard inventory contract drifted: {directory.name}",
    )
    _require(completion.get("manifest_sha256") == _sha(directory / "SHA256SUMS"),
             f"shard manifest identity drifted: {directory.name}")
    inventory_files = list(inventory.get("files", ()))
    _require(set(rows) == set(inventory_files) | {"SHARD_INVENTORY.json"},
             f"shard manifest membership drifted: {directory.name}")
    for name, digest in rows.items():
      member = directory / name
      _require(member.is_file() and _sha(member) == digest,
               f"shard member failed SHA: {directory.name}/{name}")
    for name in inventory_files:
      _require(name not in members, f"observer record appears in two shards: {name}")
      members.add(name)
    record_pairs = int(inventory.get("record_pairs", -1))
    payload_bytes = int(inventory.get("payload_bytes", -1))
    _require(
        record_pairs > 0
        and record_pairs == int(completion.get("record_pairs", -2))
        and payload_bytes > 0
        and payload_bytes == int(completion.get("payload_bytes", -2)),
        f"shard count contract drifted: {directory.name}",
    )
    total_pairs += record_pairs
    total_payload_bytes += payload_bytes
    receipts.append({
        "sequence": sequence,
        "record_pairs": record_pairs,
        "payload_bytes": payload_bytes,
        "manifest_sha256": _sha(directory / "SHA256SUMS"),
        "archive_sha256": completion.get("archive_sha256"),
    })
  _require(total_pairs == expected_pairs,
           f"flat shard record-pair count drifted: {total_pairs} != {expected_pairs}")
  _require(len(members) == expected_pairs * 2,
           "flat shard JSON/NPZ member cardinality drifted")
  return {
      "shards": expected_shards,
      "record_pairs": total_pairs,
      "payload_bytes": total_payload_bytes,
      "record_files": len(members),
      "receipts": receipts,
  }


def _live_inputs(live_root: Path, arm: str, source_commit: str) -> dict[str, Any]:
  live = _json(live_root / "LIVE.json", "selected live receipt")
  _require(
      live.get("schema") == "canon-p38-gcs-live-v1"
      and live.get("status") == "live-snapshot"
      and live.get("source_commit") == source_commit,
      f"{arm} live snapshot contract drifted",
  )
  rows = _manifest(live_root / "SHA256SUMS")
  _require(live.get("manifest_sha256") == _sha(live_root / "SHA256SUMS"),
           f"{arm} live manifest identity drifted")
  for name, digest in rows.items():
    member = live_root / name
    _require(member.is_file() and _sha(member) == digest,
             f"{arm} live member failed SHA: {name}")
  _require(set(rows) == set(live.get("files", ())),
           f"{arm} live file inventory drifted")
  required = {"pre-alignment.jsonl", "m15-replay-envelope.jsonl",
              "diagnostic-round.txt"}
  _require(required <= set(rows), f"{arm} live snapshot lacks replay inputs")
  _require((live_root / "diagnostic-round.txt").read_text().strip() == "0",
           f"{arm} live snapshot is not diagnostic round 0")
  capsules = sorted(
      path for path in live_root.glob("*.npz") if "capsule" in path.name
  )
  if arm == "on":
    _require(capsules, "on live snapshot lacks a mismatch capsule")
    round_capsules = [path for path in capsules if "round-000000" in path.name]
    capsule = round_capsules[0] if round_capsules else capsules[0]
  else:
    capsule = None
  return {
      "receipt": live,
      "pre_alignment": live_root / "pre-alignment.jsonl",
      "replay_ledger": live_root / "m15-replay-envelope.jsonl",
      "capsule": capsule,
      "manifest_sha256": _sha(live_root / "SHA256SUMS"),
  }


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")


def _replay_arm(
    *,
    arm: str,
    root: Path,
    work: Path,
    output: Path,
    source_commit: str,
    expected_shards: int,
    expected_pairs: int,
    expected_alignment: dict[str, int],
    expected_layer: int,
) -> dict[str, Any]:
  shard_summary = _validate_shards(
      root / "shards",
      expected_shards=expected_shards,
      expected_pairs=expected_pairs,
      source_commit=source_commit,
  )
  live = _live_inputs(root / "live", arm, source_commit)
  empty_live = work / f"{arm}-empty-live"
  empty_live.mkdir()
  round_root = work / f"{arm}-round-000000"
  capsule = live["capsule"] or work / f"{arm}-absent-capsule.npz"
  input_receipt = assemble(
      live_directory=empty_live,
      shard_root=root / "shards",
      output=round_root,
      round_index=0,
      pre_alignment=live["pre_alignment"],
      capsule=capsule,
      replay_ledger=live["replay_ledger"],
      observer_mode="full",
      expected_commit=source_commit,
      runtime_commit=source_commit,
  )
  _require(int(input_receipt["record_pairs"]) == expected_pairs,
           f"{arm} assembled record-pair count drifted")
  round_capsule = round_root / "mismatch-capsule.npz"
  capsules = [round_capsule] if round_capsule.is_file() else []
  classification = classify(
      directory=round_root,
      alignment_report=round_root / "pre-alignment.jsonl",
      capsules=capsules,
      mode="full",
      arm=arm,
      replay_ledger=round_root / "m15-replay-envelope.jsonl",
      expected_layer=expected_layer,
      require_first_action=arm == "on",
  )
  _require(
      classification.get("schema") == "m15-apc-wide-seam-classification-v1"
      and classification.get("status") == "PASS"
      and int(classification.get("diagnostic_round", -1)) == 0
      and classification.get("observer_mode") == "full",
      f"{arm} official classifier contract drifted",
  )
  expected_classification = (
      "M15_OBSERVER_CONTROL_EXACT"
      if arm == "off" else "M15_INTERNAL_FIRST_RED_LOCALIZED"
  )
  _require(classification.get("classification") == expected_classification,
           f"{arm} official classifier verdict drifted")
  if arm == "on":
    required_fields = {
        "anchors", "expected_layer", "first_difference_signatures",
        "mixed_first_difference_signatures", "replay_ledger_receipts",
    }
    _require(required_fields <= set(classification),
             "on official classifier omitted localization fields")
    _require(int(classification.get("expected_layer", -1)) == expected_layer,
             "on official classifier expected-layer drifted")
  alignment = classification["alignment"]
  _require(int(alignment.get("b_c_differing_bytes", -1)) == 0,
           f"{arm} official replay made B-C red")
  for key, expected in expected_alignment.items():
    _require(int(alignment.get(key, -1)) == expected,
             f"{arm} official replay {key} drifted")
  classification_path = output / f"{arm}.round-000000.classification.json"
  _write_json(classification_path, classification)
  input_path = output / f"{arm}.round-000000.input-receipt.json"
  _write_json(input_path, input_receipt)
  bundle_path = work / f"{arm}.m15_wide_seam_bundle.tar"
  bundle = package(
      directory=round_root,
      classification_path=classification_path,
      alignment_report=round_root / "pre-alignment.jsonl",
      capsules=capsules,
      replay_ledger=round_root / "m15-replay-envelope.jsonl",
      output=bundle_path,
  )
  return {
      "arm": arm,
      "classification": classification["classification"],
      "alignment": alignment,
      "last_exact_boundary": classification.get("last_exact_boundary"),
      "first_red_boundary": classification.get("first_red_boundary"),
      "source_interval": classification.get("source_interval"),
      "shard_union": shard_summary,
      "live_sequence": int(live["receipt"].get("sequence", -1)),
      "live_manifest_sha256": live["manifest_sha256"],
      "compact_bundle": {
          "returned": False,
          "bytes": int(bundle["bytes"]),
          "sha256": bundle["sha256"],
      },
  }


def replay(
    *,
    off_root: Path,
    on_root: Path,
    work: Path,
    output: Path,
    source_commit: str = SOURCE_COMMIT,
    expected_shards: dict[str, int] = EXPECTED_SHARDS,
    expected_pairs: dict[str, int] = EXPECTED_RECORD_PAIRS,
    expected_alignment: dict[str, dict[str, int]] = EXPECTED_ALIGNMENT,
    expected_layer: int = EXPECTED_LAYER,
) -> dict[str, Any]:
  _require(source_commit == SOURCE_COMMIT,
           "Attempt-13 source commit is not admitted")
  _require(not output.exists(), f"refusing to overwrite replay output: {output}")
  _require(not work.exists(), f"refusing to overwrite replay work: {work}")
  output_partial = output.with_name(output.name + ".partial")
  _require(not output_partial.exists(), f"stale replay partial exists: {output_partial}")
  work.mkdir(parents=True)
  output_partial.mkdir(parents=True)
  try:
    arms = {}
    for arm, root in (("off", off_root), ("on", on_root)):
      arms[arm] = _replay_arm(
          arm=arm,
          root=root,
          work=work,
          output=output_partial,
          source_commit=source_commit,
          expected_shards=expected_shards[arm],
          expected_pairs=expected_pairs[arm],
          expected_alignment=expected_alignment[arm],
          expected_layer=expected_layer,
      )
    on = arms["on"]
    boundary_repeat = (
        on["last_exact_boundary"] is not None
        and on["first_red_boundary"] is not None
        and on["last_exact_boundary"].get("layer") == expected_layer
        and on["last_exact_boundary"].get("checkpoint") == "k_post_rope"
        and on["first_red_boundary"].get("layer") == expected_layer
        and on["first_red_boundary"].get("checkpoint") == "rpa_output"
    )
    decision = (
        "SINGLE_ROUND_ATTENTION_INTERVAL_REPRODUCED"
        if boundary_repeat else "SINGLE_ROUND_OFFICIAL_REPLAY_DISAGREES"
    )
    result = {
        "schema": "m15-attempt13-flat-shard-official-replay-v1",
        "status": "PASS",
        "attempt": 13,
        "source_commit": source_commit,
        "diagnostic_round": 0,
        "layout": "wide/shards plus live snapshot",
        "decision": decision,
        "arms": arms,
        "official_classifier_replay": "PERFORMED_FROM_VERIFIED_FLAT_SHARD_UNIONS",
        "three_round_repeat": "NOT_PERFORMED",
        "numerical_repair_authorized": False,
        "claim_ceiling": (
            "One historical DP8xTP8 round was independently replayed from "
            "sealed flat shards. This does not establish three-round stability "
            "or localize a degree of freedom inside the attention call."
        ),
    }
    _write_json(output_partial / "ATTEMPT13_FLAT_REPLAY.json", result)
    (output_partial / "PACKAGING.txt").write_text(
        "M15 Attempt-13 flat-shard official replay\n"
        f"decision={decision}\n"
        "diagnostic_rounds=1\n"
        "token_bearing_bundle_returned=0\n"
        "remote_state_mutated=0\n"
        "numerical_repair_authorized=0\n",
        encoding="utf-8",
    )
    names = sorted(path.name for path in output_partial.iterdir())
    (output_partial / "SHA256SUMS").write_text(
        "".join(f"{_sha(output_partial / name)}  {name}\n" for name in names),
        encoding="ascii",
    )
    output_partial.replace(output)
    return result
  except Exception:
    shutil.rmtree(output_partial, ignore_errors=True)
    raise


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--off-root", required=True, type=Path)
  parser.add_argument("--on-root", required=True, type=Path)
  parser.add_argument("--work", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = replay(
      off_root=args.off_root,
      on_root=args.on_root,
      work=args.work,
      output=args.output,
  )
  print(
      "M15_ATTEMPT13_FLAT_REPLAY_COMPLETE "
      f"decision={result['decision']} rounds=1 numerical_repair_authorized=0"
  )
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (Attempt13FlatReplayError, M15WideShardError, OSError, ValueError) as error:
    print(f"M15_ATTEMPT13_FLAT_REPLAY_RED {error}", file=sys.stderr)
    raise SystemExit(2) from error
