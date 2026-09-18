#!/usr/bin/env python3
"""Audit three independent P38s22 round seals without a root postflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from typing import Any


CONTRACT_SCHEMA = "p38s22-round-salvage-contract-v1"
RESULT_SCHEMA = "p38s22-round-salvage-audit-v1"
ACQUISITION_SCHEMA = "p38s22-round-salvage-acquisition-v1"
_ROUND_LINE = re.compile(
    r"\[CANON_P38\] PRECHECK_ROUND_COMPLETE .*"
    r"backward=0 optimizer_commits=0"
)


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise ValueError(f"cannot load helper module: {path.name}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = _load_module(
    "p38s22_offsite_base", SCRIPT_DIR / "audit_p38s22_offsite.py")
ARCHIVE = _load_module(
    "p38s22_round_archive", SCRIPT_DIR / "p38_evidence_archive.py")


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  return BASE._load_json(path)  # pylint: disable=protected-access


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
  return BASE._load_jsonl(  # pylint: disable=protected-access
      path, label=label)


def _load_acquisition(path: Path) -> dict[str, dict[str, Any]]:
  records = _load_jsonl(path, "acquisition")
  by_label: dict[str, dict[str, Any]] = {}
  for record in records:
    _require(record.get("schema") == ACQUISITION_SCHEMA,
             "acquisition schema drifted")
    label = record.get("label")
    _require(isinstance(label, str) and label and label not in by_label,
             f"invalid or duplicate acquisition label: {label!r}")
    _require(record.get("status") in ("downloaded", "missing_or_unreadable"),
             f"acquisition status drifted: {label}")
    _require(isinstance(record.get("required"), bool),
             f"acquisition required flag drifted: {label}")
    by_label[label] = record
  return by_label


def _verify_acquired(
    acquisition: dict[str, dict[str, Any]], label: str, path: Path
) -> None:
  _require(label in acquisition, f"acquisition record is absent: {label}")
  record = acquisition[label]
  _require(record["status"] == "downloaded",
           f"required source object is unavailable: {label}")
  _require(path.is_file() and not path.is_symlink(),
           f"downloaded source object is absent or unsafe: {label}")
  _require(int(record.get("size_bytes", -1)) == path.stat().st_size,
           f"downloaded source size drifted: {label}")
  _require(record.get("sha256") == _sha256(path),
           f"downloaded source SHA drifted: {label}")


def _root_postflight(
    acquisition: dict[str, dict[str, Any]], source_root: Path
) -> dict[str, Any]:
  fields = {
      "preflight": "PREFLIGHT.json",
      "collected": "COLLECTED.json",
      "complete": "COMPLETE.json",
      "root_manifest": "SHA256SUMS",
  }
  result: dict[str, Any] = {}
  for key, name in fields.items():
    label = f"root/{name}"
    record = acquisition.get(label, {})
    result[f"{key}_present"] = (
        record.get("status") == "downloaded" and
        (source_root / name).is_file()
    )
  result["receipts_present"] = all(result.values())
  result["admitted"] = False
  result["reason"] = (
      "round_salvage_does_not_authenticate_root_manifest_members"
  )
  return result


def _validate_stage_jsonl(
    extracted: Path,
    inventory: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
  pre_records = _load_jsonl(
      extracted / "pre-alignment.jsonl", "round pre-alignment")
  _require(len(pre_records) == 1,
           f"round {round_index} pre-alignment record count drifted")
  pre_record = pre_records[0]
  _require(int(pre_record.get("diagnostic_round", -1)) == round_index,
           f"round {round_index} pre-alignment scope drifted")
  _require(int(inventory.get("pre_alignment_records", -1)) == 1,
           f"round {round_index} pre-alignment inventory drifted")

  journals = _load_jsonl(
      extracted / "request-journal.jsonl", "request journal")
  _require(inventory.get("journal_scope") == "cumulative-unscoped",
           f"round {round_index} request-journal scope drifted")
  _require(len(journals) == int(inventory.get("journal_records", -1)),
           f"round {round_index} request-journal count drifted")
  _require(all(item.get("schema") == "p38-request-journal-v1"
               for item in journals),
           f"round {round_index} request-journal schema drifted")

  incidents = _load_jsonl(
      extracted / "incident-ledger.jsonl", "incident ledger")
  _require(len(incidents) == int(inventory.get("incident_records", -1)),
           f"round {round_index} incident-ledger count drifted")
  _require(all(
      item.get("schema") == "p38-incident-ledger-v1" and
      int(item.get("diagnostic_round", -1)) == round_index
      for item in incidents
  ), f"round {round_index} incident-ledger scope/schema drifted")
  return pre_record


def _validate_observers(
    extracted: Path,
    inventory: dict[str, Any],
    round_index: int,
) -> dict[str, int]:
  counts = {
      "kv_records": BASE._observer_pair_count(  # pylint: disable=protected-access
          extracted, "p38_kv_observer", "p38-live-kv-prefix-table-v1",
          round_index),
      "seam_records": BASE._observer_pair_count(  # pylint: disable=protected-access
          extracted, "p38_seam", "p38-seam-fingerprint-v1", round_index),
      "tail_records": BASE._observer_pair_count(  # pylint: disable=protected-access
          extracted, "p38_tail", "p38-tail-values-v1", round_index),
      "terminal_records": BASE._observer_pair_count(  # pylint: disable=protected-access
          extracted, "p38_terminal", "p38-terminal-discriminator-v1",
          round_index),
  }
  for name, observed in counts.items():
    _require(int(inventory.get(name, -1)) == observed,
             f"round {round_index} {name} inventory differs from files")
  _require(counts["kv_records"] > 0,
           f"round {round_index} required KV evidence is empty")
  _require(
      counts["seam_records"] == 0 and
      counts["tail_records"] == 0 and
      counts["terminal_records"] == 0,
      f"round {round_index} forbidden observer evidence is present",
  )
  return counts


def _validate_run_log(
    path: Path,
    round_index: int,
    preset: str,
) -> dict[str, int]:
  text = path.read_text(encoding="utf-8", errors="replace")
  _require(f"[PATHTRACE] CANON_MM_ALGO on preset={preset}" in text,
           f"round {round_index} algorithm PATHTRACE is absent")
  round_lines = _ROUND_LINE.findall(text)
  _require(round_index + 1 <= len(round_lines) <= 3,
           f"round {round_index} frozen-round marker count drifted")
  _require("[CANON_P38_TERMINAL_DISCRIMINATOR_INIT]" not in text and
           "[CANON_P38_TERMINAL_DISCRIMINATOR_RECORD]" not in text,
           f"round {round_index} terminal discriminator unexpectedly ran")
  for completed in range(1, round_index + 1):
    marker = (
        "[CANON_P38] DIAGNOSTIC_ROUND_SKIPPED_UPDATE "
        f"completed={completed}/3 backward=0 optimizer_commits=0 "
        "weights=frozen"
    )
    _require(marker in text,
             f"round {round_index} lacks frozen prior-round marker {completed}")
  return {
      "precheck_round_complete_markers": len(round_lines),
      "prior_frozen_update_skips": round_index,
  }


def _audit_round(
    round_index: int,
    expected: dict[str, Any],
    round_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
  marker = _load_json(round_dir / "ROUND_COMPLETE.json")
  manifest = round_dir / "SHA256SUMS"
  archive = round_dir / "ROUND_ARCHIVE.tar"
  _require(marker.get("schema") == "canon-p38-round-completion-v1",
           f"round {round_index} marker schema drifted")
  _require(marker.get("status") == "sealed-and-verified",
           f"round {round_index} is not sealed-and-verified")
  _require(marker.get("source_commit") == contract["expected_source_commit"],
           f"round {round_index} source commit drifted")
  _require(str(marker.get("attempt")) == contract["expected_attempt"],
           f"round {round_index} attempt drifted")
  _require(int(marker.get("diagnostic_round", -1)) == round_index,
           f"round {round_index} marker identity drifted")
  _require(marker.get("archive_name") == "ROUND_ARCHIVE.tar" and
           marker.get("transport") == "single-deterministic-tar-v1",
           f"round {round_index} transport drifted")

  archive_sha = _sha256(archive)
  manifest_sha = _sha256(manifest)
  _require(archive_sha == expected["archive_sha256"] and
           marker.get("archive_sha256") == archive_sha,
           f"round {round_index} archive SHA differs from contract/marker")
  _require(manifest_sha == expected["manifest_sha256"] and
           marker.get("manifest_sha256") == manifest_sha,
           f"round {round_index} manifest SHA differs from contract/marker")
  logical_count, verified_sha = ARCHIVE.verify_archive(archive, archive_sha)
  _require(verified_sha == archive_sha,
           f"round {round_index} archive verifier SHA drifted")
  _require(logical_count == int(expected["logical_file_count"]) and
           logical_count == int(marker.get("logical_file_count", -1)),
           f"round {round_index} logical file count drifted")

  extracted = round_dir / "extracted"
  _require(not extracted.exists(),
           f"round {round_index} extraction destination already exists")
  ARCHIVE.extract_archive(archive, extracted)
  _require((extracted / "SHA256SUMS").read_bytes() == manifest.read_bytes(),
           f"round {round_index} archived/remote manifests differ")
  members = BASE._verify_manifest(  # pylint: disable=protected-access
      extracted, extracted / "SHA256SUMS")
  required = set(contract["required_round_members"])
  _require(required <= set(members),
           f"round {round_index} required members are absent: "
           f"{sorted(required - set(members))}")

  inventory = _load_json(extracted / "ROUND_INVENTORY.json")
  _require(inventory.get("schema") == "canon-p38-round-stage-v1",
           f"round {round_index} inventory schema drifted")
  _require(int(inventory.get("diagnostic_round", -1)) == round_index,
           f"round {round_index} inventory scope drifted")
  pre_record = _validate_stage_jsonl(extracted, inventory, round_index)
  observer_counts = _validate_observers(extracted, inventory, round_index)
  run_log_counts = _validate_run_log(
      extracted / "run.log", round_index,
      contract["expected_mm_algo_preset"])

  capsule = extracted / "mismatch-capsule.npz"
  capsule_sha = _sha256(capsule)
  _require(capsule_sha == expected["capsule_sha256"],
           f"round {round_index} capsule SHA drifted")
  counts = BASE._capsule_counts(capsule)  # pylint: disable=protected-access
  ab = BASE._boundary(  # pylint: disable=protected-access
      pre_record, "S_decode_vs_S_prefill")
  bc = BASE._boundary(  # pylint: disable=protected-access
      pre_record, "S_prefill_vs_T_old")
  _require(int(pre_record.get("N_action", -1)) == int(expected["n_action"]),
           f"round {round_index} N_action drifted")
  expected_fields = (
      "a_b_differing_elements", "a_b_differing_bytes",
      "b_c_differing_elements", "b_c_differing_bytes",
  )
  for field in expected_fields:
    _require(int(counts[field]) == int(expected[field]),
             f"round {round_index} capsule {field} drifted")
  _require(float(counts["a_b_max_abs"]) == float(expected["a_b_max_abs"]),
           f"round {round_index} capsule A-B max_abs drifted")
  _require(float(counts["b_c_max_abs"]) == float(expected["b_c_max_abs"]),
           f"round {round_index} capsule B-C max_abs drifted")
  boundary_pairs = (
      (ab, "differing_elements", "a_b_differing_elements"),
      (ab, "differing_bytes", "a_b_differing_bytes"),
      (bc, "differing_elements", "b_c_differing_elements"),
      (bc, "differing_bytes", "b_c_differing_bytes"),
  )
  for boundary, source_field, expected_field in boundary_pairs:
    _require(int(boundary.get(source_field, -1)) == int(expected[expected_field]),
             f"round {round_index} pre-alignment {expected_field} drifted")
  _require(float(ab.get("max_abs")) == float(expected["a_b_max_abs"]),
           f"round {round_index} pre-alignment A-B max_abs drifted")
  _require(float(bc.get("max_abs")) == float(expected["b_c_max_abs"]),
           f"round {round_index} pre-alignment B-C max_abs drifted")
  return {
      "diagnostic_round": round_index,
      "n_action": int(expected["n_action"]),
      "a_b_differing_elements": int(expected["a_b_differing_elements"]),
      "a_b_differing_bytes": int(expected["a_b_differing_bytes"]),
      "a_b_max_abs": float(expected["a_b_max_abs"]),
      "b_c_differing_elements": int(expected["b_c_differing_elements"]),
      "b_c_differing_bytes": int(expected["b_c_differing_bytes"]),
      "b_c_max_abs": float(expected["b_c_max_abs"]),
      "archive_sha256": archive_sha,
      "manifest_sha256": manifest_sha,
      "capsule_sha256": capsule_sha,
      "logical_file_count": logical_count,
      "observer_counts": observer_counts,
      "run_log_counts": run_log_counts,
  }


def audit(args: argparse.Namespace) -> dict[str, Any]:
  contract = _load_json(args.contract)
  _require(contract.get("schema") == CONTRACT_SCHEMA,
           "round-salvage contract schema drifted")
  _require(contract.get("root_postflight_required_for_round_verdict") is False,
           "round-salvage root policy drifted")
  _require(contract.get("forbid_terminal_observer") is True,
           "round-salvage terminal-observer policy drifted")
  expected_rounds = contract.get("expected_rounds")
  _require(isinstance(expected_rounds, list) and len(expected_rounds) == 3,
           "round-salvage contract must define three rounds")
  by_round = {int(item["diagnostic_round"]): item for item in expected_rounds}
  _require(set(by_round) == {0, 1, 2}, "round identities drifted")

  acquisition = _load_acquisition(args.acquisition)
  _verify_acquired(
      acquisition, "root/PREFLIGHT.json", args.source_root / "PREFLIGHT.json")
  preflight = _load_json(args.source_root / "PREFLIGHT.json")
  _require(preflight.get("schema") == "canon-p38-gcs-preflight-v1" and
           preflight.get("status") == "writable",
           "PREFLIGHT contract drifted")
  _require(preflight.get("source_commit") == contract["expected_source_commit"] and
           str(preflight.get("attempt")) == contract["expected_attempt"] and
           preflight.get("prefix") == contract["source_gcs_uri"],
           "PREFLIGHT source identity drifted")
  collected_path = args.source_root / "COLLECTED.json"
  if collected_path.is_file():
    collected = _load_json(collected_path)
    _require(collected.get("schema") == "canon-p38-gcs-collection-v1" and
             collected.get("status") == "collected" and
             collected.get("source_commit") == contract["expected_source_commit"] and
             str(collected.get("attempt")) == contract["expected_attempt"] and
             collected.get("jobset") == contract["expected_jobset"] and
             collected.get("prefix") == contract["source_gcs_uri"],
             "optional COLLECTED receipt identity drifted")
  complete_path = args.source_root / "COMPLETE.json"
  if complete_path.is_file():
    complete = _load_json(complete_path)
    _require(complete.get("schema") == "canon-p38-gcs-completion-v1" and
             complete.get("status") == "postflight-accepted" and
             complete.get("source_commit") == contract["expected_source_commit"] and
             str(complete.get("attempt")) == contract["expected_attempt"] and
             complete.get("prefix") == contract["source_gcs_uri"],
             "optional COMPLETE receipt identity drifted")

  for round_index in range(3):
    for name in ("ROUND_ARCHIVE.tar", "SHA256SUMS", "ROUND_COMPLETE.json"):
      label = f"rounds/{round_index:06d}/{name}"
      _verify_acquired(
          acquisition, label, args.round_root / f"{round_index:06d}/{name}")

  rendered = args.reference_evidence / "rendered-stock.yaml"
  _require(rendered.is_file(), "registered rendered-stock.yaml is absent")
  rendered_text = rendered.read_text(encoding="utf-8")
  _require("name: CANON_P38_TERMINAL_DISCRIMINATOR" not in rendered_text,
           "P38s22 rendered contract unexpectedly enabled terminal evidence")

  rounds = [
      _audit_round(i, by_round[i], args.round_root / f"{i:06d}", contract)
      for i in range(3)
  ]
  totals = {
      "n_action": sum(item["n_action"] for item in rounds),
      "a_b_differing_elements": sum(
          item["a_b_differing_elements"] for item in rounds),
      "a_b_differing_bytes": sum(
          item["a_b_differing_bytes"] for item in rounds),
      "b_c_differing_elements": sum(
          item["b_c_differing_elements"] for item in rounds),
      "b_c_differing_bytes": sum(
          item["b_c_differing_bytes"] for item in rounds),
  }
  _require(totals["a_b_differing_elements"] > 0 and
           totals["b_c_differing_elements"] == 0,
           "round-sealed numerical result differs from preregistration")
  return {
      "schema": RESULT_SCHEMA,
      "status": "PASS",
      "verdict": "ROUND_SEALED_GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED",
      "claim_ceiling": (
          "P38s22_three_round_forward_discriminator_only;_root_postflight,_"
          "terminal_localization,_backward,_and_optimizer_completion_unadmitted"
      ),
      "analysis_source_commit": args.analysis_source_commit,
      "contract_sha256": _sha256(args.contract),
      "tool_sha256": {
          "auditor": _sha256(Path(__file__).resolve()),
          "archive": _sha256(SCRIPT_DIR / "p38_evidence_archive.py"),
          "base_auditor": _sha256(SCRIPT_DIR / "audit_p38s22_offsite.py"),
          "wrapper": _sha256(SCRIPT_DIR / "run_p38s22_round_salvage.sh"),
      },
      "source_uri_sha256": hashlib.sha256(
          contract["source_gcs_uri"].encode("utf-8")).hexdigest(),
      "root_postflight": _root_postflight(acquisition, args.source_root),
      "rounds": rounds,
      "totals": totals,
      "terminal_classification": {
          "admitted": False,
          "reason": "P38s22_forbade_terminal_observer",
      },
      "next_gate": "dedicated_fixed_tile_Pallas_lm_head_onehost_then_P38s23",
  }


def _copy_if_present(source: Path, destination: Path) -> None:
  if source.is_file():
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_output(args: argparse.Namespace, result: dict[str, Any]) -> None:
  _require(not args.output.exists(), f"output already exists: {args.output}")
  args.output.mkdir(parents=True, mode=0o700)
  shutil.copyfile(args.acquisition, args.output / "ACQUISITION.jsonl")
  receipts = args.output / "receipts"
  receipts.mkdir()
  for name in ("PREFLIGHT.json", "COLLECTED.json", "COMPLETE.json", "SHA256SUMS"):
    output_name = "ROOT_SHA256SUMS" if name == "SHA256SUMS" else name
    _copy_if_present(args.source_root / name, receipts / output_name)
  for round_index in range(3):
    source = args.round_root / f"{round_index:06d}"
    _copy_if_present(
        source / "ROUND_COMPLETE.json",
        receipts / f"ROUND_COMPLETE.round-{round_index:06d}.json")
    _copy_if_present(
        source / "SHA256SUMS",
        receipts / f"ROUND_SHA256SUMS.round-{round_index:06d}")
  (args.output / "AUDIT.json").write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  verdict = {
      "schema": "p38s22-round-salvage-verdict-v1",
      "status": result["status"],
      "verdict": result["verdict"],
      "claim_ceiling": result.get("claim_ceiling", "none"),
      "next_gate": result.get("next_gate", "repair_round_evidence_without_TPU"),
  }
  (args.output / "verdict.json").write_text(
      json.dumps(verdict, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  totals = result.get("totals", {})
  root = result.get("root_postflight", {})
  (args.output / "SUMMARY.txt").write_text("\n".join((
      f"status={result['status']}",
      f"verdict={result['verdict']}",
      f"n_action={totals.get('n_action', 'unknown')}",
      f"a_b_differing_elements={totals.get('a_b_differing_elements', 'unknown')}",
      f"a_b_differing_bytes={totals.get('a_b_differing_bytes', 'unknown')}",
      f"b_c_differing_elements={totals.get('b_c_differing_elements', 'unknown')}",
      f"b_c_differing_bytes={totals.get('b_c_differing_bytes', 'unknown')}",
      f"root_postflight_receipts_present={root.get('receipts_present', 'unknown')}",
      f"root_postflight_admitted={root.get('admitted', False)}",
      f"next_gate={result.get('next_gate', 'repair_round_evidence_without_TPU')}",
  )) + "\n", encoding="utf-8")
  BASE._seal(args.output)  # pylint: disable=protected-access


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--contract", required=True, type=Path)
  parser.add_argument("--source-root", required=True, type=Path)
  parser.add_argument("--round-root", required=True, type=Path)
  parser.add_argument("--acquisition", required=True, type=Path)
  parser.add_argument("--reference-evidence", required=True, type=Path)
  parser.add_argument("--analysis-source-commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = audit(args)
    rc = 0
  except Exception as exc:  # pylint: disable=broad-exception-caught
    result = {
        "schema": RESULT_SCHEMA,
        "status": "INCONCLUSIVE",
        "verdict": "ROUND_SEAL_SALVAGE_FAILED",
        "failure": str(exc),
        "analysis_source_commit": args.analysis_source_commit,
        "contract_sha256": (
            _sha256(args.contract) if args.contract.is_file() else None),
        "claim_ceiling": "none",
        "next_gate": "repair_round_evidence_without_TPU_relaunch",
    }
    rc = 4
  _write_output(args, result)
  print(
      "[P38S22.ROUND_SALVAGE] COMPLETE "
      f"status={result['status']} verdict={result['verdict']} "
      f"output={args.output} rc={rc}"
  )
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
