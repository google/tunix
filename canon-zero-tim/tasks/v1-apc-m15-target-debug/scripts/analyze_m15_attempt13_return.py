#!/usr/bin/env python3
"""Fail-closed analysis of the small Attempt-13 three-round GCS return."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class Attempt13ReturnError(RuntimeError):
  pass


SOURCE_COMMIT = "7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
ROUNDS = 3
_MANIFEST_ROW = re.compile(r"([0-9a-f]{64})  ([^/]+)")
_RED_CLASSIFICATION = "M15_INTERNAL_FIRST_RED_LOCALIZED"


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise Attempt13ReturnError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  _require(isinstance(value, dict), f"JSON is not an object: {path.name}")
  return value


def _verify_manifest(directory: Path) -> dict[str, str]:
  manifest_path = directory / "SHA256SUMS"
  _require(manifest_path.is_file(), "small return lacks SHA256SUMS")
  rows: dict[str, str] = {}
  for line in manifest_path.read_text(encoding="ascii").splitlines():
    match = _MANIFEST_ROW.fullmatch(line)
    _require(match is not None, f"invalid SHA256SUMS row: {line!r}")
    digest, name = match.groups()
    _require(name not in rows, f"duplicate SHA256SUMS member: {name}")
    path = directory / name
    _require(path.is_file() and _sha256(path) == digest,
             f"small-return SHA failed: {name}")
    rows[name] = digest
  expected = {
      path.name for path in directory.iterdir()
      if path.is_file() and path.name not in {"SHA256SUMS", "ATTEMPT13_ANALYSIS.json"}
  }
  _require(set(rows) == expected,
           f"small-return manifest membership drifted: {sorted(set(rows) ^ expected)}")
  return rows


def _classifier(
    directory: Path, *, arm: str, round_index: int
) -> tuple[dict[str, Any], tuple[Any, ...] | None]:
  path = directory / f"{arm}.round-{round_index:06d}.classification.json"
  value = _json(path)
  alignment = value.get("alignment")
  _require(
      value.get("schema") == "m15-apc-wide-seam-classification-v1"
      and value.get("status") == "PASS"
      and value.get("arm") == arm
      and int(value.get("diagnostic_round", -1)) == round_index
      and isinstance(alignment, dict)
      and int(alignment.get("b_c_differing_bytes", -1)) == 0,
      f"{arm} round {round_index} official classifier contract is incomplete",
  )
  classification = str(value.get("classification", ""))
  ab_bytes = int(alignment.get("a_b_differing_bytes", -1))
  if arm == "off":
    _require(
        classification == "M15_OBSERVER_CONTROL_EXACT" and ab_bytes == 0,
        f"off round {round_index} is not an exact control",
    )
    return value, None

  if classification == "M15_OBSERVER_TREATMENT_EXACT":
    _require(ab_bytes == 0, f"on round {round_index} exact result has A-B bytes")
    return value, ("exact",)

  _require(
      classification == _RED_CLASSIFICATION and ab_bytes > 0,
      f"on round {round_index} has an unregistered red classification",
  )
  anchors = value.get("anchors")
  signatures = value.get("first_difference_signatures")
  receipts = value.get("replay_ledger_receipts")
  first = value.get("first_red_boundary")
  last = value.get("last_exact_boundary")
  source_interval = value.get("source_interval")
  _require(
      value.get("observer_mode") == "full"
      and int(value.get("expected_layer", -1)) == 0
      and isinstance(value.get("mixed_first_difference_signatures"), bool)
      and isinstance(anchors, list) and anchors
      and isinstance(signatures, list) and signatures
      and isinstance(receipts, list) and receipts
      and len(receipts) == len(anchors)
      and isinstance(first, dict)
      and isinstance(last, dict)
      and isinstance(source_interval, dict),
      f"on round {round_index} lacks official full-observer provenance fields",
  )
  _require(
      int(first.get("layer", -1)) == 0
      and int(last.get("layer", -1)) == 0,
      f"on round {round_index} first-red interval is not Layer 0",
  )
  signature = (
      "red",
      str(last.get("checkpoint", "")),
      str(first.get("checkpoint", "")),
      json.dumps(signatures, sort_keys=True, separators=(",", ":")),
  )
  return value, signature


def analyze(directory: Path) -> dict[str, Any]:
  _require(directory.is_dir(), f"small return is absent: {directory}")
  rows = _verify_manifest(directory)
  summary = _json(directory / "MULTIROUND_SUMMARY.json")
  _require(
      summary.get("schema") == "m15-apc-multiround-small-return-v1"
      and summary.get("source_commit") == SOURCE_COMMIT
      and int(summary.get("expected_rounds_per_arm", -1)) == ROUNDS,
      "Attempt-13 multiround summary identity drifted",
  )
  status = str(summary.get("status", ""))
  _require(status in {
      "COMPLETE",
      "ROUNDS_RECOVERED_ROOT_INCOMPLETE",
      "PARTIAL_ROUNDS_RECOVERED",
      "NO_DURABLE_ROUND",
  }, f"Attempt-13 return status is invalid: {status}")

  arms = summary.get("arms")
  _require(isinstance(arms, dict) and set(arms) == {"off", "on"},
           "Attempt-13 arm summary drifted")
  classifiers: list[dict[str, Any]] = []
  on_signatures: list[tuple[Any, ...]] = []
  for arm in ("off", "on"):
    arm_value = arms[arm]
    rounds = arm_value.get("rounds")
    _require(isinstance(rounds, list) and len(rounds) == ROUNDS,
             f"{arm} round inventory drifted")
    sealed = 0
    for round_index, round_value in enumerate(rounds):
      _require(int(round_value.get("diagnostic_round", -1)) == round_index,
               f"{arm} round ordering drifted")
      filename = f"{arm}.round-{round_index:06d}.classification.json"
      if round_value.get("status") == "ABSENT":
        _require(filename not in rows,
                 f"{arm} absent round unexpectedly has a classifier")
        continue
      _require(round_value.get("status") == "SEALED" and filename in rows,
               f"{arm} round {round_index} is neither sealed nor absent")
      classifier, signature = _classifier(
          directory, arm=arm, round_index=round_index
      )
      classifiers.append({
          "arm": arm,
          "diagnostic_round": round_index,
          "sha256": rows[filename],
          "classification": classifier["classification"],
          "a_b_differing_bytes": int(
              classifier["alignment"]["a_b_differing_bytes"]
          ),
          "b_c_differing_bytes": 0,
      })
      if arm == "on" and signature is not None:
        on_signatures.append(signature)
      sealed += 1
    _require(sealed == int(arm_value.get("sealed_rounds", -1)),
             f"{arm} sealed-round count drifted")

  off_sealed = int(arms["off"].get("sealed_rounds", -1))
  on_sealed = int(arms["on"].get("sealed_rounds", -1))
  stable = bool(on_signatures) and len(set(on_signatures)) == 1
  all_on_red = (
      len(on_signatures) == ROUNDS
      and all(signature[0] == "red" for signature in on_signatures)
  )
  all_on_exact = (
      len(on_signatures) == ROUNDS
      and all(signature == ("exact",) for signature in on_signatures)
  )
  if off_sealed == on_sealed == ROUNDS:
    if all_on_red and stable:
      decision = "THREE_ROUND_ATTENTION_INTERVAL_REPEAT_READY"
      next_action = (
          "replay the official classifier from each compact bundle on the "
          "bucket executor, then instrument RPA sub-boundaries"
      )
    elif all_on_exact:
      decision = "THREE_ROUND_TREATMENT_EXACT"
      next_action = "record the exact repeat; do not claim an APC repair"
    else:
      decision = "THREE_ROUND_SIGNATURE_UNSTABLE"
      next_action = "preserve all rounds and explain the mixed first-red signatures"
  elif off_sealed or on_sealed:
    decision = "PARTIAL_EVIDENCE_ONLY"
    next_action = "use recovered rounds without claiming paired target completion"
  else:
    decision = "NO_DURABLE_ROUND"
    next_action = "repair worker/upload durability before another target launch"

  return {
      "schema": "m15-attempt13-return-analysis-v1",
      "status": "PASS",
      "source_commit": SOURCE_COMMIT,
      "multiround_status": status,
      "decision": decision,
      "off_sealed_rounds": off_sealed,
      "on_sealed_rounds": on_sealed,
      "on_first_red_signature_stable": stable,
      "classifiers": classifiers,
      "official_classifier_replay": "NOT_PERFORMED_FROM_SMALL_RETURN",
      "claim_ceiling": (
          "The small return can establish repeated checkpoint-fingerprint "
          "intervals only. It cannot prove an RPA block-table or KV-read cause."
      ),
      "numerical_repair_authorized": False,
      "next_action": next_action,
  }


def _write_analysis(directory: Path, result: dict[str, Any]) -> Path:
  output = directory / "ATTEMPT13_ANALYSIS.json"
  _require(not output.exists(), f"refusing to overwrite analysis: {output}")
  partial = output.with_suffix(".json.partial")
  partial.write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  partial.replace(output)
  names = sorted(
      path.name for path in directory.iterdir()
      if path.is_file() and path.name != "SHA256SUMS"
  )
  manifest = directory / "SHA256SUMS"
  manifest_partial = directory / "SHA256SUMS.partial"
  manifest_partial.write_text(
      "".join(f"{_sha256(directory / name)}  {name}\n" for name in names),
      encoding="ascii",
  )
  manifest_partial.replace(manifest)
  return output


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--return-dir", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = analyze(args.return_dir)
    output = _write_analysis(args.return_dir, result)
  except (OSError, ValueError, json.JSONDecodeError, Attempt13ReturnError) as exc:
    raise SystemExit(f"M15_ATTEMPT13_ANALYSIS_RED {exc}") from exc
  print(
      "M15_ATTEMPT13_ANALYSIS_COMPLETE "
      f"decision={result['decision']} "
      f"off_rounds={result['off_sealed_rounds']} "
      f"on_rounds={result['on_sealed_rounds']} "
      f"output={output}"
  )


if __name__ == "__main__":
  main()
