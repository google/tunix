#!/usr/bin/env python3
"""Derive missing P57cal6 map provenance without changing measured outcomes.

The p57cal6 recorder preserved the exact group/pair identity and every measured
rollout scalar, but its trajectory object no longer contained the originating
dataset row.  This tool accepts only that known sentinel shape, rematerializes
the signed deterministic calibration inventory, joins by the orchestrator's
registered group id, and writes a new receipt plus a self-hashing proof.  The
source receipt is never modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from examples.frozenlake import p57_workloads


_RECIPES = ("m10", "m15", "m20")
_GENERATIONS = 8
_SENTINELS = {
    "p57_index": -1,
    "grid_side": -1,
    "shortest_path": -1,
    "map_sha256": "",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_new(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")


def derive(source: Path, output: Path, proof_path: Path) -> dict[str, object]:
  receipt = json.loads(source.read_text(encoding="utf-8"))
  if receipt.get("schema") != "p57-frozenlake-stock-rollout-calibration-v2":
    raise ValueError("P57 provenance derivation requires the v2 receipt")
  if tuple(receipt.get("recipe_order", ())) != _RECIPES:
    raise ValueError("P57 calibration recipe order drifted")
  if receipt.get("generations") != _GENERATIONS:
    raise ValueError("P57 calibration generation count drifted")

  derived = copy.deepcopy(receipt)
  dataset_shas: dict[str, str] = {}
  replacements = 0
  for recipe_name in _RECIPES:
    result = derived.get("results", {}).get(recipe_name)
    if not isinstance(result, dict):
      raise ValueError(f"P57 result is missing for {recipe_name}")
    rows = p57_workloads.materialize_records(
        recipe_name, "calibration", "eval", 100
    )
    dataset_sha = p57_workloads.attest_records(
        rows,
        recipe_name,
        "calibration",
        "eval",
        expected_count=100,
    )
    dataset_shas[recipe_name] = dataset_sha
    if result.get("dataset_eval_sha256") != dataset_sha:
      raise ValueError(f"P57 {recipe_name} dataset SHA does not rematerialize")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != 100 * _GENERATIONS:
      raise ValueError(f"P57 {recipe_name} trajectory coverage drifted")
    seen_pairs: set[tuple[int, int]] = set()
    for record in records:
      if not isinstance(record, dict):
        raise ValueError(f"P57 {recipe_name} contains a non-object record")
      actual_sentinels = {key: record.get(key) for key in _SENTINELS}
      if actual_sentinels != _SENTINELS:
        raise ValueError(
            f"P57 {recipe_name} source provenance is not the known sentinel: "
            f"{actual_sentinels}"
        )
      group_id = int(record.get("group_id", -1))
      pair_index = int(record.get("pair_index", -1))
      if group_id not in range(100) or pair_index not in range(_GENERATIONS):
        raise ValueError(
            f"P57 {recipe_name} group/pair is outside the signed inventory"
        )
      pair = (group_id, pair_index)
      if pair in seen_pairs:
        raise ValueError(f"P57 {recipe_name} contains duplicate group/pair {pair}")
      seen_pairs.add(pair)
      row = rows[group_id]
      record.update({
          "p57_index": int(row["p57_index"]),
          "grid_side": int(row["size"]),
          "shortest_path": int(row["shortest_path"]),
          "map_sha256": str(row["map_sha256"]),
      })
      replacements += 1
    if len(seen_pairs) != 100 * _GENERATIONS:
      raise ValueError(f"P57 {recipe_name} group/pair coverage is incomplete")

  source_sha = _sha256(source)
  derived["provenance_derivation"] = {
      "schema": "p57-calibration-provenance-derivation-v1",
      "method": "deterministic-dataset-rematerialization-plus-group-id-join",
      "source_sha256": source_sha,
      "derived_fields": tuple(_SENTINELS),
      "records_derived": replacements,
      "dataset_eval_sha256": dataset_shas,
      "measured_fields_modified": False,
  }
  _write_new(output, derived)
  output_sha = _sha256(output)
  proof = {
      **derived["provenance_derivation"],
      "output_sha256": output_sha,
      "source_path": str(source),
      "output_path": str(output),
      "verdict": "PASS",
  }
  _write_new(proof_path, proof)
  return proof


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--proof", required=True, type=Path)
  args = parser.parse_args()
  print(json.dumps(derive(args.source, args.output, args.proof), sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
