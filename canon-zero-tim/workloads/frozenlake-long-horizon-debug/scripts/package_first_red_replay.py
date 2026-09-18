#!/usr/bin/env python3
"""Package the first exact M15 APC red row into a bounded return artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class PackageError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise PackageError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise PackageError(f"invalid {label}: {path}: {exc}") from exc
  _require(isinstance(value, dict), f"{label} is not a JSON object")
  return value


def package(
    *,
    capsule_path: Path,
    capture_classification_path: Path,
    m15_classification_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
  _require(not output_dir.exists(), f"refusing to overwrite output directory: {output_dir}")
  capture = _load_json(capture_classification_path, "capture classification")
  m15 = _load_json(m15_classification_path, "M15 classification")
  _require(capture.get("verdict") == "PASS", "capture classification is not PASS")
  _require(
      m15.get("status") == "FRESH_TARGET_RED_FROZEN" and m15.get("arm") == "on",
      "replay packaging requires a valid fresh APC-on target red",
  )
  capsule_sha = _sha256(capsule_path)
  _require(
      capture.get("mismatch_capsule", {}).get("sha256") == capsule_sha,
      "capture classification capsule SHA drifted",
  )
  _require(
      m15.get("artifacts", {}).get("mismatch_capsule_sha256") == capsule_sha,
      "M15 classification capsule SHA drifted",
  )
  joins = capture.get("incident_exact_joins", [])
  _require(isinstance(joins, list) and joins, "capture has no exact incident joins")
  first = min(
      joins,
      key=lambda item: (
          int(item["num_computed_tokens"]),
          int(item["source_row"]),
          int(item["completion_position"]),
      ),
  )
  source_row = int(first["source_row"])
  row_joins = sorted(
      (item for item in joins if int(item["source_row"]) == source_row),
      key=lambda item: (
          int(item["num_computed_tokens"]),
          int(item["completion_position"]),
      ),
  )

  with np.load(capsule_path, allow_pickle=False) as archive:
    arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  try:
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
  except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise PackageError("mismatch capsule metadata is invalid") from exc
  _require(
      metadata.get("schema") == "p38-frozenlake-mismatch-capsule-v1",
      "mismatch capsule schema drifted",
  )
  selected_rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  matches = np.flatnonzero(selected_rows == source_row)
  _require(matches.size == 1, f"source row {source_row} is not unique in capsule")
  capsule_index = int(matches[0])
  extracted: dict[str, np.ndarray] = {
      "selected_rows": np.ascontiguousarray(selected_rows[[capsule_index]])
  }
  for name, value in arrays.items():
    if name in ("selected_rows", "metadata_json"):
      continue
    _require(
        value.ndim > 0 and value.shape[0] == selected_rows.size,
        f"capsule array {name} is not row-aligned",
    )
    extracted[name] = np.ascontiguousarray(value[[capsule_index]])

  metadata["selected_rows"] = [source_row]
  metadata["row_identity"] = [
      item
      for item in metadata.get("row_identity", [])
      if int(item.get("source_row", -1)) == source_row
  ]
  metadata["arrays"] = {
      name: {
          "shape": list(value.shape),
          "dtype": str(value.dtype),
          "sha256": _array_sha256(value),
      }
      for name, value in extracted.items()
      if name != "selected_rows"
  }
  metadata["derived_replay_bundle"] = {
      "schema": "m15-apc-first-red-row-v1",
      "source_capsule_sha256": capsule_sha,
      "source_row": source_row,
      "first_num_computed_tokens": int(first["num_computed_tokens"]),
  }
  metadata_json = json.dumps(
      metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")
  extracted["metadata_json"] = np.frombuffer(metadata_json, dtype=np.uint8)

  output_dir.mkdir(parents=True, exist_ok=False)
  output_capsule = output_dir / "first_red_capsule.npz"
  with output_capsule.open("xb") as target:
    np.savez_compressed(target, **extracted)
    target.flush()
    os.fsync(target.fileno())
  output_capsule_sha = _sha256(output_capsule)
  contract = {
      "schema": "m15-apc-first-red-replay-contract-v1",
      "status": "FIRST_RED_ROW_FROZEN",
      "source_commit": m15.get("source_commit"),
      "source_row": source_row,
      "first_incident": first,
      "source_row_incidents": row_joins,
      "input_sha256": {
          "mismatch_capsule": capsule_sha,
          "capture_classification": _sha256(capture_classification_path),
          "m15_classification": _sha256(m15_classification_path),
      },
      "output_sha256": {"first_red_capsule": output_capsule_sha},
      "preserved": [
          "prompt_ids_and_mask",
          "completion_ids_and_valid_mask",
          "action_mask",
          "S_decode_S_prefill_T_old",
          "policy_version_and_sampling_values_when_present",
          "exact_request_call_position_physical_pages_and_generations",
          "co_batch_request_ids",
      ],
      "limitations": [
          "co_batch_request_token_payloads_are_not_present",
          "full_scheduler_interleaving_is_not_reversible_from_this_bundle",
          "this_is_a_first_red_row_carrier_not_a_mechanism_verdict",
      ],
  }
  contract_path = output_dir / "first_red_contract.json"
  contract_path.write_text(
      json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  sums_path = output_dir / "SHA256SUMS"
  sums_path.write_text(
      f"{output_capsule_sha}  {output_capsule.name}\n"
      f"{_sha256(contract_path)}  {contract_path.name}\n",
      encoding="utf-8",
  )
  return contract


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--capsule", required=True, type=Path)
  parser.add_argument("--capture-classification", required=True, type=Path)
  parser.add_argument("--m15-classification", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = package(
        capsule_path=args.capsule,
        capture_classification_path=args.capture_classification,
        m15_classification_path=args.m15_classification,
        output_dir=args.output_dir,
    )
  except (OSError, ValueError, PackageError) as exc:
    print(json.dumps({"status": "INCONCLUSIVE", "error": str(exc)}, sort_keys=True))
    raise SystemExit(2) from exc
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
