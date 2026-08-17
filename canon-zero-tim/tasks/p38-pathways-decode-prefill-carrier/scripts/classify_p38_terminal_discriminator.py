#!/usr/bin/env python3
"""Classify paired P38 terminal-discriminator rows without GCS access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "p38-terminal-discriminator-v1"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
  return bytes(np.asarray(value).item()).rstrip(b"\x00")


def _row_key(metadata: dict, arrays: dict[str, np.ndarray], row: int) -> tuple:
  return (
      int(metadata["diagnostic_round"]),
      _bytes(arrays["token_prefix_sha256"][row]),
      int(arrays["positions"][row]),
      int(arrays["token_ids"][row]),
      int(arrays["target_ids"][row]),
  )


def _row_payload(arrays: dict[str, np.ndarray], row: int) -> dict[str, np.ndarray]:
  return {
      name: np.asarray(value[row]).copy()
      for name, value in arrays.items()
      if name not in {
          "row_indices", "positions", "token_ids", "request_ordinals",
          "token_prefix_sha256", "logit_row_indices", "target_ids",
      }
  }


def _payload_digest(payload: dict[str, np.ndarray]) -> str:
  digest = hashlib.sha256()
  for name in sorted(payload):
    value = np.ascontiguousarray(payload[name])
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
  return digest.hexdigest()


def load_rows(root: Path) -> tuple[dict[str, dict[tuple, dict]], int]:
  by_arm: dict[str, dict[tuple, dict]] = {"A": {}, "B": {}}
  record_count = 0
  for json_path in sorted(root.glob("p38_terminal_*.json")):
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
      raise ValueError(f"unexpected terminal schema: {json_path}")
    if metadata.get("reduction_program") != "shared-fixed-four-row-v1":
      raise ValueError(f"terminal reduction program drifted: {json_path}")
    arm = metadata.get("arm")
    if arm not in by_arm:
      raise ValueError(f"unexpected terminal arm: {arm!r}")
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
      raise ValueError(f"terminal NPZ is missing: {npz_path}")
    if _sha256(npz_path) != metadata.get("npz_sha256"):
      raise ValueError(f"terminal NPZ SHA mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: archive[name] for name in archive.files}
    if set(arrays) != set(metadata.get("array_keys", ())):
      raise ValueError(f"terminal array inventory drifted: {npz_path}")
    rows = int(arrays["positions"].size)
    for name, value in arrays.items():
      if value.shape[0] != rows:
        raise ValueError(f"terminal row count drifted: {npz_path}:{name}")
    for row in range(rows):
      key = _row_key(metadata, arrays, row)
      payload = _row_payload(arrays, row)
      prior = by_arm[arm].get(key)
      if prior is not None:
        if _payload_digest(prior) != _payload_digest(payload):
          raise ValueError(
              f"conflicting duplicate terminal row for arm={arm} key={key}")
        continue
      by_arm[arm][key] = payload
    record_count += 1
  if record_count == 0:
    raise ValueError("P38 terminal discriminator produced no records")
  return by_arm, record_count


def _different(left: np.ndarray, right: np.ndarray) -> bool:
  return not np.array_equal(left, right)


def _capsule_red_points(capsules: list[Path]) -> list[dict[str, Any]]:
  seam_path = Path(__file__).with_name("classify_p38_seam.py")
  spec = importlib.util.spec_from_file_location(
      "p38_terminal_seam_contract", seam_path)
  if spec is None or spec.loader is None:
    raise ValueError("cannot load the P38 seam capsule contract")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module._red_points(capsules)


def _required_keys(
    by_arm: dict[str, dict[tuple, dict]],
    capsules: list[Path],
) -> list[tuple]:
  points = _capsule_red_points(capsules)
  keys_a = set(by_arm["A"])
  keys_b = set(by_arm["B"])
  shared = keys_a & keys_b
  required = []
  for point in points:
    base = (
        int(point["diagnostic_round"]),
        bytes(point["token_prefix_sha256"]),
        int(point["source_position"]),
        int(point["target_id"]),
    )
    matches = [
        key for key in shared
        if (key[0], key[1], key[2], key[4]) == base
    ]
    if len(matches) != 1:
      raise ValueError(
          "P38 terminal red-point join is missing or ambiguous: "
          f"round={base[0]} position={base[2]} target={base[3]} "
          f"matches={len(matches)}")
    required.append(matches[0])
  if len(set(required)) != len(required):
    raise ValueError("P38 terminal capsule red-point identity is duplicated")
  return sorted(required)


def _first_stage(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> str:
  if _different(a["final_hidden_rows"], b["final_hidden_rows"]):
    return "pre_lm_head_hidden"
  if (_different(a["raw_logit_signatures"], b["raw_logit_signatures"])
      or _different(a["raw_block_max"], b["raw_block_max"])
      or _different(a["raw_row_max"], b["raw_row_max"])):
    return "lm_head_logits"
  if (_different(a["raw_block_exp_sum"], b["raw_block_exp_sum"])
      or _different(
          a["raw_block_observer_log_normalizer"],
          b["raw_block_observer_log_normalizer"],
      )):
    return "vocab_block_reduction"
  if (_different(
          a["processed_logit_signatures"],
          b["processed_logit_signatures"])
      or _different(a["processed_block_max"], b["processed_block_max"])
      or _different(a["processed_row_max"], b["processed_row_max"])):
    return "logits_processing"
  if (_different(
          a["processed_block_exp_sum"], b["processed_block_exp_sum"])
      or _different(
          a["processed_block_observer_log_normalizer"],
          b["processed_block_observer_log_normalizer"],
      )):
    return "processed_vocab_block_reduction"
  if _different(a["tail_values"][-1:], b["tail_values"][-1:]):
    return "production_tail_only"
  return "exact"


def _legacy_tail_fields(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[int]:
  return [
      int(value) for value in np.flatnonzero(a["tail_values"][:-1] != b["tail_values"][:-1])
  ]


def classify(
    root: Path,
    capsules: list[Path] | None = None,
    require_red_join: bool = False,
) -> dict:
  by_arm, record_count = load_rows(root)
  keys_a = set(by_arm["A"])
  keys_b = set(by_arm["B"])
  all_shared = sorted(keys_a & keys_b)
  if require_red_join and not capsules:
    raise ValueError("P38 terminal red-point join requires a capsule")
  shared = (
      _required_keys(by_arm, list(capsules or ()))
      if capsules else all_shared
  )
  if not shared:
    raise ValueError("P38 terminal discriminator has no target-aware A/B joins")
  stages: dict[str, int] = {}
  legacy_tail_drift_rows = []
  red_rows = []
  for key in shared:
    a = by_arm["A"][key]
    b = by_arm["B"][key]
    stage = _first_stage(a, b)
    legacy_fields = _legacy_tail_fields(a, b)
    if legacy_fields:
      legacy_tail_drift_rows.append({
          "diagnostic_round": key[0],
          "position": key[2],
          "target_id": key[4],
          "differing_checkpoint_indices": legacy_fields,
      })
    stages[stage] = stages.get(stage, 0) + 1
    if stage != "exact":
      red_rows.append({
          "diagnostic_round": key[0],
          "position": key[2],
          "source_token_id": key[3],
          "target_id": key[4],
          "first_differing_stage": stage,
      })
  red_stages = sorted(stage for stage in stages if stage != "exact")
  if not red_stages:
    classification = "terminal_rows_exact"
  elif len(red_stages) == 1:
    classification = red_stages[0]
  else:
    classification = "mixed_terminal_carrier"
  return {
      "classification": classification,
      "claim_ceiling": (
          "exact_final_hidden_rows_plus_diagnostic_multiword_logit_"
          "fingerprints_not_full_logits_byte_equality"
      ),
      "records": record_count,
      "rows_a": len(keys_a),
      "rows_b": len(keys_b),
      "joined_rows": len(shared),
      "joined_red_points": len(shared) if capsules else None,
      "missing_a_rows": len(keys_b - keys_a),
      "missing_b_rows": len(keys_a - keys_b),
      "legacy_shape_dependent_tail_drift_rows": legacy_tail_drift_rows,
      "stage_counts": stages,
      "red_rows": red_rows,
      "selection_scope": "capsule_red_points" if capsules else "all_shared_rows",
      "schema": "p38-terminal-discriminator-classification-v1",
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--capsule", action="append", default=[], type=Path)
  parser.add_argument("--require-red-join", action="store_true")
  args = parser.parse_args()
  result = classify(
      args.input,
      capsules=args.capsule,
      require_red_join=args.require_red_join,
  )
  encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
  if args.output:
    args.output.write_text(encoded, encoding="utf-8")
  print(encoded, end="")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
