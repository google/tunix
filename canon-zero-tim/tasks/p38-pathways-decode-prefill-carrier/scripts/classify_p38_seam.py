#!/usr/bin/env python3
"""Join P38 seam fingerprints to red action positions and name the first seam."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class SeamError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SeamError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix_sha256(values: np.ndarray) -> bytes:
  canonical = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
  return hashlib.sha256(canonical.tobytes()).hexdigest().encode()


def _load_records(directory: Path, mode: str) -> dict[tuple[int, bytes, str], dict]:
  paths = sorted(directory.glob("p38_seam_*.json"))
  _require(paths, "P38 seam observer produced no records")
  result = {}
  indices = []
  expected_keys = {
      "row_indices", "positions", "token_ids", "request_ordinals",
      "token_prefix_sha256", "layer_fingerprints",
      "final_norm_fingerprints",
  }
  for path in paths:
    record = json.loads(path.read_text())
    _require(record.get("schema") == "p38-seam-fingerprint-v1",
             f"invalid seam schema: {path.name}")
    _require(record.get("observer_mode") == mode,
             f"seam mode drifted: {path.name}")
    index = int(record.get("record_index", -1))
    indices.append(index)
    expected_name = f"p38_seam_{index:06d}.json"
    _require(index >= 0 and path.name == expected_name,
             f"seam record identity drifted: {path.name}")
    npz_path = directory / f"p38_seam_{index:06d}.npz"
    _require(npz_path.is_file(), f"seam NPZ missing: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"seam NPZ SHA failed: {npz_path.name}")
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name], copy=True)
                for name in archive.files}
    _require(set(arrays) == expected_keys,
             f"seam array inventory drifted: {npz_path.name}")
    rows = arrays["row_indices"].reshape(-1)
    positions = arrays["positions"].reshape(-1)
    hashes = arrays["token_prefix_sha256"].reshape(-1)
    layer_values = arrays["layer_fingerprints"]
    final_values = arrays["final_norm_fingerprints"]
    _require(layer_values.shape[0] == final_values.shape[0] == rows.size,
             f"seam row geometry drifted: {npz_path.name}")
    _require(positions.size == hashes.size == rows.size,
             f"seam provenance geometry drifted: {npz_path.name}")
    _require(layer_values.ndim == 4 and layer_values.shape[-1] == 8,
             f"seam layer geometry drifted: {npz_path.name}")
    _require(final_values.shape == (rows.size, 8),
             f"seam final geometry drifted: {npz_path.name}")
    checkpoint_names = list(record.get("checkpoint_names", ()))
    layer_indices = list(record.get("layer_indices", ()))
    _require(layer_values.shape[1] == len(layer_indices)
             and layer_values.shape[2] == len(checkpoint_names),
             f"seam metadata geometry drifted: {path.name}")
    arm = record.get("arm")
    diagnostic_round = int(record.get("diagnostic_round", -1))
    _require(arm in ("A", "B") and 0 <= diagnostic_round < 8,
             f"seam record provenance drifted: {path.name}")
    for row_index in range(rows.size):
      key = (diagnostic_round, bytes(hashes[row_index]), arm)
      _require(key not in result, "duplicate seam token-prefix record")
      result[key] = {
          "record_index": index,
          "row_index": int(rows[row_index]),
          "position": int(positions[row_index]),
          "layer_fingerprints": layer_values[row_index],
          "final_norm_fingerprints": final_values[row_index],
          "checkpoint_names": checkpoint_names,
          "layer_indices": [int(value) for value in layer_indices],
      }
  _require(indices == list(range(len(indices))),
           "seam record indices are not contiguous")
  return result


def _red_points(capsules: list[Path]) -> list[dict[str, Any]]:
  points = []
  seen_rounds = set()
  for path in capsules:
    with np.load(path, allow_pickle=False) as archive:
      arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "metadata_json", "selected_rows", "prompt_ids", "prompt_mask",
        "completion_ids", "completion_valid_mask", "action_mask",
        "s_decode", "s_prefill",
    }
    _require(required.issubset(arrays), f"capsule is incomplete: {path}")
    metadata = json.loads(arrays["metadata_json"].tobytes().decode())
    diagnostic_round = int(metadata.get("diagnostic_round", -1))
    _require(diagnostic_round not in seen_rounds,
             f"duplicate capsule round: {diagnostic_round}")
    seen_rounds.add(diagnostic_round)
    for capsule_row, source_row_raw in enumerate(
        arrays["selected_rows"].reshape(-1)
    ):
      prompt = arrays["prompt_ids"][capsule_row][
          np.asarray(arrays["prompt_mask"][capsule_row], dtype=np.bool_)]
      completion = arrays["completion_ids"][capsule_row][
          np.asarray(
              arrays["completion_valid_mask"][capsule_row], dtype=np.bool_)]
      action = np.asarray(arrays["action_mask"][capsule_row], dtype=np.bool_)
      decode = np.asarray(arrays["s_decode"][capsule_row])
      prefill = np.asarray(arrays["s_prefill"][capsule_row])
      _require(action.shape == decode.shape == prefill.shape,
               f"capsule row geometry drifted: {path.name}")
      byte_diff = (
          np.ascontiguousarray(decode).view(np.uint8)
          != np.ascontiguousarray(prefill).view(np.uint8)
      ).reshape(decode.size, decode.dtype.itemsize).any(axis=1).reshape(
          decode.shape)
      tokens = np.concatenate((prompt, completion)).astype(np.int32, copy=False)
      for completion_position in np.flatnonzero(action & byte_diff):
        source_position = int(prompt.size) + int(completion_position) - 1
        _require(source_position >= 0,
                 "red action has no causal source-token position")
        points.append({
            "diagnostic_round": diagnostic_round,
            "source_row": int(source_row_raw),
            "completion_position": int(completion_position),
            "source_position": source_position,
            "token_prefix_sha256": _prefix_sha256(
                tokens[:source_position + 1]),
            "capsule": path.name,
        })
  _require(points, "capsules contain no A-B-red action positions")
  return points


def _first_difference(a: dict, b: dict) -> dict | None:
  _require(a["checkpoint_names"] == b["checkpoint_names"]
           and a["layer_indices"] == b["layer_indices"],
           "A/B seam metadata differs")
  for layer_offset, layer in enumerate(a["layer_indices"]):
    for checkpoint_offset, checkpoint in enumerate(a["checkpoint_names"]):
      av = a["layer_fingerprints"][layer_offset, checkpoint_offset]
      bv = b["layer_fingerprints"][layer_offset, checkpoint_offset]
      if not np.array_equal(av, bv):
        return {
            "layer": layer,
            "checkpoint": checkpoint,
            "differing_fingerprint_fields": [
                int(value) for value in np.flatnonzero(av != bv)
            ],
        }
  if not np.array_equal(
      a["final_norm_fingerprints"], b["final_norm_fingerprints"]
  ):
    return {
        "layer": None,
        "checkpoint": "final_norm",
        "differing_fingerprint_fields": [
            int(value) for value in np.flatnonzero(
                a["final_norm_fingerprints"]
                != b["final_norm_fingerprints"])
        ],
    }
  return None


def classify(directory: Path, capsules: list[Path], mode: str) -> dict:
  records = _load_records(directory, mode)
  red_points = _red_points(capsules)
  joins = []
  for point in red_points:
    base = (point["diagnostic_round"], point["token_prefix_sha256"])
    a = records.get((*base, "A"))
    b = records.get((*base, "B"))
    _require(a is not None and b is not None,
             "not every red action joined A/B seam records")
    _require(a["position"] == b["position"] == point["source_position"],
             "joined seam source position drifted")
    first = _first_difference(a, b)
    joins.append({
        **{key: value for key, value in point.items()
           if key != "token_prefix_sha256"},
        "token_prefix_sha256": point["token_prefix_sha256"].decode(),
        "a_record_index": a["record_index"],
        "b_record_index": b["record_index"],
        "first_difference": first,
    })
  divergent = [join for join in joins if join["first_difference"] is not None]
  _require(divergent, "red actions have no divergent seam fingerprint")
  signatures = {
      (item["first_difference"]["layer"],
       item["first_difference"]["checkpoint"])
      for item in divergent
  }
  return {
      "schema": "p38-seam-classification-v1",
      "status": "PASS",
      "classification": "decode_seam_first_difference_measured",
      "observer_mode": mode,
      "red_points": len(red_points),
      "joined_red_points": len(joins),
      "divergent_red_points": len(divergent),
      "first_difference_signatures": [
          {"layer": layer, "checkpoint": checkpoint}
          for layer, checkpoint in sorted(
              signatures,
              key=lambda value: (
                  -1 if value[0] is None else value[0], value[1]),
          )
      ],
      "joins": joins,
      "claim_ceiling": (
          "Exact integer diagnostic fingerprints localize the first observed "
          "seam; they do not prove equality of every unobserved tensor byte."
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", type=Path, required=True)
  parser.add_argument("--capsule", type=Path, action="append", required=True)
  parser.add_argument("--mode", choices=("layer", "full"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(args.directory, args.capsule, args.mode)
  args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
