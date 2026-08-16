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


_TAIL_CHECKPOINTS = (
    "raw_target_logit",
    "raw_log_normalizer",
    "processed_target_logit",
    "processed_log_normalizer",
    "observer_target_logprob",
    "production_target_logprob",
)


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SeamError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix_sha256(values: np.ndarray) -> bytes:
  canonical = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
  return hashlib.sha256(canonical.tobytes()).hexdigest().encode()


def _load_reduction_manifest(path: Path, directory: Path) -> dict[str, Any]:
  _require(path.is_file(), f"reduction manifest is absent: {path}")
  manifest = json.loads(path.read_text(encoding="utf-8"))
  schema = manifest.get("schema")
  _require(schema in ("p38-seam-reduction-v1", "p38-seam-reduction-v2"),
           "invalid seam reduction schema")
  _require(manifest.get("selection_complete") is True,
           "seam reduction did not join every red action")
  _require(not manifest.get("unmatched_keys"),
           "seam reduction contains unmatched red actions")
  _require(not manifest.get("ambiguous_keys"),
           "seam reduction contains ambiguous red actions")
  if schema == "p38-seam-reduction-v1":
    directory_name = str(manifest.get("selected_directory", "selected"))
    inventory = manifest.get("selected_files")
  else:
    directory_name = str(manifest.get("records_directory", "records"))
    inventory = manifest.get("record_files")
  expected_directory = (path.parent / directory_name).resolve()
  _require(directory.resolve() == expected_directory,
           "seam reduction selected-directory provenance drifted")
  _require(isinstance(inventory, list) and inventory,
           "seam reduction selected-file inventory is empty")
  expected_paths = set()
  for item in inventory:
    _require(isinstance(item, dict),
             "seam reduction selected-file entry is invalid")
    relative = Path(str(item.get("path", "")))
    _require(
        len(relative.parts) == 2
        and relative.parts[0] == directory_name
        and relative.name.startswith("p38_seam_")
        and relative.suffix in (".json", ".npz"),
        f"seam reduction selected path is invalid: {relative}",
    )
    target = (path.parent / relative).resolve()
    _require(target.parent == directory.resolve(),
             f"seam reduction selected path escaped its directory: {relative}")
    _require(target.is_file(),
             f"seam reduction selected file is absent: {relative}")
    _require(_sha256(target) == item.get("sha256"),
             f"seam reduction selected SHA failed: {relative}")
    _require(target.stat().st_size == int(item.get("bytes", -1)),
             f"seam reduction selected size drifted: {relative}")
    expected_paths.add(target.name)
  actual_paths = {
      item.name for item in directory.iterdir()
      if item.is_file() and item.name.startswith("p38_seam_")
      and item.suffix in (".json", ".npz")
  }
  _require(actual_paths == expected_paths,
           "seam reduction selected-file inventory drifted")
  return manifest


def _validate_reduced_capsules(
    manifest_path: Path,
    manifest: dict[str, Any],
    capsules: list[Path],
) -> None:
  expected = manifest.get("capsules")
  _require(isinstance(expected, list) and expected,
           "seam reduction capsule inventory is empty")
  expected_paths = set()
  for item in expected:
    _require(isinstance(item, dict),
             "seam reduction capsule entry is invalid")
    relative = Path(str(item.get("path", "")))
    _require(
        len(relative.parts) == 2
        and relative.parts[0] == "capsules"
        and relative.suffix == ".npz",
        f"seam reduction capsule path is invalid: {relative}",
    )
    target = (manifest_path.parent / relative).resolve()
    _require(target.is_file(),
             f"seam reduction capsule is absent: {relative}")
    _require(_sha256(target) == item.get("sha256"),
             f"seam reduction capsule SHA failed: {relative}")
    _require(target.stat().st_size == int(item.get("bytes", -1)),
             f"seam reduction capsule size drifted: {relative}")
    expected_paths.add(target)
  actual_paths = {path.resolve() for path in capsules}
  _require(actual_paths == expected_paths,
           "seam reduction classifier capsule inventory drifted")


def _load_records(
    directory: Path,
    mode: str,
    *,
    allow_sparse_indices: bool = False,
) -> dict[tuple[int, bytes, str], dict]:
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
  _require(len(indices) == len(set(indices)),
           "seam record indices are not unique")
  if not allow_sparse_indices:
    _require(indices == list(range(len(indices))),
             "seam record indices are not contiguous")
  return result


def _load_tail_records(
    directory: Path,
) -> dict[tuple[int, bytes, str], dict[str, Any]]:
  paths = sorted(directory.glob("p38_tail_*.json"))
  _require(paths, "P38 terminal-tail observer produced no records")
  result = {}
  expected_keys = {
      "row_indices", "positions", "token_ids", "request_ordinals",
      "token_prefix_sha256", "logit_row_indices", "target_ids",
      "tail_values",
  }
  for path in paths:
    record = json.loads(path.read_text(encoding="utf-8"))
    _require(record.get("schema") == "p38-tail-values-v1",
             f"invalid terminal-tail schema: {path.name}")
    index = int(record.get("record_index", -1))
    _require(index >= 0 and path.name == f"p38_tail_{index:06d}.json",
             f"terminal-tail record identity drifted: {path.name}")
    npz_path = directory / f"p38_tail_{index:06d}.npz"
    _require(npz_path.is_file(), f"terminal-tail NPZ missing: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"terminal-tail NPZ SHA failed: {npz_path.name}")
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name], copy=True)
                for name in archive.files}
    _require(set(arrays) == expected_keys,
             f"terminal-tail array inventory drifted: {npz_path.name}")
    rows = arrays["row_indices"].reshape(-1)
    positions = arrays["positions"].reshape(-1)
    source_tokens = arrays["token_ids"].reshape(-1)
    hashes = arrays["token_prefix_sha256"].reshape(-1)
    logit_rows = arrays["logit_row_indices"].reshape(-1)
    target_ids = arrays["target_ids"].reshape(-1)
    values = arrays["tail_values"]
    _require(
        rows.size == positions.size == source_tokens.size == hashes.size
        == logit_rows.size == target_ids.size == values.shape[0]
        and values.shape == (rows.size, len(_TAIL_CHECKPOINTS)),
        f"terminal-tail row geometry drifted: {npz_path.name}",
    )
    _require(tuple(record.get("checkpoint_names", ())) == _TAIL_CHECKPOINTS,
             f"terminal-tail checkpoint contract drifted: {path.name}")
    arm = record.get("arm")
    diagnostic_round = int(record.get("diagnostic_round", -1))
    _require(arm in ("A", "B") and 0 <= diagnostic_round < 8,
             f"terminal-tail provenance drifted: {path.name}")
    for row_offset in range(rows.size):
      key = (diagnostic_round, bytes(hashes[row_offset]), arm)
      _require(key not in result,
               "duplicate terminal-tail token-prefix record")
      result[key] = {
          "record_index": index,
          "row_index": int(rows[row_offset]),
          "position": int(positions[row_offset]),
          "source_token_id": int(source_tokens[row_offset]),
          "target_id": int(target_ids[row_offset]),
          "logit_row_index": int(logit_rows[row_offset]),
          "checkpoint_names": list(_TAIL_CHECKPOINTS),
          "values": values[row_offset],
      }
  return result


def _array_sha256(value: np.ndarray) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  digest = hashlib.sha256()
  digest.update(array.dtype.str.encode("ascii"))
  digest.update(json.dumps(list(array.shape)).encode("ascii"))
  digest.update(array.tobytes())
  return digest.hexdigest()


def _numeric_payload_sha256(
    *,
    position: int,
    token_id: int,
    checkpoint_names: list[str],
    layer_indices: list[int],
    layer_fingerprints: np.ndarray,
    final_norm_fingerprints: np.ndarray,
) -> str:
  digest = hashlib.sha256()
  digest.update(json.dumps({
      "position": int(position),
      "token_id": int(token_id),
      "checkpoint_names": checkpoint_names,
      "layer_indices": layer_indices,
  }, sort_keys=True).encode("utf-8"))
  digest.update(_array_sha256(layer_fingerprints).encode("ascii"))
  digest.update(_array_sha256(final_norm_fingerprints).encode("ascii"))
  return digest.hexdigest()


def _load_manifest_selected_row(
    directory: Path,
    mode: str,
    key: tuple[int, bytes, str],
    selected: dict[str, Any],
) -> dict[str, Any]:
  index = int(selected.get("record_index", -1))
  row_offset = int(selected.get("row_offset", -1))
  json_path = directory / f"p38_seam_{index:06d}.json"
  npz_path = directory / f"p38_seam_{index:06d}.npz"
  _require(json_path.is_file() and npz_path.is_file(),
           f"selected seam record is absent: {index}")
  record = json.loads(json_path.read_text(encoding="utf-8"))
  _require(
      record.get("schema") == "p38-seam-fingerprint-v1"
      and record.get("observer_mode") == mode
      and int(record.get("record_index", -1)) == index
      and int(record.get("diagnostic_round", -1)) == key[0]
      and record.get("arm") == key[2],
      f"selected seam record provenance drifted: {index}",
  )
  _require(_sha256(npz_path) == record.get("npz_sha256"),
           f"selected seam NPZ SHA failed: {index}")
  with np.load(npz_path, allow_pickle=False) as archive:
    arrays = {name: np.asarray(archive[name]) for name in archive.files}
  expected_keys = {
      "row_indices", "positions", "token_ids", "request_ordinals",
      "token_prefix_sha256", "layer_fingerprints",
      "final_norm_fingerprints",
  }
  _require(set(arrays) == expected_keys,
           f"selected seam array inventory drifted: {index}")
  rows = arrays["row_indices"].reshape(-1)
  positions = arrays["positions"].reshape(-1)
  token_ids = arrays["token_ids"].reshape(-1)
  hashes = arrays["token_prefix_sha256"].reshape(-1)
  layer_values = arrays["layer_fingerprints"]
  final_values = arrays["final_norm_fingerprints"]
  _require(
      0 <= row_offset < rows.size
      and rows.size == positions.size == token_ids.size == hashes.size
      == layer_values.shape[0] == final_values.shape[0],
      f"selected seam row offset/geometry drifted: {index}",
  )
  _require(bytes(hashes[row_offset]) == key[1],
           f"selected seam token prefix drifted: {index}")
  checkpoint_names = [str(value) for value in record.get(
      "checkpoint_names", ())]
  layer_indices = [int(value) for value in record.get("layer_indices", ())]
  layer_row = np.asarray(layer_values[row_offset])
  final_row = np.asarray(final_values[row_offset])
  observed = {
      "record_index": index,
      "row_offset": row_offset,
      "row_index": int(rows[row_offset]),
      "position": int(positions[row_offset]),
      "token_id": int(token_ids[row_offset]),
      "checkpoint_names": checkpoint_names,
      "layer_indices": layer_indices,
      "layer_fingerprint_sha256": _array_sha256(layer_row),
      "final_norm_fingerprint_sha256": _array_sha256(final_row),
      "numeric_payload_sha256": _numeric_payload_sha256(
          position=int(positions[row_offset]),
          token_id=int(token_ids[row_offset]),
          checkpoint_names=checkpoint_names,
          layer_indices=layer_indices,
          layer_fingerprints=layer_row,
          final_norm_fingerprints=final_row,
      ),
  }
  for name, value in observed.items():
    _require(selected.get(name) == value,
             f"selected seam manifest field drifted: {index}/{name}")
  return {
      "record_index": index,
      "row_index": observed["row_index"],
      "position": observed["position"],
      "layer_fingerprints": layer_row,
      "final_norm_fingerprints": final_row,
      "checkpoint_names": checkpoint_names,
      "layer_indices": layer_indices,
  }


def _load_v2_join_records(
    directory: Path,
    mode: str,
    manifest: dict[str, Any],
    required: set[tuple[int, bytes, str]],
) -> dict[tuple[int, bytes, str], dict[str, Any]]:
  entries = manifest.get("join_entries")
  _require(isinstance(entries, list) and entries,
           "seam reduction join map is absent")
  entry_map = {}
  for entry in entries:
    _require(isinstance(entry, dict), "seam reduction join entry is invalid")
    try:
      prefix = str(entry["token_prefix_sha256"]).encode("ascii")
      key = (int(entry["diagnostic_round"]), prefix, str(entry["arm"]))
    except (KeyError, UnicodeEncodeError, ValueError) as error:
      raise SeamError("seam reduction join key is invalid") from error
    _require(key not in entry_map, "seam reduction join key is duplicated")
    entry_map[key] = entry
  _require(set(entry_map) == required,
           "seam reduction join map differs from capsule red actions")
  records = {}
  for key in sorted(required, key=lambda value: (value[0], value[1], value[2])):
    entry = entry_map[key]
    _require(entry.get("resolution") in ("unique", "equivalent_alias"),
             "seam reduction join is not numerically resolved")
    selected = entry.get("selected")
    _require(isinstance(selected, dict),
             "seam reduction selected row is absent")
    records[key] = _load_manifest_selected_row(
        directory, mode, key, selected)
  _require(int(manifest.get("matched_arm_keys", -1)) == len(required),
           "seam reduction matched-key total drifted")
  return records


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
            "target_id": int(completion[completion_position]),
            "decode_logprob": float(decode[completion_position]),
            "prefill_logprob": float(prefill[completion_position]),
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


def _tail_first_difference(a: dict, b: dict) -> dict | None:
  _require(a["checkpoint_names"] == b["checkpoint_names"],
           "A/B terminal-tail checkpoint metadata differs")
  _require(a["position"] == b["position"],
           "A/B terminal-tail positions differ")
  _require(a["target_id"] == b["target_id"],
           "A/B terminal-tail target IDs differ")
  for offset, checkpoint in enumerate(a["checkpoint_names"]):
    av = np.asarray(a["values"][offset])
    bv = np.asarray(b["values"][offset])
    if not np.array_equal(av, bv):
      av_scalar = av.item()
      bv_scalar = bv.item()
      return {
          "layer": None,
          "checkpoint": checkpoint,
          "a_value": float(av_scalar),
          "b_value": float(bv_scalar),
          "max_abs": float(abs(float(av_scalar) - float(bv_scalar))),
      }
  return None


def classify(
    directory: Path,
    capsules: list[Path],
    mode: str,
    reduction_manifest: Path | None = None,
    require_tail: bool = False,
) -> dict:
  reduction = None
  if reduction_manifest is not None:
    reduction = _load_reduction_manifest(reduction_manifest, directory)
    _validate_reduced_capsules(reduction_manifest, reduction, capsules)
  red_points = _red_points(capsules)
  required = {
      (point["diagnostic_round"], point["token_prefix_sha256"], arm)
      for point in red_points for arm in ("A", "B")
  }
  if reduction is not None and reduction.get(
      "schema") == "p38-seam-reduction-v2":
    records = _load_v2_join_records(directory, mode, reduction, required)
  else:
    records = _load_records(
        directory, mode, allow_sparse_indices=reduction is not None)
  tail_records = _load_tail_records(directory) if require_tail else {}
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
    tail_a = tail_records.get((*base, "A")) if require_tail else None
    tail_b = tail_records.get((*base, "B")) if require_tail else None
    if require_tail:
      _require(tail_a is not None and tail_b is not None,
               "not every red action joined A/B terminal-tail records")
      _require(
          tail_a["position"] == tail_b["position"] == point["source_position"],
          "joined terminal-tail source position drifted",
      )
      _require(tail_a["target_id"] == tail_b["target_id"],
               "joined terminal-tail target ID drifted")
      _require(tail_a["target_id"] == point["target_id"],
               "terminal-tail target ID differs from the mismatch capsule")
      _require(
          float(tail_a["values"][-1]) == point["decode_logprob"]
          and float(tail_b["values"][-1]) == point["prefill_logprob"],
          "terminal-tail production logprob differs from the mismatch capsule",
      )
      if first is None:
        first = _tail_first_difference(tail_a, tail_b)
      _require(first is not None,
               "red action stayed exact through the production logprob tail")
    joins.append({
        **{key: value for key, value in point.items()
           if key != "token_prefix_sha256"},
        "token_prefix_sha256": point["token_prefix_sha256"].decode(),
        "a_record_index": a["record_index"],
        "b_record_index": b["record_index"],
        "a_tail_record_index": (
            tail_a["record_index"] if tail_a is not None else None
        ),
        "b_tail_record_index": (
            tail_b["record_index"] if tail_b is not None else None
        ),
        "first_difference": first,
    })
  divergent = [join for join in joins if join["first_difference"] is not None]
  signatures = {
      (item["first_difference"]["layer"],
       item["first_difference"]["checkpoint"])
      for item in divergent
  }
  ordered_signatures = sorted(
      signatures,
      key=lambda value: (
          10**9 if value[0] is None else value[0], value[1]),
  )
  numeric_layers = sorted({
      int(layer) for layer, _ in signatures if layer is not None
  })
  tail_required = not divergent
  report = {
      "schema": "p38-seam-classification-v1",
      "status": "PASS",
      "classification": (
          "hidden_chain_exact_tail_localization_required"
          if tail_required
          else "decode_terminal_first_difference_measured"
          if require_tail
          else "decode_seam_first_difference_measured"
      ),
      "observer_mode": mode,
      "red_points": len(red_points),
      "joined_red_points": len(joins),
      "divergent_red_points": len(divergent),
      "all_observed_fingerprints_equal": tail_required,
      "tail_localization_required": tail_required,
      "tail_observer_required_and_joined": require_tail,
      "mixed_first_difference_signatures": len(signatures) > 1,
      "selected_layer": numeric_layers[0] if numeric_layers else None,
      "first_difference_signatures": [
          {"layer": layer, "checkpoint": checkpoint}
          for layer, checkpoint in ordered_signatures
      ],
      "joins": joins,
      "claim_ceiling": (
          "Exact integer diagnostic fingerprints localize the first observed "
          "seam; equality through the last observed checkpoint requires a "
          "bounded tail observer and does not prove equality of every "
          "unobserved tensor byte."
      ),
  }
  if reduction_manifest is not None:
    report["reduction_provenance"] = {
        "manifest": reduction_manifest.name,
        "manifest_sha256": _sha256(reduction_manifest),
        "source_gcs_uri": reduction.get("source_gcs_uri"),
        "source_snapshot_manifest_sha256": reduction.get(
            "source_snapshot_manifest_sha256"),
    }
  return report


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", type=Path, required=True)
  parser.add_argument("--capsule", type=Path, action="append", required=True)
  parser.add_argument("--mode", choices=("layer", "full"), required=True)
  parser.add_argument("--reduction-manifest", type=Path)
  parser.add_argument("--require-tail", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      args.directory,
      args.capsule,
      args.mode,
      reduction_manifest=args.reduction_manifest,
      require_tail=args.require_tail,
  )
  args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
