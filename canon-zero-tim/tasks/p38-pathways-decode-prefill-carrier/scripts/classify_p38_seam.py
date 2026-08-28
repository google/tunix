#!/usr/bin/env python3
"""Join P38 seam fingerprints to red action positions and name the first seam."""

from __future__ import annotations

import argparse
import hashlib
import ast
import io
import json
from pathlib import Path
import struct
from typing import Any
import zipfile



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
  if manifest.get("require_tail") is True:
    tail_inventory = manifest.get("tail_record_files")
    _require(isinstance(tail_inventory, list) and tail_inventory,
             "seam reduction selected-tail inventory is empty")
    expected_tail_paths = set()
    for item in tail_inventory:
      _require(isinstance(item, dict),
               "seam reduction selected-tail entry is invalid")
      relative = Path(str(item.get("path", "")))
      _require(
          len(relative.parts) == 2
          and relative.parts[0] == directory_name
          and relative.name.startswith("p38_tail_")
          and relative.suffix in (".json", ".npz"),
          f"seam reduction selected-tail path is invalid: {relative}",
      )
      target = (path.parent / relative).resolve()
      _require(target.parent == directory.resolve(),
               f"seam reduction selected-tail path escaped: {relative}")
      _require(target.is_file(),
               f"seam reduction selected-tail file is absent: {relative}")
      _require(_sha256(target) == item.get("sha256"),
               f"seam reduction selected-tail SHA failed: {relative}")
      _require(target.stat().st_size == int(item.get("bytes", -1)),
               f"seam reduction selected-tail size drifted: {relative}")
      expected_tail_paths.add(target.name)
    actual_tail_paths = {
        item.name for item in directory.iterdir()
        if item.is_file() and item.name.startswith("p38_tail_")
        and item.suffix in (".json", ".npz")
    }
    _require(actual_tail_paths == expected_tail_paths,
             "seam reduction selected-tail inventory drifted")
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
) -> dict[tuple[int, bytes, str, int], dict[str, Any]]:
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
      key = (
          diagnostic_round,
          bytes(hashes[row_offset]),
          arm,
          int(target_ids[row_offset]),
      )
      _require(key not in result,
               "duplicate terminal-tail token-prefix/target record")
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


def _tail_numeric_payload_sha256(
    *,
    position: int,
    source_token_id: int,
    target_id: int,
    logit_row_index: int,
    checkpoint_names: list[str],
    values: np.ndarray,
) -> str:
  digest = hashlib.sha256()
  digest.update(json.dumps({
      "position": int(position),
      "source_token_id": int(source_token_id),
      "target_id": int(target_id),
      "logit_row_index": int(logit_row_index),
      "checkpoint_names": checkpoint_names,
  }, sort_keys=True).encode("utf-8"))
  digest.update(_array_sha256(values).encode("ascii"))
  return digest.hexdigest()


def _load_manifest_selected_tail_row(
    directory: Path,
    key: tuple[int, bytes, str] | tuple[int, bytes, str, int],
    selected: dict[str, Any],
) -> dict[str, Any]:
  index = int(selected.get("record_index", -1))
  row_offset = int(selected.get("row_offset", -1))
  json_path = directory / f"p38_tail_{index:06d}.json"
  npz_path = directory / f"p38_tail_{index:06d}.npz"
  _require(json_path.is_file() and npz_path.is_file(),
           f"selected terminal-tail record is absent: {index}")
  record = json.loads(json_path.read_text(encoding="utf-8"))
  _require(
      record.get("schema") == "p38-tail-values-v1"
      and int(record.get("record_index", -1)) == index
      and int(record.get("diagnostic_round", -1)) == key[0]
      and record.get("arm") == key[2],
      f"selected terminal-tail provenance drifted: {index}",
  )
  _require(_sha256(npz_path) == record.get("npz_sha256"),
           f"selected terminal-tail NPZ SHA failed: {index}")
  with np.load(npz_path, allow_pickle=False) as archive:
    arrays = {name: np.asarray(archive[name]) for name in archive.files}
  expected_keys = {
      "row_indices", "positions", "token_ids", "request_ordinals",
      "token_prefix_sha256", "logit_row_indices", "target_ids",
      "tail_values",
  }
  _require(set(arrays) == expected_keys,
           f"selected terminal-tail array inventory drifted: {index}")
  rows = arrays["row_indices"].reshape(-1)
  positions = arrays["positions"].reshape(-1)
  source_tokens = arrays["token_ids"].reshape(-1)
  hashes = arrays["token_prefix_sha256"].reshape(-1)
  logit_rows = arrays["logit_row_indices"].reshape(-1)
  target_ids = arrays["target_ids"].reshape(-1)
  values = arrays["tail_values"]
  _require(
      0 <= row_offset < rows.size
      and rows.size == positions.size == source_tokens.size == hashes.size
      == logit_rows.size == target_ids.size == values.shape[0]
      and values.shape[1:] == (len(_TAIL_CHECKPOINTS),),
      f"selected terminal-tail row offset/geometry drifted: {index}",
  )
  _require(bytes(hashes[row_offset]) == key[1],
           f"selected terminal-tail token prefix drifted: {index}")
  if len(key) == 4:
    _require(int(target_ids[row_offset]) == key[3],
             f"selected terminal-tail target drifted: {index}")
  checkpoint_names = [str(value) for value in record.get(
      "checkpoint_names", ())]
  _require(tuple(checkpoint_names) == _TAIL_CHECKPOINTS,
           f"selected terminal-tail checkpoints drifted: {index}")
  value_row = np.asarray(values[row_offset])
  observed = {
      "record_index": index,
      "row_offset": row_offset,
      "row_index": int(rows[row_offset]),
      "position": int(positions[row_offset]),
      "source_token_id": int(source_tokens[row_offset]),
      "target_id": int(target_ids[row_offset]),
      "logit_row_index": int(logit_rows[row_offset]),
      "checkpoint_names": checkpoint_names,
      "tail_value_sha256": _array_sha256(value_row),
      "numeric_payload_sha256": _tail_numeric_payload_sha256(
          position=int(positions[row_offset]),
          source_token_id=int(source_tokens[row_offset]),
          target_id=int(target_ids[row_offset]),
          logit_row_index=int(logit_rows[row_offset]),
          checkpoint_names=checkpoint_names,
          values=value_row,
      ),
  }
  for name, value in observed.items():
    _require(selected.get(name) == value,
             f"selected terminal-tail manifest field drifted: {index}/{name}")
  return {
      "record_index": index,
      "row_index": observed["row_index"],
      "position": observed["position"],
      "source_token_id": observed["source_token_id"],
      "target_id": observed["target_id"],
      "logit_row_index": observed["logit_row_index"],
      "checkpoint_names": checkpoint_names,
      "values": value_row,
  }


def _load_v2_tail_join_records(
    directory: Path,
    manifest: dict[str, Any],
    required: set[tuple[int, bytes, str]]
    | set[tuple[int, bytes, str, int]],
) -> dict[tuple[int, bytes, str] | tuple[int, bytes, str, int],
          dict[str, Any]]:
  entries = manifest.get("tail_join_entries")
  _require(isinstance(entries, list) and entries,
           "seam reduction terminal-tail join map is absent")
  entry_map = {}
  for entry in entries:
    _require(isinstance(entry, dict),
             "seam reduction terminal-tail join entry is invalid")
    try:
      prefix = str(entry["token_prefix_sha256"]).encode("ascii")
      key = (int(entry["diagnostic_round"]), prefix, str(entry["arm"]))
      if manifest.get("tail_target_identity_required") is True:
        key = (*key, int(entry["expected_target_id"]))
    except (KeyError, UnicodeEncodeError, ValueError) as error:
      raise SeamError("seam reduction terminal-tail join key is invalid") from error
    _require(key not in entry_map,
             "seam reduction terminal-tail join key is duplicated")
    entry_map[key] = entry
  _require(set(entry_map) == required,
           "seam reduction terminal-tail join map differs from red actions")
  records = {}
  for key in sorted(required, key=lambda value: tuple(value)):
    entry = entry_map[key]
    _require(entry.get("resolution") in ("unique", "equivalent_alias"),
             "seam reduction terminal-tail join is not numerically resolved")
    selected = entry.get("selected")
    _require(isinstance(selected, dict),
             "seam reduction selected terminal-tail row is absent")
    records[key] = _load_manifest_selected_tail_row(directory, key, selected)
  _require(int(manifest.get("matched_tail_keys", -1)) == len(required),
           "seam reduction matched-tail total drifted")
  return records


def _load_npz_archive(path: Path) -> dict[str, dict[str, Any]]:
  try:
    import numpy as np  # type: ignore
    with np.load(path, allow_pickle=False) as archive:
      return {
          name: {
              "values": list(np.asarray(archive[name]).reshape(-1)),
              "shape": archive[name].shape,
              "raw": archive[name].tobytes(),
          }
          for name in archive.files
      }
  except ImportError:
    pass

  arrays: dict[str, dict[str, Any]] = {}
  with zipfile.ZipFile(path, "r") as zf:
    for name in zf.namelist():
      if not name.endswith(".npy"):
        continue
      key = name[:-4]
      raw = zf.read(name)
      bio = io.BytesIO(raw)
      magic = bio.read(6)
      _require(magic == b"\x93NUMPY", f"invalid NPY magic in {path.name}/{name}")
      major, _ = struct.unpack("BB", bio.read(2))
      if major == 1:
        hlen, = struct.unpack("<H", bio.read(2))
      else:
        hlen, = struct.unpack("<I", bio.read(4))
      hdr = ast.literal_eval(bio.read(hlen).decode("latin1").strip())
      data_bytes = bio.read()
      shape = tuple(hdr["shape"])
      descr = hdr["descr"]
      count = 1
      for s in shape:
        count *= s
      if descr in ("<i4", "<i8", "<f4", "<f8"):
        fmt = {"<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]
        values = list(struct.unpack(f"<{count}{fmt}", data_bytes))
      elif descr in ("|b1", "|u1", "|i1"):
        values = [bool(b) if descr == "|b1" else b for b in data_bytes]
      else:
        values = data_bytes
      arrays[key] = {
          "shape": shape,
          "descr": descr,
          "values": values,
          "raw": data_bytes,
      }
  return arrays


def _red_points(capsules: list[Path]) -> list[dict[str, Any]]:
  points = []
  seen_rounds = set()
  for path in capsules:
    arrays = _load_npz_archive(path)
    required = {
        "metadata_json", "selected_rows", "prompt_ids", "prompt_mask",
        "completion_ids", "completion_valid_mask", "action_mask",
        "s_decode", "s_prefill",
    }
    _require(required.issubset(arrays), f"capsule is incomplete: {path}")
    raw_meta = arrays["metadata_json"]["raw"]
    metadata = json.loads(raw_meta.decode("utf-8"))
    diagnostic_round = int(metadata.get("diagnostic_round", -1))
    _require(diagnostic_round not in seen_rounds,
             f"duplicate capsule round: {diagnostic_round}")
    seen_rounds.add(diagnostic_round)

    selected_rows = arrays["selected_rows"]["values"]
    num_selected = len(selected_rows)

    prompt_shape = arrays["prompt_ids"]["shape"]
    comp_shape = arrays["completion_ids"]["shape"]
    prompt_w = prompt_shape[1] if len(prompt_shape) > 1 else len(arrays["prompt_ids"]["values"]) // max(1, num_selected)
    comp_w = comp_shape[1] if len(comp_shape) > 1 else len(arrays["completion_ids"]["values"]) // max(1, num_selected)

    p_ids = arrays["prompt_ids"]["values"]
    p_mask = arrays["prompt_mask"]["values"]
    c_ids = arrays["completion_ids"]["values"]
    c_mask = arrays["completion_valid_mask"]["values"]
    a_mask = arrays["action_mask"]["values"]
    s_dec = arrays["s_decode"]["values"]
    s_pref = arrays["s_prefill"]["values"]
    dec_raw = arrays["s_decode"]["raw"]
    pref_raw = arrays["s_prefill"]["raw"]

    for capsule_row, source_row_raw in enumerate(selected_rows):
      p_offset = capsule_row * prompt_w
      c_offset = capsule_row * comp_w
      row_prompt = [
          int(p_ids[p_offset + j])
          for j in range(prompt_w)
          if bool(p_mask[p_offset + j])
      ]
      row_completion = [
          int(c_ids[c_offset + j])
          for j in range(comp_w)
          if bool(c_mask[c_offset + j])
      ]
      row_action = [bool(a_mask[c_offset + j]) for j in range(comp_w)]
      row_decode = [float(s_dec[c_offset + j]) for j in range(comp_w)]
      row_prefill = [float(s_pref[c_offset + j]) for j in range(comp_w)]
      row_dec_raw = dec_raw[c_offset * 4 : (c_offset + comp_w) * 4]
      row_pref_raw = pref_raw[c_offset * 4 : (c_offset + comp_w) * 4]

      tokens = row_prompt + row_completion
      for completion_position in range(comp_w):
        if not row_action[completion_position]:
          continue
        byte_diff = (
            row_dec_raw[completion_position * 4 : (completion_position + 1) * 4]
            != row_pref_raw[completion_position * 4 : (completion_position + 1) * 4]
        )
        if byte_diff:
          source_position = len(row_prompt) + completion_position - 1
          _require(source_position >= 0,
                   "red action has no causal source-token position")
          points.append({
              "diagnostic_round": diagnostic_round,
              "source_row": int(source_row_raw),
              "completion_position": int(completion_position),
              "source_position": source_position,
              "target_id": int(row_completion[completion_position]),
              "decode_logprob": row_decode[completion_position],
              "prefill_logprob": row_prefill[completion_position],
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
  required_tail = {
      (
          point["diagnostic_round"],
          point["token_prefix_sha256"],
          arm,
          int(point["target_id"]),
      )
      for point in red_points for arm in ("A", "B")
  }
  if reduction is not None and reduction.get(
      "schema") == "p38-seam-reduction-v2":
    records = _load_v2_join_records(directory, mode, reduction, required)
  else:
    records = _load_records(
        directory, mode, allow_sparse_indices=reduction is not None)
  if require_tail and reduction is not None and reduction.get(
      "schema") == "p38-seam-reduction-v2":
    _require(reduction.get("require_tail") is True,
             "seam reduction did not register terminal-tail evidence")
    tail_target_identity = (
        reduction.get("tail_target_identity_required") is True)
    tail_records = _load_v2_tail_join_records(
        directory,
        reduction,
        required_tail if tail_target_identity else required,
    )
  else:
    tail_target_identity = require_tail
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
    tail_target = int(point["target_id"])
    tail_a_key = (*base, "A", tail_target) if tail_target_identity else (
        *base, "A")
    tail_b_key = (*base, "B", tail_target) if tail_target_identity else (
        *base, "B")
    tail_a = tail_records.get(tail_a_key) if require_tail else None
    tail_b = tail_records.get(tail_b_key) if require_tail else None
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
