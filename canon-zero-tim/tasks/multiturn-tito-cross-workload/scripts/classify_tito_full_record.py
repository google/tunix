#!/usr/bin/env python3
"""Classify one P45/M15 exact-TiTO full-training record carrier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


_COLLECTION_CLASSIFIER_PATH = (
    Path(__file__).resolve().parent / "classify_tito_collection.py"
)
_COLLECTION_SPEC = importlib.util.spec_from_file_location(
    "p57_tito_collection_for_full", _COLLECTION_CLASSIFIER_PATH
)
if _COLLECTION_SPEC is None or _COLLECTION_SPEC.loader is None:
  raise RuntimeError("cannot load P57 TiTO collection classifier")
collection_classifier = importlib.util.module_from_spec(_COLLECTION_SPEC)
sys.modules[_COLLECTION_SPEC.name] = collection_classifier
_COLLECTION_SPEC.loader.exec_module(collection_classifier)

_ACTOR_SNAPSHOT_THRESHOLDS = {
    "first-any": 0.0,
    "first-ge-1": 1.0,
    "first-ge-8": 8.0,
    "first-ge-32": 32.0,
}


def _sha256_tokens(tokens: Any) -> str:
  array = np.ascontiguousarray(np.asarray(tokens, dtype="<i8"))
  if array.ndim != 1:
    raise ValueError("capsule token arrays must be one-dimensional")
  return hashlib.sha256(array.tobytes()).hexdigest()


def _json_lines(path: Path, reasons: list[str]) -> list[dict[str, Any]]:
  if not path.is_file():
    reasons.append(f"missing:{path.name}")
    return []
  rows = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      reasons.append(f"malformed:{path.name}:{line_number}")
      continue
    if not isinstance(value, dict):
      reasons.append(f"non_object:{path.name}:{line_number}")
      continue
    rows.append(value)
  return rows


def _require(condition: bool, reason: str, reasons: list[str]) -> None:
  if not condition:
    reasons.append(reason)


def _alignment_has_reds(
    record: dict[str, Any], *, prefix: str, reasons: list[str]
) -> bool:
  """Validates explicit red lists instead of trusting a verdict summary."""
  has_red = record.get("verdict") != "PASS"
  for field in ("blocking_reds", "warning_reds", "reported_reds"):
    value = record.get(field)
    if not isinstance(value, list):
      reasons.append(f"{prefix}_{field}_type")
      has_red = True
    elif value:
      has_red = True
      if record.get("verdict") == "PASS":
        reasons.append(f"{prefix}_pass_with_{field}")
  return has_red


_UPDATE_SIDECAR_ARRAYS = {
    "prompt_ids", "prompt_mask", "completion_ids", "completion_valid_mask",
    "action_mask", "s_decode", "s_prefill", "t_old", "policy_version",
    "sampling_values", "sequence_row", "trajectory_id", "group_id",
    "pair_index",
}
_ACTOR_SNAPSHOT_ROOT_RE = (
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
    r"[a-z0-9](?:[-a-z0-9]{0,62}[a-z0-9])?/attempt-(?:direct|[0-9]+)/"
    r"actor-snapshots"
)


def _sha256_array(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_update_sidecars(
    *,
    state: Path,
    recipe: str,
    expected_updates: int,
    rows_per_update: int,
    source_commit: Any,
    image_identity: Any,
    row_maps: list[dict[str, Any]],
    pre: list[dict[str, Any]],
    reasons: list[str],
) -> tuple[int, int, float]:
  paths = sorted(
      (state / "p57_tito_witness" / "update-sidecars").glob("step-*.npz")
  )
  _require(len(paths) == expected_updates, "update_sidecar_count", reasons)
  rows_by_step = {
      step: sorted(
          (row for row in row_maps if row.get("policy_step") == step),
          key=lambda row: row.get("sequence_row", -1),
      )
      for step in range(expected_updates)
  }
  pre_by_step = {
      row.get("step"): row
      for row in pre
      if type(row.get("step")) is int
  }
  _require(
      set(pre_by_step) == set(range(expected_updates)),
      "pre_alignment_steps",
      reasons,
  )
  total_bytes = 0
  total_write_seconds = 0.0
  observed_steps = set()
  for path in paths:
    match = re.fullmatch(r"step-([0-9]{6})\.npz", path.name)
    if match is None:
      reasons.append(f"update_sidecar_name:{path.name}")
      continue
    step = int(match.group(1))
    observed_steps.add(step)
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
      reasons.append(f"update_sidecar_file:{path.name}")
      continue
    try:
      with np.load(path, allow_pickle=False) as archive:
        expected_names = {"metadata_json", *_UPDATE_SIDECAR_ARRAYS}
        if set(archive.files) != expected_names:
          raise ValueError("array set")
        metadata_array = archive["metadata_json"]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
          raise ValueError("metadata encoding")
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        arrays = {name: np.asarray(archive[name]) for name in _UPDATE_SIDECAR_ARRAYS}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
      reasons.append(f"update_sidecar_unreadable:{path.name}:{error}")
      continue
    if (
        metadata.get("schema") != "canon.p57-tito-update-sidecar.v1"
        or metadata.get("workload") != recipe
        or metadata.get("step") != step
        or metadata.get("rows") != rows_per_update
        or metadata.get("dp") != 8
        or metadata.get("tp") != 8
        or metadata.get("source_commit") != source_commit
        or metadata.get("image_identity") != image_identity
    ):
      reasons.append(f"update_sidecar_identity:{path.name}")
    if any(value.dtype.hasobject for value in arrays.values()):
      reasons.append(f"update_sidecar_object_array:{path.name}")
      continue
    if any(
        value.ndim == 0 or value.shape[0] != rows_per_update
        for value in arrays.values()
    ):
      reasons.append(f"update_sidecar_row_shape:{path.name}")
    array_records = metadata.get("arrays")
    if not isinstance(array_records, dict) or set(array_records) != set(arrays):
      reasons.append(f"update_sidecar_array_manifest:{path.name}")
    else:
      for name, value in arrays.items():
        observed = array_records.get(name, {})
        if (
            observed.get("shape") != list(value.shape)
            or observed.get("dtype") != str(value.dtype)
            or observed.get("sha256") != _sha256_array(value)
        ):
          reasons.append(f"update_sidecar_array_hash:{path.name}:{name}")
    type_contract = (
        all(arrays[name].dtype.kind in "iu" for name in ("prompt_ids", "completion_ids", "policy_version", "sequence_row", "group_id", "pair_index"))
        and all(arrays[name].dtype.kind == "b" for name in ("prompt_mask", "completion_valid_mask", "action_mask"))
        and all(arrays[name].dtype.kind == "f" for name in ("s_decode", "s_prefill", "t_old", "sampling_values"))
        and arrays["trajectory_id"].dtype.kind == "S"
        and arrays["trajectory_id"].dtype.itemsize == 32
    )
    _require(type_contract, f"update_sidecar_dtypes:{path.name}", reasons)
    expected_rows = rows_by_step.get(step, [])
    try:
      observed_trajectory_ids = [
          value.decode("ascii") for value in arrays["trajectory_id"].tolist()
      ]
    except (UnicodeDecodeError, AttributeError):
      observed_trajectory_ids = []
    joins_equal = (
        len(expected_rows) == rows_per_update
        and arrays["sequence_row"].tolist() == list(range(rows_per_update))
        and observed_trajectory_ids
        == [row.get("trajectory_id") for row in expected_rows]
        and arrays["group_id"].tolist()
        == [row.get("group_id") for row in expected_rows]
        and arrays["pair_index"].tolist()
        == [row.get("pair_index") for row in expected_rows]
        and metadata.get("request_ids")
        == [row.get("request_ids") for row in expected_rows]
        and bool(np.all(arrays["policy_version"] == step))
    )
    _require(joins_equal, f"update_sidecar_row_join:{path.name}", reasons)
    pre_record = pre_by_step.get(step, {})
    receipt = pre_record.get("tito_update_sidecar", {})
    unhashed_record = dict(pre_record)
    unhashed_record.pop("tito_update_sidecar", None)
    # The snapshot request is reserved only after the sidecar has bound the
    # original alignment record. It therefore must not be folded back into
    # that earlier record digest during terminal classification.
    unhashed_record.pop("tito_actor_snapshot_request", None)
    record_payload = json.dumps(
        unhashed_record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    expected_record_sha = hashlib.sha256(record_payload).hexdigest()
    actual_file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_valid = (
        receipt.get("schema")
        == "canon.p57-tito-update-sidecar-receipt.v1"
        and receipt.get("step") == step
        and receipt.get("rows") == rows_per_update
        and isinstance(receipt.get("path"), str)
        and Path(receipt.get("path", "")).resolve() == path.resolve()
        and receipt.get("bytes") == path.stat().st_size
        and receipt.get("sha256") == actual_file_sha
        and type(receipt.get("logical_bytes")) is int
        and receipt["logical_bytes"] > 0
        and isinstance(receipt.get("write_seconds"), (int, float))
        and receipt["write_seconds"] >= 0
        and metadata.get("alignment_record_sha256") == expected_record_sha
    )
    _require(receipt_valid, f"update_sidecar_receipt:{path.name}", reasons)
    total_bytes += path.stat().st_size
    if isinstance(receipt.get("write_seconds"), (int, float)):
      total_write_seconds += float(receipt["write_seconds"])
  _require(observed_steps == set(range(expected_updates)), "update_sidecar_steps", reasons)
  return len(paths), total_bytes, total_write_seconds


def _expected_actor_snapshot_triggers(
    pre: list[dict[str, Any]],
) -> dict[int, list[str]]:
  expected: dict[int, list[str]] = {}
  reserved: set[str] = set()
  for record in sorted(pre, key=lambda row: row.get("step", -1)):
    boundary = record.get("boundaries", {}).get("S_decode_vs_S_prefill", {})
    max_abs = boundary.get("max_abs")
    red = (
        boundary.get("valid") is True
        and boundary.get("finite") is True
        and isinstance(boundary.get("differing_bytes"), int)
        and boundary["differing_bytes"] > 0
        and isinstance(max_abs, (int, float))
        and np.isfinite(max_abs)
    )
    if not red:
      continue
    categories = [
        category
        for category, threshold in _ACTOR_SNAPSHOT_THRESHOLDS.items()
        if category not in reserved and float(max_abs) >= threshold
    ]
    if categories:
      expected[int(record["step"])] = categories
      reserved.update(categories)
  return expected


def _validate_actor_snapshots(
    *,
    state: Path,
    source_commit: Any,
    image_identity: Any,
    recipe: str,
    pre: list[dict[str, Any]],
    reasons: list[str],
) -> tuple[int, int]:
  expected = _expected_actor_snapshot_triggers(pre)
  request_dir = state / "p57_tito_witness/actor-snapshot-requests"
  receipt_dir = state / "p57_tito_witness/actor-snapshot-receipts"
  request_paths = sorted(request_dir.glob("step-*.json"))
  receipt_paths = sorted(receipt_dir.glob("step-*.json"))
  expected_names = {f"step-{step:06d}.json" for step in expected}
  _require(
      {path.name for path in request_paths} == expected_names,
      "actor_snapshot_request_inventory",
      reasons,
  )
  _require(
      {path.name for path in receipt_paths} == expected_names,
      "actor_snapshot_receipt_inventory",
      reasons,
  )
  pre_by_step = {
      row.get("step"): row
      for row in pre
      if type(row.get("step")) is int
  }
  successful = 0
  for step, categories in expected.items():
    request_path = request_dir / f"step-{step:06d}.json"
    receipt_path = receipt_dir / f"step-{step:06d}.json"
    try:
      request = json.loads(request_path.read_text(encoding="utf-8"))
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      reasons.append(f"actor_snapshot_unreadable:{step}:{error}")
      continue
    if (
        request_path.is_symlink()
        or receipt_path.is_symlink()
        or request_path.stat().st_mode & 0o077
        or receipt_path.stat().st_mode & 0o077
    ):
      reasons.append(f"actor_snapshot_mode:{step}")
    pre_record = pre_by_step.get(step, {})
    request_receipt = pre_record.get("tito_actor_snapshot_request", {})
    sidecar_receipt = pre_record.get("tito_update_sidecar", {})
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    request_valid = (
        request.get("schema")
        == "canon.p57-tito-actor-snapshot-request.v1"
        and request.get("status") == "PENDING"
        and request.get("step") == step
        and request.get("policy_version") == step
        and request.get("categories") == categories
        and request.get("workload") == recipe
        and request.get("source_commit") == source_commit
        and request.get("image_identity") == image_identity
        and request.get("dp") == 8
        and request.get("tp") == 8
        and request.get("sidecar_sha256") == sidecar_receipt.get("sha256")
        and isinstance(request.get("max_abs"), (int, float))
        and np.isfinite(request["max_abs"])
        and all(
            float(request["max_abs"])
            >= _ACTOR_SNAPSHOT_THRESHOLDS[category]
            for category in categories
        )
        and request_receipt.get("schema")
        == "canon.p57-tito-actor-snapshot-request-receipt.v1"
        and request_receipt.get("path") == str(request_path)
        and request_receipt.get("sha256") == request_sha
        and request_receipt.get("bytes") == request_path.stat().st_size
        and request_receipt.get("step") == step
        and request_receipt.get("categories") == categories
        and request_receipt.get("max_abs") == request.get("max_abs")
    )
    _require(request_valid, f"actor_snapshot_request:{step}", reasons)
    inventory = receipt.get("model_inventory")
    inventory_valid = (
        isinstance(inventory, dict)
        and type(inventory.get("leaf_count")) is int
        and inventory["leaf_count"] > 0
        and type(inventory.get("logical_bytes")) is int
        and inventory["logical_bytes"] > 0
        and isinstance(inventory.get("leaves"), list)
        and len(inventory["leaves"]) == inventory["leaf_count"]
        and isinstance(inventory.get("bounded_fingerprint"), dict)
    )
    receipt_valid = (
        receipt.get("schema")
        == "canon.p57-tito-actor-snapshot-receipt.v1"
        and receipt.get("status") == "PASS"
        and receipt.get("step") == step
        and receipt.get("policy_version") == step
        and receipt.get("categories") == categories
        and receipt.get("max_abs") == request.get("max_abs")
        and receipt.get("source_commit") == source_commit
        and receipt.get("image_identity") == image_identity
        and receipt.get("workload") == recipe
        and receipt.get("dp") == 8
        and receipt.get("tp") == 8
        and receipt.get("request_path") == str(request_path)
        and receipt.get("request_sha256") == request_sha
        and isinstance(receipt.get("snapshot_root"), str)
        and re.fullmatch(
            _ACTOR_SNAPSHOT_ROOT_RE, receipt["snapshot_root"]
        ) is not None
        and isinstance(receipt.get("snapshot_root_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["snapshot_root_sha256"])
        is not None
        and receipt.get("snapshot_root_sha256")
        == hashlib.sha256(receipt["snapshot_root"].encode()).hexdigest()
        and receipt.get("latest_step") == step
        and receipt.get("optimizer_included") is False
        and receipt.get("resumable") is False
        and receipt.get("actor_train_steps_before") == step
        and receipt.get("actor_train_steps_after") == step
        and isinstance(receipt.get("save_seconds"), (int, float))
        and receipt["save_seconds"] >= 0
        and receipt.get("failure_type") is None
        and inventory_valid
    )
    _require(receipt_valid, f"actor_snapshot_receipt:{step}", reasons)
    if receipt.get("status") == "PASS" and receipt_valid:
      successful += 1
    else:
      reasons.append(f"actor_snapshot_failed:{step}")
  return len(expected), successful


def _validate_startup_receipts(
    *,
    state: Path,
    source_commit: Any,
    image_identity: Any,
    recipe: str,
    reasons: list[str],
) -> dict[str, Any]:
  """Requires the singleton controller and real Orbax admission receipts."""
  writer_path = state / "p57_tito_witness/single-writer.json"
  orbax_path = state / "p57_tito_gcs/orbax-probe.json"
  try:
    writer = json.loads(writer_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    writer = {}
    reasons.append(f"single_writer_unreadable:{error}")
  try:
    orbax = json.loads(orbax_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    orbax = {}
    reasons.append(f"orbax_probe_unreadable:{error}")
  for path, label in ((writer_path, "single_writer"), (orbax_path, "orbax_probe")):
    _require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_mode & 0o077 == 0,
        f"{label}_mode",
        reasons,
    )
  _require(
      writer.get("schema") == "canon.p57-tito-single-writer.v1"
      and writer.get("status") == "PASS"
      and writer.get("workload") == recipe
      and writer.get("source_commit") == source_commit
      and writer.get("image_identity") == image_identity
      and writer.get("dp") == 8
      and writer.get("tp") == 8
      and writer.get("writer_contract") == "one-python-controller-o-excl"
      and writer.get("neutrality_arm") is None,
      "single_writer_identity",
      reasons,
  )
  _require(
      orbax.get("schema")
      == "canon.p57-tito-orbax-admission-receipt.v1"
      and orbax.get("status") == "PASS"
      and orbax.get("workload") == recipe
      and orbax.get("source_commit") == source_commit
      and orbax.get("image_identity") == image_identity
      and orbax.get("dp") == 8
      and orbax.get("tp") == 8
      and orbax.get("saved_step") == 0
      and orbax.get("restored_step") == 0
      and orbax.get("restored_equal") is True
      and isinstance(orbax.get("probe_root_sha256"), str)
      and re.fullmatch(r"[0-9a-f]{64}", orbax["probe_root_sha256"])
      is not None
      and isinstance(orbax.get("elapsed_seconds"), (int, float))
      and orbax["elapsed_seconds"] >= 0
      and orbax.get("failure_type") is None,
      "orbax_probe_identity",
      reasons,
  )
  return {
      "single_writer": writer.get("status"),
      "orbax_probe": orbax.get("status"),
  }


def _validate_capsule(
    path: Path, reasons: list[str], *, recipe: str
) -> tuple[str | None, str | None, int | None]:
  try:
    record = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    reasons.append(f"capsule_unreadable:{path.name}:{error}")
    return None, None, None
  schema = record.get("schema")
  if schema == "canon.p57-tito-echo-diff.v1":
    witness = record.get("witness", {})
    submitted = record.get("submitted_token_ids")
    echoed = record.get("engine_echo_token_ids")
    try:
      valid = (
          witness.get("schema") == "canon.p57-tito-host-witness.v1"
          and witness.get("workload") == recipe
          and isinstance(submitted, list)
          and isinstance(echoed, list)
          and len(submitted) == witness.get("submitted_tokens")
          and len(echoed) == witness.get("engine_echo_tokens")
          and _sha256_tokens(submitted) == witness.get("submitted_sha256")
          and _sha256_tokens(echoed) == witness.get("engine_echo_sha256")
          and submitted != echoed
          and witness.get("submitted_equals_engine_echo") is False
      )
    except (TypeError, ValueError):
      valid = False
    _require(valid, f"echo_capsule_invalid:{path.name}", reasons)
    trajectory_id = witness.get("trajectory_id")
    request_id = witness.get("request_id")
    _require(
        isinstance(trajectory_id, str)
        and len(trajectory_id) == 32
        and all(character in "0123456789abcdef" for character in trajectory_id),
        f"echo_capsule_trajectory:{path.name}",
        reasons,
    )
    _require(
        isinstance(request_id, str) and bool(request_id),
        f"echo_capsule_request:{path.name}",
        reasons,
    )
    return trajectory_id, request_id, None
  if schema != "p57-token-first-diff-capsule-v1":
    reasons.append(f"capsule_schema:{path.name}")
    return None, None, None
  try:
    collection_classifier._validate_capsule(
        record, workload=recipe, path=path
    )
  except ValueError as error:
    reasons.append(f"capsule_invalid:{path.name}:{error}")
    return None, None, None
  header = record.get("header", {})
  _require(
      isinstance(header.get("trajectory_id"), str)
      and len(header["trajectory_id"]) == 32
      and type(header.get("policy_step")) is int,
      f"capsule_join_identity:{path.name}",
      reasons,
  )
  return header.get("trajectory_id"), None, header.get("policy_step")


def classify(
    *,
    state: Path,
    recipe: str,
    base_classification: Path,
    v1_classification: Path,
    _expected_updates: int = 300,
    _rows_per_update: int = 256,
) -> dict[str, Any]:
  reasons: list[str] = []
  expected_updates = _expected_updates
  rows_per_update = _rows_per_update
  _require(recipe in ("p45", "m15"), "recipe", reasons)
  summary_path = state / "p57_tito_witness/full-record-summary.json"
  row_map_path = state / "p57_tito_witness/full-row-map.jsonl"
  try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    summary = {}
    reasons.append(f"summary_unreadable:{error}")
  _require(summary.get("schema") == "canon.p57-tito-full-record.v1", "summary_schema", reasons)
  _require(summary.get("workload") == recipe, "summary_workload", reasons)
  _require(summary.get("expected_updates") == expected_updates, "summary_expected_updates", reasons)
  source_commit = summary.get("source_commit")
  image_identity = summary.get("image_identity")
  _require(
      isinstance(source_commit, str)
      and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
      "summary_source_commit",
      reasons,
  )
  _require(
      isinstance(image_identity, str) and bool(image_identity),
      "summary_image_identity",
      reasons,
  )
  _require(summary.get("dp") == 8 and summary.get("tp") == 8, "summary_mesh", reasons)
  startup_receipts = _validate_startup_receipts(
      state=state,
      source_commit=source_commit,
      image_identity=image_identity,
      recipe=recipe,
      reasons=reasons,
  )

  collection = summary.get("collection", {})
  expected_trajectories = expected_updates * rows_per_update
  numeric_fields = (
      "trajectories", "compared_trajectories",
      "unexercised_single_turn_trajectories", "equal_trajectories",
      "different_trajectories", "later_turn_comparisons",
      "engine_echo_comparisons", "engine_echo_differences",
      "capsules_reserved", "capsules_emitted", "capsules_omitted",
      "emission_failures", "backward_transactions",
      "gradient_microbatches", "optimizer_commits", "alignment_updates",
  )
  _require(
      all(type(collection.get(name)) is int and collection[name] >= 0 for name in numeric_fields),
      "collection_numeric_fields",
      reasons,
  )
  if all(name in collection for name in numeric_fields):
    _require(collection["trajectories"] == expected_trajectories, "trajectory_count", reasons)
    _require(
        collection["compared_trajectories"]
        + collection["unexercised_single_turn_trajectories"]
        == collection["trajectories"],
        "trajectory_coverage_partition",
        reasons,
    )
    _require(
        collection["equal_trajectories"] + collection["different_trajectories"]
        == collection["compared_trajectories"],
        "trajectory_verdict_partition",
        reasons,
    )
    _require(
        collection["engine_echo_comparisons"]
        == collection["trajectories"] + collection["later_turn_comparisons"],
        "engine_echo_coverage",
        reasons,
    )
    _require(
        collection["capsules_reserved"]
        == collection["capsules_emitted"] + collection["emission_failures"],
        "capsule_emission_accounting",
        reasons,
    )
    _require(
        collection["capsules_reserved"] + collection["capsules_omitted"]
        == collection["different_trajectories"],
        "capsule_difference_accounting",
        reasons,
    )
    _require(
        collection["capsules_reserved"] <= 64,
        "capsule_bound",
        reasons,
    )
    _require(
        collection["engine_echo_differences"]
        <= collection["different_trajectories"],
        "engine_echo_difference_accounting",
        reasons,
    )

  row_maps = _json_lines(row_map_path, reasons)
  _require(len(row_maps) == expected_trajectories, "row_map_count", reasons)
  ids = [row.get("trajectory_id") for row in row_maps]
  _require(len(set(ids)) == len(ids), "row_map_trajectory_ids", reasons)
  row_by_trajectory = {}
  all_request_ids = []
  by_step: dict[int, set[int]] = {}
  for row in row_maps:
    _require(
        row.get("schema") == "canon.p57-tito-row-map.v1",
        "row_map_schema",
        reasons,
    )
    step = row.get("policy_step")
    sequence_row = row.get("sequence_row")
    if type(step) is not int or type(sequence_row) is not int:
      reasons.append("row_map_identity_type")
      continue
    by_step.setdefault(step, set()).add(sequence_row)
    trajectory_id = row.get("trajectory_id")
    request_ids = row.get("request_ids")
    if (
        not isinstance(trajectory_id, str)
        or len(trajectory_id) != 32
        or not isinstance(request_ids, list)
        or not request_ids
        or any(not isinstance(value, str) or not value for value in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
      reasons.append("row_map_request_identity")
      continue
    row_by_trajectory[trajectory_id] = row
    all_request_ids.extend(request_ids)
  _require(set(by_step) == set(range(expected_updates)), "row_map_steps", reasons)
  _require(
      all(rows == set(range(rows_per_update)) for rows in by_step.values()),
      "row_map_sequence_rows",
      reasons,
  )
  _require(
      not any(
          row.get("policy_step") == 0 and bool(row.get("token_different"))
          for row in row_maps
      ),
      "first_update_token_admission",
      reasons,
  )
  if collection:
    _require(
        sum(int(row.get("later_turns", 0)) for row in row_maps)
        == collection.get("later_turn_comparisons"),
        "row_map_later_turns",
        reasons,
    )
    _require(
        sum(bool(row.get("token_different")) for row in row_maps)
        == collection.get("different_trajectories"),
        "row_map_different",
        reasons,
    )
    _require(
        len(all_request_ids) == collection.get("engine_echo_comparisons"),
        "row_map_request_coverage",
        reasons,
    )
    _require(
        len(set(all_request_ids)) == len(all_request_ids),
        "row_map_request_uniqueness",
        reasons,
    )

  updates = _json_lines(state / "updates.jsonl", reasons)
  pre = _json_lines(state / "pre_alignment.jsonl", reasons)
  alignment = _json_lines(state / "alignment.jsonl", reasons)
  _require(len(updates) == expected_updates, "update_count", reasons)
  _require(len(pre) == expected_updates, "pre_alignment_count", reasons)
  microsteps = sum(
      row.get("microsteps", 0) for row in updates
      if type(row.get("microsteps")) is int
  )
  alignment_updates = sum(
      len(row.get("alignment_hashes", [])) for row in updates
      if isinstance(row.get("alignment_hashes"), list)
  )
  _require(collection.get("backward_transactions") == len(updates), "backward_transactions", reasons)
  _require(collection.get("gradient_microbatches") == microsteps, "gradient_microbatches", reasons)
  _require(collection.get("optimizer_commits") == len(updates), "optimizer_commits", reasons)
  _require(collection.get("alignment_updates") == alignment_updates == len(alignment), "alignment_updates", reasons)
  _require(summary.get("optimizer_commits") == len(updates), "summary_optimizer_commits", reasons)
  _require(summary.get("checkpoint_writes") == 0, "checkpoint_writes", reasons)
  for prefix in ("train", "global"):
    before = summary.get(f"{prefix}_steps_before")
    after = summary.get(f"{prefix}_steps_after")
    _require(
        type(before) is int
        and type(after) is int
        and after - before == expected_updates,
        f"summary_{prefix}_step_delta",
        reasons,
    )
  checkpoint = summary.get("checkpoint_observation", {})
  _require(
      isinstance(checkpoint, dict)
      and checkpoint.get("configured_root") is None
      and checkpoint.get("latest_before") is None
      and checkpoint.get("latest_after") is None,
      "checkpoint_observation",
      reasons,
  )

  sidecar_count, sidecar_bytes, sidecar_write_seconds = (
      _validate_update_sidecars(
          state=state,
          recipe=recipe,
          expected_updates=expected_updates,
          rows_per_update=rows_per_update,
          source_commit=source_commit,
          image_identity=image_identity,
          row_maps=row_maps,
          pre=pre,
          reasons=reasons,
      )
  )
  requested_snapshots, successful_snapshots = _validate_actor_snapshots(
      state=state,
      source_commit=source_commit,
      image_identity=image_identity,
      recipe=recipe,
      pre=pre,
      reasons=reasons,
  )
  snapshot_trigger_steps = sorted(_expected_actor_snapshot_triggers(pre))

  capsules = sorted((state / "token-continuity-first-diff").glob("*.json"))
  capsule_trajectories = set()
  for capsule in capsules:
    _require(capsule.stat().st_mode & 0o077 == 0, f"capsule_mode:{capsule.name}", reasons)
    trajectory_id, request_id, policy_step = _validate_capsule(
        capsule, reasons, recipe=recipe
    )
    _require(
        trajectory_id in row_by_trajectory,
        f"capsule_trajectory_join:{capsule.name}",
        reasons,
    )
    if trajectory_id in row_by_trajectory:
      row = row_by_trajectory[trajectory_id]
      _require(
          bool(row.get("token_different")),
          f"capsule_nonred_row:{capsule.name}",
          reasons,
      )
      if request_id is not None:
        _require(
            request_id in row.get("request_ids", ()),
            f"capsule_request_join:{capsule.name}",
            reasons,
        )
      if policy_step is not None:
        _require(
            policy_step == row.get("policy_step"),
            f"capsule_step_join:{capsule.name}",
            reasons,
        )
    _require(
        trajectory_id not in capsule_trajectories,
        f"capsule_duplicate_trajectory:{capsule.name}",
        reasons,
    )
    capsule_trajectories.add(trajectory_id)
  _require(len(capsules) == collection.get("capsules_emitted"), "capsule_inventory", reasons)
  _require(collection.get("emission_failures") == 0, "capsule_emission_failure", reasons)

  try:
    base = json.loads(base_classification.read_text(encoding="utf-8"))
    v1 = json.loads(v1_classification.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    base = v1 = {}
    reasons.append(f"upstream_classification:{error}")
  _require(base.get("verdict") in ("PASS", "PASS_WITH_ALIGNMENT_WARNINGS"), "base_classification", reasons)
  _require(v1.get("verdict") == "PASS", "v1_classification", reasons)

  token_verdict = summary.get("token_verdict", "UNEXERCISED")
  expected_token_verdict = (
      "DIFFERENT" if collection.get("different_trajectories", 0)
      else "EQUAL" if collection.get("compared_trajectories", 0)
      else "UNEXERCISED"
  )
  _require(token_verdict == expected_token_verdict, "token_verdict", reasons)
  alignment_red = any(
      _alignment_has_reds(row, prefix=f"pre_alignment_{index}", reasons=reasons)
      for index, row in enumerate(pre)
  )
  alignment_red = any(
      _alignment_has_reds(row, prefix=f"alignment_{index}", reasons=reasons)
      for index, row in enumerate(alignment)
  ) or alignment_red
  zero_tim_verdict = (
      "PASS"
      if token_verdict == "EQUAL" and not alignment_red and not reasons
      else "FAIL"
  )
  evidence_verdict = "PASS" if not reasons else "FAIL"
  execution_verdict = "PASS" if not reasons else "FAIL"
  return {
      "schema": "canon.p57-tito-full-record-classification.v1",
      "verdict": execution_verdict,
      "execution_verdict": execution_verdict,
      "token_verdict": token_verdict,
      "zero_tim_verdict": zero_tim_verdict,
      "evidence_verdict": evidence_verdict,
      "claim": (
          "STRICT_ZERO_TIM" if zero_tim_verdict == "PASS"
          else "NON_ZERO_TIM_DATA_COLLECTION"
      ),
      "recipe": recipe,
      "updates": len(updates),
      "trajectories": collection.get("trajectories", 0),
      "compared_trajectories": collection.get("compared_trajectories", 0),
      "unexercised_trajectories": collection.get(
          "unexercised_single_turn_trajectories", 0
      ),
      "different_trajectories": collection.get("different_trajectories", 0),
      "capsules": len(capsules),
      "update_sidecars": sidecar_count,
      "update_sidecar_bytes": sidecar_bytes,
      "update_sidecar_write_seconds": sidecar_write_seconds,
      "actor_snapshots_requested": requested_snapshots,
      "actor_snapshots_complete": successful_snapshots,
      "actor_snapshot_trigger_steps": snapshot_trigger_steps,
      "timing_excluded_from_step": (
          snapshot_trigger_steps[0] if snapshot_trigger_steps else None
      ),
      "performance_evidence": False,
      "performance_claim": "DIAGNOSTIC_ONLY",
      "startup_receipts": startup_receipts,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--recipe", choices=("p45", "m15"), required=True)
  parser.add_argument("--base-classification", type=Path, required=True)
  parser.add_argument("--v1-classification", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite TiTO full classification: {args.output}")
  record = classify(
      state=args.state,
      recipe=args.recipe,
      base_classification=args.base_classification,
      v1_classification=args.v1_classification,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
  descriptor = os.open(
      args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
  )
  with os.fdopen(descriptor, "wb") as output:
    output.write(payload)
    output.flush()
    os.fsync(output.fileno())
  print(
      "P57_TITO_FULL_RECORD_CLASSIFICATION "
      f"execution={record['execution_verdict']} token={record['token_verdict']} "
      f"zero_tim={record['zero_tim_verdict']} evidence={record['evidence_verdict']} "
      f"claim={record['claim']}",
      flush=True,
  )
  return 0 if record["execution_verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
