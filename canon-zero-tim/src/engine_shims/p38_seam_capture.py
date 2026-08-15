"""Host-side contracts for bounded P38 per-token seam evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


P38_LAYER_CHECKPOINTS = ("layer_input", "layer_output")


P38_SEAM_CHECKPOINTS = (
    "layer_input",
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "q_post_rope",
    "k_post_rope",
    "rpa_output",
    "o_proj",
    "attention_residual",
    "post_attention_norm",
    "mlp_output",
    "layer_output",
)


def _prefix_sha256(values: np.ndarray) -> str:
  canonical = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
  return hashlib.sha256(canonical.tobytes()).hexdigest()


def select_standard_seam_rows(
    *,
    req_ids_dp: Mapping[int, Sequence[str]],
    scheduled_tokens: Mapping[str, int],
    req_id_to_index: Mapping[str, int],
    num_computed_tokens: Sequence[int],
    num_tokens: Sequence[int],
    token_ids_cpu: np.ndarray,
    padded_rows_per_dp: int,
    total_rows: int,
    min_position: int,
    max_position: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
  """Map real standard-path packed rows without admitting padding rows."""
  if not (0 <= int(min_position) < int(max_position)):
    raise ValueError("P38 seam position bounds are invalid")
  if int(padded_rows_per_dp) <= 0 or int(total_rows) <= 0:
    raise ValueError("P38 seam padded geometry must be positive")
  dp_size = len(req_ids_dp)
  if dp_size <= 0 or int(total_rows) != dp_size * int(padded_rows_per_dp):
    raise ValueError(
        "P38 seam token rows do not equal DP times padded rows: "
        f"rows={total_rows} dp={dp_size} padded={padded_rows_per_dp}"
    )

  selected_rows: list[int] = []
  positions: list[int] = []
  token_ids: list[int] = []
  request_ordinals: list[int] = []
  prefix_hashes: list[bytes] = []
  requests: list[dict] = []
  populated = 0
  expected = sum(int(value) for value in scheduled_tokens.values())
  for dp_rank in range(dp_size):
    offset = 0
    for request_id in req_ids_dp.get(dp_rank, ()):
      count = int(scheduled_tokens.get(request_id, 0))
      if count <= 0:
        continue
      if request_id not in req_id_to_index:
        raise ValueError(
            f"P38 seam request is absent from the input batch: {request_id}"
        )
      request_index = int(req_id_to_index[request_id])
      prefix = int(num_computed_tokens[request_index])
      available = int(num_tokens[request_index])
      if prefix < 0 or prefix + count > available:
        raise ValueError(
            "P38 seam scheduled range exceeds request tokens: "
            f"request={request_id} prefix={prefix} count={count} "
            f"available={available}"
        )
      if offset + count > int(padded_rows_per_dp):
        raise ValueError(
            f"P38 seam DP row overflow: dp={dp_rank} offset={offset} count={count}"
        )
      request_ordinal = len(requests)
      request_selected = 0
      for local_offset in range(count):
        position = prefix + local_offset
        if not int(min_position) <= position < int(max_position):
          continue
        row = dp_rank * int(padded_rows_per_dp) + offset + local_offset
        selected_rows.append(row)
        positions.append(position)
        token_id = int(token_ids_cpu[request_index, position])
        token_ids.append(token_id)
        request_ordinals.append(request_ordinal)
        prefix_hashes.append(
            _prefix_sha256(token_ids_cpu[request_index, :position + 1]).encode()
        )
        request_selected += 1
      requests.append({
          "request_id": str(request_id),
          "request_index": request_index,
          "dp_rank": int(dp_rank),
          "packed_row_range": [
              int(dp_rank * int(padded_rows_per_dp) + offset),
              int(dp_rank * int(padded_rows_per_dp) + offset + count),
          ],
          "position_range": [prefix, prefix + count],
          "scheduled_tokens": count,
          "selected_tokens": request_selected,
          "token_history_sha256": _prefix_sha256(
              token_ids_cpu[request_index, :available]
          ),
      })
      offset += count
      populated += count
  if populated != expected:
    raise ValueError(
        f"P38 seam packed rows incomplete: populated={populated} expected={expected}"
    )

  arrays = {
      "row_indices": np.asarray(selected_rows, dtype=np.int32),
      "positions": np.asarray(positions, dtype=np.int32),
      "token_ids": np.asarray(token_ids, dtype=np.int32),
      "request_ordinals": np.asarray(request_ordinals, dtype=np.int32),
      "token_prefix_sha256": np.asarray(prefix_hashes, dtype="S64"),
  }
  return arrays["row_indices"], arrays, requests


def next_power_of_two(value: int) -> int:
  if int(value) <= 0:
    raise ValueError("P38 seam gather requires a positive row count")
  return 1 << (int(value) - 1).bit_length()


def write_seam_record(
    directory: str,
    state: dict[str, int],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping,
    max_bytes: int,
) -> tuple[int, str]:
  """Persist one exclusive, self-checking NPZ+JSON evidence record."""
  record_index = int(state.get("records", 0))
  current_bytes = int(state.get("bytes", 0))
  root = Path(directory)
  root.mkdir(parents=True, exist_ok=True)
  base = root / f"p38_seam_{record_index:06d}"
  npz_path = Path(str(base) + ".npz")
  json_path = Path(str(base) + ".json")
  if npz_path.exists() or json_path.exists():
    raise RuntimeError(f"P38 seam evidence collision at {base}")

  payload = io.BytesIO()
  np.savez(payload, **arrays)
  npz_bytes = payload.getvalue()
  npz_sha256 = hashlib.sha256(npz_bytes).hexdigest()
  record = {
      **metadata,
      "schema": "p38-seam-fingerprint-v1",
      "record_index": record_index,
      "array_keys": sorted(arrays),
      "npz_sha256": npz_sha256,
  }
  json_bytes = (
      json.dumps(record, sort_keys=True, indent=2, default=str) + "\n"
  ).encode()
  new_bytes = len(npz_bytes) + len(json_bytes)
  if current_bytes + new_bytes > int(max_bytes):
    raise RuntimeError(
        "P38 seam evidence exceeded its registered output byte bound"
    )

  npz_fd = os.open(npz_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
  try:
    if os.write(npz_fd, npz_bytes) != len(npz_bytes):
      raise RuntimeError("P38 seam NPZ write was incomplete")
    os.fsync(npz_fd)
  finally:
    os.close(npz_fd)
  json_fd = os.open(json_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
  try:
    if os.write(json_fd, json_bytes) != len(json_bytes):
      raise RuntimeError("P38 seam JSON write was incomplete")
    os.fsync(json_fd)
  finally:
    os.close(json_fd)
  if hashlib.sha256(npz_path.read_bytes()).hexdigest() != npz_sha256:
    raise RuntimeError("P38 seam NPZ self-check failed")
  state["records"] = record_index + 1
  state["bytes"] = current_bytes + new_bytes
  return record_index, npz_sha256
