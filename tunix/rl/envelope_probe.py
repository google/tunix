# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default-off P35 pre-backward envelope discriminator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tunix.rl import alignment


ENV = "CANON_P35_ENVELOPE"
REPORT_ENV = "CANON_P35_ENVELOPE_REPORT"
METADATA_DIR_ENV = "CANON_P35_METADATA_DIR"
DATA_SIZE_ENV = "CANON_DP_SIZE"
LOCAL_M_ENV = "CANON_LOGPROB_M"

PAIR_AB = "A_native_vs_B_canonical_serving"
PAIR_BC = "B_canonical_serving_vs_C_adapter"
PAIR_AC = "A_native_vs_C_adapter"


class EnvelopeProbeError(RuntimeError):
  """Raised when a P35 measurement cannot satisfy its input contract."""


class EnvelopeProbeComplete(RuntimeError):
  """Intentional pre-backward terminal used only by the diagnostic runner."""


def enabled() -> bool:
  return os.environ.get(ENV, "") == "1"


def rank_strided_row_groups(num_rows: int, data_size: int) -> np.ndarray:
  """Returns the exact global-row groups used by the DP canonical adapter."""
  if data_size <= 0:
    raise EnvelopeProbeError(f"data_size must be positive, got {data_size}")
  if num_rows <= 0 or num_rows % data_size:
    raise EnvelopeProbeError(
        "global row count must be a nonzero multiple of data size: "
        f"rows={num_rows} data={data_size}"
    )
  return np.arange(num_rows, dtype=np.int64).reshape(data_size, -1).T


def _element_bitwise_difference(a: Any, b: Any) -> np.ndarray:
  aa = np.ascontiguousarray(np.asarray(a))
  bb = np.ascontiguousarray(np.asarray(b))
  if aa.shape != bb.shape or aa.dtype != bb.dtype:
    raise EnvelopeProbeError(
        "bitwise comparison requires equal shape and dtype: "
        f"{aa.shape}/{aa.dtype} vs {bb.shape}/{bb.dtype}"
    )
  return (
      aa.view(np.uint8).reshape(aa.shape + (aa.dtype.itemsize,))
      != bb.view(np.uint8).reshape(bb.shape + (bb.dtype.itemsize,))
  ).any(axis=-1)


def select_reproducing_group(
    a_full: Any,
    c_full: Any,
    action_mask: Any,
    *,
    data_size: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
  """Selects the C rank-strided group containing the first current A-C red."""
  aa = np.asarray(a_full)
  cc = np.asarray(c_full)
  mask = np.asarray(action_mask, dtype=np.bool_)
  if aa.shape != cc.shape or aa.shape != mask.shape or aa.ndim < 2:
    raise EnvelopeProbeError(
        "A, C and action mask must have one equal rank-2-or-higher shape"
    )
  red = _element_bitwise_difference(aa, cc) & mask
  coordinates = np.argwhere(red)
  if coordinates.size == 0:
    raise EnvelopeProbeError("known A-C red was not reproduced in the current batch")
  first_coordinate = tuple(int(value) for value in coordinates[0])
  groups = rank_strided_row_groups(aa.shape[0], data_size)
  local_batch = aa.shape[0] // data_size
  group_index = first_coordinate[0] % local_batch
  return groups[group_index].copy(), first_coordinate


def compact_sequences(
    prompt_ids: Any,
    completion_ids: Any,
    prompt_mask: Any,
    completion_valid_mask: Any,
) -> tuple[tuple[int, ...], ...]:
  """Returns the exact compact token sequences consumed by A, B and C."""
  prompts = np.asarray(prompt_ids)
  completions = np.asarray(completion_ids)
  prompt_valid = np.asarray(prompt_mask, dtype=np.bool_)
  completion_valid = np.asarray(completion_valid_mask, dtype=np.bool_)
  if (
      prompts.ndim != 2
      or completions.ndim != 2
      or prompts.shape != prompt_valid.shape
      or completions.shape != completion_valid.shape
      or prompts.shape[0] != completions.shape[0]
  ):
    raise EnvelopeProbeError("token arrays and validity masks have incompatible shapes")
  sequences = []
  for row in range(prompts.shape[0]):
    valid_completion = completion_valid[row]
    count = int(valid_completion.sum())
    if not np.array_equal(
        valid_completion,
        np.arange(valid_completion.size, dtype=np.int64) < count,
    ):
      raise EnvelopeProbeError(
          f"completion validity mask is not a contiguous prefix at row {row}"
      )
    sequence = np.concatenate(
        (prompts[row][prompt_valid[row]], completions[row][:count])
    )
    if sequence.size == 0:
      raise EnvelopeProbeError(f"row {row} has no valid tokens")
    sequences.append(tuple(int(token) for token in sequence))
  return tuple(sequences)


def _load_metadata_records(directory: str | os.PathLike[str]) -> list[dict[str, Any]]:
  root = Path(directory)
  records = []
  for json_path in sorted(root.glob("p35_metadata_*.json")):
    record = json.loads(json_path.read_text(encoding="utf-8"))
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
      raise EnvelopeProbeError(f"P35 metadata arrays are missing: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
      record["arrays"] = {key: arrays[key].copy() for key in arrays.files}
    records.append(record)
  if not records:
    raise EnvelopeProbeError(f"no P35 metadata records in {root}")
  return records


def attest_metadata(
    *,
    directory: str | os.PathLike[str],
    expected_b_sequences: tuple[tuple[int, ...], ...],
    expected_a_rows: int,
    data_size: int,
    local_m: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
  """Checks all arm-labelled serving chunks against C sequence semantics."""
  records = _load_metadata_records(directory)
  a_records = [record for record in records if record.get("arm") == "A"]
  b_records = [record for record in records if record.get("arm") == "B"]
  if not a_records:
    raise EnvelopeProbeError("compact metadata contains no native A record")
  if not b_records:
    raise EnvelopeProbeError("compact metadata contains no grouped B record")
  if len(expected_b_sequences) != data_size:
    raise EnvelopeProbeError(
        "B sequence group must contain exactly one row per data rank: "
        f"{len(expected_b_sequences)} vs {data_size}"
    )
  required_arrays = {
      "input_ids",
      "input_positions",
      "md_input_positions",
      "md_seq_lens",
      "md_query_start_loc",
      "md_request_distribution",
      "md_block_tables",
  }
  expected_lengths = np.asarray(
      [len(sequence) for sequence in expected_b_sequences], dtype=np.int64
  )
  consumed = np.zeros(data_size, dtype=np.int64)
  local_m_values: list[int] = []
  query_lengths: list[list[int]] = []
  block_table_entries = 0
  local_m_ok = True
  token_rows_ok = True
  positions_ok = True
  one_per_rank = True
  block_tables_ok = True
  cache_fresh = True
  first_page_ids = np.full(data_size, -1, dtype=np.int64)

  for record_index, b_record in enumerate(b_records):
    arrays = b_record.get("arrays", {})
    if not required_arrays.issubset(arrays):
      raise EnvelopeProbeError(
          f"B metadata record {record_index} is missing arrays: "
          f"{sorted(required_arrays - set(arrays))}"
      )
    input_ids = np.asarray(arrays["input_ids"]).reshape(-1)
    positions = np.asarray(arrays["input_positions"]).reshape(-1)
    metadata_positions = np.asarray(arrays["md_input_positions"]).reshape(-1)
    local_m_observed = (
        input_ids.size // data_size
        if data_size and input_ids.size % data_size == 0
        else 0
    )
    local_m_values.append(int(local_m_observed))
    record_local_m_ok = bool(
        input_ids.size == data_size * local_m
        and positions.size == input_ids.size
        and metadata_positions.size == positions.size
        and local_m_observed == local_m
    )
    metadata_positions_ok = bool(
        metadata_positions.size == positions.size
        and np.array_equal(metadata_positions, positions)
    )
    local_m_ok &= record_local_m_ok
    positions_ok &= metadata_positions_ok

    meta = b_record.get("meta", {})
    padded_num_reqs = int(meta.get("md_padded_num_reqs", 0) or 0)
    request_slots_ok = bool(
        padded_num_reqs > 0 and padded_num_reqs % data_size == 0
    )
    local_slots = padded_num_reqs // data_size if request_slots_ok else 0
    seq_lens = np.asarray(arrays["md_seq_lens"]).reshape(-1)
    query_start = np.asarray(arrays["md_query_start_loc"]).reshape(-1)
    distribution = np.asarray(arrays["md_request_distribution"]).reshape(-1)
    block_tables = np.asarray(arrays["md_block_tables"]).reshape(-1)
    block_table_entries += int(block_tables.size)
    record_shapes_ok = bool(
        request_slots_ok
        and seq_lens.size == data_size * local_slots
        and query_start.size == data_size * (local_slots + 1)
        and distribution.size == data_size * 3
        and block_tables.size > 0
        and block_tables.size % padded_num_reqs == 0
    )
    one_per_rank &= record_shapes_ok
    block_tables_ok &= record_shapes_ok
    if not (record_local_m_ok and metadata_positions_ok and record_shapes_ok):
      token_rows_ok = False
      positions_ok = False
      cache_fresh = False
      continue

    input_by_rank = input_ids.reshape(data_size, local_m)
    position_by_rank = positions.reshape(data_size, local_m)
    seq_by_rank = seq_lens.reshape(data_size, local_slots)
    query_by_rank = query_start.reshape(data_size, local_slots + 1)
    distribution_by_rank = distribution.reshape(data_size, 3)
    blocks_per_request = block_tables.size // padded_num_reqs
    block_by_rank = block_tables.reshape(
        data_size, local_slots, blocks_per_request
    )
    record_query_lengths = []
    for rank, sequence in enumerate(expected_b_sequences):
      active_slots = int(distribution_by_rank[rank, 2])
      q_len = int(query_by_rank[rank, 1] - query_by_rank[rank, 0])
      record_query_lengths.append(q_len)
      remaining = int(expected_lengths[rank] - consumed[rank])
      rank_active = q_len > 0
      rank_contract_ok = bool(
          active_slots in (0, 1)
          and active_slots == int(rank_active)
          and query_by_rank[rank, 0] == 0
          and 0 <= q_len <= local_m
          and q_len <= remaining
          and int((seq_by_rank[rank] > 0).sum()) == int(rank_active)
      )
      if rank_active:
        page_id = int(block_by_rank[rank, 0, 0])
        rank_contract_ok &= bool(
            int(seq_by_rank[rank, 0]) == int(consumed[rank] + q_len)
            and page_id >= 0
        )
        if first_page_ids[rank] < 0:
          first_page_ids[rank] = page_id
        else:
          block_tables_ok &= bool(first_page_ids[rank] == page_id)
      one_per_rank &= rank_contract_ok
      block_tables_ok &= bool(
          not rank_active or int(block_by_rank[rank, 0, 0]) >= 0
      )
      start = int(consumed[rank])
      stop = start + q_len
      expected_tokens = np.asarray(sequence[start:stop], dtype=input_ids.dtype)
      expected_positions = np.arange(start, stop, dtype=positions.dtype)
      token_rows_ok &= bool(
          rank_contract_ok
          and np.array_equal(input_by_rank[rank, :q_len], expected_tokens)
      )
      positions_ok &= bool(
          rank_contract_ok
          and np.array_equal(
              position_by_rank[rank, :q_len], expected_positions
          )
      )
      if rank_active and start == 0:
        cache_fresh &= bool(position_by_rank[rank, 0] == 0)
      consumed[rank] = stop
    one_per_rank &= bool(any(length > 0 for length in record_query_lengths))
    query_lengths.append(record_query_lengths)

  complete_sequences = bool(np.array_equal(consumed, expected_lengths))
  token_rows_ok &= complete_sequences
  positions_ok &= complete_sequences
  one_per_rank &= complete_sequences
  cache_fresh &= bool(
      complete_sequences and positions_ok and block_tables_ok and one_per_rank
  )

  mesh_descriptions = [record.get("meta", {}).get("mesh") for record in records]
  mesh_equal = all(description == mesh_descriptions[0] for description in mesh_descriptions)
  mesh = mesh_descriptions[0] if mesh_descriptions else None
  shape = mesh.get("shape", {}) if isinstance(mesh, dict) else {}
  mesh_shape_ok = int(shape.get("data", 0)) == data_size
  device_ids = mesh.get("device_ids", []) if isinstance(mesh, dict) else []
  device_order_ok = mesh_equal and len(device_ids) > 0

  a_request_ids = set()
  for record in a_records:
    prompt_logprobs = record.get("meta", {}).get("num_prompt_logprobs", {})
    if isinstance(prompt_logprobs, dict):
      a_request_ids.update(str(key) for key in prompt_logprobs)
  native_a_observed = len(a_request_ids) == expected_a_rows

  attestations = {
      "native_A_observed": bool(native_a_observed),
      "grouped_B_observed": bool(b_records and complete_sequences),
      "mesh_shape_expected": bool(mesh_shape_ok),
      "device_order_expected": bool(device_order_ok),
      "local_m256_B": bool(local_m_ok and local_m == 256),
      "positions_equal": bool(positions_ok),
      "block_tables_B_observed": bool(block_tables_ok),
      "request_distribution_B_one_per_rank": bool(one_per_rank),
      "metadata_B_matches_C": bool(
          token_rows_ok and positions_ok and one_per_rank and block_tables_ok
      ),
      "cache_fresh_B": bool(cache_fresh),
  }
  summary = {
      "records": len(records),
      "A_records": len(a_records),
      "B_records": len(b_records),
      "A_request_ids": len(a_request_ids),
      "expected_A_rows": int(expected_a_rows),
      "B_local_m": int(local_m) if local_m_ok else None,
      "B_local_m_values": local_m_values,
      "B_sequence_lengths": [len(sequence) for sequence in expected_b_sequences],
      "B_consumed_lengths": consumed.tolist(),
      "B_query_lengths_by_record": query_lengths,
      "B_block_table_entries": int(block_table_entries),
      "B_first_page_ids": first_page_ids.tolist(),
      "mesh": mesh,
  }
  return attestations, summary


def _pair(a: Any, b: Any, mask: Any) -> dict[str, Any]:
  difference = alignment._masked_bitwise_difference(a, b, mask)  # pylint: disable=protected-access
  hash_a = alignment._masked_hash(a, mask)  # pylint: disable=protected-access
  hash_b = alignment._masked_hash(b, mask)  # pylint: disable=protected-access
  return {
      **difference,
      "masked_hash_a": hash_a,
      "masked_hash_b": hash_b,
      "masked_hashes_equal": hash_a == hash_b,
  }


def _negative_control(value: Any, mask: Any) -> dict[str, Any]:
  source = np.asarray(value)
  bool_mask = np.asarray(mask, dtype=np.bool_)
  coordinates = np.argwhere(bool_mask)
  if coordinates.size == 0:
    raise EnvelopeProbeError("negative control requires at least one action element")
  changed = np.ascontiguousarray(source).copy()
  flat_index = int(np.ravel_multi_index(tuple(coordinates[0]), source.shape))
  bytes_by_element = changed.reshape(-1).view(np.uint8).reshape(
      changed.size, changed.dtype.itemsize
  )
  bytes_by_element[flat_index, 0] ^= np.uint8(1)
  result = _pair(source, changed, bool_mask)
  return {
      "injected": True,
      "differing_elements": result["differing_elements"],
      "differing_bytes": result["differing_bytes"],
      "masked_hashes_equal": result["masked_hashes_equal"],
  }


def build_report(
    *,
    a: Any,
    b: Any,
    c: Any,
    action_mask: Any,
    selected_row_indices: Any,
    first_full_ac_mismatch: tuple[int, ...],
    attestations: Mapping[str, bool],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
  """Builds one complete schema-v2 report without changing model values."""
  aa = np.asarray(a)
  bb = np.asarray(b)
  cc = np.asarray(c)
  mask = np.asarray(action_mask, dtype=np.bool_)
  if aa.shape != bb.shape or aa.shape != cc.shape or aa.shape != mask.shape:
    raise EnvelopeProbeError(
        "selected A/B/C values and action mask must have exactly one shape"
    )
  rows = np.asarray(selected_row_indices, dtype=np.int64).reshape(-1)
  if rows.size != aa.shape[0] or np.unique(rows).size != rows.size:
    raise EnvelopeProbeError("selected row indices are incomplete or duplicated")
  return {
      "schema_version": 2,
      "measurement_rows": 1,
      "arms": ["A", "B", "C"],
      "selected_row_indices": rows.tolist(),
      "first_full_A_vs_C_mismatch": list(first_full_ac_mismatch),
      "attestations": {key: bool(value) for key, value in attestations.items()},
      "metadata": dict(metadata),
      "negative_control": _negative_control(aa, mask),
      "pairs": {
          PAIR_AB: _pair(aa, bb, mask),
          PAIR_BC: _pair(bb, cc, mask),
          PAIR_AC: _pair(aa, cc, mask),
      },
  }


def write_report(report: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
  """Writes exactly one immutable report and refuses evidence collisions."""
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
  return output
