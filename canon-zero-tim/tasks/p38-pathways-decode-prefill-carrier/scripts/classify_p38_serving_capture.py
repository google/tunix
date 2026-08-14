#!/usr/bin/env python3
"""Fail-closed classifier for bounded P38 continue-decode captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PRE_ARRAYS = {
    "input_ids",
    "input_positions",
    "active_mask",
    "md_input_positions",
    "md_block_tables",
    "md_seq_lens",
    "md_query_start_loc",
    "md_request_distribution",
    "tokens_indices_selector",
    "rng",
}
POST_ARRAYS = {
    "generated_tokens",
    "final_input_positions",
    "final_seq_lens",
    "logprob_token_ids",
    "logprob_values",
    "logprob_ranks",
}
CAPSULE_ARRAYS = {
    "selected_rows",
    "metadata_json",
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_valid_mask",
    "action_mask",
    "s_decode",
    "s_prefill",
    "t_old",
}


class CaptureError(RuntimeError):
  """Raised when a serving capture is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise CaptureError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  return hashlib.sha256(array.tobytes()).hexdigest()


def _token_history_sha256(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value, dtype="<i8"))
  return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_prefix_bounds(value: str) -> tuple[int, ...]:
  parts = value.split(",")
  _require(len(parts) >= 2 and all(part.strip() for part in parts),
           "prefix bounds must contain at least two integers")
  try:
    bounds = tuple(int(part) for part in parts)
  except ValueError as error:
    raise CaptureError("prefix bounds must contain integers") from error
  _require(all(left < right for left, right in zip(bounds, bounds[1:])),
           "prefix bounds must be strictly increasing")
  return bounds


def _validate_implementation_identity(meta: dict[str, Any], seq: int) -> None:
  identity = meta.get("implementation_identity")
  _require(isinstance(identity, dict),
           f"sequence {seq} has no implementation identity")
  runner_class = identity.get("runner_class", {})
  _require(runner_class.get("module") and runner_class.get("qualname"),
           f"sequence {seq} has an invalid runner-class identity")
  for name in (
      "continue_decode", "execute_model", "model_fn", "compute_logits_fn",
      "sample_fn",
  ):
    item = identity.get(name)
    _require(isinstance(item, dict) and item.get("chain"),
             f"sequence {seq} has no {name} identity")
    for link in item["chain"]:
      _require(link.get("type_module") and link.get("type_name"),
               f"sequence {seq} has an invalid {name} wrapper identity")
    source_sha = item.get("source_sha256")
    if source_sha is not None:
      _require(len(source_sha) == 64 and all(
          character in "0123456789abcdef" for character in source_sha),
          f"sequence {seq} has an invalid {name} source SHA")


def _primary_block_ids(value: Any) -> list[int]:
  """Returns the single Qwen KV group's physical page list."""
  _require(isinstance(value, list) and value, "request has invalid physical page IDs")
  if isinstance(value[0], list):
    _require(len(value) == 1, "capture contains more than one KV cache group")
    value = value[0]
  _require(all(isinstance(item, int) for item in value), "physical page IDs are not integers")
  return [int(item) for item in value]


def _block_table_matrix(
    value: Any, meta: dict[str, Any], seq: int
) -> np.ndarray:
  """Restores the runner's logical block-table matrix.

  The standard Pathways runner may serialize ``AttentionMetadata.block_tables``
  as one flat device buffer even though the host request contract is a
  ``[dp * rows_per_dp, max_blocks]`` matrix.  The capture metadata contains the
  two dimensions needed to restore that view without guessing from request
  contents.
  """
  block_tables = np.asarray(value)
  if block_tables.ndim == 1:
    dp_size = int(meta.get("dp_size", 0))
    rows_per_dp = int(meta.get("max_attention_rows_per_dp", 0))
    rows = dp_size * rows_per_dp
    _require(
        rows > 0 and block_tables.size % rows == 0,
        f"sequence {seq} flattened block tables are not row-divisible",
    )
    block_tables = block_tables.reshape(rows, -1)
  _require(
      block_tables.ndim == 2,
      f"sequence {seq} block tables are not rank two",
  )
  return block_tables


def _load_mismatch_capsule(path: Path) -> dict[str, Any]:
  _require(path.is_file(), f"mismatch capsule does not exist: {path}")
  with np.load(path, allow_pickle=False) as archive:
    arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  _require(CAPSULE_ARRAYS.issubset(arrays), f"mismatch capsule inventory is incomplete: {sorted(CAPSULE_ARRAYS - set(arrays))}")
  try:
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise CaptureError("mismatch capsule metadata is invalid") from error
  _require(metadata.get("schema") == "p38-frozenlake-mismatch-capsule-v1", "mismatch capsule schema is invalid")
  diagnostic_round = metadata.get("diagnostic_round")
  _require(
      isinstance(diagnostic_round, int) and 0 <= diagnostic_round < 8,
      "mismatch capsule diagnostic round is invalid",
  )
  selected_rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  _require(selected_rows.size > 0, "mismatch capsule selected no rows")
  _require(len(set(int(row) for row in selected_rows)) == selected_rows.size, "mismatch capsule selected duplicate rows")
  for name in CAPSULE_ARRAYS - {"selected_rows", "metadata_json"}:
    value = arrays[name]
    _require(value.ndim > 0 and value.shape[0] == selected_rows.size, f"mismatch capsule {name} is not row aligned")
    expected = metadata.get("arrays", {}).get(name, {})
    _require(expected.get("shape") == list(value.shape), f"mismatch capsule {name} shape attestation failed")
    _require(expected.get("dtype") == str(value.dtype), f"mismatch capsule {name} dtype attestation failed")
    _require(expected.get("sha256") == _array_sha256(value), f"mismatch capsule {name} SHA attestation failed")
  return {"arrays": arrays, "metadata": metadata}


def _validate_request_mapping(
    meta: dict[str, Any], arrays: dict[str, np.ndarray], seq: int
) -> list[dict[str, Any]]:
  request_ids = meta.get("request_ids", [])
  requests = meta.get("requests", [])
  by_dp = meta.get("request_ids_by_dp", {})
  index_map = meta.get("req_id_to_index", {})
  _require(request_ids and len(set(request_ids)) == len(request_ids), f"sequence {seq} has invalid request IDs")
  _require(len(requests) == len(request_ids), f"sequence {seq} request metadata count mismatch")
  _require(int(meta.get("scheduled_request_count", -1)) == len(request_ids), f"sequence {seq} scheduled request count mismatch")
  _require(isinstance(by_dp, dict) and by_dp, f"sequence {seq} has no DP request mapping")
  _require(set(index_map) == set(request_ids), f"sequence {seq} request-index keys mismatch")

  flattened = [request_id for rank in sorted(by_dp, key=int) for request_id in by_dp[rank]]
  _require(len(flattened) == len(set(flattened)), f"sequence {seq} DP mapping contains duplicates")
  _require(set(flattened) == set(request_ids), f"sequence {seq} DP mapping request set mismatch")
  records = {item.get("request_id"): item for item in requests}
  _require(set(records) == set(request_ids), f"sequence {seq} request record IDs mismatch")

  padded_rows = int(meta.get("padded_rows_per_dp", 0))
  attention_rows = int(meta.get("max_attention_rows_per_dp", 0))
  program_path = meta.get("program_path")
  _require(program_path in ("standard", "continue_decode"),
           f"sequence {seq} has an invalid program path")
  _require(padded_rows > 0 and attention_rows > 0, f"sequence {seq} row geometry is invalid")
  selector = np.asarray(arrays["tokens_indices_selector"]).reshape(-1)
  positions = np.asarray(arrays["input_positions"]).reshape(-1)
  active = np.asarray(arrays["active_mask"], dtype=np.bool_).reshape(-1)
  seq_lens = np.asarray(arrays["md_seq_lens"]).reshape(-1)
  query_start = np.asarray(arrays["md_query_start_loc"]).reshape(-1)
  block_tables = _block_table_matrix(arrays["md_block_tables"], meta, seq)

  seen_indices: set[int] = set()
  seen_slots: set[tuple[int, int]] = set()
  for rank_text in sorted(by_dp, key=int):
    dp_rank = int(rank_text)
    for request_id in by_dp[rank_text]:
      item = records[request_id]
      local_slot = int(item.get("local_scheduler_slot", -1))
      input_index = int(item.get("input_batch_index", -1))
      scheduled_tokens = int(item.get("scheduled_tokens", 0))
      _require(0 <= local_slot < attention_rows, f"sequence {seq} request {request_id} local slot out of range")
      _require((dp_rank, local_slot) not in seen_slots, f"sequence {seq} local scheduler slot is duplicated")
      seen_slots.add((dp_rank, local_slot))
      if program_path == "standard":
        packed_token_offset = int(item.get("packed_token_offset", -1))
        _require(packed_token_offset >= 0,
                 f"sequence {seq} request {request_id} has no packed offset")
        global_row = dp_rank * padded_rows + packed_token_offset
      else:
        global_row = dp_rank * padded_rows + local_slot
      attention_row = dp_rank * attention_rows + local_slot
      query_index = dp_rank * (attention_rows + 1) + local_slot
      _require(scheduled_tokens == 1, f"sequence {seq} request {request_id} is not a one-token decode")
      _require(input_index == int(index_map[request_id]), f"sequence {seq} request {request_id} input index mismatch")
      _require(input_index not in seen_indices, f"sequence {seq} input index is duplicated")
      seen_indices.add(input_index)
      _require(int(item.get("dp_rank", -1)) == dp_rank, f"sequence {seq} request {request_id} DP rank mismatch")
      _require(int(item.get("global_row", -1)) == global_row, f"sequence {seq} request {request_id} global row mismatch")
      _require(int(item.get("attention_row", -1)) == attention_row, f"sequence {seq} request {request_id} attention row mismatch")
      _require(int(item.get("selector_index", -1)) == input_index, f"sequence {seq} request {request_id} selector index mismatch")
      _require(0 <= input_index < selector.size, f"sequence {seq} request {request_id} selector index out of range")
      _require(int(selector[input_index]) == global_row, f"sequence {seq} request {request_id} selector value mismatch")
      _require(item.get("selector_range") == [global_row, global_row + 1], f"sequence {seq} request {request_id} selector range mismatch")
      _require(global_row < positions.size and global_row < active.size and bool(active[global_row]), f"sequence {seq} request {request_id} active row mismatch")
      computed = int(item.get("num_computed_tokens", -1))
      expected_seq_len = computed + scheduled_tokens
      _require(int(positions[global_row]) == computed, f"sequence {seq} request {request_id} position mismatch")
      _require(int(item.get("expected_seq_len", -1)) == expected_seq_len, f"sequence {seq} request {request_id} expected length mismatch")
      _require(attention_row < seq_lens.size and int(seq_lens[attention_row]) == expected_seq_len, f"sequence {seq} request {request_id} sequence length mismatch")
      _require(query_index + 1 < query_start.size, f"sequence {seq} request {request_id} query range out of bounds")
      expected_query = [int(query_start[query_index]), int(query_start[query_index + 1])]
      _require(item.get("query_start_range") == expected_query and expected_query[1] - expected_query[0] == 1, f"sequence {seq} request {request_id} query range mismatch")
      token_ids = item.get("token_ids", [])
      _require(token_ids and len(token_ids) == int(item.get("num_tokens", -1)), f"sequence {seq} request {request_id} token history length mismatch")
      _require(item.get("token_history_sha256") == _token_history_sha256(token_ids), f"sequence {seq} request {request_id} token-history SHA mismatch")
      logical_blocks = int(item.get("logical_blocks", 0))
      _require(logical_blocks > 0 and attention_row < block_tables.shape[0], f"sequence {seq} request {request_id} block-table row mismatch")
      metadata_pages = [int(value) for value in block_tables[attention_row, :logical_blocks]]
      _require(item.get("metadata_block_ids") == metadata_pages, f"sequence {seq} request {request_id} metadata page mismatch")
      _require(_primary_block_ids(item.get("block_ids"))[:logical_blocks] == metadata_pages, f"sequence {seq} request {request_id} physical page mapping mismatch")
  return requests


def _join_mismatch_capsule(
    requests: list[dict[str, Any]], capsule: dict[str, Any], seq: int
) -> list[dict[str, Any]]:
  arrays = capsule["arrays"]
  selected_rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  joins = []
  joined_request_ids: set[str] = set()
  for capsule_index, source_row in enumerate(selected_rows):
    prompt = np.asarray(arrays["prompt_ids"][capsule_index])[
        np.asarray(arrays["prompt_mask"][capsule_index], dtype=np.bool_)
    ]
    completion = np.asarray(arrays["completion_ids"][capsule_index])[
        np.asarray(arrays["completion_valid_mask"][capsule_index], dtype=np.bool_)
    ]
    full_history = np.concatenate((prompt, completion)).astype(np.int64, copy=False)
    candidates = []
    for item in requests:
      captured = np.asarray(item["token_ids"], dtype=np.int64)
      if captured.size <= full_history.size and np.array_equal(captured, full_history[:captured.size]):
        candidates.append((item, int(source_row), full_history[:captured.size]))
    if not candidates:
      continue
    _require(
        len(candidates) == 1,
        "sequence "
        f"{seq} source row {int(source_row)} token-history join expected "
        f"at most one candidate, found {len(candidates)}",
    )
    item, joined_row, matching_prefix = candidates[0]
    request_id = str(item["request_id"])
    _require(
        request_id not in joined_request_ids,
        f"sequence {seq} request {request_id} joins multiple source rows",
    )
    joined_request_ids.add(request_id)
    prefix_sha = _token_history_sha256(matching_prefix)
    _require(
        prefix_sha == item["token_history_sha256"],
        f"sequence {seq} mismatch join hash mismatch",
    )
    joins.append({
        "request_id": request_id,
        "source_row": joined_row,
        "captured_tokens": len(item["token_ids"]),
        "token_history_sha256": prefix_sha,
    })
  return joins


def _load_request_journal(
    directory: Path,
    prefix_bounds: tuple[int, ...],
    expected_program_path: str,
) -> list[dict[str, Any]]:
  path = directory / "p38_request_journal.jsonl"
  _require(path.is_file() and path.stat().st_size > 0,
           "P38 request journal is absent or empty")
  records = []
  seen: set[tuple[str, int]] = set()
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    _require(line.strip(), f"request journal line {line_number} is empty")
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      raise CaptureError(
          f"request journal line {line_number} is not JSON"
      ) from error
    _require(record.get("schema") == "p38-request-journal-v1",
             f"request journal line {line_number} has an invalid schema")
    _require(record.get("program_path") == expected_program_path,
             f"request journal line {line_number} changed program path")
    request_id = record.get("request_id")
    _require(isinstance(request_id, str) and request_id,
             f"request journal line {line_number} has no request ID")
    stratum_index = int(record.get("stratum_index", -1))
    _require(0 <= stratum_index < len(prefix_bounds) - 1,
             f"request journal line {line_number} has an invalid stratum")
    expected_stratum = [
        prefix_bounds[stratum_index], prefix_bounds[stratum_index + 1]
    ]
    _require(record.get("stratum") == expected_stratum,
             f"request journal line {line_number} has drifted bounds")
    prefix = int(record.get("num_computed_tokens", -1))
    _require(expected_stratum[0] <= prefix < expected_stratum[1],
             f"request journal line {line_number} is outside its stratum")
    key = (request_id, stratum_index)
    _require(key not in seen,
             f"request journal repeats request/stratum {key}")
    seen.add(key)
    token_ids = record.get("token_ids")
    _require(isinstance(token_ids, list) and token_ids and
             all(isinstance(token, int) for token in token_ids),
             f"request journal line {line_number} has invalid tokens")
    _require(len(token_ids) == int(record.get("num_tokens", -1)),
             f"request journal line {line_number} token count drifted")
    _require(record.get("token_history_sha256") ==
             _token_history_sha256(token_ids),
             f"request journal line {line_number} token SHA drifted")
    block_size = int(record.get("block_size", 0))
    logical_blocks = int(record.get("logical_blocks", 0))
    _require(block_size > 0 and
             logical_blocks == (prefix + 1 + block_size - 1) // block_size,
             f"request journal line {line_number} block count drifted")
    pages = record.get("physical_pages")
    generations = record.get("page_generations")
    _require(isinstance(pages, list) and len(pages) == logical_blocks and
             all(isinstance(page, int) for page in pages),
             f"request journal line {line_number} page list drifted")
    _require(isinstance(generations, list) and
             len(generations) == logical_blocks,
             f"request journal line {line_number} generation list drifted")
    for logical_page, (physical_page, generation) in enumerate(
        zip(pages, generations, strict=True)
    ):
      _require(
          isinstance(generation, dict) and
          generation.get("logical_page") == logical_page and
          generation.get("physical_page") == physical_page and
          int(generation.get("observation_generation", -1)) >= 0,
          f"request journal line {line_number} has invalid page provenance",
      )
    co_batch = record.get("co_batch_request_ids")
    _require(isinstance(co_batch, list) and request_id in co_batch and
             len(co_batch) == int(record.get("scheduled_request_count", -1)),
             f"request journal line {line_number} co-batch contract drifted")
    decode_batch = record.get("one_token_decode_request_ids")
    _require(isinstance(decode_batch, list) and request_id in decode_batch and
             set(decode_batch) <= set(co_batch) and
             len(decode_batch) == int(
                 record.get("one_token_decode_request_count", -1)
             ),
             f"request journal line {line_number} decode-batch contract drifted")
    _require(int(record.get("call_index", 0)) > 0 and
             int(record.get("dp_rank", -1)) >= 0 and
             int(record.get("local_scheduler_slot", -1)) >= 0,
             f"request journal line {line_number} scheduler identity drifted")
    records.append(record)
  return records


def _join_journal_to_capsule(
    records: list[dict[str, Any]], capsule: dict[str, Any]
) -> list[dict[str, Any]]:
  arrays = capsule["arrays"]
  selected_rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  histories = []
  for capsule_index, source_row in enumerate(selected_rows):
    prompt = np.asarray(arrays["prompt_ids"][capsule_index])[
        np.asarray(arrays["prompt_mask"][capsule_index], dtype=np.bool_)
    ]
    completion = np.asarray(arrays["completion_ids"][capsule_index])[
        np.asarray(
            arrays["completion_valid_mask"][capsule_index], dtype=np.bool_
        )
    ]
    histories.append((
        int(source_row),
        np.concatenate((prompt, completion)).astype(np.int64, copy=False),
    ))
  joins = []
  for record in records:
    captured = np.asarray(record["token_ids"], dtype=np.int64)
    candidates = [
        source_row for source_row, history in histories
        if captured.size <= history.size and
        np.array_equal(captured, history[:captured.size])
    ]
    if not candidates:
      continue
    _require(
        len(candidates) == 1,
        "request journal token history ambiguously joins source rows "
        f"{candidates}",
    )
    joins.append({
        "request_id": record["request_id"],
        "source_row": candidates[0],
        "captured_tokens": len(record["token_ids"]),
        "num_computed_tokens": int(record["num_computed_tokens"]),
        "stratum_index": int(record["stratum_index"]),
        "dp_rank": int(record["dp_rank"]),
        "local_scheduler_slot": int(record["local_scheduler_slot"]),
        "token_history_sha256": record["token_history_sha256"],
        "physical_pages": record["physical_pages"],
        "page_generations": record["page_generations"],
        "scheduled_request_count": int(record["scheduled_request_count"]),
    })
  return joins


def _load_incident_ledger(
    directory: Path,
    expected_program_path: str,
    incident_min_prefix: int,
    incident_max_prefix: int,
) -> list[dict[str, Any]]:
  """Load the bounded per-call host ledger and validate every join field."""
  _require(
      0 <= incident_min_prefix < incident_max_prefix,
      "incident prefix bounds are invalid",
  )
  path = directory / "p38_incident_ledger.jsonl"
  _require(
      path.is_file() and path.stat().st_size > 0,
      "P38 incident ledger is absent or empty",
  )
  entries = []
  previous_call = 0
  previous_round = -1
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    _require(line.strip(), f"incident ledger line {line_number} is empty")
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      raise CaptureError(
          f"incident ledger line {line_number} is not JSON"
      ) from error
    _require(
        record.get("schema") == "p38-incident-ledger-v1",
        f"incident ledger line {line_number} has an invalid schema",
    )
    _require(
        record.get("program_path") == expected_program_path,
        f"incident ledger line {line_number} changed program path",
    )
    _require(
        int(record.get("incident_min_prefix", -1)) == incident_min_prefix
        and int(record.get("incident_max_prefix", -1))
        == incident_max_prefix,
        f"incident ledger line {line_number} changed prefix bounds",
    )
    call_index = int(record.get("call_index", 0))
    _require(
        call_index > previous_call,
        f"incident ledger line {line_number} has a nonmonotonic call index",
    )
    previous_call = call_index
    diagnostic_round = int(record.get("diagnostic_round", -1))
    _require(
        0 <= diagnostic_round < 8 and diagnostic_round >= previous_round,
        f"incident ledger line {line_number} has an invalid diagnostic round",
    )
    previous_round = diagnostic_round
    co_batch = record.get("co_batch_request_ids")
    _require(
        isinstance(co_batch, list)
        and len(co_batch)
        == int(record.get("scheduled_request_count", -1))
        and len(co_batch) == len(set(co_batch)),
        f"incident ledger line {line_number} has an invalid co-batch",
    )
    compile_geometry = record.get("compile_geometry")
    if compile_geometry is not None:
      _require(
          isinstance(compile_geometry, dict),
          f"incident ledger line {line_number} has invalid compile geometry",
      )
      dp_size = int(compile_geometry.get("dp_size", 0))
      padded_rows = int(compile_geometry.get("padded_rows_per_dp", 0))
      global_rows = int(compile_geometry.get("global_padded_rows", 0))
      canonical_rows = int(
          compile_geometry.get("canonical_logprob_rows", 0)
      )
      _require(
          dp_size > 0
          and padded_rows > 0
          and global_rows == dp_size * padded_rows
          and canonical_rows > 0,
          f"incident ledger line {line_number} fixed-M scalars drifted",
      )
      for name in (
          "input_ids",
          "input_positions",
          "block_tables",
          "seq_lens",
          "query_start_loc",
          "request_distribution",
      ):
        aval = compile_geometry.get(name)
        _require(
            isinstance(aval, dict)
            and isinstance(aval.get("shape"), list)
            and aval["shape"]
            and all(int(dimension) >= 0 for dimension in aval["shape"])
            and isinstance(aval.get("dtype"), str)
            and aval["dtype"],
            f"incident ledger line {line_number} has invalid {name} aval",
        )
      _require(
          int(compile_geometry["input_ids"]["shape"][0]) == global_rows
          and compile_geometry["input_positions"]["shape"]
          == compile_geometry["input_ids"]["shape"],
          f"incident ledger line {line_number} fixed-M input aval drifted",
      )
    requests = record.get("requests")
    _require(
        isinstance(requests, list) and requests,
        f"incident ledger line {line_number} has no incident requests",
    )
    seen_requests = set()
    for request in requests:
      request_id = request.get("request_id")
      _require(
          isinstance(request_id, str)
          and request_id
          and request_id in co_batch
          and request_id not in seen_requests,
          f"incident ledger line {line_number} has an invalid request ID",
      )
      seen_requests.add(request_id)
      prefix = int(request.get("num_computed_tokens", -1))
      num_tokens = int(request.get("num_tokens", -1))
      _require(
          incident_min_prefix <= prefix < incident_max_prefix
          and num_tokens == prefix + 1,
          f"incident ledger line {line_number} request geometry drifted",
      )
      token_sha = request.get("token_history_sha256")
      _require(
          isinstance(token_sha, str)
          and len(token_sha) == 64
          and all(character in "0123456789abcdef" for character in token_sha),
          f"incident ledger line {line_number} token SHA is invalid",
      )
      single_active_tokens = request.get("single_active_token_ids")
      if compile_geometry is not None and len(co_batch) == 1:
        _require(
            isinstance(single_active_tokens, list)
            and len(single_active_tokens) == num_tokens
            and all(isinstance(token, int) for token in single_active_tokens)
            and _token_history_sha256(single_active_tokens) == token_sha,
            f"incident ledger line {line_number} single-active tokens drifted",
        )
      if compile_geometry is not None and len(co_batch) != 1:
        _require(
            single_active_tokens is None,
            f"incident ledger line {line_number} leaked non-single tokens",
        )
      block_size = int(request.get("block_size", 0))
      logical_blocks = int(request.get("logical_blocks", 0))
      pages = request.get("physical_pages")
      generations = request.get("page_generations")
      _require(
          block_size > 0
          and logical_blocks == (prefix + 1 + block_size - 1) // block_size
          and isinstance(pages, list)
          and len(pages) == logical_blocks
          and all(isinstance(page, int) for page in pages)
          and isinstance(generations, list)
          and len(generations) == logical_blocks,
          f"incident ledger line {line_number} page geometry drifted",
      )
      for logical_page, (physical_page, generation) in enumerate(
          zip(pages, generations, strict=True)
      ):
        _require(
            generation.get("logical_page") == logical_page
            and generation.get("physical_page") == physical_page
            and generation.get("observed_request_id") == request_id
            and generation.get("observed_logical_page") == logical_page
            and int(generation.get("observation_generation", -1)) >= 0,
            f"incident ledger line {line_number} page ownership drifted",
        )
      entries.append({
          **request,
          "call_index": call_index,
          "diagnostic_round": diagnostic_round,
          "co_batch_request_ids": co_batch,
          "scheduled_request_count": len(co_batch),
          "compile_geometry": compile_geometry,
      })
  return entries


def _join_incident_to_capsule(
    entries: list[dict[str, Any]], capsule: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
  """Join every selected A/B mismatch to its exact serving decode call."""
  arrays = capsule["arrays"]
  diagnostic_round = int(capsule["metadata"]["diagnostic_round"])
  selected_rows = np.asarray(arrays["selected_rows"]).reshape(-1)
  expected = []
  histories = {}
  for capsule_index, source_row_raw in enumerate(selected_rows):
    source_row = int(source_row_raw)
    prompt = np.asarray(arrays["prompt_ids"][capsule_index])[
        np.asarray(arrays["prompt_mask"][capsule_index], dtype=np.bool_)
    ]
    completion = np.asarray(arrays["completion_ids"][capsule_index])[
        np.asarray(
            arrays["completion_valid_mask"][capsule_index], dtype=np.bool_
        )
    ]
    history = np.concatenate((prompt, completion)).astype(
        np.int64, copy=False
    )
    histories[source_row] = history
    action = np.asarray(arrays["action_mask"][capsule_index], dtype=np.bool_)
    a = np.asarray(arrays["s_decode"][capsule_index])
    b = np.asarray(arrays["s_prefill"][capsule_index])
    _require(
        action.shape == a.shape == b.shape,
        f"incident join source row {source_row} shape drifted",
    )
    bytes_differ = (
        np.ascontiguousarray(a).view(np.uint8)
        != np.ascontiguousarray(b).view(np.uint8)
    ).reshape(a.size, a.dtype.itemsize).any(axis=1).reshape(a.shape)
    mismatch_positions = np.flatnonzero(action & bytes_differ)
    _require(
        mismatch_positions.size > 0,
        f"incident join source row {source_row} has no A/B mismatch",
    )
    prompt_length = int(np.asarray(
        arrays["prompt_mask"][capsule_index], dtype=np.bool_
    ).sum())
    for position in mismatch_positions:
      expected.append({
          "source_row": source_row,
          "completion_position": int(position),
          "num_computed_tokens": prompt_length + int(position),
      })

  joins = []
  missing = []
  for target in expected:
    source_row = target["source_row"]
    prefix = target["num_computed_tokens"]
    history = histories[source_row]
    expected_tokens = prefix + 1
    _require(
        expected_tokens <= history.size,
        f"incident join source row {source_row} exceeds token history",
    )
    expected_sha = _token_history_sha256(history[:expected_tokens])
    candidates = [
        entry for entry in entries
        if int(entry["diagnostic_round"]) == diagnostic_round
        and int(entry["num_computed_tokens"]) == prefix
        and int(entry["num_tokens"]) == expected_tokens
        and entry["token_history_sha256"] == expected_sha
    ]
    if not candidates:
      missing.append(target)
      continue
    _require(
        len(candidates) == 1,
        "incident ledger ambiguously joins source row "
        f"{source_row} prefix {prefix}: {len(candidates)} candidates",
    )
    entry = candidates[0]
    joins.append({
        **target,
        "request_id": entry["request_id"],
        "call_index": int(entry["call_index"]),
        "diagnostic_round": diagnostic_round,
        "dp_rank": int(entry["dp_rank"]),
        "local_scheduler_slot": int(entry["local_scheduler_slot"]),
        "token_history_sha256": expected_sha,
        "physical_pages": entry["physical_pages"],
        "page_generations": entry["page_generations"],
        "scheduled_request_count": int(entry["scheduled_request_count"]),
        "co_batch_request_ids": entry["co_batch_request_ids"],
        "compile_geometry": entry.get("compile_geometry"),
        "single_active_token_ids_present": (
            "single_active_token_ids" in entry
        ),
    })
  return joins, missing


def _load_stage(directory: Path, seq: int, stage: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  base = directory / f"p38_serving_{seq:04d}_{stage}"
  json_path = Path(str(base) + ".json")
  npz_path = Path(str(base) + ".npz")
  _require(json_path.is_file(), f"missing {stage} JSON for sequence {seq}")
  _require(npz_path.is_file(), f"missing {stage} NPZ for sequence {seq}")
  record = json.loads(json_path.read_text(encoding="utf-8"))
  _require(record.get("schema_version") == 2, f"bad schema for {stage} sequence {seq}")
  _require(record.get("stage") == stage, f"bad stage for sequence {seq}: {record.get('stage')!r}")
  _require(record.get("seq") == seq, f"bad sequence number in {stage} record")
  _require(record.get("npz_sha256") == _sha256(npz_path), f"NPZ SHA mismatch for {stage} sequence {seq}")
  storage = record.get("storage_guard", {})
  _require(int(storage.get("multiplier", 0)) >= 5,
           f"missing five-times storage guard for {stage} sequence {seq}")
  _require(int(storage.get("payload_bytes", 0)) > 0,
           f"invalid payload estimate for {stage} sequence {seq}")
  _require(int(storage.get("estimated_total_bytes", 0)) >= int(
      storage["payload_bytes"]),
      f"invalid total-size estimate for {stage} sequence {seq}")
  _require(int(storage.get("free_bytes", -1)) >= int(
      storage.get("required_free_bytes", 0)) > 0,
      f"free-space guard failed for {stage} sequence {seq}")
  with np.load(npz_path, allow_pickle=False) as archive:
    arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  _require(sorted(arrays) == sorted(record.get("arrays", [])), f"array inventory mismatch for {stage} sequence {seq}")
  return record, arrays


def classify(
    directory: Path,
    expected_records: int,
    mismatch_capsule: Path | None,
    prefix_bounds: tuple[int, ...],
    expected_program_path: str,
    *,
    require_mismatch_join: bool = True,
    incident_min_prefix: int = 0,
    incident_max_prefix: int = 1 << 30,
) -> dict[str, Any]:
  _require(expected_records > 0, "expected_records must be positive")
  _require(expected_program_path in ("standard", "continue_decode"),
           "expected program path must be standard or continue_decode")
  _require(len(prefix_bounds) == expected_records + 1,
           "prefix-bound count must equal expected records plus one")
  _require(all(left < right for left, right in zip(
      prefix_bounds, prefix_bounds[1:])),
      "prefix bounds must be strictly increasing")
  _require(directory.is_dir(), f"capture directory does not exist: {directory}")
  pre_files = sorted(directory.glob("p38_serving_*_pre.json"))
  post_files = sorted(directory.glob("p38_serving_*_post.json"))
  _require(len(pre_files) == expected_records, f"expected {expected_records} pre records, found {len(pre_files)}")
  _require(len(post_files) == expected_records, f"expected {expected_records} post records, found {len(post_files)}")

  capsule = None
  if mismatch_capsule is not None and mismatch_capsule.is_file():
    capsule = _load_mismatch_capsule(mismatch_capsule)
  _require(
      capsule is not None or not require_mismatch_join,
      "required mismatch capsule is absent",
  )
  request_journal = _load_request_journal(
      directory, prefix_bounds, expected_program_path
  )
  incident_entries = _load_incident_ledger(
      directory,
      expected_program_path,
      incident_min_prefix,
      incident_max_prefix,
  )
  journal_joins = (
      _join_journal_to_capsule(request_journal, capsule)
      if capsule is not None else []
  )
  selected_source_rows = (
      {
          int(row) for row in np.asarray(
              capsule["arrays"]["selected_rows"]
          ).reshape(-1)
      }
      if capsule is not None else set()
  )
  journal_joined_source_rows = {
      int(join["source_row"]) for join in journal_joins
  }
  # The bounded request journal is a coarse path/shape witness and may not
  # coincide with a red row. Exact all-mismatch attribution belongs solely to
  # the per-call incident ledger below; requiring a journal coincidence would
  # reintroduce the selection bias this phase removes.
  incident_joins, missing_incident_joins = (
      _join_incident_to_capsule(incident_entries, capsule)
      if capsule is not None else ([], [])
  )
  _require(
      not require_mismatch_join or not missing_incident_joins,
      "incident ledger did not join every selected A/B mismatch: missing "
      f"{missing_incident_joins[:16]}",
  )
  summaries = []
  captured_strata: set[int] = set()
  successful_joins = 0
  implementation_identity = None
  source_commit = None
  for seq in range(expected_records):
    pre, pre_arrays = _load_stage(directory, seq, "pre")
    post, post_arrays = _load_stage(directory, seq, "post")
    _require(PRE_ARRAYS.issubset(pre_arrays), f"missing required pre arrays for sequence {seq}: {sorted(PRE_ARRAYS - set(pre_arrays))}")
    _require(POST_ARRAYS.issubset(post_arrays), f"missing required post arrays for sequence {seq}: {sorted(POST_ARRAYS - set(post_arrays))}")
    _require(
        any(name.startswith("sampling_leaf_") for name in pre_arrays),
        f"sequence {seq} has no sampling-metadata leaves",
    )

    meta = pre.get("meta", {})
    _require(meta.get("program_path") == expected_program_path,
             f"sequence {seq} captured the wrong program path")
    _validate_implementation_identity(meta, seq)
    current_identity = json.dumps(
        meta["implementation_identity"], sort_keys=True, separators=(",", ":")
    )
    if implementation_identity is None:
      implementation_identity = current_identity
    _require(current_identity == implementation_identity,
             f"sequence {seq} implementation identity drifted")
    current_source = meta.get("env", {}).get("CANON_EXPECT_COMMIT")
    _require(isinstance(current_source, str) and len(current_source) == 40 and
             all(character in "0123456789abcdef" for character in current_source),
             f"sequence {seq} source commit identity is invalid")
    if source_commit is None:
      source_commit = current_source
    _require(current_source == source_commit,
             f"sequence {seq} source commit identity drifted")
    requests = _validate_request_mapping(meta, pre_arrays, seq)
    _require(
        meta.get("continue_decode_enabled") is
        (expected_program_path == "continue_decode"),
        f"sequence {seq} continue-decode configuration contradicts its path",
    )
    _require(meta.get("caller_update_kv_cache") is True, f"sequence {seq} has an invalid caller cache-update contract")
    _require(
        meta.get("output_update_kv_cache") is (not bool(meta.get("kv_unified"))),
        f"sequence {seq} has an inconsistent output cache-update contract",
    )
    _require(meta.get("request_ids"), f"sequence {seq} has no request IDs")
    _require(requests, f"sequence {seq} has no scheduled request metadata")
    _require(all(item.get("block_ids") for item in requests), f"sequence {seq} has a request without physical page IDs")
    _require(all(item.get("token_ids") for item in requests), f"sequence {seq} has a request without token history")
    _require(meta.get("kv_caches_spec"), f"sequence {seq} has no KV-cache specification")
    _require(meta.get("block_size", 0) > 0, f"sequence {seq} has an invalid page size")
    _require(meta.get("capture_prefix_bounds") == list(prefix_bounds),
             f"sequence {seq} prefix bounds drifted")
    stratum_index = int(meta.get("capture_stratum_index", -1))
    _require(0 <= stratum_index < expected_records,
             f"sequence {seq} has an invalid capture stratum")
    _require(stratum_index not in captured_strata,
             f"sequence {seq} duplicates capture stratum {stratum_index}")
    captured_strata.add(stratum_index)
    expected_stratum = [
        prefix_bounds[stratum_index], prefix_bounds[stratum_index + 1]
    ]
    _require(meta.get("capture_stratum") == expected_stratum,
             f"sequence {seq} capture stratum bounds drifted")
    observed_min_prefix = int(meta.get("observed_min_prefix", -1))
    observed_prefix = int(meta.get("observed_max_prefix", -1))
    _require(0 <= observed_min_prefix <= observed_prefix,
             f"sequence {seq} observed prefix range is invalid")
    anchor_request_id = meta.get("capture_anchor_request_id")
    anchor_prefix = int(meta.get("capture_anchor_prefix", -1))
    anchors = [
        item for item in requests if item.get("request_id") == anchor_request_id
    ]
    _require(len(anchors) == 1,
             f"sequence {seq} capture anchor request is invalid")
    _require(int(anchors[0].get("num_computed_tokens", -1)) == anchor_prefix,
             f"sequence {seq} capture anchor prefix mapping drifted")
    _require(expected_stratum[0] <= anchor_prefix < expected_stratum[1],
             f"sequence {seq} anchor prefix is outside its capture stratum")
    _require(meta.get("capture_min_prefix") == prefix_bounds[0],
             f"sequence {seq} minimum prefix drifted")
    _require(all(meta.get("rpa_block_tuples", {}).get(name) for name in ("CANON_RPA_D", "CANON_RPA_P", "CANON_RPA_M")), f"sequence {seq} is missing a pinned RPA block tuple")

    post_meta = post.get("meta", {})
    _require(post_meta.get("program_path") == expected_program_path,
             f"sequence {seq} post record changed program path")
    actual_steps = int(post_meta.get("actual_steps", 0))
    _require(actual_steps > 0, f"sequence {seq} completed zero decode steps")
    _require(post_arrays["generated_tokens"].shape[0] == actual_steps, f"sequence {seq} generated-token step count mismatch")
    _require(post_arrays["logprob_values"].shape[0] == actual_steps, f"sequence {seq} logprob step count mismatch")
    _require(int(post_meta.get("completed_records", -1)) == seq + 1, f"sequence {seq} has a bad completed-record count")
    _require(int(post_meta.get("expected_max_records", -1)) >= expected_records, f"sequence {seq} reports an undersized record budget")

    mismatch_joins = []
    if capsule is not None:
      mismatch_joins = _join_mismatch_capsule(requests, capsule, seq)
      successful_joins += len(mismatch_joins)
    summaries.append({
        "seq": seq,
        "requests": len(requests),
        "actual_steps": actual_steps,
        "observed_max_prefix": observed_prefix,
        "observed_min_prefix": observed_min_prefix,
        "capture_anchor_request_id": anchor_request_id,
        "capture_anchor_prefix": anchor_prefix,
        "capture_stratum_index": stratum_index,
        "capture_stratum": expected_stratum,
        "program_path": expected_program_path,
        "kv_unified": bool(meta.get("kv_unified")),
        # Keep the singular field for old one-row fixtures while making the
        # complete per-row evidence explicit for production captures.
        "mismatch_join": (
            mismatch_joins[0] if len(mismatch_joins) == 1 else None
        ),
        "mismatch_joins": mismatch_joins,
    })

  _require(captured_strata == set(range(expected_records)),
           "capture strata are incomplete")
  # The four coarse serving records prove path/shape/implementation identity.
  # Exact mismatch-call attribution is now owned by the incident ledger, so a
  # mismatch is not required to coincide with one of the four old anchors.
  # Requiring such a coincidence recreated the selection bias P38.2l removes.
  _require(
      not require_mismatch_join or bool(incident_joins),
      "no exact incident call joins the mismatch capsule",
  )
  return {
      "schema_version": 1,
      "verdict": "PASS",
      "scope": "p38-serving-capture",
      "prefix_bounds": list(prefix_bounds),
      "program_path": expected_program_path,
      "successful_mismatch_joins": successful_joins,
      "request_journal_records": len(request_journal),
      "request_journal_joins": journal_joins,
      "request_journal_joined_source_rows": sorted(
          journal_joined_source_rows
      ),
      "incident_ledger_records": len(incident_entries),
      "incident_exact_joins": incident_joins,
      "incident_missing_joins": missing_incident_joins,
      "joined_source_rows": sorted({
          int(join["source_row"])
          for record in summaries
          for join in record["mismatch_joins"]
      }),
      "source_commit": source_commit,
      "records": summaries,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--expected-records", required=True, type=int)
  parser.add_argument("--expected-program-path", required=True)
  parser.add_argument("--mismatch-capsule", required=True, type=Path)
  parser.add_argument("--prefix-bounds", required=True)
  parser.add_argument("--require-mismatch-join", action="store_true")
  parser.add_argument("--incident-min-prefix", required=True, type=int)
  parser.add_argument("--incident-max-prefix", required=True, type=int)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  try:
    report = classify(
        args.directory,
        args.expected_records,
        args.mismatch_capsule,
        _parse_prefix_bounds(args.prefix_bounds),
        args.expected_program_path,
        require_mismatch_join=args.require_mismatch_join,
        incident_min_prefix=args.incident_min_prefix,
        incident_max_prefix=args.incident_max_prefix,
    )
  except CaptureError as error:
    report = {
        "schema_version": 1,
        "verdict": "INCONCLUSIVE",
        "scope": "p38-serving-capture",
        "error": str(error),
    }
    if args.output:
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(2) from error
  if args.output:
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
