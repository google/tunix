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

"""Fail-closed inputs and schedules for the P38 FrozenLake replay gate.

The P38 capsule records tokens and masks, but not the original serving
scheduler calls.  Consequently, the schedules in this module are explicitly
mask-derived counterfactuals.  They must never be reported as an exact replay
of the captured serving schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "p38-frozenlake-mismatch-capsule-v1"
SCHEDULE_PROVENANCE = "mask-derived-v1"
REQUIRED_ARRAYS = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_valid_mask",
    "action_mask",
    "s_decode",
    "s_prefill",
    "t_old",
    "policy_version",
    "sampling_values",
)


class P38ReplayError(RuntimeError):
  """Raised when a P38 replay input cannot satisfy its evidence contract."""


def _array_sha256(value: Any) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


@dataclass(frozen=True)
class CapsuleRow:
  """One hash-verified compact row selected by the target mismatch gate."""

  source_row: int
  prompt_ids: np.ndarray
  completion_ids: np.ndarray
  action_mask: np.ndarray
  s_decode: np.ndarray
  s_prefill: np.ndarray
  t_old: np.ndarray
  policy_version: np.ndarray
  sampling_values: np.ndarray

  @property
  def prompt_length(self) -> int:
    return int(self.prompt_ids.size)

  @property
  def completion_length(self) -> int:
    return int(self.completion_ids.size)


@dataclass(frozen=True)
class VerifiedCapsule:
  """A capsule whose transport-independent embedded hashes all match."""

  path: Path
  sha256: str
  metadata: Mapping[str, Any]
  rows: tuple[CapsuleRow, ...]


@dataclass(frozen=True)
class ReplayCall:
  """One fixed-M model invocation in a mask-derived replay schedule."""

  ordinal: int
  kind: str
  distribution: tuple[int, int, int]
  logical_start: int
  logical_stop: int
  kv_length_before: int
  kv_length_after: int
  action_targets: tuple[int, ...]

  @property
  def query_length(self) -> int:
    return self.logical_stop - self.logical_start

  def as_dict(self) -> dict[str, Any]:
    return {
        "ordinal": self.ordinal,
        "kind": self.kind,
        "request_distribution": list(self.distribution),
        "logical_start": self.logical_start,
        "logical_stop": self.logical_stop,
        "query_length": self.query_length,
        "kv_length_before": self.kv_length_before,
        "kv_length_after": self.kv_length_after,
        "action_targets": list(self.action_targets),
    }


@dataclass(frozen=True)
class ReplaySchedule:
  """A validated sequence of calls and its explicit provenance ceiling."""

  arm: str
  provenance: str
  source_row: int
  prompt_length: int
  completion_length: int
  local_m: int
  logical_input_ids: np.ndarray
  action_mask: np.ndarray
  calls: tuple[ReplayCall, ...]

  def as_dict(self) -> dict[str, Any]:
    return {
        "arm": self.arm,
        "provenance": self.provenance,
        "source_row": self.source_row,
        "prompt_length": self.prompt_length,
        "completion_length": self.completion_length,
        "local_m": self.local_m,
        "logical_input_sha256": _array_sha256(self.logical_input_ids),
        "action_targets": np.flatnonzero(self.action_mask).tolist(),
        "calls": [call.as_dict() for call in self.calls],
    }


def _require_bool_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
  array = np.asarray(value)
  if array.shape != shape:
    raise P38ReplayError(
        f"{name} shape mismatch: expected {shape}, observed {array.shape}"
    )
  if array.dtype != np.bool_:
    raise P38ReplayError(f"{name} must have bool dtype, observed {array.dtype}")
  return np.ascontiguousarray(array)


def _compact_prompt(ids: np.ndarray, mask: np.ndarray, row: int) -> np.ndarray:
  valid = np.flatnonzero(mask)
  if valid.size == 0:
    raise P38ReplayError(f"capsule row {row} has no valid prompt token")
  expected = np.arange(valid[0], valid[-1] + 1, dtype=valid.dtype)
  if not np.array_equal(valid, expected):
    raise P38ReplayError(f"prompt mask contains a hole at capsule row {row}")
  return np.ascontiguousarray(ids[valid])


def _completion_length(mask: np.ndarray, row: int) -> int:
  length = int(mask.sum())
  expected = np.arange(mask.size, dtype=np.int64) < length
  if not np.array_equal(mask, expected):
    raise P38ReplayError(
        f"completion validity mask is not a contiguous prefix at capsule row {row}"
    )
  if length == 0:
    raise P38ReplayError(f"capsule row {row} has no valid completion token")
  return length


def load_verified_capsule(path: str | Path) -> VerifiedCapsule:
  """Loads a capsule and verifies its schema, arrays, and embedded hashes."""
  source = Path(path)
  if not source.is_file():
    raise P38ReplayError(f"P38 capsule does not exist: {source}")
  try:
    with np.load(source, allow_pickle=False) as archive:
      files = set(archive.files)
      missing = set(REQUIRED_ARRAYS) - files
      if missing:
        raise P38ReplayError(f"P38 capsule arrays are missing: {sorted(missing)}")
      if "metadata_json" not in files or "selected_rows" not in files:
        raise P38ReplayError("P38 capsule metadata or selected rows are missing")
      metadata = json.loads(archive["metadata_json"].tobytes())
      if metadata.get("schema") != SCHEMA:
        raise P38ReplayError(
            f"unexpected P38 capsule schema: {metadata.get('schema')!r}"
        )
      arrays = {
          name: np.ascontiguousarray(archive[name]) for name in REQUIRED_ARRAYS
      }
      selected_rows = np.asarray(archive["selected_rows"])
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    if isinstance(exc, P38ReplayError):
      raise
    raise P38ReplayError(f"cannot read P38 capsule {source}: {exc}") from exc

  metadata_arrays = metadata.get("arrays")
  if not isinstance(metadata_arrays, dict):
    raise P38ReplayError("P38 capsule metadata has no array hash table")
  for name, value in arrays.items():
    expected = metadata_arrays.get(name)
    if not isinstance(expected, dict):
      raise P38ReplayError(f"P38 capsule metadata is missing array {name}")
    observed_shape = list(value.shape)
    observed_dtype = str(value.dtype)
    observed_sha = _array_sha256(value)
    if expected.get("shape") != observed_shape:
      raise P38ReplayError(
          f"P38 capsule array shape mismatch for {name}: "
          f"{observed_shape}/{expected.get('shape')}"
      )
    if expected.get("dtype") != observed_dtype:
      raise P38ReplayError(
          f"P38 capsule array dtype mismatch for {name}: "
          f"{observed_dtype}/{expected.get('dtype')}"
      )
    if expected.get("sha256") != observed_sha:
      raise P38ReplayError(f"P38 capsule array hash mismatch: {name}")

  if selected_rows.ndim != 1 or selected_rows.dtype.kind not in "iu":
    raise P38ReplayError("P38 selected_rows must be a rank-1 integer array")
  selected = tuple(int(value) for value in selected_rows)
  if selected != tuple(metadata.get("selected_rows", ())):
    raise P38ReplayError("P38 selected_rows disagree with capsule metadata")
  if len(set(selected)) != len(selected):
    raise P38ReplayError("P38 selected_rows contain duplicates")
  batch = len(selected)
  if batch == 0:
    raise P38ReplayError("P38 capsule contains no selected row")
  bad_batch = {
      name: value.shape for name, value in arrays.items()
      if value.ndim == 0 or value.shape[0] != batch
  }
  if bad_batch:
    raise P38ReplayError(f"P38 capsule arrays are not batch aligned: {bad_batch}")

  prompt_ids = arrays["prompt_ids"]
  completion_ids = arrays["completion_ids"]
  if prompt_ids.ndim != 2 or completion_ids.ndim != 2:
    raise P38ReplayError("P38 prompt and completion token arrays must be rank 2")
  prompt_mask = _require_bool_array(
      "prompt_mask", arrays["prompt_mask"], prompt_ids.shape
  )
  completion_valid = _require_bool_array(
      "completion_valid_mask",
      arrays["completion_valid_mask"],
      completion_ids.shape,
  )
  action_mask = _require_bool_array(
      "action_mask", arrays["action_mask"], completion_ids.shape
  )
  if np.any(action_mask & ~completion_valid):
    raise P38ReplayError("P38 action mask includes an invalid completion token")

  rows = []
  for index, source_row in enumerate(selected):
    compact_prompt = _compact_prompt(prompt_ids[index], prompt_mask[index], index)
    length = _completion_length(completion_valid[index], index)
    compact_action = np.ascontiguousarray(action_mask[index, :length])
    if not compact_action.any():
      raise P38ReplayError(f"capsule row {index} has no action target")
    per_completion = {}
    for name in ("s_decode", "s_prefill", "t_old"):
      value = arrays[name][index]
      if value.shape != completion_ids[index].shape:
        raise P38ReplayError(
            f"{name} is not completion-aligned at capsule row {index}"
        )
      per_completion[name] = np.ascontiguousarray(value[:length])
    rows.append(CapsuleRow(
        source_row=source_row,
        prompt_ids=compact_prompt,
        completion_ids=np.ascontiguousarray(completion_ids[index, :length]),
        action_mask=compact_action,
        s_decode=per_completion["s_decode"],
        s_prefill=per_completion["s_prefill"],
        t_old=per_completion["t_old"],
        policy_version=np.ascontiguousarray(arrays["policy_version"][index]),
        sampling_values=np.ascontiguousarray(arrays["sampling_values"][index]),
    ))
  return VerifiedCapsule(
      path=source,
      sha256=_file_sha256(source),
      metadata=metadata,
      rows=tuple(rows),
  )


def _targets_for_span(
    *,
    prompt_length: int,
    action_mask: np.ndarray,
    logical_start: int,
    logical_stop: int,
) -> tuple[int, ...]:
  targets = []
  for completion_index in np.flatnonzero(action_mask):
    predictor_position = prompt_length + int(completion_index) - 1
    if logical_start <= predictor_position < logical_stop:
      targets.append(int(completion_index))
  return tuple(targets)


def _append_call(
    calls: list[ReplayCall],
    *,
    kind: str,
    distribution: tuple[int, int, int],
    start: int,
    stop: int,
    prompt_length: int,
    action_mask: np.ndarray,
    local_m: int,
) -> None:
  if stop <= start or stop - start > local_m:
    raise P38ReplayError(
        f"invalid {kind} query span [{start}, {stop}) for local M={local_m}"
    )
  calls.append(ReplayCall(
      ordinal=len(calls),
      kind=kind,
      distribution=distribution,
      logical_start=start,
      logical_stop=stop,
      kv_length_before=start,
      kv_length_after=stop,
      action_targets=_targets_for_span(
          prompt_length=prompt_length,
          action_mask=action_mask,
          logical_start=start,
          logical_stop=stop,
      ),
  ))


def _append_initial_prompt(
    calls: list[ReplayCall], row: CapsuleRow, local_m: int
) -> None:
  start = 0
  while start < row.prompt_length:
    stop = min(start + local_m, row.prompt_length)
    if start == 0:
      kind = "initial_prefill"
      distribution = (0, 1, 1)
    else:
      kind = "continued_prefill"
      distribution = (0, 0, 1)
    _append_call(
        calls,
        kind=kind,
        distribution=distribution,
        start=start,
        stop=stop,
        prompt_length=row.prompt_length,
        action_mask=row.action_mask,
        local_m=local_m,
    )
    start = stop


def _validate_schedule(schedule: ReplaySchedule) -> ReplaySchedule:
  expected_targets = tuple(int(value) for value in np.flatnonzero(schedule.action_mask))
  observed_targets = tuple(
      target for call in schedule.calls for target in call.action_targets
  )
  if sorted(observed_targets) != sorted(expected_targets):
    raise P38ReplayError(
        "replay calls do not cover each action predictor exactly once: "
        f"observed={observed_targets} expected={expected_targets}"
    )
  if len(set(observed_targets)) != len(observed_targets):
    raise P38ReplayError("an action predictor is covered by multiple replay calls")
  expected_start = 0
  for call in schedule.calls:
    if call.logical_start != expected_start:
      raise P38ReplayError(
          f"replay calls contain a logical gap before call {call.ordinal}"
      )
    if call.kv_length_before != call.logical_start:
      raise P38ReplayError(f"call {call.ordinal} has inconsistent KV provenance")
    expected_start = call.logical_stop
  if expected_start != schedule.logical_input_ids.size:
    raise P38ReplayError(
        "replay calls do not consume the full predictor-token sequence"
    )
  return schedule


def build_r0_mask_derived_schedule(
    row: CapsuleRow, *, local_m: int = 256
) -> ReplaySchedule:
  """Builds the multi-turn counterfactual from action and validity masks."""
  if local_m <= 0:
    raise P38ReplayError(f"local M must be positive, got {local_m}")
  logical_inputs = np.concatenate((row.prompt_ids, row.completion_ids[:-1]))
  calls: list[ReplayCall] = []
  _append_initial_prompt(calls, row, local_m)
  completion_input = 0
  while completion_input < row.completion_length - 1:
    logical_start = row.prompt_length + completion_input
    if row.action_mask[completion_input]:
      _append_call(
          calls,
          kind="decode",
          distribution=(1, 1, 1),
          start=logical_start,
          stop=logical_start + 1,
          prompt_length=row.prompt_length,
          action_mask=row.action_mask,
          local_m=local_m,
      )
      completion_input += 1
      continue
    run_stop = completion_input + 1
    while (
        run_stop < row.completion_length - 1
        and not row.action_mask[run_stop]
    ):
      run_stop += 1
    while completion_input < run_stop:
      chunk_stop = min(completion_input + local_m, run_stop)
      _append_call(
          calls,
          kind="environment_prefill",
          distribution=(0, 0, 1),
          start=row.prompt_length + completion_input,
          stop=row.prompt_length + chunk_stop,
          prompt_length=row.prompt_length,
          action_mask=row.action_mask,
          local_m=local_m,
      )
      completion_input = chunk_stop
  return _validate_schedule(ReplaySchedule(
      arm="R0",
      provenance=SCHEDULE_PROVENANCE,
      source_row=row.source_row,
      prompt_length=row.prompt_length,
      completion_length=row.completion_length,
      local_m=local_m,
      logical_input_ids=np.ascontiguousarray(logical_inputs),
      action_mask=np.ascontiguousarray(row.action_mask),
      calls=tuple(calls),
  ))


def build_r1_continuous_decode_schedule(
    row: CapsuleRow, *, local_m: int = 256
) -> ReplaySchedule:
  """Builds the same-token, same-depth, continuous-decode counterfactual."""
  if local_m <= 0:
    raise P38ReplayError(f"local M must be positive, got {local_m}")
  logical_inputs = np.concatenate((row.prompt_ids, row.completion_ids[:-1]))
  calls: list[ReplayCall] = []
  _append_initial_prompt(calls, row, local_m)
  for completion_input in range(row.completion_length - 1):
    logical_start = row.prompt_length + completion_input
    _append_call(
        calls,
        kind="continuous_decode",
        distribution=(1, 1, 1),
        start=logical_start,
        stop=logical_start + 1,
        prompt_length=row.prompt_length,
        action_mask=row.action_mask,
        local_m=local_m,
    )
  return _validate_schedule(ReplaySchedule(
      arm="R1",
      provenance=SCHEDULE_PROVENANCE,
      source_row=row.source_row,
      prompt_length=row.prompt_length,
      completion_length=row.completion_length,
      local_m=local_m,
      logical_input_ids=np.ascontiguousarray(logical_inputs),
      action_mask=np.ascontiguousarray(row.action_mask),
      calls=tuple(calls),
  ))


def build_fixed_chunk_reference_schedule(
    row: CapsuleRow, *, local_m: int = 256
) -> ReplaySchedule:
  """Builds the unchanged adapter reference schedule with bounded observers."""
  if local_m <= 0:
    raise P38ReplayError(f"local M must be positive, got {local_m}")
  logical_inputs = np.concatenate((row.prompt_ids, row.completion_ids))
  calls: list[ReplayCall] = []
  start = 0
  while start < logical_inputs.size:
    stop = min(start + local_m, logical_inputs.size)
    _append_call(
        calls,
        kind="fixed_chunk_reference",
        distribution=(0, 0, 1),
        start=start,
        stop=stop,
        prompt_length=row.prompt_length,
        action_mask=row.action_mask,
        local_m=local_m,
    )
    start = stop
  return _validate_schedule(ReplaySchedule(
      arm="REF",
      provenance="canonical-fixed-chunk-v1",
      source_row=row.source_row,
      prompt_length=row.prompt_length,
      completion_length=row.completion_length,
      local_m=local_m,
      logical_input_ids=np.ascontiguousarray(logical_inputs),
      action_mask=np.ascontiguousarray(row.action_mask),
      calls=tuple(calls),
  ))


def schedules_report(
    capsule: VerifiedCapsule,
    schedules: Sequence[ReplaySchedule],
) -> dict[str, Any]:
  """Builds a JSON-safe admission report without claiming a TPU result."""
  if not schedules:
    raise P38ReplayError("at least one replay schedule is required")
  return {
      "schema": "p38-frozenlake-replay-schedule-v1",
      "verdict": "LOCALLY_ADMITTED",
      "tpu_status": "NOT_RUN",
      "capsule": {
          "path": str(capsule.path),
          "sha256": capsule.sha256,
          "schema": capsule.metadata.get("schema"),
          "selected_rows": [row.source_row for row in capsule.rows],
      },
      "claim_ceiling": (
          "Schedules are derived from token masks and do not reproduce the "
          "captured serving scheduler metadata."
      ),
      "schedules": [schedule.as_dict() for schedule in schedules],
  }


def build_engine_records(
    schedule: ReplaySchedule,
    *,
    max_num_reqs: int,
    blocks_per_request: int,
    cache_blocks: int,
) -> tuple[dict[str, Any], ...]:
  """Lowers one DP1 schedule to fixed-M engine metadata records."""
  if max_num_reqs <= 0 or blocks_per_request <= 0:
    raise P38ReplayError("engine request and page-table sizes must be positive")
  if blocks_per_request > cache_blocks:
    raise P38ReplayError(
        "replay page table exceeds the independently allocated cache: "
        f"{blocks_per_request}>{cache_blocks}"
    )
  page_table = np.zeros(
      (max_num_reqs, blocks_per_request), dtype=np.int32
  )
  page_table[0] = np.arange(blocks_per_request, dtype=np.int32)
  records = []
  for call in schedule.calls:
    input_ids = np.zeros((schedule.local_m,), dtype=np.int32)
    positions = np.zeros((schedule.local_m,), dtype=np.int32)
    query_ids = schedule.logical_input_ids[
        call.logical_start : call.logical_stop
    ]
    input_ids[: call.query_length] = query_ids
    positions[: call.query_length] = np.arange(
        call.logical_start, call.logical_stop, dtype=np.int32
    )
    query_start = np.full(
        (max_num_reqs + 1,), call.query_length, dtype=np.int32
    )
    query_start[0] = 0
    seq_lens = np.zeros((max_num_reqs,), dtype=np.int32)
    seq_lens[0] = call.kv_length_after
    records.append({
        "arm": schedule.arm,
        "meta": {
            "md_padded_num_reqs": max_num_reqs,
            "schedule_provenance": schedule.provenance,
            "call": call.as_dict(),
        },
        "arrays": {
            "input_ids": input_ids,
            "input_positions": positions,
            "md_input_positions": positions.copy(),
            "md_block_tables": page_table.reshape(-1).copy(),
            "md_seq_lens": seq_lens,
            "md_query_start_loc": query_start,
            "md_request_distribution": np.asarray(
                call.distribution, dtype=np.int32
            ),
        },
    })
  return tuple(records)
