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

"""Fail-closed four-boundary observability for zero-TIM integration gates.

This module is deliberately host-side.  ``ObservedTrainExample`` carries the
three frozen boundaries next to a normal TrainExample while batches are merged
and sliced.  ``rl.trainer.Trainer._prepare_inputs`` removes the wrapper before
sharding/JIT, so diagnostic arrays cannot alter the train-program signature.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import flax
import numpy as np


ALIGN_ENV = "CANON_ALIGNMENT_GATE"
GATE_ONLY_ENV = "CANON_ALIGNMENT_GATE_ONLY"
UPDATE_CANARY_ENV = "CANON_ALIGNMENT_UPDATE_CANARY"
TRAIN_ENV = "CANON_ALIGNMENT_TRAIN"
REPORT_ENV = "CANON_ALIGN_REPORT"
PRE_GATE_ENV = "CANON_PRE_ALIGN_GATE"
PRE_REPORT_ENV = "CANON_PRE_ALIGN_REPORT"
PRECHECK_ONLY_ENV = "CANON_P38_PRECHECK_ONLY"
_MAX_MISMATCH_DETAILS = 1024


class AlignmentGateError(RuntimeError):
  """Raised when an alignment run is incomplete or numerically red."""


class PreAlignmentProbeComplete(RuntimeError):
  """Raised after an exact P38 precheck to stop before backward."""


@flax.struct.dataclass(frozen=True)
class ObservedTrainExample:
  """Host-only observability wrapper; never pass this object to JIT."""

  train_example: Any
  s_decode: Any
  s_prefill: Any
  t_old: Any
  action_mask: Any
  completion_valid_mask: Any
  prompt_mask: Any
  tokens: Any
  policy_version: Any
  sampling_values: Any
  source_name: str = flax.struct.field(
      pytree_node=False, default="VllmRollout.get_prefill_rescore_logps"
  )

  # AgenticRLLearner reads these attributes before Trainer unwraps the object.
  @property
  def completion_ids(self):
    return self.train_example.completion_ids

  @property
  def completion_mask(self):
    return self.train_example.completion_mask

  @property
  def advantages(self):
    return self.train_example.advantages

  @property
  def is_update_step(self):
    return self.train_example.is_update_step


def enabled() -> bool:
  return os.environ.get(ALIGN_ENV, "") == "1"


def precheck_enabled() -> bool:
  """Returns whether the pre-backward value-boundary gate is enabled."""
  return os.environ.get(PRE_GATE_ENV, "") == "1"


def precheck_only_enabled() -> bool:
  """Return the fail-closed P38 diagnostic stop policy."""
  value = os.environ.get(PRECHECK_ONLY_ENV, "")
  if value not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{PRECHECK_ONLY_ENV} must be exactly 0 or 1, got {value!r}"
    )
  return value == "1"


def stop_after_exact_precheck(record: dict[str, Any]) -> None:
  """Stop a P38 diagnostic after its durable exact record."""
  if not precheck_only_enabled():
    return
  if record.get("verdict") != "PASS":
    raise AlignmentGateError(
        "P38 precheck-only stop requires a passing pre-backward record"
    )
  print(
      "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD "
      f"step={record.get('step')} N_action={record.get('N_action')}",
      flush=True,
  )
  raise PreAlignmentProbeComplete(
      "P38 precheck-only diagnostic completed before backward"
  )


def execution_mode() -> str:
  """Return the single admitted alignment execution mode.

  ``gate-only`` is the release observability path and cannot mutate model or
  optimizer state. ``update-canary`` is a one-step, no-checkpoint systems test
  whose mutation is confined to the ephemeral process.  Requiring exactly one
  mode prevents an unset/empty Docker environment variable from silently
  changing the safety contract.
  """
  modes = {
      "gate-only": os.environ.get(GATE_ONLY_ENV, "") == "1",
      "update-canary": os.environ.get(UPDATE_CANARY_ENV, "") == "1",
      "train": os.environ.get(TRAIN_ENV, "") == "1",
  }
  enabled_modes = [name for name, is_enabled in modes.items() if is_enabled]
  if len(enabled_modes) != 1:
    raise AlignmentGateError(
        "exactly one alignment execution mode is required: "
        f"{GATE_ONLY_ENV}=1, {UPDATE_CANARY_ENV}=1, or {TRAIN_ENV}=1"
    )
  return enabled_modes[0]


def wrap_train_example(
    train_example: Any,
    *,
    s_decode: Any,
    s_prefill: Any,
    t_old: Any,
    action_mask: Any,
    completion_valid_mask: Any | None = None,
    prompt_mask: Any | None = None,
    tokens: Any,
    policy_version: Any,
    temperature: float,
    top_k: int,
    top_p: float,
    s_prefill_source: Any,
) -> ObservedTrainExample:
  """Validate real-rescore provenance and create a merge/slice-safe wrapper."""
  if not getattr(s_prefill_source, "is_real_rescore", False):
    raise AlignmentGateError(
        "S_prefill producer does not declare is_real_rescore=True; refusing "
        "a cached-decode alias"
    )
  sd = np.asarray(s_decode)
  sp = np.asarray(s_prefill)
  to = np.asarray(t_old)
  mask = np.asarray(action_mask)
  expected = tuple(np.shape(train_example.completion_ids))
  completion_valid = np.asarray(
      action_mask if completion_valid_mask is None else completion_valid_mask,
      dtype=np.bool_,
  )
  if prompt_mask is None:
    prompt_valid = np.zeros((expected[0], 0), dtype=np.bool_)
  else:
    prompt_valid = np.asarray(prompt_mask, dtype=np.bool_)
  tok = np.asarray(tokens)
  for name, value in (
      ("S_decode", sd),
      ("S_prefill", sp),
      ("T_old", to),
      ("action_mask", mask),
      ("completion_valid_mask", completion_valid),
      ("tokens", tok),
  ):
    if tuple(value.shape) != expected:
      raise AlignmentGateError(
          f"{name} shape {value.shape} != completion shape {expected}"
      )
  if prompt_valid.ndim != 2 or prompt_valid.shape[0] != expected[0]:
    raise AlignmentGateError(
        "prompt_mask must be rank two and batch-aligned with completions: "
        f"{prompt_valid.shape} vs {expected}"
    )
  if np.any(mask.astype(np.bool_) & ~completion_valid):
    raise AlignmentGateError(
        "action_mask must be a subset of completion_valid_mask"
    )
  if np.shares_memory(sd, sp) or s_decode is s_prefill:
    raise AlignmentGateError(
        "S_prefill aliases S_decode; the decode-vs-rescore gate would be vacuous"
    )
  return ObservedTrainExample(
      train_example=train_example,
      s_decode=sd.copy(),
      s_prefill=sp.copy(),
      t_old=to.copy(),
      action_mask=mask.astype(np.bool_, copy=True),
      completion_valid_mask=completion_valid.copy(),
      prompt_mask=prompt_valid.copy(),
      tokens=tok.copy(),
      policy_version=np.asarray(policy_version).copy(),
      sampling_values=np.repeat(
          np.asarray(
              [[temperature, float(top_k), top_p]], dtype=np.float32
          ),
          expected[0],
          axis=0,
      ),
  )


def unwrap_train_example(value: Any) -> tuple[Any, ObservedTrainExample | None]:
  if isinstance(value, ObservedTrainExample):
    return value.train_example, value
  return value, None


def _hash(value: Any) -> str:
  array = np.ascontiguousarray(np.asarray(value))
  return hashlib.sha256(array.tobytes()).hexdigest()


def _masked_hash(value: Any, mask: Any) -> str:
  array = np.asarray(value)
  bool_mask = np.asarray(mask, dtype=np.bool_)
  if array.shape != bool_mask.shape:
    return "INVALID_SHAPE"
  return _hash(np.ascontiguousarray(array[bool_mask]))


def _scalar_bits(value: Any, dtype: np.dtype) -> tuple[int | None, str | None]:
  """Returns the exact in-memory scalar bits as an integer and hex string."""
  unsigned = {
      1: np.uint8,
      2: np.uint16,
      4: np.uint32,
      8: np.uint64,
  }.get(dtype.itemsize)
  if unsigned is None:
    return None, None
  scalar = np.asarray([value], dtype=dtype)
  bits = int(scalar.view(unsigned)[0])
  return bits, f"0x{bits:0{dtype.itemsize * 2}x}"


def _float_ulp_distance(a_bits: int, b_bits: int, bit_width: int) -> int:
  """Returns ordered-representation distance for two IEEE floating values."""
  sign_mask = 1 << (bit_width - 1)
  value_mask = (1 << bit_width) - 1

  def ordered(bits: int) -> int:
    if bits & sign_mask:
      return (~bits) & value_mask
    return bits | sign_mask

  return abs(ordered(a_bits) - ordered(b_bits))


def _json_number(value: Any) -> float | str:
  """Returns a strict-JSON representation without losing nonfinite state."""
  number = float(value)
  if np.isnan(number):
    return "nan"
  if np.isposinf(number):
    return "inf"
  if np.isneginf(number):
    return "-inf"
  return number


def _mismatch_detail(
    av: np.ndarray,
    bv: np.ndarray,
    coordinates: np.ndarray,
    byte_diff_by_element: np.ndarray,
    masked_index: int,
) -> dict[str, Any]:
  """Builds one JSON-safe exact-value record for a masked mismatch."""
  coordinate = tuple(int(value) for value in coordinates[masked_index])
  a_value = av[masked_index]
  b_value = bv[masked_index]
  abs_delta = abs(np.float64(a_value) - np.float64(b_value))
  a_bits, a_bits_hex = _scalar_bits(a_value, av.dtype)
  b_bits, b_bits_hex = _scalar_bits(b_value, bv.dtype)
  detail = {
      "masked_index": int(masked_index),
      "coordinate": list(coordinate),
      "a": _json_number(a_value),
      "b": _json_number(b_value),
      "abs_delta": _json_number(abs_delta),
      "a_bits": a_bits_hex,
      "b_bits": b_bits_hex,
      "xor_bits": (
          f"0x{(a_bits ^ b_bits):0{av.dtype.itemsize * 2}x}"
          if a_bits is not None and b_bits is not None
          else None
      ),
      "differing_byte_offsets": [
          int(value)
          for value in np.flatnonzero(byte_diff_by_element[masked_index])
      ],
      "ulp_distance": None,
  }
  if len(coordinate) == 2:
    detail.update({
        "sequence_row": coordinate[0],
        "completion_position": coordinate[1],
    })
  if (
      a_bits is not None
      and b_bits is not None
      and av.dtype.kind == "f"
      and np.isfinite(a_value)
      and np.isfinite(b_value)
  ):
    detail["ulp_distance"] = _float_ulp_distance(
        a_bits, b_bits, av.dtype.itemsize * 8
    )
  return detail


def _masked_bitwise_difference(a: Any, b: Any, mask: Any) -> dict[str, Any]:
  """Returns byte- and element-level bitwise differences under ``mask``."""
  aa = np.asarray(a)
  bb = np.asarray(b)
  mm = np.asarray(mask, dtype=np.bool_)
  if aa.shape != bb.shape or aa.dtype != bb.dtype or aa.shape != mm.shape:
    return {
        "valid": False,
        "differing_bytes": -1,
        "total_bytes": -1,
        "byte_fraction": None,
        "differing_elements": -1,
        "total_elements": -1,
        "element_fraction": None,
        "first_mismatch": None,
        "mismatches": [],
        "reported_mismatches": 0,
        "mismatches_truncated": False,
    }

  av = np.ascontiguousarray(aa[mm]).reshape(-1)
  bv = np.ascontiguousarray(bb[mm]).reshape(-1)
  byte_diff = (av.view(np.uint8) != bv.view(np.uint8)).reshape(-1)
  differing_bytes = int(byte_diff.sum())
  total_bytes = int(av.nbytes)
  total_elements = int(av.size)
  byte_diff_by_element = byte_diff.reshape(total_elements, av.dtype.itemsize)
  element_diff = byte_diff_by_element.any(axis=1)
  differing_elements = int(element_diff.sum())
  coordinates = np.argwhere(mm)
  mismatch_indices = np.flatnonzero(element_diff)
  reported_indices = mismatch_indices[:_MAX_MISMATCH_DETAILS]
  mismatches = [
      _mismatch_detail(
          av,
          bv,
          coordinates,
          byte_diff_by_element,
          int(index),
      )
      for index in reported_indices
  ]
  first = mismatches[0] if mismatches else None
  return {
      "valid": True,
      "differing_bytes": differing_bytes,
      "total_bytes": total_bytes,
      "byte_fraction": (
          float(differing_bytes / total_bytes) if total_bytes else 0.0
      ),
      "differing_elements": differing_elements,
      "total_elements": total_elements,
      "element_fraction": (
          float(differing_elements / total_elements) if total_elements else 0.0
      ),
      "first_mismatch": first,
      "mismatches": mismatches,
      "reported_mismatches": len(mismatches),
      "mismatches_truncated": differing_elements > len(mismatches),
  }


def _attach_tokens(
    difference: dict[str, Any], tokens: Any, expected_shape: tuple[int, ...]
) -> None:
  """Adds token ids to localized records when the sidecar shape is valid."""
  token_array = np.asarray(tokens)
  if token_array.shape != expected_shape:
    return
  for detail in difference.get("mismatches", []):
    coordinate = tuple(detail.get("coordinate", ()))
    if len(coordinate) == token_array.ndim:
      detail["token_id"] = int(token_array[coordinate])
  first = difference.get("first_mismatch")
  if first is not None and difference.get("mismatches"):
    difference["first_mismatch"] = difference["mismatches"][0]


def _attach_sequence_context(
    difference: dict[str, Any],
    *,
    prompt_mask: Any,
    completion_valid_mask: Any,
    action_mask: Any,
    chunk_size: int = 256,
) -> None:
  """Attach logical turn, chunk, and KV coordinates to mismatch records."""
  prompt = np.asarray(prompt_mask, dtype=np.bool_)
  valid = np.asarray(completion_valid_mask, dtype=np.bool_)
  action = np.asarray(action_mask, dtype=np.bool_)
  if (
      prompt.ndim != 2
      or valid.ndim != 2
      or action.shape != valid.shape
      or prompt.shape[0] != valid.shape[0]
      or chunk_size <= 0
  ):
    return

  prompt_lengths = prompt.sum(axis=1, dtype=np.int64)
  valid_lengths = valid.sum(axis=1, dtype=np.int64)
  action_starts = action & np.concatenate(
      (np.ones((action.shape[0], 1), dtype=np.bool_), ~action[:, :-1]),
      axis=1,
  )
  turn_indices = np.cumsum(action_starts, axis=1, dtype=np.int64) - 1

  for detail in difference.get("mismatches", []):
    coordinate = tuple(detail.get("coordinate", ()))
    if len(coordinate) != 2:
      continue
    row, position = coordinate
    if (
        row < 0
        or row >= valid.shape[0]
        or position < 0
        or position >= valid.shape[1]
    ):
      continue
    prompt_length = int(prompt_lengths[row])
    logical_position = prompt_length + position
    current_action_start = bool(action_starts[row, position])
    previous_starts = np.flatnonzero(action_starts[row, : position + 1])
    action_run_start = (
        int(previous_starts[-1]) if previous_starts.size else None
    )
    detail.update({
        "prompt_length": prompt_length,
        "completion_valid_length": int(valid_lengths[row]),
        "logical_kv_prefix_length": logical_position,
        "completion_chunk_index": int(position // chunk_size),
        "sequence_chunk_index": int(logical_position // chunk_size),
        "offset_in_sequence_chunk": int(logical_position % chunk_size),
        "distance_to_next_sequence_chunk": int(
            (-logical_position) % chunk_size
        ),
        "turn_index": (
            int(turn_indices[row, position])
            if action[row, position] and turn_indices[row, position] >= 0
            else None
        ),
        "action_run_start": current_action_start,
        "action_run_end": bool(
            action[row, position]
            and (
                position + 1 >= action.shape[1]
                or not action[row, position + 1]
            )
        ),
        "offset_in_action_run": (
            int(position - action_run_start)
            if action[row, position] and action_run_start is not None
            else None
        ),
        "previous_token_is_environment": bool(
            position > 0
            and valid[row, position - 1]
            and not action[row, position - 1]
        ),
    })
  if difference.get("mismatches"):
    difference["first_mismatch"] = difference["mismatches"][0]


def _max_abs_mismatch(a: Any, b: Any, mask: Any) -> dict[str, Any] | None:
  """Returns an exact record for the largest numerical masked mismatch."""
  aa = np.asarray(a)
  bb = np.asarray(b)
  mm = np.asarray(mask, dtype=np.bool_)
  if aa.shape != bb.shape or aa.dtype != bb.dtype or aa.shape != mm.shape:
    return None
  av = np.ascontiguousarray(aa[mm]).reshape(-1)
  bv = np.ascontiguousarray(bb[mm]).reshape(-1)
  if not av.size:
    return None
  byte_diff_by_element = (
      av.view(np.uint8) != bv.view(np.uint8)
  ).reshape(av.size, av.dtype.itemsize)
  mismatch_indices = np.flatnonzero(byte_diff_by_element.any(axis=1))
  if not mismatch_indices.size:
    return None
  deltas = np.abs(
      av[mismatch_indices].astype(np.float64)
      - bv[mismatch_indices].astype(np.float64)
  )
  masked_index = int(mismatch_indices[int(np.argmax(deltas))])
  return _mismatch_detail(
      av,
      bv,
      np.argwhere(mm),
      byte_diff_by_element,
      masked_index,
  )


def _report_sha256(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as report_file:
    for chunk in iter(lambda: report_file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _masked_bytes_differ(a: Any, b: Any, mask: Any) -> tuple[int, dict | None]:
  """Compatibility wrapper for callers that only consume the legacy fields."""
  result = _masked_bitwise_difference(a, b, mask)
  return result["differing_bytes"], result["first_mismatch"]


def check_pre_backward(
    sidecar: ObservedTrainExample,
    *,
    step: int,
    fail_closed: bool = True,
) -> dict[str, Any]:
  """Checks decode, engine-prefill and trainer-old values before backward."""
  if not precheck_enabled():
    raise AlignmentGateError(
        f"pre-backward alignment requires {PRE_GATE_ENV}=1"
    )
  sd = np.asarray(sidecar.s_decode)
  sp = np.asarray(sidecar.s_prefill)
  to = np.asarray(sidecar.t_old)
  mask = np.asarray(sidecar.action_mask, dtype=np.bool_)
  n_action = int(mask.sum())
  reds: list[str] = []
  if n_action == 0:
    reds.append("N_action=0")
  boundaries = {}
  for name, a, b in (
      ("S_decode_vs_S_prefill", sd, sp),
      ("S_prefill_vs_T_old", sp, to),
  ):
    difference = _masked_bitwise_difference(a, b, mask)
    _attach_tokens(difference, sidecar.tokens, mask.shape)
    _attach_sequence_context(
        difference,
        prompt_mask=sidecar.prompt_mask,
        completion_valid_mask=sidecar.completion_valid_mask,
        action_mask=sidecar.action_mask,
    )
    max_abs = None
    max_abs_mismatch = _max_abs_mismatch(a, b, mask)
    if max_abs_mismatch is not None:
      coordinate = tuple(max_abs_mismatch.get("coordinate", ()))
      token_array = np.asarray(sidecar.tokens)
      if token_array.shape == mask.shape and len(coordinate) == token_array.ndim:
        max_abs_mismatch["token_id"] = int(token_array[coordinate])
      max_abs_wrapper = {"mismatches": [max_abs_mismatch]}
      _attach_sequence_context(
          max_abs_wrapper,
          prompt_mask=sidecar.prompt_mask,
          completion_valid_mask=sidecar.completion_valid_mask,
          action_mask=sidecar.action_mask,
      )
      max_abs_mismatch = max_abs_wrapper["mismatches"][0]
    if a.shape == b.shape == mask.shape and n_action:
      max_abs = _json_number(
          np.max(
              np.abs(
                  a.astype(np.float64)[mask] - b.astype(np.float64)[mask]
              )
          )
      )
    boundaries[name] = {
        **difference,
        "max_abs": max_abs,
        "max_abs_mismatch": max_abs_mismatch,
    }
    if difference["differing_bytes"] != 0:
      reds.append(name)
  record = {
      "timestamp": time.time(),
      "step": int(step),
      "verdict": "PASS" if not reds else "FAIL",
      "reds": reds,
      "N_action": n_action,
      "boundaries": boundaries,
      "hashes": {
          "S_decode": _hash(sd),
          "S_prefill": _hash(sp),
          "T_old": _hash(to),
          "tokens": _hash(sidecar.tokens),
          "action_mask": _hash(mask),
          "policy_version": _hash(sidecar.policy_version),
      },
      "masked_hashes": {
          "S_decode": _masked_hash(sd, mask),
          "S_prefill": _masked_hash(sp, mask),
          "T_old": _masked_hash(to, mask),
      },
      "context": {
          "source": sidecar.source_name,
          "mesh": os.environ.get("FL_SHARED_MESH", ""),
          "bucket": os.environ.get("MIN_TOKEN_BUCKET", ""),
          "run_stage": os.environ.get("CANON_P33_RUN_STAGE", ""),
      },
  }
  report_path = os.environ.get(
      PRE_REPORT_ENV,
      "/mnt/disks/tunix-data/frozenlake/logs/pre_alignment_report.jsonl",
  )
  os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
  with open(report_path, "a", encoding="utf-8") as report_file:
    report_file.write(json.dumps(record, sort_keys=True) + "\n")
    report_file.flush()
    os.fsync(report_file.fileno())
  compact_record = json.dumps(
      record, sort_keys=True, separators=(",", ":"), allow_nan=False
  )
  print(f"[CANON_ALIGN_PRE_JSON] {compact_record}", flush=True)
  print(
      "[CANON_ALIGN_PRE_EVIDENCE] "
      f"path={report_path} sha256={_report_sha256(report_path)}",
      flush=True,
  )
  print(
      "[CANON_ALIGN_PRE] "
      f"step={step} verdict={record['verdict']} N_action={n_action} "
      f"bounds={[(name, value['differing_bytes']) for name, value in boundaries.items()]}",
      flush=True,
  )
  if reds and fail_closed:
    raise AlignmentGateError(
        f"pre-backward alignment gate RED: {reds}; report={report_path}"
    )
  return record


def check_batch(
    sidecar: ObservedTrainExample,
    *,
    t_current: Any,
    gradient_norm: Any,
    optimizer_skipped: Any,
    step: int,
    fail_closed: bool = True,
) -> dict[str, Any]:
  """Check four boundaries and two ratios after one value_and_grad call."""
  mode = execution_mode()
  skipped = int(np.asarray(optimizer_skipped).item())
  expected_skipped = 1 if mode == "gate-only" else 0
  if skipped != expected_skipped:
    raise AlignmentGateError(
        "compiled train step optimizer attestation mismatch: "
        f"mode={mode} optimizer_skipped={skipped} expected={expected_skipped}"
    )

  sd = np.asarray(sidecar.s_decode)
  sp = np.asarray(sidecar.s_prefill)
  to = np.asarray(sidecar.t_old)
  tc = np.asarray(t_current)
  mask = np.asarray(sidecar.action_mask, dtype=np.bool_)
  sampling_values = np.asarray(sidecar.sampling_values, dtype=np.float32)
  n_action = int(mask.sum())
  reds: list[str] = []
  if n_action == 0:
    reds.append("N_action=0")
  canonical_c = None
  if os.environ.get("CANON_ENGINE_MODULE_C", "") != "1":
    reds.append("CANON_ENGINE_MODULE_C!=1")
  else:
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

    canonical_c = canonical_forward.attestation()
  if sampling_values.shape != (sd.shape[0], 3):
    reds.append(
        "sampling_values_shape="
        f"{sampling_values.shape},expected={(sd.shape[0], 3)}"
    )
  elif not np.all(sampling_values == sampling_values[:1]):
    reds.append("sampling_values_vary_within_batch")
  sampling_row = (
      sampling_values[0]
      if sampling_values.shape == (sd.shape[0], 3) and sd.shape[0]
      else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
  )

  boundaries = {}
  for name, a, b in (
      ("S_decode_vs_S_prefill", sd, sp),
      ("S_prefill_vs_T_old", sp, to),
      ("T_old_vs_T_current", to, tc),
  ):
    difference = _masked_bitwise_difference(a, b, mask)
    max_abs = float("nan")
    if a.shape == b.shape == mask.shape and n_action:
      max_abs = float(
          np.max(np.abs(a.astype(np.float64)[mask] - b.astype(np.float64)[mask]))
      )
    boundaries[name] = {**difference, "max_abs": max_abs}
    if difference["differing_bytes"] != 0:
      reds.append(name)

  w = np.exp(to.astype(np.float64) - sd.astype(np.float64))
  r = np.exp(tc.astype(np.float64) - to.astype(np.float64))
  wr = w * r
  exact = {
      "w_all_exactly_1": bool(np.all(w[mask] == 1.0)),
      "r_all_exactly_1": bool(np.all(r[mask] == 1.0)),
      "wr_all_exactly_1": bool(np.all(wr[mask] == 1.0)),
  }
  for key, ok in exact.items():
    if not ok:
      reds.append(key)
  clip_hits = int(np.sum((r[mask] < 0.8) | (r[mask] > 1.28)))
  tis_hits = int(np.sum(w[mask] > 2.0))
  if clip_hits:
    reds.append(f"clip_hits={clip_hits}")
  if tis_hits:
    reds.append(f"tis_hits={tis_hits}")

  grad_norm = float(np.asarray(gradient_norm))
  gradient = {
      "norm": grad_norm,
      "finite": bool(np.isfinite(grad_norm)),
      "nonzero": bool(grad_norm > 0.0),
  }
  if not gradient["finite"]:
    reds.append("gradient_nonfinite")
  # A real GRPO group may legitimately have identical rewards and therefore a
  # zero advantage/gradient.  Keep that measurement visible, but do not turn
  # it into a numerical alignment red in the multi-step training mode.  The
  # P26 stage classifier separately requires a nonzero learning signal before
  # promotion.  The historical gate-only/update-canary modes retain their
  # stricter nonzero-gradient contract.
  p27_real_update = (
      mode == "update-canary"
      and os.environ.get("CANON_FROZENLAKE_P27", "") == "1"
  )
  if not gradient["nonzero"] and mode != "train" and not p27_real_update:
    reds.append("gradient_zero")

  delta = (tc.astype(np.float64) - sd.astype(np.float64))[mask]
  record = {
      "timestamp": time.time(),
      "step": int(step),
      "execution_mode": mode,
      "verdict": "PASS" if not reds else "FAIL",
      "reds": reds,
      "N_action": n_action,
      "boundaries": boundaries,
      "exact": exact,
      "clip_hits": clip_hits,
      "tis_hits": tis_hits,
      "optimizer_skipped": skipped,
      "gradient": gradient,
      "kl_protocol": {
          "first_order_-mean_delta": float(-delta.mean()) if n_action else 0.0,
          "second_order_half_mean_delta2": (
              float(0.5 * np.mean(delta**2)) if n_action else 0.0
          ),
      },
      "hashes": {
          "S_decode": _hash(sd),
          "S_prefill": _hash(sp),
          "T_old": _hash(to),
          "T_current": _hash(tc),
          "tokens": _hash(sidecar.tokens),
          "action_mask": _hash(mask),
          "policy_version": _hash(sidecar.policy_version),
      },
      "masked_hashes": {
          "S_decode": _masked_hash(sd, mask),
          "S_prefill": _masked_hash(sp, mask),
          "T_old": _masked_hash(to, mask),
          "T_current": _masked_hash(tc, mask),
      },
      "context": {
          "source": sidecar.source_name,
          "temperature": (
              float(sampling_row[0]) if np.isfinite(sampling_row[0]) else None
          ),
          "top_k": (
              int(sampling_row[1]) if np.isfinite(sampling_row[1]) else None
          ),
          "top_p": (
              float(sampling_row[2]) if np.isfinite(sampling_row[2]) else None
          ),
          "mesh": os.environ.get("FL_SHARED_MESH", ""),
          "bucket": os.environ.get("MIN_TOKEN_BUCKET", ""),
      },
  }
  record["context"]["canonical_c"] = canonical_c
  report_path = os.environ.get(
      REPORT_ENV,
      "/mnt/disks/tunix-data/frozenlake/logs/alignment_report.jsonl",
  )
  os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
  with open(report_path, "a", encoding="utf-8") as report_file:
    report_file.write(json.dumps(record, sort_keys=True) + "\n")
  print(
      "[CANON_ALIGN] "
      f"step={step} verdict={record['verdict']} N_action={n_action} "
      f"bounds={[(k, v['differing_bytes']) for k, v in boundaries.items()]} "
      f"w/r/wr={exact} clip={clip_hits} tis={tis_hits} grad_norm={grad_norm:.6g}",
      flush=True,
  )
  if reds and fail_closed:
    raise AlignmentGateError(
        f"alignment gate RED mode={mode}: {reds}; report={report_path}"
    )
  return record
