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

"""Matched-work receipts for the one-host GSM8K XProf pair."""

from __future__ import annotations

import contextlib
import hashlib
import json
import numbers
import os
import re
from typing import Any, Mapping


ARM_ENV = "CANON_V1_GSM8K_XPROF_ARM"
_ARMS = ("native", "zero-hp")
_WORK_FIELDS = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_mask",
    "completion_valid_mask",
    "advantages",
    "policy_version",
)
_LABEL_ENV = "CANON_XPROF_LABELS"
_ANNOTATION_NAME = re.compile(r"[a-z][a-z0-9_]*")


def labels_enabled(values: Mapping[str, str] | None = None) -> bool:
  """Returns the exact default-off XProf annotation contract."""
  values = os.environ if values is None else values
  value = values.get(_LABEL_ENV, "")
  if value not in ("", "0", "1"):
    raise ValueError(
        f"{_LABEL_ENV} must be unset/0/1 (empty is disabled), got {value!r}"
    )
  return value == "1"


def _annotation_metadata(metadata: Mapping[str, Any]) -> dict[str, int]:
  normalized = {}
  for name, value in metadata.items():
    if not _ANNOTATION_NAME.fullmatch(name):
      raise ValueError(f"invalid XProf annotation metadata name {name!r}")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
      raise ValueError(
          "XProf annotation metadata must be integer-valued: "
          f"{name}={value!r}"
      )
    normalized[name] = int(value)
  return normalized


def trace_annotation(name: str, **metadata: Any):
  """Returns a bounded host TraceAnnotation or an exact no-op context."""
  if not labels_enabled():
    return contextlib.nullcontext()
  if not _ANNOTATION_NAME.fullmatch(name):
    raise ValueError(f"invalid XProf annotation name {name!r}")
  normalized = _annotation_metadata(metadata)
  import jax  # pylint: disable=g-import-not-at-top

  return jax.profiler.TraceAnnotation(name, **normalized)


def train_step_annotation(*, step_num: int):
  """Matches Native's ``StepTraceAnnotation('train')`` Steps-row contract."""
  if not labels_enabled():
    return contextlib.nullcontext()
  normalized = _annotation_metadata({"step_num": step_num})
  import jax  # pylint: disable=g-import-not-at-top

  return jax.profiler.StepTraceAnnotation("train", **normalized)


class ZeroHpTrainMicrostepSchedule:
  """Owns the 16 real Zero-HP train annotations through optimizer commit.

  The final train annotation deliberately outlives its reverse transaction so
  the real optimizer commit is its child, as in Native. The schedule is
  profiling-only: it owns context-manager lifetimes and never touches arrays.
  """

  def __init__(self, *, update_step: int, microsteps: int = 16):
    if isinstance(update_step, bool) or not isinstance(
        update_step, numbers.Integral
    ):
      raise ValueError("Zero-HP XProf update_step must be an integer")
    if microsteps != 16:
      raise ValueError(
          f"Zero-HP XProf requires exactly 16 microsteps, got {microsteps}"
      )
    self.update_step = int(update_step)
    self.microsteps = microsteps
    self._entered = False
    self._next_microstep = 0
    self._last_annotation = None
    self._last_open = False
    self._last_closed_by_optimizer = False
    self._optimizer_seen = False

  def __enter__(self):
    if self._entered:
      raise RuntimeError("Zero-HP XProf train schedule cannot be re-entered")
    self._entered = True
    return self

  def _require_entered(self) -> None:
    if not self._entered:
      raise RuntimeError("Zero-HP XProf train schedule is not entered")

  def _close_last(self, error: BaseException | None = None) -> None:
    if not self._last_open:
      return
    self._last_open = False
    if error is None:
      self._last_annotation.__exit__(None, None, None)
    else:
      self._last_annotation.__exit__(
          type(error), error, error.__traceback__
      )

  @contextlib.contextmanager
  def transaction(self, micro_step: int):
    """Wraps one real reverse/reduce/accumulate transaction."""
    self._require_entered()
    if micro_step != self._next_microstep:
      raise RuntimeError(
          "Zero-HP XProf microsteps must be sequential: "
          f"got={micro_step} expected={self._next_microstep}"
      )
    if self._last_open:
      raise RuntimeError("Zero-HP XProf last train annotation is already open")
    annotation = train_step_annotation(
        step_num=self.update_step * self.microsteps + micro_step
    )
    annotation.__enter__()
    try:
      yield
    except BaseException as error:
      annotation.__exit__(type(error), error, error.__traceback__)
      raise
    else:
      self._next_microstep += 1
      if micro_step == self.microsteps - 1:
        self._last_annotation = annotation
        self._last_open = True
      else:
        annotation.__exit__(None, None, None)

  @contextlib.contextmanager
  def optimizer_commit(self):
    """Keeps the last train annotation open through the real optimizer."""
    self._require_entered()
    if (
        self._next_microstep != self.microsteps
        or self._last_annotation is None
        or not self._last_open
    ):
      raise RuntimeError(
          "Zero-HP XProf optimizer requires 16 completed transactions"
      )
    if self._optimizer_seen:
      raise RuntimeError("Zero-HP XProf optimizer annotation is duplicated")
    self._optimizer_seen = True
    try:
      with trace_annotation(
          "optimizer_commit", update_step=self.update_step
      ):
        yield
    except BaseException as error:
      self._close_last(error)
      raise
    else:
      self._close_last()
      self._last_closed_by_optimizer = True

  def __exit__(self, exc_type, exc_value, traceback):
    del traceback
    self._entered = False
    if self._last_open:
      self._close_last(exc_value)
    if exc_type is None:
      if self._next_microstep != self.microsteps:
        raise RuntimeError(
            "Zero-HP XProf train schedule ended before all transactions: "
            f"{self._next_microstep}/{self.microsteps}"
        )
      if not self._optimizer_seen or not self._last_closed_by_optimizer:
        raise RuntimeError(
            "Zero-HP XProf train schedule ended without optimizer ownership"
        )
    return False


def zero_hp_train_microsteps(
    *, update_step: int, microsteps: int = 16
) -> ZeroHpTrainMicrostepSchedule:
  """Builds the signed Zero-HP-only Native-like train schedule."""
  if arm() != "zero-hp" or not labels_enabled():
    raise RuntimeError(
        "Native-like Zero-HP train microsteps require the signed arm and "
        "CANON_XPROF_LABELS=1"
    )
  return ZeroHpTrainMicrostepSchedule(
      update_step=update_step, microsteps=microsteps
  )


def arm(values: Mapping[str, str] | None = None) -> str:
  """Returns the signed pair arm and rejects partial/mixed configurations."""
  values = os.environ if values is None else values
  selected = values.get(ARM_ENV, "")
  if not selected:
    return ""
  if selected not in _ARMS:
    raise ValueError(
        f"{ARM_ENV} must be unset or one of {_ARMS}, got {selected!r}"
    )

  common = {
      "CANON_GSM8K_TRAIN": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_P60_DETERMINISTIC_AB": "1",
      "CANON_XPROF_SKIP_STEPS": "2",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_LABELS": "1",
  }
  wrong = {
      name: values.get(name)
      for name, expected in common.items()
      if values.get(name) != expected
  }
  # The capture phase is a signed two-value contract paired with the TPU
  # trace mode: update (backward window, the certification default) takes
  # TRACE_ONLY_XLA, while step (rollout diagnostic window) must leave the
  # TPU trace mode empty -- the learner admits it only for update.
  phase = values.get("CANON_XPROF_PHASE")
  tpu_trace_mode = values.get("CANON_XPROF_TPU_TRACE_MODE")
  if phase == "update":
    if tpu_trace_mode != "TRACE_ONLY_XLA":
      wrong["CANON_XPROF_TPU_TRACE_MODE"] = tpu_trace_mode
  elif phase == "step":
    if tpu_trace_mode not in ("", None):
      wrong["CANON_XPROF_TPU_TRACE_MODE"] = tpu_trace_mode
  else:
    wrong["CANON_XPROF_PHASE"] = phase
  if not values.get("CANON_XPROF_DIR"):
    wrong["CANON_XPROF_DIR"] = values.get("CANON_XPROF_DIR")
  if not values.get("CANON_PERF_TRACE_DIR"):
    wrong["CANON_PERF_TRACE_DIR"] = values.get("CANON_PERF_TRACE_DIR")
  if wrong:
    raise ValueError(
        f"{ARM_ENV}={selected} has an invalid common capture contract: {wrong}"
    )

  vanilla = values.get("CANON_GSM8K_VANILLA", "")
  workload = values.get("CANON_P32_WORKLOAD", "")
  rank_parallel = values.get("CANON_P59_RANK_PARALLEL_BACKWARD", "")
  g6_update = values.get("CANON_P28_G6_UPDATE", "")
  if selected == "native":
    if (
        vanilla != "1"
        or workload
        or rank_parallel not in ("", "0")
        or g6_update not in ("", "0")
    ):
      raise ValueError(
          "native GSM8K XProf requires the vanilla stock trainer with no "
          "P32/P59/G6 numerical program"
      )
  else:
    if (
        vanilla
        or workload != "gsm8k-p59-dp4-tp1"
        or rank_parallel != "1"
        or g6_update != "1"
        or values.get("CANON_GSM8K_ALIGNMENT_WARN_ONLY") != "0"
    ):
      raise ValueError(
          "zero-hp GSM8K XProf requires strict V1 DP4xTP1 P59 training"
      )
  return selected


def _array_receipt(value: Any) -> dict[str, Any]:
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
  digest = hashlib.sha256()
  digest.update(str(array.dtype).encode("ascii"))
  digest.update(b"\0")
  digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
  digest.update(b"\0")
  digest.update(array.tobytes(order="C"))
  return {
      "dtype": str(array.dtype),
      "shape": list(array.shape),
      "sha256": digest.hexdigest(),
  }


def work_receipt(
    train_example: Any,
    *,
    selected_arm: str,
    train_step: int,
    global_step: int,
) -> dict[str, Any]:
  """Builds a host receipt for the exact batch consumed by one update."""
  if selected_arm not in _ARMS:
    raise ValueError(f"invalid GSM8K XProf arm: {selected_arm!r}")
  fields = {
      name: _array_receipt(getattr(train_example, name))
      for name in _WORK_FIELDS
      if getattr(train_example, name, None) is not None
  }
  missing = {"prompt_ids", "completion_ids", "advantages"} - set(fields)
  if missing:
    raise ValueError(
        "GSM8K XProf work receipt is missing fields: "
        + ",".join(sorted(missing))
    )
  shape_signature = hashlib.sha256(
      json.dumps(
          {name: item["shape"] for name, item in fields.items()},
          sort_keys=True,
          separators=(",", ":"),
      ).encode("utf-8")
  ).hexdigest()
  return {
      "schema": "canon.v1.gsm8k-onehost-xprof.work.v1",
      "arm": selected_arm,
      "train_step": int(train_step),
      "global_step": int(global_step),
      "fields": fields,
      "shape_signature": shape_signature,
  }


def emit_work_receipt(
    train_example: Any,
    *,
    train_step: int,
    global_step: int,
) -> None:
  """Prints one deterministic receipt when the matched pair is active."""
  selected_arm = arm()
  if not selected_arm:
    return
  receipt = work_receipt(
      train_example,
      selected_arm=selected_arm,
      train_step=train_step,
      global_step=global_step,
  )
  print(
      "[V1.GSM8K.XPROF.WORK] "
      + json.dumps(receipt, sort_keys=True, separators=(",", ":")),
      flush=True,
  )
