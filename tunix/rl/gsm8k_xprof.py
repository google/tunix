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

import hashlib
import json
import os
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
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": "1",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
  }
  wrong = {
      name: values.get(name)
      for name, expected in common.items()
      if values.get(name) != expected
  }
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
