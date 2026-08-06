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


class AlignmentGateError(RuntimeError):
  """Raised when an alignment run is incomplete or numerically red."""


@flax.struct.dataclass(frozen=True)
class ObservedTrainExample:
  """Host-only observability wrapper; never pass this object to JIT."""

  train_example: Any
  s_decode: Any
  s_prefill: Any
  t_old: Any
  action_mask: Any
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
  tok = np.asarray(tokens)
  expected = tuple(np.shape(train_example.completion_ids))
  for name, value in (
      ("S_decode", sd),
      ("S_prefill", sp),
      ("T_old", to),
      ("action_mask", mask),
      ("tokens", tok),
  ):
    if tuple(value.shape) != expected:
      raise AlignmentGateError(
          f"{name} shape {value.shape} != completion shape {expected}"
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


def _masked_bytes_differ(a: Any, b: Any, mask: Any) -> tuple[int, dict | None]:
  aa = np.asarray(a)
  bb = np.asarray(b)
  mm = np.asarray(mask, dtype=np.bool_)
  if aa.shape != bb.shape or aa.dtype != bb.dtype or aa.shape != mm.shape:
    return -1, None
  av = np.ascontiguousarray(aa[mm])
  bv = np.ascontiguousarray(bb[mm])
  byte_diff = av.view(np.uint8) != bv.view(np.uint8)
  count = int(byte_diff.sum())
  if not count:
    return 0, None
  first_byte = int(np.flatnonzero(byte_diff.reshape(-1))[0])
  index = first_byte // av.dtype.itemsize
  return count, {
      "masked_index": index,
      "a": float(av.reshape(-1)[index]),
      "b": float(bv.reshape(-1)[index]),
  }


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
    count, first = _masked_bytes_differ(a, b, mask)
    max_abs = float("nan")
    if a.shape == b.shape == mask.shape and n_action:
      max_abs = float(
          np.max(np.abs(a.astype(np.float64)[mask] - b.astype(np.float64)[mask]))
      )
    boundaries[name] = {
        "differing_bytes": count,
        "max_abs": max_abs,
        "first_mismatch": first,
    }
    if count != 0:
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
