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
FORWARD_ONLY_ENV = "CANON_ALIGNMENT_FORWARD_ONLY"
UPDATE_CANARY_ENV = "CANON_ALIGNMENT_UPDATE_CANARY"
TRAIN_ENV = "CANON_ALIGNMENT_TRAIN"
REPORT_ENV = "CANON_ALIGN_REPORT"
DEBUG_ARRAYS_ENV = "CANON_ALIGNMENT_DEBUG_NPZ"
D2B_REPORT_ENV = "CANON_P32_D2B_REPORT"
D2B_ARRAYS_ENV = "CANON_P32_D2B_NPZ"


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
  def prompt_ids(self):
    return self.train_example.prompt_ids

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
      "forward-only": os.environ.get(FORWARD_ONLY_ENV, "") == "1",
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


def _full_bytes_differ(a: Any, b: Any) -> int:
  aa = np.ascontiguousarray(np.asarray(a))
  bb = np.ascontiguousarray(np.asarray(b))
  if aa.shape != bb.shape or aa.dtype != bb.dtype:
    return -1
  return int(np.count_nonzero(aa.view(np.uint8) != bb.view(np.uint8)))


def _reference_processed_logprobs(processed_logits: Any) -> np.ndarray:
  """Materializes the full categorical distribution for the D2b artifact."""
  values = np.asarray(processed_logits, dtype=np.float64)
  support = np.isfinite(values)
  if not np.all(np.any(support, axis=-1)):
    raise AlignmentGateError("D2b processed row has empty finite support")
  masked = np.where(support, values, -np.inf)
  maximum = np.max(masked, axis=-1, keepdims=True)
  normalizer = maximum + np.log(
      np.sum(np.exp(masked - maximum), axis=-1, keepdims=True)
  )
  return np.where(support, masked - normalizer, -np.inf).astype(np.float32)


def check_p32_d2b_full_distribution(
    *,
    engine_result: dict[str, Any],
    t_old_logps: Any,
    t_old_diagnostics: dict[str, Any],
    t_current_logps: Any,
    t_current_diagnostics: dict[str, Any],
    fail_closed: bool = True,
) -> dict[str, Any]:
  """Checks two DP-covering full-vocabulary sentinels at all four boundaries."""
  if os.environ.get("CANON_P32_D2B_FULL_DISTRIBUTION", "") != "1":
    raise AlignmentGateError(
        "P32 D2b requires CANON_P32_D2B_FULL_DISTRIBUTION=1"
    )
  tokens = np.asarray(engine_result["generated_tokens"], dtype=np.int32)
  if tokens.shape != (2, 2):
    raise AlignmentGateError(f"D2b generated token shape {tokens.shape} != (2, 2)")
  arms = {
      "S_decode": {
          "raw": np.asarray(engine_result["decode"]["raw_rows"]),
          "processed": np.asarray(engine_result["decode"]["processed_rows"]),
          "target_logps": np.asarray(engine_result["decode_target_logps"]),
      },
      "S_prefill": {
          "raw": np.asarray(engine_result["prefill"]["raw_rows"]),
          "processed": np.asarray(engine_result["prefill"]["processed_rows"]),
          "target_logps": np.asarray(engine_result["prefill_target_logps"]),
      },
      "T_old": {
          "raw": np.asarray(t_old_diagnostics["raw_rows"])[:, 1, :],
          "processed": np.asarray(t_old_diagnostics["processed_rows"])[:, 1, :],
          "target_logps": np.asarray(t_old_logps)[:, 1],
          "target_ids": np.asarray(t_old_diagnostics["target_ids"])[:, 1],
      },
      "T_current": {
          "raw": np.asarray(t_current_diagnostics["raw_rows"])[:, 1, :],
          "processed": np.asarray(t_current_diagnostics["processed_rows"])[:, 1, :],
          "target_logps": np.asarray(t_current_logps)[:, 1],
          "target_ids": np.asarray(t_current_diagnostics["target_ids"])[:, 1],
      },
  }
  vocab = arms["S_decode"]["processed"].shape[-1]
  expected_row_shape = (2, vocab)
  reasons = []
  for name, arm in arms.items():
    for field in ("raw", "processed"):
      value = arm[field]
      if value.shape != expected_row_shape or value.dtype != np.float32:
        reasons.append(
            f"{name}_{field}_contract={value.shape}/{value.dtype}"
        )
    if arm["target_logps"].shape != (2,) or arm["target_logps"].dtype != np.float32:
      reasons.append(
          f"{name}_target_contract={arm['target_logps'].shape}/"
          f"{arm['target_logps'].dtype}"
      )
    if name in ("T_old", "T_current") and not np.array_equal(
        arm["target_ids"].astype(np.int32), tokens[:, 1]
    ):
      reasons.append(f"{name}_target_ids")
    arm["support"] = np.isfinite(arm["processed"])
    arm["distribution"] = _reference_processed_logprobs(arm["processed"])
    target_from_distribution = np.take_along_axis(
        arm["distribution"], tokens[:, 1, None], axis=-1
    )[:, 0]
    arm["reference_target_logps"] = target_from_distribution

  baseline = arms["S_decode"]
  comparisons = {}
  expected_comparisons = 0
  for name in ("S_prefill", "T_old", "T_current"):
    other = arms[name]
    for field in ("raw", "processed", "support", "distribution", "target_logps"):
      key = f"S_decode_vs_{name}.{field}"
      comparisons[key] = _full_bytes_differ(baseline[field], other[field])
      expected_comparisons += 1
      if comparisons[key] != 0:
        reasons.append(key)
  if expected_comparisons != 15 or len(comparisons) != 15:
    reasons.append(
        f"comparison_count={len(comparisons)} expected=15"
    )
  dp_coverage = {
      name: list(map(int, engine_result[name]["dp_ranks"]))
      for name in ("decode", "prefill")
  }
  if any(sorted(ranks) != [0, 1] for ranks in dp_coverage.values()):
    reasons.append(f"dp_coverage={dp_coverage}")
  target_consistency = {}
  for name, arm in arms.items():
    diff = _full_bytes_differ(
        arm["target_logps"], arm["reference_target_logps"]
    )
    target_consistency[name] = diff
    # The host reference uses float64 normalization and is a semantic artifact,
    # not the deployed scorer. Keep its delta visible without using it as a
    # bitwise execution gate; the deployed target logps are compared above.

  report = {
      "schema_version": 1,
      "status": "pass" if not reasons else "fail",
      "verdict": (
          "P32_DP2TP2_D2B_PASS" if not reasons else "P32_DP2TP2_D2B_FAIL"
      ),
      "reasons": reasons,
      "sentinel_count": 2,
      "vocab_size": int(vocab),
      "dp_coverage": dp_coverage,
      "comparison_count": len(comparisons),
      "comparisons": comparisons,
      "target_reference_differing_bytes": target_consistency,
      "sampling": dict(engine_result["sampling"]),
      "hashes": {
          f"{name}.{field}": _hash(arm[field])
          for name, arm in arms.items()
          for field in ("raw", "processed", "support", "distribution", "target_logps")
      },
  }
  arrays_path = os.environ.get(D2B_ARRAYS_ENV, "")
  report_path = os.environ.get(D2B_REPORT_ENV, "")
  if not arrays_path or not report_path:
    raise AlignmentGateError(
        "P32 D2b requires CANON_P32_D2B_NPZ and CANON_P32_D2B_REPORT"
    )
  os.makedirs(os.path.dirname(arrays_path) or ".", exist_ok=True)
  os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
  with open(arrays_path, "xb") as arrays_file:
    np.savez_compressed(
        arrays_file,
        generated_tokens=tokens,
        **{
            f"{name}_{field}": arm[field]
            for name, arm in arms.items()
            for field in ("raw", "processed", "support", "distribution", "target_logps")
        },
    )
  report["artifact_npz"] = arrays_path
  with open(report_path, "x", encoding="utf-8") as report_file:
    report_file.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(
      "[P32.D2B] "
      f"verdict={report['verdict']} sentinels=2 vocab={vocab} "
      f"comparisons={len(comparisons)} dp={dp_coverage}",
      flush=True,
  )
  if reasons and fail_closed:
    raise AlignmentGateError(
        f"P32 D2b full-distribution gate RED: {reasons}; report={report_path}"
    )
  return report


def check_batch(
    sidecar: ObservedTrainExample,
    *,
    t_current: Any,
    gradient_norm: Any,
    optimizer_skipped: Any,
    step: int,
    backward_executed: Any = 1,
    fail_closed: bool = True,
) -> dict[str, Any]:
  """Check four boundaries and two ratios after one value_and_grad call."""
  mode = execution_mode()
  skipped = int(np.asarray(optimizer_skipped).item())
  expected_skipped = 1 if mode in ("forward-only", "gate-only") else 0
  if skipped != expected_skipped:
    raise AlignmentGateError(
        "compiled train step optimizer attestation mismatch: "
        f"mode={mode} optimizer_skipped={skipped} expected={expected_skipped}"
    )
  backward = int(np.asarray(backward_executed).item())
  expected_backward = 0 if mode == "forward-only" else 1
  if backward != expected_backward:
    raise AlignmentGateError(
        "compiled train step backward attestation mismatch: "
        f"mode={mode} backward_executed={backward} "
        f"expected={expected_backward}"
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
      "executed": bool(backward),
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
  if (
      not gradient["nonzero"]
      and mode not in ("forward-only", "train")
      and not p27_real_update
  ):
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
  debug_path = os.environ.get(DEBUG_ARRAYS_ENV, "")
  if debug_path:
    os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
    with open(debug_path, "xb") as debug_file:
      np.savez_compressed(
          debug_file,
          s_decode=sd,
          s_prefill=sp,
          t_old=to,
          t_current=tc,
          action_mask=mask,
          tokens=np.asarray(sidecar.tokens),
          policy_version=np.asarray(sidecar.policy_version),
          sampling_values=sampling_values,
      )
    record["context"]["debug_npz"] = debug_path
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
