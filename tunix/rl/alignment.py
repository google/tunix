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
import sys
import time
from typing import Any, Mapping

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
P38_CONTROLLED_EXIT_ENV = "CANON_P38_CONTROLLED_EXIT"
P38_CONTROLLED_EXIT_CODE = 42
P38_DIAGNOSTIC_ROUNDS_ENV = "CANON_P38_DIAGNOSTIC_ROUNDS"
P38_DIAGNOSTIC_ROUND_FILE_ENV = "CANON_P38_DIAGNOSTIC_ROUND_FILE"
P38_ROUND_SEAL_REQUEST_DIR_ENV = "CANON_P38_ROUND_SEAL_REQUEST_DIR"
P38_ROUND_SEAL_ACK_DIR_ENV = "CANON_P38_ROUND_SEAL_ACK_DIR"
P38_ONEHOST_REHEARSAL_ENV = "CANON_P38_ONEHOST_REHEARSAL"
P38_MISMATCH_CAPSULE_ENV = "CANON_P38_MISMATCH_CAPSULE"
P38_MISMATCH_CAPSULE_MAX_ROWS_ENV = "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS"
GSM8K_AB_REPORT_ONLY_ENV = "CANON_GSM8K_AB_REPORT_ONLY"
GSM8K_ALIGNMENT_WARN_ONLY_ENV = "CANON_GSM8K_ALIGNMENT_WARN_ONLY"
FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV = "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"
DEEPSWE_ALIGNMENT_WARN_ONLY_ENV = "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"
_GSM8K_AB_POLICY_ID = "gsm8k-full-ab-report-v1"
_GSM8K_ALIGNMENT_WARNING_POLICY_ID = "gsm8k-full-alignment-warning-v2"
_FROZENLAKE_ALIGNMENT_WARNING_POLICY_ID = (
    "frozenlake-full-alignment-warning-v1"
)
_DEEPSWE_ALIGNMENT_WARNING_POLICY_ID = "deepswe-pilot-alignment-warning-v1"
_GSM8K_AB_MAX_ABS = 1.0e-4
_GSM8K_AB_MAX_BYTE_FRACTION = 4.0e-3
_MAX_MISMATCH_DETAILS = 1024


class AlignmentGateError(RuntimeError):
  """Raised when an alignment run is incomplete or numerically red."""


class PreAlignmentProbeComplete(RuntimeError):
  """Raised after an exact P38 precheck to stop before backward."""


class P38DiagnosticRoundComplete(RuntimeError):
  """Raised after a nonterminal frozen-weight P38 diagnostic round."""


_P38_DIAGNOSTIC_ROUNDS_COMPLETED = 0


def p38_diagnostic_rounds() -> int:
  """Return the bounded number of frozen-weight P38 rollout rounds."""
  raw = os.environ.get(P38_DIAGNOSTIC_ROUNDS_ENV, "1")
  try:
    rounds = int(raw)
  except ValueError as exc:
    raise AlignmentGateError(
        f"{P38_DIAGNOSTIC_ROUNDS_ENV} must be an integer"
    ) from exc
  if rounds < 1 or rounds > 8:
    raise AlignmentGateError(
        f"{P38_DIAGNOSTIC_ROUNDS_ENV} must be in [1, 8]"
    )
  return rounds


def p38_diagnostic_round_index() -> int:
  """Return the zero-based round currently being materialized."""
  return int(_P38_DIAGNOSTIC_ROUNDS_COMPLETED)


def _publish_p38_diagnostic_round(round_index: int) -> None:
  """Atomically publish the active host-only incident-ledger round."""
  path = os.environ.get(P38_DIAGNOSTIC_ROUND_FILE_ENV, "")
  if not path:
    raise AlignmentGateError(
        f"{P38_DIAGNOSTIC_ROUND_FILE_ENV} is required for multi-round P38"
    )
  directory = os.path.dirname(path)
  os.makedirs(directory, exist_ok=True)
  temporary = f"{path}.tmp"
  with open(temporary, "x", encoding="utf-8") as stream:
    stream.write(f"{round_index}\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def _seal_p38_diagnostic_round(round_index: int) -> None:
  """Block until the survivor worker durably seals one completed round."""
  request_dir = os.environ.get(P38_ROUND_SEAL_REQUEST_DIR_ENV, "")
  ack_dir = os.environ.get(P38_ROUND_SEAL_ACK_DIR_ENV, "")
  if (
      not request_dir
      and not ack_dir
      and os.environ.get(P38_ONEHOST_REHEARSAL_ENV, "0") == "1"
  ):
    print(
        "[CANON_P38] ROUND_SEAL_SKIPPED "
        f"round={round_index} scope=onehost-rehearsal",
        flush=True,
    )
    return
  if not request_dir or not ack_dir:
    raise AlignmentGateError(
        "multi-round P38 requires round-seal request and acknowledgement dirs"
    )
  os.makedirs(request_dir, exist_ok=True)
  os.makedirs(ack_dir, exist_ok=True)
  stem = f"round-{int(round_index):06d}"
  request_path = os.path.join(request_dir, f"{stem}.request")
  ack_path = os.path.join(ack_dir, f"{stem}.ack")
  if os.path.exists(request_path) or os.path.exists(ack_path):
    raise AlignmentGateError(
        f"P38 round-seal control path already exists for round {round_index}"
    )
  temporary = f"{request_path}.tmp"
  payload = {
      "action": "seal-round",
      "diagnostic_round": int(round_index),
      "schema": "canon-p38-round-seal-request-v1",
  }
  with open(temporary, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, request_path)
  print(
      "[CANON_P38] ROUND_SEAL_REQUESTED "
      f"round={round_index} request={request_path}",
      flush=True,
  )
  deadline = time.monotonic() + 900.0
  while time.monotonic() < deadline:
    if os.path.isfile(ack_path) and os.path.getsize(ack_path) > 0:
      try:
        with open(ack_path, encoding="utf-8") as stream:
          acknowledgement = json.load(stream)
      except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentGateError(
            f"P38 round-seal acknowledgement is invalid: {ack_path}"
        ) from exc
      expected = {
          "action": "seal-round",
          "diagnostic_round": int(round_index),
          "schema": "canon-p38-round-seal-ack-v1",
          "status": "PASS",
      }
      if acknowledgement != expected:
        raise AlignmentGateError(
            "P38 round-seal acknowledgement drifted: "
            f"expected={expected} observed={acknowledgement}"
        )
      print(
          "[CANON_P38] ROUND_SEAL_ACKNOWLEDGED "
          f"round={round_index} ack={ack_path}",
          flush=True,
      )
      return
    time.sleep(1.0)
  raise AlignmentGateError(
      f"timed out waiting for P38 round {round_index} durability acknowledgement"
  )


def _finish_p38_precheck(message: str) -> None:
  """Terminate a target P38 diagnostic without waiting on backend threads."""
  controlled = os.environ.get(P38_CONTROLLED_EXIT_ENV, "")
  if controlled not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{P38_CONTROLLED_EXIT_ENV} must be exactly 0 or 1, got {controlled!r}"
    )
  if controlled == "1":
    print(
        "[CANON_P38] CONTROLLED_EXIT "
        f"code={P38_CONTROLLED_EXIT_CODE} backward=0 optimizer_commits=0",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(P38_CONTROLLED_EXIT_CODE)  # pylint: disable=protected-access
  raise PreAlignmentProbeComplete(message)


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
  _finish_p38_precheck(
      "P38 precheck-only diagnostic completed before backward"
  )


def stop_after_diagnostic_precheck(record: dict[str, Any]) -> None:
  """Stop after a finite P38 A/B diagnostic, including the expected red arm.

  P38 is specifically intended to capture a known decode-versus-prefill
  mismatch.  A durable finite A/B red with exact B/C is an admitted diagnostic
  result, not a training admission.  Invalid arrays, non-finite values, an
  empty action set, or B/C drift remain fatal.
  """
  if not precheck_only_enabled():
    return
  boundaries = record.get("boundaries", {})
  a_b = boundaries.get("S_decode_vs_S_prefill", {})
  b_c = boundaries.get("S_prefill_vs_T_old", {})
  admitted = (
      int(record.get("N_action", 0)) > 0
      and a_b.get("valid") is True
      and a_b.get("finite") is True
      and isinstance(a_b.get("differing_bytes"), int)
      and b_c.get("valid") is True
      and b_c.get("finite") is True
      and b_c.get("differing_bytes") == 0
  )
  if not admitted:
    raise AlignmentGateError(
        "P38 diagnostic precheck requires finite A/B evidence and exact B/C"
    )
  global _P38_DIAGNOSTIC_ROUNDS_COMPLETED
  rounds = p38_diagnostic_rounds()
  round_index = p38_diagnostic_round_index()
  if round_index >= rounds:
    raise AlignmentGateError(
        "P38 diagnostic round counter exceeded its registered bound"
    )
  print(
      "[CANON_P38] PRECHECK_ROUND_COMPLETE "
      f"round={round_index + 1}/{rounds} "
      f"step={record.get('step')} N_action={record.get('N_action')} "
      f"verdict={record.get('verdict')} "
      f"a_b_differing_bytes={a_b.get('differing_bytes')} "
      "backward=0 optimizer_commits=0",
      flush=True,
  )
  if rounds > 1:
    _seal_p38_diagnostic_round(round_index)
  if round_index + 1 < rounds:
    _P38_DIAGNOSTIC_ROUNDS_COMPLETED += 1
    _publish_p38_diagnostic_round(_P38_DIAGNOSTIC_ROUNDS_COMPLETED)
    raise P38DiagnosticRoundComplete(
        f"P38 frozen-weight diagnostic round {round_index + 1} completed"
    )
  print(
      "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD "
      f"rounds={rounds} step={record.get('step')} "
      f"N_action={record.get('N_action')} verdict={record.get('verdict')} "
      f"a_b_differing_bytes={a_b.get('differing_bytes')}",
      flush=True,
  )
  _finish_p38_precheck(
      "P38 diagnostic precheck completed before backward"
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


def gsm8k_ab_report_policy() -> dict[str, Any]:
  """Returns the narrow, preregistered full-run alignment policy."""
  raw = os.environ.get(GSM8K_AB_REPORT_ONLY_ENV, "")
  if raw not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{GSM8K_AB_REPORT_ONLY_ENV} must be exactly 0 or 1, got {raw!r}"
    )
  warn_raw = os.environ.get(GSM8K_ALIGNMENT_WARN_ONLY_ENV, "")
  if warn_raw not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{GSM8K_ALIGNMENT_WARN_ONLY_ENV} must be exactly 0 or 1, "
        f"got {warn_raw!r}"
    )
  bounded_ab = raw == "1"
  warning_only = warn_raw == "1"
  frozenlake_warn_raw = os.environ.get(
      FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV, ""
  )
  if frozenlake_warn_raw not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV} must be exactly 0 or 1, "
        f"got {frozenlake_warn_raw!r}"
    )
  frozenlake_warning_only = frozenlake_warn_raw == "1"
  deepswe_warn_raw = os.environ.get(DEEPSWE_ALIGNMENT_WARN_ONLY_ENV, "")
  if deepswe_warn_raw not in ("", "0", "1"):
    raise AlignmentGateError(
        f"{DEEPSWE_ALIGNMENT_WARN_ONLY_ENV} must be exactly 0 or 1, "
        f"got {deepswe_warn_raw!r}"
    )
  deepswe_warning_only = deepswe_warn_raw == "1"
  p58_active = os.environ.get("CANON_P58_DEEPSWE_TIM", "") == "1"
  p58_arm = os.environ.get("CANON_P58_TIM_ARM", "") if p58_active else ""
  if p58_active and p58_arm not in ("native", "zero"):
    raise AlignmentGateError("P58 alignment arm must be native or zero")
  if p58_active and (deepswe_warning_only != (p58_arm == "native")):
    raise AlignmentGateError(
        "P58 native requires observer-only A-B warnings and zero requires "
        "strict alignment"
    )
  warning_policies = sum(
      (warning_only, frozenlake_warning_only, deepswe_warning_only)
  )
  if warning_policies > 1:
    raise AlignmentGateError(
        "GSM8K, FrozenLake, and DeepSWE warning-only policies are mutually "
        "exclusive"
    )
  if bounded_ab and warning_only:
    raise AlignmentGateError(
        f"{GSM8K_AB_REPORT_ONLY_ENV} and "
        f"{GSM8K_ALIGNMENT_WARN_ONLY_ENV} are mutually exclusive"
    )
  enabled_policy = (
      bounded_ab
      or warning_only
      or frozenlake_warning_only
      or deepswe_warning_only
  )
  workload = os.environ.get("CANON_P32_WORKLOAD", "")
  if not workload and os.environ.get("CANON_GSM8K_TRAIN", "") == "1":
    workload = "gsm8k"
  stage = os.environ.get("CANON_P33_RUN_STAGE", "")
  if not stage and os.environ.get("CANON_GSM8K_TRAIN", "") == "1":
    stage = "full"
  no_commit = os.environ.get("CANON_P33_NO_COMMIT", "") or "0"
  if deepswe_warning_only:
    p34_stage = os.environ.get("CANON_P34_RUN_STAGE", "")
    p39_pilot = os.environ.get("CANON_P39_64CHIP_PILOT", "") == "1"
    p43_debug = os.environ.get("CANON_P43_DEEPSWE_DEBUG", "") == "1"
    p44_parity = os.environ.get("CANON_P44_DEEPSWE_PARITY", "") == "1"
    p58_tim = os.environ.get("CANON_P58_DEEPSWE_TIM", "") == "1"
    production_full = (
        not any((p39_pilot, p43_debug, p44_parity, p58_tim))
        and p34_stage == "full"
    )
    admitted = (
        os.environ.get("CANON_P34_DEEPSWE", "") == "1"
        and (
            production_full
            or (
                sum((p39_pilot, p43_debug, p44_parity, p58_tim)) == 1
                and p34_stage in ("one-update", "three-update")
            )
        )
        and os.environ.get("CANON_P34_NO_COMMIT", "") == "0"
        and execution_mode() == "train"
    )
    if not admitted:
      raise AlignmentGateError(
          "DeepSWE warning policy is admitted only for committed P34 full "
          "training or a committed P39, P43, or P44 debug update"
      )
    workload = "deepswe"
    stage = p34_stage
  elif frozenlake_warning_only:
    admitted = (
        workload in ("frozenlake", "frozenlake-dp8-tp8")
        and stage == "full"
        and no_commit == "0"
        and execution_mode() == "train"
    )
    if not admitted:
      raise AlignmentGateError(
          "FrozenLake alignment warning policy is admitted only for committed "
          "FrozenLake full training"
      )
    # The policy schema describes the workload family, while topology remains
    # attested independently by the P33/P45 update record.
    workload = "frozenlake"
  elif enabled_policy:
    admitted = (
        workload == "gsm8k"
        and stage == "full"
        and no_commit == "0"
        and execution_mode() == "train"
    )
    if not admitted:
      raise AlignmentGateError(
          "GSM8K alignment reporting is admitted only for committed GSM8K "
          "full training"
      )
  return {
      "id": (
          _DEEPSWE_ALIGNMENT_WARNING_POLICY_ID
          if deepswe_warning_only
          else _FROZENLAKE_ALIGNMENT_WARNING_POLICY_ID
          if frozenlake_warning_only
          else
          _GSM8K_ALIGNMENT_WARNING_POLICY_ID
          if warning_only
          else _GSM8K_AB_POLICY_ID
      ),
      "enabled": enabled_policy,
      "warning_only": (
          warning_only or frozenlake_warning_only or deepswe_warning_only
      ),
      "warning_boundaries": (
          ("S_decode_vs_S_prefill",) if p58_arm == "native" else None
      ),
      "bounded_ab_only": bounded_ab,
      "workload": workload,
      "stage": stage,
      "max_abs_limit": (
          None
          if warning_only or frozenlake_warning_only or deepswe_warning_only
          else _GSM8K_AB_MAX_ABS
      ),
      "byte_fraction_limit": (
          None
          if warning_only or frozenlake_warning_only or deepswe_warning_only
          else _GSM8K_AB_MAX_BYTE_FRACTION
      ),
      "claim_level": (
          "convergence-only"
          if warning_only or frozenlake_warning_only or deepswe_warning_only
          else "alignment-degraded"
          if enabled_policy
          else "strict-zero-tim"
      ),
  }


def _policy_warns(policy: Mapping[str, Any], item: str) -> bool:
  """Returns whether a finite mismatch is observer-only for this boundary."""
  if not policy.get("warning_only", False):
    return False
  boundaries = policy.get("warning_boundaries")
  if boundaries is None:
    return True
  if item in boundaries:
    return True
  # w and w*r include the registered native A-B treatment.  r is B-C and
  # therefore remains exact/fail-closed.
  return item in ("w_all_exactly_1", "wr_all_exactly_1", "clip_hits", "tis_hits")


def _masked_pair_is_finite(a: Any, b: Any, mask: Any) -> bool:
  """Returns whether a shape-valid masked pair contains only finite values."""
  aa = np.asarray(a)
  bb = np.asarray(b)
  mm = np.asarray(mask, dtype=np.bool_)
  if aa.shape != bb.shape or aa.shape != mm.shape:
    return False
  return bool(np.all(np.isfinite(aa[mm])) and np.all(np.isfinite(bb[mm])))


def _ab_drift_is_reportable(
    difference: dict[str, Any],
    *,
    max_abs: float | str | None,
    finite: bool,
    policy: dict[str, Any],
) -> bool:
  """Returns whether an A/B drift is inside the preregistered report budget."""
  return bool(
      policy["enabled"]
      and not policy.get("warning_only", False)
      and difference.get("valid") is True
      and difference.get("differing_bytes", 0) > 0
      and finite
      and isinstance(max_abs, (int, float))
      and np.isfinite(max_abs)
      and max_abs <= policy["max_abs_limit"]
      and isinstance(difference.get("byte_fraction"), (int, float))
      and difference["byte_fraction"] <= policy["byte_fraction_limit"]
  )


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


def _action_geometry(
    *, prompt_mask: Any, action_mask: Any
) -> dict[str, Any]:
  """Summarize the logical-KV depth reached by all scored action tokens."""
  prompt = np.asarray(prompt_mask, dtype=np.bool_)
  action = np.asarray(action_mask, dtype=np.bool_)
  if (
      prompt.ndim != 2
      or action.ndim != 2
      or prompt.shape[0] != action.shape[0]
  ):
    return {"valid": False, "reason": "shape_mismatch"}
  rows, positions = np.nonzero(action)
  if rows.size == 0:
    return {"valid": False, "reason": "no_action_tokens"}
  prompt_lengths = prompt.sum(axis=1, dtype=np.int64)
  logical_kv = prompt_lengths[rows] + positions
  return {
      "valid": True,
      "min_logical_kv_prefix_length": int(logical_kv.min()),
      "max_logical_kv_prefix_length": int(logical_kv.max()),
      "rows_reaching_1686": int(np.unique(rows[logical_kv >= 1686]).size),
  }


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


def _p38_capsule_rows(record: dict[str, Any], max_rows: int) -> list[int]:
  """Returns the first unique sequence rows represented by red boundaries."""
  rows = []
  for boundary in record.get("boundaries", {}).values():
    for mismatch in boundary.get("mismatches", []):
      row = mismatch.get("sequence_row")
      if isinstance(row, int) and row not in rows:
        rows.append(row)
        if len(rows) == max_rows:
          return rows
  return rows


def _persist_p38_mismatch_capsule(
    sidecar: ObservedTrainExample,
    record: dict[str, Any],
) -> dict[str, Any] | None:
  """Persists a bounded, replayable pre-backward mismatch capsule."""
  base_path = os.environ.get(P38_MISMATCH_CAPSULE_ENV, "")
  if not base_path or not record.get("blocking_reds"):
    return None
  if not base_path.endswith(".npz"):
    raise AlignmentGateError(
        f"{P38_MISMATCH_CAPSULE_ENV} must end in .npz"
    )
  rounds = p38_diagnostic_rounds() if precheck_only_enabled() else 1
  round_index = p38_diagnostic_round_index() if rounds > 1 else 0
  path = (
      base_path
      if rounds == 1
      else f"{base_path[:-4]}.round-{round_index:06d}.npz"
  )
  try:
    max_rows = int(
        os.environ.get(P38_MISMATCH_CAPSULE_MAX_ROWS_ENV, "2")
    )
  except ValueError as exc:
    raise AlignmentGateError(
        f"{P38_MISMATCH_CAPSULE_MAX_ROWS_ENV} must be an integer"
    ) from exc
  if max_rows < 1 or max_rows > 256:
    raise AlignmentGateError(
        f"{P38_MISMATCH_CAPSULE_MAX_ROWS_ENV} must be in [1, 256]"
    )
  rows = _p38_capsule_rows(record, max_rows)
  if not rows:
    raise AlignmentGateError(
        "P38 mismatch capsule requested but no red sequence rows were localized"
    )
  prompt_ids = getattr(sidecar.train_example, "prompt_ids", None)
  if prompt_ids is None:
    raise AlignmentGateError(
        "P38 mismatch capsule requires train_example.prompt_ids"
    )
  row_arrays = {
      "prompt_ids": np.asarray(prompt_ids),
      "prompt_mask": np.asarray(sidecar.prompt_mask, dtype=np.bool_),
      "completion_ids": np.asarray(sidecar.tokens),
      "completion_valid_mask": np.asarray(
          sidecar.completion_valid_mask, dtype=np.bool_
      ),
      "action_mask": np.asarray(sidecar.action_mask, dtype=np.bool_),
      "s_decode": np.asarray(sidecar.s_decode),
      "s_prefill": np.asarray(sidecar.s_prefill),
      "t_old": np.asarray(sidecar.t_old),
      "policy_version": np.asarray(sidecar.policy_version),
      "sampling_values": np.asarray(sidecar.sampling_values),
  }
  batch_rows = np.asarray(sidecar.tokens).shape[0]
  invalid = {
      name: value.shape
      for name, value in row_arrays.items()
      if value.ndim == 0 or value.shape[0] != batch_rows
  }
  if invalid:
    raise AlignmentGateError(
        f"P38 mismatch capsule contains non-batch-aligned arrays: {invalid}"
    )
  selected = np.asarray(rows, dtype=np.int32)
  try:
    num_generations = int(os.environ.get("CANON_NUM_GENERATIONS", "0"))
  except ValueError as exc:
    raise AlignmentGateError(
        "CANON_NUM_GENERATIONS must be an integer for a P38 capsule"
    ) from exc
  if num_generations <= 0:
    raise AlignmentGateError(
        "P38 mismatch capsule requires positive CANON_NUM_GENERATIONS"
    )
  captured = {
      name: np.ascontiguousarray(value[selected])
      for name, value in row_arrays.items()
  }
  record_json = json.dumps(
      record, sort_keys=True, separators=(",", ":"), allow_nan=False
  )
  metadata = {
      "schema": "p38-frozenlake-mismatch-capsule-v1",
      "step": int(record["step"]),
      "diagnostic_round": round_index,
      "diagnostic_rounds": rounds,
      "selected_rows": rows,
      "num_generations": num_generations,
      "row_identity": [
          {
              "source_row": row,
              "batch_group_index": row // num_generations,
              "generation_index": row % num_generations,
          }
          for row in rows
      ],
      "source": sidecar.source_name,
      "record_sha256": hashlib.sha256(record_json.encode()).hexdigest(),
      "boundaries": {
          name: {
              "differing_bytes": value.get("differing_bytes"),
              "differing_elements": value.get("differing_elements"),
              "max_abs": value.get("max_abs"),
          }
          for name, value in record.get("boundaries", {}).items()
      },
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": _hash(value),
          }
          for name, value in captured.items()
      },
  }
  metadata_json = json.dumps(
      metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
  )
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  if os.path.exists(path):
    raise AlignmentGateError(
        f"P38 mismatch capsule path already exists: {path}"
    )
  temporary = f"{path}.tmp"
  try:
    with open(temporary, "xb") as capsule_file:
      np.savez_compressed(
          capsule_file,
          selected_rows=selected,
          metadata_json=np.frombuffer(metadata_json.encode(), dtype=np.uint8),
          **captured,
      )
      capsule_file.flush()
      os.fsync(capsule_file.fileno())
    os.replace(temporary, path)
    if rounds > 1:
      # Keep every red round immutable while maintaining the legacy/base path
      # as an atomic alias to the most recent red round.  A later exact round
      # therefore cannot erase an earlier incident, and outer postflight can
      # keep one stable input path without guessing which round was red.
      latest_temporary = f"{base_path}.latest.tmp"
      if os.path.exists(latest_temporary):
        raise AlignmentGateError(
            f"P38 mismatch latest-link path already exists: {latest_temporary}"
        )
      try:
        os.link(path, latest_temporary)
        os.replace(latest_temporary, base_path)
      finally:
        if os.path.exists(latest_temporary):
          os.unlink(latest_temporary)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)
  result = {
      "path": path,
      "latest_path": base_path,
      "sha256": _report_sha256(path),
      "selected_rows": rows,
      "logical_bytes": sum(value.nbytes for value in captured.values()),
  }
  print(
      "[CANON_P38_CAPSULE] "
      f"path={path} sha256={result['sha256']} rows={rows} "
      f"logical_bytes={result['logical_bytes']}",
      flush=True,
  )
  return result


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
  policy = gsm8k_ab_report_policy()
  blocking_reds: list[str] = []
  reported_reds: list[str] = []
  warning_reds: list[str] = []
  if n_action == 0:
    blocking_reds.append("N_action=0")
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
    finite = _masked_pair_is_finite(a, b, mask)
    boundaries[name] = {
        **difference,
        "max_abs": max_abs,
        "max_abs_mismatch": max_abs_mismatch,
        "finite": finite,
    }
    if difference["valid"] is not True or not finite:
      blocking_reds.append(name)
    elif difference["differing_bytes"] != 0:
      if _policy_warns(policy, name):
        warning_reds.append(name)
      elif name == "S_decode_vs_S_prefill" and _ab_drift_is_reportable(
          difference, max_abs=max_abs, finite=finite, policy=policy
      ):
        reported_reds.append(name)
      else:
        blocking_reds.append(name)
  verdict = (
      "FAIL"
      if blocking_reds
      else "PASS_WITH_ALIGNMENT_WARNINGS"
      if warning_reds
      else "PASS_WITH_REPORTED_DRIFT"
      if reported_reds
      else "PASS"
  )
  reds = blocking_reds + warning_reds + reported_reds
  record = {
      "timestamp": time.time(),
      "step": int(step),
      "verdict": verdict,
      "reds": reds,
      "blocking_reds": blocking_reds,
      "warning_reds": warning_reds,
      "reported_reds": reported_reds,
      "admission_policy": policy,
      "N_action": n_action,
      "action_geometry": _action_geometry(
          prompt_mask=sidecar.prompt_mask,
          action_mask=sidecar.action_mask,
      ),
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
  if precheck_only_enabled():
    # Frozen diagnostic rounds are a control-flow counter, not optimizer
    # steps.  They can advance while the training step remains unchanged.
    record["diagnostic_round"] = p38_diagnostic_round_index()
  report_path = os.environ.get(
      PRE_REPORT_ENV,
      "/mnt/disks/tunix-data/frozenlake/logs/pre_alignment_report.jsonl",
  )
  capsule = _persist_p38_mismatch_capsule(sidecar, record)
  if capsule is not None:
    record["mismatch_capsule"] = capsule
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
  if warning_reds:
    print(
        "[CANON_ALIGN_WARNING] boundary=pre_backward "
        f"step={step} warnings={warning_reds}",
        flush=True,
    )
  if blocking_reds and fail_closed:
    raise AlignmentGateError(
        "pre-backward alignment gate RED: "
        f"{blocking_reds}; report={report_path}"
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
  no_commit = os.environ.get("CANON_P33_NO_COMMIT", "") == "1"
  expected_skipped = 1 if (mode == "gate-only" or no_commit) else 0
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
  policy = gsm8k_ab_report_policy()
  blocking_reds: list[str] = []
  reported_reds: list[str] = []
  warning_reds: list[str] = []
  if n_action == 0:
    blocking_reds.append("N_action=0")
  canonical_c = None
  p58_native = (
      os.environ.get("CANON_P58_DEEPSWE_TIM", "") == "1"
      and os.environ.get("CANON_P58_TIM_ARM", "") == "native"
  )
  p57_stock = (
      os.environ.get("CANON_P57_RUN_KIND", "") == "train"
      and os.environ.get("CANON_P57_TIM_ARM", "") == "mismatch"
      and os.environ.get("CANON_P57_INFERENCE_REGIME", "") == "stock-fast"
  )
  if p58_native or p57_stock:
    canonical_c = {
        "mode": "native-stock-trainer",
        "canonical_engine_registered": False,
    }
  elif os.environ.get("CANON_ENGINE_MODULE_C", "") != "1":
    blocking_reds.append("CANON_ENGINE_MODULE_C!=1")
  else:
    from tunix.rl import canonical_forward  # pylint: disable=g-import-not-at-top

    canonical_c = canonical_forward.attestation()
  if sampling_values.shape != (sd.shape[0], 3):
    blocking_reds.append(
        "sampling_values_shape="
        f"{sampling_values.shape},expected={(sd.shape[0], 3)}"
    )
  elif not np.all(sampling_values == sampling_values[:1]):
    blocking_reds.append("sampling_values_vary_within_batch")
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
    max_abs: float | str = "nan"
    if a.shape == b.shape == mask.shape and n_action:
      max_abs = _json_number(
          np.max(np.abs(a.astype(np.float64)[mask] - b.astype(np.float64)[mask]))
      )
    finite = _masked_pair_is_finite(a, b, mask)
    boundaries[name] = {
        **difference,
        "max_abs": max_abs,
        "finite": finite,
    }
    if difference["valid"] is not True or not finite:
      blocking_reds.append(name)
    elif difference["differing_bytes"] != 0:
      if _policy_warns(policy, name):
        warning_reds.append(name)
      elif name == "S_decode_vs_S_prefill" and _ab_drift_is_reportable(
          difference, max_abs=max_abs, finite=finite, policy=policy
      ):
        reported_reds.append(name)
      else:
        blocking_reds.append(name)

  with np.errstate(over="ignore", invalid="ignore"):
    w = np.exp(to.astype(np.float64) - sd.astype(np.float64))
    r = np.exp(tc.astype(np.float64) - to.astype(np.float64))
    wr = w * r
  ratio_finite = bool(
      np.all(np.isfinite(w[mask]))
      and np.all(np.isfinite(r[mask]))
      and np.all(np.isfinite(wr[mask]))
  )
  ratio_stats = {}
  for ratio_name, ratio_values in (("w", w), ("r", r), ("wr", wr)):
    selected = ratio_values[mask]
    ratio_stats[ratio_name] = {
        "min": float(np.min(selected)) if selected.size and ratio_finite else None,
        "max": float(np.max(selected)) if selected.size and ratio_finite else None,
    }
  if not ratio_finite:
    blocking_reds.append("ratio_nonfinite")
  exact = {
      "w_all_exactly_1": bool(np.all(w[mask] == 1.0)),
      "r_all_exactly_1": bool(np.all(r[mask] == 1.0)),
      "wr_all_exactly_1": bool(np.all(wr[mask] == 1.0)),
  }
  ab_reported = "S_decode_vs_S_prefill" in reported_reds
  for key, ok in exact.items():
    if ok:
      continue
    if _policy_warns(policy, key):
      warning_reds.append(key)
    elif ab_reported and key in ("w_all_exactly_1", "wr_all_exactly_1"):
      reported_reds.append(key)
    else:
      blocking_reds.append(key)
  # Canonical GSM8K keeps rollout logprobs as the PPO old values.  Therefore
  # the ratio that actually reaches the loss is w*r = exp(T_current-A), while
  # r separately attests the trainer-old/current program boundary.
  clip_hits = int(np.sum((wr[mask] < 0.8) | (wr[mask] > 1.28)))
  tis_hits = int(np.sum(w[mask] > 2.0))
  if clip_hits:
    target = warning_reds if _policy_warns(policy, "clip_hits") else blocking_reds
    target.append(f"clip_hits={clip_hits}")
  if tis_hits:
    target = warning_reds if _policy_warns(policy, "tis_hits") else blocking_reds
    target.append(f"tis_hits={tis_hits}")

  grad_norm = float(np.asarray(gradient_norm))
  gradient = {
      "norm": grad_norm,
      "finite": bool(np.isfinite(grad_norm)),
      "nonzero": bool(grad_norm > 0.0),
  }
  if not gradient["finite"]:
    blocking_reds.append("gradient_nonfinite")
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
    blocking_reds.append("gradient_zero")

  delta = (tc.astype(np.float64) - sd.astype(np.float64))[mask]
  verdict = (
      "FAIL"
      if blocking_reds
      else "PASS_WITH_ALIGNMENT_WARNINGS"
      if warning_reds
      else "PASS_WITH_REPORTED_DRIFT"
      if reported_reds
      else "PASS"
  )
  reds = blocking_reds + warning_reds + reported_reds
  record = {
      "timestamp": time.time(),
      "step": int(step),
      "execution_mode": mode,
      "verdict": verdict,
      "reds": reds,
      "blocking_reds": blocking_reds,
      "warning_reds": warning_reds,
      "reported_reds": reported_reds,
      "admission_policy": policy,
      "N_action": n_action,
      "boundaries": boundaries,
      "exact": exact,
      "ratio_finite": ratio_finite,
      "ratio_stats": ratio_stats,
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
  if warning_reds:
    print(
        "[CANON_ALIGN_WARNING] boundary=post_backward "
        f"step={step} warnings={warning_reds}",
        flush=True,
    )
  if blocking_reds and fail_closed:
    raise AlignmentGateError(
        "alignment gate RED mode="
        f"{mode}: {blocking_reds}; report={report_path}"
    )
  return record
