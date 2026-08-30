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

"""Exact-token reconstruction and M15 continuity receipts."""

import hashlib
import os
from typing import Any, Mapping, Sequence

import numpy as np


M15_TOKEN_CONTINUITY_ENV = "CANON_M15_TOKEN_CONTINUITY"

_M15_FULL_IDENTITY = {
    "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    "CANON_PROFILE_FILE": (
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
    ),
    "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-v1-hp",
    "CANON_V1_HP_FULL": "1",
    "CANON_P57_TIM_ARM": "zero",
    "CANON_P57_RUN_KIND": "train",
    "CANON_P57_EXPECTED_UPDATES": "300",
    "CANON_P57_STOP_AFTER_STEP": "300",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "full",
    "CANON_P33_NO_COMMIT": "0",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_CKPT_MODE": "disabled",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "1",
    "CANON_DP_SIZE": "8",
    "CANON_TP_SIZE": "8",
}

_M15_APC_DEBUG_PROFILE = (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env"
)
_M15_APC_DEBUG_IDENTITY = {
    "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    "CANON_PROFILE_FILE": _M15_APC_DEBUG_PROFILE,
    "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-apc-debug",
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "backward-no-commit",
    "CANON_P33_NO_COMMIT": "1",
    "CANON_P38_PRECHECK_ONLY": "1",
    "CANON_P38_CONTROLLED_EXIT": "1",
    "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
    "CANON_P38_DURABILITY_PROFILE": "m15-wide-v1",
    "CANON_P38_SEAM_OBSERVER": "layer",
    "CANON_P38_TAIL_OBSERVER": "1",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_DP_SIZE": "8",
    "CANON_TP_SIZE": "8",
}
_M15_APC_DEBUG_ABSENT = (
    "CANON_P57_TIM_ARM",
    "CANON_P57_RUN_KIND",
    "CANON_P57_EXPECTED_UPDATES",
    "CANON_P57_STOP_AFTER_STEP",
    "CANON_FROZENLAKE_CKPT_MODE",
)
_M15_ONEHOST_IDENTITY = {
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "backward-no-commit",
    "CANON_P33_NO_COMMIT": "1",
    "CANON_P38_PRECHECK_ONLY": "1",
    "CANON_P38_CONTROLLED_EXIT": "1",
    "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
    "CANON_P38_ONEHOST_REHEARSAL": "1",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_DP_SIZE": "1",
    "CANON_TP_SIZE": "4",
}
_M15_ONEHOST_ABSENT = (
    "CANON_P32_WORKLOAD",
    "CANON_PROFILE_FILE",
    "CANON_PROFILE",
    "CANON_APC_M15_TARGET_DEBUG",
    "CANON_P57_TIM_ARM",
    "CANON_P57_RUN_KIND",
    "CANON_P57_EXPECTED_UPDATES",
    "CANON_P57_STOP_AFTER_STEP",
    "CANON_FROZENLAKE_CKPT_MODE",
    "CANON_P38_DURABILITY_PROFILE",
    "CANON_P38_SEAM_OBSERVER",
    "CANON_P38_TAIL_OBSERVER",
)


def m15_token_continuity_mode(
    values: Mapping[str, str] | None = None,
) -> str | None:
  """Returns the admitted M15 continuity mode, failing closed on drift."""
  env = os.environ if values is None else values
  if M15_TOKEN_CONTINUITY_ENV not in env:
    return None
  mode = env[M15_TOKEN_CONTINUITY_ENV]
  if mode not in ("verify", "exact"):
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY must be absent, 'verify', or 'exact'"
    )
  onehost_identity = env.get("CANON_P38_ONEHOST_REHEARSAL") == "1"
  debug_identity = env.get("CANON_PROFILE_FILE") == _M15_APC_DEBUG_PROFILE
  identity = (
      _M15_ONEHOST_IDENTITY
      if onehost_identity
      else _M15_APC_DEBUG_IDENTITY
      if debug_identity
      else _M15_FULL_IDENTITY
  )
  drift = {
      name: (env.get(name), expected)
      for name, expected in identity.items()
      if env.get(name) != expected
  }
  if onehost_identity:
    apc = env.get("CANON_VLLM_ENABLE_PREFIX_CACHING")
    admitted_apc = ("0", "1") if mode == "exact" else ("0",)
    if apc not in admitted_apc:
      drift["CANON_VLLM_ENABLE_PREFIX_CACHING"] = (
          apc,
          "|".join(admitted_apc),
      )
    for name in _M15_ONEHOST_ABSENT:
      if env.get(name) not in (None, ""):
        drift[name] = (env.get(name), "absent")
  elif debug_identity:
    if mode != "exact":
      raise ValueError("M15 APC debug admits exact token continuity only")
    arm = env.get("CANON_APC_M15_TARGET_DEBUG")
    if arm not in ("off", "on"):
      drift["CANON_APC_M15_TARGET_DEBUG"] = (arm, "off|on")
    for name in _M15_APC_DEBUG_ABSENT:
      if env.get(name) not in (None, ""):
        drift[name] = (env.get(name), "absent")
  elif mode != "exact":
    raise ValueError("M15 full training admits exact continuity only")
  if drift:
    details = ", ".join(
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in sorted(drift.items())
    )
    raise ValueError(
        f"M15 token-continuity {mode} is outside its registered identity: "
        + details
    )
  forbidden_checkpoint_values = "".join(
      env.get(name, "")
      for name in (
          "CANON_FROZENLAKE_CKPT_ROOT",
          "CANON_FROZENLAKE_CKPT_TAG",
          "CANON_FROZENLAKE_CKPT_INTERVAL",
          "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP",
          "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
      )
  )
  if forbidden_checkpoint_values:
    raise ValueError(
        f"M15 token-continuity {mode} requires its checkpoint-free "
        "registered run identity"
    )
  return mode


def _integer_vector(value: Any, *, field: str) -> np.ndarray:
  array = np.asarray(value)
  if array.ndim != 1 or array.dtype.kind not in "iu":
    raise TypeError(f"{field} must be a 1-D integer array")
  if np.any(array < 0):
    raise ValueError(f"{field} contains a negative token id")
  if np.any(array > np.iinfo(np.int32).max):
    raise ValueError(f"{field} contains a token id outside int32")
  return np.asarray(array, dtype=np.int32)


def reconstruct_continuation_prompt_tokens(
    trajectory: Any,
    response_token_count: int,
    *,
    contract: str,
) -> np.ndarray:
  """Reconstructs the exact token stream sampled across completed turns."""
  if not trajectory.steps:
    raise RuntimeError(f"{contract} token continuity requires a completed turn")
  if (
      not isinstance(response_token_count, (int, np.integer))
      or int(response_token_count) < 0
  ):
    raise ValueError(f"{contract} response token counter must be nonnegative")

  raw_prompt = _integer_vector(
      getattr(trajectory, "prompt_tokens", None),
      field=f"{contract} trajectory prompt tokens",
  )
  prompt_length = getattr(trajectory, "prompt_length", None)
  if (
      not isinstance(prompt_length, (int, np.integer))
      or not 0 < int(prompt_length) <= raw_prompt.size
  ):
    raise ValueError(
        f"{contract} trajectory prompt length is absent or outside its token "
        "width"
    )
  parts = [raw_prompt[-int(prompt_length):]]

  for step_index, step in enumerate(trajectory.steps):
    assistant_tokens = getattr(step, "assistant_tokens", None)
    if assistant_tokens is None:
      raise ValueError(
          f"{contract} turn {step_index} has no exact sampled assistant tokens"
      )
    parts.append(
        _integer_vector(
            assistant_tokens,
            field=f"{contract} turn {step_index} assistant tokens",
        )
    )

    env_tokens = getattr(step, "env_tokens", None)
    if env_tokens is None:
      if not bool(getattr(step, "done", False)):
        raise ValueError(
            f"{contract} nonterminal turn {step_index} has no environment "
            "tokens"
        )
      continue
    parts.append(
        _integer_vector(
            env_tokens,
            field=f"{contract} turn {step_index} environment tokens",
        )
    )

  prompt_token_ids = np.concatenate(parts, axis=0)
  expected = int(prompt_length) + int(response_token_count)
  if prompt_token_ids.size != expected:
    raise ValueError(
        f"{contract} exact prompt width differs from the trajectory response "
        f"counter: {prompt_token_ids.size} vs {expected}"
    )
  return prompt_token_ids


def unpadded_rollout_prompt_tokens(rollout_output: Any) -> np.ndarray:
  """Extracts the single prompt actually consumed by the rollout worker."""
  raw_prompts = np.asarray(rollout_output.left_padded_prompt_tokens)
  lengths = np.asarray(rollout_output.prompt_lengths)
  if raw_prompts.ndim != 2 or raw_prompts.shape[0] != 1:
    raise ValueError(
        "M15 token observer expected one 2-D left-padded prompt, got "
        f"{raw_prompts.shape}"
    )
  if raw_prompts.dtype.kind not in "iu":
    raise TypeError("M15 rollout prompt tokens must be integers")
  if lengths.shape != (1,) or lengths.dtype.kind not in "iu":
    raise ValueError(
        "M15 token observer expected one integer prompt length, got "
        f"shape={lengths.shape} dtype={lengths.dtype}"
    )
  prompt_length = int(lengths[0])
  if not 0 < prompt_length <= raw_prompts.shape[1]:
    raise ValueError(
        "M15 rollout prompt length is outside its padded token width: "
        f"{prompt_length} vs {raw_prompts.shape[1]}"
    )
  return _integer_vector(
      raw_prompts[0, -prompt_length:], field="M15 rollout prompt tokens"
  )


def _digest(tokens: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()


def continuity_receipt(
    actual: Sequence[int] | np.ndarray,
    expected: Sequence[int] | np.ndarray,
    *,
    turn: int,
    mode: str = "verify",
) -> str:
  """Builds a bounded, token-content-free equality receipt."""
  if mode not in ("verify", "exact"):
    raise ValueError(f"unsupported M15 token-continuity mode: {mode!r}")
  actual_tokens = _integer_vector(actual, field="M15 actual prompt tokens")
  expected_tokens = _integer_vector(
      expected, field="M15 expected prompt tokens"
  )
  common = min(actual_tokens.size, expected_tokens.size)
  unequal = np.flatnonzero(actual_tokens[:common] != expected_tokens[:common])
  if unequal.size:
    first_mismatch = int(unequal[0])
    actual_token = str(int(actual_tokens[first_mismatch]))
    expected_token = str(int(expected_tokens[first_mismatch]))
  elif actual_tokens.size != expected_tokens.size:
    first_mismatch = common
    actual_token = (
        str(int(actual_tokens[common]))
        if common < actual_tokens.size
        else "NA"
    )
    expected_token = (
        str(int(expected_tokens[common]))
        if common < expected_tokens.size
        else "NA"
    )
  else:
    first_mismatch = -1
    actual_token = "NA"
    expected_token = "NA"
  equal = first_mismatch == -1
  verdict = "TOKEN_STREAM_EQUAL" if equal else "TOKEN_STREAM_DIFFERENT"
  return (
      "[CANON_M15_TOKEN_CONTINUITY] "
      f"mode={mode} turn={turn} verdict={verdict} "
      f"actual_tokens={actual_tokens.size} "
      f"expected_tokens={expected_tokens.size} "
      f"actual_sha256={_digest(actual_tokens)} "
      f"expected_sha256={_digest(expected_tokens)} "
      f"first_mismatch={first_mismatch} actual_token={actual_token} "
      f"expected_token={expected_token}"
  )


def token_streams_equal(
    actual: Sequence[int] | np.ndarray,
    expected: Sequence[int] | np.ndarray,
) -> bool:
  """Returns exact token-stream equality after validating both operands."""
  actual_tokens = _integer_vector(actual, field="M15 actual prompt tokens")
  expected_tokens = _integer_vector(
      expected, field="M15 expected prompt tokens"
  )
  return bool(np.array_equal(actual_tokens, expected_tokens))
