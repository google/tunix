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

"""Exact-token reconstruction and observer-only continuity receipts."""

import hashlib
import os
from typing import Any, Mapping, Sequence

import numpy as np


M15_TOKEN_CONTINUITY_ENV = "CANON_M15_TOKEN_CONTINUITY"

_M15_VERIFY_IDENTITY = {
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


def m15_token_continuity_mode(
    values: Mapping[str, str] | None = None,
) -> str | None:
  """Returns the admitted M15 observer mode, failing closed on drift.

  Exact input is deliberately not admitted here. It changes model input and is
  reserved until a real M15 verify receipt proves token-stream divergence.
  """
  env = os.environ if values is None else values
  if M15_TOKEN_CONTINUITY_ENV not in env:
    return None
  mode = env[M15_TOKEN_CONTINUITY_ENV]
  if mode == "exact":
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY=exact is reserved until M15 verify "
        "evidence admits the numerical input change"
    )
  if mode != "verify":
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY must be absent or exactly 'verify'"
    )
  drift = {
      name: (env.get(name), expected)
      for name, expected in _M15_VERIFY_IDENTITY.items()
      if env.get(name) != expected
  }
  if drift:
    details = ", ".join(
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in sorted(drift.items())
    )
    raise ValueError(
        "M15 token-continuity verify is outside its registered identity: "
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
        "M15 token-continuity verify requires the checkpoint-free concept "
        "run identity"
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
) -> str:
  """Builds a bounded, token-content-free equality receipt."""
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
      f"mode=verify turn={turn} verdict={verdict} "
      f"actual_tokens={actual_tokens.size} "
      f"expected_tokens={expected_tokens.size} "
      f"actual_sha256={_digest(actual_tokens)} "
      f"expected_sha256={_digest(expected_tokens)} "
      f"first_mismatch={first_mismatch} actual_token={actual_token} "
      f"expected_token={expected_token}"
  )
