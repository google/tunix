#!/usr/bin/env python3
"""Admission and fail-closed tests for M15 token continuity."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TOKEN_SPEC = importlib.util.spec_from_file_location(
    "m15_token_continuity", ROOT / "tunix/rl/agentic/token_continuity.py"
)
if TOKEN_SPEC is None or TOKEN_SPEC.loader is None:
  raise RuntimeError("cannot import token continuity policy")
token_continuity = importlib.util.module_from_spec(TOKEN_SPEC)
sys.modules[TOKEN_SPEC.name] = token_continuity
TOKEN_SPEC.loader.exec_module(token_continuity)


def _environment(mode: str = "verify") -> dict[str, str]:
  return {
      token_continuity.M15_TOKEN_CONTINUITY_ENV: mode,
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


def _debug_environment(
    mode: str = "exact", arm: str = "on"
) -> dict[str, str]:
  return {
      token_continuity.M15_TOKEN_CONTINUITY_ENV: mode,
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env"
      ),
      "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-apc-debug",
      "CANON_APC_M15_TARGET_DEBUG": arm,
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


def _onehost_environment(
    mode: str = "exact", apc: str = "1"
) -> dict[str, str]:
  return {
      token_continuity.M15_TOKEN_CONTINUITY_ENV: mode,
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
      "CANON_VLLM_ENABLE_PREFIX_CACHING": apc,
  }


def _trajectory(*, env_tokens=np.asarray([301, 302], dtype=np.int32)):
  return types.SimpleNamespace(
      prompt_tokens=np.asarray([0, 0, 101], dtype=np.int32),
      prompt_length=1,
      steps=[
          types.SimpleNamespace(
              assistant_tokens=np.asarray([201, 202], dtype=np.int32),
              env_tokens=env_tokens,
              done=False,
          )
      ],
  )


class M15TokenContinuityTest(unittest.TestCase):

  def test_selector_is_absence_sensitive_and_exact_identity_only(self):
    self.assertIsNone(token_continuity.m15_token_continuity_mode({}))
    self.assertEqual(
        token_continuity.m15_token_continuity_mode(_environment("exact")),
        "exact",
    )
    for field, value in (
        (token_continuity.M15_TOKEN_CONTINUITY_ENV, ""),
        (token_continuity.M15_TOKEN_CONTINUITY_ENV, "verify"),
        (token_continuity.M15_TOKEN_CONTINUITY_ENV, "unknown"),
        ("CANON_P57_WORKLOAD_CANDIDATE", ""),
        ("CANON_P57_TIM_ARM", "mismatch"),
        ("CANON_P33_ENABLE_EVAL", "1"),
        ("CANON_DP_SIZE", "16"),
    ):
      values = _environment()
      values[field] = value
      with self.subTest(field=field, value=value), self.assertRaises(ValueError):
        token_continuity.m15_token_continuity_mode(values)

  def test_exact_selector_admits_only_registered_apc_layer_rebaseline(self):
    for arm in ("off", "on"):
      self.assertEqual(
          token_continuity.m15_token_continuity_mode(
              _debug_environment(arm=arm)
          ),
          "exact",
      )
    for field, value in (
        (token_continuity.M15_TOKEN_CONTINUITY_ENV, "verify"),
        ("CANON_APC_M15_TARGET_DEBUG", ""),
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "1"),
        ("CANON_P38_DURABILITY_PROFILE", "m15-e0-kv-v1"),
        ("CANON_P38_SEAM_OBSERVER", "full"),
        ("CANON_P57_TIM_ARM", "zero"),
        ("CANON_DP_SIZE", "16"),
    ):
      values = _debug_environment()
      values[field] = value
      with self.subTest(field=field, value=value), self.assertRaises(ValueError):
        token_continuity.m15_token_continuity_mode(values)

  def test_exact_selector_admits_only_bounded_onehost_rehearsal(self):
    for apc in ("0", "1"):
      self.assertEqual(
          token_continuity.m15_token_continuity_mode(
              _onehost_environment(apc=apc)
          ),
          "exact",
      )
    self.assertEqual(
        token_continuity.m15_token_continuity_mode(
            _onehost_environment(mode="verify", apc="0")
        ),
        "verify",
    )
    for field, value in (
        ("CANON_VLLM_ENABLE_PREFIX_CACHING", ""),
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "1"),
        ("CANON_P38_ONEHOST_REHEARSAL", "0"),
        ("CANON_APC_M15_TARGET_DEBUG", "on"),
        ("CANON_P32_WORKLOAD", "frozenlake"),
        ("CANON_PROFILE_FILE", "unexpected.env"),
        ("CANON_DP_SIZE", "8"),
    ):
      values = _onehost_environment()
      values[field] = value
      with self.subTest(field=field, value=value), self.assertRaises(ValueError):
        token_continuity.m15_token_continuity_mode(values)

    with self.assertRaises(ValueError):
      token_continuity.m15_token_continuity_mode(
          _onehost_environment(mode="verify", apc="1")
      )

  def test_reconstruction_preserves_exact_turn_tokens_and_padding_tail(self):
    actual = token_continuity.reconstruct_continuation_prompt_tokens(
        _trajectory(), 4, contract="M15"
    )
    np.testing.assert_array_equal(
        actual, np.asarray([101, 201, 202, 301, 302], dtype=np.int32)
    )

  def test_missing_or_invalid_token_arrays_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "no environment tokens"):
      token_continuity.reconstruct_continuation_prompt_tokens(
          _trajectory(env_tokens=None), 2, contract="M15"
      )
    with self.assertRaisesRegex(ValueError, "negative token id"):
      token_continuity.reconstruct_continuation_prompt_tokens(
          _trajectory(env_tokens=np.asarray([-1], dtype=np.int32)),
          3,
          contract="M15",
      )
    with self.assertRaisesRegex(ValueError, "outside int32"):
      token_continuity.continuity_receipt(
          np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
          np.asarray([1], dtype=np.int32),
          turn=1,
      )

  def test_unpadding_and_bounded_different_bpe_receipt(self):
    rollout = types.SimpleNamespace(
        left_padded_prompt_tokens=np.asarray([[0, 0, 101, 28, 1725]]),
        prompt_lengths=np.asarray([3], dtype=np.int32),
    )
    actual = token_continuity.unpadded_rollout_prompt_tokens(rollout)
    np.testing.assert_array_equal(actual, np.asarray([101, 28, 1725]))
    receipt = token_continuity.continuity_receipt(
        actual,
        np.asarray([101, 97183], dtype=np.int32),
        turn=1,
    )
    self.assertIn("verdict=TOKEN_STREAM_DIFFERENT", receipt)
    self.assertIn("first_mismatch=1", receipt)
    self.assertIn("actual_token=28", receipt)
    self.assertIn("expected_token=97183", receipt)
    self.assertNotIn("[101", receipt)

  def test_equal_receipt_is_exact(self):
    tokens = np.asarray([101, 201, 202], dtype=np.int32)
    receipt = token_continuity.continuity_receipt(tokens, tokens, turn=1)
    self.assertIn("verdict=TOKEN_STREAM_EQUAL", receipt)
    self.assertIn("first_mismatch=-1", receipt)
    exact_receipt = token_continuity.continuity_receipt(
        tokens, tokens, turn=1, mode="exact"
    )
    self.assertIn("mode=exact", exact_receipt)
    self.assertTrue(token_continuity.token_streams_equal(tokens, tokens))
    self.assertFalse(
        token_continuity.token_streams_equal(tokens, tokens[:-1])
    )


if __name__ == "__main__":
  unittest.main()
