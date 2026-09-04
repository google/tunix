#!/usr/bin/env python3
"""Admission and fail-closed tests for M15 token continuity."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

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
      "CANON_EXPECT_COMMIT": "a" * 40,
      "CANON_CLIENT_IMAGE": "example/image@sha256:" + "b" * 64,
  }


def _p57_environment(workload: str, mode: str = "exact") -> dict[str, str]:
  values = _environment("exact")
  values.pop(token_continuity.M15_TOKEN_CONTINUITY_ENV)
  values[token_continuity.P57_TOKEN_CONTINUITY_ENV] = mode
  if workload == "p45":
    values["CANON_P57_WORKLOAD_CANDIDATE"] = ""
    values["CANON_P57_DATA_SPLIT"] = ""
  elif workload != "m15":
    raise ValueError(f"unsupported test workload: {workload}")
  return values


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


def _p57_collect_environment(workload: str) -> dict[str, str]:
  state = "/tmp/canon-state/p57-tito-test"
  values = _p57_environment(workload)
  values.update({
      "CANON_PROFILE_FILE": (
          "cluster/profiles/"
          "qwen3-8b-dp8-tp8-frozenlake-tito-diagnostic.env"
      ),
      "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-tito-diagnostic",
      "CANON_V1_HP_FULL": "0",
      "CANON_P57_RUN_KIND": "tito-diagnostic",
      "CANON_P57_TIM_ARM": "zero",
      "CANON_P57_EXPECTED_UPDATES": "1",
      "CANON_P57_STOP_AFTER_STEP": "1",
      "CANON_P33_RUN_STAGE": "rollout-only",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P57_TITO_ROLLOUT_ONLY": "1",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_STATE": state,
      token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV: "collect-64",
      token_continuity.P57_TITO_RUNNER_WITNESS_DIR_ENV: (
          f"{state}/p57_tito_witness/runner"
      ),
  })
  return values


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


def _p57_tito_neutrality_environment(
    arm: str, *, state: str = "/tmp/canon-state/p57-tito-neutrality"
) -> dict[str, str]:
  values = {
      token_continuity.P57_TOKEN_CONTINUITY_ENV: "exact",
      token_continuity.P57_TITO_ONEHOST_NEUTRALITY_ENV: arm,
      "CANON_V1_HP_FULL": "0",
      "CANON_P57_TIM_ARM": "zero",
      "CANON_P57_RUN_KIND": "train",
      "CANON_P57_EXPECTED_UPDATES": "3",
      "CANON_P57_STOP_AFTER_STEP": "3",
      "CANON_P57_WORKLOAD_CANDIDATE": "",
      "CANON_P57_DATA_SPLIT": "",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
      "CANON_P31_ENABLE_EVAL": "0",
      "CANON_FROZENLAKE_CKPT_MODE": "disabled",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_DP_SIZE": "1",
      "CANON_TP_SIZE": "4",
      "CANON_EXPECT_COMMIT": "a" * 40,
      "CANON_CLIENT_IMAGE": "example/image@sha256:" + "b" * 64,
      "CANON_STATE": state,
  }
  if arm == "on":
    values[token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV] = "record-full"
  return values


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


def _multiturn_trajectory():
  return types.SimpleNamespace(
      # The leading zeros model the exact left-padded sampler submission. The
      # prompt_length boundary, not token value, decides what reaches B/C.
      prompt_tokens=np.asarray([0, 0, 101, 102], dtype=np.int32),
      prompt_length=2,
      steps=[
          types.SimpleNamespace(
              assistant_tokens=np.asarray([201, 202], dtype=np.int32),
              env_tokens=np.asarray([301], dtype=np.int32),
              done=False,
          ),
          types.SimpleNamespace(
              assistant_tokens=np.asarray([203], dtype=np.int32),
              env_tokens=np.asarray([302, 303], dtype=np.int32),
              done=False,
          ),
          types.SimpleNamespace(
              assistant_tokens=np.asarray([204, 205], dtype=np.int32),
              env_tokens=None,
              done=True,
          ),
      ],
  )


class M15TokenContinuityTest(unittest.TestCase):

  def test_p57_tito_onehost_neutrality_is_a_closed_exact_p45_identity(self):
    for arm in ("off", "on"):
      values = _p57_tito_neutrality_environment(arm)
      with self.subTest(arm=arm):
        contract = token_continuity.frozenlake_token_continuity(values)
        self.assertEqual(contract.workload, "p45")
        self.assertEqual(
            token_continuity.frozenlake_token_continuity_debug_mode(values),
            "record-full" if arm == "on" else None,
        )
    for field, value in (
        ("CANON_DP_SIZE", "2"),
        ("CANON_TP_SIZE", "2"),
        ("CANON_P57_EXPECTED_UPDATES", "300"),
        ("CANON_VLLM_ENABLE_PREFIX_CACHING", "1"),
        ("CANON_PROFILE", "foreign"),
    ):
      values = _p57_tito_neutrality_environment("on")
      values[field] = value
      with self.subTest(field=field), self.assertRaises(ValueError):
        token_continuity.frozenlake_token_continuity(values)
    missing_debug = _p57_tito_neutrality_environment("on")
    missing_debug.pop(token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV)
    with self.assertRaisesRegex(ValueError, "requires record-full"):
      token_continuity.frozenlake_token_continuity_debug_mode(missing_debug)

  def test_generic_first_diff_debug_is_exact_full_only(self):
    for workload in ("p45", "m15"):
      values = _p57_environment(workload)
      values[token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV] = "first-diff"
      with self.subTest(workload=workload):
        self.assertTrue(
            token_continuity.frozenlake_token_continuity_debug_enabled(values)
        )
    self.assertFalse(
        token_continuity.frozenlake_token_continuity_debug_enabled({})
    )
    for values in (
        {
            token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV: "first-diff"
        },
        {
            **_environment("exact"),
            token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV: "first-diff",
        },
        {
            **_p57_environment("p45"),
            token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV: "0",
        },
    ):
      with self.assertRaises(ValueError):
        token_continuity.frozenlake_token_continuity_debug_enabled(values)

  def test_collect_64_is_exact_rollout_only_and_path_bound(self):
    for workload in ("p45", "m15"):
      values = _p57_collect_environment(workload)
      with self.subTest(workload=workload):
        self.assertEqual(
            token_continuity.frozenlake_token_continuity_debug_mode(values),
            "collect-64",
        )
        self.assertEqual(
            token_continuity.frozenlake_token_continuity(values).workload,
            workload,
        )

    for field, value in (
        ("CANON_P57_TITO_ROLLOUT_ONLY", "0"),
        ("CANON_P33_RUN_STAGE", "full"),
        ("CANON_P33_NO_COMMIT", "0"),
        ("CANON_VLLM_ENABLE_PREFIX_CACHING", "1"),
        (
            token_continuity.P57_TITO_RUNNER_WITNESS_DIR_ENV,
            "/tmp/canon-state/p57-tito-test/wrong",
        ),
    ):
      values = _p57_collect_environment("m15")
      values[field] = value
      with self.subTest(field=field), self.assertRaises(ValueError):
        token_continuity.frozenlake_token_continuity_debug_mode(values)

  def test_prompt_witness_persistence_is_atomic_bounded_metadata(self):
    witness = types.SimpleNamespace(
        request_id="request-17",
        submitted_tokens=3,
        submitted_sha256="1" * 64,
        engine_echo_tokens=3,
        engine_echo_sha256="1" * 64,
    )
    record = token_continuity.prompt_token_witness_record(
        witness,
        workload="p45",
        trajectory_id="a" * 32,
        turn=2,
        pair_index=3,
        group_id=4,
    )
    self.assertTrue(record["submitted_equals_engine_echo"])
    self.assertNotIn("prompt_token_ids", record)
    self.assertNotIn("submitted_token_ids", record)
    self.assertNotIn("engine_echo_token_ids", record)
    with tempfile.TemporaryDirectory() as tmp:
      path, digest, size = token_continuity.write_prompt_token_witness(
          record, state_dir=tmp
      )
      payload = path.read_bytes()
      self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
      self.assertEqual(len(payload), size)
      self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
      self.assertEqual(json.loads(payload), record)
      with self.assertRaisesRegex(FileExistsError, "duplicate"):
        token_continuity.write_prompt_token_witness(record, state_dir=tmp)

  def test_first_diff_capsule_round_trip_and_corruption_negative(self):
    trajectory = _trajectory()
    expected = token_continuity.reconstruct_continuation_prompt_tokens(
        trajectory, 4, contract="P45"
    )
    actual = expected.copy()
    actual[2] = 999
    lines = token_continuity.continuity_debug_receipts(
        trajectory,
        actual,
        expected,
        turn=1,
        workload="p45",
        pair_index=7,
        group_id=9,
        chunk_size=2,
    )
    self.assertTrue(
        lines[0].startswith("[CANON_P57_TOKEN_CONTINUITY_DEBUG] ")
    )
    production_lines = token_continuity.continuity_debug_receipts(
        trajectory,
        actual,
        expected,
        turn=1,
        workload="p45",
    )
    self.assertLess(max(len(line.encode("utf-8")) for line in production_lines), 8192)
    capsule = token_continuity.debug_capsule_from_receipts(lines)
    self.assertEqual(capsule["header"]["first_mismatch"], 2)
    self.assertEqual(capsule["header"]["pair_index"], "7")
    self.assertEqual(capsule["actual"]["tokens"], actual.tolist())
    reconstructed = [
        token
        for segment in capsule["expected_segments"]
        for token in segment["tokens"]
    ]
    self.assertEqual(reconstructed, expected.tolist())
    m15_lines = token_continuity.continuity_debug_receipts(
        trajectory,
        actual,
        expected,
        turn=1,
        workload="m15",
        chunk_size=2,
    )
    self.assertEqual(
        token_continuity.debug_capsule_from_receipts(m15_lines)["header"][
            "workload"
        ],
        "m15",
    )
    p45_id = json.loads(lines[0].split(" ", 1)[1])["capsule_id"]
    interleaved = [lines[0], m15_lines[0]]
    for p45_line, m15_line in zip(lines[1:], m15_lines[1:], strict=True):
      interleaved.extend((m15_line, p45_line))
    with self.assertRaisesRegex(ValueError, "exactly one header"):
      token_continuity.debug_capsule_from_receipts(interleaved)
    selected = token_continuity.debug_capsule_from_receipts(
        interleaved, capsule_id=p45_id
    )
    self.assertEqual(selected, capsule)
    serialized = json.dumps(capsule, sort_keys=True)
    self.assertNotIn("model_response", serialized)
    self.assertNotIn("observation", serialized)
    with tempfile.TemporaryDirectory() as tmp:
      path, digest, size = token_continuity.write_continuity_debug_capsule(
          lines, state_dir=tmp
      )
      payload = path.read_bytes()
      self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
      self.assertEqual(len(payload), size)
      self.assertEqual(json.loads(payload), capsule)
      raw_log = Path(tmp) / "worker.log"
      raw_log.write_text("noise\n" + "\n".join(lines) + "\n")
      extracted = Path(tmp) / "extracted.json"
      completed = subprocess.run(
          [
              sys.executable,
              str(
                  ROOT
                  / "canon-zero-tim/tasks/multiturn-tito-cross-workload/"
                  "scripts/extract_first_diff_capsule.py"
              ),
              "--log",
              str(raw_log),
              "--output",
              str(extracted),
          ],
          cwd=ROOT,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(completed.returncode, 0, msg=completed.stderr)
      self.assertIn("P57_TOKEN_FIRST_DIFF_EXTRACT_PASS", completed.stdout)
      self.assertEqual(json.loads(extracted.read_bytes()), capsule)
      self.assertEqual(stat.S_IMODE(extracted.stat().st_mode), 0o600)

      raw_log.write_text("noise\n" + "\n".join(interleaved) + "\n")
      selected_output = Path(tmp) / "selected.json"
      selected_completed = subprocess.run(
          [
              sys.executable,
              str(
                  ROOT
                  / "canon-zero-tim/tasks/multiturn-tito-cross-workload/"
                  "scripts/extract_first_diff_capsule.py"
              ),
              "--log",
              str(raw_log),
              "--output",
              str(selected_output),
              "--capsule-id",
              p45_id,
          ],
          cwd=ROOT,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(
          selected_completed.returncode, 0, msg=selected_completed.stderr
      )
      self.assertEqual(json.loads(selected_output.read_bytes()), capsule)

    corrupt = list(lines)
    corrupt[-1] = corrupt[-1].replace(
        '"tokens":[301,302]', '"tokens":[301,303]'
    )
    with self.assertRaisesRegex(ValueError, "chunk hash differs"):
      token_continuity.debug_capsule_from_receipts(corrupt)
    with self.assertRaisesRegex(ValueError, "chunk count differs"):
      token_continuity.debug_capsule_from_receipts(lines[:-1])
    metadata_corrupt = list(lines)
    metadata_record = json.loads(metadata_corrupt[-1].split(" ", 1)[1])
    metadata_record["workload"] = "m15"
    metadata_corrupt[-1] = (
        "[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] "
        + json.dumps(metadata_record, sort_keys=True, separators=(",", ":"))
    )
    with self.assertRaisesRegex(ValueError, "workload differs"):
      token_continuity.debug_capsule_from_receipts(metadata_corrupt)

    parsed_header = json.loads(lines[0].split(" ", 1)[1])
    parsed_records = [json.loads(line.split(" ", 1)[1]) for line in lines[1:]]
    extra_actual = dict(parsed_records[0])
    extra_actual["segment_index"] = 1
    parsed_records.append(extra_actual)
    parsed_header["token_chunk_records"] = len(parsed_records)
    parsed_header["records_metadata_sha256"] = (
        token_continuity._debug_records_metadata_digest(parsed_records)
    )
    extra_actual_lines = [
        "[CANON_P57_TOKEN_CONTINUITY_DEBUG] "
        + json.dumps(parsed_header, sort_keys=True, separators=(",", ":")),
        *(
            "[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] "
            + json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in parsed_records
        ),
    ]
    with self.assertRaisesRegex(ValueError, "topology differs"):
      token_continuity.debug_capsule_from_receipts(extra_actual_lines)

    attribution_records = [dict(record) for record in parsed_records[:-1]]
    multi_chunk_segment = next(
        key
        for key in {
            (record["stream"], record["segment_index"])
            for record in attribution_records
        }
        if sum(
            record["stream"] == key[0] and record["segment_index"] == key[1]
            for record in attribution_records
        ) > 1
    )
    changed = False
    for record in attribution_records:
      if (
          (record["stream"], record["segment_index"]) == multi_chunk_segment
          and record["chunk_index"] == 1
      ):
        record["kind"] = "environment"
        changed = True
    self.assertTrue(changed)
    attribution_header = dict(parsed_header)
    attribution_header["token_chunk_records"] = len(attribution_records)
    attribution_header["records_metadata_sha256"] = (
        token_continuity._debug_records_metadata_digest(attribution_records)
    )
    attribution_lines = [
        "[CANON_P57_TOKEN_CONTINUITY_DEBUG] "
        + json.dumps(attribution_header, sort_keys=True, separators=(",", ":")),
        *(
            "[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] "
            + json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in attribution_records
        ),
    ]
    with self.assertRaisesRegex(ValueError, "attribution drifted"):
      token_continuity.debug_capsule_from_receipts(attribution_lines)
    with self.assertRaisesRegex(ValueError, "require unequal"):
      token_continuity.continuity_debug_receipts(
          trajectory,
          expected,
          expected,
          turn=1,
          workload="p45",
      )

  def test_first_diff_default_chunks_stay_below_worker_log_limit(self):
    maximum = np.int32(2**31 - 1)
    trajectory = types.SimpleNamespace(
        prompt_tokens=np.full(256, maximum, dtype=np.int32),
        prompt_length=256,
        steps=[types.SimpleNamespace(
            assistant_tokens=np.full(256, maximum, dtype=np.int32),
            env_tokens=np.full(256, maximum, dtype=np.int32),
            done=False,
        )],
    )
    expected = token_continuity.reconstruct_continuation_prompt_tokens(
        trajectory, 512, contract="P45"
    )
    actual = expected.copy()
    actual[-1] -= 1
    lines = token_continuity.continuity_debug_receipts(
        trajectory,
        actual,
        expected,
        turn=1,
        workload="p45",
    )
    self.assertLess(max(len(line.encode("utf-8")) for line in lines), 8192)

  def test_generic_selector_admits_exact_p45_and_m15_full_only(self):
    for workload in ("p45", "m15"):
      with self.subTest(workload=workload):
        contract = token_continuity.frozenlake_token_continuity(
            _p57_environment(workload)
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.workload, workload)
        self.assertEqual(contract.mode, "exact")
        self.assertEqual(
            contract.selector, token_continuity.P57_TOKEN_CONTINUITY_ENV
        )

  def test_generic_selector_is_absence_sensitive_and_fails_closed(self):
    for workload in ("p45", "m15"):
      for field, value in (
          (token_continuity.P57_TOKEN_CONTINUITY_ENV, ""),
          (token_continuity.P57_TOKEN_CONTINUITY_ENV, "0"),
          (token_continuity.P57_TOKEN_CONTINUITY_ENV, "verify"),
          ("CANON_P57_TIM_ARM", "mismatch"),
          ("CANON_P57_RUN_KIND", "eval"),
          ("CANON_P33_ENABLE_EVAL", "1"),
          ("CANON_FROZENLAKE_CKPT_MODE", "new"),
          ("CANON_DP_SIZE", "16"),
          ("CANON_TP_SIZE", "4"),
          ("CANON_PROFILE", "neighbor-profile"),
      ):
        values = _p57_environment(workload)
        values[field] = value
        with self.subTest(
            workload=workload, field=field, value=value
        ), self.assertRaises(ValueError):
          token_continuity.frozenlake_token_continuity(values)

    wrong_workload = _p57_environment("m15")
    wrong_workload["CANON_P57_WORKLOAD_CANDIDATE"] = "m20"
    with self.assertRaisesRegex(ValueError, "requires P45 readiness or M15"):
      token_continuity.frozenlake_token_continuity(wrong_workload)

  def test_legacy_and_generic_selectors_are_mutually_exclusive(self):
    values = _p57_environment("m15")
    values[token_continuity.M15_TOKEN_CONTINUITY_ENV] = "exact"
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      token_continuity.frozenlake_token_continuity(values)
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      token_continuity.m15_token_continuity_mode(values)

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

  def test_each_later_turn_equals_trainer_bc_prompt_prefix(self):
    frozen = _multiturn_trajectory()
    conversation = np.concatenate([
        frozen.steps[0].assistant_tokens,
        frozen.steps[0].env_tokens,
        frozen.steps[1].assistant_tokens,
        frozen.steps[1].env_tokens,
        frozen.steps[2].assistant_tokens,
    ])
    completed = 0
    for later_turn in (1, 2, 3):
      prefix = types.SimpleNamespace(
          prompt_tokens=frozen.prompt_tokens,
          prompt_length=frozen.prompt_length,
          steps=frozen.steps[:later_turn],
      )
      completed += sum(
          0 if value is None else len(value)
          for value in (
              frozen.steps[later_turn - 1].assistant_tokens,
              frozen.steps[later_turn - 1].env_tokens,
          )
      )
      reconstructed = token_continuity.reconstruct_continuation_prompt_tokens(
          prefix, completed, contract="M15"
      )
      trainer_prefix = token_continuity.trainer_bc_prompt_prefix(
          frozen.prompt_tokens,
          frozen.prompt_length,
          conversation,
          completed,
          contract="M15",
      )
      with self.subTest(later_turn=later_turn):
        np.testing.assert_array_equal(reconstructed, trainer_prefix)

    # A one-token poison in the independently assembled B/C conversation
    # must be visible at the exact injected boundary.
    poisoned = conversation.copy()
    poisoned[3] += 1
    poisoned_prefix = token_continuity.trainer_bc_prompt_prefix(
        frozen.prompt_tokens,
        frozen.prompt_length,
        poisoned,
        completed,
        contract="M15",
    )
    mismatch = np.flatnonzero(reconstructed != poisoned_prefix)
    self.assertEqual(mismatch.tolist(), [frozen.prompt_length + 3])

  def test_segment_ledger_preserves_first_prompt_submission_provenance(self):
    frozen = _multiturn_trajectory()
    one_turn = types.SimpleNamespace(
        prompt_tokens=frozen.prompt_tokens,
        prompt_length=frozen.prompt_length,
        steps=frozen.steps[:1],
    )
    segments = token_continuity.continuation_prompt_segments(
        one_turn, contract="P45"
    )
    self.assertEqual(
        [(item.kind, item.turn_index) for item in segments],
        [("initial_prompt", -1), ("assistant", 0), ("environment", 0)],
    )
    np.testing.assert_array_equal(
        segments[0].tokens,
        np.asarray([101, 102], dtype=np.int32),
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

  def test_generic_receipts_are_workload_labelled(self):
    tokens = np.asarray([101, 201, 202], dtype=np.int32)
    for workload in ("p45", "m15"):
      receipt = token_continuity.continuity_receipt(
          tokens,
          tokens,
          turn=1,
          mode="exact",
          workload=workload,
          selector=token_continuity.P57_TOKEN_CONTINUITY_ENV,
      )
      self.assertTrue(receipt.startswith("[CANON_P57_TOKEN_CONTINUITY] "))
      self.assertIn(f"workload={workload}", receipt)
      self.assertIn("verdict=TOKEN_STREAM_EQUAL", receipt)
    with self.assertRaisesRegex(ValueError, "cannot attest P45"):
      token_continuity.continuity_receipt(
          tokens,
          tokens,
          turn=1,
          mode="exact",
          workload="p45",
      )

  def test_record_full_accounts_real_work_and_unexercised_rows(self):
    with tempfile.TemporaryDirectory() as tmp:
      values = _p57_environment("p45")
      values.update({
          "CANON_P57_TOKEN_CONTINUITY_DEBUG": "record-full",
          "CANON_STATE": tmp,
      })
      token_continuity._reset_token_collection_for_test()
      token_continuity.begin_token_continuity_collection(values)
      token_continuity.record_prompt_echo_comparison(equal=True)
      token_continuity.record_token_collection_trajectory(
          different=False, later_turns=0
      )
      token_continuity.record_full_update({
          "verdict": "PASS",
          "microsteps": 8,
          "commits": 1,
          "alignment_hashes": ["a" * 64],
      })
      snapshot = token_continuity.token_collection_snapshot()
      self.assertEqual(snapshot["mode"], "record-full")
      self.assertEqual(snapshot["trajectories"], 1)
      self.assertEqual(snapshot["compared_trajectories"], 0)
      self.assertEqual(snapshot["unexercised_single_turn_trajectories"], 1)
      self.assertEqual(snapshot["backward_transactions"], 1)
      self.assertEqual(snapshot["gradient_microbatches"], 8)
      self.assertEqual(snapshot["optimizer_commits"], 1)
      self.assertEqual(snapshot["alignment_updates"], 1)
      token_continuity._reset_token_collection_for_test()

  def test_record_full_single_writer_and_update_zero_observation(self):
    rows = [{"token_different": False}, {"token_different": False}]
    with tempfile.TemporaryDirectory() as tmp:
      values = _p57_tito_neutrality_environment("on", state=tmp)
      token_continuity._reset_token_collection_for_test()
      token_continuity.begin_token_continuity_collection(values)
      writer = Path(tmp) / "p57_tito_witness/single-writer.json"
      record = json.loads(writer.read_text(encoding="utf-8"))
      self.assertEqual(record["writer_contract"], "one-python-controller-o-excl")
      self.assertEqual(record["neutrality_arm"], "on")
      self.assertEqual(stat.S_IMODE(writer.stat().st_mode), 0o600)
      admitted = token_continuity.enforce_record_full_first_update_token_admission(
          rows, step=0, values=values
      )
      self.assertEqual(admitted["verdict"], "PASS")
      self.assertTrue(admitted["continue_training"])
      self.assertIsNone(
          token_continuity.enforce_record_full_first_update_token_admission(
              [{"token_different": True}], step=1, values=values
          )
      )
      observed = token_continuity.enforce_record_full_first_update_token_admission(
          [{"token_different": True}], step=0, values=values
      )
      self.assertEqual(observed["verdict"], "OBSERVED_DIFFERENT")
      self.assertTrue(observed["continue_training"])
      with self.assertRaisesRegex(FileExistsError, "second immutable"):
        token_continuity.write_tito_single_writer_receipt(values=values)
      token_continuity._reset_token_collection_for_test()

  def test_record_full_event_reservations_are_not_collect_64_bounded(self):
    with tempfile.TemporaryDirectory() as tmp:
      values = _p57_tito_neutrality_environment("on", state=tmp)
      token_continuity._reset_token_collection_for_test()
      token_continuity.begin_token_continuity_collection(values)
      events = [
          token_continuity.reserve_record_full_token_difference_event()
          for _ in range(token_continuity.P57_TOKEN_CONTINUITY_COLLECT_LIMIT + 2)
      ]
      self.assertEqual(events, list(range(1, 67)))
      snapshot = token_continuity.token_collection_snapshot()
      self.assertEqual(snapshot["token_difference_events"], 66)
      self.assertEqual(snapshot["capsules_reserved"], 66)
      self.assertEqual(snapshot["capsules_omitted"], 0)
      token_continuity._reset_token_collection_for_test()

  def test_orbax_probe_is_independent_and_fail_closed(self):
    class Model:

      def __init__(self, value):
        self.value = np.asarray(value, dtype=np.int32)

    class Manager:

      def __init__(self, root, *, fail_restore=False):
        self.root = root
        self.fail_restore = fail_restore
        self.saved = None

      def latest_step(self):
        return None if self.saved is None else 0

      def save(self, step, model, **kwargs):
        self.saved = (np.asarray(model.value).copy(), kwargs["custom_metadata"])
        return step == 0

      def maybe_restore(self, model, *, step):
        model.value = self.saved[0].copy()
        if self.fail_restore:
          model.value[0] += 1
        return step, self.saved[1]

      def close(self):
        return None

    with tempfile.TemporaryDirectory() as tmp:
      values = _p57_environment("p45")
      values.update({
          token_continuity.P57_TOKEN_CONTINUITY_DEBUG_ENV: "record-full",
          "CANON_STATE": tmp,
          "CANON_P57_TITO_GCS_PREFIX": (
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
              "p57-tito-test/attempt-direct"
          ),
      })
      managers = []

      def manager_factory(root):
        manager = Manager(root)
        managers.append(manager)
        return manager

      record = token_continuity.run_tito_orbax_admission_probe(
          values,
          manager_factory=manager_factory,
          model_factory=Model,
          value_reader=lambda model: model.value,
      )
      self.assertEqual(record["status"], "PASS")
      self.assertTrue(record["restored_equal"])
      self.assertTrue(managers[0].root.endswith("/orbax-admission-probe"))
      receipt = Path(tmp) / "p57_tito_gcs/orbax-probe.json"
      self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)

    with tempfile.TemporaryDirectory() as tmp:
      values["CANON_STATE"] = tmp
      with self.assertRaisesRegex(RuntimeError, "Orbax admission probe failed"):
        token_continuity.run_tito_orbax_admission_probe(
            values,
            manager_factory=lambda root: Manager(root, fail_restore=True),
            model_factory=Model,
            value_reader=lambda model: model.value,
        )
      failed = json.loads(
          (Path(tmp) / "p57_tito_gcs/orbax-probe.json").read_text()
      )
      self.assertEqual(failed["status"], "FAIL")

  def test_record_full_row_map_accepts_group_chunks_and_rejects_gaps(self):
    def row(index: int) -> dict[str, object]:
      return {
          "trajectory_id": f"{index + 1:032x}",
          "request_ids": [f"request-{index}-0", f"request-{index}-1"],
          "policy_step": 0,
          "group_id": index // 2,
          "pair_index": index % 2,
          "sequence_row": index,
          "later_turns": 1,
          "token_different": False,
      }

    with tempfile.TemporaryDirectory() as tmp:
      output = token_continuity.append_full_record_batch_map(
          [row(0), row(1)], state_dir=tmp
      )
      token_continuity.append_full_record_batch_map(
          [row(2), row(3)], state_dir=tmp
      )
      records = [json.loads(line) for line in output.read_text().splitlines()]
      self.assertEqual([record["sequence_row"] for record in records], [0, 1, 2, 3])
      self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
      with self.assertRaisesRegex(ValueError, "contiguous policy group"):
        token_continuity.append_full_record_batch_map(
            [row(0), row(2)], state_dir=Path(tmp) / "bad"
        )
      invalid = row(0)
      invalid["request_ids"] = []
      with self.assertRaisesRegex(ValueError, "identity is malformed"):
        token_continuity.append_full_record_batch_map(
            [invalid], state_dir=Path(tmp) / "missing-request"
        )

  def test_actor_snapshot_request_is_consumed_before_update(self):
    class FakeManager:

      def __init__(self, root):
        self.root = root
        self.saved = None

      def save(self, step, model, **kwargs):
        self.saved = (step, model, kwargs)
        return True

      def latest_step(self):
        return self.saved[0]

      def close(self):
        return None

    with tempfile.TemporaryDirectory() as tmp:
      values = _p57_environment("p45")
      values.update({
          "CANON_P57_TOKEN_CONTINUITY_DEBUG": "record-full",
          "CANON_STATE": tmp,
          "CANON_P57_TITO_GCS_PREFIX": (
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
              "p57-tito-test/attempt-direct"
          ),
      })
      request_dir = Path(tmp) / "p57_tito_witness/actor-snapshot-requests"
      request_dir.mkdir(parents=True, mode=0o700)
      request = {
          "schema": "canon.p57-tito-actor-snapshot-request.v1",
          "status": "PENDING",
          "step": 5,
          "policy_version": 5,
          "categories": ["first-any", "first-ge-1", "first-ge-8"],
          "max_abs": 9.0,
          "sidecar_sha256": "b" * 64,
          "source_commit": "a" * 40,
          "image_identity": values["CANON_CLIENT_IMAGE"],
          "workload": "p45",
          "dp": 8,
          "tp": 8,
      }
      request_path = request_dir / "step-000005.json"
      request_path.write_text(
          json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
      )
      request_path.chmod(0o600)
      managers = []

      def manager_factory(root):
        manager = FakeManager(root)
        managers.append(manager)
        return manager

      trainer = types.SimpleNamespace(train_steps=5, model=object())
      inspection = {
          "leaves": [{"path": ".x", "shape": [2], "dtype": "float32"}],
          "leaf_count": 1,
          "logical_bytes": 8,
          "bounded_fingerprint": {"leaves": {".x": {"sha256": "c" * 64}}},
      }
      with mock.patch.dict(os.environ, values, clear=True):
        receipt = token_continuity.consume_actor_snapshot_request(
            trainer,
            step=5,
            manager_factory=manager_factory,
            state_inspector=lambda unused: inspection,
        )
      self.assertEqual(receipt["status"], "PASS")
      self.assertFalse(receipt["optimizer_included"])
      self.assertFalse(receipt["resumable"])
      self.assertEqual(receipt["image_identity"], values["CANON_CLIENT_IMAGE"])
      self.assertEqual(receipt["dp"], 8)
      self.assertEqual(receipt["tp"], 8)
      self.assertEqual(
          receipt["categories"], ["first-any", "first-ge-1", "first-ge-8"]
      )
      self.assertEqual(receipt["actor_train_steps_before"], 5)
      self.assertEqual(
          managers[0].root,
          values["CANON_P57_TITO_GCS_PREFIX"] + "/actor-snapshots",
      )
      self.assertIsNone(managers[0].saved[2]["optimizer"])
      saved_metadata = managers[0].saved[2]["custom_metadata"]
      self.assertEqual(saved_metadata["artifact_kind"], "actor-only-nonresumable")
      receipt_path = Path(tmp) / (
          "p57_tito_witness/actor-snapshot-receipts/step-000005.json"
      )
      self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
      with mock.patch.dict(os.environ, values, clear=True), self.assertRaisesRegex(
          FileExistsError, "consumed twice"
      ):
        token_continuity.consume_actor_snapshot_request(
            trainer,
            step=5,
            manager_factory=manager_factory,
            state_inspector=lambda unused: inspection,
        )


if __name__ == "__main__":
  unittest.main()
