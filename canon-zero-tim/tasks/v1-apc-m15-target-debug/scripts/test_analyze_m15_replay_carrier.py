#!/usr/bin/env python3
"""Contract tests for the M15 replay-input analyzer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


analysis = _load("analyze_m15_replay_carrier", SCRIPT_DIR / "analyze_m15_replay_carrier.py")
carrier = _load("package_full_replay_carrier", SCRIPT_DIR / "package_full_replay_carrier.py")


def _token_sha(tokens: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayAnalysisTest(unittest.TestCase):

  def _inputs(self, root: Path):
    rows = 256
    prompt_ids = np.zeros((rows, 3), dtype=np.int32)
    completion_ids = np.zeros((rows, 4), dtype=np.int32)
    for row in range(rows):
      prompt_ids[row] = [1000 + row, 2000 + row, 3000 + row]
      completion_ids[row] = [4000 + row, 5000 + row, 6000 + row, 7000 + row]
    prompt_mask = np.ones_like(prompt_ids, dtype=np.bool_)
    completion_valid = np.ones_like(completion_ids, dtype=np.bool_)
    action_mask = np.ones_like(completion_ids, dtype=np.bool_)
    s_prefill = np.zeros((rows, 4), dtype=np.float32)
    s_decode = s_prefill.copy()
    s_decode[201, 0] = np.float32(-5.0)
    s_decode[245, 0] = np.float32(-3.0)
    t_old = s_prefill.copy()
    policy = np.zeros((rows, 1), dtype=np.int32)
    sampling = np.zeros((rows, 1), dtype=np.float32)
    arrays = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_valid_mask": completion_valid,
        "action_mask": action_mask,
        "s_decode": s_decode,
        "s_prefill": s_prefill,
        "t_old": t_old,
        "policy_version": policy,
        "sampling_values": sampling,
    }
    source = "7" * 40
    metadata = {
        "schema": "m15-apc-producer-unit-v1",
        "arm": "on",
        "source_commit": source,
        "rows": rows,
        "prompt_groups": 32,
        "num_generations": 8,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": carrier._array_sha256(value),
            }
            for name, value in arrays.items()
        },
    }
    producer = root / "m15_producer_unit.npz"
    np.savez_compressed(
        producer,
        source_rows=np.arange(rows, dtype=np.int32),
        metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
        **arrays,
    )

    h245 = np.concatenate((prompt_ids[245], completion_ids[245]))
    h201 = np.concatenate((prompt_ids[201], completion_ids[201]))
    h0 = np.concatenate((prompt_ids[0], completion_ids[0]))
    records = [
        self._record(1, "A", "standard", "a245", h245, 3, 0, 3, [11]),
        self._record(2, "A", "continue_decode", "a245", h245, 4, 3, 1, [11]),
        self._record(3, "A", "standard", "a201", h201, 3, 0, 3, [12]),
        self._record(4, "A", "continue_decode", "a201", h201, 4, 3, 1, [12]),
        self._record(5, "A", "continue_decode", "a245", h245, 5, 4, 1, [11, 20]),
        self._record(6, "A", "standard", "a245", h245, 6, 5, 1, [11, 20]),
        self._record(7, "B", "standard", "b0", h0, 3, 0, 3, [19]),
    ]
    envelope = root / "m15_replay_envelope.jsonl"
    envelope.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    valid = action_mask & completion_valid
    ab_elements, ab_bytes, _ = analysis._byte_difference(s_decode, s_prefill, valid)
    m15 = root / "m15.json"
    m15.write_text(json.dumps({
        "status": "FRESH_TARGET_RED_FROZEN",
        "arm": "on",
        "source_commit": source,
        "a_b_differing_bytes": [ab_bytes],
        "a_b_differing_elements": [int(np.count_nonzero(ab_elements))],
        "b_c_differing_bytes": [0],
    }), encoding="utf-8")
    first = root / "first.json"
    first.write_text(json.dumps({
        "status": "FIRST_RED_ROW_FROZEN",
        "source_commit": source,
        "source_row": 245,
        "first_incident": {
            "source_row": 245,
            "completion_position": 0,
            "call_index": 6,
            "request_id": "a245",
            "num_computed_tokens": 5,
            "dp_rank": 0,
            "local_scheduler_slot": 0,
        },
    }), encoding="utf-8")
    replay = root / "replay.json"
    replay.write_text(json.dumps({
        "status": "FULL_REPLAY_CARRIER_FROZEN",
        "source_commit": source,
        "producer_rows": 256,
        "serving_call_count": 7,
        "request_count": 3,
        "program_paths_by_arm": {"A": ["continue_decode", "standard"], "B": ["standard"]},
        "first_red": {"source_row": 245, "call_index": 6},
    }), encoding="utf-8")
    audit = root / "audit.json"
    audit.write_text(json.dumps({
        "status": "FRESH_TARGET_RED_FROZEN",
        "source_commit": source,
        "source_gcs_uri": (
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
            "canon-test-analysis/attempt-0"
        ),
    }), encoding="utf-8")
    return producer, envelope, first, replay, m15, audit

  def _record(self, call, arm, path, request_id, history, length, computed, scheduled, pages):
    request = {
        "request_id": request_id,
        "request_index": 0,
        "dp_rank": 0,
        "local_scheduler_slot": 0,
        "scheduled_tokens": scheduled,
        "num_computed_tokens": computed,
        "num_prompt_tokens": 3,
        "num_tokens": length,
        "token_history_sha256": _token_sha(history[:length]),
        "request_kind": "prefill" if computed == 0 else "decode",
        "block_size": 4,
        "logical_blocks_before": (computed + 3) // 4,
        "logical_blocks_after": (computed + scheduled + 3) // 4,
        "physical_pages": pages,
    }
    return {
        "schema": "m15-apc-serving-envelope-v1",
        "arm": "on",
        "serving_arm": arm,
        "call_index": call,
        "program_path": path,
        "request_order": [request_id],
        "requests": [request],
    }

  def _analyze(self, root: Path):
    producer, envelope, first, replay, m15, audit = self._inputs(root)
    return analysis.analyze(
        producer_path=producer,
        envelope_path=envelope,
        first_red_contract_path=first,
        replay_contract_path=replay,
        m15_classification_path=m15,
        upstream_audit_receipt_path=audit,
        source_gcs_uri=(
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
            "canon-test-analysis/attempt-0"
        ),
        output_dir=root / "output",
    )

  def test_positive_distinguishes_onset_from_captured_incident(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = self._analyze(root)
      self.assertEqual(result["status"], "M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED")
      self.assertEqual(result["coordinates"]["canonical_first_mismatch"]["source_row"], 201)
      self.assertEqual(result["coordinates"]["canonical_first_mismatch_request"]["first_call"], 3)
      self.assertEqual(result["coordinates"]["earliest_red_request"]["source_row"], 245)
      self.assertEqual(result["coordinates"]["earliest_red_request"]["first_request"]["first_call"], 1)
      self.assertEqual(result["coordinates"]["first_fully_captured_incident"]["call_index"], 6)
      self.assertEqual(result["carrier"]["replay_prefix_end_call"], 4)
      self.assertTrue((root / "output/SHA256SUMS").is_file())

  def test_rejects_classification_count_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      producer, envelope, first, replay, m15, audit = self._inputs(root)
      value = json.loads(m15.read_text())
      value["a_b_differing_bytes"] = [value["a_b_differing_bytes"][0] + 1]
      m15.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(analysis.AnalysisError, "A-B byte count"):
        analysis.analyze(
            producer_path=producer,
            envelope_path=envelope,
            first_red_contract_path=first,
            replay_contract_path=replay,
            m15_classification_path=m15,
            upstream_audit_receipt_path=audit,
            source_gcs_uri=(
                "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
                "canon-test-analysis/attempt-0"
            ),
            output_dir=root / "output",
        )

  def test_rejects_b_c_red(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      producer, envelope, first, replay, m15, audit = self._inputs(root)
      with np.load(producer, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
      values["t_old"][201, 0] = np.float32(1.0)
      metadata = json.loads(values["metadata_json"].tobytes().decode())
      metadata["arrays"]["t_old"]["sha256"] = carrier._array_sha256(values["t_old"])
      values["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
      np.savez_compressed(producer, **values)
      target = json.loads(m15.read_text())
      _, target["b_c_differing_bytes"][0], _ = analysis._byte_difference(
          values["s_prefill"], values["t_old"], values["action_mask"]
      )
      m15.write_text(json.dumps(target), encoding="utf-8")
      with self.assertRaisesRegex(analysis.AnalysisError, "B-C is red"):
        analysis.analyze(
            producer_path=producer,
            envelope_path=envelope,
            first_red_contract_path=first,
            replay_contract_path=replay,
            m15_classification_path=m15,
            upstream_audit_receipt_path=audit,
            source_gcs_uri=(
                "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
                "canon-test-analysis/attempt-0"
            ),
            output_dir=root / "output",
        )

  def test_gcs_wrapper_prepares_and_seals_derived_plan(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      capture = root / "capture"
      capture.mkdir()
      producer, envelope, first, replay, m15, _ = self._inputs(capture)
      first_dir = capture / "m15_first_red_replay"
      full_dir = capture / "m15_full_replay_carrier"
      first_dir.mkdir()
      full_dir.mkdir()
      shutil.copyfile(first, first_dir / "first_red_contract.json")
      shutil.copyfile(replay, full_dir / "replay_contract.json")
      (full_dir / "request_row_joins.jsonl").write_text("{}\n", encoding="utf-8")
      shutil.copyfile(m15, capture / "m15_apc_target.classification.json")
      for nested, names in (
          (first_dir, ("first_red_contract.json",)),
          (full_dir, ("replay_contract.json", "request_row_joins.jsonl")),
      ):
        (nested / "SHA256SUMS").write_text(
            "".join(f"{_file_sha(nested / name)}  {name}\n" for name in names),
            encoding="utf-8",
        )

      source_uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          "canon-test-prepare/attempt-0"
      )
      source = root / "source"
      source.mkdir()
      source_commit = "7" * 40
      for name, status in (
          ("PREFLIGHT.json", "writable"),
          ("COLLECTED.json", "collected"),
          ("COMPLETE.json", "postflight-accepted"),
      ):
        (source / name).write_text(json.dumps({
            "prefix": source_uri,
            "source_commit": source_commit,
            "status": status,
        }), encoding="utf-8")
      (source / "serving-classification.json").write_text(
          json.dumps({"verdict": "PASS"}), encoding="utf-8"
      )
      (source / "run.log").write_text(
          "[CANON_ALIGN_PRE] verdict=FAIL\n", encoding="utf-8"
      )
      with tarfile.open(source / "serving-capture.tar", "w") as archive:
        for path in sorted(capture.rglob("*")):
          if path.is_file():
            archive.add(path, arcname=path.relative_to(capture))
      manifest_names = ("serving-capture.tar", "serving-classification.json", "run.log")
      (source / "SHA256SUMS").write_text(
          "".join(f"{_file_sha(source / name)}  {name}\n" for name in manifest_names),
          encoding="utf-8",
      )

      fake_root = root / "fake-gcs"
      remote = fake_root / source_uri[5:]
      remote.parent.mkdir(parents=True)
      shutil.copytree(source, remote)
      fake_bin = root / "bin"
      fake_bin.mkdir()
      fake_gcloud = fake_bin / "gcloud"
      fake_gcloud.write_text(
          """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

root = Path(os.environ["FAKE_GCS_ROOT"])
def resolve(value):
  return root / value[5:] if value.startswith("gs://") else Path(value)
args = sys.argv[1:]
if args[:2] == ["storage", "ls"]:
  raise SystemExit(0 if resolve(args[2]).exists() else 1)
if args[:2] == ["storage", "cp"]:
  source, destination = resolve(args[2]), resolve(args[3])
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(source, destination)
  raise SystemExit(0)
if args[:3] == ["storage", "rsync", "--recursive"]:
  source, destination = resolve(args[3]), resolve(args[4])
  destination.mkdir(parents=True, exist_ok=True)
  for path in source.iterdir():
    shutil.copyfile(path, destination / path.name)
  raise SystemExit(0)
raise SystemExit(f"unsupported fake gcloud invocation: {args}")
""",
          encoding="utf-8",
      )
      fake_gcloud.chmod(0o755)
      environment = os.environ.copy()
      environment["FAKE_GCS_ROOT"] = str(fake_root)
      environment["PATH"] = (
          f"{fake_bin}:{Path(sys.executable).parent}:"
          f"{environment.get('PATH', '')}"
      )
      command = [
          "bash",
          str(SCRIPT_DIR / "run_m15_replay_gcs_prepare.sh"),
          source_uri,
          str(root),
      ]
      result = subprocess.run(
          command, check=False, capture_output=True, text=True, env=environment
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("[M15.APC.REPLAY.PREPARE] COMPLETE", result.stdout)
      self.assertIn("red_rows=201,245 replay_prefix_end_call=4", result.stdout)
      derived = remote / "derived/m15-replay-input-plan-v1/files"
      check = subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=derived,
          check=False,
      )
      self.assertEqual(check.returncode, 0)
      derived_result = json.loads((derived / "REPLAY_ANALYSIS.json").read_text())
      self.assertEqual(derived_result["claim_ceiling"], "INPUT_PLAN_ONLY_MODEL_REPLAY_NOT_RUN")
      second = subprocess.run(
          command, check=False, capture_output=True, text=True, env=environment
      )
      self.assertEqual(second.returncode, 3)
      self.assertIn("immutable analysis already exists", second.stderr)


if __name__ == "__main__":
  program = unittest.main(exit=False)
  if not program.result.wasSuccessful():
    raise SystemExit(1)
  print("M15_REPLAY_ANALYSIS_TEST_PASS tests=4 gcs_wrapper=1")
