#!/usr/bin/env python3
"""Regression tests for the complete M15 serving replay carrier."""

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


MODULE = Path(__file__).with_name("package_full_replay_carrier.py")
SPEC = importlib.util.spec_from_file_location("package_full_replay_carrier", MODULE)
assert SPEC and SPEC.loader
carrier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = carrier
SPEC.loader.exec_module(carrier)
AUDIT_MODULE = Path(__file__).with_name("audit_m15_replay_capture.py")
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_m15_replay_capture", AUDIT_MODULE
)
assert AUDIT_SPEC and AUDIT_SPEC.loader
audit_module = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit_module
AUDIT_SPEC.loader.exec_module(audit_module)


def _array_sha(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha(tokens: np.ndarray) -> str:
  return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


class PackageFullReplayCarrierTest(unittest.TestCase):

  def _inputs(self, root: Path):
    arrays = {
        "prompt_ids": np.arange(256 * 4, dtype=np.int32).reshape(256, 4) + 1000,
        "prompt_mask": np.ones((256, 4), dtype=np.bool_),
        "completion_ids": np.arange(256 * 3, dtype=np.int32).reshape(256, 3) + 9000,
        "completion_valid_mask": np.ones((256, 3), dtype=np.bool_),
        "action_mask": np.ones((256, 3), dtype=np.bool_),
        "s_decode": np.zeros((256, 3), dtype=np.float32),
        "s_prefill": np.zeros((256, 3), dtype=np.float32),
        "t_old": np.zeros((256, 3), dtype=np.float32),
        "policy_version": np.zeros((256, 1), dtype=np.int32),
        "sampling_values": np.tile(
            np.array([[0.7, 0.0, 1.0]], dtype=np.float32), (256, 1)
        ),
    }
    arrays["s_decode"][9, 0] = np.float32(0.5)
    metadata = {
        "schema": "m15-apc-producer-unit-v1",
        "arm": "on",
        "source_commit": "7" * 40,
        "rows": 256,
        "prompt_groups": 32,
        "num_generations": 8,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha(value),
            }
            for name, value in arrays.items()
        },
    }
    producer = root / "m15_producer_unit.npz"
    np.savez_compressed(
        producer,
        source_rows=np.arange(256, dtype=np.int32),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        ),
        **arrays,
    )
    history = np.concatenate((arrays["prompt_ids"][9], arrays["completion_ids"][9]))
    ledger = root / "m15_replay_envelope.jsonl"
    records = [
        {
            "schema": "m15-apc-serving-envelope-v1",
            "arm": "on",
            "serving_arm": "A",
            "call_index": 1,
            "program_path": "standard",
            "request_order": ["rollout-9"],
            "requests": [{
                "request_id": "rollout-9",
                "request_index": 0,
                "dp_rank": 2,
                "local_scheduler_slot": 1,
                "scheduled_tokens": 1,
                "num_computed_tokens": 2,
                "num_prompt_tokens": 2,
                "num_tokens": 3,
                "token_history_sha256": _token_sha(history[:3]),
                "request_kind": "decode",
                "block_size": 4,
                "logical_blocks_before": 1,
                "logical_blocks_after": 1,
                "physical_pages": [11],
            }],
        },
        {
            "schema": "m15-apc-serving-envelope-v1",
            "arm": "on",
            "serving_arm": "A",
            "call_index": 2,
            "program_path": "continue_decode",
            "request_order": ["rollout-9"],
            "requests": [{
                "request_id": "rollout-9",
                "request_index": 0,
                "dp_rank": 2,
                "local_scheduler_slot": 1,
                "scheduled_tokens": 1,
                "num_computed_tokens": 3,
                "num_prompt_tokens": 2,
                "num_tokens": 4,
                "token_history_sha256": _token_sha(history[:4]),
                "request_kind": "decode",
                "block_size": 4,
                "logical_blocks_before": 1,
                "logical_blocks_after": 1,
                "physical_pages": [11],
            }],
        },
        {
            "schema": "m15-apc-serving-envelope-v1",
            "arm": "on",
            "serving_arm": "B",
            "call_index": 3,
            "program_path": "standard",
            "request_order": ["rescore-9"],
            "requests": [{
                "request_id": "rescore-9",
                "request_index": 0,
                "dp_rank": 2,
                "local_scheduler_slot": 0,
                "scheduled_tokens": 3,
                "num_computed_tokens": 0,
                "num_prompt_tokens": 3,
                "num_tokens": 3,
                "token_history_sha256": _token_sha(history[:3]),
                "request_kind": "prefill",
                "block_size": 4,
                "logical_blocks_before": 0,
                "logical_blocks_after": 1,
                "physical_pages": [19],
            }],
        },
    ]
    ledger.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    first_dir = root / "m15_first_red_replay"
    first_dir.mkdir()
    first_capsule = first_dir / "first_red_capsule.npz"
    np.savez_compressed(
        first_capsule,
        selected_rows=np.array([9], dtype=np.int32),
        metadata_json=np.frombuffer(b"{}", dtype=np.uint8),
        **{name: value[[9]] for name, value in arrays.items()},
    )
    first_contract = first_dir / "first_red_contract.json"
    first_contract.write_text(json.dumps({
        "status": "FIRST_RED_ROW_FROZEN",
        "source_row": 9,
        "first_incident": {
            "source_row": 9,
            "request_id": "rollout-9",
            "call_index": 1,
            "num_computed_tokens": 2,
            "physical_pages": [11],
        },
    }), encoding="utf-8")
    first_sums = first_dir / "SHA256SUMS"
    first_sums.write_text(
        f"{_file_sha(first_capsule)}  first_red_capsule.npz\n"
        f"{_file_sha(first_contract)}  first_red_contract.json\n",
        encoding="utf-8",
    )
    capture = root / "capture.json"
    capture.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    m15 = root / "m15_apc_target.classification.json"
    m15.write_text(json.dumps({
        "status": "FRESH_TARGET_RED_FROZEN",
        "arm": "on",
        "source_commit": "7" * 40,
    }), encoding="utf-8")
    return producer, ledger, first_dir, capture, m15, records

  def _package(self, root: Path, inputs):
    producer, ledger, first_dir, capture, m15, _ = inputs
    return carrier.package(
        producer_path=producer,
        ledger_path=ledger,
        first_red_dir=first_dir,
        capture_classification_path=capture,
        m15_classification_path=m15,
        output_dir=root / "m15_full_replay_carrier",
    )

  def _gcs_root(self, root: Path, capture: Path) -> Path:
    gcs = root / "gcs-root"
    gcs.mkdir()
    prefix = (
        "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
        "canon-test/attempt-0"
    )
    source = "7" * 40
    for name, status in (
        ("PREFLIGHT.json", "writable"),
        ("COLLECTED.json", "collected"),
        ("COMPLETE.json", "postflight-accepted"),
    ):
      (gcs / name).write_text(json.dumps({
          "prefix": prefix,
          "source_commit": source,
          "status": status,
      }), encoding="utf-8")
    (gcs / "serving-classification.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    (gcs / "run.log").write_text(
        "[CANON_ALIGN_PRE] verdict=FAIL\n"
        "[CANON_" "APC_M15_FULL_REPLAY_CARRIER]\n",
        encoding="utf-8",
    )
    members = ("serving-classification.json", "run.log")
    (gcs / "SHA256SUMS").write_text(
        "".join(f"{_file_sha(gcs / name)}  {name}\n" for name in members),
        encoding="utf-8",
    )
    return gcs

  def test_freezes_full_carrier_and_relative_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = self._package(root, self._inputs(root))
      self.assertEqual(result["status"], "FULL_REPLAY_CARRIER_FROZEN")
      self.assertEqual(result["producer_rows"], 256)
      self.assertEqual(result["serving_call_count"], 3)
      self.assertEqual(result["serving_arms"], ["A", "B"])
      self.assertEqual(result["program_paths"], ["continue_decode", "standard"])
      self.assertEqual(
          result["program_paths_by_arm"],
          {"A": ["continue_decode", "standard"], "B": ["standard"]},
      )
      joins = (root / "m15_full_replay_carrier/request_row_joins.jsonl").read_text()
      self.assertIn('"candidate_source_rows":[9]', joins)
      check = subprocess.run(
          ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
          cwd=root / "m15_full_replay_carrier",
          check=False,
      )
      self.assertEqual(check.returncode, 0)

  def test_rejects_noncontiguous_call_chronology(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      records = inputs[-1]
      records[1]["call_index"] = 4
      inputs[1].write_text(
          "".join(json.dumps(record) + "\n" for record in records),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(carrier.CarrierError, "not contiguous"):
        self._package(root, inputs)

  def test_rejects_missing_continue_decode_attestation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      records = [inputs[-1][0], inputs[-1][2]]
      records[1]["call_index"] = 2
      inputs[1].write_text(
          "".join(json.dumps(record) + "\n" for record in records),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(carrier.CarrierError, "must attest standard and continue_decode"):
        self._package(root, inputs)

  def test_rejects_continue_decode_on_full_reset_arm(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      records = inputs[-1]
      records[2]["program_path"] = "continue_decode"
      inputs[1].write_text(
          "".join(json.dumps(record) + "\n" for record in records),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(carrier.CarrierError, "full-reset standard"):
        self._package(root, inputs)

  def test_rejects_unknown_program_path(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      records = inputs[-1]
      records[1]["program_path"] = "vendor_magic"
      inputs[1].write_text(
          "".join(json.dumps(record) + "\n" for record in records),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(carrier.CarrierError, "program path drifted"):
        self._package(root, inputs)

  def test_rejects_token_history_outside_full_producer(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      records = inputs[-1]
      records[0]["requests"][0]["token_history_sha256"] = "f" * 64
      inputs[1].write_text(
          "".join(json.dumps(record) + "\n" for record in records),
          encoding="utf-8",
      )
      with self.assertRaisesRegex(carrier.CarrierError, "do not join"):
        self._package(root, inputs)

  def test_rejects_producer_array_hash_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      producer = inputs[0]
      with np.load(producer, allow_pickle=False) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
      changed["s_decode"][9, 0] += np.float32(0.25)
      np.savez_compressed(producer, **changed)
      with self.assertRaisesRegex(carrier.CarrierError, "SHA drifted"):
        self._package(root, inputs)

  def test_rejects_first_red_payload_not_from_producer(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      first_capsule = inputs[2] / "first_red_capsule.npz"
      with np.load(first_capsule, allow_pickle=False) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
      changed["completion_ids"][0, 0] += 1
      np.savez_compressed(first_capsule, **changed)
      with self.assertRaisesRegex(carrier.CarrierError, "differs from the full producer"):
        self._package(root, inputs)

  def test_gcs_audit_returns_only_small_receipts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      self._package(root, inputs)
      gcs = self._gcs_root(root, root)
      result = audit_module.audit(
          root_dir=gcs,
          capture_dir=root,
          source_gcs_uri=(
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
              "canon-test/attempt-0"
          ),
          output_dir=root / "return",
      )
      self.assertEqual(result["status"], "FRESH_TARGET_RED_FROZEN")
      self.assertFalse((root / "return/m15_producer_unit.npz").exists())
      self.assertTrue((root / "return/replay-contract.json").is_file())

  def test_gcs_audit_rejects_missing_terminal_marker(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      inputs = self._inputs(root)
      self._package(root, inputs)
      gcs = self._gcs_root(root, root)
      (gcs / "COMPLETE.json").unlink()
      with self.assertRaisesRegex(audit_module.AuditError, "missing JSON"):
        audit_module.audit(
            root_dir=gcs,
            capture_dir=root,
            source_gcs_uri=(
                "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
                "canon-test/attempt-0"
            ),
            output_dir=root / "return",
        )

  def test_gcs_wrapper_downloads_verifies_and_uploads_small_receipts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      capture = root / "capture"
      capture.mkdir()
      inputs = self._inputs(capture)
      self._package(capture, inputs)
      source = self._gcs_root(root, capture)
      with tarfile.open(source / "serving-capture.tar", "w") as archive:
        for path in sorted(capture.rglob("*")):
          if path.is_file():
            archive.add(path, arcname=path.relative_to(capture))
      manifest_names = (
          "serving-capture.tar",
          "serving-classification.json",
          "run.log",
      )
      (source / "SHA256SUMS").write_text(
          "".join(
              f"{_file_sha(source / name)}  {name}\n"
              for name in manifest_names
          ),
          encoding="utf-8",
      )

      fake_root = root / "fake-gcs"
      remote = (
          fake_root
          / "yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-test/attempt-0"
      )
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
    target = destination / path.name
    if path.is_dir():
      shutil.copytree(path, target)
    else:
      shutil.copyfile(path, target)
  raise SystemExit(0)
raise SystemExit(f"unsupported fake gcloud invocation: {args}")
""",
          encoding="utf-8",
      )
      fake_gcloud.chmod(0o755)
      attempt_uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
          "canon-test/attempt-0"
      )
      environment = os.environ.copy()
      environment["FAKE_GCS_ROOT"] = str(fake_root)
      interpreter_dir = Path(sys.executable).resolve().parent
      environment["PATH"] = (
          f"{fake_bin}:{interpreter_dir}:{environment.get('PATH', '')}"
      )
      result = subprocess.run(
          [
              "bash",
              str(Path(__file__).with_name("run_m15_replay_gcs_audit.sh")),
              attempt_uri,
              str(root),
          ],
          check=False,
          capture_output=True,
          text=True,
          env=environment,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("[M15.APC.GCS] COMPLETE", result.stdout)
      derived = remote / "derived/m15-replay-audit-v1/files"
      self.assertTrue((derived / "SHA256SUMS").is_file())
      self.assertTrue((derived / "RETURN_RECEIPT.json").is_file())
      self.assertFalse((derived / "m15_producer_unit.npz").exists())
      second = subprocess.run(
          [
              "bash",
              str(Path(__file__).with_name("run_m15_replay_gcs_audit.sh")),
              attempt_uri,
              str(root),
          ],
          check=False,
          capture_output=True,
          text=True,
          env=environment,
      )
      self.assertEqual(second.returncode, 3)
      self.assertIn("immutable derived audit already exists", second.stderr)


if __name__ == "__main__":
  unittest.main()
