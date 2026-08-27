#!/usr/bin/env python3
"""Host positives and negatives for the M15 wide seam classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

import numpy as np

from assemble_m15_wide_round import assemble
from stage_m15_wide_shard import _sha256, stage
from verify_m15_wide_round import verify


MODULE_PATH = Path(__file__).with_name("classify_m15_apc_wide_seam.py")
SPEC = importlib.util.spec_from_file_location("classify_m15_apc_wide_seam", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PACKAGER_PATH = Path(__file__).with_name("package_m15_apc_wide_seam.py")
PACKAGER_SPEC = importlib.util.spec_from_file_location(
    "package_m15_apc_wide_seam", PACKAGER_PATH
)
assert PACKAGER_SPEC and PACKAGER_SPEC.loader
PACKAGER = importlib.util.module_from_spec(PACKAGER_SPEC)
PACKAGER_SPEC.loader.exec_module(PACKAGER)


def _prefix(values: list[int]) -> bytes:
  array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
  return hashlib.sha256(array.tobytes()).hexdigest().encode()


class Fixture:

  def __init__(
      self,
      *,
      mode: str = "layer",
      completion_position: int = 0,
      diagnostic_round: int = 0,
  ):
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.capture = self.root / "capture"
    self.capture.mkdir()
    self.report = self.root / "pre_alignment.jsonl"
    self.capsule = self.root / "mismatch.npz"
    self.ledger = self.capture / "m15_replay_envelope.jsonl"
    self.mode = mode
    self.diagnostic_round = diagnostic_round
    self.completion_position = completion_position
    self.expected_layer = 5 if mode == "full" else None
    prompt = np.asarray([[10, 11, 12]], dtype=np.int32)
    completion = np.asarray([[13, 14]], dtype=np.int32)
    decode = np.asarray([[-2.0, -3.0]], dtype=np.float32)
    prefill = decode.copy()
    prefill[0, completion_position] += np.float32(0.25)
    np.savez(
        self.capsule,
        metadata_json=np.frombuffer(
            json.dumps({"diagnostic_round": diagnostic_round}).encode(),
            dtype=np.uint8,
        ),
        selected_rows=np.asarray([201], dtype=np.int32),
        prompt_ids=prompt,
        prompt_mask=np.asarray([[1, 1, 1]], dtype=np.bool_),
        completion_ids=completion,
        completion_valid_mask=np.asarray([[1, 1]], dtype=np.bool_),
        action_mask=np.asarray([[1, 1]], dtype=np.bool_),
        s_decode=decode,
        s_prefill=prefill,
    )
    self.source_position = 2 + completion_position
    tokens = [10, 11, 12, 13, 14]
    self.prefix = _prefix(tokens[:self.source_position + 1])
    self.target = int(completion[0, completion_position])
    self.decode = float(decode[0, completion_position])
    self.prefill = float(prefill[0, completion_position])
    self._write_report(ab_bytes=1, bc_bytes=0)
    self._write_records()
    self._write_ledger()

  def close(self):
    self.holder.cleanup()

  def _write_report(self, *, ab_bytes: int, bc_bytes: int):
    row = {
        "diagnostic_round": self.diagnostic_round,
        "N_action": 2,
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "valid": True,
                "finite": True,
                "differing_bytes": ab_bytes,
                "differing_elements": 1 if ab_bytes else 0,
            },
            "S_prefill_vs_T_old": {
                "valid": True,
                "finite": True,
                "differing_bytes": bc_bytes,
                "differing_elements": 1 if bc_bytes else 0,
            },
        },
    }
    self.report.write_text(json.dumps(row) + "\n", encoding="utf-8")

  def _write_npz_record(self, prefix: str, index: int, record: dict, arrays: dict):
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    raw = payload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    record = {
        **record,
        "record_index": index,
        "npz_sha256": digest,
        "schema": (
            "p38-seam-fingerprint-v1"
            if prefix == "p38_seam" else "p38-tail-values-v1"
        ),
    }
    (self.capture / f"{prefix}_{index:06d}.npz").write_bytes(raw)
    (self.capture / f"{prefix}_{index:06d}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

  def _write_records(self):
    checkpoints = (
        MODULE._LAYER_CHECKPOINTS  # pylint: disable=protected-access
        if self.mode == "layer"
        else MODULE._FULL_CHECKPOINTS  # pylint: disable=protected-access
    )
    layers = list(range(36)) if self.mode == "layer" else [5]
    shape = (1, len(layers), len(checkpoints), 8)
    a_values = np.zeros(shape, dtype=np.uint64)
    b_values = np.zeros(shape, dtype=np.uint64)
    if self.mode == "layer":
      a_values[0, 5, 1, 2] = 1
    else:
      a_values[0, 0, checkpoints.index("q_post_rope"), 3] = 1
    final_values = np.zeros((1, 8), dtype=np.uint64)
    for index, arm in enumerate(("A", "B")):
      request_id = f"request-{arm.lower()}"
      values = a_values if arm == "A" else b_values
      self._write_npz_record(
          "p38_seam",
          index,
          {
              "arm": arm,
              "diagnostic_round": self.diagnostic_round,
              "call_index": 10 + index,
              "observer_mode": self.mode,
              "checkpoint_names": list(checkpoints),
              "layer_indices": layers,
              "gather_bucket": 1,
              "layer_fingerprint_shape": list(shape),
              "final_fingerprint_shape": [1, 8],
              "layer_fingerprint_sharding": "test-sharding",
              "final_fingerprint_sharding": "test-sharding",
              "requests": [{
                  "request_id": request_id,
                  "dp_rank": 0,
                  "dp_request_slot": 0,
                  "position_range": [self.source_position, self.source_position + 1],
                  "token_history_sha256": "a" * 64,
              }],
          },
          {
              "row_indices": np.asarray([0], dtype=np.int32),
              "positions": np.asarray([self.source_position], dtype=np.int32),
              "token_ids": np.asarray([12 + self.completion_position], dtype=np.int32),
              "request_ordinals": np.asarray([0], dtype=np.int32),
              "token_prefix_sha256": np.asarray([self.prefix], dtype="S64"),
              "layer_fingerprints": values,
              "final_norm_fingerprints": final_values,
              "final_hidden_rows": np.zeros((1, 0), dtype=np.float32),
          },
      )
    if self.mode == "layer":
      for index, arm in enumerate(("A", "B")):
        values = np.zeros((1, len(MODULE._TAIL_CHECKPOINTS)), dtype=np.float32)  # pylint: disable=protected-access
        values[0, -1] = self.decode if arm == "A" else self.prefill
        self._write_npz_record(
            "p38_tail",
            index,
            {
                "arm": arm,
                "diagnostic_round": self.diagnostic_round,
                "call_index": 10 + index,
                "checkpoint_names": list(MODULE._TAIL_CHECKPOINTS),  # pylint: disable=protected-access
            },
            {
                "row_indices": np.asarray([0], dtype=np.int32),
                "positions": np.asarray([self.source_position], dtype=np.int32),
                "token_ids": np.asarray([12 + self.completion_position], dtype=np.int32),
                "request_ordinals": np.asarray([0], dtype=np.int32),
                "token_prefix_sha256": np.asarray([self.prefix], dtype="S64"),
                "logit_row_indices": np.asarray([0], dtype=np.int32),
                "target_ids": np.asarray([self.target], dtype=np.int32),
                "tail_values": values,
            },
        )

  def _write_ledger(self):
    rows = []
    for index, arm in enumerate(("A", "B")):
      request_id = f"request-{arm.lower()}"
      rows.append({
          "schema": "m15-apc-serving-envelope-v1",
          "diagnostic_round": self.diagnostic_round,
          "arm": "on",
          "serving_arm": arm,
          "call_index": 10 + index,
          "program_path": "standard",
          "request_order": [request_id],
          "requests": [{
              "request_id": request_id,
              "num_computed_tokens": self.source_position,
              "scheduled_tokens": 1,
              "physical_pages": [3, 4],
              "block_size": 16,
          }],
      })
    self.ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

  def classify(self, *, arm: str = "on", require_first_action: bool = True):
    return MODULE.classify(
        directory=self.capture,
        alignment_report=self.report,
        capsules=[self.capsule],
        mode=self.mode,
        arm=arm,
        replay_ledger=self.ledger,
        expected_layer=self.expected_layer,
        require_first_action=require_first_action,
    )


class M15WideSeamClassifierTest(unittest.TestCase):

  def _fixture(self, **kwargs) -> Fixture:
    fixture = Fixture(**kwargs)
    self.addCleanup(fixture.close)
    return fixture

  def test_layer_mode_localizes_first_action_and_preserves_ledger_geometry(self):
    result = self._fixture().classify()
    self.assertEqual(result["status"], "PASS")
    self.assertEqual(result["classification"], "M15_LAYER_FIRST_RED_LOCALIZED")
    self.assertEqual(result["selected_layer"], 5)
    self.assertEqual(result["first_red_boundary"]["checkpoint"], "layer_output")
    self.assertEqual(result["last_exact_boundary"], {
        "layer": 5, "checkpoint": "layer_input"
    })
    self.assertEqual(result["coverage"]["first_action_joinable_red_points"], 1)
    self.assertEqual(len(result["replay_ledger_receipts"]), 2)

  def test_full_mode_localizes_q_post_rope(self):
    result = self._fixture(mode="full").classify()
    self.assertEqual(result["gate"], "FIRST_RED_LOCALIZED")
    self.assertEqual(result["selected_layer"], 5)
    self.assertEqual(result["first_red_boundary"]["checkpoint"], "q_post_rope")
    self.assertEqual(result["last_exact_boundary"]["checkpoint"], "k_norm")

  def test_nonzero_diagnostic_round_is_bound_end_to_end(self):
    result = self._fixture(diagnostic_round=2).classify()
    self.assertEqual(result["diagnostic_round"], 2)
    self.assertEqual(result["coverage"]["first_action_joinable_red_points"], 1)

  def test_exact_control_is_reachability_not_localization(self):
    fixture = self._fixture()
    fixture._write_report(ab_bytes=0, bc_bytes=0)  # pylint: disable=protected-access
    result = MODULE.classify(
        directory=fixture.capture,
        alignment_report=fixture.report,
        capsules=[],
        mode="layer",
        arm="off",
    )
    self.assertEqual(result["classification"], "M15_OBSERVER_CONTROL_EXACT")
    self.assertEqual(result["gate"], "OBSERVER_REACHED_EXACT_ENDPOINT")

  def test_rejects_red_without_first_action_anchor(self):
    fixture = self._fixture(completion_position=1)
    with self.assertRaisesRegex(
        MODULE.M15WideSeamError, "completion-position-zero"
    ):
      fixture.classify()

  def test_rejects_b_c_red(self):
    fixture = self._fixture()
    fixture._write_report(ab_bytes=1, bc_bytes=1)  # pylint: disable=protected-access
    with self.assertRaisesRegex(MODULE.M15WideSeamError, "B-C"):
      fixture.classify()

  def test_one_bit_seam_change_is_detected(self):
    result = self._fixture().classify()
    fields = result["first_red_boundary"]["differing_fingerprint_fields"]
    self.assertEqual(fields, [2])

  def test_compact_bundle_contains_selected_raw_inputs_and_valid_manifest(self):
    fixture = self._fixture()
    classification = fixture.classify()
    classification_path = fixture.root / "classification.json"
    classification_path.write_text(
        json.dumps(classification), encoding="utf-8"
    )
    output = fixture.root / "bundle.tar"
    receipt = PACKAGER.package(
        directory=fixture.capture,
        classification_path=classification_path,
        alignment_report=fixture.report,
        capsules=[fixture.capsule],
        replay_ledger=fixture.ledger,
        output=output,
    )
    self.assertEqual(receipt["status"], "PASS")
    self.assertEqual(receipt["selected_seam_records"], [0, 1])
    self.assertEqual(receipt["selected_tail_records"], [0, 1])
    with tarfile.open(output) as archive:
      names = archive.getnames()
      self.assertIn("classification.json", names)
      self.assertIn("records/p38_seam_000000.npz", names)
      self.assertIn("capsules/capsule-00.npz", names)
      manifest = archive.extractfile("SHA256SUMS").read().decode("ascii")
      for line in manifest.splitlines():
        expected, name = line.split("  ", 1)
        self.assertEqual(
            hashlib.sha256(archive.extractfile(name).read()).hexdigest(),
            expected,
        )

  def test_classifier_and_bundle_run_only_from_completed_shard_union(self):
    fixture = self._fixture()
    commit = "9" * 40
    shards = fixture.root / "shards"
    shard = shards / "000000"
    inventory = stage(
        directory=fixture.capture,
        shard_root=shards,
        output=shard,
        round_index=0,
        sequence=0,
        max_records=32,
        max_bytes=256 * 1024 * 1024,
        expected_commit=commit,
        runtime_commit=commit,
    )
    completion = {
        "schema": "m15-wide-observer-shard-completion-v1",
        "status": "sealed-uploaded-verified",
        "claim_ceiling": (
            "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
        ),
        "sequence": 0,
        "diagnostic_round": 0,
        "record_pairs": inventory["record_pairs"],
        "manifest_sha256": _sha256(shard / "SHA256SUMS"),
        "expected_source_commit": commit,
        "runtime_source_commit": commit,
    }
    (shard / "SHARD_COMPLETE.json").write_text(
        json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8"
    )
    round_dir = fixture.root / "round"
    receipt = assemble(
        live_directory=fixture.capture,
        shard_root=shards,
        output=round_dir,
        round_index=0,
        pre_alignment=fixture.report,
        capsule=fixture.capsule,
        replay_ledger=fixture.ledger,
        observer_mode="layer",
        expected_commit=commit,
        runtime_commit=commit,
    )
    result = MODULE.classify(
        directory=round_dir,
        alignment_report=round_dir / "pre-alignment.jsonl",
        capsules=[round_dir / "mismatch-capsule.npz"],
        mode="layer",
        arm="on",
        replay_ledger=round_dir / "m15-replay-envelope.jsonl",
        require_first_action=True,
    )
    classification = round_dir / "p38_seam.classification.json"
    classification.write_text(json.dumps(result), encoding="utf-8")
    bundle = round_dir / "m15_wide_seam_bundle.tar"
    PACKAGER.package(
        directory=round_dir,
        classification_path=classification,
        alignment_report=round_dir / "pre-alignment.jsonl",
        capsules=[round_dir / "mismatch-capsule.npz"],
        replay_ledger=round_dir / "m15-replay-envelope.jsonl",
        output=bundle,
    )
    members = [
        "ROUND_INPUT_RECEIPT.json",
        "p38_seam.classification.json",
        "m15_wide_seam_bundle.tar",
    ]
    manifest = round_dir / "WIDE_SHA256SUMS"
    manifest.write_text("".join(
        f"{_sha256(round_dir / name)}  {name}\n" for name in members
    ), encoding="ascii")
    (round_dir / "WIDE_ROUND_COMPLETE.json").write_text(json.dumps({
        "schema": "m15-wide-round-completion-v1",
        "status": "classified-and-uploaded",
        "diagnostic_round": 0,
        "classification": result["classification"],
        "record_pairs": receipt["record_pairs"],
        "shards": receipt["shards"],
        "manifest_sha256": _sha256(manifest),
        "expected_source_commit": commit,
        "runtime_source_commit": commit,
    }, sort_keys=True) + "\n", encoding="utf-8")
    verified = verify(
        round_directory=round_dir,
        classification=classification,
        bundle=bundle,
        expected_commit=commit,
        runtime_commit=commit,
    )
    self.assertEqual(verified["classification"], "M15_LAYER_FIRST_RED_LOCALIZED")
    self.assertEqual(verified["record_pairs"], 4)


if __name__ == "__main__":
  unittest.main()
