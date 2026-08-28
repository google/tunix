#!/usr/bin/env python3
"""Host gates for bounded, sealed M15 wide-observer evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from assemble_m15_wide_round import assemble  # noqa: E402
from checkpoint_m15_classifier_input import (  # noqa: E402
    M15ClassifierInputError,
    checkpoint,
)
from stage_m15_wide_shard import (  # noqa: E402
    M15WideShardError,
    _sha256,
    stage,
)
from verify_m15_wide_round import VerificationError, verify  # noqa: E402


COMMIT = "1" * 40


def _load_round_seal_function():
  source_path = SCRIPT_DIR.parents[3] / "tunix" / "rl" / "alignment.py"
  source = source_path.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(source_path))
  function = next(
      node for node in tree.body
      if isinstance(node, ast.FunctionDef)
      and node.name == "_seal_p38_diagnostic_round"
  )
  namespace = {
      "AlignmentGateError": type("AlignmentGateError", (RuntimeError,), {}),
      "P38_ONEHOST_REHEARSAL_ENV": "CANON_P38_ONEHOST_REHEARSAL",
      "P38_ROUND_SEAL_ACK_DIR_ENV": "CANON_P38_ROUND_SEAL_ACK_DIR",
      "P38_ROUND_SEAL_REQUEST_DIR_ENV": "CANON_P38_ROUND_SEAL_REQUEST_DIR",
      "json": json,
      "os": os,
      "re": re,
  }
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
  return namespace


class RoundSealFailureTest(unittest.TestCase):

  def test_learner_fails_immediately_on_worker_failure_receipt(self):
    namespace = _load_round_seal_function()
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      requests = root / "requests"
      acknowledgements = root / "acks"
      requests.mkdir()
      acknowledgements.mkdir()
      failure = acknowledgements / "round-000000.failure.json"

      def publish_failure(_seconds):
        failure.write_text(json.dumps({
            "action": "seal-round",
            "diagnostic_round": 0,
            "exit_code": 17,
            "schema": "canon-p38-round-seal-failure-v1",
            "stage": "classify",
            "status": "FAIL",
        }, sort_keys=True) + "\n", encoding="utf-8")

      namespace["time"] = SimpleNamespace(
          monotonic=lambda: 0.0,
          sleep=publish_failure,
      )
      environment = {
          "CANON_P38_ROUND_SEAL_REQUEST_DIR": str(requests),
          "CANON_P38_ROUND_SEAL_ACK_DIR": str(acknowledgements),
      }
      with mock.patch.dict(os.environ, environment, clear=False):
        with self.assertRaisesRegex(
            namespace["AlignmentGateError"],
            "round=0 stage=classify exit_code=17",
        ):
          namespace["_seal_p38_diagnostic_round"](0)
      self.assertTrue((requests / "round-000000.request").is_file())
      self.assertFalse((acknowledgements / "round-000000.ack").exists())


class M15WideDurabilityTest(unittest.TestCase):

  def setUp(self) -> None:
    self.temp = tempfile.TemporaryDirectory()
    self.root = Path(self.temp.name)
    self.live = self.root / "live"
    self.shards = self.root / "shards"
    self.live.mkdir()
    self.shards.mkdir()

  def tearDown(self) -> None:
    self.temp.cleanup()

  def _pair(
      self,
      prefix: str,
      index: int,
      payload_bytes: int = 32,
      diagnostic_round: int = 0,
  ) -> None:
    schema = {
        "p38_seam": "p38-seam-fingerprint-v1",
        "p38_tail": "p38-tail-values-v1",
    }[prefix]
    npz_path = self.live / f"{prefix}_{index:06d}.npz"
    npz_path.write_bytes(bytes([index % 251 + 1]) * payload_bytes)
    record = {
        "schema": schema,
        "diagnostic_round": diagnostic_round,
        "record_index": index,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    (self.live / f"{prefix}_{index:06d}.json").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )

  def _complete(self, shard: Path) -> None:
    inventory = json.loads(
        (shard / "SHARD_INVENTORY.json").read_text(encoding="utf-8")
    )
    completion = {
        "schema": "m15-wide-observer-shard-completion-v1",
        "status": "sealed-uploaded-verified",
        "claim_ceiling": (
            "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
        ),
        "sequence": inventory["sequence"],
        "diagnostic_round": inventory["diagnostic_round"],
        "record_pairs": inventory["record_pairs"],
        "manifest_sha256": _sha256(shard / "SHA256SUMS"),
        "expected_source_commit": COMMIT,
        "runtime_source_commit": COMMIT,
    }
    (shard / "SHARD_COMPLETE.json").write_text(
        json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8"
    )

  def _stage(
      self, sequence: int, max_records: int = 2, diagnostic_round: int = 0
  ):
    round_root = self.shards / f"round-{diagnostic_round:06d}"
    output = round_root / f"{sequence:06d}"
    result = stage(
        directory=self.live,
        shard_root=round_root,
        output=output,
        round_index=diagnostic_round,
        sequence=sequence,
        max_records=max_records,
        max_bytes=1024 * 1024,
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    return output, result

  def _round_inputs(self, diagnostic_round: int = 0) -> tuple[Path, Path]:
    pre = self.root / "pre.jsonl"
    replay = self.root / "replay.jsonl"
    pre.write_text(json.dumps({
        "diagnostic_round": diagnostic_round,
        "boundaries": {
            "S_decode_vs_S_prefill": {"differing_bytes": 0},
            "S_prefill_vs_T_old": {"differing_bytes": 0},
        },
    }) + "\n", encoding="utf-8")
    replay.write_text(json.dumps({
        "schema": "m15-apc-serving-envelope-v1",
        "diagnostic_round": diagnostic_round,
    }) + "\n", encoding="utf-8")
    return pre, replay

  def test_bounded_shards_are_disjoint_and_live_mutation_isolated(self) -> None:
    for index in range(5):
      self._pair("p38_seam", index)
    shard0, inventory0 = self._stage(0)
    self.assertEqual(inventory0["record_pairs"], 2)
    self._complete(shard0)
    original = (shard0 / "p38_seam_000000.npz").read_bytes()
    (self.live / "p38_seam_000000.npz").write_bytes(b"mutated-live")
    self.assertEqual((shard0 / "p38_seam_000000.npz").read_bytes(), original)

    # The immutable shard remains authoritative; later snapshots must not
    # reread or rehash this mutated live copy.
    shard1, inventory1 = self._stage(1)
    self.assertEqual(inventory1["record_pairs"], 2)
    self._complete(shard1)
    shard2, inventory2 = self._stage(2)
    self.assertEqual(inventory2["record_pairs"], 1)
    self._complete(shard2)
    unused, empty = self._stage(3)
    self.assertIsNone(empty)
    self.assertFalse(unused.exists())

    names = []
    for shard in (shard0, shard1, shard2):
      inventory = json.loads(
          (shard / "SHARD_INVENTORY.json").read_text(encoding="utf-8")
      )
      names.extend(inventory["files"])
    self.assertEqual(len(names), len(set(names)))
    pre, replay = self._round_inputs()
    receipt = assemble(
        live_directory=self.live,
        shard_root=self.shards / "round-000000",
        output=self.root / "round",
        round_index=0,
        pre_alignment=pre,
        capsule=self.root / "absent.npz",
        replay_ledger=replay,
        observer_mode="layer",
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    self.assertEqual(receipt["record_pairs"], 5)

  def test_absent_observer_directory_returns_empty(self) -> None:
    absent_dir = self.root / "does_not_exist"
    round_root = self.shards / "round-000000"
    output = round_root / "000000"
    result = stage(
        directory=absent_dir,
        shard_root=round_root,
        output=output,
        round_index=0,
        sequence=0,
        max_records=2,
        max_bytes=1024 * 1024,
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    self.assertIsNone(result)
    self.assertFalse(output.exists())

  def test_unsealed_stage_is_not_accepted_and_tamper_is_rejected(self) -> None:
    self._pair("p38_seam", 0)
    shard0, _ = self._stage(0)
    # An upload interrupted before SHARD_COMPLETE is not durable authority.
    shard1, duplicate = self._stage(1)
    self.assertEqual(duplicate["record_pairs"], 1)
    self._complete(shard1)
    (shard1 / "p38_seam_000000.npz").write_bytes(b"tampered")
    pre, replay = self._round_inputs()
    with self.assertRaisesRegex(M15WideShardError, "failed SHA"):
      assemble(
          live_directory=self.live,
          shard_root=self.shards / "round-000000",
          output=self.root / "round",
          round_index=0,
          pre_alignment=pre,
          capsule=self.root / "absent.npz",
          replay_ledger=replay,
          observer_mode="layer",
          expected_commit=COMMIT,
          runtime_commit=COMMIT,
      )
    self.assertFalse((shard0 / "SHARD_COMPLETE.json").exists())

  def test_missing_pair_and_source_mismatch_fail_closed(self) -> None:
    json_path = self.live / "p38_seam_000000.json"
    json_path.write_text(json.dumps({
        "schema": "p38-seam-fingerprint-v1",
        "diagnostic_round": 0,
        "record_index": 0,
        "npz_sha256": "0" * 64,
    }) + "\n", encoding="utf-8")
    with self.assertRaisesRegex(M15WideShardError, "lacks its NPZ"):
      self._stage(0)

    json_path.unlink()
    self._pair("p38_seam", 0)
    shard, _ = self._stage(0)
    self._complete(shard)
    pre, replay = self._round_inputs()
    with self.assertRaisesRegex(M15WideShardError, "runtime source"):
      assemble(
          live_directory=self.live,
          shard_root=self.shards / "round-000000",
          output=self.root / "round",
          round_index=0,
          pre_alignment=pre,
          capsule=self.root / "absent.npz",
          replay_ledger=replay,
          observer_mode="full",
          expected_commit=COMMIT,
          runtime_commit="2" * 40,
      )

  def test_oversize_pair_and_shard_overwrite_are_rejected(self) -> None:
    self._pair("p38_seam", 0, payload_bytes=1024 * 1024)
    with self.assertRaisesRegex(M15WideShardError, "exceeds shard byte cap"):
      self._stage(0)
    (self.live / "p38_seam_000000.json").unlink()
    (self.live / "p38_seam_000000.npz").unlink()
    self._pair("p38_seam", 0)
    shard, _ = self._stage(0)
    self.assertTrue(shard.is_dir())
    with self.assertRaisesRegex(M15WideShardError, "already exists"):
      self._stage(0)

  def test_second_round_uses_isolated_shards_and_filters_combined_ledger(self):
    self._pair("p38_seam", 0, diagnostic_round=0)
    shard0, _ = self._stage(0, diagnostic_round=0)
    self._complete(shard0)
    self._pair("p38_seam", 1, diagnostic_round=1)
    shard1, _ = self._stage(1, diagnostic_round=1)
    self._complete(shard1)
    pre = self.root / "pre-rounds.jsonl"
    replay = self.root / "replay-rounds.jsonl"
    pre.write_text("".join(
        json.dumps({
            "diagnostic_round": round_index,
            "boundaries": {
                "S_decode_vs_S_prefill": {"differing_bytes": 0},
                "S_prefill_vs_T_old": {"differing_bytes": 0},
            },
        }) + "\n" for round_index in (0, 1)
    ), encoding="utf-8")
    replay.write_text("".join(
        json.dumps({
            "schema": "m15-apc-serving-envelope-v1",
            "diagnostic_round": round_index,
        }) + "\n" for round_index in (0, 1)
    ), encoding="utf-8")
    receipt = assemble(
        live_directory=self.live,
        shard_root=self.shards / "round-000001",
        output=self.root / "round-1",
        round_index=1,
        pre_alignment=pre,
        capsule=self.root / "absent.npz",
        replay_ledger=replay,
        observer_mode="full",
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    self.assertEqual(receipt["diagnostic_round"], 1)
    replay_rows = (self.root / "round-1/m15-replay-envelope.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    self.assertEqual(len(replay_rows), 1)
    self.assertEqual(json.loads(replay_rows[0])["diagnostic_round"], 1)

  def test_round_verifier_rejects_published_output_drift(self) -> None:
    self._pair("p38_seam", 0)
    shard, _ = self._stage(0)
    self._complete(shard)
    pre = self.root / "pre.jsonl"
    replay = self.root / "replay.jsonl"
    pre.write_text(json.dumps({
        "diagnostic_round": 0,
        "boundaries": {
            "S_decode_vs_S_prefill": {"differing_bytes": 0},
            "S_prefill_vs_T_old": {"differing_bytes": 0},
        },
    }) + "\n", encoding="utf-8")
    replay.write_text(json.dumps({
        "schema": "m15-apc-serving-envelope-v1",
        "diagnostic_round": 0,
    }) + "\n", encoding="utf-8")
    round_dir = self.root / "round"
    receipt = assemble(
        live_directory=self.live,
        shard_root=self.shards / "round-000000",
        output=round_dir,
        round_index=0,
        pre_alignment=pre,
        capsule=self.root / "absent.npz",
        replay_ledger=replay,
        observer_mode="full",
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    classification = round_dir / "p38_seam.classification.json"
    bundle = round_dir / "m15_wide_seam_bundle.tar"
    classification.write_text(json.dumps({
        "status": "PASS",
        "classification": "FIRST_RED_LOCALIZED",
        "diagnostic_round": 0,
    }) + "\n", encoding="utf-8")
    bundle.write_bytes(b"bundle")
    members = [
        "ROUND_INPUT_RECEIPT.json",
        "p38_seam.classification.json",
        "m15_wide_seam_bundle.tar",
    ]
    manifest = round_dir / "WIDE_SHA256SUMS"
    manifest.write_text("".join(
        f"{_sha256(round_dir / name)}  {name}\n" for name in members
    ), encoding="ascii")
    completion = {
        "schema": "m15-wide-round-completion-v1",
        "status": "classified-and-uploaded",
        "diagnostic_round": 0,
        "classification": "FIRST_RED_LOCALIZED",
        "record_pairs": receipt["record_pairs"],
        "shards": receipt["shards"],
        "manifest_sha256": _sha256(manifest),
        "expected_source_commit": COMMIT,
        "runtime_source_commit": COMMIT,
    }
    (round_dir / "WIDE_ROUND_COMPLETE.json").write_text(
        json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8"
    )
    published_classification = self.root / "published.json"
    published_bundle = self.root / "published.tar"
    published_classification.write_bytes(classification.read_bytes())
    published_bundle.write_bytes(bundle.read_bytes())
    result = verify(
        round_directory=round_dir,
        classification=published_classification,
        bundle=published_bundle,
        expected_commit=COMMIT,
        runtime_commit=COMMIT,
    )
    self.assertEqual(result["record_pairs"], 1)
    published_bundle.write_bytes(b"wrong")
    with self.assertRaisesRegex(VerificationError, "published bundle"):
      verify(
          round_directory=round_dir,
          classification=published_classification,
          bundle=published_bundle,
          expected_commit=COMMIT,
          runtime_commit=COMMIT,
      )


class M15ClassifierInputCheckpointTest(unittest.TestCase):

  def setUp(self) -> None:
    self.temp = tempfile.TemporaryDirectory()
    self.root = Path(self.temp.name)
    (self.root / "ROUND_INPUT_RECEIPT.json").write_text(json.dumps({
        "schema": "m15-wide-sealed-input-v1",
        "status": "PASS",
        "diagnostic_round": 0,
        "record_pairs": 7,
        "shards": [{"sequence": 0}],
        "expected_source_commit": COMMIT,
        "runtime_source_commit": COMMIT,
    }) + "\n", encoding="utf-8")
    (self.root / "m15-replay-envelope.jsonl").write_text(
        '{"schema":"m15-apc-serving-envelope-v1","diagnostic_round":0}\n',
        encoding="utf-8",
    )

  def tearDown(self) -> None:
    self.temp.cleanup()

  def _alignment(self, differing_bytes: int) -> None:
    (self.root / "pre-alignment.jsonl").write_text(json.dumps({
        "diagnostic_round": 0,
        "boundaries": {
            "S_decode_vs_S_prefill": {
                "differing_bytes": differing_bytes,
            },
        },
    }) + "\n", encoding="utf-8")

  def test_exact_input_checkpoint_is_self_hashed(self) -> None:
    self._alignment(0)
    receipt = checkpoint(self.root, arm="off")
    self.assertEqual(receipt["status"], "prepared-for-durable-upload")
    self.assertEqual(receipt["record_pairs"], 7)
    self.assertNotIn("mismatch-capsule.npz", receipt["files"])
    self.assertEqual(
        receipt["manifest_sha256"],
        _sha256(self.root / "CLASSIFIER_INPUT_SHA256SUMS"),
    )

  def test_red_input_checkpoint_requires_and_hashes_capsule(self) -> None:
    self._alignment(1)
    (self.root / "mismatch-capsule.npz").write_bytes(b"capsule")
    receipt = checkpoint(self.root, arm="on")
    self.assertIn("mismatch-capsule.npz", receipt["files"])
    rows = (self.root / "CLASSIFIER_INPUT_SHA256SUMS").read_text().splitlines()
    self.assertEqual(len(rows), 4)

  def test_red_input_without_capsule_fails_closed(self) -> None:
    self._alignment(1)
    with self.assertRaisesRegex(M15ClassifierInputError, "capsule presence"):
      checkpoint(self.root, arm="on")


if __name__ == "__main__":
  unittest.main()
