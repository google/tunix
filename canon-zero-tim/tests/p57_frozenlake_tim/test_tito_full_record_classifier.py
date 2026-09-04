#!/usr/bin/env python3
"""Tests for the P45/M15 exact-TiTO full-record classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tasks/multiturn-tito-cross-workload/scripts/"
    "classify_tito_full_record.py"
)
SPEC = importlib.util.spec_from_file_location("p57_tito_full_classifier", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P57 TiTO full classifier")
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


def _write_json(path: Path, value: dict, *, mode: int = 0o600) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
  path.chmod(mode)


def _write_jsonl(path: Path, values: list[dict], *, mode: int = 0o600) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
      encoding="utf-8",
  )
  path.chmod(mode)


def _write_sidecar(
    state: Path,
    *,
    step: int,
    rows: list[dict],
    pre_record: dict,
) -> None:
  count = len(rows)
  arrays = {
      "prompt_ids": np.arange(count * 2, dtype=np.int32).reshape(count, 2),
      "prompt_mask": np.ones((count, 2), dtype=np.bool_),
      "completion_ids": np.arange(count * 3, dtype=np.int32).reshape(count, 3),
      "completion_valid_mask": np.ones((count, 3), dtype=np.bool_),
      "action_mask": np.ones((count, 3), dtype=np.bool_),
      "s_decode": np.zeros((count, 3), dtype=np.float32),
      "s_prefill": np.zeros((count, 3), dtype=np.float32),
      "t_old": np.zeros((count, 3), dtype=np.float32),
      "policy_version": np.full((count,), step, dtype=np.int32),
      "sampling_values": np.tile(
          np.asarray([[0.7, 0.0, 1.0]], dtype=np.float32), (count, 1)
      ),
      "sequence_row": np.arange(count, dtype=np.int32),
      "trajectory_id": np.asarray(
          [row["trajectory_id"] for row in rows], dtype="S32"
      ),
      "group_id": np.asarray([row["group_id"] for row in rows], dtype=np.int64),
      "pair_index": np.asarray(
          [row["pair_index"] for row in rows], dtype=np.int32
      ),
  }
  record_payload = json.dumps(
      pre_record, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode()
  metadata = {
      "schema": "canon.p57-tito-update-sidecar.v1",
      "workload": "p45",
      "step": step,
      "rows": count,
      "dp": 8,
      "tp": 8,
      "source_commit": "a" * 40,
      "image_identity": "example/image@sha256:" + "b" * 64,
      "alignment_record_sha256": hashlib.sha256(record_payload).hexdigest(),
      "request_ids": [row["request_ids"] for row in rows],
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": hashlib.sha256(
                  np.ascontiguousarray(value).tobytes()
              ).hexdigest(),
          }
          for name, value in arrays.items()
      },
  }
  directory = state / "p57_tito_witness/update-sidecars"
  directory.mkdir(parents=True, exist_ok=True, mode=0o700)
  directory.chmod(0o700)
  path = directory / f"step-{step:06d}.npz"
  with path.open("wb") as output:
    np.savez(
        output,
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        ),
        **arrays,
    )
  path.chmod(0o600)
  pre_record["tito_update_sidecar"] = {
      "schema": "canon.p57-tito-update-sidecar-receipt.v1",
      "path": str(path),
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "bytes": path.stat().st_size,
      "logical_bytes": sum(value.nbytes for value in arrays.values()),
      "write_seconds": 0.01,
      "rows": count,
      "step": step,
  }


def _fixture(root: Path, *, red: bool) -> tuple[Path, Path, Path]:
  state = root / "state"
  rows = []
  for step in range(2):
    for sequence_row in range(4):
      ordinal = step * 4 + sequence_row
      rows.append({
          "schema": "canon.p57-tito-row-map.v1",
          "trajectory_id": f"{ordinal + 1:032x}",
          "request_ids": [
              f"request-{ordinal}-{turn}"
              for turn in range(2 if sequence_row < 2 else 1)
          ],
          "policy_step": step,
          "group_id": step * 2 + sequence_row // 2,
          "pair_index": sequence_row % 2,
          "sequence_row": sequence_row,
          "later_turns": 1 if sequence_row < 2 else 0,
          "token_different": red and ordinal == 4,
      })
  collection = {
      "active": True,
      "mode": "record-full",
      "trajectories": 8,
      "compared_trajectories": 4,
      "unexercised_single_turn_trajectories": 4,
      "equal_trajectories": 3 if red else 4,
      "different_trajectories": 1 if red else 0,
      "later_turn_comparisons": 4,
      "engine_echo_comparisons": 12,
      "engine_echo_differences": 1 if red else 0,
      "token_difference_events": 1 if red else 0,
      "capsules_reserved": 1 if red else 0,
      "capsules_emitted": 1 if red else 0,
      "capsules_omitted": 0,
      "emission_failures": 0,
      "backward_transactions": 2,
      "gradient_microbatches": 4,
      "optimizer_commits": 2,
      "alignment_updates": 2,
  }
  summary = {
      "schema": "canon.p57-tito-full-record.v1",
      "workload": "p45",
      "source_commit": "a" * 40,
      "image_identity": "example/image@sha256:" + "b" * 64,
      "dp": 8,
      "tp": 8,
      "expected_updates": 2,
      "train_steps_before": 0,
      "train_steps_after": 2,
      "global_steps_before": 0,
      "global_steps_after": 2,
      "optimizer_commits": 2,
      "global_updates": 2,
      "checkpoint_writes": 0,
      "checkpoint_observation": {
          "configured_root": None,
          "latest_before": None,
          "latest_after": None,
      },
      "token_verdict": "DIFFERENT" if red else "EQUAL",
      "collection": collection,
  }
  _write_json(state / "p57_tito_witness/full-record-summary.json", summary)
  _write_json(
      state / "p57_tito_witness/single-writer.json",
      {
          "schema": "canon.p57-tito-single-writer.v1",
          "status": "PASS",
          "workload": "p45",
          "source_commit": "a" * 40,
          "image_identity": "example/image@sha256:" + "b" * 64,
          "dp": 8,
          "tp": 8,
          "controller_pid": 123,
          "controller_hostname": "controller",
          "writer_contract": "one-python-controller-o-excl",
          "neutrality_arm": None,
      },
  )
  _write_json(
      state / "p57_tito_gcs/orbax-probe.json",
      {
          "schema": "canon.p57-tito-orbax-admission-receipt.v1",
          "status": "PASS",
          "workload": "p45",
          "source_commit": "a" * 40,
          "image_identity": "example/image@sha256:" + "b" * 64,
          "dp": 8,
          "tp": 8,
          "probe_root_sha256": "c" * 64,
          "saved_step": 0,
          "restored_step": 0,
          "restored_equal": True,
          "elapsed_seconds": 1.0,
          "failure_type": None,
      },
  )
  _write_jsonl(state / "p57_tito_witness/full-row-map.jsonl", rows)
  updates = [
      {
          "verdict": "PASS",
          "microsteps": 2,
          "commits": 1,
          "alignment_hashes": [f"{step + 1:064x}"],
      }
      for step in range(2)
  ]
  _write_jsonl(state / "updates.jsonl", updates)
  pre_rows = [
      {
          "step": step,
          "verdict": "PASS",
          "blocking_reds": [],
          "warning_reds": [],
          "reported_reds": [],
      }
      for step in range(2)
  ]
  for step, pre_record in enumerate(pre_rows):
    _write_sidecar(
        state,
        step=step,
        rows=[row for row in rows if row["policy_step"] == step],
        pre_record=pre_record,
    )
  _write_jsonl(state / "pre_alignment.jsonl", pre_rows)
  _write_jsonl(
      state / "alignment.jsonl",
      [
          {
              "verdict": "PASS",
              "blocking_reds": [],
              "warning_reds": [],
              "reported_reds": [],
          }
          for _ in range(2)
      ],
  )
  base = state / "base.json"
  v1 = state / "v1.json"
  _write_json(base, {"verdict": "PASS"})
  _write_json(v1, {"verdict": "PASS"})
  if red:
    submitted = [1, 2]
    echoed = [1, 3]
    digest = lambda tokens: hashlib.sha256(  # noqa: E731
        b"".join(int(token).to_bytes(8, "little", signed=True) for token in tokens)
    ).hexdigest()
    _write_json(
        state / "token-continuity-first-diff/echo.json",
        {
            "schema": "canon.p57-tito-echo-diff.v1",
            "event_index": 1,
            "witness": {
                "schema": "canon.p57-tito-host-witness.v1",
                "request_id": "request-4-0",
                "trajectory_id": f"{5:032x}",
                "workload": "p45",
                "turn": 0,
                "pair_index": "0",
                "group_id": "0",
                "submitted_tokens": 2,
                "submitted_sha256": digest(submitted),
                "engine_echo_tokens": 2,
                "engine_echo_sha256": digest(echoed),
                "submitted_equals_engine_echo": False,
            },
            "submitted_token_ids": submitted,
            "engine_echo_token_ids": echoed,
        },
    )
  return state, base, v1


class TitoFullRecordClassifierTest(unittest.TestCase):

  def test_snapshot_trigger_ladder_is_bounded_and_first_per_threshold(self):
    pre = []
    for step, max_abs in enumerate((0.2, 2.0, 9.0, 33.0, 64.0)):
      pre.append({
          "step": step,
          "boundaries": {
              "S_decode_vs_S_prefill": {
                  "valid": True,
                  "finite": True,
                  "differing_bytes": 1,
                  "max_abs": max_abs,
              }
          },
      })
    self.assertEqual(
        classifier._expected_actor_snapshot_triggers(pre),
        {
            0: ["first-any"],
            1: ["first-ge-1"],
            2: ["first-ge-8"],
            3: ["first-ge-32"],
        },
    )

  def _classify(self, state: Path, base: Path, v1: Path) -> dict:
    return classifier.classify(
        state=state,
        recipe="p45",
        base_classification=base,
        v1_classification=v1,
        _expected_updates=2,
        _rows_per_update=4,
    )

  def test_green_full_record_is_strict_zero_tim(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, base, v1 = _fixture(Path(tmp), red=False)
      result = self._classify(state, base, v1)
      self.assertEqual(result["execution_verdict"], "PASS")
      self.assertEqual(result["token_verdict"], "EQUAL")
      self.assertEqual(result["zero_tim_verdict"], "PASS")
      self.assertEqual(result["claim"], "STRICT_ZERO_TIM")

  def test_red_full_record_completes_but_cannot_claim_zero_tim(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, base, v1 = _fixture(Path(tmp), red=True)
      result = self._classify(state, base, v1)
      self.assertEqual(result["execution_verdict"], "PASS")
      self.assertEqual(result["token_verdict"], "DIFFERENT")
      self.assertEqual(result["zero_tim_verdict"], "FAIL")
      self.assertEqual(result["claim"], "NON_ZERO_TIM_DATA_COLLECTION")

  def test_update_zero_token_red_is_completed_data_collection(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, base, v1 = _fixture(Path(tmp), red=True)
      row_map = state / "p57_tito_witness/full-row-map.jsonl"
      rows = [json.loads(line) for line in row_map.read_text().splitlines()]
      rows[4]["token_different"] = False
      rows[0]["token_different"] = True
      _write_jsonl(row_map, rows)
      capsule = state / "token-continuity-first-diff/echo.json"
      value = json.loads(capsule.read_text())
      value["witness"]["trajectory_id"] = f"{1:032x}"
      value["witness"]["request_id"] = "request-0-0"
      _write_json(capsule, value)
      result = self._classify(state, base, v1)
      self.assertEqual(result["execution_verdict"], "PASS")
      self.assertEqual(result["token_verdict"], "DIFFERENT")
      self.assertEqual(result["claim"], "NON_ZERO_TIM_DATA_COLLECTION")

  def test_missing_or_duplicate_token_event_fails_evidence(self):
    for mutation in ("missing", "duplicate"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, base, v1 = _fixture(Path(tmp), red=True)
        capsule = state / "token-continuity-first-diff/echo.json"
        if mutation == "missing":
          capsule.unlink()
        else:
          duplicate = json.loads(capsule.read_text())
          _write_json(
              state / "token-continuity-first-diff/echo-duplicate.json",
              duplicate,
          )
          summary_path = state / "p57_tito_witness/full-record-summary.json"
          summary = json.loads(summary_path.read_text())
          summary["collection"]["token_difference_events"] = 2
          summary["collection"]["capsules_reserved"] = 2
          summary["collection"]["capsules_emitted"] = 2
          _write_json(summary_path, summary)
        result = self._classify(state, base, v1)
        self.assertEqual(result["execution_verdict"], "FAIL")
        self.assertTrue(
            any(
                reason.startswith("capsule_")
                for reason in result["reasons"]
            )
        )

  def test_missing_join_and_false_counter_claims_fail(self):
    for mutation in ("request", "step", "counter", "checkpoint"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, base, v1 = _fixture(Path(tmp), red=True)
        if mutation == "request":
          row_map = state / "p57_tito_witness/full-row-map.jsonl"
          rows = [json.loads(line) for line in row_map.read_text().splitlines()]
          rows[0]["request_ids"] = ["foreign"]
          _write_jsonl(row_map, rows)
        elif mutation == "step":
          capsule = state / "token-continuity-first-diff/echo.json"
          value = json.loads(capsule.read_text())
          value["witness"]["trajectory_id"] = "f" * 32
          _write_json(capsule, value)
        else:
          summary_path = state / "p57_tito_witness/full-record-summary.json"
          summary = json.loads(summary_path.read_text())
          if mutation == "counter":
            summary["collection"]["optimizer_commits"] = 1
          else:
            summary["checkpoint_observation"]["latest_after"] = 2
          _write_json(summary_path, summary)
        result = self._classify(state, base, v1)
        self.assertEqual(result["execution_verdict"], "FAIL")
        self.assertEqual(result["zero_tim_verdict"], "FAIL")
        self.assertTrue(result["reasons"])

  def test_nominal_pass_with_hidden_red_list_fails_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, base, v1 = _fixture(Path(tmp), red=False)
      path = state / "pre_alignment.jsonl"
      rows = [json.loads(line) for line in path.read_text().splitlines()]
      rows[0]["warning_reds"] = ["S_decode_vs_S_prefill"]
      _write_jsonl(path, rows)
      result = self._classify(state, base, v1)
      self.assertEqual(result["execution_verdict"], "FAIL")
      self.assertEqual(result["zero_tim_verdict"], "FAIL")
      self.assertIn("pre_alignment_0_pass_with_warning_reds", result["reasons"])

  def test_missing_or_tampered_update_sidecar_fails(self):
    for mutation in ("missing", "tampered"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
        state, base, v1 = _fixture(Path(tmp), red=False)
        path = state / "p57_tito_witness/update-sidecars/step-000001.npz"
        if mutation == "missing":
          path.unlink()
        else:
          with path.open("ab") as output:
            output.write(b"tamper")
        result = self._classify(state, base, v1)
        self.assertEqual(result["execution_verdict"], "FAIL")
        self.assertEqual(result["zero_tim_verdict"], "FAIL")
        self.assertTrue(
            any(reason.startswith("update_sidecar") for reason in result["reasons"])
        )

  def test_snapshot_request_does_not_change_prior_sidecar_record_digest(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, _, _ = _fixture(Path(tmp), red=False)
      row_maps = [
          json.loads(line)
          for line in (
              state / "p57_tito_witness/full-row-map.jsonl"
          ).read_text().splitlines()
      ]
      pre = [
          json.loads(line)
          for line in (state / "pre_alignment.jsonl").read_text().splitlines()
      ]
      pre[0]["tito_actor_snapshot_request"] = {
          "schema": "canon.p57-tito-actor-snapshot-request-receipt.v1",
          "step": 0,
      }
      reasons = []
      classifier._validate_update_sidecars(
          state=state,
          recipe="p45",
          expected_updates=2,
          rows_per_update=4,
          source_commit="a" * 40,
          image_identity="example/image@sha256:" + "b" * 64,
          row_maps=row_maps,
          pre=pre,
          reasons=reasons,
      )
      self.assertEqual(reasons, [])

  def test_actor_snapshot_receipt_proves_pre_update_actor_only_state(self):
    with tempfile.TemporaryDirectory() as tmp:
      state = Path(tmp) / "state"
      request_path = state / (
          "p57_tito_witness/actor-snapshot-requests/step-000003.json"
      )
      request = {
          "schema": "canon.p57-tito-actor-snapshot-request.v1",
          "status": "PENDING",
          "step": 3,
          "policy_version": 3,
          "categories": ["first-any", "first-ge-1"],
          "max_abs": 2.0,
          "sidecar_sha256": "b" * 64,
          "source_commit": "a" * 40,
          "image_identity": "example/image@sha256:" + "b" * 64,
          "workload": "p45",
          "dp": 8,
          "tp": 8,
      }
      _write_json(request_path, request)
      request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
      pre = [{
          "step": 3,
          "boundaries": {
              "S_decode_vs_S_prefill": {
                  "valid": True,
                  "finite": True,
                  "differing_bytes": 4,
                  "max_abs": 2.0,
              }
          },
          "tito_update_sidecar": {"sha256": "b" * 64},
          "tito_actor_snapshot_request": {
              "schema": "canon.p57-tito-actor-snapshot-request-receipt.v1",
              "path": str(request_path),
              "sha256": request_sha,
              "bytes": request_path.stat().st_size,
              "step": 3,
              "categories": ["first-any", "first-ge-1"],
              "max_abs": 2.0,
          },
      }]
      receipt_path = state / (
          "p57_tito_witness/actor-snapshot-receipts/step-000003.json"
      )
      receipt = {
          "schema": "canon.p57-tito-actor-snapshot-receipt.v1",
          "status": "PASS",
          "step": 3,
          "policy_version": 3,
          "categories": ["first-any", "first-ge-1"],
          "max_abs": 2.0,
          "source_commit": "a" * 40,
          "image_identity": "example/image@sha256:" + "b" * 64,
          "workload": "p45",
          "dp": 8,
          "tp": 8,
          "request_path": str(request_path),
          "request_sha256": request_sha,
          "snapshot_root": (
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
              "p57-tito-test/attempt-direct/actor-snapshots"
          ),
          "snapshot_root_sha256": "c" * 64,
          "latest_step": 3,
          "optimizer_included": False,
          "resumable": False,
          "actor_train_steps_before": 3,
          "actor_train_steps_after": 3,
          "save_seconds": 1.0,
          "model_inventory": {
              "leaves": [{
                  "path": ".x", "shape": [2], "dtype": "float32",
                  "logical_bytes": 8,
              }],
              "leaf_count": 1,
              "logical_bytes": 8,
              "bounded_fingerprint": {"leaves": {".x": {"sha256": "d" * 64}}},
          },
          "failure_type": None,
      }
      receipt["snapshot_root_sha256"] = hashlib.sha256(
          receipt["snapshot_root"].encode()
      ).hexdigest()
      _write_json(receipt_path, receipt)
      reasons = []
      counts = classifier._validate_actor_snapshots(
          state=state,
          source_commit="a" * 40,
          image_identity="example/image@sha256:" + "b" * 64,
          recipe="p45",
          pre=pre,
          reasons=reasons,
      )
      self.assertEqual(counts, (1, 1))
      self.assertEqual(reasons, [])

      receipt["optimizer_included"] = True
      _write_json(receipt_path, receipt)
      reasons = []
      classifier._validate_actor_snapshots(
          state=state,
          source_commit="a" * 40,
          image_identity="example/image@sha256:" + "b" * 64,
          recipe="p45",
          pre=pre,
          reasons=reasons,
      )
      self.assertIn("actor_snapshot_receipt:3", reasons)


if __name__ == "__main__":
  unittest.main()
