#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
MODULE = (
    ROOT / "canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/"
    "review_m15_attempt18_e0_return.py"
)
RECOVERY_WRAPPER = (
    ROOT / "canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/"
    "run_m15_attempt18_e0_return_recovery.sh"
)
COMMITTED_971_RETURN = (
    ROOT / "canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/"
    "v1_apc_m15_attempt18_e0_kv_20260829"
)
SPEC = importlib.util.spec_from_file_location("m15_e0_return_review", MODULE)
assert SPEC and SPEC.loader
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)
SOURCE = "12207e3281db13461350fe7ef68dbaadfe713a58"
CLASSIFIER_SHA = (
    "99cc7d9c50777a9be182e2edd33a3cdca3daabaa396c019e4925e0ac531049f6"
)


def _digest(label: str) -> str:
  return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(path: Path, value: dict) -> None:
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class Attempt18E0ReturnReviewTest(unittest.TestCase):

  def _classifier(self, arm: str) -> dict:
    comparisons = [{
        "source_a_record_index": index * 2,
        "source_a_request_id": f"decode-{index * 2}",
        "clean_b_record_index": index * 2 + 1,
        "clean_b_request_id": f"clean-{index * 2 + 1}",
        "diagnostic_round": 0,
        "target_seq_len": 1226,
        "valid_tokens": [16] * 76 + [10],
        "aggregate_prefix_cells_differing": 0,
        "sample_prefix_cells_differing": 0,
        "differing_layers": [],
        "differing_logical_pages": [],
        "first_difference": None,
        "fingerprint_equal": True,
    } for index in range(8)]
    source_inputs = {
        "classifier": {
            "path": "classify_p38_kv_observer.py",
            "sha256": CLASSIFIER_SHA,
        },
        "observer_records": [{
            "arm": "A" if index % 2 == 0 else "B",
            "record_index": index,
            "json": f"kv-observer-{index}.json",
            "json_sha256": _digest(f"observer-json-{arm}-{index}"),
            "npz": f"kv-observer-{index}.npz",
            "npz_sha256": _digest(f"observer-npz-{arm}-{index}"),
            "valid_tokens": [16] * 76 + [10],
        } for index in range(16)],
        "capsules": [],
    }
    binding = None
    classification = "observer_pairs_valid_red_join_pending"
    if arm == "on":
      classification = "live_kv_fingerprint_equal_on_red_row"
      source_inputs["capsules"] = [{
          "path": "capsule.npz",
          "sha256": _digest("capsule"),
      }]
      source_inputs["replay_ledger"] = {
          "path": "ledger.jsonl",
          "sha256": _digest("replay-ledger"),
      }
      binding = {
          "schema": "m15-kv-source-request-binding-v1",
          "status": "UNIQUE_FUTURE_PREFIX_BINDING",
          "diagnostic_round": 0,
          "source_row": 217,
          "capsule": "capsule.npz",
          "anchor_prefix_tokens": 1226,
          "selected_request_id": "decode-0",
          "selected_source_a_record_index": 0,
          "required_elimination_horizon": 1227,
          "selected_proof_prefix_tokens": 1300,
          "candidates": [
              {
                  "source_a_record_index": 0,
                  "request_id": "decode-0",
                  "status": "FUTURE_PREFIX_MATCH",
                  "matching_prefix_lengths": [1300],
                  "conflicting_prefix_lengths": [],
              },
              *[{
                  "source_a_record_index": index * 2,
                  "request_id": f"decode-{index * 2}",
                  "status": "FUTURE_PREFIX_CONFLICT",
                  "matching_prefix_lengths": [],
                  "conflicting_prefix_lengths": [1227],
              } for index in range(1, 8)],
          ],
      }
    return {
        "schema": "p38-live-kv-classification-v2",
        "status": "PASS",
        "classification": classification,
        "records": 16,
        "pairs": 8,
        "comparisons": comparisons,
        "red_joins": [] if arm == "off" else [{
            "source_a_record_index": index * 2,
            "diagnostic_round": 0,
            "source_row": 217,
            "capsule": "capsule.npz",
            "mismatch_positions": [88],
            "mismatch_count": 1,
            "target_seq_len": 1226,
        } for index in range(8)],
        "source_request_binding": binding,
        "source_inputs": source_inputs,
        "claim_level": "bit-level-diagnostic-fingerprint-not-full-kv-bytes",
        "claim_ceiling": [
            "A/B token prefixes and valid extents are exact.",
            "The integer aggregates and fixed samples are diagnostic fingerprints, not cryptographic hashes.",
            "An equal fingerprint does not mathematically prove full KV byte equality.",
            "Only a candidate joined to an A/B-red capsule row can choose the mechanism branch.",
        ],
    }

  def _fixture(self, root: Path) -> Path:
    classifiers = {}
    for arm in ("off", "on"):
      path = root / f"{arm}.kv-observer-classification.json"
      _canonical(path, self._classifier(arm))
      classifiers[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema": "m15-attempt18-e0-kv-return-v1",
        "status": "LIVE_KV_FINGERPRINT_EQUAL",
        "source_commit": SOURCE,
        "target_executed": True,
        "remote_mutation": False,
        "numerical_repair_authorized": False,
        "claim_ceiling": (
            "The KV result is a diagnostic fingerprint over the uniquely bound "
            "red request, not a collision-free proof of all KV bytes."
        ),
        "arms": {
            "off": {
                "a_b_differing_bytes": 0,
                "a_b_differing_elements": 0,
                "b_c_differing_bytes": 0,
                "n_action": 123010,
                "kv_all_pairs_equal": True,
                "kv_classification": "observer_pairs_valid_red_join_pending",
                "source_request_binding": None,
                "root_manifest_sha256": _digest("off-root-manifest"),
                "kv_classification_sha256": classifiers["off"],
                "execution_receipts": {
                    "run_log_sha256": _digest("off-run-log"),
                    "runtime_source_exact": True,
                    "b_full_reset": True,
                    "all_num_cached_tokens_zero": True,
                    "zero_backward": True,
                    "zero_optimizer_commit": True,
                },
            },
            "on": {
                "a_b_differing_bytes": 1499,
                "a_b_differing_elements": 88,
                "b_c_differing_bytes": 0,
                "n_action": 117834,
                "kv_all_pairs_equal": True,
                "kv_classification": "live_kv_fingerprint_equal_on_red_row",
                "source_request_binding": self._classifier("on")[
                    "source_request_binding"
                ],
                "root_manifest_sha256": _digest("on-root-manifest"),
                "kv_classification_sha256": classifiers["on"],
                "execution_receipts": {
                    "run_log_sha256": _digest("on-run-log"),
                    "runtime_source_exact": True,
                    "b_full_reset": True,
                    "all_num_cached_tokens_zero": True,
                    "zero_backward": True,
                    "zero_optimizer_commit": True,
                },
            },
        },
    }
    _canonical(root / "E0_KV_RETURN.json", report)
    names = sorted(path.name for path in root.glob("*.json"))
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    ))
    raw = root.parent / "return.log"
    raw.write_text(
        "M15_E0_KV_RETURN_PASS status=LIVE_KV_FINGERPRINT_EQUAL "
        "control_a_b=0 treatment_a_b=1499 b_c=0\n"
        "[M15.E0.KV.RETURN] COMPLETE status=LIVE_KV_FINGERPRINT_EQUAL\n"
        "[M15.E0.KV.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0\n"
    )
    return raw

  def test_official_equal_return_passes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      raw = self._fixture(root)
      result = reviewer.review(root, SOURCE, raw)
      self.assertEqual(result["status"], "LIVE_KV_FINGERPRINT_EQUAL")
      self.assertEqual(result["inventory_members"], 3)
      self.assertFalse(result["numerical_repair_authorized"])

  def test_committed_971_schema_shaped_return_is_rejected(self):
    with self.assertRaisesRegex(
        reviewer.ReturnReviewError, "source identity/provenance"
    ):
      reviewer.review(COMMITTED_971_RETURN, SOURCE)

  def test_collapsed_observer_json_digests_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "off.kv-observer-classification.json"
      value = json.loads(path.read_text())
      for record in value["source_inputs"]["observer_records"]:
        record["json_sha256"] = _digest("one-impossible-json")
      _canonical(path, value)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(
          reviewer.ReturnReviewError, "collapse distinct records"
      ):
        reviewer.review(root, SOURCE)

  def test_absolute_observer_path_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "on.kv-observer-classification.json"
      value = json.loads(path.read_text())
      value["source_inputs"]["observer_records"][0]["json"] = (
          "/tmp/kv-observer-0.json"
      )
      _canonical(path, value)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(
          reviewer.ReturnReviewError, "source identity/digest"
      ):
        reviewer.review(root, SOURCE)

  def test_collapsed_off_on_root_manifests_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "E0_KV_RETURN.json"
      report = json.loads(path.read_text())
      report["arms"]["on"]["root_manifest_sha256"] = (
          report["arms"]["off"]["root_manifest_sha256"]
      )
      _canonical(path, report)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(
          reviewer.ReturnReviewError, "off/on provenance digests"
      ):
        reviewer.review(root, SOURCE)

  def test_missing_classifier_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      (root / "off.kv-observer-classification.json").unlink()
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "inventory"):
        reviewer.review(root, SOURCE)

  def test_md5_length_summary_digest_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "E0_KV_RETURN.json"
      report = json.loads(path.read_text())
      report["arms"]["on"]["kv_classification_sha256"] = (
          "d41d8cd98f00b204e9800998ecf8427e"
      )
      _canonical(path, report)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "arm summary"):
        reviewer.review(root, SOURCE)

  def test_truncated_binding_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "on.kv-observer-classification.json"
      value = json.loads(path.read_text())
      value["source_request_binding"].pop("candidates")
      _canonical(path, value)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "binding"):
        reviewer.review(root, SOURCE)

  def test_missing_terminal_marker_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      raw = self._fixture(root)
      raw.write_text("M15_E0_KV_RETURN_PASS status=LIVE_KV_FINGERPRINT_EQUAL\n")
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "terminal"):
        reviewer.review(root, SOURCE, raw)

  def test_missing_b_full_reset_receipt_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "E0_KV_RETURN.json"
      report = json.loads(path.read_text())
      report["arms"]["on"]["execution_receipts"]["b_full_reset"] = False
      _canonical(path, report)
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "arm summary"):
        reviewer.review(root, SOURCE)

  def test_noncanonical_manual_summary_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "return"
      root.mkdir()
      self._fixture(root)
      path = root / "E0_KV_RETURN.json"
      value = json.loads(path.read_text())
      path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
      (root / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.name}\n"
          for item in sorted(root.glob("*.json"))
      ))
      with self.assertRaisesRegex(reviewer.ReturnReviewError, "canonical"):
        reviewer.review(root, SOURCE)

  def test_recovery_wrapper_has_fail_closed_local_preflight(self):
    text = RECOVERY_WRAPPER.read_text(encoding="utf-8")
    self.assertIn('runtime_source="12207e3281db13461350fe7ef68dbaadfe713a58"', text)
    self.assertIn('case "$branch" in', text)
    self.assertIn('local/*)', text)
    self.assertIn('[ "$head" = "$analysis_source" ]', text)
    self.assertIn('status --porcelain', text)
    self.assertIn('preflight_runtime.py', text)
    self.assertIn('test_review_m15_attempt18_e0_return.py', text)
    self.assertIn('sha256sum -c SHA256SUMS --quiet', text)

  def test_recovery_wrapper_delegates_read_only_and_preserves_raw_log(self):
    text = RECOVERY_WRAPPER.read_text(encoding="utf-8")
    self.assertIn('run_m15_attempt18_e0_kv_gcs_return.sh', text)
    self.assertIn('review_m15_attempt18_e0_return.py', text)
    self.assertIn('raw_log="$(mktemp', text)
    self.assertNotIn('rm -f -- "$raw_log"', text)
    self.assertNotIn("kubectl", text)
    self.assertIn(
        "READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0", text
    )
    official = RECOVERY_WRAPPER.with_name(
        "run_m15_attempt18_e0_kv_gcs_return.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('run.log pre-alignment.jsonl', official)
    self.assertIn('all_num_cached_tokens_zero=True', official)
    self.assertIn('"execution_receipts"', official)

  def test_official_wrapper_round_trips_fake_read_only_transport(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      render = root / "render"
      render.mkdir()
      remote_root = root / "remote"
      for arm in ("off", "on"):
        remote = remote_root / arm
        remote.mkdir(parents=True)
        prefix = f"gs://test-evidence/{arm}/attempt-0"
        raw = (
            f"[sync] HEAD={SOURCE}\n"
            "[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
            "all_num_cached_tokens_zero=True\n"
            f"[CANON_APC_M15_TARGET_CONTRACT] arm={arm} topology=DP8xTP8 "
            "workload=m15/main backward=0 optimizer_commits=0\n"
            "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 "
            "optimizer_commits=0\n"
        )
        (remote / "run.log").write_text(raw, encoding="utf-8")
        ab_bytes = 0 if arm == "off" else 1499
        ab_elements = 0 if arm == "off" else 88
        alignment = {
            "N_action": 123010 if arm == "off" else 117834,
            "boundaries": {
                "S_decode_vs_S_prefill": {
                    "differing_bytes": ab_bytes,
                    "differing_elements": ab_elements,
                },
                "S_prefill_vs_T_old": {
                    "differing_bytes": 0,
                    "differing_elements": 0,
                },
            },
        }
        (remote / "pre-alignment.jsonl").write_text(
            json.dumps(alignment, sort_keys=True) + "\n", encoding="utf-8"
        )
        _canonical(remote / "serving-classification.json", {
            "schema_version": 1,
            "verdict": "PASS",
            "scope": "p38-serving-capture",
            "source_commit": SOURCE,
        })
        _canonical(
            remote / "kv-observer-classification.json",
            self._classifier(arm),
        )
        data_names = (
            "run.log",
            "pre-alignment.jsonl",
            "serving-classification.json",
            "kv-observer-classification.json",
        )
        (remote / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256((remote / name).read_bytes()).hexdigest()}  {name}\n"
            for name in data_names
        ), encoding="ascii")
        manifest_sha = hashlib.sha256(
            (remote / "SHA256SUMS").read_bytes()
        ).hexdigest()
        _canonical(remote / "PREFLIGHT.json", {
            "schema": "canon-p38-gcs-preflight-v1",
            "status": "writable-and-source-verified",
            "source_verified": True,
            "source_commit": SOURCE,
            "runtime_source_commit": SOURCE,
            "prefix": prefix,
        })
        _canonical(remote / "COLLECTED.json", {
            "schema": "canon-p38-gcs-collection-v1",
            "status": "collected",
            "source_commit": SOURCE,
            "runtime_source_commit": SOURCE,
            "prefix": prefix,
        })
        _canonical(remote / "COMPLETE.json", {
            "schema": "canon-p38-gcs-completion-v1",
            "status": "postflight-accepted",
            "source_commit": SOURCE,
            "runtime_source_commit": SOURCE,
            "prefix": prefix,
            "manifest_sha256": manifest_sha,
        })

        document = {
            "spec": {"replicatedJobs": [{"template": {"spec": {"template": {
                "spec": {"containers": [{
                    "name": "jax-tpu",
                    "env": [
                        {"name": "CANON_APC_M15_TARGET_DEBUG", "value": arm},
                        {"name": "CANON_EXPECT_COMMIT", "value": SOURCE},
                        {"name": "CANON_P38_GCS_PREFIX", "value": prefix},
                        {"name": "CANON_P38_KV_OBSERVER_LAYER", "value": "0"},
                    ],
                }]},
            }}}}]},
        }
        import yaml
        (render / f"jobset-v1-apc-m15-{arm}-kv.yaml").write_text(
            yaml.safe_dump(document), encoding="utf-8"
        )

      _canonical(render / "RUN_CONTRACT.json", {
          "schema": "m15-attempt18-e0-kv-render-v1",
          "rounds": 1,
          "launch_authorized": False,
          "observer": {"layer": 0, "target_aliases": 8},
      })
      render_names = sorted(path.name for path in render.iterdir())
      (render / "SHA256SUMS").write_text("".join(
          f"{hashlib.sha256((render / name).read_bytes()).hexdigest()}  {name}\n"
          for name in render_names
      ), encoding="ascii")

      fake_bin = root / "bin"
      fake_bin.mkdir()
      fake_gcloud = fake_bin / "gcloud"
      fake_gcloud.write_text(
          """#!/usr/bin/env bash
set -euo pipefail
[ "$1" = storage ] && [ "$2" = cp ]
uri="$3"
destination="$4"
rest="${uri#gs://test-evidence/}"
arm="${rest%%/*}"
name="${rest##*/}"
cp "$FAKE_E0_REMOTE/$arm/$name" "$destination"
""",
          encoding="utf-8",
      )
      fake_gcloud.chmod(0o755)
      output = root / "return"
      scratch = root / "scratch"
      scratch.mkdir()
      env = dict(os.environ)
      env["PATH"] = f"{fake_bin}:{env['PATH']}"
      env["FAKE_E0_REMOTE"] = str(remote_root)
      official = RECOVERY_WRAPPER.with_name(
          "run_m15_attempt18_e0_kv_gcs_return.sh"
      )
      result = subprocess.run(
          ["bash", str(official), str(render), str(output), str(scratch)],
          check=False,
          capture_output=True,
          text=True,
          env=env,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIn("M15_E0_KV_RETURN_PASS", result.stdout)
      self.assertIn("READ_ONLY gcs_read=1 gcs_write=0", result.stdout)
      raw_log = root / "official.log"
      raw_log.write_text(result.stdout, encoding="utf-8")
      admitted = reviewer.review(output, SOURCE, raw_log)
      self.assertEqual(admitted["status"], "LIVE_KV_FINGERPRINT_EQUAL")
      report = json.loads((output / "E0_KV_RETURN.json").read_text())
      self.assertTrue(
          report["arms"]["on"]["execution_receipts"]
          ["all_num_cached_tokens_zero"]
      )


if __name__ == "__main__":
  unittest.main()
