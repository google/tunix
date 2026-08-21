"""CPU contracts for resumable P46 DeepSWE evaluation artifacts."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "deepswe"
sys.path.insert(0, str(EXAMPLES))

import deepswe_eval_artifacts as artifacts  # pylint: disable=wrong-import-position


class EvalArtifactsTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    root = Path(self.temporary.name)
    self.config = artifacts.EvalConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_path=str(root / "Qwen3-4B-Instruct-2507"),
        dataset_name="R2E-Gym/R2E-Gym-Subset",
        dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
        dataset_split="train",
        dataset_rows=4578,
        whitelist_path=str(root / "clean.jsonl"),
        whitelist_sha256=(
            "2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f"
            "7b3474b0b307ed7"
        ),
        whitelist_rows=1851,
        source_commit="6" * 40,
        harness_commit="6" * 40,
        client_image="example.invalid/tunix@sha256:" + "7" * 64,
        topology="64",
        resume_tag="eval-resume-test",
    )

  def tearDown(self):
    self.temporary.cleanup()

  def test_signed_config_and_fingerprint(self):
    self.config.validate()
    self.assertEqual(len(self.config.fingerprint), 64)
    self.assertIn(self.config.fingerprint[:16], self.config.run_tag)
    other = dataclasses.replace(self.config, topology="128")
    other.validate()
    self.assertNotEqual(self.config.fingerprint, other.fingerprint)
    self.assertEqual(self.config.evaluation_mode, "reward_only")
    self.assertEqual(
        self.config.trajectory_mode, "reward_only_no_logprobs"
    )
    self.assertEqual(self.config.sampled_by, "stock@" + "6" * 40)
    self.assertEqual(
        self.config.sampling_rng_mode, "engine_global_sequential"
    )
    self.assertFalse(self.config.collect_logprobs)
    self.assertEqual(
        self.config.action_compat_mode, "q4_r2egym_xml_v2"
    )
    self.assertEqual(self.config.resume_tag, "eval-resume-test")
    with self.assertRaisesRegex(ValueError, "restricted"):
      dataclasses.replace(self.config, evaluation_mode="training").validate()
    with self.assertRaisesRegex(ValueError, "contract mismatch"):
      dataclasses.replace(self.config, max_response_length=4096).validate()
    with self.assertRaisesRegex(ValueError, "prefix cache"):
      dataclasses.replace(self.config, prefix_cache=True).validate()

  def test_onehost_probe_is_exact_and_does_not_relax_production(self):
    onehost = dataclasses.replace(
        self.config,
        topology="4",
        onehost_probe=True,
        max_model_len=4096,
        max_response_length=512,
        max_steps=1,
        n_sample=1,
        logical_tasks=1,
        shard_tasks=1,
        max_concurrency=1,
        trajectory_timeout_secs=900,
        step_timeout_secs=300,
        reward_timeout_secs=300,
        cleanup_timeout_secs=120,
        shard_timeout_secs=1200,
    )
    onehost.validate()
    self.assertIn("onehost", onehost.run_tag)
    self.assertEqual(onehost.max_steps, 1)
    self.assertEqual(self.config.max_steps, 50)
    with self.assertRaisesRegex(ValueError, "topology"):
      dataclasses.replace(self.config, topology="4").validate()
    with self.assertRaisesRegex(ValueError, "contract mismatch"):
      dataclasses.replace(onehost, n_sample=2).validate()

  def test_logprob_observer_is_restricted_to_64chip_n16_canary(self):
    observer = dataclasses.replace(
        self.config,
        parity_canary=True,
        evaluation_mode="logprob_observer",
        logical_tasks=1,
        shard_tasks=1,
        max_concurrency=16,
    )
    observer.validate()
    self.assertTrue(observer.collect_logprobs)
    self.assertEqual(
        observer.trajectory_mode, "observer_with_sampled_logprobs"
    )
    self.assertIn("parity-logprob_observer", observer.run_tag)
    with self.assertRaisesRegex(ValueError, "restricted"):
      dataclasses.replace(self.config, evaluation_mode="logprob_observer").validate()
    with self.assertRaisesRegex(ValueError, "topology"):
      dataclasses.replace(observer, topology="128").validate()

    entry = {"docker_image": "img-a"}
    record = artifacts.trajectory_record(
        observer,
        entry=entry,
        sample_index=0,
        trajectory={
            "status": "SUCCEEDED",
            "reward": 0.0,
            "steps": [{"logprobs": [-0.2]}],
        },
        elapsed_secs=1.0,
    )
    self.assertEqual(record["trajectory"]["steps"][0]["logprobs"], [-0.2])

  def test_record_is_complete_redacted_and_seeded(self):
    entry = {"docker_image": "img-a", "instance_id": "task-a"}
    record = artifacts.trajectory_record(
        self.config,
        entry=entry,
        sample_index=3,
        trajectory={
            "status": "SUCCEEDED",
            "reward": 1.0,
            "steps": [{"action": "ok", "api_token": "hf-secretsecretsecret"}],
        },
        elapsed_secs=12.5,
    )
    self.assertTrue(record["valid"])
    self.assertTrue(record["solved"])
    self.assertEqual(record["sample_index"], 3)
    self.assertEqual(
        record["sample_nonce"], self.config.sample_nonce("img-a", 3)
    )
    self.assertEqual(record["engine_seed"], 42)
    self.assertEqual(record["sampling_rng_mode"], "engine_global_sequential")
    self.assertEqual(
        record["trajectory"]["steps"][0]["api_token"], "<redacted>"
    )
    self.assertEqual(record["trajectory_mode"], "reward_only_no_logprobs")
    self.assertEqual(record["resume_tag"], "eval-resume-test")
    self.assertEqual(record["sampled_by"], "stock@" + "6" * 40)
    self.assertEqual(
        record["validity_reason"], "completed_under_signed_budget"
    )

  def test_signed_budget_terminals_are_valid_unsolved_without_retry(self):
    entry = {"docker_image": "img-a"}
    for status in (
        "MAX_STEPS_REACHED",
        "MAX_CONTEXT_LIMIT_REACHED",
        "MODEL_TIMEOUT",
        "TIMEOUT",
    ):
      with self.subTest(status=status):
        record = artifacts.trajectory_record(
            self.config,
            entry=entry,
            sample_index=0,
            trajectory={"status": status, "reward": 0.0, "steps": []},
            elapsed_secs=1.0,
        )
        self.assertTrue(record["valid"])
        self.assertFalse(record["solved"])
        expected_reason = (
            "completed_model_timeout"
            if status == "MODEL_TIMEOUT"
            else "completed_under_signed_budget"
        )
        self.assertEqual(record["validity_reason"], expected_reason)

  def test_model_tool_syntax_failure_is_valid_unsolved_without_retry(self):
    entry = {"docker_image": "img-a"}
    for status in ("SUCCEEDED", "MODEL_TIMEOUT"):
      with self.subTest(status=status):
        record = artifacts.trajectory_record(
            self.config,
            entry=entry,
            sample_index=0,
            trajectory={
                "status": status,
                "reward": 0.0,
                "steps": [{
                    "observation": (
                        "file_editor: error: unrecognized arguments: "
                        "--command=view"
                    )
                }],
            },
            elapsed_secs=1.0,
        )
        self.assertTrue(record["valid"])
        self.assertFalse(record["solved"])
        self.assertEqual(
            record["validity_reason"],
            (
                "completed_model_timeout"
                if status == "MODEL_TIMEOUT"
                else "completed_with_model_action_errors"
            ),
        )
        self.assertEqual(record["model_action_errors"], 1)
        remaining = artifacts.remaining_samples(
            [entry], [record], config=self.config
        )
        self.assertEqual(len(remaining), 15)
        self.assertNotIn((entry, 0, 1), remaining)

  def test_q4_action_repair_provenance_is_recorded(self):
    record = artifacts.trajectory_record(
        self.config,
        entry={"docker_image": "img-a"},
        sample_index=0,
        trajectory={
            "status": "SUCCEEDED",
            "reward": 0.0,
            "steps": [{
                "model_response": (
                    "thinking\n<function=execute_bash>"
                    "<parameter=cmd=ls</parameter></function>"
                ),
                "action": (
                    "<function=execute_bash>"
                    "<parameter=cmd>ls</parameter></function>"
                ),
                "observation": "ok",
            }],
        },
        elapsed_secs=1.0,
    )
    self.assertTrue(record["valid"])
    self.assertEqual(record["action_compat_mode"], "q4_r2egym_xml_v2")
    self.assertEqual(record["action_compat_repairs"], 1)

  def test_reward_only_logprobs_are_absent_or_null_never_numeric(self):
    entry = {"docker_image": "img-a", "instance_id": "task-a"}
    record = artifacts.trajectory_record(
        self.config,
        entry=entry,
        sample_index=0,
        trajectory={
            "status": "SUCCEEDED",
            "reward": 0.0,
            "steps": [{"action": "ok", "logprobs": []}],
            "old_logprobs": None,
        },
        elapsed_secs=1.0,
    )
    self.assertIsNone(record["trajectory"]["steps"][0]["logprobs"])
    self.assertIsNone(record["trajectory"]["old_logprobs"])
    for payload in (0.0, [0.0], [-0.0001]):
      with self.subTest(payload=payload):
        with self.assertRaisesRegex(ValueError, "never numeric"):
          artifacts.trajectory_record(
              self.config,
              entry=entry,
              sample_index=0,
              trajectory={
                  "status": "SUCCEEDED",
                  "reward": 0.0,
                  "steps": [{"logprobs": payload}],
              },
              elapsed_secs=1.0,
          )

  def test_resume_retries_invalid_but_rejects_attempts_after_valid(self):
    root = Path(self.temporary.name)
    path = root / "raw.jsonl"
    entry = {"docker_image": "img-a"}
    invalid = artifacts.trajectory_record(
        self.config,
        entry=entry,
        sample_index=0,
        attempt_index=0,
        trajectory={"status": "ENV_TIMEOUT", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    artifacts.append_record(path, invalid)
    loaded = artifacts.load_records(
        [path], config=self.config, allowed_task_keys={"img-a"}
    )
    remaining = artifacts.remaining_samples([entry], loaded, config=self.config)
    self.assertEqual(len(remaining), 16)
    self.assertEqual(remaining[0][1:], (0, 1))
    unattempted = artifacts.unattempted_samples(
        [entry], loaded, config=self.config
    )
    self.assertEqual(len(unattempted), 15)
    self.assertNotIn(0, {sample_index for _, sample_index, _ in unattempted})
    deferred = artifacts.deferred_samples(
        [entry], loaded, config=self.config
    )
    self.assertEqual(
        collections.Counter(item["state"] for item in deferred),
        {"invalid": 1, "unattempted": 15},
    )

    valid_retry = artifacts.trajectory_record(
        self.config,
        entry=entry,
        sample_index=0,
        attempt_index=1,
        trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    artifacts.append_record(path, valid_retry)
    loaded = artifacts.load_records(
        [path], config=self.config, allowed_task_keys={"img-a"}
    )
    remaining = artifacts.remaining_samples([entry], loaded, config=self.config)
    self.assertEqual(len(remaining), 15)
    self.assertNotIn(0, {sample_index for _, sample_index, _ in remaining})
    report = artifacts.aggregate_tasks([entry], loaded, config=self.config)[0]
    self.assertEqual(report["attempts"], 2)
    self.assertEqual(report["invalid_attempts"], 1)
    self.assertEqual(report["valid_n"], 1)

    duplicate_valid = dict(valid_retry, attempt_index=2)
    artifacts.append_record(path, duplicate_valid)
    with self.assertRaisesRegex(ValueError, "duplicate valid"):
      artifacts.load_records(
          [path], config=self.config, allowed_task_keys={"img-a"}
      )

    drift = root / "drift.jsonl"
    changed = dict(invalid)
    changed["config_fingerprint"] = "0" * 64
    artifacts.append_record(drift, changed)
    with self.assertRaisesRegex(ValueError, "fingerprint"):
      artifacts.load_records(
          [drift], config=self.config, allowed_task_keys={"img-a"}
      )

  def test_resume_rejects_nonconsecutive_attempt_indices(self):
    path = Path(self.temporary.name) / "attempt-gap.jsonl"
    record = artifacts.trajectory_record(
        self.config,
        entry={"docker_image": "img-a"},
        sample_index=0,
        attempt_index=1,
        trajectory={"status": "ENV_TIMEOUT", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    artifacts.append_record(path, record)
    with self.assertRaisesRegex(ValueError, "consecutive"):
      artifacts.load_records(
          [path], config=self.config, allowed_task_keys={"img-a"}
      )

  def test_resume_ignores_only_a_torn_tail_and_loads_the_next_file(self):
    root = Path(self.temporary.name)
    first = root / "a.jsonl"
    second = root / "b.jsonl"
    records = []
    for sample_index, path in ((0, first), (1, second)):
      record = artifacts.trajectory_record(
          self.config,
          entry={"docker_image": "img-a"},
          sample_index=sample_index,
          trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
          elapsed_secs=1.0,
      )
      artifacts.append_record(path, record)
      records.append(record)
    with first.open("ab") as output:
      output.write(b'{"torn":')
      output.flush()
    loaded = artifacts.load_records(
        [first, second], config=self.config, allowed_task_keys={"img-a"}
    )
    self.assertEqual(
        [(item["sample_index"], item["valid"]) for item in loaded],
        [(0, True), (1, True)],
    )

  def test_resume_contract_and_single_writer_lease_fail_closed(self):
    root = Path(self.temporary.name) / "campaign"
    first = artifacts.ensure_resume_contract(root, config=self.config)
    second = artifacts.ensure_resume_contract(root, config=self.config)
    self.assertEqual(first["sha256"], second["sha256"])
    with artifacts.campaign_lease(
        root, config=self.config, launch_id="launch-a"
    ):
      lease = json.loads((root / "resume_lease.json").read_text())
      self.assertEqual(lease["state"], "active")
      with self.assertRaisesRegex(RuntimeError, "active writer"):
        with artifacts.campaign_lease(
            root, config=self.config, launch_id="launch-b"
        ):
          self.fail("a second writer acquired the same resume tag")
    lease = json.loads((root / "resume_lease.json").read_text())
    self.assertEqual(lease["state"], "released")

    changed = dataclasses.replace(self.config, source_commit="8" * 40)
    with self.assertRaisesRegex(ValueError, "differs from exact payload"):
      artifacts.ensure_resume_contract(root, config=changed)

  def test_resume_tag_rejects_paths_and_uppercase(self):
    for value in ("../escape", "UPPER", "a" * 64, "dash-"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "resume_tag"):
          dataclasses.replace(self.config, resume_tag=value).validate()

  def test_frozen_legacy_v5_snapshot_imports_once_with_full_provenance(self):
    config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="adopt-old-run",
    )
    entry = {"docker_image": "img-a", "instance_id": "task-a"}
    record = artifacts.trajectory_record(
        config,
        entry=entry,
        sample_index=3,
        trajectory={"status": "SUCCEEDED", "reward": 1.0, "steps": []},
        elapsed_secs=2.0,
    )
    record.update({
        "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
        "config_fingerprint": artifacts.legacy_v5_fingerprint(config),
        "run_tag": artifacts.legacy_v5_run_tag(config),
    })
    record.pop("resume_tag")
    record.pop("harness_commit")

    snapshot = Path(self.temporary.name) / "imports" / "old-run"
    trajectory = snapshot / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    trajectory.write_bytes(payload)
    artifacts.seal_legacy_v5_snapshot(
        snapshot, config=config, allowed_task_keys=["img-a"]
    )

    output = Path(self.temporary.name) / "resume" / "outputs"
    validation = artifacts.validate_legacy_v5_snapshot_contract(
        snapshot, config=config, allowed_task_keys={"img-a"}
    )
    with self.assertRaisesRegex(ValueError, "changed after pre-lease"):
      artifacts.import_legacy_v5_snapshot(
          snapshot,
          output,
          config=config,
          allowed_task_keys={"img-a"},
          validated_snapshot_manifest_sha256="0" * 64,
      )
    first = artifacts.import_legacy_v5_snapshot(
        snapshot,
        output,
        config=config,
        allowed_task_keys={"img-a"},
        validated_snapshot_manifest_sha256=validation[
            "snapshot_manifest_sha256"
        ],
    )
    second = artifacts.import_legacy_v5_snapshot(
        snapshot, output, config=config, allowed_task_keys={"img-a"}
    )
    self.assertEqual(first["outputs"], second["outputs"])
    self.assertEqual(first["records"], 1)
    output_path = first["outputs"][0]["path"]
    self.assertTrue(Path(output_path).name.startswith(config.run_tag))
    loaded = artifacts.load_records(
        [output_path], config=config, allowed_task_keys={"img-a"}
    )
    self.assertEqual(len(loaded), 1)
    self.assertEqual(loaded[0]["schema"], artifacts.TRAJECTORY_SCHEMA)
    self.assertEqual(loaded[0]["resume_tag"], "adopt-old-run")
    self.assertEqual(loaded[0]["harness_commit"], "6" * 40)
    self.assertEqual(
        loaded[0]["imported_from"]["legacy_config_fingerprint"],
        artifacts.legacy_v5_fingerprint(config),
    )
    receipt = json.loads(Path(first["receipt_path"]).read_text())
    self.assertEqual(receipt["source_commit"], "5" * 40)
    self.assertEqual(receipt["harness_commit"], "6" * 40)

  def test_legacy_import_rejects_live_or_contract_drifted_snapshot(self):
    config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="reject-old-run",
    )
    entry = {"docker_image": "img-a"}
    record = artifacts.trajectory_record(
        config,
        entry=entry,
        sample_index=0,
        trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    record.update({
        "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
        "config_fingerprint": artifacts.legacy_v5_fingerprint(config),
        "run_tag": artifacts.legacy_v5_run_tag(config),
    })
    record.pop("resume_tag")
    record.pop("harness_commit")
    snapshot = Path(self.temporary.name) / "imports" / "old-run-bad"
    trajectory = snapshot / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = (json.dumps(record) + "\n").encode()
    trajectory.write_bytes(payload)
    artifacts.seal_legacy_v5_snapshot(
        snapshot, config=config, allowed_task_keys=["img-a"]
    )
    trajectory.write_bytes(payload + b" ")
    with self.assertRaisesRegex(ValueError, "digest mismatch"):
      artifacts.import_legacy_v5_snapshot(
          snapshot,
          Path(self.temporary.name) / "resume-bad" / "outputs",
          config=config,
          allowed_task_keys={"img-a"},
      )
    trajectory.write_bytes(payload)
    validation = artifacts.validate_legacy_v5_snapshot_contract(
        snapshot, config=config, allowed_task_keys={"img-a"}
    )
    self.assertEqual(validation["schema"], artifacts.LEGACY_TRAJECTORY_SCHEMA)
    self.assertEqual(validation["sampled_by"], f"stock@{'5' * 40}")
    later_drift = dict(record)
    later_drift["sampled_by"] = f"stock@{'4' * 40}"
    two_record_payload = payload + (
        json.dumps(later_drift, sort_keys=True) + "\n"
    ).encode()
    trajectory.write_bytes(two_record_payload)
    with self.assertRaisesRegex(ValueError, "digest mismatch"):
      artifacts.validate_legacy_v5_snapshot_contract(
          snapshot, config=config, allowed_task_keys={"img-a"}
      )
    trajectory.write_bytes(payload)
    drifted = dataclasses.replace(config, source_commit="4" * 40)
    with self.assertRaisesRegex(ValueError, "legacy source contract mismatch"):
      artifacts.import_legacy_v5_snapshot(
          snapshot,
          Path(self.temporary.name) / "resume-drift" / "outputs",
          config=drifted,
          allowed_task_keys={"img-a"},
      )
    (snapshot / "resume_contract.json").write_text("{}\n", encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "must not contain resume_contract"):
      artifacts.validate_legacy_v5_snapshot_contract(
          snapshot, config=config, allowed_task_keys={"img-a"}
      )

  def test_legacy_seal_accepts_path_drift_but_rejects_mixed_cohort(self):
    destination = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="sealed-path-drift",
    )
    historical = dataclasses.replace(
        destination,
        model_path="/historical/model/path",
        whitelist_path="/historical/whitelist.jsonl",
        client_image="historical.invalid/tunix@sha256:" + "8" * 64,
    )
    records = []
    for sample_index in (0, 1):
      record = artifacts.trajectory_record(
          historical,
          entry={"docker_image": "img-a"},
          sample_index=sample_index,
          trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
          elapsed_secs=1.0,
      )
      record.update({
          "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
          "config_fingerprint": artifacts.legacy_v5_fingerprint(historical),
          "run_tag": artifacts.legacy_v5_run_tag(historical),
      })
      record.pop("resume_tag")
      record.pop("harness_commit")
      records.append(record)

    accepted = Path(self.temporary.name) / "imports" / "path-drift"
    trajectory = accepted / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    validation = artifacts.seal_legacy_v5_snapshot(
        accepted, config=destination, allowed_task_keys=["img-a"]
    )
    self.assertEqual(validation["records"], 2)

    mixed = Path(self.temporary.name) / "imports" / "mixed-cohort"
    mixed_trajectory = mixed / "trajectories" / "wave.jsonl"
    mixed_trajectory.parent.mkdir(parents=True)
    drifted = dict(records[1])
    drifted["config_fingerprint"] = "9" * 64
    drifted["run_tag"] = "q4i16k-n16-64-" + "9" * 16
    mixed_trajectory.write_text(
        json.dumps(records[0]) + "\n" + json.dumps(drifted) + "\n",
        encoding="utf-8",
    )
    with self.assertRaisesRegex(ValueError, "mixes source cohorts"):
      artifacts.seal_legacy_v5_snapshot(
          mixed, config=destination, allowed_task_keys=["img-a"]
      )

  def test_legacy_import_rejects_manifest_without_source_contract(self):
    config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="missing-source-contract",
    )
    record = artifacts.trajectory_record(
        config,
        entry={"docker_image": "img-a"},
        sample_index=0,
        trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    record.update({
        "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
        "config_fingerprint": artifacts.legacy_v5_fingerprint(config),
        "run_tag": artifacts.legacy_v5_run_tag(config),
    })
    record.pop("resume_tag")
    record.pop("harness_commit")
    snapshot = Path(self.temporary.name) / "imports" / "unsealed"
    trajectory = snapshot / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = (json.dumps(record) + "\n").encode()
    trajectory.write_bytes(payload)
    (snapshot / "SHA256SUMS").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  trajectories/wave.jsonl\n",
        encoding="utf-8",
    )
    with self.assertRaisesRegex(ValueError, "legacy_source_contract"):
      artifacts.validate_legacy_v5_snapshot_contract(
          snapshot, config=config, allowed_task_keys=["img-a"]
      )

  def test_legacy_import_preserves_per_logical_shard_fingerprints(self):
    config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="multi-logical-import",
    )
    keys = [f"img-{index:02d}" for index in range(33)]
    legacy_records = []
    for key_index in (0, 32):
      logical_index = key_index // config.logical_tasks
      logical_config = dataclasses.replace(config, shard_index=logical_index)
      record = artifacts.trajectory_record(
          logical_config,
          entry={"docker_image": keys[key_index]},
          sample_index=0,
          trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
          elapsed_secs=1.0,
      )
      record.update({
          "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
          "config_fingerprint": artifacts.legacy_v5_fingerprint(logical_config),
          "run_tag": artifacts.legacy_v5_run_tag(logical_config),
      })
      record.pop("resume_tag")
      record.pop("harness_commit")
      legacy_records.append(record)

    snapshot = Path(self.temporary.name) / "imports" / "multi-old-run"
    trajectory = snapshot / "trajectories" / "waves.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode()
        for record in legacy_records
    )
    trajectory.write_bytes(payload)
    artifacts.seal_legacy_v5_snapshot(
        snapshot, config=config, allowed_task_keys=keys
    )
    receipt = artifacts.import_legacy_v5_snapshot(
        snapshot,
        Path(self.temporary.name) / "multi-resume" / "outputs",
        config=config,
        allowed_task_keys=keys,
    )
    self.assertEqual(receipt["records"], 2)
    self.assertEqual(
        [item["logical_shard_index"] for item in receipt["outputs"]], [0, 1]
    )
    for item in receipt["outputs"]:
      logical_config = dataclasses.replace(
          config, shard_index=item["logical_shard_index"]
      )
      loaded = artifacts.load_records(
          [item["path"]], config=logical_config, allowed_task_keys=keys
      )
      self.assertEqual(len(loaded), 1)

  def test_frozen_v6_snapshot_migrates_to_fresh_harness_and_resume_tag(self):
    old_config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="old-v6-run",
    )
    new_config = dataclasses.replace(
        old_config,
        harness_commit="7" * 40,
        resume_tag="new-v6-run",
    )
    entry = {"docker_image": "img-a", "instance_id": "task-a"}
    records = [
        artifacts.trajectory_record(
            old_config,
            entry=entry,
            sample_index=3,
            attempt_index=0,
            trajectory={"status": "FAILED", "reward": 0.0, "steps": []},
            elapsed_secs=1.0,
        ),
        artifacts.trajectory_record(
            old_config,
            entry=entry,
            sample_index=3,
            attempt_index=1,
            trajectory={"status": "SUCCEEDED", "reward": 1.0, "steps": []},
            elapsed_secs=2.0,
        ),
    ]
    source_root = Path(self.temporary.name) / "source-v6"
    artifacts.ensure_resume_contract(source_root, config=old_config)
    snapshot = Path(self.temporary.name) / "imports" / "sealed-old-v6"
    trajectory = snapshot / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )
    trajectory.write_bytes(payload)
    contract_payload = (source_root / "resume_contract.json").read_bytes()
    (snapshot / "resume_contract.json").write_bytes(contract_payload)
    (snapshot / "SHA256SUMS").write_text(
        f"{hashlib.sha256(contract_payload).hexdigest()}  resume_contract.json\n"
        f"{hashlib.sha256(payload).hexdigest()}  trajectories/wave.jsonl\n",
        encoding="utf-8",
    )

    output = Path(self.temporary.name) / "new-v6" / "outputs"
    first = artifacts.import_frozen_v6_snapshot(
        snapshot, output, config=new_config, allowed_task_keys={"img-a"}
    )
    second = artifacts.import_frozen_v6_snapshot(
        snapshot, output, config=new_config, allowed_task_keys={"img-a"}
    )
    self.assertEqual(first["outputs"], second["outputs"])
    self.assertEqual(first["records"], 2)
    self.assertEqual(first["valid_records"], 1)
    self.assertEqual(first["source_resume_tag"], "old-v6-run")
    loaded = artifacts.load_records(
        [first["outputs"][0]["path"]],
        config=new_config,
        allowed_task_keys={"img-a"},
    )
    self.assertEqual([item["attempt_index"] for item in loaded], [0, 1])
    self.assertEqual(loaded[0]["sampled_by"], f"stock@{'5' * 40}")
    self.assertEqual(loaded[0]["harness_commit"], "7" * 40)
    self.assertEqual(
        loaded[0]["migrated_from"]["source_harness_commit"], "6" * 40
    )
    self.assertEqual(
        loaded[0]["migrated_from"]["source_resume_tag"], "old-v6-run"
    )

    with self.assertRaisesRegex(ValueError, "fresh resume tag"):
      artifacts.import_frozen_v6_snapshot(
          snapshot,
          Path(self.temporary.name) / "same-v6" / "outputs",
          config=old_config,
          allowed_task_keys={"img-a"},
      )

  def test_frozen_v6_snapshot_rejects_sampling_contract_drift(self):
    old_config = dataclasses.replace(
        self.config,
        source_commit="5" * 40,
        harness_commit="6" * 40,
        resume_tag="old-v6-drift",
    )
    source_root = Path(self.temporary.name) / "source-v6-drift"
    artifacts.ensure_resume_contract(source_root, config=old_config)
    record = artifacts.trajectory_record(
        old_config,
        entry={"docker_image": "img-a"},
        sample_index=0,
        trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    snapshot = Path(self.temporary.name) / "imports" / "sealed-v6-drift"
    trajectory = snapshot / "trajectories" / "wave.jsonl"
    trajectory.parent.mkdir(parents=True)
    payload = (json.dumps(record) + "\n").encode()
    trajectory.write_bytes(payload)
    contract_payload = (source_root / "resume_contract.json").read_bytes()
    (snapshot / "resume_contract.json").write_bytes(contract_payload)
    (snapshot / "SHA256SUMS").write_text(
        f"{hashlib.sha256(contract_payload).hexdigest()}  resume_contract.json\n"
        f"{hashlib.sha256(payload).hexdigest()}  trajectories/wave.jsonl\n",
        encoding="utf-8",
    )
    drifted = dataclasses.replace(
        old_config,
        source_commit="4" * 40,
        harness_commit="7" * 40,
        resume_tag="new-v6-drift",
    )
    with self.assertRaisesRegex(ValueError, "sampling contract drift"):
      artifacts.import_frozen_v6_snapshot(
          snapshot,
          Path(self.temporary.name) / "new-v6-drift" / "outputs",
          config=drifted,
          allowed_task_keys={"img-a"},
      )

  def test_exact_n_classification_and_q32_hard_tier(self):
    entries = [
        {"docker_image": "partial"},
        {"docker_image": "all-fail"},
        {"docker_image": "all-pass"},
        {"docker_image": "broken"},
        {"docker_image": "missing"},
    ]
    records = []
    for entry in entries[:-1]:
      for sample_index in range(16):
        key = entry["docker_image"]
        status = "FAILED" if key == "broken" else "SUCCEEDED"
        reward = 0.0
        if key == "all-pass" or (key == "partial" and sample_index < 8):
          reward = 1.0
        records.append(artifacts.trajectory_record(
            self.config,
            entry=entry,
            sample_index=sample_index,
            trajectory={"status": status, "reward": reward, "steps": []},
            elapsed_secs=1.0,
        ))
    reports = artifacts.aggregate_tasks(entries, records, config=self.config)
    categories = {item["task_key"]: item["category"] for item in reports}
    self.assertEqual(categories, {
        "partial": "partial",
        "all-fail": "all_fail",
        "all-pass": "all_pass",
        "broken": "broken",
        "missing": "incomplete",
    })
    summary = artifacts.write_reports(
        Path(self.temporary.name) / "reports", reports, config=self.config
    )
    self.assertEqual(summary["trajectory_mode"], "reward_only_no_logprobs")
    self.assertEqual(summary["sampled_by"], "stock@" + "6" * 40)
    self.assertTrue(all(
        item["trajectory_mode"] == "reward_only_no_logprobs"
        and item["sampled_by"] == "stock@" + "6" * 40
        for item in reports
    ))
    q32_path = Path(summary["paths"]["q32_candidates"])
    q32_keys = {
        json.loads(line)["task_key"]
        for line in q32_path.read_text(encoding="utf-8").splitlines()
    }
    self.assertEqual(q32_keys, {"partial", "all-fail"})
    self.assertEqual(summary["category_counts"]["incomplete"], 1)

    census = artifacts.write_census(
        Path(self.temporary.name) / "census",
        reports,
        artifacts.deferred_samples(entries, records, config=self.config),
        config=self.config,
        launch_id="census-test",
    )
    self.assertEqual(census["scheduled_identities"], 80)
    self.assertEqual(census["attempted_identities"], 64)
    self.assertEqual(census["valid_identities"], 48)
    self.assertEqual(census["deferred_invalid_identities"], 16)
    self.assertEqual(census["unattempted_identities"], 16)
    self.assertEqual(census["q4_learnable_provisional"], 1)
    self.assertFalse(census["first_pass_complete"])
    self.assertFalse(census["strict_campaign_complete"])
    self.assertTrue(Path(census["summary_path"]).is_file())
    self.assertTrue(Path(census["paths"]["mixed_complete"]).is_file())

  def test_identical_report_writers_converge_but_drift_is_rejected(self):
    entry = {"docker_image": "partial"}
    records = [
        artifacts.trajectory_record(
            self.config,
            entry=entry,
            sample_index=sample_index,
            trajectory={
                "status": "SUCCEEDED",
                "reward": float(sample_index == 0),
                "steps": [],
            },
            elapsed_secs=1.0,
        )
        for sample_index in range(self.config.n_sample)
    ]
    reports = artifacts.aggregate_tasks(
        [entry], records, config=self.config
    )
    root = Path(self.temporary.name) / "concurrent-reports"
    first = artifacts.write_reports(root, reports, config=self.config)
    second = artifacts.write_reports(root, reports, config=self.config)
    self.assertEqual(first["summary_sha256"], second["summary_sha256"])
    changed = [dict(reports[0], category="all_fail", k=0)]
    with self.assertRaisesRegex(ValueError, "differs from exact payload"):
      artifacts.write_reports(root, changed, config=self.config)

  def test_campaign_finalizer_requires_every_exact_n_logical_summary(self):
    summaries = []
    for shard_index, keys in enumerate((("partial", "all-fail"), ("all-pass",))):
      config = dataclasses.replace(self.config, shard_index=shard_index)
      entries = [{"docker_image": key} for key in keys]
      records = []
      for entry in entries:
        for sample_index in range(config.n_sample):
          reward = float(
              entry["docker_image"] == "all-pass"
              or (
                  entry["docker_image"] == "partial" and sample_index == 0
              )
          )
          records.append(artifacts.trajectory_record(
              config,
              entry=entry,
              sample_index=sample_index,
              trajectory={
                  "status": "SUCCEEDED",
                  "reward": reward,
                  "steps": [],
              },
              elapsed_secs=1.0,
          ))
      reports = artifacts.aggregate_tasks(entries, records, config=config)
      summaries.append(artifacts.write_reports(
          Path(self.temporary.name) / f"logical-{shard_index}",
          reports,
          config=config,
      ))

    with self.assertRaisesRegex(ValueError, "every logical summary"):
      artifacts._finalize_campaign(  # pylint: disable=protected-access
          [summaries[0]["summary_path"]],
          Path(self.temporary.name) / "campaign",
          expected_tasks=3,
          expected_logical_shards=2,
          tasks_per_logical_shard=2,
      )
    result = artifacts._finalize_campaign(  # pylint: disable=protected-access
        [summary["summary_path"] for summary in reversed(summaries)],
        Path(self.temporary.name) / "campaign",
        expected_tasks=3,
        expected_logical_shards=2,
        tasks_per_logical_shard=2,
    )
    self.assertEqual(result["tasks"], 3)
    self.assertEqual(result["valid_trajectories"], 48)
    self.assertEqual(result["logical_shards"], 2)
    self.assertEqual(result["category_counts"], {
        "all_fail": 1,
        "all_pass": 1,
        "partial": 1,
    })
    q32_keys = {
        json.loads(line)["task_key"]
        for line in Path(result["paths"]["q32_candidates"])
        .read_text(encoding="utf-8")
        .splitlines()
    }
    self.assertEqual(q32_keys, {"partial", "all-fail"})


if __name__ == "__main__":
  unittest.main()
