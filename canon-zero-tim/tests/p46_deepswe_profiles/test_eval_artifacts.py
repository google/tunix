"""CPU contracts for resumable P46 DeepSWE evaluation artifacts."""

from __future__ import annotations

import dataclasses
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
        client_image="example.invalid/tunix@sha256:" + "7" * 64,
        topology="64",
    )

  def tearDown(self):
    self.temporary.cleanup()

  def test_signed_config_and_fingerprint(self):
    self.config.validate()
    self.assertEqual(len(self.config.fingerprint), 64)
    self.assertIn(self.config.fingerprint[:16], self.config.run_tag)
    other = dataclasses.replace(self.config, topology="256")
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
      dataclasses.replace(observer, topology="256").validate()

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
    self.assertEqual(record["sampled_by"], "stock@" + "6" * 40)

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

  def test_resume_rejects_duplicates_and_fingerprint_drift(self):
    root = Path(self.temporary.name)
    path = root / "raw.jsonl"
    entry = {"docker_image": "img-a"}
    record = artifacts.trajectory_record(
        self.config,
        entry=entry,
        sample_index=0,
        trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
        elapsed_secs=1.0,
    )
    artifacts.append_record(path, record)
    loaded = artifacts.load_records(
        [path], config=self.config, allowed_task_keys={"img-a"}
    )
    remaining = artifacts.remaining_samples([entry], loaded, config=self.config)
    self.assertEqual(len(remaining), 15)
    artifacts.append_record(path, record)
    with self.assertRaisesRegex(ValueError, "duplicate"):
      artifacts.load_records(
          [path], config=self.config, allowed_task_keys={"img-a"}
      )

    drift = root / "drift.jsonl"
    changed = dict(record)
    changed["config_fingerprint"] = "0" * 64
    artifacts.append_record(drift, changed)
    with self.assertRaisesRegex(ValueError, "fingerprint"):
      artifacts.load_records(
          [drift], config=self.config, allowed_task_keys={"img-a"}
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


if __name__ == "__main__":
  unittest.main()
