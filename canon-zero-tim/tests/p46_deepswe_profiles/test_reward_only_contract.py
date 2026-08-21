"""Static L1 contracts for P46 stock reward-only evaluation."""

import asyncio
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "deepswe"))

import eval_deepswe  # pylint: disable=wrong-import-position
import deepswe_eval_artifacts as artifacts  # pylint: disable=wrong-import-position
import r2egym_runtime_patch  # pylint: disable=wrong-import-position


class RewardOnlyContractTest(unittest.TestCase):

  def test_imported_v5_identities_are_skipped_by_census_resume(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      output_dir = root / "fresh-resume" / "outputs"
      entries = [
          {"docker_image": f"image-{index}"} for index in range(4)
      ]
      config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2" * 64,
          whitelist_rows=1851,
          source_commit="5" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="128",
          resume_tag="fresh-resume",
      )
      legacy_records = []
      for entry in entries:
        for sample_index in range(config.n_sample):
          record = artifacts.trajectory_record(
              config,
              entry=entry,
              sample_index=sample_index,
              trajectory={
                  "status": "SUCCEEDED",
                  "reward": 0.0,
                  "steps": [],
              },
              elapsed_secs=1.0,
          )
          record.update({
              "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
              "config_fingerprint": artifacts.legacy_v5_fingerprint(config),
              "run_tag": artifacts.legacy_v5_run_tag(config),
          })
          record.pop("resume_tag")
          record.pop("harness_commit")
          legacy_records.append(record)
      legacy_records = legacy_records[:17]
      snapshot = output_dir.parent / "imports" / "old-v5"
      trajectory = snapshot / "trajectories" / "wave.jsonl"
      trajectory.parent.mkdir(parents=True)
      payload = b"".join(
          (json.dumps(record, sort_keys=True) + "\n").encode()
          for record in legacy_records
      )
      trajectory.write_bytes(payload)
      artifacts.seal_legacy_v5_snapshot(
          snapshot,
          config=config,
          allowed_task_keys=(artifacts.task_key(entry) for entry in entries),
      )
      validation = artifacts.validate_legacy_v5_snapshot_contract(
          snapshot,
          config=config,
          allowed_task_keys=(artifacts.task_key(entry) for entry in entries),
      )
      receipt = artifacts.import_legacy_v5_snapshot(
          snapshot,
          output_dir,
          config=config,
          allowed_task_keys=(artifacts.task_key(entry) for entry in entries),
          validated_snapshot_manifest_sha256=validation[
              "snapshot_manifest_sha256"
          ],
      )
      self.assertEqual(receipt["records"], 17)
      wave_sizes = []

      async def fake_run(
          logical_config,
          unused_entries,
          pending,
          output_path,
          *,
          runtime,
          timeout_secs,
      ):
        del unused_entries, runtime
        self.assertEqual(timeout_secs, 3600)
        wave_sizes.append(len(pending))
        for entry, sample_index, attempt_index in pending:
          artifacts.append_record(
              output_path,
              artifacts.trajectory_record(
                  logical_config,
                  entry=entry,
                  sample_index=sample_index,
                  attempt_index=attempt_index,
                  trajectory={
                      "status": "SUCCEEDED",
                      "reward": 0.0,
                      "steps": [],
                  },
                  elapsed_secs=1.0,
              ),
          )
        return len(pending), False

      with (
          mock.patch.object(eval_deepswe, "_Runtime", return_value=object()),
          mock.patch.object(
              eval_deepswe, "_run_evaluation", side_effect=fake_run
          ),
      ):
        result = asyncio.run(eval_deepswe._run_full_campaign(
            config,
            entries,
            output_dir,
            first_pass_census=True,
            launch_id="resume-census",
        ))
      self.assertEqual(result, 0)
      self.assertEqual(wave_sizes, [47])
      loaded = artifacts.load_records(
          list((output_dir / "trajectories").glob("*.jsonl")),
          config=config,
          allowed_task_keys=(artifacts.task_key(entry) for entry in entries),
      )
      self.assertEqual(len(loaded), 64)

  def test_wrong_legacy_sampler_fails_before_target_resume_tag_is_claimed(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      output_dir = root / "fresh-resume" / "outputs"
      entry = {"docker_image": "image-a"}
      source_config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2" * 64,
          whitelist_rows=1851,
          source_commit="5" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="128",
          resume_tag="source-resume",
      )
      destination_config = dataclasses.replace(
          source_config,
          source_commit="6" * 40,
          resume_tag="fresh-resume",
      )
      record = artifacts.trajectory_record(
          source_config,
          entry=entry,
          sample_index=0,
          trajectory={"status": "SUCCEEDED", "reward": 0.0, "steps": []},
          elapsed_secs=1.0,
      )
      record.update({
          "schema": artifacts.LEGACY_TRAJECTORY_SCHEMA,
          "config_fingerprint": artifacts.legacy_v5_fingerprint(source_config),
          "run_tag": artifacts.legacy_v5_run_tag(source_config),
      })
      record.pop("resume_tag")
      record.pop("harness_commit")
      snapshot = output_dir.parent / "imports" / "source-v5"
      trajectory = snapshot / "trajectories" / "wave.jsonl"
      trajectory.parent.mkdir(parents=True)
      payload = (json.dumps(record, sort_keys=True) + "\n").encode()
      trajectory.write_bytes(payload)
      artifacts.seal_legacy_v5_snapshot(
          snapshot,
          config=source_config,
          allowed_task_keys=[artifacts.task_key(entry)],
      )

      environment = {
          "CANON_P46_FULL_CAMPAIGN": "1",
          "CANON_P46_CENSUS_FIRST_PASS": "1",
          "CANON_P46_ONEHOST_PROBE": "0",
          "CANON_P46_PARITY_CANARY": "0",
          "CANON_P46_EVALUATION_MODE": "reward_only",
          "CANON_P46_LEGACY_IMPORT_ID": "source-v5",
          "CANON_P46_FROZEN_V6_IMPORT_ID": "",
          "CANON_RUN_ID": "wrong-sampler-test",
      }
      with (
          mock.patch.dict(os.environ, environment, clear=False),
          mock.patch.object(
              eval_deepswe,
              "_build_config",
              return_value=(destination_config, 0, output_dir),
          ),
          mock.patch.object(
              eval_deepswe, "_load_clean_entries", return_value=[entry]
          ),
      ):
        with self.assertRaisesRegex(
            ValueError, "legacy source contract mismatch"
        ):
          eval_deepswe.main()
      self.assertFalse((output_dir / "resume_contract.json").exists())

  def test_vllm_false_path_is_none_none_and_skips_extraction(self):
    sampler = (ROOT / "tunix/generate/vllm_sampler.py").read_text(
        encoding="utf-8"
    )
    self.assertIn("sampling_params.logprobs = None", sampler)
    self.assertIn("sampling_params.prompt_logprobs = None", sampler)
    self.assertNotIn("sampling_params.logprobs = 0\n", sampler)
    conditional = "if self.config.return_logprobs:\n          logprobs = utils.get_logprobs_from_vllm_output"
    self.assertIn(conditional, sampler)

  def test_eval_uses_engine_seed_not_unsupported_request_seed(self):
    evaluator = (ROOT / "examples/deepswe/eval_deepswe.py").read_text(
        encoding="utf-8"
    )
    self.assertIn('"seed": config.seed_base', evaluator)
    self.assertNotIn('seed=env.extra_kwargs["sample_seed"]', evaluator)
    self.assertIn("sample_nonce", evaluator)
    self.assertIn(
        "left_padded_prompt_tokens=output.padded_prompt_tokens", evaluator
    )
    self.assertIn("generation_steps = min(generation_steps, 256)", evaluator)
    self.assertIn("if timed_out or physical_pending:", evaluator)
    self.assertIn("P46_EVAL_PHYSICAL_INCOMPLETE", evaluator)
    self.assertIn("pending_valid_samples=", evaluator)
    self.assertIn("P46_EVAL_CAMPAIGN_PASS", evaluator)
    self.assertIn("runtime_reused=1", evaluator)
    self.assertEqual(evaluator.count("runtime = _Runtime(base_config)"), 1)
    self.assertIn(
        "SWEAgent(action_compat_mode=config.action_compat_mode)", evaluator
    )
    finalizer = (ROOT / "examples/deepswe/finalize_deepswe_eval.py").read_text()
    self.assertIn("P46_EVAL_CAMPAIGN_PASS", finalizer)

  def test_resume_cleanup_deletes_only_same_tag_sandboxes(self):
    class FakeCore:

      def __init__(self):
        self.list_calls = []
        self.deleted = []

      def list_namespaced_pod(self, **kwargs):
        self.list_calls.append(kwargs)
        items = (
            [
                types.SimpleNamespace(
                    metadata=types.SimpleNamespace(name="sandbox-a")
                ),
                types.SimpleNamespace(
                    metadata=types.SimpleNamespace(name="sandbox-b")
                ),
            ]
            if len(self.list_calls) == 1
            else []
        )
        return types.SimpleNamespace(items=items)

      def delete_namespaced_pod(self, **kwargs):
        self.deleted.append(kwargs)

    core = FakeCore()
    deleted = r2egym_runtime_patch._cleanup_orphaned_kubernetes_pods(  # pylint: disable=protected-access
        core,
        namespace="default",
        resume_tag="wash-q4-001",
        api_exception_type=RuntimeError,
    )
    self.assertEqual(deleted, 2)
    self.assertEqual(
        core.list_calls[0]["label_selector"],
        "app.kubernetes.io/managed-by=tunix-deepswe,"
        "canon.zero-tim/resume-tag=wash-q4-001",
    )
    self.assertEqual(
        [item["name"] for item in core.deleted],
        ["sandbox-a", "sandbox-b"],
    )

  def test_q32_agent_path_keeps_strict_parser_default(self):
    agent = (ROOT / "examples/deepswe/swe_agent.py").read_text(
        encoding="utf-8"
    )
    trainer = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text(
        encoding="utf-8"
    )
    self.assertIn("action_compat_mode: str = STRICT_XML_MODE", agent)
    self.assertIn("agent_kwargs={}", trainer)
    self.assertNotIn("q4_r2egym_xml_v2", trainer)

  def test_eval_adds_only_the_swe_env_batch_dimension(self):
    row = {
        "docker_image": "example/image",
        "modified_files": ["a.py", "b.py"],
        "modified_entity_summaries": ["one", "two", "three"],
    }
    batched = eval_deepswe._batch_entry_for_swe_env(row)  # pylint: disable=protected-access
    self.assertEqual(batched["docker_image"], ["example/image"])
    self.assertEqual(batched["modified_files"], [["a.py", "b.py"]])
    self.assertEqual(
        batched["modified_entity_summaries"],
        [["one", "two", "three"]],
    )
    # The evaluator must not mutate or flatten the clean source row.
    self.assertEqual(row["modified_files"], ["a.py", "b.py"])

  def test_full_campaign_reuses_one_runtime_for_all_29616_identities(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7",
          whitelist_rows=1851,
          source_commit="6" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="64",
          resume_tag="full-campaign-test",
      )
      entries = [
          {"docker_image": f"image-{index:04d}"}
          for index in range(1851)
      ]
      records: dict[int, list[dict[str, object]]] = {}
      wave_sizes = []
      runtime_instances = []

      def fake_runtime(_):
        runtime = object()
        runtime_instances.append(runtime)
        return runtime

      def fake_load(logical_config, logical_entries, unused_dir):
        del logical_entries, unused_dir
        return list(records.get(logical_config.shard_index, []))

      async def fake_run(
          logical_config,
          unused_entries,
          pending,
          unused_path,
          *,
          runtime,
          timeout_secs,
      ):
        del unused_entries, unused_path
        self.assertIs(runtime, runtime_instances[0])
        self.assertGreater(timeout_secs, 0)
        self.assertLessEqual(timeout_secs, 3600)
        wave_sizes.append(len(pending))
        target = records.setdefault(logical_config.shard_index, [])
        for entry, sample_index, attempt_index in pending:
          target.append({
              "task_key": entry["docker_image"],
              "sample_index": sample_index,
              "attempt_index": attempt_index,
              "valid": True,
              "solved": False,
              "status": "SUCCEEDED",
          })
        return len(pending), False

      def fake_reports(unused_dir, reports, *, config):
        del unused_dir
        return {
            "summary_path": str(root / f"l{config.shard_index}.json"),
            "tasks": len(reports),
            "valid_trajectories": len(reports) * config.n_sample,
        }

      with (
          mock.patch.object(eval_deepswe, "_Runtime", side_effect=fake_runtime),
          mock.patch.object(
              eval_deepswe, "_load_logical_records", side_effect=fake_load
          ),
          mock.patch.object(
              eval_deepswe, "_run_evaluation", side_effect=fake_run
          ),
          mock.patch.object(
              eval_deepswe, "write_reports", side_effect=fake_reports
          ),
          mock.patch.object(
              eval_deepswe,
              "finalize_campaign",
              return_value={
                  "tasks": 1851,
                  "n_sample": 16,
                  "valid_trajectories": 29616,
                  "logical_shards": 58,
                  "summary_sha256": "8" * 64,
              },
          ),
      ):
        result = asyncio.run(
            eval_deepswe._run_full_campaign(config, entries, root / "out")
        )
      self.assertEqual(result, 0)
      self.assertEqual(len(runtime_instances), 1)
      self.assertEqual(len(wave_sizes), 463)
      self.assertEqual(sum(wave_sizes), 29616)
      self.assertEqual(wave_sizes[-1], 48)
      self.assertTrue(all(size == 64 for size in wave_sizes[:-1]))

  def test_full_campaign_restart_picks_up_only_missing_wave_identities(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7",
          whitelist_rows=1851,
          source_commit="6" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="64",
          resume_tag="restart-wave-test",
      )
      entries = [{"docker_image": f"image-{index}"} for index in range(4)]
      durable = []
      wave_sizes = []
      runtime_instances = []

      def fake_runtime(_):
        runtime = object()
        runtime_instances.append(runtime)
        return runtime

      def fake_load(unused_config, unused_entries, unused_dir):
        del unused_config, unused_entries, unused_dir
        return list(durable)

      async def fake_run(
          unused_config,
          unused_entries,
          pending,
          unused_path,
          *,
          runtime,
          timeout_secs,
      ):
        del unused_config, unused_entries, unused_path, runtime, timeout_secs
        wave_sizes.append(len(pending))
        selected = pending[:17] if len(wave_sizes) == 1 else pending
        for entry, sample_index, attempt_index in selected:
          durable.append({
              "task_key": entry["docker_image"],
              "sample_index": sample_index,
              "attempt_index": attempt_index,
              "valid": True,
              "solved": False,
              "status": "SUCCEEDED",
          })
        return len(selected), len(wave_sizes) == 1

      def fake_aggregate(logical_entries, unused_records, *, config):
        del unused_records, config
        return [
            {"category": "all_fail", "task_key": entry["docker_image"]}
            for entry in logical_entries
        ]

      with (
          mock.patch.object(eval_deepswe, "_Runtime", side_effect=fake_runtime),
          mock.patch.object(
              eval_deepswe, "_load_logical_records", side_effect=fake_load
          ),
          mock.patch.object(
              eval_deepswe, "_run_evaluation", side_effect=fake_run
          ),
          mock.patch.object(
              eval_deepswe, "aggregate_tasks", side_effect=fake_aggregate
          ),
          mock.patch.object(
              eval_deepswe,
              "write_reports",
              return_value={
                  "summary_path": str(root / "summary.json"),
                  "tasks": 4,
                  "valid_trajectories": 64,
              },
          ),
          mock.patch.object(
              eval_deepswe,
              "finalize_campaign",
              return_value={
                  "tasks": 1851,
                  "n_sample": 16,
                  "valid_trajectories": 29616,
                  "logical_shards": 58,
                  "summary_sha256": "8" * 64,
              },
          ),
      ):
        first = asyncio.run(
            eval_deepswe._run_full_campaign(config, entries, root / "out")
        )
        second = asyncio.run(
            eval_deepswe._run_full_campaign(config, entries, root / "out")
        )
      self.assertEqual(first, 2)
      self.assertEqual(second, 0)
      self.assertEqual(wave_sizes, [64, 47])
      self.assertEqual(len(durable), 64)
      self.assertEqual(len(runtime_instances), 2)

  def test_census_defers_invalid_identity_and_covers_later_work(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7",
          whitelist_rows=1851,
          source_commit="6" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="64",
          resume_tag="census-wave-test",
      )
      entries = [{"docker_image": f"image-{index}"} for index in range(4)]
      durable = []
      wave_sizes = []
      runtime_instances = []

      def fake_runtime(_):
        runtime = object()
        runtime_instances.append(runtime)
        return runtime

      def fake_load(unused_config, unused_entries, unused_dir):
        del unused_config, unused_entries, unused_dir
        return list(durable)

      async def fake_run(
          unused_config,
          unused_entries,
          pending,
          unused_path,
          *,
          runtime,
          timeout_secs,
      ):
        del unused_config, unused_entries, unused_path
        self.assertIs(runtime, runtime_instances[-1])
        self.assertEqual(timeout_secs, 3600)
        wave_sizes.append(len(pending))
        for entry, sample_index, attempt_index in pending:
          invalid = entry["docker_image"] == "image-0" and sample_index == 0
          durable.append({
              "task_key": entry["docker_image"],
              "sample_index": sample_index,
              "attempt_index": attempt_index,
              "valid": not invalid,
              "validity_reason": (
                  "retryable_runtime_failure"
                  if invalid
                  else "completed_under_signed_budget"
              ),
              "solved": False,
              "status": "FAILED" if invalid else "SUCCEEDED",
          })
        return len(pending), False

      with (
          mock.patch.object(eval_deepswe, "_Runtime", side_effect=fake_runtime),
          mock.patch.object(
              eval_deepswe, "_load_logical_records", side_effect=fake_load
          ),
          mock.patch.object(
              eval_deepswe, "_run_evaluation", side_effect=fake_run
          ),
      ):
        first = asyncio.run(eval_deepswe._run_full_campaign(
            config,
            entries,
            root / "out",
            first_pass_census=True,
            launch_id="census-a",
        ))
        second = asyncio.run(eval_deepswe._run_full_campaign(
            config,
            entries,
            root / "out",
            first_pass_census=True,
            launch_id="census-b",
        ))
      self.assertEqual(first, 0)
      self.assertEqual(second, 0)
      self.assertEqual(wave_sizes, [64])
      self.assertEqual(len(durable), 64)
      self.assertEqual(len(runtime_instances), 2)
      summaries = sorted((root / "out" / "census").glob("*.summary.json"))
      self.assertEqual(len(summaries), 2)
      latest = json.loads(summaries[-1].read_text(encoding="utf-8"))
      self.assertEqual(latest["attempted_identities"], 64)
      self.assertEqual(latest["valid_identities"], 63)
      self.assertEqual(latest["deferred_invalid_identities"], 1)
      self.assertEqual(latest["unattempted_identities"], 0)
      self.assertTrue(latest["first_pass_complete"])
      self.assertFalse(latest["strict_campaign_complete"])

  def test_census_continues_after_wave_timeout_then_resumes_unattempted(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      config = artifacts.EvalConfig(
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          model_path=str(root / "model"),
          dataset_name="R2E-Gym/R2E-Gym-Subset",
          dataset_revision="2e8108ff942f24fcb5686badfaf7f9a8808566d5",
          dataset_split="train",
          dataset_rows=4578,
          whitelist_path=str(root / "clean.jsonl"),
          whitelist_sha256="2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7",
          whitelist_rows=1851,
          source_commit="6" * 40,
          harness_commit="6" * 40,
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="64",
          resume_tag="census-timeout-test",
      )
      entries = [{"docker_image": f"image-{index}"} for index in range(8)]
      durable = []
      wave_sizes = []

      def fake_load(unused_config, unused_entries, unused_dir):
        del unused_config, unused_entries, unused_dir
        return list(durable)

      async def fake_run(
          unused_config,
          unused_entries,
          pending,
          unused_path,
          *,
          runtime,
          timeout_secs,
      ):
        del unused_config, unused_entries, unused_path, runtime, timeout_secs
        wave_sizes.append(len(pending))
        timed_out = len(wave_sizes) == 1
        selected = pending[:-1] if timed_out else pending
        for entry, sample_index, attempt_index in selected:
          durable.append({
              "task_key": entry["docker_image"],
              "sample_index": sample_index,
              "attempt_index": attempt_index,
              "valid": True,
              "validity_reason": "completed_under_signed_budget",
              "solved": False,
              "status": "SUCCEEDED",
          })
        return len(selected), timed_out

      with (
          mock.patch.object(eval_deepswe, "_Runtime", return_value=object()),
          mock.patch.object(
              eval_deepswe, "_load_logical_records", side_effect=fake_load
          ),
          mock.patch.object(
              eval_deepswe, "_run_evaluation", side_effect=fake_run
          ),
      ):
        first = asyncio.run(eval_deepswe._run_full_campaign(
            config,
            entries,
            root / "out",
            first_pass_census=True,
            launch_id="timeout-a",
        ))
        second = asyncio.run(eval_deepswe._run_full_campaign(
            config,
            entries,
            root / "out",
            first_pass_census=True,
            launch_id="timeout-b",
        ))
      self.assertEqual(first, 2)
      self.assertEqual(second, 0)
      self.assertEqual(wave_sizes, [64, 64, 1])
      self.assertEqual(len(durable), 128)

  def test_entrypoint_skips_only_canonical_overlay_not_lifecycle(self):
    entrypoint = (ROOT / "canon-zero-tim/cluster/entrypoint.sh").read_text(
        encoding="utf-8"
    )
    resolved = entrypoint.index('source "$CANON_STATE/env.sh"')
    branch = entrypoint.index(
        'if [ "${CANON_P46_EVALUATION:-0}" = "1" ]'
    )
    self.assertLess(resolved, branch)
    normal = entrypoint.index("else", branch)
    stock_block = entrypoint[branch:normal]
    self.assertIn("step 35_install_r2egym.sh", stock_block)
    self.assertNotIn("step 30_install_canon.sh", stock_block)
    self.assertNotIn("step 40_overlay_engine.sh", stock_block)
    self.assertNotIn("step 50_verify_overlay.sh", stock_block)
    self.assertIn("step 60_wait_workers.sh", entrypoint)
    self.assertIn("step 65_probe_devices.sh", entrypoint)
    self.assertIn("step 90_run.sh", entrypoint)

  def test_onehost_runner_is_isolated_and_keeps_real_r2e(self):
    runner = (
        ROOT
        / "canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh"
    ).read_text(encoding="utf-8")
    probe = (ROOT / "examples/deepswe/probe_reward_only_v5p.py").read_text(
        encoding="utf-8"
    )
    self.assertIn("CANON_P46_ONEHOST_PROBE=1", runner)
    self.assertIn("CANON_P46_EVALUATION_MODE=reward_only", runner)
    self.assertIn("R2EGYM_SHA", runner)
    self.assertIn("_run_evaluation", probe)
    self.assertNotIn("seed=20260813", probe)
    self.assertIn("_restore_engine_rng", probe)
    self.assertIn("cleanup_new_containers", probe)
    self.assertIn("NOT_RUN_REQUIRES_64_CHIP_PAIRED_N16", probe)


if __name__ == "__main__":
  unittest.main()
