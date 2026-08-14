"""Static L1 contracts for P46 stock reward-only evaluation."""

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "deepswe"))

import eval_deepswe  # pylint: disable=wrong-import-position
import deepswe_eval_artifacts as artifacts  # pylint: disable=wrong-import-position


class RewardOnlyContractTest(unittest.TestCase):

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
          client_image="example.invalid/tunix@sha256:" + "7" * 64,
          topology="64",
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
