"""Static L1 contracts for P46 stock reward-only evaluation."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "deepswe"))

import eval_deepswe  # pylint: disable=wrong-import-position


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
    finalizer = (ROOT / "examples/deepswe/finalize_deepswe_eval.py").read_text()
    self.assertIn("P46_EVAL_CAMPAIGN_PASS", finalizer)

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
