#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
ROLLOUT = ROOT / "tunix/rl/rollout/vllm_rollout.py"
FLAGS = ROOT / "canon-zero-tim/FLAGS.md"
LEARNER = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
RUNNER = ROOT / (
    "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts/"
    "run_p3_apc_onehost.sh"
)
BOUNDARY_RUNNER = ROOT / (
    "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts/"
    "run_p3_apc_boundary_onehost.sh"
)


class PrefixCacheContractTest(unittest.TestCase):

  def test_apc_is_default_off_and_has_one_frozenlake_reader(self):
    text = TRAIN.read_text(encoding="utf-8")
    self.assertIn(
        'os.getenv(\n    "CANON_VLLM_ENABLE_PREFIX_CACHING", "0"\n)', text
    )
    self.assertIn(
        '"enable_prefix_caching": CANON_VLLM_ENABLE_PREFIX_CACHING', text
    )
    self.assertIn("[P3_APC_CONFIG]", text)

  def test_b_rescore_remains_an_independent_full_recompute(self):
    text = ROLLOUT.read_text(encoding="utf-8")
    self.assertIn("reset_prefix_cache: bool = True", text)
    self.assertIn("reset_prefix_cache=reset_prefix_cache", text)
    self.assertIn(
        "Otherwise a cached prefix makes the 'prefill' partly a cache read",
        text,
    )

  def test_boundary_probe_is_default_off_and_reuses_full_reset_b(self):
    recipe = TRAIN.read_text(encoding="utf-8")
    rollout = ROLLOUT.read_text(encoding="utf-8")
    self.assertEqual(recipe.count('"CANON_P3_APC_BOUNDARY_REPORT", ""'), 1)
    self.assertIn("run_p3_apc_boundary_probe()", recipe)
    method = rollout.split("def run_p3_apc_boundary_probe", 1)[1]
    method = method.split("def get_grouped_prefill_rescore_logps", 1)[0]
    self.assertIn("self.get_prefill_rescore_logps(", method)
    self.assertIn("reset_prefix_cache=True", method)
    self.assertIn('b_cached_tokens != (0,)', method)
    self.assertIn("prompt_logprobs=None", method)
    self.assertIn("logprobs=sampled_logprobs", method)
    self.assertIn("cached_params.skip_reading_prefix_cache is not False", method)
    self.assertIn("completion.logprobs", method)

  def test_dirty_page_negative_is_default_off_and_runner_gated(self):
    rollout = ROLLOUT.read_text(encoding="utf-8")
    runner = BOUNDARY_RUNNER.read_text(encoding="utf-8")
    self.assertIn('os.environ.get("CANON_P3_APC_DIRTY_PAGE", "")', rollout)
    self.assertIn("def _p3_dirty_one_cached_page", rollout)
    self.assertIn('dirty) apc=1; dirty_page=1', runner)
    self.assertIn('-e CANON_P3_APC_DIRTY_PAGE="$dirty_page"', runner)
    self.assertIn('--expect-dirty-page "$dirty_page"', runner)

  def test_flag_registry_records_apc_contract_and_count(self):
    text = FLAGS.read_text(encoding="utf-8")
    self.assertIn("CANON_VLLM_ENABLE_PREFIX_CACHING", text)
    self.assertIn("CANON_P3_APC_DIRTY_PAGE", text)
    self.assertIn("Count: 408 settable names", text)

  def test_v4_optimizer_escape_hatch_is_gate_only(self):
    text = TRAIN.read_text(encoding="utf-8")
    self.assertIn(
        'if _FL_STATELESS_OPTIMIZER == "1" and not CANON_P38_PRECHECK_ONLY:',
        text,
    )
    self.assertIn("optimizer = optax.sgd", text)

  def test_tpu_launch_segment_has_no_pipeline(self):
    for runner in (RUNNER, BOUNDARY_RUNNER):
      text = runner.read_text(encoding="utf-8")
      launch = text.split("sudo docker run", 1)[1].split("docker_rc=$?", 1)[0]
      self.assertNotIn("|", launch.replace("||", ""))

  def test_production_runner_has_fail_closed_certification_mode(self):
    runner = RUNNER.read_text(encoding="utf-8")
    self.assertIn("cert) apc=1; seam_mode=\"\"; purpose=certification", runner)
    self.assertIn('--purpose "$purpose"', runner)

  def test_performance_modes_are_greedy_and_change_only_apc(self):
    runner = RUNNER.read_text(encoding="utf-8")
    self.assertIn(
        'perf-control) apc=0; seam_mode=""; purpose=certification; temperature=0.0',
        runner,
    )
    self.assertIn(
        'perf-apc) apc=1; seam_mode=""; purpose=certification; temperature=0.0',
        runner,
    )
    self.assertIn("--temperature=$temperature", runner)

  def test_profile_modes_capture_one_diagnostic_round(self):
    runner = RUNNER.read_text(encoding="utf-8")
    self.assertIn(
        'xprof-control) apc=0; seam_mode=""; purpose=certification; '
        'temperature=0.0; profile=1',
        runner,
    )
    self.assertIn(
        'xprof-apc) apc=1; seam_mode=""; purpose=certification; '
        'temperature=0.0; profile=1',
        runner,
    )
    self.assertIn("-e CANON_XPROF_PHASE=diagnostic", runner)
    self.assertIn("-e CANON_XPROF_SKIP_STEPS=1", runner)
    self.assertIn("-e CANON_XPROF_STEPS=1", runner)
    self.assertIn("-e CANON_XPROF_PYTHON_TRACER=0", runner)
    self.assertIn("-e CANON_PERF_TRACE_DIR=", runner)

  def test_frozenlake_wires_official_perfetto_only_when_requested(self):
    recipe = TRAIN.read_text(encoding="utf-8")
    self.assertIn('os.environ.get("CANON_PERF_TRACE_DIR", "")', recipe)
    self.assertIn("PerfMetricsExport.from_cluster_config", recipe)
    self.assertIn("perf_config=perf_config", recipe)

  def test_diagnostic_window_stops_and_exports_before_next_round_is_queued(self):
    learner = LEARNER.read_text(encoding="utf-8")
    catch = learner.split("except alignment.P38DiagnosticRoundComplete:", 1)[1]
    catch = catch.split("continue", 1)[0]
    self.assertLess(
        catch.index("_canon_xprof_diagnostic_round_boundary(completed)"),
        catch.index("next_prompts = next(full_dataset_iterator)"),
    )
    self.assertIn("self.rl_cluster.perf_v2.export()", catch)
    self.assertIn('mode not in ("step", "update", "diagnostic")', learner)


if __name__ == "__main__":
  unittest.main()
