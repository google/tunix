"""Static launch-boundary checks for the real DeepSWE program."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DeepSWEScriptContractTest(unittest.TestCase):

  def test_canonical_entrypoint_initializes_pathways_before_program_import(self):
    text = (ROOT / "examples/deepswe/canonical_entrypoint.py").read_text()
    self.assertLess(text.index("pathwaysutils.initialize()"), text.index("runpy.run_module"))
    self.assertNotIn("import jax", text)

  def test_strict_mode_registers_scheduler_flags(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn('"--rollout_vllm_max_num_seqs"', text)
    self.assertIn('"--max_num_batched_tokens"', text)
    self.assertIn("args = parser.parse_args()", text)
    self.assertIn('"enable_prefix_caching": not P34_DEEPSWE', text)
    self.assertIn("scheduler_per_dp=4/256", text)

  def test_p34_uses_dp_axes_and_replicated_parameters(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn('rollout_dims = [("dp", 16), ("tp", 8)]', text)
    self.assertIn('train_dims = [("dp", 16), ("tp", 8)]', text)
    self.assertIn("configure_replicated_parameter_sharding", text)
    self.assertIn("P34 forbids FSDP", text)

  def test_backward_no_commit_requires_full_array_repeat(self):
    adapter = (ROOT / "tunix/rl/canonical_qwen3_adapter.py").read_text()
    learner = (ROOT / "tunix/rl/agentic/agentic_rl_learner.py").read_text()
    self.assertIn("deterministic_repeat_exact", adapter)
    self.assertIn("jnp.array_equal(first, second)", adapter)
    self.assertIn("deterministic_repeat=(p34_workload and p33_no_commit)", learner)

  def test_cross_role_weights_are_checked_before_rescore(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    weight_gate = learner.index("persist_weight_attestation(")
    rescore = learner.index(
        "rescore_source = self.rl_cluster.rollout.get_prefill_rescore_logps"
    )
    self.assertLess(weight_gate, rescore)
    self.assertIn("attest_actor_anchor_matches_engine()", learner)

  def test_weight_evidence_is_fail_closed_and_classified(self):
    runner = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    self.assertIn("report_keys+=(CANON_P34_WEIGHT_REPORT)", runner)
    self.assertIn("[P34.WEIGHT_ARTIFACT_JSON]", runner)
    self.assertIn('--weight-report "$CANON_P34_WEIGHT_REPORT"', runner)


if __name__ == "__main__":
  unittest.main()
