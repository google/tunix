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


if __name__ == "__main__":
  unittest.main()
