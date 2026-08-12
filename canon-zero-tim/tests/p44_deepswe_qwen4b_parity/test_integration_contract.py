"""Static guards for the P44 training and evidence integration path."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class P44IntegrationContractTest(unittest.TestCase):

  def test_both_dataset_entrypoints_use_the_modern_datasets_api(self):
    notebook = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    dataset_call = notebook.split("datasets_lib.load_dataset(", 1)[1].split(
        ")", 1
    )[0]
    self.assertNotIn("trust_remote_code", dataset_call)
    cli_data = (ROOT / "examples/deepswe/deepswe_data.py").read_text()
    self.assertNotIn("trust_remote_code", cli_data)

  def test_training_selects_physical_split_from_role_size(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn("if p34.devices_per_role == 32", text)
    self.assertIn("deepswe_contract.split_4x4x4_role_devices", text)
    self.assertIn("deepswe_contract.split_4x8x8_role_devices", text)
    self.assertIn("[P34.DEVICE_INVENTORY] PASS", text)

  def test_reviewed_agentic_batch_semantics_are_present(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    grpo = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    self.assertIn("prompts = [chat_lists]", learner)
    self.assertIn("configured_compute_logps", grpo)
    self.assertIn(
        "configured_compute_logps * self.algo_config.num_generations", grpo
    )
    self.assertIn("[{marker}.LOGPS_BATCH]", grpo)

  def test_artifacts_precede_alignment_and_use_active_directory(self):
    text = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    artifact = text.index("deepswe_debug.persist_batch(")
    alignment = text.index("if alignment.enabled():", artifact)
    self.assertLess(artifact, alignment)
    self.assertIn("output_dir=deepswe_debug.artifact_directory()", text)
    self.assertIn("deepswe_debug.rollout_only()", text)
    self.assertIn('"p44-qwen4b-parity-64",', text)
    self.assertIn('"p44-qwen4b-parity-256",', text)

  def test_rollout_only_marker_and_alignment_are_shared(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    alignment = (ROOT / "tunix/rl/alignment.py").read_text()
    self.assertIn("marker = deepswe_debug.marker_prefix()", learner)
    self.assertIn("deepswe_debug.rollout_only()", learner)
    self.assertIn("p44_parity", alignment)
    self.assertIn("sum((p39_pilot, p43_debug, p44_parity)) == 1", alignment)

  def test_postflight_routes_p44_before_existing_deepswe_lanes(self):
    text = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    reservation = text.index("report_keys+=(CANON_P44_DEBUG_DIR)")
    p44 = text.index(
        "tests/p44_deepswe_qwen4b_parity/classify_run.py"
    )
    p43 = text.index("tests/p43_deepswe_debug/classify_run.py", p44)
    p39 = text.index("tests/p39_deepswe_pilot/classify_run.py", p43)
    p34 = text.index("tests/p34_deepswe/classify_run.py", p39)
    self.assertLess(reservation, p44)
    self.assertLess(p44, p43)
    self.assertLess(p43, p39)
    self.assertLess(p39, p34)
    self.assertIn('--topology "$CANON_P44_TOPOLOGY"', text[p44 - 400:p43])
    self.assertIn('--debug-dir "$CANON_P44_DEBUG_DIR"', text[p44:p43])


if __name__ == "__main__":
  unittest.main()
