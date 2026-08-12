"""Static integration guards for the P43 training and postflight path."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"


class P43IntegrationContractTest(unittest.TestCase):

  def test_qwen8b_swiglu_remains_on_unpadded_bf256_path(self):
    path = (
        ROOT
        / "canon-zero-tim/src/engine_shims/models/qwen8b/p22xf_contract.py"
    )
    spec = importlib.util.spec_from_file_location("p43_qwen8b_contract", path)
    if spec is None or spec.loader is None:
      raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    self.assertEqual(module.INTERMEDIATE_SIZE // module.TP_SIZE, 3072)
    self.assertEqual(module.SWIGLU_FEATURE_PADDING, {})
    self.assertEqual(3072 % 256, 0)

  def test_dataset_loader_uses_modern_datasets_api(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    dataset_call = text.split("datasets_lib.load_dataset(", 1)[1].split(
        ")", 1
    )[0]
    self.assertNotIn("trust_remote_code", dataset_call)
    cli_data = (ROOT / "examples/deepswe/deepswe_data.py").read_text()
    self.assertNotIn("trust_remote_code", cli_data)

  def test_artifact_write_precedes_alignment_and_update(self):
    grpo = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    writer = grpo.index("deepswe_debug.persist_batch(")
    alignment = grpo.index("if alignment.enabled():", writer)
    self.assertLess(writer, alignment)

    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    rollout_exit = learner.index("marker = deepswe_debug.marker_prefix()")
    segmented_update = learner.index("p28_g6_update =", rollout_exit)
    self.assertLess(rollout_exit, segmented_update)
    self.assertIn("deepswe_debug.rollout_only()", learner)

  def test_postflight_selects_p43_before_p39(self):
    postflight = (PKG / "cluster/steps/90_run.sh").read_text()
    p43 = postflight.index("tests/p43_deepswe_debug/classify_run.py")
    p39 = postflight.index("tests/p39_deepswe_pilot/classify_run.py")
    self.assertLess(p43, p39)

  def test_training_batch_uses_active_workload_geometry(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn(
        "p34.global_trajectories if P34_DEEPSWE else None", text
    )
    self.assertIn("if p34.devices_per_role == 32", text)


if __name__ == "__main__":
  unittest.main()
