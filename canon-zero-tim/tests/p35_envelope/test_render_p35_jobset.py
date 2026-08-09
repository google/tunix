"""Fail-closed tests for the single P35 envelope-short JobSet."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p35_jobset.py"
SPEC = importlib.util.spec_from_file_location("p35_renderer", RENDERER)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class RenderP35JobSetTest(unittest.TestCase):

  def test_renders_one_attempt_zero_pre_backward_job(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    env = renderer.p33._env_values(document)
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
    self.assertEqual(env["CANON_P35_ENVELOPE"], "1")
    self.assertEqual(env["CANON_P33_RUN_STAGE"], "envelope-short")
    self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
    self.assertIn("--max_response_length=64", env["CANON_RUN_CMD"])
    self.assertIn("--max_steps=1", env["CANON_RUN_CMD"])

  def test_negative_control_rejects_training_stage(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    env = renderer._main_env(document)
    next(item for item in env if item["name"] == "CANON_P33_RUN_STAGE")["value"] = "full"
    with self.assertRaisesRegex(ValueError, "drifted"):
      renderer.validate(document, source_commit="1" * 40, run_id="r20")


if __name__ == "__main__":
  unittest.main()
