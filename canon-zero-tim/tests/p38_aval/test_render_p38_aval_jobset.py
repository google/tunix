"""Fail-closed tests for the P38 model-free aval JobSet."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p38_aval_jobset.py"
BASE = ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
SOURCE_COMMIT = "2" * 40
RUN_ID = "aval-a0"
SPEC = importlib.util.spec_from_file_location("p38_aval_renderer", RENDERER)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import the P38 aval renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _document():
  return renderer.render(
      base=renderer.p33.load_base(BASE),
      source_commit=SOURCE_COMMIT,
      run_id=RUN_ID,
  )


class RenderP38AvalJobSetTest(unittest.TestCase):

  def test_model_free_attempt_zero_contract(self):
    document = _document()
    env = renderer.p33._env_values(document)
    self.assertEqual(env["CANON_MODE"], "gate-only")
    self.assertEqual(env["CANON_RUN_P38_AVAL"], "1")
    self.assertNotIn("CANON_RUN_CMD", env)
    self.assertNotIn("WANDB_MODE", env)
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
    self.assertTrue(all(
        job["template"]["spec"]["backoffLimit"] == 0
        for job in document["spec"]["replicatedJobs"]
    ))

  def test_negative_controls_reject_probe_or_proxy_drift(self):
    document = _document()
    main = renderer._main(document)
    next(
        entry
        for entry in main["env"]
        if entry["name"] == "CANON_RUN_P38_AVAL"
    )["value"] = "0"
    with self.assertRaisesRegex(ValueError, "environment drifted"):
      renderer.validate(
          document, source_commit=SOURCE_COMMIT, run_id=RUN_ID
      )

    document = _document()
    proxy = renderer._proxy(document)
    next(
        entry
        for entry in proxy["env"]
        if entry["name"] == renderer.p36.PROXY_XLA_ENV
    )["value"] = "--xla_allow_excess_precision=true"
    with self.assertRaisesRegex(ValueError, "lost the canonical"):
      renderer.validate(
          document, source_commit=SOURCE_COMMIT, run_id=RUN_ID
      )

  def test_negative_control_rejects_workload_command(self):
    document = _document()
    main = renderer._main(document)
    main["env"].append({"name": "CANON_RUN_CMD", "value": "python train.py"})
    with self.assertRaisesRegex(ValueError, "workload environment"):
      renderer.validate(
          document, source_commit=SOURCE_COMMIT, run_id=RUN_ID
      )


if __name__ == "__main__":
  unittest.main()
