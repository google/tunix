#!/usr/bin/env python3
"""Contracts for the P38.2h actual-model backward-no-commit renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p38_backward_jobset.py"
BASE = ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
SOURCE = "2" * 40
RUN_ID = "p38h1"
SPEC = importlib.util.spec_from_file_location("render_p38_backward", RENDERER)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class RenderP38BackwardJobsetTest(unittest.TestCase):

  def test_single_variable_backward_contract(self):
    document = renderer.render(
        base_path=BASE, source_commit=SOURCE, run_id=RUN_ID
    )
    renderer.validate(document, SOURCE, RUN_ID)
    env = renderer.p33._env_values(document)
    self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], "1")
    self.assertEqual(env["CANON_P33_RUN_STAGE"], "backward-no-commit")
    self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
    self.assertEqual(env["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], "0")
    self.assertEqual(env["CANON_P33_DISABLE_EVAL"], "1")
    for key in renderer._FORBIDDEN_ENV:
      self.assertNotIn(key, env)
    labels = document["metadata"]["labels"]
    self.assertEqual(
        labels["canon.zero-tim/diagnostic"], "p38-fixed-lm-head-backward"
    )
    self.assertEqual(labels["canon.zero-tim/mutation"], "backward-no-commit")
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)

  def test_cli_writes_one_valid_yaml_and_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      command = [
          sys.executable, str(RENDERER),
          "--source-commit", SOURCE,
          "--run-id", RUN_ID,
          "--output-dir", str(directory),
      ]
      first = subprocess.run(
          command, check=False, text=True, capture_output=True
      )
      self.assertEqual(first.returncode, 0, first.stderr)
      path = directory / renderer._FILENAME
      loaded = yaml.safe_load(path.read_text())
      renderer.validate(loaded, SOURCE, RUN_ID)
      second = subprocess.run(
          command, check=False, text=True, capture_output=True
      )
      self.assertNotEqual(second.returncode, 0)
      self.assertIn("refusing to overwrite", second.stderr)

  def test_invalid_scope_fails_closed(self):
    document = renderer.render(
        base_path=BASE, source_commit=SOURCE, run_id=RUN_ID
    )
    main = renderer._main(document)
    main["env"].append({
        "name": "CANON_P38_PRECHECK_ONLY", "value": "1"
    })
    with self.assertRaisesRegex(ValueError, "precheck/observer"):
      renderer.validate(document, SOURCE, RUN_ID)

  def test_runtime_scripts_enforce_vjp_receipt(self):
    env_script = (ROOT / "canon-zero-tim/cluster/steps/00_env.sh").read_text()
    run_script = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    self.assertIn("P38.2h fixed lm-head backward-no-commit enabled", env_script)
    self.assertIn("classify_p38_fixed_lm_head_receipts.py", run_script)
    self.assertIn("p38_fixed_receipt_args+=(--require-vjp)", run_script)
    self.assertIn("fixed lm-head executable receipt contract failed", run_script)
    self.assertIn("[P38.FIXED_LM_HEAD] RECEIPT_ARTIFACT", run_script)
    self.assertIn("p38_fixed_lm_head_receipts.json", run_script)


if __name__ == "__main__":
  unittest.main()
