"""Fail-closed tests for the P36 Pathways proxy-XLA JobSet."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py"
BASE = ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
SOURCE_COMMIT = "1" * 40
RUN_ID = "flag-on"
SPEC = importlib.util.spec_from_file_location("p36_renderer", RENDERER)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import the P36 JobSet renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _document():
  return renderer.render(
      base=renderer.p33.load_base(BASE),
      source_commit=SOURCE_COMMIT,
      run_id=RUN_ID,
  )


class RenderP36ProxyXlaJobSetTest(unittest.TestCase):

  def test_renders_attempt_zero_gate_only_with_proxy_environment(self):
    document = _document()
    env = renderer.p33._env_values(document)
    proxy = renderer._proxy(document)
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
    self.assertTrue(all(
        job["template"]["spec"]["backoffLimit"] == 0
        for job in document["spec"]["replicatedJobs"]
    ))
    self.assertEqual(env["CANON_MODE"], "gate-only")
    self.assertEqual(env["CANON_EXPECT_COMMIT"], SOURCE_COMMIT)
    self.assertEqual(env["CANON_WAYCOUNT_WIDTHS"], "2,4,8")
    self.assertEqual(env["CANON_WAYCOUNT_DEPTHS"], "8,15")
    self.assertEqual(env["CANON_GCS_CACHE_BUCKET"], "")
    self.assertNotIn("CANON_P32_RC_STAGE", env)
    self.assertFalse(any(
        arg.startswith(renderer.PROXY_XLA_PREFIX) for arg in proxy["args"]
    ))
    self.assertEqual(
        [
            entry
            for entry in proxy["env"]
            if entry["name"] == renderer.PROXY_XLA_ENV
        ],
        [{"name": renderer.PROXY_XLA_ENV, "value": renderer.PROXY_XLA_FLAG}],
    )

  def test_negative_controls_reject_missing_duplicate_and_true_environment(self):
    for case in ("missing", "duplicate", "true"):
      with self.subTest(case=case):
        document = copy.deepcopy(_document())
        proxy = renderer._proxy(document)
        matches = [
            entry
            for entry in proxy["env"]
            if entry["name"] == renderer.PROXY_XLA_ENV
        ]
        self.assertEqual(len(matches), 1)
        if case == "missing":
          proxy["env"].remove(matches[0])
        elif case == "duplicate":
          proxy["env"].append(copy.deepcopy(matches[0]))
        else:
          matches[0]["value"] = "--xla_allow_excess_precision=true"
        with self.assertRaisesRegex(ValueError, "through XLA_FLAGS"):
          renderer.validate(
              document, source_commit=SOURCE_COMMIT, run_id=RUN_ID
          )

  def test_negative_control_rejects_raw_proxy_argument(self):
    document = _document()
    renderer._proxy(document)["args"].append(renderer.PROXY_XLA_FLAG)
    with self.assertRaisesRegex(ValueError, "must not receive a raw"):
      renderer.validate(document, source_commit=SOURCE_COMMIT, run_id=RUN_ID)

  def test_negative_control_rejects_training_mode(self):
    document = _document()
    main = renderer._main(document)
    next(
        entry for entry in main["env"] if entry["name"] == "CANON_MODE"
    )["value"] = "run"
    with self.assertRaisesRegex(ValueError, "environment drifted"):
      renderer.validate(document, source_commit=SOURCE_COMMIT, run_id=RUN_ID)

  def test_negative_control_rejects_environment_on_resource_manager(self):
    document = _document()
    head = renderer.p33._head_pod(document)
    manager = renderer.p33._container(
        head["initContainers"], "pathways-rm"
    )
    manager["env"].append({
        "name": renderer.PROXY_XLA_ENV,
        "value": renderer.PROXY_XLA_FLAG,
    })
    with self.assertRaisesRegex(ValueError, "only through the proxy"):
      renderer.validate(document, source_commit=SOURCE_COMMIT, run_id=RUN_ID)

  def test_cli_refuses_to_overwrite_rendered_manifest(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "p36.yaml"
      output.write_text("occupied\n", encoding="utf-8")
      result = subprocess.run(
          [
              sys.executable,
              str(RENDERER),
              "--source-commit",
              SOURCE_COMMIT,
              "--run-id",
              RUN_ID,
              "--output",
              str(output),
          ],
          cwd=ROOT,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("refusing to overwrite", result.stderr)
      self.assertEqual(output.read_text(encoding="utf-8"), "occupied\n")

  def test_rendered_yaml_contains_no_literal_credentials(self):
    serialized = yaml.safe_dump(_document(), sort_keys=False)
    self.assertNotIn("wandb_v1_", serialized)
    self.assertNotIn("github_pat_", serialized)
    self.assertNotIn("ghp_", serialized)


if __name__ == "__main__":
  unittest.main()
