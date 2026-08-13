"""End-to-end environment admission tests for the rendered P34 JobSet."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
RENDERER = PKG / "cluster/render_p34_jobset.py"
SPEC = importlib.util.spec_from_file_location("p34_env_renderer", RENDERER)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P34 JobSet renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _main_env(document) -> dict[str, str]:
  head = renderer._head(document)
  main = renderer._container(head["containers"], "jax-tpu")
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


class P34EnvironmentContractTest(unittest.TestCase):

  def _run_env(
      self, *, stage: str = "backward-no-commit",
      override_profile: str | None = None
  ):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      if stage == "full":
        whitelist = Path(renderer.P34_CLEAN_WHITELIST)
        whitelist_sha = renderer.P34_CLEAN_WHITELIST_SHA256
      else:
        whitelist = root / "gold.jsonl"
        whitelist.write_text('{"docker_image":"test-image"}\n')
        whitelist_sha = hashlib.sha256(whitelist.read_bytes()).hexdigest()
      base = yaml.safe_load(
          (PKG / "cluster/jobset-256cluster-64chip.yaml").read_text()
      )
      document = renderer.render(
          base,
          source_commit="1" * 40,
          source_branch=renderer.DEFAULT_SOURCE_BRANCH,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          run_id="env-gate",
          stage=stage,
          cpu_nodepool="cpu-pool",
          worker_nodepool="tpu-pool",
          model_pvc="model-pvc",
          whitelist=str(whitelist),
          whitelist_sha256=whitelist_sha,
      )
      environ = os.environ.copy()
      environ.update(_main_env(document))
      state = root / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
      })
      if override_profile is not None:
        wrapper = root / "profile.env"
        wrapper.write_text(
            "source "
            + str(
                PKG
                / "cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env"
            )
            + "\n"
            + override_profile
            + "\n"
        )
        environ["CANON_PROFILE_FILE"] = str(wrapper)
      result = subprocess.run(
          ["bash", str(PKG / "cluster/steps/00_env.sh")],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      resolved = root / "state/env.sh"
      resolved_text = resolved.read_text() if resolved.is_file() else ""
      return result, resolved_text

  def test_rendered_production_environment_passes_preflight(self):
    result, resolved = self._run_env()
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("[env] P34 contract OK: DP16xTP8", result.stdout)
    self.assertIn("export CANON_PRE_ALIGN_GATE=1", resolved)
    self.assertIn("export CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=1", resolved)
    self.assertIn("export CANON_EXPECT_MODEL_MESH_IDS=''", resolved)
    self.assertNotIn("test-only", resolved)

  def test_onehost_model_mesh_assertion_is_rejected(self):
    result, _ = self._run_env(
        override_profile="export CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("must not inherit a one-host model mesh ID", result.stdout)

  def test_rendered_full_environment_pins_capture_device_and_warning_policy(self):
    result, resolved = self._run_env(stage="full")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_P34_TRAJECTORY_CAPTURE=1", resolved)
    self.assertIn("export CANON_OPT_STATE_RESIDENT=1", resolved)
    self.assertIn("export CANON_P30_OPT_STATE_OFFLOAD=0", resolved)
    self.assertIn("export CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=1", resolved)

  def test_full_environment_rejects_strict_alignment_override(self):
    result, _ = self._run_env(
        stage="full",
        override_profile="export CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=0",
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("finite alignment warning-only", result.stdout)

  def test_missing_pre_backward_gate_is_rejected(self):
    result, _ = self._run_env(override_profile="export CANON_PRE_ALIGN_GATE=0")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("pre-backward gate", result.stdout)

  def test_missing_deepswe_sampler_policy_is_rejected(self):
    result, _ = self._run_env(
        override_profile="export CANON_P34_DISABLE_SAMPLER_IS=0"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("neutral importance paths", result.stdout)

  def test_missing_weight_report_is_rejected(self):
    result, _ = self._run_env(
        override_profile="unset CANON_P34_WEIGHT_REPORT"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("CANON_P34_WEIGHT_REPORT", result.stdout)


if __name__ == "__main__":
  unittest.main()
