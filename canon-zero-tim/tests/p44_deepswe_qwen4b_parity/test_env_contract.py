"""End-to-end CPU preflight for both rendered P44 topologies."""

from __future__ import annotations

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
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p44_env_renderer", PKG / "cluster/render_p44_deepswe_parity.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P44 parity renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P44EnvironmentContractTest(unittest.TestCase):

  def _run(self, topology: str, stage: str = "three-update", override: str = ""):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      document = renderer.render(
          yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text()),
          source_commit="1" * 40,
          source_branch="yuxzhang/canon-zero-tim",
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          run_id="env-test",
          stage=stage,
          topology=topology,
          cpu_nodepool="cpu-pool",
          worker_nodepool="tpu-pool",
          model_pvc="model-pvc",
          whitelist=renderer.p34.P34_CLEAN_WHITELIST,
          whitelist_sha256=renderer.p34.P34_CLEAN_WHITELIST_SHA256,
      )
      environ = os.environ.copy()
      environ.update(renderer.p34._env(document))
      state = root / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
      })
      if override:
        wrapper = root / "profile.env"
        wrapper.write_text(
            "source "
            + str(
                PKG
                / "cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env"
            )
            + "\n"
            + override
            + "\n"
        )
        environ["CANON_PROFILE_FILE"] = str(wrapper)
      return subprocess.run(
          ["bash", str(PKG / "cluster/steps/00_env.sh")],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )

  def test_all_rendered_stages_pass_on_both_topologies(self):
    for topology, dp in (("64", 4), ("256", 16)):
      for stage in ("rollout-only", "one-update", "three-update"):
        with self.subTest(topology=topology, stage=stage):
          result = self._run(topology, stage)
          self.assertEqual(result.returncode, 0, result.stdout)
          self.assertIn(f"P34 contract OK: DP{dp}xTP8", result.stdout)

  def test_topology_and_batch_drift_are_rejected(self):
    result = self._run("64", override="export CANON_DP_SIZE=16")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("role topology", result.stdout)
    result = self._run(
        "256", override="export CANON_GLOBAL_TRAJECTORIES=64"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("trajectory geometry", result.stdout)

  def test_recipe_mode_overlap_is_rejected(self):
    result = self._run("64", override="export CANON_P43_DEEPSWE_DEBUG=1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("cannot overlap", result.stdout)


if __name__ == "__main__":
  unittest.main()
