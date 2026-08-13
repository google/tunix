"""End-to-end CPU preflight for the P46 JobSet families."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
CLUSTER = PKG / "cluster"
sys.path.insert(0, str(CLUSTER))

import render_p34_jobset as p34  # pylint: disable=wrong-import-position
import render_p46_deepswe_profiles as renderer  # pylint: disable=wrong-import-position


class P46EnvironmentContractTest(unittest.TestCase):

  def _render(self, workload: str, topology: str):
    base_name = (
        "jobset-64chip.yaml"
        if topology == "64"
        else "jobset-256cluster-64chip.yaml"
    )
    return renderer.render(
        yaml.safe_load((CLUSTER / base_name).read_text(encoding="utf-8")),
        workload=workload,
        topology=topology,
        source_commit="6" * 40,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image="example.invalid/tunix@sha256:" + "7" * 64,
        run_id="envtest",
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="models-pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
        logical_shard_index=0,
        physical_shard_index=0,
    )

  def _run(self, workload: str, topology: str, override: str = ""):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      document = self._render(workload, topology)
      environ = os.environ.copy()
      environ.update(p34._env(document))
      state = root / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
      })
      if override:
        original = Path(environ["CANON_PROFILE_FILE"])
        if not original.is_absolute():
          original = PKG / original
        wrapper = root / "profile.env"
        wrapper.write_text(
            f"source {original}\n{override}\n", encoding="utf-8"
        )
        environ["CANON_PROFILE_FILE"] = str(wrapper)
      return subprocess.run(
          ["bash", str(CLUSTER / "steps/00_env.sh")],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )

  def test_q32_training_preflight_passes_on_both_topologies(self):
    for topology, dp, global_m in (
        ("64", 4, 1024),
        ("256", 16, 4096),
    ):
      with self.subTest(topology=topology):
        result = self._run("q32-train", topology)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"P34 contract OK: DP{dp}xTP8 per role, local M256, "
            f"global M{global_m}",
            result.stdout,
        )

  def test_clean_evaluation_preflight_passes_without_trainer(self):
    for topology in ("64", "256"):
      with self.subTest(topology=topology):
        result = self._run("q4-clean-eval", topology)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"P46 evaluation contract OK: topology={topology}", result.stdout
        )

  def test_q32_topology_drift_and_eval_trainer_overlap_fail_closed(self):
    result = self._run(
        "q32-train", "64", "export CANON_DP_SIZE=16"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("role topology does not match", result.stdout)
    result = self._run(
        "q4-clean-eval", "64", "export CANON_P32_TRAIN_ADMITTED=1"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("must not admit a trainer", result.stdout)


if __name__ == "__main__":
  unittest.main()
