#!/usr/bin/env python3
"""End-to-end P58 renderer -> shell profile -> Python contract validation."""

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
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "p58_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py"
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE contract")
deepswe_contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = deepswe_contract
CONTRACT_SPEC.loader.exec_module(deepswe_contract)
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p58_environment_renderer", PKG / "cluster/render_p58_deepswe_tim.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P58EnvironmentContractTest(unittest.TestCase):

  def _rendered_env(self, arm: str, stage: str) -> dict[str, str]:
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    document = renderer.render(
        base,
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="env-test",
        stage=stage,
        arm=arm,
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
    )
    return dict(renderer.p34._env(document))

  def _resolved(self, arm: str, stage: str) -> dict[str, str]:
    supplied = os.environ.copy()
    supplied.update(self._rendered_env(arm, stage))
    command = (
        "set -a; "
        f"source {PKG / 'cluster/profiles/_canonical_engine.env'}; "
        f"source {PKG / 'cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env'}; "
        "env -0"
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        env=supplied,
        check=True,
        capture_output=True,
    )
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in completed.stdout.decode().split("\0")
        if "=" in item
    }

  def _persisted(self, arm: str, stage: str):
    supplied = os.environ.copy()
    supplied.update(self._rendered_env(arm, stage))
    supplied.update({
        "CANON_PKG": str(PKG),
        "HF_TOKEN": "test-hf-runtime-token",
        "WANDB_API_KEY": "test-wandb-runtime-key",
        "INJECTED_HF_TOKEN": "test-hf-token",
        "INJECTED_WANDB_API_KEY": "test-wandb-key",
    })
    with tempfile.TemporaryDirectory() as state_dir:
      supplied["CANON_STATE"] = state_dir
      completed = subprocess.run(
          ["bash", str(PKG / "cluster/steps/00_env.sh")],
          cwd=ROOT,
          env=supplied,
          check=True,
          text=True,
          capture_output=True,
      )
      resolved = (Path(state_dir) / "env.sh").read_text()
      reloaded = subprocess.run(
          [
              "bash",
              "-c",
              f"source {Path(state_dir) / 'env.sh'}; env -0",
          ],
          env=supplied,
          check=True,
          capture_output=True,
      )
    values = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in reloaded.stdout.decode().split("\0")
        if "=" in item
    }
    return completed, resolved, values

  def test_native_renderer_environment_passes_real_00_env(self):
    completed, resolved, values = self._persisted("native", "three-update")
    self.assertIn("[env] P34 contract OK: DP8xTP8", completed.stdout)
    self.assertNotIn("REFUSING TO CONTINUE", completed.stderr)
    self.assertIn("export CANON_P32_DP_REDUCTION_ADMITTED=0", resolved)
    self.assertIn("export CANON_FROZENLAKE_L3=0", resolved)
    self.assertIn("export CANON_FROZENLAKE_P27=0", resolved)
    self.assertIn(
        "export CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0", resolved
    )
    self.assertNotIn("test-hf-runtime-token", resolved)
    self.assertNotIn("test-wandb-runtime-key", resolved)
    # 00_env.sh is a child of the entrypoint. Exercise the actual reload
    # boundary with the raw renderer environment still present, rather than
    # merely inspecting the generated exports. This is the p58c03 regression:
    # the native profile unset CANON_LOGPROB_M in the child, but a layered
    # source left the renderer's CANON_LOGPROB_M=256 alive in the parent.
    self.assertNotIn("CANON_LOGPROB_M", values)
    self.assertNotIn("CANON_FIXED_AR", values)
    self.assertEqual(values["HF_TOKEN"], "test-hf-runtime-token")
    self.assertEqual(values["WANDB_API_KEY"], "test-wandb-runtime-key")
    deepswe_contract.validate_environment(values)

  def test_zero_renderer_environment_survives_authoritative_reload(self):
    _, resolved, values = self._persisted("zero", "three-update")
    self.assertIn("export CANON_LOGPROB_M=256", resolved)
    self.assertEqual(values["CANON_LOGPROB_M"], "256")
    self.assertEqual(values["CANON_FIXED_AR"], "1")
    self.assertEqual(values["HF_TOKEN"], "test-hf-runtime-token")
    self.assertEqual(values["WANDB_API_KEY"], "test-wandb-runtime-key")
    deepswe_contract.validate_environment(values)

  def test_both_arms_resolve_to_the_signed_contract(self):
    for arm in ("native", "zero"):
      for stage in ("three-update", "full"):
        with self.subTest(arm=arm, stage=stage):
          values = self._resolved(arm, stage)
          deepswe_contract.validate_environment(values)
          workload = deepswe_contract.active_workload(values)
          self.assertEqual(workload.global_trajectories, 128)
          self.assertEqual(workload.local_trajectories, 16)
          if arm == "native":
            self.assertNotIn("CANON_FIXED_AR", values)
            self.assertNotIn("CANON_LOGPROB_M", values)
          else:
            self.assertEqual(values["CANON_FIXED_AR"], "1")
            self.assertEqual(values["CANON_LOGPROB_M"], "256")


if __name__ == "__main__":
  unittest.main()
