#!/usr/bin/env python3
"""End-to-end P58 renderer -> shell profile -> Python contract validation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
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

  def _resolved(self, arm: str, stage: str) -> dict[str, str]:
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
    supplied = os.environ.copy()
    supplied.update(renderer.p34._env(document))
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
