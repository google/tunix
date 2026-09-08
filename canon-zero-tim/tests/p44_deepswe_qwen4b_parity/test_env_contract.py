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

  def _run(
      self,
      topology: str,
      stage: str = "three-update",
      override: str = "",
      *,
      system_optimization_arm: str | None = None,
      raw_overrides: dict[str, str] | None = None,
  ):
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
          system_optimization_arm=system_optimization_arm,
      )
      environ = os.environ.copy()
      environ.update(renderer.p34._env(document))
      state = root / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
          "JAX_PLATFORMS": "cpu",
          "PYTHONPATH": str(ROOT),
      })
      if raw_overrides:
        environ.update(raw_overrides)
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
      command = (
          f'"{PKG / "cluster/steps/00_env.sh"}"'
          ' && source "$CANON_STATE/env.sh"'
          f' && "{sys.executable}" -c '
          "'import os; from tunix.rl import deepswe_contract; "
          "deepswe_contract.validate_environment(os.environ); "
          "print(\"[P44.PYTHON_CONTRACT] PASS\")'"
      )
      return subprocess.run(
          ["bash", "-c", command],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )

  def test_all_rendered_stages_pass_on_both_topologies(self):
    for topology, dp in (("64", 4), ("128", 8)):
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
        "128", override="export CANON_GLOBAL_TRAJECTORIES=64"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("trajectory geometry", result.stdout)

  def test_recipe_mode_overlap_is_rejected(self):
    result = self._run("64", override="export CANON_P43_DEEPSWE_DEBUG=1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("cannot overlap", result.stdout)

  def test_strict_system_optimization_arms_pass_both_topologies(self):
    for topology in ("64", "128"):
      for arm in ("control", "treatment"):
        with self.subTest(topology=topology, arm=arm):
          result = self._run(
              topology, system_optimization_arm=arm
          )
          self.assertEqual(result.returncode, 0, result.stdout)
          self.assertIn(
              f"[P44.V2] system optimization arm={arm} "
              f"topology={topology} strict=1",
              result.stdout,
          )
          self.assertIn("[P44.PYTHON_CONTRACT] PASS", result.stdout)

  def test_strict_arms_fail_closed_on_cross_arm_and_policy_drift(self):
    cases = (
        (
            "control",
            {"CANON_DP_REDUCE_ONCE": "0"},
            "control arm must keep stream and reduce-once absent",
        ),
        (
            "treatment",
            {"CANON_DP_REDUCE_ONCE": "0"},
            "treatment arm requires stream and reduce-once",
        ),
        (
            "control",
            {"CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1"},
            "common system-optimization tuple drifted",
        ),
        (
            "control",
            {"CANON_P59_CHECKED_VMA": "0"},
            "common system-optimization tuple drifted",
        ),
    )
    for arm, raw_overrides, message in cases:
      with self.subTest(arm=arm, drift=raw_overrides):
        result = self._run(
            "64",
            system_optimization_arm=arm,
            raw_overrides=raw_overrides,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stdout)


if __name__ == "__main__":
  unittest.main()
