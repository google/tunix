"""Renderer parity contracts for the dual-topology P44 ladder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p44_parity_renderer", PKG / "cluster/render_p44_deepswe_parity.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P44 parity renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P44RendererTest(unittest.TestCase):

  def _render(self, topology: str, stage: str = "three-update"):
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    return renderer.render(
        base,
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="parity-test",
        stage=stage,
        topology=topology,
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
        whitelist="/data/gold.jsonl",
        whitelist_sha256="3" * 64,
    )

  def test_all_six_bounded_jobsets_render(self):
    expected_workers = {"64": 16, "256": 64}
    expected_dp = {"64": 4, "256": 16}
    for topology in ("64", "256"):
      for stage in ("rollout-only", "one-update", "three-update"):
        with self.subTest(topology=topology, stage=stage):
          document = self._render(topology, stage)
          env = renderer.p34._env(document)
          worker = renderer.p34._worker(document)
          self.assertEqual(worker["completions"], expected_workers[topology])
          self.assertEqual(worker["parallelism"], expected_workers[topology])
          self.assertEqual(env["CANON_P44_TOPOLOGY"], topology)
          self.assertEqual(env["CANON_P44_DEEPSWE_PARITY"], "1")
          self.assertIn("--model_version=Qwen3-4B", env["CANON_RUN_CMD"])
          self.assertIn(
              f"--rollout_mesh_dp={expected_dp[topology]}",
              env["CANON_RUN_CMD"],
          )

  def test_normalized_rendered_recipes_are_identical(self):
    for stage in ("rollout-only", "one-update", "three-update"):
      with self.subTest(stage=stage):
        small = renderer.recipe_signature(self._render("64", stage))
        large = renderer.recipe_signature(self._render("256", stage))
        self.assertEqual(small, large)

  def test_invalid_topology_and_unbounded_stage_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "exactly 64 or 256"):
      self._render("128")
    for stage in ("backward-no-commit", "full"):
      with self.subTest(stage=stage):
        with self.assertRaisesRegex(ValueError, "P44 parity admits"):
          self._render("64", stage)

  def test_model_drift_is_rejected(self):
    document = self._render("64", "one-update")
    head = renderer.p34._head(document)
    main = renderer.p34._container(head["containers"], "jax-tpu")
    for item in main["env"]:
      if item["name"] == "CANON_RUN_CMD":
        item["value"] = item["value"].replace(
            "--model_version=Qwen3-4B", "--model_version=Qwen3-8B"
        )
    with self.assertRaisesRegex(ValueError, "lost a signed field"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="one-update",
          topology="64",
      )


if __name__ == "__main__":
  unittest.main()
