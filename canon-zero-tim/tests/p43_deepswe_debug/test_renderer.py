"""Renderer contracts for the P43 DeepSWE debug ladder."""

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
    "p43_debug_renderer", PKG / "cluster/render_p43_deepswe_debug.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P43 debug renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P43RendererTest(unittest.TestCase):

  def _render(self, stage="three-update"):
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    return renderer.render(
        base,
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="debug-test",
        stage=stage,
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
        whitelist="/data/gold.jsonl",
        whitelist_sha256="3" * 64,
    )

  def test_all_debug_stages_render_exact_geometry(self):
    expected = {
        "rollout-only": ("1", "--max_steps=1"),
        "one-update": ("0", "--max_steps=1"),
        "three-update": ("0", "--max_steps=3"),
    }
    for stage, (no_commit, step_arg) in expected.items():
      with self.subTest(stage=stage):
        document = self._render(stage)
        env = renderer.p34._env(document)
        worker = renderer.p34._worker(document)
        self.assertEqual(
            (worker["completions"], worker["parallelism"]), (16, 16)
        )
        self.assertEqual(env["CANON_P43_DEEPSWE_DEBUG"], "1")
        self.assertEqual(env["CANON_P34_NO_COMMIT"], no_commit)
        self.assertIn("--model_version=Qwen3-8B", env["CANON_RUN_CMD"])
        self.assertIn("--rollout_mesh_tp=8", env["CANON_RUN_CMD"])
        self.assertIn("--num_generations=4", env["CANON_RUN_CMD"])
        self.assertIn(step_arg, env["CANON_RUN_CMD"])
        self.assertTrue(env["CANON_P43_DEBUG_DIR"].endswith("/debug"))

  def test_production_and_unbounded_stages_are_rejected(self):
    for stage in ("backward-no-commit", "full"):
      with self.subTest(stage=stage):
        with self.assertRaisesRegex(ValueError, "P43 debug admits"):
          self._render(stage)

  def test_renderer_refuses_model_drift(self):
    document = self._render("one-update")
    env = renderer.p34._env(document)
    env["CANON_RUN_CMD"] = env["CANON_RUN_CMD"].replace(
        "--model_version=Qwen3-8B", "--model_version=Qwen3-32B"
    )
    head = renderer.p34._head(document)
    main = renderer.p34._container(head["containers"], "jax-tpu")
    for item in main["env"]:
      if item["name"] == "CANON_RUN_CMD":
        item["value"] = env["CANON_RUN_CMD"]
    with self.assertRaisesRegex(ValueError, "lost a signed field"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="one-update",
      )


if __name__ == "__main__":
  unittest.main()
