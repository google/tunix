"""Renderer contracts for the P39 DeepSWE pilot."""

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
    "p39_pilot_renderer", PKG / "cluster/render_p39_deepswe_pilot.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P39 pilot renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P39RendererTest(unittest.TestCase):

  def _render(self, stage="three-update"):
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    return renderer.render(
        base,
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="pilot-test",
        stage=stage,
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
        whitelist="/data/gold.jsonl",
        whitelist_sha256="3" * 64,
    )

  def test_three_update_is_dp4_tp8_resident(self):
    document = self._render()
    head = renderer.p34._head(document)
    main = renderer.p34._container(head["containers"], "jax-tpu")
    env = {item["name"]: item["value"] for item in main["env"] if "value" in item}
    worker = renderer.p34._worker(document)
    self.assertEqual((worker["completions"], worker["parallelism"]), (16, 16))
    self.assertEqual(env["CANON_P39_64CHIP_PILOT"], "1")
    self.assertEqual(env["CANON_OPT_STATE_RESIDENT"], "1")
    self.assertEqual(env["CANON_P30_OPT_STATE_OFFLOAD"], "0")
    self.assertIn("--rollout_mesh_dp=4", env["CANON_RUN_CMD"])
    self.assertIn("--optimizer_offload=False", env["CANON_RUN_CMD"])

  def test_full_stage_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "only one-update or three-update"):
      self._render(stage="full")


if __name__ == "__main__":
  unittest.main()
