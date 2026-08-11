"""End-to-end CPU preflight for the rendered P39 pilot environment."""

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
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p39_env_renderer", PKG / "cluster/render_p39_deepswe_pilot.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P39 renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _main_env(document) -> dict[str, str]:
  head = renderer.p34._head(document)
  main = renderer.p34._container(head["containers"], "jax-tpu")
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


class P39EnvironmentContractTest(unittest.TestCase):

  def test_postflight_selects_the_p39_classifier(self):
    postflight = (PKG / "cluster/steps/90_run.sh").read_text()
    self.assertIn(
        'if [ "${CANON_P39_64CHIP_PILOT:-0}" = "1" ]; then',
        postflight,
    )
    self.assertIn(
        'python3 "$CANON_PKG/tests/p39_deepswe_pilot/classify_run.py"',
        postflight,
    )
    self.assertIn(
        'python3 "$CANON_PKG/tests/p34_deepswe/classify_run.py"',
        postflight,
    )

  def _run(self, override=""):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      whitelist = root / "gold.jsonl"
      whitelist.write_text('{"docker_image":"test-image"}\n')
      digest = hashlib.sha256(whitelist.read_bytes()).hexdigest()
      document = renderer.render(
          yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text()),
          source_commit="1" * 40,
          source_branch="yuxzhang/canon-zero-tim",
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          run_id="env-test",
          stage="three-update",
          cpu_nodepool="cpu-pool",
          worker_nodepool="tpu-pool",
          model_pvc="model-pvc",
          whitelist=str(whitelist),
          whitelist_sha256=digest,
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
      if override:
        wrapper = root / "profile.env"
        wrapper.write_text(
            "source "
            + str(
                PKG
                / "cluster/profiles/qwen3-32b-dp4-tp8-deepswe-pilot.env"
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

  def test_rendered_pilot_passes_preflight(self):
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("P34 contract OK: DP4xTP8", result.stdout)

  def test_offload_override_is_rejected(self):
    result = self._run(
        "export CANON_OPT_STATE_RESIDENT=0\n"
        "export CANON_P30_OPT_STATE_OFFLOAD=1"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("device-resident optimizer state", result.stdout)


if __name__ == "__main__":
  unittest.main()
