#!/usr/bin/env python3
"""Exercise rendered M15 APC JobSets through the real CPU preflight resolver."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[4]
CANON = ROOT / "canon-zero-tim"
RENDERER_PATH = CANON / "cluster/render_v1_apc_m15_target_debug.py"
BASE = CANON / "cluster/jobset-64chip.yaml"
SOURCE = "7" * 40
SPEC = importlib.util.spec_from_file_location("render_v1_apc_m15_target_debug", RENDERER_PATH)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _literal_env(document: dict) -> dict[str, str]:
  container = renderer._container(document)  # pylint: disable=protected-access
  return {
      item["name"]: str(item["value"])
      for item in container["env"]
      if "value" in item
  }


class ResolvedEnvironmentTest(unittest.TestCase):

  def _resolve(
      self,
      arm: str,
      *,
      wrong_profile: bool = False,
      wrong_replay_path: bool = False,
  ):
    run_id = f"cpu-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="v1-apc-render-", dir="/tmp") as output:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(output),
          source_commit=SOURCE,
          run_id=run_id,
      )
      path = next(path for path in paths if path.stem.endswith(f"-{arm}"))
      document = yaml.safe_load(path.read_text(encoding="utf-8"))
      env = os.environ.copy()
      env.update(_literal_env(document))
      state = Path(env["CANON_STATE"])
      state.mkdir(parents=True, exist_ok=False)
      self.addCleanup(shutil.rmtree, state)
      env.update({
          "CANON_PKG": str(CANON),
          "JOBSET_RESTART_ATTEMPT": "0",
          "CANON_POD_NAME": f"cpu-resolver-{arm}",
          "INJECTED_HF_TOKEN": "cpu-test-token",
          "INJECTED_WANDB_API_KEY": "cpu-test-key",
      })
      if wrong_profile:
        env["CANON_PROFILE_FILE"] = (
            "cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env"
        )
      if wrong_replay_path:
        env["CANON_APC_M15_REPLAY_LEDGER"] = str(
            state / "outside-capture.jsonl"
        )
      result = subprocess.run(
          ["bash", str(CANON / "cluster/steps/00_env.sh")],
          env=env,
          cwd=ROOT,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      resolved = state / "env.sh"
      resolved_text = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
      return result, resolved_text

  def test_off_arm_resolves_apc_off(self):
    result, resolved = self._resolve("off")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_APC_M15_TARGET_DEBUG=off", resolved)
    self.assertIn("export CANON_VLLM_ENABLE_PREFIX_CACHING=0", resolved)
    self.assertIn("export CANON_DP_SIZE=8", resolved)
    self.assertIn("export CANON_TP_SIZE=8", resolved)

  def test_on_arm_resolves_apc_on(self):
    result, resolved = self._resolve("on")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_APC_M15_TARGET_DEBUG=on", resolved)
    self.assertIn("export CANON_VLLM_ENABLE_PREFIX_CACHING=1", resolved)
    self.assertIn("export CANON_P38_DIAGNOSTIC_ROUNDS=1", resolved)
    self.assertIn("export CANON_APC_M15_REPLAY_LEDGER=", resolved)

  def test_wrong_profile_is_rejected_before_runtime(self):
    result, resolved = self._resolve("on", wrong_profile=True)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("requires its exact profile", result.stdout)

  def test_replay_ledger_outside_capture_is_rejected(self):
    result, resolved = self._resolve("on", wrong_replay_path=True)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("replay ledger must live in the capture directory", result.stdout)


if __name__ == "__main__":
  unittest.main()
