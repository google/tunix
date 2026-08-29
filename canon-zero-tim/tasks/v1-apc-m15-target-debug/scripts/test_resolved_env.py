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
      observer: str = "none",
      seam_layer: int | None = None,
      wrong_profile: bool = False,
      wrong_replay_path: bool = False,
      wrong_incident_bound: bool = False,
      wrong_seam_bound: bool = False,
      wrong_workload_identity: bool = False,
      wrong_entrypoint: bool = False,
  ):
    run_id = f"cpu-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="v1-apc-render-", dir="/tmp") as output:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(output),
          source_commit=SOURCE,
          run_id=run_id,
          observer=observer,
          seam_layer=seam_layer,
      )
      expected_suffix = f"-{arm}" if observer == "none" else f"-{arm}-{observer}"
      path = next(path for path in paths if path.stem.endswith(expected_suffix))
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
      if wrong_incident_bound:
        env["CANON_P38_INCIDENT_MAX_BYTES"] = "268435456"
      if wrong_seam_bound:
        env["CANON_P38_SEAM_MIN_POSITION"] = "1400"
      if wrong_workload_identity:
        env["CANON_P57_DATA_SPLIT"] = "selection"
      if wrong_entrypoint:
        env["CANON_RUN_CMD"] = env["CANON_RUN_CMD"].replace(
            "-m examples.frozenlake.train_frozenlake_qwen3",
            "examples/frozenlake/train_frozenlake_qwen3.py",
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
    self.assertIn("export CANON_P57_WORKLOAD_CANDIDATE=m15", resolved)
    self.assertIn("export CANON_P57_DATA_SPLIT=main", resolved)
    self.assertIn("export CANON_CONTINUE_DECODE=8", resolved)
    self.assertIn("export CANON_P38_INCIDENT_MAX_BYTES=2147483648", resolved)

  def test_on_arm_resolves_apc_on(self):
    result, resolved = self._resolve("on")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_APC_M15_TARGET_DEBUG=on", resolved)
    self.assertIn("export CANON_VLLM_ENABLE_PREFIX_CACHING=1", resolved)
    self.assertIn("export CANON_P38_DIAGNOSTIC_ROUNDS=1", resolved)
    self.assertIn("export CANON_APC_M15_REPLAY_LEDGER=", resolved)

  def test_layer_observer_resolves_with_m15_bounds_and_tail(self):
    result, resolved = self._resolve("on", observer="layer")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_P38_SEAM_OBSERVER=layer", resolved)
    self.assertIn("export CANON_P38_SEAM_MIN_POSITION=960", resolved)
    self.assertIn("export CANON_P38_SEAM_MAX_POSITION=4096", resolved)
    self.assertIn("export CANON_P38_SEAM_MAX_BYTES=8589934592", resolved)
    self.assertIn("export CANON_P38_TAIL_OBSERVER=1", resolved)
    self.assertIn("export CANON_P38_DURABILITY_PROFILE=m15-wide-v1", resolved)
    self.assertIn("export CANON_P38_DIAGNOSTIC_ROUNDS=3", resolved)

  def test_full_observer_resolves_with_exact_layer(self):
    result, resolved = self._resolve("off", observer="full", seam_layer=17)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_P38_SEAM_OBSERVER=full", resolved)
    self.assertIn("export CANON_P38_SEAM_LAYER=17", resolved)
    self.assertNotIn("export CANON_P38_TAIL_OBSERVER=", resolved)
    self.assertIn("export CANON_P38_DURABILITY_PROFILE=m15-wide-v1", resolved)
    self.assertIn("export CANON_P38_DIAGNOSTIC_ROUNDS=3", resolved)

  def test_targeted_kv_observer_resolves_with_exact_alias_contract(self):
    result, resolved = self._resolve("on", observer="kv")
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("export CANON_P38_KV_OBSERVER_LAYER=0", resolved)
    self.assertIn(
        "export CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS=1226", resolved
    )
    self.assertIn("export CANON_P38_KV_OBSERVER_MAX_CANDIDATES=8", resolved)
    self.assertIn("export CANON_P38_KV_OBSERVER_MAX_PAGES=96", resolved)
    self.assertIn("export CANON_P38_DURABILITY_PROFILE=round-alignment-v1", resolved)
    self.assertIn("export CANON_P38_DIAGNOSTIC_ROUNDS=1", resolved)
    self.assertNotIn("export CANON_P38_SEAM_OBSERVER=", resolved)

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

  def test_legacy_incident_bound_is_rejected(self):
    result, resolved = self._resolve("off", wrong_incident_bound=True)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("incident ledger bounds drifted", result.stdout)

  def test_legacy_seam_bound_is_rejected(self):
    result, resolved = self._resolve(
        "on", observer="layer", wrong_seam_bound=True
    )
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("seam observer bounds drifted", result.stdout)

  def test_cli_and_signed_workload_identity_cannot_diverge(self):
    result, resolved = self._resolve("off", wrong_workload_identity=True)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("requires signed m15/main workload identity", result.stdout)

  def test_file_path_entrypoint_is_rejected_before_runtime(self):
    result, resolved = self._resolve("off", wrong_entrypoint=True)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertFalse(resolved)
    self.assertIn("requires the package-safe module entrypoint", result.stdout)


if __name__ == "__main__":
  unittest.main()
