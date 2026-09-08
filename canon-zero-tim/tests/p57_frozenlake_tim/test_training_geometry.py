"""New full geometry must reach real admission without weakening legacy gates."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "canon-zero-tim/cluster"))
from examples.frozenlake import training_geometry as geometry
import render_p57_frozenlake_tim as p57


class TrainingGeometryTest(unittest.TestCase):

  def _render(self, directory, candidate="", mode=geometry.SMALL):
    return p57.render_all(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        output_dir=directory, source_commit="a" * 40, run_id="t9g",
        campaign_tag="t9g", checkpoint_mode="disabled", expected_updates=300,
        arm="zero", high_performance=True, disable_eval=True,
        workload_candidate=candidate, data_split="main" if candidate else "",
        train_geometry=mode,
    )[0]

  def _preflight(self, env, state):
    state.mkdir()
    return subprocess.run(
        ["bash", "cluster/steps/00_env.sh"], cwd=ROOT / "canon-zero-tim",
        env={**os.environ, **env, "CANON_PKG": str(ROOT / "canon-zero-tim"),
             "CANON_STATE": str(state), "INJECTED_HF_TOKEN": "test-token",
             "INJECTED_WANDB_API_KEY": "test-key"},
        capture_output=True, text=True, check=False,
    )

  def test_new_p45_m15_resolve_the_same_closed_shape(self):
    for candidate in ("", "m15"):
      with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as tmp:
        path = self._render(Path(tmp) / "render", candidate)
        doc = yaml.safe_load(path.read_text())
        env = p57._env(doc)
        expected = geometry.geometry(geometry.SMALL)
        for key, value in expected.environment().items():
          self.assertEqual(env[key], value, key)
        cmd = env["CANON_RUN_CMD"].split()
        for arg in ("--mesh_dp=4", "--mesh_tp=8", "--batch_size=16",
                    "--mini_batch_size=16", "--num_generations=8",
                    "--train_trajectory_micro_batch_size=4", "--max_concurrency=128",
                    "--vllm_max_num_seqs=32", "--vllm_max_num_batched_tokens=256"):
          self.assertEqual(cmd.count(arg), 1, arg)
        worker = doc["spec"]["replicatedJobs"][1]["template"]["spec"]
        self.assertEqual((worker["parallelism"], worker["completions"]), (8, 8))
        self.assertEqual(worker["template"]["spec"]["nodeSelector"][
            "cloud.google.com/gke-tpu-topology"], "2x4x4")
        result = self._preflight(env, Path(tmp) / "state")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_selector_is_closed(self):
    for value in ("", "0", "dp2-tp8-b128", geometry.LEGACY):
      with self.assertRaises(ValueError):
        geometry.from_env({geometry.SELECTOR: value})
    self.assertEqual(geometry.from_env({}).name, geometry.LEGACY)

  def test_invalid_raw_tuple_cannot_be_repaired_by_profile(self):
    with tempfile.TemporaryDirectory() as tmp:
      env = p57._env(yaml.safe_load(self._render(Path(tmp) / "render").read_text()))
      for i, (key, value) in enumerate((
          ("CANON_DP_SIZE", "8"), ("CANON_GLOBAL_PROMPTS", "32"),
          ("CANON_GLOBAL_TRAJECTORIES", "256"), ("MIN_TOKEN_BUCKET", "2048"),
          ("CANON_TP_SIZE", "4"), ("CANON_LOCAL_TRAJECTORIES", "16"),
          ("CANON_PROFILE_FILE", geometry.geometry().profile_file),
          ("CANON_P57_TIM_ARM", "mismatch"),
      )):
        with self.subTest(key=key):
          result = self._preflight({**env, key: value}, Path(tmp) / f"bad-{i}")
          self.assertNotEqual(result.returncode, 0)
          self.assertIn("P57.GEOMETRY", result.stderr)

  def test_new_command_shape_and_duplicates_fail_before_runtime(self):
    with tempfile.TemporaryDirectory() as tmp:
      env = p57._env(yaml.safe_load(self._render(Path(tmp) / "render").read_text()))
      cmd = env["CANON_RUN_CMD"]
      wrong_commands = [
          cmd.replace("--batch_size=16", "--batch_size=32"),
          cmd.replace("--mini_batch_size=16", "--mini_batch_size=32"),
          cmd.replace("--max_concurrency=128", "--max_concurrency=256"),
          cmd + " --batch_size=16",
          cmd.replace("--num_generations=8", "--num_generations=4"),
          cmd.replace("--train_trajectory_micro_batch_size=4", "--train_trajectory_micro_batch_size=8"),
      ]
      for i, command in enumerate(wrong_commands):
        result = self._preflight({**env, "CANON_RUN_CMD": command}, Path(tmp) / f"cmd-{i}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("P57.GEOMETRY", result.stderr)


if __name__ == "__main__":
  unittest.main()
