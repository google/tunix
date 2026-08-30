"""Tests for the P67 P45/M15 full-wave renderer and render-only wrapper."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO
    / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
    / "render_p67_frozenlake_two_full_recipes.py"
)
_PREPARE = (
    _REPO
    / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
    / "prepare_p67_frozenlake_two_full_wave.sh"
)
_SPEC = importlib.util.spec_from_file_location("v1_p67_two_renderer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
renderer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer)


def _pod(document: dict) -> dict:
  return document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]


def _worker_template(document: dict) -> dict:
  worker = next(
      job
      for job in document["spec"]["replicatedJobs"]
      if job["name"] == "pathways-worker"
  )
  return worker["template"]["spec"]["template"]


def _env(document: dict) -> dict[str, str]:
  main = next(
      item for item in _pod(document)["containers"] if item["name"] == "jax-tpu"
  )
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


class P67FrozenLakeTwoFullRendererTest(unittest.TestCase):

  def _render(self, output: Path):
    return renderer.render_two(
        source_commit="b" * 40,
        output_dir=output,
        p45_run_id="p45p67a",
        m15_run_id="m15p67a",
        campaign_root="v1p67-a",
        base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
    )

  def test_renders_exactly_two_scoped_full_recipes_without_topology_drift(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "rendered"
      outputs = self._render(root)
      self.assertEqual(len(outputs), 2)
      documents = [
          yaml.safe_load(path.read_text(encoding="utf-8")) for path in outputs
      ]
      envs = [_env(document) for document in documents]
      p45, m15 = envs
      for document, values in zip(documents, envs, strict=True):
        self.assertEqual(values["CANON_P33_SHARED_MESH"], "8,8")
        self.assertEqual(values["CANON_P57_TIM_ARM"], "zero")
        self.assertEqual(values["CANON_P57_EXPECTED_UPDATES"], "300")
        self.assertEqual(values["CANON_P59_CHECKED_VMA"], "1")
        self.assertEqual(values["CANON_P67_P66_VMA_P59_ONLY"], "1")
        self.assertEqual(values["CANON_V1_HP_FIRST_UPDATE_GATE"], "1")
        self.assertEqual(
            values["CANON_DP_COMPARE_MODE"], "fingerprint-hybrid"
        )
        self.assertEqual(
            values["CANON_DP_DISTINCT_SCHEDULE"], "first-group-warmup"
        )
        self.assertEqual(values["CANON_DP_FINITE_FETCH"], "batched-commit")
        self.assertEqual(values["CANON_P71_SCAN"], "fwd")
        self.assertNotIn("CANON_DP_COLLECTIVE_REDUCE", values)
        self.assertEqual(values["CANON_P33_ENABLE_EVAL"], "0")
        self.assertEqual(values["CANON_P33_DISABLE_EVAL"], "1")
        self.assertEqual(values["CANON_P31_ENABLE_EVAL"], "0")
        self.assertIn("--eval_every_n_steps=0", values["CANON_RUN_CMD"])
        self.assertNotIn("--num_test_batches=", values["CANON_RUN_CMD"])
        self.assertEqual(values["CANON_FROZENLAKE_CKPT_MODE"], "disabled")
        for name in (
            "CANON_FROZENLAKE_CKPT_ROOT",
            "CANON_FROZENLAKE_CKPT_TAG",
            "CANON_FROZENLAKE_CKPT_INTERVAL",
            "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP",
            "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
        ):
          self.assertEqual(values[name], "")
        worker = _worker_template(document)
        annotations = worker["metadata"]["annotations"]
        self.assertEqual(
            annotations["alpha.jobset.sigs.k8s.io/exclusive-topology"],
            "cloud.google.com/gke-nodepool",
        )
        selectors = worker["spec"]["nodeSelector"]
        self.assertEqual(selectors["cloud.google.com/gke-tpu-topology"], "4x4x4")
      self.assertEqual(p45["CANON_P57_WORKLOAD_CANDIDATE"], "")
      self.assertEqual(p45["CANON_P57_DATA_SPLIT"], "")
      self.assertEqual(p45["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], "0")
      self.assertNotIn("CANON_M15_TOKEN_CONTINUITY", p45)
      self.assertEqual(m15["CANON_P57_WORKLOAD_CANDIDATE"], "m15")
      self.assertEqual(m15["CANON_P57_DATA_SPLIT"], "main")
      self.assertEqual(m15["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], "1")
      self.assertEqual(m15["CANON_M15_TOKEN_CONTINUITY"], "exact")
      index = (root / "manifest-index.json").read_text(encoding="utf-8")
      self.assertIn('"schema": "v1-p67-frozenlake-two-full-v1"', index)

  def test_both_manifests_pass_real_env_resolution(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      outputs = self._render(root / "rendered")
      env_step = _REPO / "canon-zero-tim/cluster/steps/00_env.sh"
      for index, path in enumerate(outputs):
        values = _env(yaml.safe_load(path.read_text(encoding="utf-8")))
        state = root / f"state-{index}"
        state.mkdir()
        completed = subprocess.run(
            ["bash", str(env_step)],
            cwd=_REPO,
            env={
                **os.environ,
                **values,
                "CANON_PKG": str(_REPO / "canon-zero-tim"),
                "CANON_STATE": str(state),
                "JOBSET_RESTART_ATTEMPT": "0",
                "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{path}\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        snapshot = (state / "env.sh").read_text(encoding="utf-8")
        self.assertIn("export CANON_P59_CHECKED_VMA=1", snapshot)
        self.assertIn("export CANON_P66_P59_CHECK_VMA=1", snapshot)
        self.assertIn("export CANON_P67_P66_VMA_P59_ONLY=1", snapshot)
        self.assertIn(
            "export CANON_DP_COMPARE_MODE=fingerprint-hybrid", snapshot
        )
        self.assertIn(
            "export CANON_DP_DISTINCT_SCHEDULE=first-group-warmup", snapshot
        )
        self.assertIn("export CANON_DP_FINITE_FETCH=batched-commit", snapshot)
        self.assertIn("export CANON_P71_SCAN=fwd", snapshot)
        self.assertNotIn("CANON_DP_COLLECTIVE_REDUCE", snapshot)
        if index == 0:
          self.assertNotIn("CANON_M15_TOKEN_CONTINUITY", snapshot)
        else:
          self.assertIn(
              "export CANON_M15_TOKEN_CONTINUITY=exact", snapshot
          )
          self.assertIn(
              "[env] M15 exact TITO enabled mode=exact", completed.stdout
          )

  def test_wrong_profile_and_partial_scope_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      values = _env(
          yaml.safe_load(self._render(root / "rendered")[0].read_text(encoding="utf-8"))
      )
      env_step = _REPO / "canon-zero-tim/cluster/steps/00_env.sh"
      for label, mutation, expected_error in (
          (
              "wrong-profile",
              {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"},
              "P45 checkpoint-disabled mode is isolated to the optimized P45/M15 zero concept run",
          ),
          (
              "wrong-mesh",
              {"CANON_P33_SHARED_MESH": "16,4"},
              "P67 VMA scoping is restricted",
          ),
      ):
        with self.subTest(label=label):
          state = root / label
          state.mkdir()
          completed = subprocess.run(
              ["bash", str(env_step)],
              cwd=_REPO,
              env={
                  **os.environ,
                  **values,
                  **mutation,
                  "CANON_PKG": str(_REPO / "canon-zero-tim"),
                  "CANON_STATE": str(state),
                  "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
              },
              text=True,
              capture_output=True,
              check=False,
          )
          self.assertNotEqual(completed.returncode, 0)
          self.assertIn(expected_error, completed.stderr)

  def test_wrapper_is_render_only_and_emits_two_unpiped_launch_commands(self):
    script = _PREPARE.read_text(encoding="utf-8")
    self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD', script)
    self.assertIn("refusing to render from a dirty worktree", script)
    self.assertIn("V1_P67_FROZENLAKE_WAVE_READY", script)
    self.assertIn("launch=not-executed", script)
    self.assertEqual(script.count('"kubectl apply -f '), 2)
    self.assertFalse(
        any(line.strip().startswith("kubectl apply") for line in script.splitlines())
    )
    completed = subprocess.run(
        ["bash", "-n", str(_PREPARE)],
        cwd=_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, msg=completed.stderr)

  def test_refuses_reused_output_and_duplicate_ids(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "rendered"
      self._render(root)
      with self.assertRaises(FileExistsError):
        self._render(root)
      with self.assertRaisesRegex(ValueError, "distinct"):
        renderer.render_two(
            source_commit="b" * 40,
            output_dir=Path(tmp) / "duplicate",
            p45_run_id="same",
            m15_run_id="same",
            campaign_root="v1p67-b",
            base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
        )


if __name__ == "__main__":
  unittest.main()
