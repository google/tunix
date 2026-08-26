"""Tests for the V1 high-performance three-full renderer."""

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
    / "render_three_full_recipes.py"
)
_PREPARE = (
    _REPO
    / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
    / "prepare_checked_vma_three_full_wave.sh"
)
_SPEC = importlib.util.spec_from_file_location("v1_phase4_renderer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
renderer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer)


def _env(document: dict) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  main = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


class ThreeFullRendererTest(unittest.TestCase):

  def _render(self, root: Path):
    return renderer.render_three(
        source_commit="a" * 40,
        output_dir=root,
        gsm8k_run_id="g64a",
        p45_run_id="f45a",
        m15_run_id="m15a",
        campaign_root="v1hp-a",
        base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
    )

  def test_exact_image_gate_imports_the_mounted_workspace(self):
    script = (
        _REPO / "canon-zero-tim/tests/v1_phase4/run_exact_image.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('-v "$root:/workspace:ro"', script)
    self.assertIn("-e PYTHONPATH=/workspace", script)

  def test_wave_wrapper_is_render_only_and_binds_clean_head(self):
    script = _PREPARE.read_text(encoding="utf-8")
    self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD', script)
    self.assertIn(
        'git -C "$REPO_ROOT" status --porcelain --untracked-files=all',
        script,
    )
    self.assertIn("refusing to render from a dirty worktree", script)
    self.assertIn("V1_HP_CHECKED_VMA_WAVE_READY", script)
    self.assertIn("launch=not-executed", script)
    self.assertEqual(script.count('"kubectl apply -f '), 3)
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

  def test_renders_exactly_three_strict_full_recipes(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "rendered"
      outputs = self._render(root)
      self.assertEqual(len(outputs), 3)
      documents = [
          yaml.safe_load(path.read_text(encoding="utf-8")) for path in outputs
      ]
      envs = [_env(document) for document in documents]

      gsm8k, p45, m15 = envs
      self.assertEqual(gsm8k["CANON_PROFILE_FILE"], renderer._GSM8K_PROFILE)
      self.assertEqual(gsm8k["CANON_P33_SHARED_MESH"], "16,4")
      self.assertEqual(gsm8k["CANON_GSM8K_ALIGNMENT_WARN_ONLY"], "0")
      self.assertEqual(gsm8k["CANON_P38_FIXED_LM_HEAD"], "1")
      self.assertIn("--max_steps=200", gsm8k["CANON_RUN_CMD"])

      for frozen in (p45, m15):
        self.assertEqual(frozen["CANON_PROFILE_FILE"], renderer.p57._V1_HP_PROFILE)
        self.assertEqual(frozen["CANON_P33_SHARED_MESH"], "8,8")
        self.assertEqual(frozen["CANON_P57_TIM_ARM"], "zero")
        self.assertEqual(frozen["CANON_P57_EXPECTED_UPDATES"], "300")
        self.assertEqual(frozen["CANON_P33_ENABLE_EVAL"], "1")
        self.assertEqual(frozen["CANON_P33_DISABLE_EVAL"], "0")
        self.assertEqual(frozen["CANON_P31_ENABLE_EVAL"], "1")
        self.assertEqual(frozen["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], "0")
        self.assertEqual(
            frozen["CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL"], "0"
        )
        self.assertIn("--eval_every_n_steps=50", frozen["CANON_RUN_CMD"])
        self.assertIn("--num_test_batches=4", frozen["CANON_RUN_CMD"])
      self.assertNotIn("--p57_workload_candidate=", p45["CANON_RUN_CMD"])
      self.assertIn("--p57_workload_candidate=m15", m15["CANON_RUN_CMD"])
      self.assertIn("--p57_data_split=main", m15["CANON_RUN_CMD"])
      self.assertIn("--max_response_length=8192", m15["CANON_RUN_CMD"])

      for values in envs:
        self.assertEqual(values["CANON_V1_HP_FULL"], "1")
        self.assertEqual(values["CANON_P59_RANK_PARALLEL_BACKWARD"], "1")
        self.assertEqual(values["CANON_P59_CHECKED_VMA"], "1")
        self.assertEqual(values["CANON_V1_HP_FIRST_UPDATE_GATE"], "1")
        self.assertEqual(values["CANON_P33_RUN_STAGE"], "full")
        self.assertEqual(values["CANON_P33_NO_COMMIT"], "0")
        self.assertEqual(
            values["JAX_COMPILATION_CACHE_DIR"],
            "/tmp/jax_compilation_cache",
        )
        self.assertEqual(
            values["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"], "0"
        )
        self.assertEqual(
            values["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "all"
        )
        self.assertEqual(
            values["CANON_GCS_CACHE_BUCKET"],
            "gs://yuxzhang-tunix-models/cache/p33_compilation_cache",
        )

  def test_profiles_resolve_complete_workload_scoped_bundle(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = self._render(Path(tmp) / "rendered")
      for path in outputs:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        values = _env(document)
        profile = _REPO / "canon-zero-tim" / values["CANON_PROFILE_FILE"]
        command = (
            f"source {profile}; "
            "test \"$CANON_CONTINUE_DECODE\" = 8; "
            "test \"$CANON_FIXED_AR_GATHER\" = 1; "
            "test \"$CANON_PALLAS_GATHERED_LOGPROBS\" = 1; "
            "test \"$CANON_LOGPROB_STEP_FUSION\" = 1; "
            "test \"$CANON_P59_RANK_PARALLEL_BACKWARD\" = 1; "
            "test \"$CANON_P59_CHECKED_VMA\" = 1; "
            "test \"$CANON_V1_HP_FIRST_UPDATE_GATE\" = 1; "
            "test \"$CANON_P63_OVERFLOW_SAFE_CLIP\" = 1; "
            "test \"$CANON_P28_BATCHED_REPORT\" = 1; "
            "test \"$CANON_XPROF_PHASE\" = update; "
            "test \"$CANON_XPROF_SKIP_STEPS\" = 2; "
            "test \"$CANON_XPROF_STEPS\" = 1"
            "; test \"$CANON_XPROF_LABELS\" = 1"
            "; test \"$CANON_PERF_TRACE_EXPORT_STEP\" = 2"
            "; test \"$CANON_VLLM_ENABLE_PREFIX_CACHING\" = 0"
            "; test \"$JAX_COMPILATION_CACHE_DIR\" = /tmp/jax_compilation_cache"
            "; test \"$JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS\" = 0"
            "; test \"$JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES\" = all"
            "; test \"$CANON_GCS_CACHE_BUCKET\" = "
            "gs://yuxzhang-tunix-models/cache/p33_compilation_cache"
            "; test \"$CANON_XPROF_DIR\" = "
            "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/"
            "${CANON_STATE##*/}/attempt-direct/xprof-update"
        )
        completed = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=_REPO,
            env={**os.environ, **values},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"{path}\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        resolved = subprocess.run(
            [
                "bash", "-euo", "pipefail", "-c",
                f"source {profile}; "
                "test \"$CANON_VLLM_ENABLE_PREFIX_CACHING\" = 0",
            ],
            cwd=_REPO,
            env={**os.environ, **values},
            check=False,
        )
        self.assertEqual(resolved.returncode, 0)

      frozen_profile = (
          _REPO
          / "canon-zero-tim/cluster/profiles/"
          "qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
      )
      m15_values = _env(
          yaml.safe_load(outputs[2].read_text(encoding="utf-8"))
      )
      wrong_m15 = subprocess.run(
          [
              "bash", "-euo", "pipefail", "-c",
              f"source {frozen_profile}",
          ],
          cwd=_REPO,
          env={
              **os.environ,
              **m15_values,
              "CANON_P57_WORKLOAD_CANDIDATE": "m15",
              "CANON_P57_DATA_SPLIT": "selection",
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(wrong_m15.returncode, 0)
      self.assertTrue(wrong_m15.stderr)

  def test_p63_rejects_neighboring_stock_profile(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      outputs = self._render(root / "rendered")
      values = _env(yaml.safe_load(outputs[0].read_text(encoding="utf-8")))
      values["CANON_PROFILE_FILE"] = (
          "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env"
      )
      values["CANON_P63_OVERFLOW_SAFE_CLIP"] = "1"
      state = root / "wrong-profile"
      state.mkdir()
      completed = subprocess.run(
          [
              "bash",
              str(_REPO / "canon-zero-tim/cluster/steps/00_env.sh"),
          ],
          cwd=_REPO,
          env={
              **os.environ,
              **values,
              "CANON_PKG": str(_REPO / "canon-zero-tim"),
              "CANON_STATE": str(state),
              "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(completed.returncode, 0)
      self.assertIn("P63 overflow-safe clip is restricted", completed.stderr)

  def test_refuses_reused_output_and_duplicate_ids(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "rendered"
      self._render(root)
      with self.assertRaises(FileExistsError):
        self._render(root)
      with self.assertRaisesRegex(ValueError, "distinct"):
        renderer.render_three(
            source_commit="a" * 40,
            output_dir=Path(tmp) / "duplicate",
            gsm8k_run_id="same",
            p45_run_id="same",
            m15_run_id="m15b",
            campaign_root="v1hp-b",
            base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
        )

  def test_p57_high_performance_rejects_nonzero_or_short_job(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(ValueError, "requires a new 300-update zero"):
        renderer.p57.render_all(
            base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
            output_dir=Path(tmp) / "bad",
            source_commit="a" * 40,
            run_id="bad1",
            campaign_tag="v1hp-bad",
            checkpoint_mode="new",
            expected_updates=200,
            run_kind="train",
            arm="mismatch",
          high_performance=True,
        )

  def test_all_three_pass_real_env_resolution_and_partial_bundle_fails(self):
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
        self.assertIn("export CANON_V1_HP_FULL=1", snapshot)
        self.assertIn("export CANON_P59_RANK_PARALLEL_BACKWARD=1", snapshot)
        self.assertIn("export CANON_P59_CHECKED_VMA=1", snapshot)
        self.assertIn("export CANON_P66_P59_CHECK_VMA=1", snapshot)
        self.assertIn("export CANON_V1_HP_FIRST_UPDATE_GATE=1", snapshot)
        self.assertIn("export CANON_P63_OVERFLOW_SAFE_CLIP=1", snapshot)
        self.assertIn("export CANON_PERF_TRACE_EXPORT_STEP=2", snapshot)
        expected_xprof = (
            "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/"
            f"{state.name}/attempt-0/xprof-update"
        )
        self.assertIn(f"export CANON_XPROF_DIR={expected_xprof}", snapshot)

      gsm_values = _env(
          yaml.safe_load(outputs[0].read_text(encoding="utf-8"))
      )
      bad_state = root / "bad-state"
      bad_state.mkdir()
      rejected = subprocess.run(
          ["bash", str(env_step)],
          cwd=_REPO,
          env={
              **os.environ,
              **gsm_values,
              "CANON_PKG": str(_REPO / "canon-zero-tim"),
              "CANON_STATE": str(bad_state),
              "CANON_P33_NO_COMMIT": "1",
              "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(rejected.returncode, 0)
      self.assertIn("GSM8K v1-hp requires", rejected.stderr)


if __name__ == "__main__":
  unittest.main()
