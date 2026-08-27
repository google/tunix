"""Renderer contracts for P57 calibration and frozen-recipe training."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[3]
CLUSTER = ROOT / "canon-zero-tim/cluster"
if str(CLUSTER) not in sys.path:
  sys.path.insert(0, str(CLUSTER))


def _module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


calibration = _module("p57_calibration_renderer", CLUSTER / "render_p57_calibration.py")
paired = _module("p57_paired_renderer", CLUSTER / "render_p57_frozenlake_tim.py")
THREE_ARM_WRAPPER = (
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts"
    / "render_three_arm_wave.sh"
)
manifest_preflight = _module(
    "p57_calibration_manifest_preflight",
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/verify_calibration_manifest.py",
)
BASE = CLUSTER / "jobset-64chip.yaml"


def _env(document):
  return {item["name"]: item.get("value") for item in _container_env(document)}


def _container_env(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
  container = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
  return container["env"]


def _run_env_preflight(rendered_env, state: Path):
  state.mkdir()
  return subprocess.run(
      ["bash", "cluster/steps/00_env.sh"],
      cwd=ROOT / "canon-zero-tim",
      env={
          **os.environ,
          **{k: v for k, v in rendered_env.items() if v is not None},
          "CANON_PKG": str(ROOT / "canon-zero-tim"),
          "CANON_STATE": str(state),
          "INJECTED_HF_TOKEN": "test-token",
          "INJECTED_WANDB_API_KEY": "test-key",
      },
      text=True,
      capture_output=True,
      check=False,
  )


class P57RendererTest(unittest.TestCase):

  def test_calibration_renders_one_stock_stochastic_no_update_job(self):
    with tempfile.TemporaryDirectory() as tmp:
      paths = calibration.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )
      self.assertEqual(len(paths), 1)
      documents = [yaml.safe_load(path.read_text()) for path in paths]
    modes = {_env(document)["CANON_P57_CALIBRATION_MODE"] for document in documents}
    self.assertEqual(modes, {"stochastic"})
    for document in documents:
      env = _env(document)
      self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], "0")
      self.assertEqual(env["CANON_P57_TIM_ARM"], "mismatch")
      self.assertEqual(env["CANON_P57_RUN_KIND"], "calibration")
      self.assertEqual(env["CANON_P57_INFERENCE_REGIME"], "stock-fast")
      for name in (
          "CANON_P32_TRAIN_ADMITTED",
          "CANON_P32_DP_REDUCTION_ADMITTED",
          "CANON_P33_WORKLOAD_LAUNCH_ADMITTED",
          "CANON_PRE_ALIGN_GATE",
          "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY",
      ):
        self.assertEqual(env[name], "0")
      self.assertEqual(env["CANON_P57_CALIBRATION_RECIPES"], "m10,m15,m20")
      self.assertEqual(
          env["CANON_RUN_CMD"].split()[:4],
          [
              "python3",
              "-u",
              "-m",
              "examples.frozenlake.train_frozenlake_qwen3",
          ],
      )
      self.assertIn("--evaluation_only", env["CANON_RUN_CMD"])
      self.assertIn("--num_generations=8", env["CANON_RUN_CMD"])
      self.assertIn("--temperature=0.7", env["CANON_RUN_CMD"])
      self.assertIn("--max_prompt_length=16384", env["CANON_RUN_CMD"])
      self.assertIn("--max_response_length=16384", env["CANON_RUN_CMD"])
      result = subprocess.run(
          [
              "bash",
              "-c",
              "set -euo pipefail; "
              "source cluster/profiles/_canonical_engine.env; "
              "source \"$CANON_PROFILE_FILE\"; "
              "printf 'P57_CAL_PROFILE_PASS mode=%s\\n' "
              "\"$CANON_P57_CALIBRATION_MODE\"; env",
          ],
          cwd=ROOT / "canon-zero-tim",
          env={**os.environ, **{k: v for k, v in env.items() if v is not None}},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("P57_CAL_PROFILE_PASS", result.stdout)
      resolved = dict(
          line.split("=", 1)
          for line in result.stdout.splitlines()
          if "=" in line
      )
      for name in (
          "CANON_FIXED_AR",
          "CANON_FIXED_AR_EMBED",
          "CANON_RPA_D",
          "CANON_RPA_P",
          "CANON_RPA_M",
          "CANON_LOGPROB_M",
          "CANON_PALLAS_ALL_PROJ",
          "CANON_PALLAS_ALL_RMSNORM",
          "CANON_PALLAS_SWIGLU",
          "CANON_PALLAS_MPAD",
          "CANON_PALLAS_SWIGLU_MPAD",
          "CANON_PALLAS_CANONICAL_VJP",
      ):
        self.assertNotIn(name, resolved)
      for name in (
          "CANON_RPA_VJP2",
          "CANON_PROMPT_PROCESSED_LOGPROBS",
          "CANON_PALLAS_LOGSOFTMAX",
          "CANON_ENGINE_MODULE_C",
          "CANON_FROZENLAKE_L3",
          "CANON_ALIGNMENT_GATE",
          "CANON_P38_FIXED_LM_HEAD",
      ):
        self.assertEqual(resolved[name], "0")
      self.assertNotIn(
          "--xla_allow_excess_precision=false", resolved["XLA_FLAGS"]
      )

  def test_calibration_profile_rejects_greedy_or_l0_inventory(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = calibration.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
    for name, value in (
        ("CANON_P57_CALIBRATION_MODE", "greedy"),
        ("CANON_P57_CALIBRATION_RECIPES", "l0,m10,m15,m20"),
        ("CANON_P57_INFERENCE_REGIME", "canonical"),
        (
            "CANON_RUN_CMD",
            env["CANON_RUN_CMD"].replace(
                "-m examples.frozenlake.train_frozenlake_qwen3",
                "examples/frozenlake/train_frozenlake_qwen3.py",
            ),
        ),
    ):
      with self.subTest(name=name):
        bad_env = {**env, name: value}
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail; source \"$CANON_PROFILE_FILE\""],
            cwd=ROOT / "canon-zero-tim",
            env={**os.environ, **{k: v for k, v in bad_env.items() if v is not None}},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

  def test_calibration_resolved_env_proves_zero_tim_bundle_is_off(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      path = calibration.render_all(
          base_path=BASE,
          output_dir=root / "rendered",
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
      state = root / "state"
      state.mkdir()
      result = subprocess.run(
          ["bash", "cluster/steps/00_env.sh"],
          cwd=ROOT / "canon-zero-tim",
          env={
              **os.environ,
              **{k: v for k, v in env.items() if v is not None},
              "CANON_PKG": str(ROOT / "canon-zero-tim"),
              "CANON_STATE": str(state),
              "INJECTED_HF_TOKEN": "test-token",
              "INJECTED_WANDB_API_KEY": "test-key",
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn(
          "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25",
          result.stdout,
      )
      resolved = (state / "env.sh").read_text(encoding="utf-8")
      self.assertNotIn("CANON_FIXED_AR=", resolved)
      self.assertNotIn("CANON_PALLAS_ALL_PROJ=", resolved)
      self.assertIn("export CANON_ENGINE_MODULE_C=0", resolved)
      self.assertIn("export CANON_P38_FIXED_LM_HEAD=0", resolved)

  def test_calibration_refuses_overwrite_and_bad_tag(self):
    with tempfile.TemporaryDirectory() as tmp:
      args = dict(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )
      calibration.render_all(**args)
      with self.assertRaises(FileExistsError):
        calibration.render_all(**args)
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(ValueError, "campaign tag"):
        calibration.render_all(
            base_path=BASE,
            output_dir=Path(tmp),
            source_commit="a" * 40,
            run_id="p57cal",
            campaign_tag="Bad/Tag",
        )

  def test_manifest_preflight_rejects_regime_drift(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      path = calibration.render_all(
          base_path=BASE,
          output_dir=root / "good",
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )[0]
      self.assertEqual(manifest_preflight.verify(path)["regime"], "stock-fast")
      document = yaml.safe_load(path.read_text())
      pod_env = _env(document)
      self.assertEqual(pod_env["CANON_P57_INFERENCE_REGIME"], "stock-fast")
      for item in _container_env(document):
        if item["name"] == "CANON_P57_INFERENCE_REGIME":
          item["value"] = "canonical"
      bad = root / "bad.yaml"
      bad.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "manifest drifted"):
        manifest_preflight.verify(bad)

  def test_manifest_preflight_rejects_file_path_entrypoint(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      path = calibration.render_all(
          base_path=BASE,
          output_dir=root / "good",
          source_commit="a" * 40,
          run_id="p57cal",
          campaign_tag="p57-calibration",
      )[0]
      document = yaml.safe_load(path.read_text())
      for item in _container_env(document):
        if item["name"] == "CANON_RUN_CMD":
          item["value"] = item["value"].replace(
              "-m examples.frozenlake.train_frozenlake_qwen3",
              "examples/frozenlake/train_frozenlake_qwen3.py",
          )
      bad = root / "bad-entrypoint.yaml"
      bad.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "manifest drifted"):
        manifest_preflight.verify(bad)

  def test_selected_main_recipe_can_render_stock_or_native_training(self):
    for stock_only, expected_count in ((True, 1), (False, 1)):
      with self.subTest(stock_only=stock_only), tempfile.TemporaryDirectory() as tmp:
        data_split = "selection" if stock_only else "main"
        expected_updates = 200 if stock_only else 300
        paths = paired.render_all(
            base_path=BASE,
            output_dir=Path(tmp),
            source_commit="a" * 40,
            run_id="p57main",
            campaign_tag="p57-m15-main",
            checkpoint_mode="new",
            expected_updates=expected_updates,
            workload_candidate="m15",
            data_split=data_split,
            stock_only=stock_only,
            arm="" if stock_only else "mismatch",
        )
        self.assertEqual(len(paths), expected_count)
        for path in paths:
          env = _env(yaml.safe_load(path.read_text()))
          self.assertEqual(env["CANON_P57_WORKLOAD_CANDIDATE"], "m15")
          self.assertEqual(env["CANON_P57_DATA_SPLIT"], data_split)
          self.assertEqual(
              env["CANON_RUN_CMD"].split()[:4],
              [
                  "python3",
                  "-u",
                  "-m",
                  "examples.frozenlake.train_frozenlake_qwen3",
              ],
          )
          self.assertIn("--env_max_steps=15", env["CANON_RUN_CMD"])
          self.assertIn("--max_prompt_length=4096", env["CANON_RUN_CMD"])
          self.assertIn("--max_response_length=8192", env["CANON_RUN_CMD"])
          self.assertEqual(
              env["CANON_RUN_CMD"].split().count("--sampler_is=none"), 1
          )
          self.assertEqual(
              env["CANON_P57_STOP_AFTER_STEP"], str(expected_updates)
          )
          self.assertEqual(
              env["CANON_FROZENLAKE_CKPT_INTERVAL"],
              "10" if stock_only else "300",
          )
          self.assertEqual(env["CANON_FROZENLAKE_CKPT_MAX_TO_KEEP"], "1")
          if stock_only:
            self.assertEqual(env["CANON_P57_INFERENCE_REGIME"], "stock-fast")

  def test_stock_curve_renders_registered_segment_and_eval(self):
    with tempfile.TemporaryDirectory() as tmp:
      train_path = paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp) / "train",
          source_commit="a" * 40,
          run_id="p57stock50",
          campaign_tag="p57-m15-selection",
          checkpoint_mode="new",
          expected_updates=200,
          run_kind="train",
          workload_candidate="m15",
          data_split="selection",
          stock_only=True,
          stop_after_step=50,
      )[0]
      eval_path = paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp) / "eval",
          source_commit="a" * 40,
          run_id="p57stockeval0",
          campaign_tag="p57-m15-selection",
          checkpoint_mode="new",
          expected_updates=200,
          run_kind="eval",
          checkpoint_step=0,
          workload_candidate="m15",
          data_split="selection",
          stock_only=True,
      )[0]
      train_env = _env(yaml.safe_load(train_path.read_text()))
      eval_env = _env(yaml.safe_load(eval_path.read_text()))
      train_preflight = _run_env_preflight(train_env, Path(tmp) / "train-state")
      eval_preflight = _run_env_preflight(eval_env, Path(tmp) / "eval-state")
    self.assertEqual(train_env["CANON_P57_STOP_AFTER_STEP"], "50")
    self.assertEqual(train_env["CANON_FROZENLAKE_CKPT_INTERVAL"], "10")
    self.assertIn("--max_steps=200", train_env["CANON_RUN_CMD"])
    self.assertEqual(eval_env["CANON_P57_INFERENCE_REGIME"], "stock-fast")
    self.assertEqual(eval_env["CANON_P57_EVAL_CHECKPOINT_STEP"], "0")
    self.assertIn("--evaluation_only", eval_env["CANON_RUN_CMD"])
    self.assertIn("--num_generations=8", eval_env["CANON_RUN_CMD"])
    self.assertNotIn("--num_generations=2", eval_env["CANON_RUN_CMD"])
    self.assertIn("--max_prompt_length=4096", eval_env["CANON_RUN_CMD"])
    self.assertIn("--max_response_length=8192", eval_env["CANON_RUN_CMD"])
    self.assertEqual(
        train_env["CANON_RUN_CMD"].split().count("--sampler_is=none"), 1
    )
    self.assertEqual(
        eval_env["CANON_RUN_CMD"].split().count("--sampler_is=none"), 1
    )
    self.assertEqual(train_preflight.returncode, 0, train_preflight.stderr)
    self.assertEqual(eval_preflight.returncode, 0, eval_preflight.stderr)
    self.assertIn(
        "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train absent=12 "
        "observer=train processed_b=on",
        train_preflight.stdout,
    )
    self.assertIn(
        "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=eval absent=12 observer=off",
        eval_preflight.stdout,
    )

  def test_stock_discovery_defaults_to_one_full_horizon_run(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57stockfull",
          campaign_tag="p57-m15-selection",
          checkpoint_mode="new",
          expected_updates=200,
          run_kind="train",
          workload_candidate="m15",
          data_split="selection",
          stock_only=True,
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
      preflight = _run_env_preflight(env, Path(tmp) / "state")
    self.assertEqual(env["CANON_P57_STOP_AFTER_STEP"], "200")
    self.assertIn("--max_steps=200", env["CANON_RUN_CMD"])
    self.assertNotIn("--evaluation_only", env["CANON_RUN_CMD"])
    self.assertIn("--sampler_is=none", env["CANON_RUN_CMD"])
    self.assertEqual(preflight.returncode, 0, preflight.stderr)
    self.assertIn(
        "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train absent=12 "
        "observer=train processed_b=on",
        preflight.stdout,
    )

  def test_p57_training_rejects_token_sampler_correction(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp) / "rendered",
          source_commit="a" * 40,
          run_id="p57pure",
          campaign_tag="p57-m15-selection",
          checkpoint_mode="new",
          expected_updates=200,
          run_kind="train",
          workload_candidate="m15",
          data_split="selection",
          stock_only=True,
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
      env["CANON_RUN_CMD"] = env["CANON_RUN_CMD"].replace(
          "--sampler_is=none", "--sampler_is=token"
      )
      preflight = _run_env_preflight(env, Path(tmp) / "state")
    self.assertNotEqual(preflight.returncode, 0)
    self.assertIn(
        "P57 command/horizon contract drifted",
        preflight.stderr,
    )

  def test_two_workloads_render_all_three_registered_treatments(self):
    workloads = (
        ("p45", "", "", 300, 5, 2048),
        ("m15", "m15", "main", 300, 15, 8192),
    )
    arms = (
        ("mismatch", "none", "stock-fast", "0", "1"),
        ("is", "token", "stock-fast", "0", "1"),
        ("zero", "none", "", "1", "0"),
    )
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for workload, candidate, split, updates, turns, response in workloads:
        for arm, sampler, regime, fixed_head, warning_only in arms:
          with self.subTest(workload=workload, arm=arm):
            path = paired.render_all(
                base_path=BASE,
                output_dir=root / f"{workload}-{arm}",
                source_commit="a" * 40,
                run_id=f"p57{workload[:3]}{arm[:2]}",
                campaign_tag=f"p57-{workload}-{arm}",
                checkpoint_mode="new",
                expected_updates=updates,
                run_kind="train",
                workload_candidate=candidate,
                data_split=split,
                arm=arm,
                high_performance=arm == "zero",
            )[0]
            env = _env(yaml.safe_load(path.read_text()))
            command = env["CANON_RUN_CMD"].split()
            self.assertEqual(command.count(f"--sampler_is={sampler}"), 1)
            self.assertEqual(
                sum(value.startswith("--sampler_is=") for value in command), 1
            )
            self.assertIn(f"--max_steps={updates}", command)
            self.assertEqual(command.count("--seed=42"), 1)
            self.assertIn(f"--env_max_steps={turns}", command)
            self.assertIn(f"--max_response_length={response}", command)
            self.assertEqual(env["CANON_P57_TIM_ARM"], arm)
            self.assertEqual(
                env["CANON_PROFILE_FILE"],
                paired._V1_HP_PROFILE if arm == "zero" else paired._PROFILE,
            )
            self.assertEqual(
                env["CANON_V1_HP_FULL"], "1" if arm == "zero" else "0"
            )
            self.assertEqual(
                env["CANON_P59_RANK_PARALLEL_BACKWARD"],
                "1" if arm == "zero" else "0",
            )
            self.assertEqual(env["CANON_P57_INFERENCE_REGIME"], regime)
            self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], fixed_head)
            self.assertEqual(
                env["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], warning_only
            )
            self.assertEqual(env["CANON_P57_STOP_AFTER_STEP"], str(updates))
            self.assertEqual(
                env["CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL"], "0"
            )
            self.assertEqual(env["CANON_FROZENLAKE_CKPT_INTERVAL"], "300")
            self.assertEqual(env["CANON_FROZENLAKE_CKPT_MAX_TO_KEEP"], "1")
            self.assertEqual(env["CANON_P33_ENABLE_EVAL"], "1")
            self.assertEqual(env["CANON_P33_DISABLE_EVAL"], "0")
            self.assertEqual(env["CANON_P31_ENABLE_EVAL"], "1")
            self.assertIn("--num_test_batches=4", command)
            self.assertIn("--eval_every_n_steps=50", command)
            preflight = _run_env_preflight(
                env, root / f"state-{workload}-{arm}"
            )
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            resolved = (root / f"state-{workload}-{arm}" / "env.sh").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "export CANON_WANDB_PROJECT=zero-tim-p57-frozenlake-tim",
                resolved,
            )
            expected_group = f"p57-{arm}"
            if candidate:
              expected_group += f"-{candidate}-{split}"
            self.assertIn(
                f"export CANON_WANDB_GROUP={expected_group}",
                resolved,
            )
            if arm == "zero":
              for receipt in (
                  "export CANON_P59_CHECKED_VMA=1",
                  "export CANON_P66_P59_CHECK_VMA=1",
                  "export CANON_P67_P66_VMA_P59_ONLY=1",
                  "export CANON_V1_HP_FIRST_UPDATE_GATE=1",
              ):
                self.assertIn(receipt, resolved)
            if regime == "stock-fast":
              self.assertIn(
                  "[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train",
                  preflight.stdout,
              )

  def test_primary_zero_train_rejects_baseline_profile(self):
    with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
        ValueError, "requires the registered v1 high-performance path"
    ):
      paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57zerobad",
          campaign_tag="p57-zero-baseline-negative",
          checkpoint_mode="new",
          expected_updates=300,
          run_kind="train",
          arm="zero",
      )

  def test_three_arm_wrapper_routes_zero_to_hp_and_requires_clean_exact_sha(self):
    script = THREE_ARM_WRAPPER.read_text(encoding="utf-8")
    self.assertIn("renderer_mode+=(--high-performance)", script)
    self.assertEqual(script.count('"${renderer_mode[@]}"'), 2)
    self.assertIn('git -C "$repo" rev-parse HEAD', script)
    self.assertIn("refusing to render from a dirty worktree", script)

  def test_paired_arms_reject_unregistered_200_step_horizon(self):
    workloads = (
        ("p45", "", "", "P45 workload is frozen to 300 updates"),
        ("m15", "m15", "main", "M15 workload is frozen to 300 updates"),
    )
    for workload, candidate, split, message in workloads:
      with (
          self.subTest(workload=workload),
          tempfile.TemporaryDirectory() as tmp,
          self.assertRaisesRegex(ValueError, message),
      ):
        paired.render_all(
            base_path=BASE,
            output_dir=Path(tmp),
            source_commit="a" * 40,
            run_id=f"p57{workload}short",
            campaign_tag=f"p57-{workload}-short-horizon",
            checkpoint_mode="new",
            expected_updates=200,
            run_kind="train",
            workload_candidate=candidate,
            data_split=split,
            arm="mismatch",
        )

  def test_paired_train_preflight_rejects_seed_drift(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      path = paired.render_all(
          base_path=BASE,
          output_dir=root / "rendered",
          source_commit="a" * 40,
          run_id="p57seed",
          campaign_tag="p57-p45-seed-negative",
          checkpoint_mode="new",
          expected_updates=300,
          run_kind="train",
          arm="mismatch",
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
      env["CANON_RUN_CMD"] = env["CANON_RUN_CMD"].replace(
          "--seed=42", "--seed=43"
      )
      preflight = _run_env_preflight(env, root / "state")
    self.assertNotEqual(preflight.returncode, 0)
    self.assertIn("P57 command/horizon contract drifted", preflight.stderr)

  def test_paired_eval_keeps_only_zero_and_final_recovery_checkpoints(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp) / "eval-300",
          source_commit="a" * 40,
          run_id="p57eval300",
          campaign_tag="p57-native-300-a-p45",
          checkpoint_mode="resume",
          expected_updates=300,
          run_kind="eval",
          checkpoint_step=300,
          arm="mismatch",
      )[0]
      env = _env(yaml.safe_load(path.read_text()))
      preflight = _run_env_preflight(env, Path(tmp) / "state")
    self.assertEqual(env["CANON_P57_EVAL_CHECKPOINT_STEP"], "300")
    self.assertEqual(env["CANON_FROZENLAKE_CKPT_INTERVAL"], "300")
    self.assertEqual(env["CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL"], "0")
    self.assertIn("--evaluation_only", env["CANON_RUN_CMD"])
    self.assertIn("--temperature=0", env["CANON_RUN_CMD"])
    self.assertEqual(preflight.returncode, 0, preflight.stderr)

    with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
        ValueError, "registered training horizon"
    ):
      paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57eval250",
          campaign_tag="p57-native-300-a-p45",
          checkpoint_mode="resume",
          expected_updates=300,
          run_kind="eval",
          checkpoint_step=250,
          arm="mismatch",
      )

  def test_primary_train_rejects_partial_stop_without_recovery_checkpoint(self):
    with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
        ValueError, "only durable checkpoint is the final step"
    ):
      paired.render_all(
          base_path=BASE,
          output_dir=Path(tmp),
          source_commit="a" * 40,
          run_id="p57partial",
          campaign_tag="p57-primary-partial",
          checkpoint_mode="new",
          expected_updates=300,
          run_kind="train",
          arm="mismatch",
          stop_after_step=50,
      )

  def test_eval_rejects_generation_count_not_divisible_by_dp(self):
    with mock.patch.object(paired, "_EVAL_GENERATIONS", 2), self.assertRaisesRegex(
        ValueError, "generations must be divisible by the trainer DP axis"
    ):
      paired._spec(
          paired._ARMS[1],
          200,
          run_kind="eval",
          checkpoint_step=0,
          workload_candidate="m15",
          data_split="selection",
      )

  def test_stock_curve_rejects_unregistered_stop_or_recipe(self):
    common = dict(
        base_path=BASE,
        source_commit="a" * 40,
        run_id="p57stock",
        campaign_tag="p57-m15-selection",
        checkpoint_mode="new",
        expected_updates=200,
        run_kind="train",
        data_split="selection",
        stock_only=True,
    )
    with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
        ValueError, "50-step boundary"
    ):
      paired.render_all(
          **common,
          output_dir=Path(tmp),
          workload_candidate="m15",
          stop_after_step=60,
      )
    with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
        ValueError, "frozen to M15"
    ):
      paired.render_all(
          **common,
          output_dir=Path(tmp),
          workload_candidate="m10",
          stop_after_step=50,
      )

  def test_paired_renderer_rejects_calibration_split(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(ValueError, "selection"):
        paired.render_all(
            base_path=BASE,
            output_dir=Path(tmp),
            source_commit="a" * 40,
            run_id="p57main",
            campaign_tag="p57-m15-main",
            checkpoint_mode="new",
            expected_updates=200,
            workload_candidate="m15",
            data_split="calibration",
            stock_only=True,
        )


if __name__ == "__main__":
  unittest.main()
