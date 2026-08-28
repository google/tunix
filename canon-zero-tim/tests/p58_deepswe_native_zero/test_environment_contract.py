#!/usr/bin/env python3
"""End-to-end P58 renderer -> shell profile -> Python contract validation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from examples.deepswe.swe_env import SWEEnv


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "p58_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py"
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE contract")
deepswe_contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = deepswe_contract
CONTRACT_SPEC.loader.exec_module(deepswe_contract)
ALIGNMENT_SPEC = importlib.util.spec_from_file_location(
    "p58_environment_alignment", ROOT / "tunix/rl/alignment.py"
)
if ALIGNMENT_SPEC is None or ALIGNMENT_SPEC.loader is None:
  raise RuntimeError("cannot import alignment policy")
alignment = importlib.util.module_from_spec(ALIGNMENT_SPEC)
sys.modules[ALIGNMENT_SPEC.name] = alignment
ALIGNMENT_SPEC.loader.exec_module(alignment)
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p58_environment_renderer", PKG / "cluster/render_p58_deepswe_tim.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P58EnvironmentContractTest(unittest.TestCase):

  def test_swe_env_preserves_normalized_prompt_before_reset(self):
    env = SWEEnv({
        "problem_statement": np.array(["raw problem"]),
        "prompts": np.array(["normalized problem"]),
    })

    self.assertEqual(env.entry["problem_statement"], "raw problem")
    self.assertEqual(env.task, {"prompts": ["normalized problem"]})

  def test_swe_env_falls_back_to_problem_statement(self):
    env = SWEEnv({"problem_statement": np.array(["raw problem"])})

    self.assertEqual(env.task, {"prompts": ["raw problem"]})

  def test_swe_env_rejects_missing_prompt_source(self):
    with self.assertRaisesRegex(ValueError, "must contain a non-empty string"):
      SWEEnv({"docker_image": np.array(["example/image"])})

  def _rendered_env(
      self,
      arm: str,
      stage: str,
      *,
      sampler_is: bool = False,
      high_performance: bool = False,
      checked_vma_off_diagnostic: bool = False,
      checked_vma_on_diagnostic: bool = False,
      seam_localization: str = "",
  ) -> dict[str, str]:
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    document = renderer.render(
        base,
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="env-test",
        stage=stage,
        arm=arm,
        cpu_nodepool="cpu-np",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
        sampler_is=sampler_is,
        high_performance=high_performance,
        checked_vma_off_diagnostic=checked_vma_off_diagnostic,
        checked_vma_on_diagnostic=checked_vma_on_diagnostic,
        seam_localization=seam_localization,
    )
    return dict(renderer.p34._env(document))

  def _resolved(
      self,
      arm: str,
      stage: str,
      *,
      sampler_is: bool = False,
      high_performance: bool = False,
      checked_vma_off_diagnostic: bool = False,
      checked_vma_on_diagnostic: bool = False,
      seam_localization: str = "",
  ) -> dict[str, str]:
    supplied = os.environ.copy()
    supplied.update(
        self._rendered_env(
            arm,
            stage,
            sampler_is=sampler_is,
            high_performance=high_performance,
            checked_vma_off_diagnostic=checked_vma_off_diagnostic,
            checked_vma_on_diagnostic=checked_vma_on_diagnostic,
            seam_localization=seam_localization,
        )
    )
    profile = supplied["CANON_PROFILE_FILE"]
    command = (
        "set -a; "
        f"source {PKG / 'cluster/profiles/_canonical_engine.env'}; "
        f"source {PKG / profile}; "
        "env -0"
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        env=supplied,
        check=True,
        capture_output=True,
    )
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in completed.stdout.decode().split("\0")
        if "=" in item
    }

  def _persisted(
      self,
      arm: str,
      stage: str,
      *,
      sampler_is: bool = False,
      high_performance: bool = False,
      checked_vma_off_diagnostic: bool = False,
      checked_vma_on_diagnostic: bool = False,
      seam_localization: str = "",
  ):
    supplied = os.environ.copy()
    rendered = self._rendered_env(
        arm,
        stage,
        sampler_is=sampler_is,
        high_performance=high_performance,
        checked_vma_off_diagnostic=checked_vma_off_diagnostic,
        checked_vma_on_diagnostic=checked_vma_on_diagnostic,
        seam_localization=seam_localization,
    )
    supplied.update(rendered)
    supplied.update({
        "CANON_PKG": str(PKG),
        "HF_TOKEN": "test-hf-runtime-token",
        "WANDB_API_KEY": "test-wandb-runtime-key",
        "INJECTED_HF_TOKEN": "test-hf-token",
        "INJECTED_WANDB_API_KEY": "test-wandb-key",
    })
    with tempfile.TemporaryDirectory() as state_dir:
      supplied["CANON_STATE"] = state_dir
      if seam_localization:
        rendered_state = rendered["CANON_STATE"]
        for key, value in tuple(supplied.items()):
          if isinstance(value, str) and value.startswith(rendered_state):
            supplied[key] = state_dir + value[len(rendered_state):]
        supplied["CANON_P38_GCS_PREFIX"] = (
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/"
            f"{Path(state_dir).name}/attempt-0"
        )
      if checked_vma_off_diagnostic or checked_vma_on_diagnostic:
        supplied["CANON_P38_DIAGNOSTIC_ROUND_FILE"] = str(
            Path(state_dir) / "p38_diagnostic_round"
        )
      completed = subprocess.run(
          ["bash", str(PKG / "cluster/steps/00_env.sh")],
          cwd=ROOT,
          env=supplied,
          check=False,
          text=True,
          capture_output=True,
      )
      if completed.returncode != 0:
        self.fail(
            "00_env.sh rejected the rendered contract:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
      resolved = (Path(state_dir) / "env.sh").read_text()
      reloaded = subprocess.run(
          [
              "bash",
              "-c",
              f"source {Path(state_dir) / 'env.sh'}; env -0",
          ],
          env=supplied,
          check=True,
          capture_output=True,
      )
    values = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in reloaded.stdout.decode().split("\0")
        if "=" in item
    }
    return completed, resolved, values

  def test_native_renderer_environment_passes_real_00_env(self):
    completed, resolved, values = self._persisted("native", "three-update")
    self.assertIn("[env] P34 contract OK: DP8xTP8", completed.stdout)
    self.assertNotIn("REFUSING TO CONTINUE", completed.stderr)
    self.assertIn("export CANON_P32_DP_REDUCTION_ADMITTED=0", resolved)
    self.assertIn("export CANON_FROZENLAKE_L3=0", resolved)
    self.assertIn("export CANON_FROZENLAKE_P27=0", resolved)
    self.assertIn(
        "export CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=0", resolved
    )
    self.assertNotIn("test-hf-runtime-token", resolved)
    self.assertNotIn("test-wandb-runtime-key", resolved)
    # 00_env.sh is a child of the entrypoint. Exercise the actual reload
    # boundary with the raw renderer environment still present, rather than
    # merely inspecting the generated exports. This is the p58c03 regression:
    # the native profile unset CANON_LOGPROB_M in the child, but a layered
    # source left the renderer's CANON_LOGPROB_M=256 alive in the parent.
    self.assertNotIn("CANON_LOGPROB_M", values)
    self.assertNotIn("CANON_FIXED_AR", values)
    self.assertIn("export R2E_K8S_QUEUE_NAME=multislice-queue", resolved)
    self.assertEqual(values["R2E_K8S_QUEUE_NAME"], "multislice-queue")
    self.assertEqual(values["HF_TOKEN"], "test-hf-runtime-token")
    self.assertEqual(values["WANDB_API_KEY"], "test-wandb-runtime-key")
    deepswe_contract.validate_environment(values)

  def test_zero_renderer_environment_survives_authoritative_reload(self):
    _, resolved, values = self._persisted("zero", "three-update")
    self.assertIn("export CANON_LOGPROB_M=256", resolved)
    self.assertEqual(values["CANON_LOGPROB_M"], "256")
    self.assertEqual(values["CANON_FIXED_AR"], "1")
    self.assertEqual(values["HF_TOKEN"], "test-hf-runtime-token")
    self.assertEqual(values["WANDB_API_KEY"], "test-wandb-runtime-key")
    deepswe_contract.validate_environment(values)

  def test_native_is_renderer_environment_survives_authoritative_reload(self):
    completed, resolved, values = self._persisted(
        "native", "full", sampler_is=True
    )
    self.assertIn("[env] P34 contract OK: DP8xTP8", completed.stdout)
    self.assertIn("export CANON_P34_DISABLE_SAMPLER_IS=0", resolved)
    self.assertIn("export CANON_P34_DISABLE_TIS=0", resolved)
    self.assertEqual(
        deepswe_contract.p58_sampler_recipe(values), "native-is"
    )
    deepswe_contract.validate_environment(values)

  def test_native_rejects_partial_sampler_tuple(self):
    values = self._resolved("native", "full", sampler_is=True)
    with self.assertRaisesRegex(ValueError, "sampler recipe"):
      deepswe_contract.validate_environment({
          **values,
          "CANON_P34_DISABLE_TIS": "1",
      })

  def test_zero_hp_full_survives_real_env_and_python_contract(self):
    completed, resolved, values = self._persisted(
        "zero", "full", high_performance=True
    )
    self.assertIn("P58 v1-hp Qwen3-4B TP8 fixed lm-head enabled", completed.stdout)
    for key, expected in {
        "CANON_V1_HP_FULL": "1",
        "CANON_P38_FIXED_LM_HEAD": "1",
        "CANON_CONTINUE_DECODE": "8",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P59_CHECKED_VMA": "1",
        "CANON_P66_P59_CHECK_VMA": "1",
        "CANON_P67_P66_VMA_P59_ONLY": "1",
        "CANON_V1_HP_FIRST_UPDATE_GATE": "1",
        "CANON_P63_OVERFLOW_SAFE_CLIP": "1",
        "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
    }.items():
      self.assertEqual(values[key], expected)
      self.assertIn(f"export {key}={expected}", resolved)
    deepswe_contract.validate_environment(values)

  def test_zero_hp_partial_bundle_is_rejected_by_python_contract(self):
    _, _, values = self._persisted(
        "zero", "full", high_performance=True
    )
    deepswe_contract.validate_environment(values)
    for key, replacement in (
        ("CANON_CONTINUE_DECODE", "0"),
        ("CANON_P59_RANK_PARALLEL_BACKWARD", "0"),
        ("CANON_P59_CHECKED_VMA", "0"),
        ("CANON_P66_P59_CHECK_VMA", "0"),
        ("CANON_P67_P66_VMA_P59_ONLY", "0"),
        ("CANON_V1_HP_FIRST_UPDATE_GATE", "0"),
        ("CANON_P63_OVERFLOW_SAFE_CLIP", "0"),
        ("CANON_P38_FIXED_LM_HEAD", "0"),
        ("CANON_VLLM_ENABLE_PREFIX_CACHING", "1"),
    ):
      with self.subTest(key=key), self.assertRaises(ValueError):
        deepswe_contract.validate_environment({**values, key: replacement})

  def test_checked_vma_off_diagnostic_survives_real_env_contract(self):
    completed, resolved, values = self._persisted(
        "zero", "full", checked_vma_off_diagnostic=True
    )
    self.assertIn(
        "P58 checked-VMA-off precheck admitted", completed.stdout
    )
    for key, expected in {
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC": "off",
        "CANON_P59_CHECKED_VMA": "0",
        "CANON_P66_P59_CHECK_VMA": "0",
        "CANON_P67_P66_VMA_P59_ONLY": "0",
        "CANON_V1_HP_FIRST_UPDATE_GATE": "0",
        "CANON_P63_OVERFLOW_SAFE_CLIP": "0",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
    }.items():
      self.assertEqual(values[key], expected)
      self.assertIn(f"export {key}={expected}", resolved)
    deepswe_contract.validate_environment(values)

  def test_checked_vma_off_diagnostic_rejects_partial_tuple(self):
    _, _, values = self._persisted(
        "zero", "full", checked_vma_off_diagnostic=True
    )
    for key, replacement in (
        ("CANON_P59_CHECKED_VMA", "1"),
        ("CANON_P66_P59_CHECK_VMA", "1"),
        ("CANON_P67_P66_VMA_P59_ONLY", "1"),
        ("CANON_V1_HP_FIRST_UPDATE_GATE", "1"),
        ("CANON_P63_OVERFLOW_SAFE_CLIP", "1"),
        ("CANON_P38_PRECHECK_ONLY", "0"),
        ("CANON_P38_CONTROLLED_EXIT", "0"),
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "2"),
    ):
      with self.subTest(key=key), self.assertRaises(ValueError):
        deepswe_contract.validate_environment({**values, key: replacement})

  def test_checked_vma_on_diagnostic_survives_real_env_contract(self):
    completed, resolved, values = self._persisted(
        "zero", "full", checked_vma_on_diagnostic=True
    )
    self.assertIn(
        "P58 checked-VMA-on precheck admitted", completed.stdout
    )
    for key, expected in {
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC": "on",
        "CANON_P59_CHECKED_VMA": "1",
        "CANON_P66_P59_CHECK_VMA": "1",
        "CANON_P67_P66_VMA_P59_ONLY": "1",
        "CANON_V1_HP_FIRST_UPDATE_GATE": "0",
        "CANON_P63_OVERFLOW_SAFE_CLIP": "0",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
    }.items():
      self.assertEqual(values[key], expected)
      self.assertIn(f"export {key}={expected}", resolved)
    deepswe_contract.validate_environment(values)

  def test_checked_vma_on_diagnostic_rejects_partial_tuple(self):
    _, _, values = self._persisted(
        "zero", "full", checked_vma_on_diagnostic=True
    )
    for key, replacement in (
        ("CANON_P59_CHECKED_VMA", "0"),
        ("CANON_P66_P59_CHECK_VMA", "0"),
        ("CANON_P67_P66_VMA_P59_ONLY", "0"),
        ("CANON_V1_HP_FIRST_UPDATE_GATE", "1"),
        ("CANON_P63_OVERFLOW_SAFE_CLIP", "1"),
        ("CANON_P38_PRECHECK_ONLY", "0"),
        ("CANON_P38_CONTROLLED_EXIT", "0"),
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "2"),
    ):
      with self.subTest(key=key), self.assertRaises(ValueError):
        deepswe_contract.validate_environment({**values, key: replacement})

  def test_coarse_seam_survives_real_env_and_python_contract(self):
    completed, resolved, values = self._persisted(
        "zero", "full", seam_localization="coarse"
    )
    self.assertIn("P58 coarse seam precheck admitted", completed.stdout)
    for key, expected in {
        "CANON_P58_SEAM_LOCALIZATION": "coarse",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
        "CANON_P38_DURABILITY_PROFILE": "p58-seam-v1",
        "CANON_P38_SEAM_OBSERVER": "layer",
        "CANON_P38_SEAM_MIN_POSITION": "3072",
        "CANON_P38_SEAM_MAX_POSITION": "4608",
        "CANON_P38_TAIL_OBSERVER": "1",
        "CANON_P59_CHECKED_VMA": "1",
        "CANON_P67_P66_VMA_P59_ONLY": "1",
        "CANON_V1_HP_FIRST_UPDATE_GATE": "1",
        "CANON_P63_OVERFLOW_SAFE_CLIP": "1",
    }.items():
      self.assertEqual(values[key], expected)
      self.assertIn(f"export {key}={expected}", resolved)
    deepswe_contract.validate_environment(values)

  def test_coarse_seam_rejects_partial_tuple(self):
    _, _, values = self._persisted(
        "zero", "full", seam_localization="coarse"
    )
    for key, replacement in (
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "1"),
        ("CANON_P38_DURABILITY_PROFILE", "full-v1"),
        ("CANON_P38_SEAM_OBSERVER", "full"),
        ("CANON_P38_TAIL_OBSERVER", "0"),
    ):
      with self.subTest(key=key), self.assertRaises(ValueError):
        deepswe_contract.validate_environment({**values, key: replacement})

  def test_p67_is_rejected_outside_zero_hp(self):
    for arm in ("native", "zero"):
      with self.subTest(arm=arm):
        values = self._resolved(arm, "three-update")
        deepswe_contract.validate_environment(values)
        with self.assertRaises(ValueError):
          deepswe_contract.validate_environment({
              **values,
              "CANON_P67_P66_VMA_P59_ONLY": "1",
          })

  def test_both_arms_resolve_to_the_signed_contract(self):
    for arm in ("native", "zero"):
      for stage in ("three-update", "full"):
        with self.subTest(arm=arm, stage=stage):
          values = self._resolved(arm, stage)
          deepswe_contract.validate_environment(values)
          workload = deepswe_contract.active_workload(values)
          self.assertEqual(workload.global_trajectories, 128)
          self.assertEqual(workload.local_trajectories, 16)
          self.assertEqual(
              values["R2E_K8S_QUEUE_NAME"], "multislice-queue"
          )
          if arm == "native":
            self.assertNotIn("CANON_FIXED_AR", values)
            self.assertNotIn("CANON_LOGPROB_M", values)
            self.assertEqual(values["CANON_PROMPT_PROCESSED_LOGPROBS"], "0")
            self.assertEqual(
                values["CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER"], "1"
            )
            self.assertEqual(values["CANON_P28_BATCHED_REVERSE"], "0")
            self.assertEqual(values["CANON_BATCHED_EVIDENCE"], "0")
            self.assertNotIn("CANON_P59_CHECKED_VMA", values)
            self.assertNotIn("CANON_P67_P66_VMA_P59_ONLY", values)
            self.assertNotIn("CANON_V1_HP_FIRST_UPDATE_GATE", values)
            self.assertNotIn("CANON_P63_OVERFLOW_SAFE_CLIP", values)
          else:
            self.assertEqual(values["CANON_FIXED_AR"], "1")
            self.assertEqual(values["CANON_LOGPROB_M"], "256")
            self.assertEqual(values["CANON_PROMPT_PROCESSED_LOGPROBS"], "1")
            self.assertEqual(
                values["CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER"], "0"
            )
            self.assertNotIn("CANON_P59_CHECKED_VMA", values)
            self.assertNotIn("CANON_P67_P66_VMA_P59_ONLY", values)
            self.assertNotIn("CANON_V1_HP_FIRST_UPDATE_GATE", values)
            self.assertNotIn("CANON_P63_OVERFLOW_SAFE_CLIP", values)

  def test_prompt_observer_treatments_are_mutually_exclusive(self):
    native = self._resolved("native", "three-update")
    zero = self._resolved("zero", "three-update")
    for changed in (
        {"CANON_PROMPT_PROCESSED_LOGPROBS": "1"},
        {"CANON_ENGINE_MODULE_C": "1"},
        {"CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "0"},
    ):
      with self.subTest(arm="native", changed=changed):
        with self.assertRaises(ValueError):
          deepswe_contract.validate_environment({**native, **changed})
    with self.assertRaises(ValueError):
      deepswe_contract.validate_environment({
          **zero, "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "1"
      })
    with self.assertRaises(ValueError):
      deepswe_contract.validate_environment({
          **zero, "CANON_PROMPT_PROCESSED_LOGPROBS": "0"
      })

  def test_rendered_native_full_environment_is_alignment_admitted(self):
    values = self._resolved("native", "full")
    with mock.patch.dict(os.environ, values, clear=True):
      policy = alignment.gsm8k_ab_report_policy()
    self.assertTrue(policy["warning_only"])
    self.assertEqual(policy["stage"], "full")
    self.assertEqual(
        policy["warning_boundaries"],
        (
            "S_decode_vs_S_prefill",
            "S_prefill_vs_T_old",
            "T_old_vs_T_current",
        ),
    )


if __name__ == "__main__":
  unittest.main()
