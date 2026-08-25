"""Regression tests for workload-specific sampler importance sampling."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _REPO_ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
_tree = ast.parse(_SOURCE.read_text(), filename=str(_SOURCE))
_names = {
    "_canonical_alignment_sampler_is_valid",
    "_m15_apc_target_alignment_enabled",
    "_p57_tim_purity_enabled",
    "_p57_tim_is_enabled",
    "_validate_p57_tim_purity",
    "_validate_p57_tim_is",
}
_matches = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name in _names
]
if {node.name for node in _matches} != _names:
  raise RuntimeError("cannot isolate the sampler purity contracts")
_namespace = {
    "Mapping": Mapping,
    "alignment": SimpleNamespace(AlignmentGateError=ValueError),
}
exec(
    compile(ast.Module(body=_matches, type_ignores=[]), str(_SOURCE), "exec"),
    _namespace,
)
_sampler_is_valid = _namespace["_canonical_alignment_sampler_is_valid"]
_m15_apc_target_enabled = _namespace["_m15_apc_target_alignment_enabled"]
_p57_purity_enabled = _namespace["_p57_tim_purity_enabled"]
_p57_is_enabled = _namespace["_p57_tim_is_enabled"]
_validate_p57_purity = _namespace["_validate_p57_tim_purity"]
_validate_p57_is = _namespace["_validate_p57_tim_is"]


class SamplerIsContractTest(unittest.TestCase):

  def test_gsm8k_direct_rollout_logprobs_allow_no_sampler_correction(self):
    self.assertTrue(_sampler_is_valid(None, "gsm8k"))

  def test_frozenlake_rejects_missing_token_sampler_correction(self):
    self.assertFalse(_sampler_is_valid(None, "frozenlake"))

  def test_p57_causal_study_admits_direct_rollout_logprobs(self):
    self.assertTrue(
        _sampler_is_valid(None, "frozenlake", p57_tim_study=True)
    )

  def test_signed_m15_apc_target_admits_no_is_only_at_exact_identity(self):
    good = {
        "CANON_APC_M15_TARGET_DEBUG": "on",
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env"
        ),
        "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
        "CANON_P57_WORKLOAD_CANDIDATE": "m15",
        "CANON_P57_DATA_SPLIT": "main",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_DP_SIZE": "8",
        "CANON_TP_SIZE": "8",
    }
    for arm in ("off", "on"):
      env = {**good, "CANON_APC_M15_TARGET_DEBUG": arm}
      self.assertTrue(_m15_apc_target_enabled(env))
      self.assertTrue(
          _sampler_is_valid(
              None,
              "frozenlake-dp8-tp8",
              m15_apc_target=_m15_apc_target_enabled(env),
          )
      )
    for changed in (
        {"CANON_APC_M15_TARGET_DEBUG": ""},
        {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-8b.env"},
        {"CANON_P32_WORKLOAD": "frozenlake"},
        {"CANON_P57_WORKLOAD_CANDIDATE": "p45"},
        {"CANON_P57_DATA_SPLIT": "selection"},
        {"CANON_P38_PRECHECK_ONLY": "0"},
        {"CANON_P38_CONTROLLED_EXIT": "0"},
        {"CANON_P33_RUN_STAGE": "train"},
        {"CANON_P33_NO_COMMIT": "0"},
        {"CANON_DP_SIZE": "16"},
        {"CANON_TP_SIZE": "4"},
    ):
      with self.subTest(changed=changed):
        self.assertFalse(_m15_apc_target_enabled({**good, **changed}))

  def test_unsigned_m15_workload_still_rejects_no_is(self):
    self.assertFalse(
        _sampler_is_valid(None, "frozenlake-dp8-tp8")
    )

  def test_p57_purity_scope_requires_exact_profile_and_workload(self):
    good = {
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
        ),
        "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    }
    self.assertTrue(_p57_purity_enabled(good))
    self.assertTrue(
        _p57_purity_enabled({
            **good,
            "CANON_P57_TIM_ARM": "zero",
            "CANON_PROFILE_FILE": (
                "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
            ),
        })
    )
    for changed in (
        {"CANON_P57_RUN_KIND": "eval"},
        {"CANON_P57_TIM_ARM": "other"},
        {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-8b.env"},
        {"CANON_P32_WORKLOAD": "frozenlake"},
    ):
      with self.subTest(changed=changed):
        self.assertFalse(_p57_purity_enabled({**good, **changed}))

  def test_p57_purity_contract_accepts_rollout_old_without_tis(self):
    _validate_p57_purity(
        sampler_is=None,
        use_rollout_logps=True,
        rollout_logps_present=True,
        old_logps_are_rollout=True,
        sampler_is_weights_present=False,
    )

  def test_p57_purity_contract_rejects_each_mitigation_or_source_drift(self):
    cases = (
        {"sampler_is": "token"},
        {"use_rollout_logps": False},
        {"rollout_logps_present": False},
        {"old_logps_are_rollout": False},
        {"sampler_is_weights_present": True},
    )
    base = {
        "sampler_is": None,
        "use_rollout_logps": True,
        "rollout_logps_present": True,
        "old_logps_are_rollout": True,
        "sampler_is_weights_present": False,
    }
    for changed in cases:
      with self.subTest(changed=changed), self.assertRaisesRegex(
          Exception, "P57 TIM purity contract failed"
      ):
        _validate_p57_purity(**{**base, **changed})

  def test_p57_is_scope_and_contract_require_real_token_tis(self):
    good_env = {
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "is",
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"
        ),
        "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    }
    self.assertTrue(_p57_is_enabled(good_env))
    self.assertFalse(
        _p57_is_enabled({
            **good_env,
            "CANON_PROFILE_FILE": (
                "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
            ),
        })
    )
    self.assertFalse(
        _p57_is_enabled({**good_env, "CANON_P57_TIM_ARM": "mismatch"})
    )
    base = {
        "sampler_is": "token",
        "use_rollout_logps": True,
        "rollout_logps_present": True,
        "trainer_logps_present": True,
        "old_logps_are_trainer": True,
        "sampler_is_weights_present": True,
    }
    _validate_p57_is(**base)
    for changed in (
        {"sampler_is": None},
        {"use_rollout_logps": False},
        {"rollout_logps_present": False},
        {"trainer_logps_present": False},
        {"old_logps_are_trainer": False},
        {"sampler_is_weights_present": False},
    ):
      with self.subTest(changed=changed), self.assertRaisesRegex(
          Exception, "P57 TIM IS contract failed"
      ):
        _validate_p57_is(**{**base, **changed})

  def test_frozenlake_accepts_token_sampler_correction(self):
    self.assertTrue(_sampler_is_valid("token", "frozenlake"))

  def test_frozenlake_recipe_defaults_to_token_sampler_correction(self):
    source = (
        _REPO_ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text(encoding="utf-8")
    self.assertIn('default="token",', source)
    self.assertIn(
        'sampler_is=None if args.sampler_is == "none" else args.sampler_is,',
        source,
    )
    self.assertNotIn(
        'sampler_is=None if CANON_P32_WORKLOAD else "token",', source
    )

  def test_gsm8k_recipe_keeps_direct_rollout_logprobs(self):
    source = (
        _REPO_ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        '"sampler_is": None if _P32_WORKLOAD_NAME == "gsm8k" else "token",',
        source,
    )

  def test_p59_proxy_uses_token_sampler_correction(self):
    self.assertFalse(_sampler_is_valid(None, "gsm8k-p59-dp4-tp1"))
    self.assertTrue(_sampler_is_valid("token", "gsm8k-p59-dp4-tp1"))
    source = (
        _REPO_ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertNotIn(
        '"sampler_is": None if CANON_P32_WORKLOAD else "token",', source
    )

  def test_p41_benchmark_only_uses_serial_engine_seed(self):
    source = (
        _REPO_ROOT / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn("if CANON_P41_OPTIMIZER_BENCH:", source)
    self.assertIn(
        'vllm_rollout_dict["rollout_vllm_kwargs"]["seed"] = SEED', source
    )
    self.assertIn(
        'P41 optimizer benchmark requires max_concurrency=1', source
    )
    self.assertNotIn('"seed": SEED,', source)


if __name__ == "__main__":
  unittest.main()
