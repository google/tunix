"""Cross-workload contracts for the registered full system optimization."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
sys.path.insert(0, str(PKG / "cluster"))
from v1_full_system_optimization import (  # pylint: disable=wrong-import-position
    FULL_SYSTEM_OPTIMIZATION_ENV_NAMES,
    REGISTERED_FULL_WORKLOADS,
    full_system_optimization_additions,
)


class FullSystemOptimizationTest(unittest.TestCase):

  def test_exact_registered_workload_tuples(self):
    self.assertEqual(
        REGISTERED_FULL_WORKLOADS,
        frozenset({
            "gsm8k",
            "frozenlake-p45",
            "frozenlake-m15",
            "deepswe-qwen4b",
        }),
    )
    for workload in sorted(REGISTERED_FULL_WORKLOADS):
      with self.subTest(workload=workload):
        values = full_system_optimization_additions(workload)
        self.assertEqual(values["CANON_P59_CHECKED_VMA"], "1")
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
        if workload == "gsm8k":
          self.assertNotIn("CANON_P67_P66_VMA_P59_ONLY", values)
        else:
          self.assertEqual(values["CANON_P67_P66_VMA_P59_ONLY"], "1")
        self.assertTrue(set(values).issubset(FULL_SYSTEM_OPTIMIZATION_ENV_NAMES))

  def test_returns_fresh_copy_and_rejects_unregistered_neighbors(self):
    first = full_system_optimization_additions("frozenlake-p45")
    first["CANON_P71_SCAN"] = "bwd"
    second = full_system_optimization_additions("frozenlake-p45")
    self.assertEqual(second["CANON_P71_SCAN"], "fwd")
    for workload in (
        "frozenlake-stock",
        "deepswe-native",
        "deepswe-qwen32b",
        "p58-diagnostic",
    ):
      with self.subTest(workload=workload):
        with self.assertRaisesRegex(ValueError, "unregistered"):
          full_system_optimization_additions(workload)

  def test_deepswe_full_prepare_is_clean_sha_bound_and_render_only(self):
    path = (
        PKG
        / "tasks/v1-system-optimization-workload-rollout"
        / "prepare_deepswe_zero_hp_full.sh"
    )
    source = path.read_text(encoding="utf-8")
    self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD', source)
    self.assertIn("refusing to render from a dirty worktree", source)
    self.assertIn("--stage full", source)
    self.assertIn("--arm zero", source)
    self.assertIn("--high-performance", source)
    self.assertIn("V1_DEEPSWE_ZERO_HP_RFULL_READY", source)
    self.assertIn("launch=not-executed", source)
    self.assertFalse(
        any(line.strip().startswith("kubectl apply") for line in source.splitlines())
    )
    completed = subprocess.run(
        ["bash", "-n", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, msg=completed.stderr)

  def test_operator_handoffs_route_full_training_through_registered_wrappers(self):
    frozen_handoffs = (
        PKG / "tasks/v1-phase4-three-full-recipes/HANDOFF.md",
        PKG / "tasks/p57-frozenlake-tim-causal-study/HANDOFF.md",
        PKG / "tasks/p45-frozenlake-dp8-tp8-resident/HANDOFF.md",
    )
    frozen_wrapper = "prepare_p67_frozenlake_two_full_wave.sh"
    for path in frozen_handoffs:
      with self.subTest(path=path):
        source = path.read_text(encoding="utf-8")
        first_section = next(
            line for line in source.splitlines() if line.startswith("## ")
        )
        self.assertIn("P74", first_section)
        self.assertIn(frozen_wrapper, source)
        self._assert_documented_system_tuple(source)

    deepswe_handoff = (
        PKG / "tasks/p58-deepswe-native-zero-comparison/HANDOFF.md"
    )
    source = deepswe_handoff.read_text(encoding="utf-8")
    h2_sections = [
        line for line in source.splitlines() if line.startswith("## ")
    ]
    self.assertTrue(any("P74" in sec for sec in h2_sections))
    self.assertIn("prepare_deepswe_zero_hp_full.sh", source)
    self._assert_documented_system_tuple(source)

    for path, wrapper in (
        (
            PKG / "tasks/v1-phase4-three-full-recipes/RUNBOOK.md",
            frozen_wrapper,
        ),
        (
            PKG / "tasks/p57-frozenlake-tim-causal-study/RUNBOOK.md",
            frozen_wrapper,
        ),
        (
            PKG / "cluster/P58_DEEPSWE_TIM_RUNBOOK.md",
            "prepare_deepswe_zero_hp_full.sh",
        ),
    ):
      with self.subTest(path=path):
        source = path.read_text(encoding="utf-8")
        self.assertIn(wrapper, source)
        self._assert_documented_system_tuple(source)

  def _assert_documented_system_tuple(self, source: str):
    for key_value in (
        "CANON_DP_COMPARE_MODE=fingerprint-hybrid",
        "CANON_DP_DISTINCT_SCHEDULE=first-group-warmup",
        "CANON_DP_FINITE_FETCH=batched-commit",
        "CANON_P71_SCAN=fwd",
    ):
      self.assertIn(key_value, source)
    self.assertIn("CANON_DP_COLLECTIVE_REDUCE", source)
    self.assertRegex(source, r"CANON_DP_COLLECTIVE_REDUCE.{0,80}(absent|remain)")


if __name__ == "__main__":
  unittest.main()
