"""Cross-workload contracts for the registered full system optimization."""

from __future__ import annotations

from pathlib import Path
import re
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
    full_system_optimization_base_additions,
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
        # The streamed kept tape is armed only where it is hardware-certified.
        if workload == "deepswe-qwen4b":
          self.assertNotIn("CANON_P32_KEEP_TAPE", values)
          self.assertNotIn("CANON_DP_REDUCE_ONCE", values)
        else:
          self.assertEqual(values["CANON_P32_KEEP_TAPE"], "stream")
          self.assertEqual(values["CANON_DP_REDUCE_ONCE"], "1")
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

  def test_base_additions_exclude_workload_specific_knives(self):
    for workload in sorted(REGISTERED_FULL_WORKLOADS):
      with self.subTest(workload=workload):
        values = full_system_optimization_base_additions(workload)
        self.assertNotIn("CANON_P32_KEEP_TAPE", values)
        self.assertNotIn("CANON_DP_REDUCE_ONCE", values)
        if workload == "gsm8k":
          self.assertNotIn("CANON_P67_P66_VMA_P59_ONLY", values)
        else:
          self.assertEqual(values["CANON_P67_P66_VMA_P59_ONLY"], "1")

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
        self._assert_current_frozenlake_route(source)
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
    self._assert_documented_system_tuple(source, keep_tape=False)

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
        self._assert_documented_system_tuple(
          source, keep_tape="P58" not in str(path)
      )

  def _assert_current_frozenlake_route(self, source: str):
    sections = list(re.finditer(r"^## ", source, re.MULTILINE))
    self.assertTrue(sections, "missing operator entry section")
    start = sections[0].start()
    end = sections[1].start() if len(sections) > 1 else len(source)
    current = source[start:end]
    self.assertTrue(current.startswith("## START HERE"))
    # Historical phase labels are not the contract. Require the executable
    # route and selected bundle in the current entry, not only in old notes.
    for required in (
        "prepare_p67_frozenlake_two_full_wave.sh",
        "CANON_P71_SCAN=fwd",
        "CANON_P32_KEEP_TAPE=stream",
        "CANON_DP_REDUCE_ONCE=1",
    ):
      self.assertRegex(current, rf"(?<![\w-]){re.escape(required)}(?![\w.-])")

  def test_current_route_does_not_inherit_historical_instructions(self):
    route = (
        "prepare_p67_frozenlake_two_full_wave.sh\n"
        "CANON_P71_SCAN=fwd\n"
        "CANON_P32_KEEP_TAPE=stream\n"
        "CANON_DP_REDUCE_ONCE=1\n"
    )
    current = "## START HERE — v2 integration\n" + route
    historical = "## Historical — P74\n" + route
    self._assert_current_frozenlake_route(current + historical)
    for required in route.splitlines():
      with self.subTest(missing=required), self.assertRaises(AssertionError):
        self._assert_current_frozenlake_route(
            current.replace(required, "REMOVED") + historical
        )
    for old, new in (
        ("CANON_P71_SCAN=fwd", "CANON_P71_SCAN=fwd_block"),
        ("CANON_DP_REDUCE_ONCE=1", "CANON_DP_REDUCE_ONCE=10"),
        ("CANON_P32_KEEP_TAPE=stream", "CANON_P32_KEEP_TAPE=streaming"),
    ):
      with self.subTest(wrong=new), self.assertRaises(AssertionError):
        self._assert_current_frozenlake_route(current.replace(old, new) + historical)
    for invalid in (
        "# No operator section\n",
        "## Historical\n" + route + current,
        "## START HERE\nNo route here.\n" + historical,
    ):
      with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
        self._assert_current_frozenlake_route(invalid)

  def _assert_documented_system_tuple(self, source: str, *, keep_tape=True):
    for key_value in (
        "CANON_DP_COMPARE_MODE=fingerprint-hybrid",
        "CANON_DP_DISTINCT_SCHEDULE=first-group-warmup",
        "CANON_DP_FINITE_FETCH=batched-commit",
        "CANON_P71_SCAN=fwd",
    ) + ((
        "CANON_P32_KEEP_TAPE=stream",
        "CANON_DP_REDUCE_ONCE=1",
    ) if keep_tape else ()):
      self.assertIn(key_value, source)
    self.assertIn("CANON_DP_COLLECTIVE_REDUCE", source)
    self.assertRegex(source, r"CANON_DP_COLLECTIVE_REDUCE.{0,80}(absent|remain)")


if __name__ == "__main__":
  unittest.main()
