#!/usr/bin/env python3
"""Static isolation contracts for the P58 native stock B observer."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"


class P58StockPromptObserverContractTest(unittest.TestCase):

  def test_entrypoint_verifies_stock_before_native_only_install(self):
    entrypoint = (PKG / "cluster/entrypoint.sh").read_text()
    native_start = entrypoint.index(
        'elif [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]'
    )
    next_branch = entrypoint.index("elif p57_is_stock_fast_runtime", native_start)
    native = entrypoint[native_start:next_branch]
    self.assertLess(
        native.index("step p58_verify_stock_engine.sh"),
        native.index("step p58_install_stock_prompt_observer.sh"),
    )
    self.assertIn("canonical_overlay=skipped", native)
    self.assertIn("stock_observer=installed", native)

  def test_installer_enforces_independent_native_signature(self):
    installer = (
        PKG / "cluster/steps/p58_install_stock_prompt_observer.sh"
    ).read_text()
    self.assertIn("CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER", installer)
    self.assertIn('CANON_PROMPT_PROCESSED_LOGPROBS:-}\" != \"0', installer)
    self.assertIn('CANON_ENGINE_MODULE_C:-}\" != \"0', installer)
    self.assertIn("runner was not stock before install", installer)
    self.assertIn("canonical_bundle=off treatment=observer-only", installer)
    self.assertIn("CANON_FIXED_AR CANON_FIXED_AR_EMBED CANON_LOGPROB_M", installer)

  def test_runner_patch_is_observer_only_and_not_canonical(self):
    patch = (
        PKG / "patches/p58_stock_observer/01-tpu-runner.patch"
    ).read_text()
    self.assertIn("CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER", patch)
    self.assertIn("compute_processed_prompt_logprobs", patch)
    self.assertNotIn("CANON_PROMPT_PROCESSED_LOGPROBS", patch)
    self.assertNotIn("CANON_LOGPROB_M", patch)
    self.assertNotIn("CANON_ENGINE_MODULE_C", patch)

  def test_manifest_is_exactly_runner_plus_p58_helper(self):
    manifest = (PKG / "P58_STOCK_OBSERVER_MANIFEST.sha256").read_text().splitlines()
    self.assertEqual(len(manifest), 2)
    self.assertEqual(
        {line.split(maxsplit=1)[1] for line in manifest},
        {
            "runner/tpu_runner.py",
            "runner/p58_stock_prompt_observer.py",
        },
    )

  def test_postflight_requires_one_native_marker(self):
    postflight = (PKG / "cluster/steps/90_run.sh").read_text()
    self.assertIn("n_p58_stock_observer", postflight)
    self.assertIn('n_p58_stock_observer\" -ne 1', postflight)
    self.assertIn("canonical_markers=0 canonical_overlay=skipped", postflight)
    self.assertIn("stock_observer=observer-only", postflight)

  def test_profiles_keep_native_and_zero_treatments_mutually_exclusive(self):
    profile = (
        PKG / "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
    ).read_text()
    native = profile[profile.index("  native)"):profile.index("  zero)")]
    zero_start = profile.index("  zero)")
    zero = profile[zero_start:profile.index("  *)", zero_start)]
    self.assertIn("CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1", native)
    self.assertIn("CANON_PROMPT_PROCESSED_LOGPROBS=0", native)
    self.assertIn("CANON_ENGINE_MODULE_C=0", native)
    self.assertIn("CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=0", zero)
    self.assertNotIn("CANON_PROMPT_PROCESSED_LOGPROBS=0", zero)


if __name__ == "__main__":
  unittest.main()
