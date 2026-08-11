#!/usr/bin/env python3
"""Locks the pinned R2E-Gym provisioning contract (step 35).

p39d4 proved the container ships no r2egym; the reference MLPerf launch
cloned the floating upstream HEAD at runtime.  Step 35 replaces that with a
pinned commit plus a vendored patch.  These tests lock the wiring and the
fail-closed guards without touching the network:

  * the DeepSWE profile pins a 40-hex commit and enables the step;
  * no P33 profile enables the step (scope guard);
  * the entrypoint runs the step between install and overlay;
  * the vendored patch is byte-locked by SHA-256;
  * the step script skips cleanly when disabled and fails closed on a
    missing or malformed commit pin, before any network access.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEP = ROOT / "cluster" / "steps" / "35_install_r2egym.sh"
PATCH = ROOT / "patches" / "r2egym" / "r2egym.patch"
PROFILE = ROOT / "cluster" / "profiles" / "qwen3-32b-dp16-tp8-deepswe.env"
ENTRYPOINT = ROOT / "cluster" / "entrypoint.sh"

_PATCH_SHA256 = (
    "a9ef3683f5fdb236aae90d4d3290482066d40ad307b20fb2617ce70fa876ca70"
)


def _run_step(env_lines: list[str]) -> subprocess.CompletedProcess:
  with tempfile.TemporaryDirectory() as state:
    pathlib.Path(state, "env.sh").write_text("\n".join(env_lines) + "\n")
    env = dict(os.environ, CANON_STATE=state, CANON_PKG=str(ROOT))
    return subprocess.run(
        ["bash", str(STEP)], env=env, capture_output=True, text=True,
        timeout=60, check=False,
    )


class R2egymInstallStepContractTest(unittest.TestCase):

  def test_deepswe_profile_pins_install_and_commit(self):
    text = PROFILE.read_text()
    self.assertIn("export CANON_R2EGYM_INSTALL=1", text)
    for line in text.splitlines():
      if line.startswith("export CANON_R2EGYM_COMMIT="):
        value = line.split("=", 1)[1].strip()
        self.assertRegex(value, r"^[0-9a-f]{40}$")
        break
    else:
      self.fail("profile lacks CANON_R2EGYM_COMMIT")

  def test_no_p33_profile_enables_the_step(self):
    for profile in sorted((ROOT / "cluster" / "profiles").glob("*.env")):
      if profile.name == PROFILE.name:
        continue
      self.assertNotIn(
          "CANON_R2EGYM_INSTALL=1",
          profile.read_text(),
          f"{profile.name} must not enable the R2E-Gym install step",
      )

  def test_entrypoint_orders_step_between_install_and_overlay(self):
    lines = [
        line.strip()
        for line in ENTRYPOINT.read_text().splitlines()
        if line.strip().startswith("step ")
    ]
    self.assertIn("step 35_install_r2egym.sh", lines)
    self.assertLess(
        lines.index("step 30_install_canon.sh"),
        lines.index("step 35_install_r2egym.sh"),
    )
    self.assertLess(
        lines.index("step 35_install_r2egym.sh"),
        lines.index("step 40_overlay_engine.sh"),
    )

  def test_vendored_patch_is_byte_locked(self):
    digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    self.assertEqual(digest, _PATCH_SHA256)

  def test_step_skips_when_disabled(self):
    result = _run_step(["export CANON_R2EGYM_INSTALL=0"])
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("skipped", result.stdout)

  def test_step_fails_closed_without_commit_pin(self):
    result = _run_step(["export CANON_R2EGYM_INSTALL=1"])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("CANON_R2EGYM_COMMIT", result.stderr)

  def test_step_fails_closed_on_malformed_commit(self):
    result = _run_step([
        "export CANON_R2EGYM_INSTALL=1",
        "export CANON_R2EGYM_COMMIT=12e34567",
    ])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("40-hex", result.stderr)


if __name__ == "__main__":
  unittest.main()
