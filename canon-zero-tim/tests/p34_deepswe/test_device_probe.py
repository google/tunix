#!/usr/bin/env python3
"""Locks the step-65 Pathways session admission probe contract.

p39d7 connected to the proxy during an incomplete worker-registration window,
the server cancelled the session within a second, and the Stage 1 workload
fell into a single-CPU fallback.  The blind quiet period in step 60 proves
nothing, and the mode=run path never executes the step-70 assertion.  These
tests lock the repaired contract without any network or TPU:

  * the DeepSWE profile opts in with the whole-slice expectation (256), which
    is deliberately distinct from the per-role CANON_TOTAL_DEVICES value;
  * no other profile opts in (scope guard);
  * the entrypoint runs the probe between the quiet period and the workload;
  * the step skips cleanly when the expectation is unset;
  * a malformed expectation fails closed before any probe attempt;
  * an unsatisfiable expectation exhausts the bounded window and fails with
    the archive-the-proxy-logs remedy.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEP = ROOT / "cluster" / "steps" / "65_probe_devices.sh"
PROFILE = ROOT / "cluster" / "profiles" / "qwen3-32b-dp16-tp8-deepswe.env"
ENTRYPOINT = ROOT / "cluster" / "entrypoint.sh"


def _run_step(env_lines: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
  with tempfile.TemporaryDirectory() as state:
    pathlib.Path(state, "env.sh").write_text("\n".join(env_lines) + "\n")
    env = dict(os.environ, CANON_STATE=state, CANON_PKG=str(ROOT))
    return subprocess.run(
        ["bash", str(STEP)], env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


class DeviceProbeContractTest(unittest.TestCase):

  def test_deepswe_profile_opts_in_with_whole_slice_expectation(self):
    text = PROFILE.read_text()
    self.assertIn("export CANON_EXPECTED_SLICE_DEVICES=256", text)
    self.assertIn("export CANON_TOTAL_DEVICES=128", text)

  def test_no_other_profile_opts_in(self):
    for profile in sorted((ROOT / "cluster" / "profiles").glob("*.env")):
      if profile.name == PROFILE.name:
        continue
      self.assertNotIn(
          "CANON_EXPECTED_SLICE_DEVICES",
          profile.read_text(),
          f"{profile.name} must not enable the device probe yet",
      )

  def test_entrypoint_runs_probe_between_wait_and_workload(self):
    lines = [
        line.strip()
        for line in ENTRYPOINT.read_text().splitlines()
        if line.strip().startswith("step ")
    ]
    self.assertIn("step 65_probe_devices.sh", lines)
    index = lines.index("step 65_probe_devices.sh")
    self.assertEqual(lines[index - 1], "step 60_wait_workers.sh")
    self.assertEqual(lines[index + 1], "step 90_run.sh")

  def test_step_skips_when_expectation_unset(self):
    result = _run_step(["export CANON_STATE_MARKER=1"])
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("skipping", result.stdout)

  def test_step_fails_closed_on_malformed_expectation(self):
    result = _run_step(["export CANON_EXPECTED_SLICE_DEVICES=abc"])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("positive integer", result.stderr)

  def test_step_exhausts_bounded_window_with_remedy(self):
    result = _run_step([
        "export CANON_EXPECTED_SLICE_DEVICES=999999",
        "export CANON_DEVICE_PROBE_TIMEOUT_SECS=1",
        "export CANON_DEVICE_PROBE_INTERVAL_SECS=1",
    ])
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("expected 999999", result.stderr)
    self.assertIn("pathways-proxy", result.stderr)


if __name__ == "__main__":
  unittest.main()
