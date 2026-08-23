#!/usr/bin/env python3
"""Contracts for the P58 one-sandbox Kueue admission probe."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "canon-zero-tim/cluster/render_p58_sandbox_probe.py"
SPEC = importlib.util.spec_from_file_location("p58_sandbox_probe", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 sandbox probe renderer")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class P58SandboxCapacityProbeTest(unittest.TestCase):

  def _run_verifier(self, *, phase: str, managed: str = "true"):
    verifier = (
        ROOT
        / "canon-zero-tim/cluster/steps/p58_verify_sandbox_capacity.sh"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      fake = Path(temp_dir) / "kubectl"
      fake.write_text(
          """#!/bin/sh
case "$*" in
  *localqueue*spec.clusterQueue*) printf '%s' cluster-q ;;
  *localqueue*Active*) printf '%s' True ;;
  *clusterqueue*Active*) printf '%s' True ;;
  *'get nodes'*) printf '%s\\n' 'node-a||True' ;;
  *'get pod'*status.phase*) printf '%s' "${PROBE_PHASE}" ;;
  *'get pod'*schedulingGates*) printf '%s' "${PROBE_GATES:-}" ;;
  *'get pod'*queue-name*) printf '%s' multislice-queue ;;
  *'get pod'*managed*) printf '%s' "${PROBE_MANAGED}" ;;
  *'get pod'*nodeSelector*) printf '%s' cpu-np ;;
  *'get pod'*spec.nodeName*) printf '%s' node-a ;;
  *'get node node-a'*) printf '%s' cpu-np ;;
  *) echo "unexpected fake kubectl call: $*" >&2; exit 9 ;;
esac
""",
          encoding="utf-8",
      )
      fake.chmod(0o755)
      env = os.environ.copy()
      env.update({
          "PATH": f"{temp_dir}:{env['PATH']}",
          "P58_SANDBOX_PROBE_POD": "canon-p58-sandbox-probe-p58f13",
          "PROBE_PHASE": phase,
          "PROBE_MANAGED": managed,
      })
      return subprocess.run(
          ["bash", str(verifier)],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )

  def test_probe_matches_production_sandbox_admission_shape(self):
    document = probe.render(
        run_id="p58f13",
        task_image="example.invalid/r2e-task:reviewed",
    )
    labels = document["metadata"]["labels"]
    spec = document["spec"]
    container = spec["containers"][0]
    self.assertEqual(
        labels["kueue.x-k8s.io/queue-name"], "multislice-queue"
    )
    self.assertEqual(
        spec["nodeSelector"]["cloud.google.com/gke-nodepool"], "cpu-np"
    )
    self.assertEqual(container["resources"]["requests"], {
        "cpu": "2",
        "memory": "4Gi",
    })
    self.assertEqual(container["resources"]["limits"], {
        "cpu": "4",
        "memory": "8Gi",
    })
    self.assertEqual(spec["activeDeadlineSeconds"], 900)
    self.assertIs(spec["automountServiceAccountToken"], False)

  def test_probe_rejects_unsigned_queue_or_nodepool(self):
    with self.assertRaisesRegex(ValueError, "requires queue"):
      probe.render(
          run_id="p58f13",
          task_image="example.invalid/r2e-task:reviewed",
          queue_name="other-queue",
      )
    with self.assertRaisesRegex(ValueError, "requires node pool"):
      probe.render(
          run_id="p58f13",
          task_image="example.invalid/r2e-task:reviewed",
          cpu_nodepool="other-pool",
      )

  def test_probe_rejects_invalid_identity_or_image(self):
    with self.assertRaisesRegex(ValueError, "run_id"):
      probe.render(run_id="Bad Run", task_image="example.invalid/task:tag")
    with self.assertRaisesRegex(ValueError, "task_image"):
      probe.render(run_id="p58f13", task_image="")

  def test_live_probe_verifier_passes_only_after_real_admission(self):
    passed = self._run_verifier(phase="Running")
    self.assertEqual(passed.returncode, 0, passed.stderr)
    self.assertIn(
        "P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only",
        passed.stdout,
    )

    blocked = self._run_verifier(phase="Pending")
    self.assertEqual(blocked.returncode, 3)
    self.assertIn(
        "P58_SANDBOX_CAPACITY_BLOCKED reason=probe_not_admitted",
        blocked.stderr,
    )

    unmanaged = self._run_verifier(phase="Running", managed="false")
    self.assertEqual(unmanaged.returncode, 3)
    self.assertIn("kueue_managed=false", unmanaged.stderr)


if __name__ == "__main__":
  unittest.main()
