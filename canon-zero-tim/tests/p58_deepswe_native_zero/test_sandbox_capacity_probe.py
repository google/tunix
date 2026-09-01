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
PVC_PROBE_PATH = ROOT / "canon-zero-tim/cluster/render_p58_head_pvc_probe.py"
PVC_SPEC = importlib.util.spec_from_file_location(
    "p58_head_pvc_probe", PVC_PROBE_PATH
)
if PVC_SPEC is None or PVC_SPEC.loader is None:
  raise RuntimeError("cannot import P58 head PVC probe renderer")
pvc_probe = importlib.util.module_from_spec(PVC_SPEC)
sys.modules[PVC_SPEC.name] = pvc_probe
PVC_SPEC.loader.exec_module(pvc_probe)


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
  *'get pod'*nodeSelector*) printf '%s' deepswe-cpu-pool-2 ;;
  *'get pod'*spec.nodeName*) printf '%s' node-a ;;
  *'get node node-a'*) printf '%s' deepswe-cpu-pool-2 ;;
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
        spec["nodeSelector"]["cloud.google.com/gke-nodepool"],
        "deepswe-cpu-pool-2",
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
          sandbox_nodepool="other-pool",
      )

  def test_runtime_selector_is_fail_closed_only_for_p58(self):
    from examples.deepswe import r2egym_runtime_patch

    self.assertEqual(
        r2egym_runtime_patch.resolve_node_selector_value({}), "cpu-np"
    )
    self.assertEqual(
        r2egym_runtime_patch.resolve_node_selector_value({
            "NODE_SELECTOR_VAL": "some-other-pool",
        }),
        "some-other-pool",
    )
    with self.assertRaisesRegex(ValueError, "deepswe-cpu-pool-2"):
      r2egym_runtime_patch.resolve_node_selector_value({
          "CANON_P58_DEEPSWE_TIM": "1",
          "NODE_SELECTOR_VAL": "cpu-np",
      })
    self.assertEqual(
        r2egym_runtime_patch.resolve_node_selector_value({
            "CANON_P58_DEEPSWE_TIM": "1",
            "NODE_SELECTOR_VAL": "deepswe-cpu-pool-2",
        }),
        "deepswe-cpu-pool-2",
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


class P58HeadPvcProbeTest(unittest.TestCase):

  def _run_verifier(self, *, phase: str, marker: bool = True):
    verifier = (
        ROOT / "canon-zero-tim/cluster/steps/p58_verify_head_pvc_probe.sh"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      fake = Path(temp_dir) / "kubectl"
      fake.write_text(
          """#!/bin/sh
case "$*" in
  *'get pod'*status.phase*) printf '%s' "${PROBE_PHASE}" ;;
  *'get pod'*schedulingGates*) printf '%s' '' ;;
  *'get pod'*queue-name*) printf '%s' multislice-queue ;;
  *'get pod'*managed*) printf '%s' true ;;
  *'get pod'*nodeSelector*) printf '%s' canon-cpu-pool ;;
  *'get pod'*claimName*) printf '%s' haoyugao-cpu-np-pvc ;;
  *'get pod'*readOnly*) printf '%s' true ;;
  *'get pod'*spec.nodeName*) printf '%s' head-node-a ;;
  *'get node head-node-a'*) printf '%s' canon-cpu-pool ;;
  *'logs '*'head-pvc-probe'*)
    if [ "${PROBE_MARKER}" = 1 ]; then
      printf '%s' 'P58_HEAD_PVC_MOUNT_PASS path=/mnt/disks/linchai_data/models/Qwen3-4B-Instruct-2507'
    fi
    ;;
  *) echo "unexpected fake kubectl call: $*" >&2; exit 9 ;;
esac
""",
          encoding="utf-8",
      )
      fake.chmod(0o755)
      env = os.environ.copy()
      env.update({
          "PATH": f"{temp_dir}:{env['PATH']}",
          "P58_HEAD_PVC_PROBE_POD": "canon-p58-head-pvc-probe-p58k30",
          "PROBE_PHASE": phase,
          "PROBE_MARKER": "1" if marker else "0",
      })
      return subprocess.run(
          ["bash", str(verifier)],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )

  def test_probe_is_exact_read_only_canon_head_mount(self):
    image = "registry.example/tunix@sha256:" + "2" * 64
    document = pvc_probe.render(run_id="p58k30", client_image=image)
    spec = document["spec"]
    container = spec["containers"][0]
    self.assertEqual(
        spec["nodeSelector"]["cloud.google.com/gke-nodepool"],
        "canon-cpu-pool",
    )
    self.assertEqual(
        document["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )
    self.assertEqual(
        spec["volumes"][0]["persistentVolumeClaim"],
        {"claimName": "haoyugao-cpu-np-pvc", "readOnly": True},
    )
    self.assertIs(container["volumeMounts"][0]["readOnly"], True)
    self.assertIn("Qwen3-4B-Instruct-2507", container["args"][1])

  def test_probe_rejects_wrong_pool_pvc_or_floating_image(self):
    image = "registry.example/tunix@sha256:" + "2" * 64
    with self.assertRaisesRegex(ValueError, "head pool"):
      pvc_probe.render(
          run_id="p58k30", client_image=image, head_nodepool="cpu-np"
      )
    with self.assertRaisesRegex(ValueError, "model PVC"):
      pvc_probe.render(
          run_id="p58k30", client_image=image, model_pvc="other-pvc"
      )
    with self.assertRaisesRegex(ValueError, "digest-pinned"):
      pvc_probe.render(
          run_id="p58k30", client_image="registry.example/tunix:latest"
      )

  def test_verifier_requires_completed_mount_marker(self):
    passed = self._run_verifier(phase="Succeeded")
    self.assertEqual(passed.returncode, 0, passed.stderr)
    self.assertIn(
        "P58_HEAD_PVC_PASS scope=canon-head-read-only-mount", passed.stdout
    )
    pending = self._run_verifier(phase="Pending")
    self.assertEqual(pending.returncode, 3)
    self.assertIn("reason=probe_not_complete", pending.stderr)
    missing = self._run_verifier(phase="Succeeded", marker=False)
    self.assertEqual(missing.returncode, 3)
    self.assertIn("reason=missing_mount_marker", missing.stderr)


if __name__ == "__main__":
  unittest.main()
