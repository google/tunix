#!/usr/bin/env python3
"""Contracts for fail-closed Pathways XProf GCS restoration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "canon-zero-tim/cluster/steps/xprof_gcs_sync_lib.sh"
RUN = ROOT / "canon-zero-tim/cluster/steps/90_run.sh"


class XprofGcsSyncTest(unittest.TestCase):

  def _run(self, root: Path, *, mode: str, attempt: str = "direct"):
    state = root / "state"
    state.mkdir()
    local_dir = (
        state / "xprof-update"
        if attempt == "direct"
        else state / f"attempt-{attempt}" / "xprof-update"
    )
    receipt = local_dir.parent / "xprof_gcs_restore.receipt"
    fake_bin = root / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -u
for argument in "$@"; do destination="$argument"; done
case "${FAKE_GCLOUD_MODE:?}" in
  complete)
    mkdir -p "$destination/plugins/profile/run"
    printf xplane > "$destination/plugins/profile/run/device.xplane.pb"
    printf trace > "$destination/plugins/profile/run/device.trace.json.gz"
    ;;
  xplane-only)
    mkdir -p "$destination/plugins/profile/run"
    printf xplane > "$destination/plugins/profile/run/device.xplane.pb"
    ;;
  error) exit 23 ;;
esac
""",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    remote = (
        "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/"
        f"state/attempt-{attempt}/xprof-update"
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_GCLOUD_MODE": mode,
        "CANON_STATE": str(state),
        "CANON_XPROF_DIR": remote,
    }
    command = (
        f"source {LIB}; "
        f"canon_xprof_gcs_restore {local_dir} {receipt}"
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, local_dir, receipt, remote

  def test_complete_capture_restores_both_artifact_types(self):
    with tempfile.TemporaryDirectory() as tmp:
      completed, local_dir, receipt, remote = self._run(
          Path(tmp), mode="complete"
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(len(list(local_dir.rglob("*.xplane.pb"))), 1)
      self.assertEqual(len(list(local_dir.rglob("*.trace.json.gz"))), 1)
      expected = (
          "[P51.XPROF.GCS] phase=restore status=PASS tool=gcloud rc=0 "
          f"xplanes=1 traces=1 remote={remote} local={local_dir}\n"
      )
      self.assertEqual(completed.stdout, expected)
      self.assertEqual(receipt.read_text(encoding="utf-8"), expected)

  def test_missing_trace_is_fail_closed_and_preserved(self):
    with tempfile.TemporaryDirectory() as tmp:
      completed, local_dir, receipt, _ = self._run(
          Path(tmp), mode="xplane-only"
      )
      self.assertNotEqual(completed.returncode, 0)
      self.assertTrue(local_dir.is_dir())
      self.assertIn(
          "status=MISSING_ARTIFACTS tool=gcloud rc=3 xplanes=1 traces=0",
          receipt.read_text(encoding="utf-8"),
      )

  def test_transport_error_is_fail_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      completed, _, receipt, _ = self._run(Path(tmp), mode="error")
      self.assertNotEqual(completed.returncode, 0)
      self.assertIn(
          "status=TRANSPORT_ERROR tool=gcloud rc=23",
          receipt.read_text(encoding="utf-8"),
      )

  def test_wrong_remote_identity_is_rejected_before_transport(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      state = root / "state"
      state.mkdir()
      local_dir = state / "xprof-update"
      receipt = state / "xprof_gcs_restore.receipt"
      env = {
          **os.environ,
          "CANON_STATE": str(state),
          "CANON_XPROF_DIR": "gs://wrong/xprof-update",
      }
      bad = subprocess.run(
          [
              "bash",
              "-c",
              f"source {LIB}; canon_xprof_gcs_restore {local_dir} {receipt}",
          ],
          env=env,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(bad.returncode, 0)
      self.assertIn("status=INVALID_CONTRACT", bad.stdout)

  def test_run_restores_before_full_classifier(self):
    text = RUN.read_text(encoding="utf-8")
    restore = text.index("canon_xprof_gcs_restore")
    classifier = text.index("classify_full_recipe.py")
    self.assertLess(restore, classifier)
    self.assertIn("--xprof-dir \"$xprof_local_dir\"", text)
    self.assertIn("--xprof-receipt \"$xprof_restore_receipt\"", text)


if __name__ == "__main__":
  unittest.main()
