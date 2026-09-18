#!/usr/bin/env python3
"""Contracts for auditable best-effort JAX persistent-cache sync."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "canon-zero-tim/cluster/steps/jax_cache_sync_lib.sh"
RUN = ROOT / "canon-zero-tim/cluster/steps/90_run.sh"


class JaxCacheSyncTest(unittest.TestCase):

  def _run(
      self, root: Path, *, phase: str, mode: str, local_entry: bool = False
  ):
    state = root / "state"
    cache = root / "cache"
    fake_bin = root / "bin"
    state.mkdir()
    cache.mkdir()
    if local_entry:
      (cache / "local-key").write_text("compiled", encoding="utf-8")
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -u
case "${FAKE_GCLOUD_MODE:?}" in
  restore)
    for argument in "$@"; do destination="$argument"; done
    mkdir -p "$destination"
    printf 'compiled' > "$destination/cache-key"
    ;;
  pass) ;;
  error)
    echo 'fake transport failure' >&2
    exit 23
    ;;
esac
""",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_GCLOUD_MODE": mode,
        "CANON_STATE": str(state),
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env"
        ),
        "CANON_GCS_CACHE_BUCKET": (
            "gs://yuxzhang-tunix-models/cache/p33_compilation_cache"
        ),
        "JAX_COMPILATION_CACHE_DIR": str(cache),
    }
    completed = subprocess.run(
        ["bash", "-c", f"source {LIB}; canon_jax_cache_sync {phase}"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, state, cache

  def test_restore_hit_emits_and_persists_exact_receipt(self):
    with tempfile.TemporaryDirectory() as tmp:
      completed, state, cache = self._run(
          Path(tmp), phase="restore", mode="restore"
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue((cache / "cache-key").is_file())
      self.assertIn(
          "[JAX_CACHE_SYNC] phase=restore status=hit tool=gcloud rc=0 "
          "entries=1",
          completed.stdout,
      )
      self.assertEqual(
          (state / "jax_cache_restore.receipt").read_text(encoding="utf-8"),
          completed.stdout,
      )

  def test_transport_error_is_visible_but_not_a_numerical_gate(self):
    with tempfile.TemporaryDirectory() as tmp:
      completed, state, cache = self._run(
          Path(tmp), phase="save", mode="error", local_entry=True
      )
      self.assertEqual(completed.returncode, 0)
      self.assertIn("fake transport failure", completed.stderr)
      self.assertIn(
          "phase=save status=error tool=gcloud rc=23 entries=1",
          completed.stdout,
      )
      self.assertIn(
          "status=error",
          (state / "jax_cache_save.receipt").read_text(encoding="utf-8"),
      )

  def test_save_runs_before_fail_closed_postflight(self):
    text = RUN.read_text(encoding="utf-8")
    transport = text.index('echo "[run] transport_rc=$tee_rc"')
    save = text.index("canon_jax_cache_sync save")
    postflight = text.index(
        'if [ "${CANON_P46_EVALUATION:-0}" = "1" ]', save
    )
    self.assertLess(transport, save)
    self.assertLess(save, postflight)
    tail_save = text.rindex("canon_jax_cache_sync save")
    self.assertGreater(tail_save, postflight)
    self.assertEqual(text.count("canon_jax_cache_sync save"), 2)


if __name__ == "__main__":
  unittest.main()
