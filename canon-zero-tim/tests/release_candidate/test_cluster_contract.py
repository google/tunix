#!/usr/bin/env python3
"""CPU-only fail-closed tests for the DP16 release-candidate mode."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
ENV_SCRIPT = PACKAGE / "cluster" / "steps" / "00_env.sh"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      ["bash", str(ENV_SCRIPT)],
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
  )


class ClusterContractTest(unittest.TestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.state = Path(self.tempdir.name) / "state"
    self.state.mkdir()
    self.env = os.environ.copy()
    self.env.update(
        CANON_PKG=str(PACKAGE),
        CANON_STATE=str(self.state),
        CANON_MODE="dp16-rc",
        CANON_PROFILE_FILE="cluster/profiles/qwen3-8b-dp16-tp4-rc.env",
        CANON_REQUIRE_TRAIN_MESH_PIN="0",
        CANON_EXPECT_TRAIN_MESH_IDS="",
    )

  def tearDown(self):
    self.tempdir.cleanup()

  def test_default_checkpoint_forward_contract_is_accepted(self):
    result = _run(self.env)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("P32 dp16-rc contract OK", result.stdout)
    resolved = (self.state / "env.sh").read_text(encoding="utf-8")
    self.assertIn("export CANON_P32_RC=1", resolved)
    self.assertIn("export CANON_P32_RC_STAGE=checkpoint-forward", resolved)
    self.assertIn("export CANON_P32_TRAIN_ADMITTED=0", resolved)

  def test_invalid_stage_is_rejected(self):
    profile = Path(self.tempdir.name) / "bad-stage.env"
    profile.write_text(
        f'source "{PACKAGE}/cluster/profiles/qwen3-8b-dp16-tp4-rc.env"\n'
        'export CANON_P32_RC_STAGE=full-train\n',
        encoding="utf-8",
    )
    self.env["CANON_PROFILE_FILE"] = str(profile)
    result = _run(self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("invalid CANON_P32_RC_STAGE=full-train", result.stdout)

  def test_production_training_admission_is_rejected(self):
    profile = Path(self.tempdir.name) / "bad-training.env"
    profile.write_text(
        f'source "{PACKAGE}/cluster/profiles/qwen3-8b-dp16-tp4-rc.env"\n'
        'export CANON_P32_TRAIN_ADMITTED=1\n',
        encoding="utf-8",
    )
    self.env["CANON_PROFILE_FILE"] = str(profile)
    result = _run(self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("must not admit production training", result.stdout)

  def test_wrong_optimizer_memory_kind_is_rejected(self):
    profile = Path(self.tempdir.name) / "bad-memory.env"
    profile.write_text(
        f'source "{PACKAGE}/cluster/profiles/qwen3-8b-dp16-tp4-rc.env"\n'
        'export CANON_P32_OPTIMIZER_MEMORY_KIND=device\n',
        encoding="utf-8",
    )
    self.env["CANON_PROFILE_FILE"] = str(profile)
    result = _run(self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("requires pinned-host optimizer state", result.stdout)


if __name__ == "__main__":
  unittest.main()
