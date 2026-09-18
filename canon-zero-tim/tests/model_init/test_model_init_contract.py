"""CPU-only fail-closed tests for the model-init cluster mode."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
ENV_SCRIPT = PACKAGE / "cluster" / "steps" / "00_env.sh"
ENTRYPOINT = PACKAGE / "cluster" / "entrypoint.sh"


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      ["bash", str(script)],
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
  )


class ModelInitEnvironmentContractTest(unittest.TestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.state = Path(self.tempdir.name) / "state"
    self.state.mkdir()
    self.env = os.environ.copy()
    self.env.update(
        CANON_PKG=str(PACKAGE),
        CANON_STATE=str(self.state),
        CANON_MODE="model-init-only",
        CANON_PROFILE_FILE=(
            "cluster/profiles/qwen3-8b-dp16-tp4-model-init.env"
        ),
        CANON_REQUIRE_TRAIN_MESH_PIN="0",
        CANON_EXPECT_TRAIN_MESH_IDS="",
    )

  def tearDown(self):
    self.tempdir.cleanup()

  def test_dedicated_profile_is_accepted_without_training_admission(self):
    result = _run(ENV_SCRIPT, self.env)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("P32 model-init-only contract OK", result.stdout)
    env_text = (self.state / "env.sh").read_text(encoding="utf-8")
    self.assertIn("export CANON_P32_TRAIN_ADMITTED=0", env_text)
    self.assertIn("export CANON_P32_MODEL_INIT_ONLY=1", env_text)

  def test_admission_profile_without_model_init_gate_is_rejected(self):
    self.env["CANON_PROFILE_FILE"] = (
        "cluster/profiles/qwen3-8b-dp16-tp4-admission.env"
    )
    result = _run(ENV_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("MISSING: CANON_P32_MODEL_INIT_ONLY", result.stdout)

  def test_wrong_optimizer_memory_kind_is_rejected(self):
    profile = Path(self.tempdir.name) / "bad-memory.env"
    profile.write_text(
        f'source "{PACKAGE}/cluster/profiles/'
        'qwen3-8b-dp16-tp4-model-init.env"\n'
        'export CANON_P32_OPTIMIZER_MEMORY_KIND=device\n',
        encoding="utf-8",
    )
    self.env["CANON_PROFILE_FILE"] = str(profile)
    result = _run(ENV_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("requires pinned-host optimizer state", result.stdout)

  def test_unknown_mode_is_rejected_before_any_step(self):
    self.env["CANON_MODE"] = "typo-mode"
    result = _run(ENTRYPOINT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("unknown CANON_MODE: typo-mode", result.stdout)
    self.assertNotIn("--> 00_env.sh", result.stdout)


if __name__ == "__main__":
  unittest.main()
