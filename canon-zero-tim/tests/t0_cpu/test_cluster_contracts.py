"""CPU-only negative controls for cluster configuration and provenance gates."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
ENV_SCRIPT = PACKAGE / "cluster" / "steps" / "00_env.sh"
SYNC_SCRIPT = PACKAGE / "cluster" / "steps" / "10_sync_repo.sh"
TRAIN_MESH_IDS = (
    "0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51,"
    "4,5,6,7,20,21,22,23,36,37,38,39,52,53,54,55,"
    "8,9,10,11,24,25,26,27,40,41,42,43,56,57,58,59,"
    "12,13,14,15,28,29,30,31,44,45,46,47,60,61,62,63"
)


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      ["bash", str(script)],
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
  )


class ClusterEnvironmentContractTest(unittest.TestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.state = Path(self.tempdir.name) / "state"
    self.state.mkdir()
    self.env = os.environ.copy()
    self.env.update(
        CANON_PKG=str(PACKAGE),
        CANON_STATE=str(self.state),
        CANON_PROFILE_FILE="cluster/profiles/qwen3-8b-dp16-tp4-admission.env",
        CANON_REQUIRE_TRAIN_MESH_PIN="1",
        CANON_EXPECT_TRAIN_MESH_IDS=TRAIN_MESH_IDS,
    )

  def tearDown(self):
    self.tempdir.cleanup()

  def test_pinned_train_mesh_is_accepted(self):
    result = _run(ENV_SCRIPT, self.env)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("P32 train mesh pin OK: 64 unique ids", result.stdout)

  def test_required_but_missing_train_mesh_is_rejected(self):
    self.env["CANON_EXPECT_TRAIN_MESH_IDS"] = ""
    result = _run(ENV_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("MISSING: CANON_EXPECT_TRAIN_MESH_IDS", result.stdout)

  def test_discovery_manifest_accepts_missing_train_mesh(self):
    self.env["CANON_REQUIRE_TRAIN_MESH_PIN"] = "0"
    self.env["CANON_EXPECT_TRAIN_MESH_IDS"] = ""
    result = _run(ENV_SCRIPT, self.env)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("P32 train mesh pin DISCOVERY", result.stdout)

  def test_duplicate_train_mesh_id_is_rejected(self):
    self.env["CANON_EXPECT_TRAIN_MESH_IDS"] = TRAIN_MESH_IDS.rsplit(",", 1)[0] + ",62"
    result = _run(ENV_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("must contain 64 unique ids; got 63", result.stdout)


class RepositoryProvenanceContractTest(unittest.TestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.repo = Path(self.tempdir.name) / "repo"
    self.package = self.repo / "canon-zero-tim"
    self.state = Path(self.tempdir.name) / "state"
    self.package.mkdir(parents=True)
    self.state.mkdir()
    (self.state / "env.sh").write_text("", encoding="utf-8")
    (self.package / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
    subprocess.run(
        ["git", "-C", str(self.repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(self.repo), "config", "user.name", "Cluster Contract Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(self.repo), "add", "canon-zero-tim/tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True)
    self.env = os.environ.copy()
    self.env.pop("CANON_EXPECT_COMMIT", None)
    self.env.pop("CANON_ALLOW_UNVERSIONED", None)
    self.env.update(CANON_PKG=str(self.package), CANON_STATE=str(self.state))

  def tearDown(self):
    self.tempdir.cleanup()

  def test_external_untracked_file_is_reported_but_allowed(self):
    (self.repo / "image-cache.txt").write_text("image-owned\n", encoding="utf-8")
    result = _run(SYNC_SCRIPT, self.env)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("tracked_dirty=0", result.stdout)
    self.assertIn("package_untracked=0", result.stdout)
    self.assertIn("external_untracked=1", result.stdout)

  def test_tracked_change_is_rejected(self):
    (self.package / "tracked.txt").write_text("changed\n", encoding="utf-8")
    result = _run(SYNC_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("REFUSING: tracked files differ from HEAD", result.stdout)

  def test_package_untracked_file_is_rejected(self):
    (self.package / "shadow.py").write_text("# shadow\n", encoding="utf-8")
    result = _run(SYNC_SCRIPT, self.env)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("REFUSING: untracked files can shadow", result.stdout)


if __name__ == "__main__":
  unittest.main()
