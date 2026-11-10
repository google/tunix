# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for git history scrubbing and patch extraction in openhands_utils."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.examples.deepswe_dist import openhands_utils


@dataclasses.dataclass
class FakeCommandResult:
  stdout: str
  stderr: str
  exit_code: int


class RewrittenPathWorkspace:
  """Executes OpenHands commands locally while mapping `/testbed` to a tempdir."""

  def __init__(self, repo_dir: str, home_dir: str):
    self._repo_dir = repo_dir
    self._env = {**os.environ, "HOME": home_dir}
    self.commands: list[str] = []

  def execute_command(
      self, command: str, timeout: float = 60.0
  ) -> FakeCommandResult:
    self.commands.append(command)
    rewritten = command.replace("/testbed", self._repo_dir)
    proc = subprocess.run(
        ["/bin/sh", "-c", rewritten],
        env=self._env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return FakeCommandResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
    )


class FailingWorkspace:
  """Workspace stub whose setup command fails."""

  def execute_command(
      self, command: str, timeout: float = 60.0
  ) -> FakeCommandResult:
    del command, timeout
    return FakeCommandResult(stdout="", stderr="permission denied", exit_code=1)


def _create_cloned_repo_with_future_commit(
    root_dir: str,
) -> tuple[str, str, str]:
  """Creates a packed clone checked out at `future^`, mirroring R2E-Gym builds."""
  upstream = os.path.join(root_dir, "upstream")
  clone = os.path.join(root_dir, "clone")
  home = os.path.join(root_dir, "home")
  os.makedirs(upstream)
  os.makedirs(home)
  env = {**os.environ, "HOME": home}

  def _git(cwd: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, env=env, text=True
    ).strip()

  _git(upstream, "init", "-b", "main")
  _git(upstream, "config", "user.email", "jdoe@example.com")
  _git(upstream, "config", "user.name", "John Doe")
  pathlib.Path(upstream, "module.py").write_text(
      "def compute() -> int:\n  return 1\n", encoding="utf-8"
  )
  _git(upstream, "add", ".")
  _git(upstream, "commit", "-m", "initial commit")
  base_sha = _git(upstream, "rev-parse", "HEAD")

  pathlib.Path(upstream, "module.py").write_text(
      "def compute() -> int:\n  return 42  # FUTURE_GOLD_SENTINEL\n",
      encoding="utf-8",
  )
  _git(upstream, "add", ".")
  _git(upstream, "commit", "-m", "FUTURE_COMMIT_MESSAGE_SENTINEL")
  _git(upstream, "tag", "v2.0.0")
  _git(upstream, "gc", "--quiet")
  future_sha = _git(upstream, "rev-parse", "HEAD")

  subprocess.run(
      ["git", "clone", "--quiet", f"file://{upstream}", clone],
      env=env,
      check=True,
  )
  _git(clone, "checkout", "--quiet", f"{future_sha}^")
  return clone, base_sha, future_sha


class OpenHandsUtilsScrubTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("setup_openhands_workspace", "openhands"),
      ("hide_r2e_tests_for_rollout", "r2egym"),
  )
  def test_workspace_init_scrubs_future_commits_and_preserves_patch_extraction(
      self, mode: str
  ):
    if shutil.which("git") is None:
      self.skipTest("git binary is unavailable in test environment")

    root_dir = self.create_tempdir().full_path
    clone, base_sha, future_sha = _create_cloned_repo_with_future_commit(
        root_dir
    )
    ws = RewrittenPathWorkspace(clone, os.path.join(root_dir, "ws_home"))
    os.makedirs(os.path.join(root_dir, "ws_home"), exist_ok=True)
    entry: dict[str, str] = {}

    if mode == "openhands":
      openhands_utils.setup_openhands_workspace(ws, entry)
    else:
      openhands_utils.hide_r2e_tests_for_rollout(ws, entry)

    self.assertEqual(entry["base_commit"], base_sha)

    refs = subprocess.check_output(
        ["git", "-C", clone, "for-each-ref", "--format=%(refname)"],
        text=True,
    ).strip()
    self.assertEqual(refs, "")

    reflog = subprocess.check_output(
        ["git", "-C", clone, "reflog"], text=True
    ).strip()
    self.assertEqual(reflog, "")

    all_objects = subprocess.check_output(
        [
            "git",
            "-C",
            clone,
            "cat-file",
            "--batch-check",
            "--batch-all-objects",
        ],
        text=True,
    )
    self.assertNotIn(future_sha, all_objects)

    show_future = subprocess.run(
        ["git", "-C", clone, "show", future_sha],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(show_future.returncode, 0)

    # Verify agent edits extract cleanly against the preserved base_commit.
    pathlib.Path(clone, "module.py").write_text(
        "def compute() -> int:\n  return 7\n", encoding="utf-8"
    )
    patch = openhands_utils.extract_agent_patch(
        ws,
        base_commit=base_sha,
        workspace_path=clone,
    )
    self.assertIn("-  return 1", patch)
    self.assertIn("+  return 7", patch)
    self.assertNotIn("FUTURE_GOLD_SENTINEL", patch)

  @parameterized.named_parameters(
      ("setup_openhands_workspace", "openhands"),
      ("hide_r2e_tests_for_rollout", "r2egym"),
  )
  def test_workspace_init_fails_fast_when_scrub_command_errors(self, mode: str):
    ws = FailingWorkspace()
    with self.assertRaisesRegex(RuntimeError, "exit"):
      if mode == "openhands":
        openhands_utils.setup_openhands_workspace(ws, {})
      else:
        openhands_utils.hide_r2e_tests_for_rollout(ws, {})


if __name__ == "__main__":
  absltest.main()
