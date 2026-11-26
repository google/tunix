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

"""Unit tests for tunix.experimental.examples.deepswe_dist.openhands_utils."""

from __future__ import annotations

import base64
import dataclasses
import re
import shlex

from absl.testing import absltest
from tunix.experimental.examples.deepswe_dist import openhands_utils


@dataclasses.dataclass
class _FakeClient:
  _default_container: str = "agent-runtime"


class _FakeRuntime:

  def __init__(self, existing_paths: set[str] | None = None):
    self.repo_path = "/testbed"
    self._target_container = "agent-runtime"
    self._container = "agent-runtime"
    self.client = _FakeClient()
    self._defer_setup_env = True
    self.existing_paths = existing_paths or set()
    self.commands: list[tuple[str, str]] = []
    self.applied_patches: list[str] = []
    self.setup_env_calls = 0

  def run(self, cmd: str, timeout: int = 30) -> tuple[str, int]:
    del timeout
    self.commands.append((self._container, cmd))
    argv = shlex.split(cmd)
    inner = argv[2] if len(argv) >= 3 and argv[:2] == ["/bin/sh", "-c"] else cmd

    if "for p in " in inner:
      m = re.search(r"for p in (.+?); do", inner)
      if m:
        candidates = shlex.split(m.group(1))
        found = [p for p in candidates if p in self.existing_paths]
        return "\n".join(found), 0

    if "git apply" in inner:
      m = re.search(r"printf '%s' (\S+) \| base64 -d > /tmp/model\.patch", inner)
      if m:
        b64_text = shlex.split(m.group(1))[0]
        self.applied_patches.append(
            base64.b64decode(b64_text).decode("utf-8")
        )
      return "", 0
    return "", 0

  def setup_env(self) -> None:
    self.setup_env_calls += 1


class _FakeEvalEnv:

  def __init__(self, runtime: _FakeRuntime):
    self.runtime = runtime


class OpenhandsUtilsTest(absltest.TestCase):

  def test_new_file_paths_in_patch_standard_empty_and_escaped(self):
    patch = "\n".join([
        "diff --git a/mod.py b/mod.py",
        "index 1111111..2222222 100644",
        "--- a/mod.py",
        "+++ b/mod.py",
        "@@ -1 +1 @@",
        "-a",
        "+b",
        "diff --git a/del.py b/del.py",
        "deleted file mode 100644",
        "--- a/del.py",
        "+++ /dev/null",
        "diff --git a/added.py b/added.py",
        "new file mode 100644",
        "index 0000000..1111111",
        "--- /dev/null",
        "+++ b/added.py",
        "@@ -0,0 +1 @@",
        "+x = 1",
        "diff --git a/empty_file.txt b/empty_file.txt",
        "new file mode 100644",
        "index 0000000..e69de29",
        'diff --git "a/dir/caf\\303\\251\\tfile.txt" "b/dir/caf\\303\\251\\tfile.txt"',
        "new file mode 100644",
        "index 0000000..1111111",
        "--- /dev/null",
        '+++ "b/dir/caf\\303\\251\\tfile.txt"',
        "@@ -0,0 +1 @@",
        "+hi",
        "diff --git a/trailing_empty.txt b/trailing_empty.txt",
        "new file mode 100644",
        "index 0000000..e69de29",
    ])
    self.assertEqual(
        openhands_utils._new_file_paths_in_patch(patch),
        [
            "added.py",
            "empty_file.txt",
            "dir/café\tfile.txt",
            "trailing_empty.txt",
        ],
    )

  def test_find_existing_paths_chunks_large_path_lists(self):
    runtime = _FakeRuntime(existing_paths={"path_0005.txt", "path_0045.txt"})
    paths = [f"path_{i:04d}.txt" for i in range(50)]
    found = openhands_utils._find_existing_paths_in_container(
        runtime, "/testbed", paths, chunk_max_bytes=200
    )
    self.assertEqual(found, ["path_0005.txt", "path_0045.txt"])
    self.assertGreater(len(runtime.commands), 1)

  def test_evaluate_patch_in_fresh_container_switches_container_and_drops_existing(
      self,
  ):
    runtime = _FakeRuntime(existing_paths={"existing.py"})
    env = _FakeEvalEnv(runtime)
    patch = "\n".join([
        "diff --git a/existing.py b/existing.py",
        "new file mode 100644",
        "--- /dev/null",
        "+++ b/existing.py",
        "@@ -0,0 +1 @@",
        "+1",
        "diff --git a/fresh.py b/fresh.py",
        "new file mode 100644",
        "--- /dev/null",
        "+++ b/fresh.py",
        "@@ -0,0 +1 @@",
        "+2",
    ])

    reward = openhands_utils.evaluate_patch_in_fresh_container(
        env, patch, orig_compute_reward=lambda: 1.0
    )
    self.assertEqual(reward, 1.0)
    self.assertEqual(runtime._target_container, "agent-runtime")
    self.assertEqual(runtime._container, "agent-runtime")
    self.assertEqual(runtime.setup_env_calls, 1)

    self.assertTrue(all(c == "eval" for c, _ in runtime.commands))
    self.assertLen(runtime.applied_patches, 1)
    self.assertNotIn("existing.py", runtime.applied_patches[0])
    self.assertIn("fresh.py", runtime.applied_patches[0])
    apply_cmds = [cmd for _, cmd in runtime.commands if "git apply" in cmd]
    self.assertLen(apply_cmds, 1)
    self.assertNotIn("--exclude", apply_cmds[0])
    self.assertNotIn("git ls-files", "".join(cmd for _, cmd in runtime.commands))

  def test_evaluate_patch_skips_git_apply_when_all_new_files_exist(self):
    num_files = 2500
    existing = {f"artifact_dir/existing_{i:04d}.txt" for i in range(num_files)}
    runtime = _FakeRuntime(existing_paths=existing)
    env = _FakeEvalEnv(runtime)
    patch = "\n".join(
        f"diff --git a/{p} b/{p}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{p}\n"
        "@@ -0,0 +1 @@\n"
        "+x"
        for p in sorted(existing)
    )

    reward = openhands_utils.evaluate_patch_in_fresh_container(
        env, patch, orig_compute_reward=lambda: 1.0
    )
    self.assertEqual(reward, 1.0)
    self.assertEqual(runtime.setup_env_calls, 1)
    self.assertEmpty(runtime.applied_patches)
    self.assertFalse(any("git apply" in cmd for _, cmd in runtime.commands))


if __name__ == "__main__":
  absltest.main()
