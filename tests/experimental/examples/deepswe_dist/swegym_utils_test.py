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

"""Unit tests for SWE-Gym and SWE-smith rollout setup and fresh-container evaluation."""

import json

from absl.testing import absltest
from examples.deepswe import swe_env
from tunix.experimental.examples.deepswe_dist import openhands_utils
from tunix.experimental.examples.deepswe_dist import sandbox_utils
from tunix.experimental.examples.deepswe_dist import swegym_utils

_TEST_PATCH = """diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -1 +1,2 @@
 x = 1
+y = 2
diff --git a/tests/data/new.json b/tests/data/new.json
new file mode 100644
--- /dev/null
+++ b/tests/data/new.json
@@ -0,0 +1 @@
+{}
"""


class FakeRuntime:
  """Records commands; returns canned output for the eval script run."""

  def __init__(self, test_output: str):
    self.test_output = test_output
    self.cmds = []

  def run(self, cmd, timeout=90):
    del timeout
    self.cmds.append(cmd)
    if "bash /tmp/.swegym_eval.sh" in cmd:
      return self.test_output, "Error: Exit code 1"
    if "pytest" in cmd:
      return self.test_output, "0"
    return "", "0"


class SwegymAndSwesmithTest(absltest.TestCase):

  def test_is_swegym_entry(self):
    self.assertTrue(swegym_utils.is_swegym_entry({"task_source": "swegym"}))
    self.assertFalse(
        swegym_utils.is_swegym_entry(
            {"task_source": "swesmith", "docker_image": "xingyaoww/sweb.eval.x"}
        )
    )
    self.assertTrue(
        swegym_utils.is_swegym_entry({
            "docker_image": (
                "us-central1-docker.pkg.dev/p/tunix/"
                "sweb.eval.x86_64.getmoto_s_moto-1:latest"
            )
        })
    )
    self.assertFalse(
        swegym_utils.is_swegym_entry({"docker_image": "namanjain12/foo:1"})
    )
    self.assertFalse(swegym_utils.is_swegym_entry(None))

  def test_source_image_for_instance(self):
    self.assertEqual(
        swegym_utils.source_image_for_instance("Project-MONAI__MONAI-1065"),
        "xingyaoww/sweb.eval.x86_64.project-monai_s_monai-1065:latest",
    )

  def test_test_command_fallback(self):
    entry = {"repo": "getmoto/moto", "test_patch": _TEST_PATCH}
    self.assertEqual(
        swegym_utils.get_test_command(entry), "pytest -n0 -rA tests/test_a.py"
    )
    mypy = {
        "repo": "python/mypy",
        "test_patch": "+[case testFoo]\n+[case testBar-xfail]\n",
    }
    self.assertEqual(
        swegym_utils.get_test_command(mypy),
        'pytest -n0 -rA -k "testFoo or testBar-xfail"',
    )
    self.assertEqual(
        swegym_utils.get_test_command({"test_cmd": "pytest x", "repo": "a/b"}),
        "pytest x",
    )
    self.assertEqual(
        swegym_utils.get_eval_commands({"repo": "conan-io/conan"}),
        ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
    )
    self.assertEqual(
        swegym_utils.get_eval_commands(
            {"repo": "conan-io/conan", "eval_commands": "[]"}
        ),
        [],
    )

  def test_parse_test_log(self):
    log = (
        "PASSED tests/test_a.py::test_ok\n"
        "FAILED tests/test_a.py::test_bad - AssertionError: boom\n"
        "XFAIL tests/test_a.py::test_x - reason\n"
    )
    status = swegym_utils.parse_test_log(log, "getmoto/moto")
    self.assertEqual(status["tests/test_a.py::test_ok"], "PASSED")
    self.assertEqual(status["tests/test_a.py::test_bad"], "FAILED")
    self.assertEqual(status["tests/test_a.py::test_x"], "XFAIL")
    pyd = swegym_utils.parse_test_log(
        "tests/test_main.py::test_y PASSED\nFAILED [ 50%] tests/t.py::z\n",
        "pydantic/pydantic",
    )
    self.assertEqual(pyd["tests/test_main.py::test_y"], "PASSED")
    self.assertEqual(pyd["tests/t.py::z"], "FAILED")

  def test_is_resolved(self):
    sm = {"a": "PASSED", "b": "XFAIL", "c": "FAILED"}
    self.assertTrue(swegym_utils.is_resolved(sm, ["a"], ["b"]))
    self.assertFalse(swegym_utils.is_resolved(sm, ["a"], ["c"]))
    self.assertFalse(swegym_utils.is_resolved(sm, ["missing"], []))
    self.assertFalse(swegym_utils.is_resolved(sm, [], ["a"]))

  def test_eval_script_resets_and_applies_test_patch(self):
    script = swegym_utils.build_eval_script(
        {"repo": "getmoto/moto", "test_patch": _TEST_PATCH}
    )
    self.assertIn("git checkout -q HEAD -- tests/test_a.py", script)
    self.assertIn("rm -f -- tests/data/new.json", script)
    self.assertIn("git apply -v /tmp/.swegym_test.patch", script)
    self.assertIn("pytest -n0 -rA tests/test_a.py", script)
    self.assertLess(
        script.index(swegym_utils.START_TEST_OUTPUT),
        script.index("pytest -n0 -rA"),
    )

  def _entry(self):
    return {
        "instance_id": "getmoto__moto-1",
        "repo": "getmoto/moto",
        "task_source": "swegym",
        "test_patch": _TEST_PATCH,
        "FAIL_TO_PASS": json.dumps(["tests/test_a.py::test_f"]),
        "PASS_TO_PASS": json.dumps(["tests/test_a.py::test_p"]),
    }

  def test_evaluate_resolved(self):
    out = (
        f"noise\n{swegym_utils.START_TEST_OUTPUT}\n"
        "PASSED tests/test_a.py::test_f\nPASSED tests/test_a.py::test_p\n"
        f"{swegym_utils.END_TEST_OUTPUT}\n"
    )
    rt = FakeRuntime(out)
    self.assertEqual(
        swegym_utils.evaluate_swegym_in_fresh_container(
            rt, self._entry(), "diff --git a/m.py b/m.py\n"
        ),
        1.0,
    )
    self.assertTrue(any("/tmp/model.patch" in c for c in rt.cmds))

  def test_evaluate_unresolved_and_test_patch_failure(self):
    out = (
        f"{swegym_utils.START_TEST_OUTPUT}\n"
        "FAILED tests/test_a.py::test_f - x\nPASSED tests/test_a.py::test_p\n"
    )
    self.assertEqual(
        swegym_utils.evaluate_swegym_in_fresh_container(
            FakeRuntime(out), self._entry(), ""
        ),
        0.0,
    )
    self.assertEqual(
        swegym_utils.evaluate_swegym_in_fresh_container(
            FakeRuntime(swegym_utils.TEST_PATCH_FAILED), self._entry(), ""
        ),
        0.0,
    )

  def test_rollout_setup_cmds_prune_history(self):
    cmd = " && ".join(swegym_utils.build_rollout_setup_cmds())
    for needle in ("--orphan", "update-ref -d", "reflog expire", "gc -q"):
      self.assertIn(needle, cmd)

  def test_swesmith_detection_image_extraction_and_grading(self):
    smith_entry = {
        "instance_id": "org__repo.1234",
        "image_name": "swesmith.x86_64.org__repo.1234",
        "patch": "diff --git a/x.py b/x.py\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_bug", "tests/test_x.py::test_2"],
        "PASS_TO_PASS": ["tests/test_x.py::test_ok"],
    }
    self.assertTrue(openhands_utils.is_swesmith_entry(smith_entry))
    self.assertEqual(
        sandbox_utils.extract_task_image(smith_entry),
        "jyangballin/swesmith.x86_64.org_1776_repo.1234:latest",
    )
    normalized = swegym_utils.normalize_swe_entry(dict(smith_entry))
    self.assertEqual(
        normalized["docker_image"],
        "jyangballin/swesmith.x86_64.org_1776_repo.1234:latest",
    )
    self.assertEqual(normalized["image_name"], "swesmith.x86_64.org__repo.1234")
    unpacked = swe_env._unpack_entry(normalized)
    self.assertEqual(len(unpacked["FAIL_TO_PASS"]), 2)

    pytest_out = (
        "=========================== short test summary info ===========================\n"
        "PASSED tests/test_x.py::test_bug\n"
        "PASSED tests/test_x.py::test_2\n"
        "PASSED tests/test_x.py::test_ok\n"
    )
    rt = FakeRuntime(pytest_out)
    self.assertEqual(
        openhands_utils.evaluate_swesmith_in_fresh_container(
            rt, normalized, "diff --git a/x.py b/x.py\n"
        ),
        1.0,
    )

  def test_swegym_entry_normalization_and_fleet_extraction(self):
    gym_entry = {
        "instance_id": "getmoto__moto-1",
        "repo": "getmoto/moto",
        "task_source": "swegym",
        "base_commit": "abc1234",
    }
    self.assertEqual(
        sandbox_utils.extract_task_image(gym_entry),
        "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-1:latest",
    )
    normalized = swegym_utils.normalize_swe_entry(dict(gym_entry))
    self.assertEqual(
        normalized["docker_image"],
        "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-1:latest",
    )
    self.assertEqual(normalized["repo_name"], "getmoto/moto")
    self.assertEqual(
        json.loads(normalized["parsed_commit_content"])["old_commit_hash"],
        "abc1234",
    )


if __name__ == "__main__":
  absltest.main()
