# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for examples/deepswe/template.py."""

import os
import subprocess
import sys
from unittest import mock
import pytest
from examples.deepswe import template


def _setup_mock_agent_sandbox():
  mock_as = mock.MagicMock()
  mock_as.TemplateSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.ResourceSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  return mock_as


def test_prompt_constants_exist():
  """Verify that all DeepSWE prompt constants are defined and non-empty."""
  assert template.SWE_SYSTEM_PROMPT_FN_CALL
  assert template.SWE_SYSTEM_PROMPT
  assert template.SWEAGENT_SYSTEM_PROMPT
  assert template.OPENHANDS_SYSTEM_PROMPT
  assert template.SWE_USER_PROMPT_FN_CALL
  assert template.SWE_USER_PROMPT
  assert template.SWEAGENT_USER_PROMPT
  assert template.OPENHANDS_USER_PROMPT


def test_get_system_prompt():
  """Verify get_system_prompt returns correct prompt for all scaffolds and modes."""
  assert (
      template.get_system_prompt("r2egym", use_fn_calling=False)
      == template.SWE_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("r2egym", use_fn_calling=True)
      == template.SWE_SYSTEM_PROMPT_FN_CALL
  )
  assert (
      template.get_system_prompt("sweagent", use_fn_calling=False)
      == template.SWEAGENT_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("sweagent", use_fn_calling=True)
      == template.SWEAGENT_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("openhands", use_fn_calling=False)
      == template.OPENHANDS_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("openhands", use_fn_calling=True)
      == template.OPENHANDS_SYSTEM_PROMPT
  )


def test_get_user_prompt_template():
  """Verify get_user_prompt_template returns correct prompt for all scaffolds."""
  assert (
      template.get_user_prompt_template("r2egym", use_fn_calling=False)
      == template.SWE_USER_PROMPT
  )
  assert (
      template.get_user_prompt_template("r2egym", use_fn_calling=True)
      == template.SWE_USER_PROMPT_FN_CALL
  )
  assert (
      template.get_user_prompt_template("sweagent", use_fn_calling=False)
      == template.SWEAGENT_USER_PROMPT
  )
  assert (
      template.get_user_prompt_template("openhands", use_fn_calling=False)
      == template.OPENHANDS_USER_PROMPT
  )
  assert (
      template.get_user_prompt_template("openhands", use_fn_calling=True)
      == template.OPENHANDS_USER_PROMPT
  )


def test_get_openhands_pod_template_default():
  """Verify default openhands TemplateSpec construction."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {}, clear=True):
    pod_template = template.get_openhands_pod_template(
        node_selector={"node": "worker"}
    )
    assert pod_template is not None
    assert pod_template.node_selector == {"node": "worker"}
    assert "openhands-agent-server" in pod_template.keepalive_command[2]
    assert pod_template.resources.cpu == "500m"
    assert pod_template.resources.memory == "1Gi"
    container = pod_template.extra_pod_spec["containers"][0]
    assert container["resources"]["limits"]["cpu"] == "2"
    assert container["resources"]["limits"]["memory"] == "4Gi"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["ports"] == [{"containerPort": 8000}]
    assert container["env"] == []
    assert container["volumeMounts"] == [{"name": "oh", "mountPath": "/oh"}]
    init_c = pod_template.extra_pod_spec["initContainers"][0]
    assert init_c["name"] == "oh-server"
    assert "agent-server" in init_c["image"]
    assert "cp" in init_c["command"][2]
    assert init_c["volumeMounts"] == [{"name": "oh", "mountPath": "/oh"}]
    assert pod_template.extra_pod_spec["automountServiceAccountToken"] is False
    assert pod_template.extra_pod_spec["shareProcessNamespace"] is False
    assert pod_template.extra_pod_spec["volumes"] == [
        {"name": "oh", "emptyDir": {}}
    ]


def test_get_openhands_pod_template_with_overrides():
  """Verify openhands TemplateSpec honors env var overrides."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(
      os.environ,
      {
          "SANDBOX_SESSION_KEY": "secret_key_123",
          "AGENT_SERVER_COMMAND": '["custom", "entrypoint"]',
          "SANDBOX_CPU": "1",
          "SANDBOX_MEM": "2Gi",
          "SANDBOX_CPU_LIMIT": "4",
          "SANDBOX_MEM_LIMIT": "8Gi",
      },
  ):
    pod_template = template.get_openhands_pod_template()
    assert pod_template.keepalive_command == ["custom", "entrypoint"]
    assert pod_template.resources.cpu == "1"
    assert pod_template.resources.memory == "2Gi"
    container = pod_template.extra_pod_spec["containers"][0]
    assert container["resources"]["limits"]["cpu"] == "4"
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["env"] == [
        {"name": "OH_SESSION_API_KEYS_0", "value": "secret_key_123"}
    ]


def test_openhands_pod_template_keeps_venv_by_default():
  """Without SANDBOX_RELOCATE_VENV the keepalive command is unchanged."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {"SANDBOX_RELOCATE_VENV": "0"}, clear=True):
    pod_template = template.get_openhands_pod_template()
    assert pod_template.keepalive_command == (
        template.DEFAULT_OPENHANDS_KEEPALIVE_CMD
    )


def test_openhands_pod_template_relocates_venv():
  """SANDBOX_RELOCATE_VENV=1 prepends the relocation to the default command."""
  mock_as = _setup_mock_agent_sandbox()
  default_cmd = list(template.DEFAULT_OPENHANDS_KEEPALIVE_CMD)
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {"SANDBOX_RELOCATE_VENV": "true"}, clear=True):
    pod_template = template.get_openhands_pod_template()
  assert pod_template.keepalive_command == [
      "sh",
      "-c",
      template.RELOCATE_TESTBED_VENV_CMD + default_cmd[2],
  ]
  # The module-level default must not be mutated.
  assert template.DEFAULT_OPENHANDS_KEEPALIVE_CMD == default_cmd


def test_openhands_pod_template_rejects_bad_relocate_value():
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {"SANDBOX_RELOCATE_VENV": "yes"}, clear=True):
    with pytest.raises(ValueError, match="SANDBOX_RELOCATE_VENV"):
      template.get_openhands_pod_template()


def test_openhands_pod_template_relocate_conflicts_with_custom_command():
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(
      os.environ,
      {
          "SANDBOX_RELOCATE_VENV": "1",
          "AGENT_SERVER_COMMAND": '["custom", "entrypoint"]',
      },
      clear=True,
  ):
    with pytest.raises(ValueError, match="AGENT_SERVER_COMMAND"):
      template.get_openhands_pod_template()


def _relocate_cmd_for(root: str) -> str:
  """RELOCATE_TESTBED_VENV_CMD with /testbed and /opt rooted under `root`."""
  return (
      template.RELOCATE_TESTBED_VENV_CMD.replace(
          template.RELOCATED_TESTBED_VENV_PATH, "@DEST@"
      )
      .replace("/testbed", f"{root}/testbed")
      .replace("@DEST@", f"{root}/opt/testbed-venv")
  )


def _make_testbed(root: str) -> None:
  testbed = os.path.join(root, "testbed")
  os.makedirs(os.path.join(testbed, ".venv", "lib"))
  os.makedirs(os.path.join(root, "opt"))
  with open(os.path.join(testbed, ".venv", ".gitignore"), "w") as f:
    f.write("*\n")
  with open(os.path.join(testbed, ".venv", "lib", "dep.py"), "w") as f:
    f.write("x = 1\n")
  with open(os.path.join(testbed, "mod.py"), "w") as f:
    f.write("y = 2\n")
  subprocess.run(["git", "init", "-q", testbed], check=True)
  subprocess.run(["git", "-C", testbed, "add", "mod.py"], check=True)


def test_relocate_testbed_venv_cmd_moves_venv(tmp_path):
  """The shell snippet moves the venv, leaves a symlink and keeps git clean."""
  root = str(tmp_path)
  _make_testbed(root)
  testbed = os.path.join(root, "testbed")
  cmd = _relocate_cmd_for(root)

  subprocess.run(["sh", "-c", cmd], check=True)
  # Running it again (container restart) is a no-op.
  subprocess.run(["sh", "-c", cmd], check=True)

  venv_link = os.path.join(testbed, ".venv")
  assert os.path.islink(venv_link)
  assert os.readlink(venv_link) == f"{root}/opt/testbed-venv"
  assert os.path.isfile(os.path.join(venv_link, "lib", "dep.py"))
  with open(os.path.join(testbed, ".git", "info", "exclude")) as f:
    assert f.read().count("/.venv\n") == 1
  # `git add -A` (used for patch extraction) must not pick up the symlink.
  status = subprocess.run(
      ["git", "-C", testbed, "status", "--porcelain"],
      check=True,
      capture_output=True,
      text=True,
  ).stdout
  assert ".venv" not in status
  # `find` no longer descends into the venv.
  found = subprocess.run(
      ["find", testbed, "-name", "*.py"],
      check=True,
      capture_output=True,
      text=True,
  ).stdout.split()
  assert found == [os.path.join(testbed, "mod.py")]


def test_relocate_testbed_venv_cmd_fails_loudly(tmp_path):
  """If the destination is taken, the snippet exits non-zero."""
  root = str(tmp_path)
  _make_testbed(root)
  os.makedirs(os.path.join(root, "opt", "testbed-venv", "occupied"))
  res = subprocess.run(
      ["sh", "-c", _relocate_cmd_for(root) + "echo reached"],
      capture_output=True,
      text=True,
  )
  assert res.returncode == 1
  assert "relocating /testbed/.venv failed" in res.stderr.replace(root, "")
  assert "reached" not in res.stdout
  assert os.path.isdir(os.path.join(root, "testbed", ".venv"))
  assert not os.path.islink(os.path.join(root, "testbed", ".venv"))


def test_get_template_scaffolds():
  """Verify get_template delegates to openhands or returns None for other scaffolds."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {}, clear=True):
    assert template.get_template("openhands") is not None
    assert template.get_template("r2egym") is None
    assert template.get_template("sweagent") is None
