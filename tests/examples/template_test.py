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

import hashlib
import json
import os
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


def test_get_template_scaffolds():
  """Verify get_template delegates to openhands or returns None for other scaffolds."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as}
  ), mock.patch.dict(os.environ, {}, clear=True):
    assert template.get_template("openhands") is not None
    assert template.get_template("r2egym") is None
    assert template.get_template("sweagent") is None


def test_get_openhands_tools_matches_nv_openhands_reference():
  """Verify OpenHands tool schemas and rendered JSON match nv-OpenHands@0d766ad0."""
  tools = template.get_openhands_tools(
      max_timeout=60,
      workspace_mount_path_in_sandbox="/openhands_setup/OpenHands",
      enable_think=True,
      enable_task_tracker=True,
  )
  expected_required = {
      "execute_bash": ["command", "security_risk"],
      "think": ["thought"],
      "finish": ["message"],
      "task_tracker": ["command"],
      "str_replace_editor": ["command", "path", "security_risk"],
  }
  # Exact SHA-256 digests of json.dumps(tool, ensure_ascii=False) produced by
  # nv-OpenHands@0d766ad0 tool modules (bash.py, think.py, finish.py,
  # task_tracker.py, str_replace_editor.py) with COMMAND_EXEC_TIMEOUT=60 and
  # DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX="/openhands_setup/OpenHands".
  expected_sha256 = {
      "execute_bash": (
          "641b555d0056fc5b43e55c17ffa744887835d8ee55d3598e1850a55bc7595fe1"
      ),
      "think": (
          "6366dd459e63d6fa486ded5dc13307f558ecb790a4e85692815810214be1d602"
      ),
      "finish": (
          "3879fd696325f68981a5fb272fcda6524ffa3b03f77fa73ede74f9426c4824d3"
      ),
      "task_tracker": (
          "a01d346553a280aea7f01900ddcca7d0be2919bbc399de5e3d8afa436c2ff4a1"
      ),
      "str_replace_editor": (
          "dd56a537979e413afae5c05ca772be4a907e5729edb4fd2cc4d33cf2c0ce2f85"
      ),
  }

  assert [t["function"]["name"] for t in tools] == list(
      expected_required.keys()
  )
  for tool in tools:
    fn = tool["function"]
    name = fn["name"]
    assert fn["parameters"]["required"] == expected_required[name]
    rendered = json.dumps(tool, ensure_ascii=False)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    assert digest == expected_sha256[name], f"Schema mismatch for {name}"

