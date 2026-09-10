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


def test_get_system_prompt():
  """Verify get_system_prompt returns correct prompt for all scaffolds and modes."""
  assert template.get_system_prompt("r2egym", use_fn_calling=False) == template.SWE_SYSTEM_PROMPT
  assert template.get_system_prompt("r2egym", use_fn_calling=True) == template.SWE_SYSTEM_PROMPT_FN_CALL
  assert template.get_system_prompt("sweagent", use_fn_calling=False) == template.SWEAGENT_SYSTEM_PROMPT
  assert template.get_system_prompt("sweagent", use_fn_calling=True) == template.SWEAGENT_SYSTEM_PROMPT
  assert template.get_system_prompt("openhands", use_fn_calling=False) == template.OPENHANDS_SYSTEM_PROMPT
  assert template.get_system_prompt("openhands", use_fn_calling=True) == template.OPENHANDS_SYSTEM_PROMPT


def test_get_user_prompt_template():
  """Verify get_user_prompt_template returns correct prompt for all scaffolds."""
  assert template.get_user_prompt_template("r2egym", use_fn_calling=False) == template.SWE_USER_PROMPT
  assert template.get_user_prompt_template("r2egym", use_fn_calling=True) == template.SWE_USER_PROMPT_FN_CALL
  assert template.get_user_prompt_template("sweagent", use_fn_calling=False) == template.SWEAGENT_USER_PROMPT
  assert template.get_user_prompt_template("openhands", use_fn_calling=False) == template.SWE_USER_PROMPT


def test_get_openhands_pod_template_default():
  """Verify default openhands TemplateSpec construction."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as}), \
       mock.patch.dict(os.environ, {}, clear=True):
    pod_template = template.get_openhands_pod_template(node_selector={"node": "worker"})
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
    assert init_c["command"] == ["cp", "/usr/local/bin/openhands-agent-server", "/oh/"]
    assert init_c["volumeMounts"] == [{"name": "oh", "mountPath": "/oh"}]
    assert pod_template.extra_pod_spec["volumes"] == [{"name": "oh", "emptyDir": {}}]


def test_get_openhands_pod_template_with_overrides():
  """Verify openhands TemplateSpec honors env var overrides."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as}), \
       mock.patch.dict(os.environ, {
           "SANDBOX_SESSION_KEY": "secret_key_123",
           "AGENT_SERVER_COMMAND": '["custom", "entrypoint"]',
           "SANDBOX_CPU": "1",
           "SANDBOX_MEM": "2Gi",
           "SANDBOX_CPU_LIMIT": "4",
           "SANDBOX_MEM_LIMIT": "8Gi",
       }):
    pod_template = template.get_openhands_pod_template()
    assert pod_template.keepalive_command == ["custom", "entrypoint"]
    assert pod_template.resources.cpu == "1"
    assert pod_template.resources.memory == "2Gi"
    container = pod_template.extra_pod_spec["containers"][0]
    assert container["resources"]["limits"]["cpu"] == "4"
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["env"] == [{"name": "OH_SESSION_API_KEYS_0", "value": "secret_key_123"}]


def test_get_template_scaffolds():
  """Verify get_template delegates to openhands or returns None for other scaffolds."""
  mock_as = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as}), \
       mock.patch.dict(os.environ, {}, clear=True):
    assert template.get_template("openhands") is not None
    assert template.get_template("r2egym") is None
    assert template.get_template("sweagent") is None


def test_swe_agent_reexports():
  """Verify swe_agent re-exports all prompt constants and helpers for backward compatibility."""
  mock_r2e = mock.MagicMock()
  with mock.patch.dict(sys.modules, {"r2egym": mock_r2e, "r2egym.agenthub.action": mock_r2e}):
    from examples.deepswe import swe_agent
    assert swe_agent.SWE_SYSTEM_PROMPT == template.SWE_SYSTEM_PROMPT
    assert swe_agent.SWE_SYSTEM_PROMPT_FN_CALL == template.SWE_SYSTEM_PROMPT_FN_CALL
    assert swe_agent.SWEAGENT_SYSTEM_PROMPT == template.SWEAGENT_SYSTEM_PROMPT
    assert swe_agent.OPENHANDS_SYSTEM_PROMPT == template.OPENHANDS_SYSTEM_PROMPT
    assert swe_agent.SWE_USER_PROMPT == template.SWE_USER_PROMPT
    assert swe_agent.SWE_USER_PROMPT_FN_CALL == template.SWE_USER_PROMPT_FN_CALL
    assert swe_agent.SWEAGENT_USER_PROMPT == template.SWEAGENT_USER_PROMPT
    assert swe_agent.get_system_prompt == template.get_system_prompt
    assert swe_agent.get_user_prompt_template == template.get_user_prompt_template

    agent_r2e = swe_agent.SWEAgent(scaffold="r2egym")
    assert agent_r2e.system_prompt == template.SWE_SYSTEM_PROMPT
    assert agent_r2e.user_prompt_template == template.SWE_USER_PROMPT

    agent_swe = swe_agent.SWEAgent(scaffold="sweagent")
    assert agent_swe.system_prompt == template.SWEAGENT_SYSTEM_PROMPT
    assert agent_swe.user_prompt_template == template.SWEAGENT_USER_PROMPT

    agent_oh = swe_agent.SWEAgent(scaffold="openhands")
    assert agent_oh.system_prompt == template.OPENHANDS_SYSTEM_PROMPT
    assert agent_oh.user_prompt_template == template.SWE_USER_PROMPT
