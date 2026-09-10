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

"""Unit tests for SWEEnv and PrewarmDatasetIterator."""

import os
import sys
from unittest import mock
import pytest

from examples.deepswe import swe_env


def test_extract_images_gated_by_scaffold():
  """Test that AGENT_SERVER_IMAGE is gated on scaffold == 'openhands'."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]

  with mock.patch.dict(os.environ, {"AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:custom"}):
    # For r2egym scaffold, AGENT_SERVER_IMAGE must be ignored
    iter_r2e = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_r2e.scaffold = "r2egym"
    images_r2e = iter_r2e._extract_images(batch)
    assert images_r2e == ["docker.io/swebench/test-image:v1"]

    # For openhands scaffold, AGENT_SERVER_IMAGE should be used
    iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_oh.scaffold = "openhands"
    images_oh = iter_oh._extract_images(batch)
    assert images_oh == ["ghcr.io/openhands/agent-server:custom"]


def test_extract_images_without_override():
  """Test image extraction when AGENT_SERVER_IMAGE is unset."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]

  with mock.patch.dict(os.environ, {}, clear=True):
    iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_oh.scaffold = "openhands"
    assert iter_oh._extract_images(batch) == ["docker.io/swebench/test-image:v1"]


def test_warm_batch_shared_image_sizing():
  """Test warm pool sizing for shared image vs per-task images."""
  mock_fleet = mock.MagicMock()
  mock_fleet.config.max_concurrent = 64

  iter_oh_shared = object.__new__(swe_env.PrewarmDatasetIterator)
  iter_oh_shared.fleet = mock_fleet
  iter_oh_shared.scaffold = "openhands"
  iter_oh_shared.num_generations = 8
  iter_oh_shared.max_warmpool_replicas = None

  with mock.patch.dict(os.environ, {"AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:custom"}):
    iter_oh_shared._warm_batch([{"docker_image": "dummy"}])
    mock_fleet.warm_images.assert_called_with(
        ["ghcr.io/openhands/agent-server:custom"], replicas_override=64, wait=False
    )

  mock_fleet.reset_mock()
  with mock.patch.dict(os.environ, {}, clear=True):
    iter_oh_shared._warm_batch([{"docker_image": "img1"}])
    mock_fleet.warm_images.assert_called_with(
        ["img1"], replicas_override=8, wait=False
    )


def test_compute_reward_fallback_uses_tmp_eval_sh():
  """Verify fallback uses /tmp/eval.sh for non-root user compatibility."""
  env = object.__new__(swe_env.SWEEnv)
  env._cached_reward = None
  env.env = None
  env.reward_timeout = 30
  env.entry = {"eval_script": "pytest"}

  mock_workspace = mock.MagicMock()
  mock_res = mock.MagicMock(exit_code=0)
  mock_workspace.execute_command.return_value = mock_res
  env.workspace = mock_workspace

  reward = env.compute_reward()
  assert reward == 1.0

  called_cmd = mock_workspace.execute_command.call_args[0][0]
  assert "/tmp/eval.sh" in called_cmd
  assert "cat << '__EOF_EVAL__' > /eval.sh" not in called_cmd


def test_normalize_tasks_scaffold_gating():
  """Test that _normalize_tasks_for_fleet gates AGENT_SERVER_IMAGE on scaffold."""
  tasks = [{"docker_image": "r2e_img:latest", "instance_id": "task-1"}]

  with mock.patch.dict(os.environ, {"AGENT_SERVER_IMAGE": "agent_server:latest"}):
    # r2egym scaffold keeps task image
    norm_r2e = swe_env._normalize_tasks_for_fleet(tasks, scaffold="r2egym")
    assert getattr(norm_r2e[0], "image", None) == "r2e_img:latest"

    # openhands scaffold uses AGENT_SERVER_IMAGE
    norm_oh = swe_env._normalize_tasks_for_fleet(tasks, scaffold="openhands")
    assert getattr(norm_oh[0], "image", None) == "agent_server:latest"


def _setup_mock_agent_sandbox():
  mock_as = mock.MagicMock()
  mock_as.FleetConfig.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.TemplateSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.ResourceSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  return mock_as


def test_init_global_fleet_openhands_template_and_sizing():
  """Test TemplateSpec resources/limits and warmpool sizing in _init_global_fleet."""
  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as_rl}), \
       mock.patch.dict(os.environ, {"AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:1.44.1"}):
    swe_env._GLOBAL_FLEET = None
    fleet = swe_env._init_global_fleet(
        tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
        max_concurrency=32,
        num_generations=8,
        batch_size=4,
        scaffold="openhands",
    )
    assert mock_as_rl.SandboxFleet.called
    fleet_cfg = mock_as_rl.SandboxFleet.call_args[0][0]

    # Shared image mode pool size must equal effective_max_concurrent = max(32, 4*8*2) = 64
    assert fleet_cfg.max_warmpool_size == 64

    # Template resources and limits
    template = fleet_cfg.template
    assert template.resources.cpu == "500m"
    assert template.resources.memory == "1Gi"
    container = template.extra_pod_spec["containers"][0]
    assert container["resources"]["limits"]["cpu"] == "2"
    assert container["resources"]["limits"]["memory"] == "4Gi"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"

    swe_env._GLOBAL_FLEET = None


def test_extract_images_with_image_rewrite():
  """Test that image_rewrite is applied in PrewarmDatasetIterator._extract_images."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]
  iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
  iter_oh.scaffold = "openhands"
  iter_oh.image_rewrite = lambda img: f"gcr.io/custom/{img.split('/')[-1]}"

  with mock.patch.dict(os.environ, {}, clear=True):
    assert iter_oh._extract_images(batch) == ["gcr.io/custom/test-image:v1"]


def test_extract_images_with_image_rewrite_prefix_env():
  """Test that IMAGE_REWRITE_PREFIX env var is picked up in _extract_images."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]
  iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
  iter_oh.scaffold = "openhands"

  with mock.patch.dict(os.environ, {"IMAGE_REWRITE_PREFIX": "gcr.io/prefix"}):
    assert iter_oh._extract_images(batch) == ["gcr.io/prefix/test-image:v1"]


def test_keepalive_command_default_and_override():
  """Test adaptive keepalive_command and custom AGENT_SERVER_COMMAND override."""
  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as_rl}), \
       mock.patch.dict(os.environ, {}, clear=True):
    swe_env._GLOBAL_FLEET = None
    swe_env._init_global_fleet([], scaffold="openhands")
    fleet_cfg = mock_as_rl.SandboxFleet.call_args[0][0]
    cmd = fleet_cfg.template.keepalive_command
    assert "openhands-agent-server" in cmd[2]
    assert "openhands.agent_server" in cmd[2]

  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as_rl}), \
       mock.patch.dict(os.environ, {"AGENT_SERVER_COMMAND": '["custom", "server"]'}):
    swe_env._GLOBAL_FLEET = None
    swe_env._init_global_fleet([], scaffold="openhands")
    fleet_cfg = mock_as_rl.SandboxFleet.call_args[0][0]
    assert fleet_cfg.template.keepalive_command == ["custom", "server"]
  swe_env._GLOBAL_FLEET = None


def test_setup_openhands_workspace_conditional_clone_and_symlink():
  """Verify setup includes conditional git clone and symlink creation."""
  env = object.__new__(swe_env.SWEEnv)
  mock_workspace = mock.MagicMock()
  mock_res = mock.MagicMock(exit_code=0)
  mock_workspace.execute_command.return_value = mock_res
  env.workspace = mock_workspace
  env.entry = {"repo": "pandas", "commit_hash": "12345"}

  env._setup_openhands_workspace()
  assert mock_workspace.execute_command.called
  cmd = mock_workspace.execute_command.call_args[0][0]
  assert "git clone https://github.com/pandas-dev/pandas.git" in cmd
  assert "ln -s /workspace /testbed" in cmd
  assert "safe.directory /testbed" in cmd
  assert "safe.directory /workspace" in cmd


def test_load_tasks_image_rewrite_scaffold_and_agent_server_gating():
  """Test that load_tasks only rewrites images when scaffold != 'openhands' or AGENT_SERVER_IMAGE is unset."""
  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as_rl}):
    # Case 1: openhands scaffold with AGENT_SERVER_IMAGE set -> no rewrite in load_tasks
    with mock.patch.dict(os.environ, {
        "AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:1.44.1",
        "IMAGE_REWRITE_PREFIX": "gcr.io/custom",
    }):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="openhands",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" not in fleet.load_tasks.call_args[1]

    # Case 2: openhands scaffold with AGENT_SERVER_IMAGE unset (derived images) -> rewrite in load_tasks
    with mock.patch.dict(os.environ, {
        "IMAGE_REWRITE_PREFIX": "gcr.io/custom",
    }, clear=True):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="openhands",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" in fleet.load_tasks.call_args[1]

    # Case 3: r2egym scaffold -> rewrite in load_tasks
    with mock.patch.dict(os.environ, {
        "AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:1.44.1",
        "IMAGE_REWRITE_PREFIX": "gcr.io/custom",
    }):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="r2egym",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" in fleet.load_tasks.call_args[1]

  swe_env._GLOBAL_FLEET = None


def test_acquire_retry_on_transient_error():
  """Verify fleet.acquire retries on transient errors."""
  import types
  env = object.__new__(swe_env.SWEEnv)
  env.entry = {"instance_id": "test_inst", "docker_image": "test_image:latest"}
  env.scaffold = "r2egym"
  env.env = None
  env.workspace = None
  env.use_agent_sandbox = True
  env.step_timeout = 60
  env.reward_timeout = 180
  env.verbose = False
  mock_fleet = mock.MagicMock()
  mock_handle = mock.MagicMock()
  mock_fleet.acquire.side_effect = [
      RuntimeError("transient protocol error"),
      mock_handle,
  ]
  env.fleet = mock_fleet
  swe_env._fleet = mock_fleet

  mock_r2e = types.ModuleType("agent_sandbox_rl.adapters.r2egym")
  mock_make = mock.MagicMock()
  mock_repo_env = mock.MagicMock()
  mock_repo_env.reset.return_value = ("obs", {})
  mock_make.return_value = mock_repo_env
  mock_r2e.make_fleet_repo_env = mock_make
  mock_r2e.r2egym_command_files = mock.MagicMock(return_value=[])

  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {
      "agent_sandbox_rl": mock_as_rl,
      "agent_sandbox_rl.adapters.r2egym": mock_r2e,
  }):
    with mock.patch("time.sleep") as mock_sleep:
      env._initial_observation()

  assert mock_fleet.acquire.call_count == 2
  assert env.handle == mock_handle
  mock_sleep.assert_called_once_with(5)


def test_openhands_step_impl_str_replace_editor_and_reward():
  """Verify openhands delegates str_replace_editor and compute_reward to bound env."""
  env = object.__new__(swe_env.SWEEnv)
  env.scaffold = "openhands"
  env.max_steps = 30
  env.total_steps = 0
  env._cached_reward = None

  mock_workspace = mock.MagicMock()
  env.workspace = mock_workspace

  mock_inner_env = mock.MagicMock()
  mock_inner_env.step.return_value = ("replacement done", 0, False, {})
  mock_inner_env.compute_reward.return_value = 1.0
  env.env = mock_inner_env

  # Test str_replace_editor delegation
  action_editor = swe_env._ActionFallback(
      function_name="str_replace_editor",
      parameters={"command": "str_replace", "path": "/testbed/f.py"},
  )
  res = env._step_impl(action_editor)
  assert res.observation == "replacement done"
  assert res.done is False
  mock_inner_env.step.assert_called_once_with(action_editor)

  # Test compute_reward delegation
  reward = env.compute_reward()
  assert reward == 1.0
  mock_inner_env.compute_reward.assert_called_once()

  # Test tool error message mentions str_replace_editor
  action_invalid = swe_env._ActionFallback(
      function_name="invalid_tool",
      parameters={},
  )
  res_err = env._step_impl(action_invalid)
  assert "str_replace_editor" in res_err.observation


def test_openhands_initial_observation_binds_workspace_and_env():
  """Verify _initial_observation binds both OpenHands workspace and FleetRepoEnv."""
  import types
  env = object.__new__(swe_env.SWEEnv)
  env.entry = {"instance_id": "test_inst", "docker_image": "test_image:latest", "problem_statement": "fix bug"}
  env.scaffold = "openhands"
  env.env = None
  env.workspace = None
  env.use_agent_sandbox = True
  env.step_timeout = 60
  env.reward_timeout = 180
  env.verbose = False
  mock_fleet = mock.MagicMock()
  mock_handle = mock.MagicMock()
  mock_fleet.acquire.return_value = mock_handle
  env.fleet = mock_fleet
  swe_env._fleet = mock_fleet

  mock_oh = types.ModuleType("agent_sandbox_rl.adapters.openhands")
  mock_ws = mock.MagicMock()
  mock_oh.make_handle_workspace = mock.MagicMock(return_value=mock_ws)

  mock_r2e = types.ModuleType("agent_sandbox_rl.adapters.r2egym")
  mock_repo_env = mock.MagicMock()
  mock_r2e.make_fleet_repo_env = mock.MagicMock(return_value=mock_repo_env)
  mock_r2e.r2egym_command_files = mock.MagicMock(return_value=[])

  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {
      "agent_sandbox_rl": mock_as_rl,
      "agent_sandbox_rl.adapters.openhands": mock_oh,
      "agent_sandbox_rl.adapters.r2egym": mock_r2e,
  }), mock.patch.object(env, "_setup_openhands_workspace") as mock_setup:
    obs = env._initial_observation()

  assert obs == "fix bug"
  assert env.workspace == mock_ws
  assert env.env == mock_repo_env
  mock_oh.make_handle_workspace.assert_called_once()
  mock_r2e.make_fleet_repo_env.assert_called_once()
  mock_setup.assert_called_once()

