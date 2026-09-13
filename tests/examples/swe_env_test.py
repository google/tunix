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
from examples.deepswe import openhands_utils
from examples.deepswe import swe_env


def test_extract_images_always_uses_task_image():
  """Every scaffold warms the per-task benchmark image (Path B)."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]

  with mock.patch.dict(
      os.environ,
      {"AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:custom"},
  ):
    iter_r2e = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_r2e.scaffold = "r2egym"
    images_r2e = iter_r2e._extract_images(batch)
    assert images_r2e == ["docker.io/swebench/test-image:v1"]

    iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_oh.scaffold = "openhands"
    images_oh = iter_oh._extract_images(batch)
    assert images_oh == ["docker.io/swebench/test-image:v1"]


def test_extract_images_without_override():
  """Test image extraction when AGENT_SERVER_IMAGE is unset."""
  batch = [{"docker_image": "docker.io/swebench/test-image:v1"}]

  with mock.patch.dict(os.environ, {}, clear=True):
    iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
    iter_oh.scaffold = "openhands"
    assert iter_oh._extract_images(batch) == [
        "docker.io/swebench/test-image:v1"
    ]


def test_warm_batch_per_task_image_sizing():
  """Warm pool is sized per task image at num_generations replicas."""
  mock_fleet = mock.MagicMock()
  mock_fleet.config.max_concurrent = 64

  iter_oh = object.__new__(swe_env.PrewarmDatasetIterator)
  iter_oh.fleet = mock_fleet
  iter_oh.scaffold = "openhands"
  iter_oh.num_generations = 8
  iter_oh.max_warmpool_replicas = None

  with mock.patch.dict(
      os.environ,
      {"AGENT_SERVER_IMAGE": "ghcr.io/openhands/agent-server:custom"},
  ):
    iter_oh._warm_batch([{"docker_image": "dummy"}])
    mock_fleet.warm_images.assert_called_with(
        ["dummy"], replicas_override=8, wait=False
    )

  mock_fleet.reset_mock()
  with mock.patch.dict(os.environ, {}, clear=True):
    iter_oh._warm_batch([{"docker_image": "img1"}])
    mock_fleet.warm_images.assert_called_with(
        ["img1"], replicas_override=8, wait=False
    )


def test_fleet_repo_env_bind_failure_raises():
  """A failed grading-env bind must raise, not silently degrade to reward 0.0."""
  import types

  env = object.__new__(swe_env.SWEEnv)
  env.entry = {"instance_id": "test_inst", "docker_image": "test_image:latest"}
  env.scaffold = "openhands"
  env.env = None
  env.workspace = None
  env.use_agent_sandbox = True
  env.step_timeout = 60
  env.reward_timeout = 180
  env.verbose = False
  mock_fleet = mock.MagicMock()
  mock_fleet.acquire.return_value = mock.MagicMock()
  env.fleet = mock_fleet

  mock_oh = types.ModuleType("agent_sandbox_rl.adapters.openhands")
  mock_oh.make_handle_workspace = mock.MagicMock(return_value=mock.MagicMock())

  mock_r2e = types.ModuleType("agent_sandbox_rl.adapters.r2egym")
  mock_r2e.make_fleet_repo_env = mock.MagicMock(
      side_effect=RuntimeError("failed to bind warm pod")
  )
  mock_r2e.r2egym_command_files = mock.MagicMock(return_value=[])

  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules,
      {
          "agent_sandbox_rl": mock_as_rl,
          "agent_sandbox_rl.adapters.openhands": mock_oh,
          "agent_sandbox_rl.adapters.r2egym": mock_r2e,
      },
  ):
    with pytest.raises(RuntimeError, match="failed to bind warm pod"):
      env._initial_observation()


def _setup_mock_agent_sandbox():
  mock_as = mock.MagicMock()
  mock_as.FleetConfig.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.TemplateSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.ResourceSpec.side_effect = lambda **kw: mock.MagicMock(**kw)
  mock_as.Task.side_effect = lambda **kw: mock.MagicMock(**kw)
  return mock_as


def test_normalize_tasks_keeps_task_image():
  """_normalize_tasks_for_fleet always keeps the per-task benchmark image."""
  tasks = [{"docker_image": "r2e_img:latest", "instance_id": "task-1"}]
  mock_as_rl = _setup_mock_agent_sandbox()

  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as_rl}
  ), mock.patch.dict(os.environ, {"AGENT_SERVER_IMAGE": "agent_server:latest"}):
    norm_r2e = swe_env._normalize_tasks_for_fleet(tasks, scaffold="r2egym")
    assert getattr(norm_r2e[0], "image", None) == "r2e_img:latest"

    norm_oh = swe_env._normalize_tasks_for_fleet(tasks, scaffold="openhands")
    assert getattr(norm_oh[0], "image", None) == "r2e_img:latest"


def test_init_global_fleet_openhands_template_and_sizing():
  """Test TemplateSpec resources/limits and warmpool sizing in _init_global_fleet."""
  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as_rl}
  ), mock.patch.dict(os.environ, {}, clear=True):
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

    # Per-task warm pools are sized at num_generations replicas each.
    assert fleet_cfg.max_warmpool_size == 8

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
  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as_rl}
  ), mock.patch.dict(os.environ, {}, clear=True):
    swe_env._GLOBAL_FLEET = None
    swe_env._init_global_fleet([], scaffold="openhands")
    fleet_cfg = mock_as_rl.SandboxFleet.call_args[0][0]
    cmd = fleet_cfg.template.keepalive_command
    assert "openhands-agent-server" in cmd[2]
    assert "openhands.agent_server" in cmd[2]

  with mock.patch.dict(
      sys.modules, {"agent_sandbox_rl": mock_as_rl}
  ), mock.patch.dict(
      os.environ, {"AGENT_SERVER_COMMAND": '["custom", "server"]'}
  ):
    swe_env._GLOBAL_FLEET = None
    swe_env._init_global_fleet([], scaffold="openhands")
    fleet_cfg = mock_as_rl.SandboxFleet.call_args[0][0]
    assert fleet_cfg.template.keepalive_command == ["custom", "server"]
  swe_env._GLOBAL_FLEET = None


def test_setup_openhands_workspace_symlink_and_safe_directory():
  """Verify setup only wires safe.directory and the workspace symlink."""
  mock_workspace = mock.MagicMock()
  mock_res = mock.MagicMock(exit_code=0)
  mock_workspace.execute_command.return_value = mock_res
  entry = {"repo": "pandas", "commit_hash": "12345"}

  openhands_utils.setup_openhands_workspace(mock_workspace, entry)
  assert mock_workspace.execute_command.called
  cmd = mock_workspace.execute_command.call_args[0][0]
  assert "ln -s /workspace /testbed" in cmd
  assert "ln -s /testbed /workspace" in cmd
  assert "safe.directory /testbed" in cmd
  assert "safe.directory /workspace" in cmd
  # The task image already ships the repo at the target commit.
  assert "git clone" not in cmd


def test_load_tasks_image_rewrite_applies_to_all_scaffolds():
  """Test that load_tasks rewrites task images whenever a rewrite is configured."""
  mock_as_rl = _setup_mock_agent_sandbox()
  with mock.patch.dict(sys.modules, {"agent_sandbox_rl": mock_as_rl}):
    # Case 1: openhands scaffold -> rewrite in load_tasks
    with mock.patch.dict(
        os.environ,
        {
            "IMAGE_REWRITE_PREFIX": "gcr.io/custom",
        },
        clear=True,
    ):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="openhands",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" in fleet.load_tasks.call_args[1]

    # Case 2: r2egym scaffold -> rewrite in load_tasks
    with mock.patch.dict(
        os.environ,
        {
            "IMAGE_REWRITE_PREFIX": "gcr.io/custom",
        },
        clear=True,
    ):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="r2egym",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" in fleet.load_tasks.call_args[1]

    # Case 3: no rewrite configured -> load_tasks called without image_rewrite
    with mock.patch.dict(os.environ, {}, clear=True):
      swe_env._GLOBAL_FLEET = None
      fleet = swe_env._init_global_fleet(
          tasks=[{"docker_image": "r2e_img", "instance_id": "task-1"}],
          scaffold="openhands",
      )
      assert fleet.load_tasks.called
      assert "image_rewrite" not in fleet.load_tasks.call_args[1]

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
  with mock.patch.dict(
      sys.modules,
      {
          "agent_sandbox_rl": mock_as_rl,
          "agent_sandbox_rl.adapters.r2egym": mock_r2e,
      },
  ):
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

  mock_workspace = mock.MagicMock()
  env.workspace = mock_workspace

  mock_inner_env = mock.MagicMock()
  mock_inner_env.step.return_value = ("replacement done", 0, False, {})
  mock_inner_env.compute_reward.return_value = 1.0
  env.env = mock_inner_env

  mock_action_cls = mock.MagicMock()
  with mock.patch.object(swe_env, "Action", mock_action_cls):
    # Test str_replace_editor delegation
    action_editor = mock.MagicMock(
        function_name="str_replace_editor",
        parameters={"command": "str_replace", "path": "/testbed/f.py"},
    )
    res = env._step_impl(action_editor)
    assert res.observation == "replacement done"
    assert res.done is False
    mock_inner_env.step.assert_called_once_with(action_editor)
    # The tool name must be translated to R2E's registered `file_editor`,
    # otherwise run_action rejects it as an Invalid Action.
    assert action_editor.function_name == "file_editor"

    # Reward is delegated straight to the bound benchmark env, with no
    # arguments (this is how TrajectoryCollectEngine invokes it).
    env.final_reward_fn = env.env.compute_reward
    assert env.final_reward_fn() == 1.0
    mock_inner_env.compute_reward.assert_called_once_with()

    # Test tool error message mentions str_replace_editor
    action_invalid = mock.MagicMock(
        function_name="invalid_tool",
        parameters={},
    )
    res_err = env._step_impl(action_invalid)
    assert "str_replace_editor" in res_err.observation


def test_openhands_initial_observation_binds_workspace_and_env():
  """Verify _initial_observation binds both OpenHands workspace and FleetRepoEnv."""
  import types

  env = object.__new__(swe_env.SWEEnv)
  env.entry = {
      "instance_id": "test_inst",
      "docker_image": "test_image:latest",
      "problem_statement": "fix bug",
  }
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
  with mock.patch.dict(
      sys.modules,
      {
          "agent_sandbox_rl": mock_as_rl,
          "agent_sandbox_rl.adapters.openhands": mock_oh,
          "agent_sandbox_rl.adapters.r2egym": mock_r2e,
      },
  ), mock.patch.object(
      swe_env.openhands_utils, "setup_openhands_workspace"
  ) as mock_setup:
    obs = env._initial_observation()

  assert obs == "fix bug"
  assert env.workspace == mock_ws
  assert env.env == mock_repo_env
  # final_reward_fn must point straight at the benchmark grading engine.
  assert env.final_reward_fn == mock_repo_env.compute_reward
  mock_oh.make_handle_workspace.assert_called_once()
  mock_r2e.make_fleet_repo_env.assert_called_once()
  mock_setup.assert_called_once_with(mock_ws, env.entry)


def test_openhands_step_impl_submit():
  """Verify submit and finish actions mark step done."""
  env = object.__new__(swe_env.SWEEnv)
  env.scaffold = "openhands"
  env.max_steps = 30
  env.total_steps = 0
  env.workspace = mock.MagicMock()

  mock_action_cls = mock.MagicMock()
  with mock.patch.object(swe_env, "Action", mock_action_cls):
    for fn_name in ("finish", "submit"):
      action = mock.MagicMock(function_name=fn_name, parameters={})
      res = env._step_impl(action)
      assert res.observation == "Task submitted."
      assert res.reward == 0
      assert res.done is True
      assert res.info == {"max_steps": 30}
      assert env.total_steps == 0


def test_openhands_step_impl_execute_bash():
  """Verify execute_bash handling in openhands step."""
  env = object.__new__(swe_env.SWEEnv)
  env.scaffold = "openhands"
  env.max_steps = 30
  env.total_steps = 0
  env.step_timeout = 60.0

  mock_ws = mock.MagicMock()
  env.workspace = mock_ws

  mock_action_cls = mock.MagicMock()
  with mock.patch.object(swe_env, "Action", mock_action_cls):
    # Missing command
    no_cmd_action = mock.MagicMock(function_name="execute_bash", parameters={})
    res = env._step_impl(no_cmd_action)
    assert "No command specified" in res.observation
    assert res.done is False
    assert env.total_steps == 0

    # Successful command (exit_code 0)
    mock_ws.execute_command.return_value = mock.MagicMock(
        exit_code=0, stdout="hello world\n", stderr=""
    )
    cmd_action = mock.MagicMock(
        function_name="execute_bash",
        parameters={"command": "echo 'hello world'"},
    )
    res = env._step_impl(cmd_action)
    assert res.observation == "hello world\n"
    assert res.done is False
    assert env.total_steps == 1
    assert "echo 'hello world'" in mock_ws.execute_command.call_args[0][0]

    # Failed command (exit_code != 0)
    mock_ws.execute_command.return_value = mock.MagicMock(
        exit_code=1, stdout="err output", stderr="file not found"
    )
    cmd_action_err = mock.MagicMock(
        function_name="execute_bash", parameters={"cmd": "cat missing.txt"}
    )
    res = env._step_impl(cmd_action_err)
    assert "err output\nfile not found" in res.observation
    assert env.total_steps == 2

    # Command exception
    mock_ws.execute_command.side_effect = RuntimeError("timeout")
    res = env._step_impl(cmd_action)
    assert "Command execution failed: timeout" in res.observation
    assert env.total_steps == 3


def test_openhands_utils_step_openhands_direct():
  """Verify openhands_utils.step_openhands can be called directly."""
  mock_ws = mock.MagicMock()
  mock_ws.execute_command.return_value = mock.MagicMock(
      exit_code=0, stdout="pytest output", stderr=""
  )
  env = mock.MagicMock(
      workspace=mock_ws,
      env=None,
      max_steps=20,
      step_timeout=15.0,
      total_steps=5,
  )
  action = mock.MagicMock(
      function_name="execute_bash", parameters={"command": "pytest"}
  )
  res = openhands_utils.step_openhands(env, action)
  assert res.observation == "pytest output"
  assert res.done is False
  assert res.info == {"max_steps": 20}
  assert env.total_steps == 6


def test_openhands_utils_get_image_rewrite_fn():
  """Verify get_image_rewrite_fn in openhands_utils."""

  def custom_fn(img):
    return f"custom/{img}"

  assert openhands_utils.get_image_rewrite_fn(custom_fn) is custom_fn

  with mock.patch.dict(os.environ, {"IMAGE_REWRITE_PREFIX": "gcr.io/mirror"}):
    rewrite = openhands_utils.get_image_rewrite_fn()
    assert rewrite("swebench/test:v1") == "gcr.io/mirror/test:v1"

  with mock.patch.dict(os.environ, {}, clear=True):
    assert openhands_utils.get_image_rewrite_fn() is None


def test_prewarm_dataset_iterator_max_in_flight_batches():
  """Verify that PrewarmDatasetIterator unwarming honors max_in_flight_batches."""
  mock_fleet = mock.MagicMock()
  mock_fleet._image_rewrite_fn = None
  dataset = [
      [{"docker_image": "img1"}],
      [{"docker_image": "img2"}],
      [{"docker_image": "img3"}],
      [{"docker_image": "img4"}],
  ]
  it = swe_env.PrewarmDatasetIterator(dataset, fleet=mock_fleet)
  assert it.max_in_flight_batches == swe_env._MAX_IN_FLIGHT_BATCHES
  assert swe_env._MAX_IN_FLIGHT_BATCHES == 2

  b1 = next(it)
  assert b1[0]["docker_image"] == "img1"
  assert len(it.in_flight_batches) == 1
  mock_fleet.unwarm_image.assert_not_called()

  b2 = next(it)
  assert b2[0]["docker_image"] == "img2"
  assert len(it.in_flight_batches) == 2
  mock_fleet.unwarm_image.assert_not_called()

  b3 = next(it)
  assert b3[0]["docker_image"] == "img3"
  assert len(it.in_flight_batches) == 2
  mock_fleet.unwarm_image.assert_called_with("img1")
