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

"""Software Engineering Environment (SWEEnv) for code-related tasks."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, cast

import numpy as np
from examples.deepswe import openhands_utils
from examples.deepswe import sandbox_utils
try:
  from tunix.rl.agentic.environments.base_environment import BaseTaskEnv
  from tunix.rl.agentic.environments.base_environment import EnvStepResult
except ImportError:
  import dataclasses

  @dataclasses.dataclass
  class EnvStepResult:
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any]

  class BaseTaskEnv:

    def __init__(self, max_steps: int = 1):
      self.max_steps = max_steps

    def reset(self) -> Any:
      return self._initial_observation()

    def step(self, action: Any) -> EnvStepResult:
      return self._step_impl(action)

    def close(self) -> None:
      pass

# Re-exports for backward compatibility
PrewarmDatasetIterator = sandbox_utils.PrewarmDatasetIterator
_init_global_fleet = sandbox_utils.init_global_fleet
_get_global_fleet = sandbox_utils.get_global_fleet
_teardown_global_fleet = sandbox_utils.teardown_global_fleet
_patch_r2egym_for_agent_sandbox = sandbox_utils.patch_r2egym_for_agent_sandbox
_normalize_tasks_for_fleet = sandbox_utils.normalize_tasks_for_fleet
_get_image_rewrite_fn = openhands_utils.get_image_rewrite_fn
_MAX_IN_FLIGHT_BATCHES = 2
_GLOBAL_FLEET = None


# pylint: disable=g-import-not-at-top,g-blanket-type-suppression,g-multiple-import
try:
  import r2egym  # pytype: disable=import-error
  from r2egym.agenthub.action import Action  # pytype: disable=import-error
  from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error
except ImportError:
  r2egym = cast(Any, None)
  EnvArgs = cast(Any, None)
  RepoEnv = cast(Any, None)
  Action = cast(Any, None)
# pylint: enable=g-import-not-at-top,g-blanket-type-suppression,g-multiple-import

if r2egym:
  R2EGYM_PATH = os.path.dirname(r2egym.__file__)
else:
  R2EGYM_PATH = ""
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]


def _unpack_entry(entry: dict) -> dict:
  """Utility to clean up and unpack the dataset entry."""
  unpacked_entry = {}
  for k, v in entry.items():
    if isinstance(v, np.ndarray):
      unpacked_entry[k] = v.item()
    elif isinstance(v, list):
      if len(v) != 1:
        raise ValueError(
            f"Can only convert a list of size 1; got size {len(v)}"
        )
      unpacked_entry[k] = v[0]
    else:
      unpacked_entry[k] = v
  return unpacked_entry

def configure_claim_lifecycle(handle: Any, ttl_seconds: int | None = None) -> None:
  """Defensively sets shutdownPolicy: Delete and ttlSecondsAfterFinished on the SandboxClaim.

  This ensures that even if the Python process is abruptly terminated or OOM-killed,
  the Kubernetes agent-sandbox controller will automatically garbage collect the
  SandboxClaim and Sandbox once the backing pod completes or fails.
  """
  cluster = getattr(handle, "_cluster", None)
  claim_name = getattr(handle, "claim_name", None)
  if cluster is None or not claim_name:
    return

  custom_api = getattr(cluster, "custom_api", None)
  namespace = getattr(cluster, "namespace", None) or "default"
  if custom_api is None:
    return

  if ttl_seconds is None:
    try:
      ttl_seconds = int(os.getenv("SANDBOX_TTL_SECONDS_AFTER_FINISHED", "60"))
    except ValueError:
      ttl_seconds = 60

  try:
    custom_api.patch_namespaced_custom_object(
        group="extensions.agents.x-k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="sandboxclaims",
        name=claim_name,
        body={
            "spec": {
                "lifecycle": {
                    "shutdownPolicy": "Delete",
                    "ttlSecondsAfterFinished": ttl_seconds,
                }
            }
        },
    )
    logging.debug(
        "[SWEEnv] Configured SandboxClaim '%s' lifecycle: shutdownPolicy=Delete, ttlSecondsAfterFinished=%d",
        claim_name,
        ttl_seconds,
    )
  except Exception as e:
    logging.debug("[SWEEnv] Note configuring claim lifecycle: %s", e)


def cleanup_k8s_sandbox_handle(handle: Any) -> None:
  """Defensively ensure SandboxClaim and Sandbox CRDs are deleted from the cluster."""
  # 1. Direct call on handle's internal sandbox instance if available
  sb = getattr(handle, "sandbox", None)
  if sb is not None and hasattr(sb, "terminate"):
    try:
      sb.terminate()
    except Exception as e:
      logging.debug("[SWEEnv] sandbox.terminate note: %s", e)

  # 2. Delete via handle._cluster resources
  cluster = getattr(handle, "_cluster", None)
  claim_name = getattr(handle, "claim_name", None)
  sandbox_id = getattr(handle, "sandbox_id", None) or getattr(handle, "sandbox_name", None)

  if cluster is not None:
    resources = getattr(cluster, "resources", None)
    if resources is not None:
      if claim_name and hasattr(resources, "delete_claim"):
        try:
          resources.delete_claim(claim_name)
        except Exception as e:
          logging.debug("[SWEEnv] resources.delete_claim note: %s", e)
      if sandbox_id and hasattr(resources, "delete_sandbox"):
        try:
          resources.delete_sandbox(sandbox_id)
        except Exception as e:
          logging.debug("[SWEEnv] resources.delete_sandbox note: %s", e)

    # 3. Direct CustomObjectsApi deletion if resources call didn't delete
    custom_api = getattr(cluster, "custom_api", None)
    namespace = getattr(cluster, "namespace", None) or "default"
    if custom_api is not None and namespace:
      if claim_name:
        try:
          custom_api.delete_namespaced_custom_object(
              group="extensions.agents.x-k8s.io",
              version="v1beta1",
              namespace=namespace,
              plural="sandboxclaims",
              name=claim_name,
          )
        except Exception as e:
          if getattr(e, "status", None) != 404:
            logging.debug("[SWEEnv] custom_api delete_claim note: %s", e)
      if sandbox_id:
        try:
          custom_api.delete_namespaced_custom_object(
              group="agents.x-k8s.io",
              version="v1beta1",
              namespace=namespace,
              plural="sandboxes",
              name=sandbox_id,
          )
        except Exception as e:
          if getattr(e, "status", None) != 404:
            logging.debug("[SWEEnv] custom_api delete_sandbox note: %s", e)


class SWEEnv(BaseTaskEnv):
  """Software Engineering Environment for code-related tasks."""

  def __init__(
      self,
      entry: dict,
      group_id: int | None = None,
      pair_index: int | None = None,
      step_timeout: int = 30 * 60,
      reward_timeout: int = 30 * 60,
      backend: str = "kubernetes",
      delete_image: bool = False,
      verbose: bool = False,
      scaffold: str = "r2egym",
      max_steps: int = 1,
      use_agent_sandbox: bool = False,
      fleet: Any | None = None,
  ):
    """Initialize the SWE environment.

    Args:
        entry: Dataset containing the tasks. If None, uses default dataset.
        group_id: ID of the group to which the task belongs.
        pair_index: Index of the pair to use. If None, selects a random pair.
        step_timeout: Timeout for each step in seconds.
        reward_timeout: Timeout for reward computation in seconds.
        backend: Backend to use for the environment.
        delete_image: Whether to delete the Docker image after closing.
        verbose: Verbose output toggle.
        scaffold: Scaffold tool set ('r2egym', 'sweagent', or 'openhands').
        max_steps: Maximum interaction steps.
        use_agent_sandbox: If True, strictly forces SandboxFleet and
          AgentSandboxRuntime.
        fleet: Optional SandboxFleet instance to use.
    """
    self.entry = _unpack_entry(entry)
    self.step_timeout = step_timeout
    self.reward_timeout = reward_timeout
    self.total_steps = 0
    self.delete_image = delete_image
    self.backend = backend
    self.env: Any = None
    self.workspace: Any = None
    self.handle: Any = None
    self.verbose = verbose
    self.scaffold = scaffold
    if not use_agent_sandbox:
      env_flag = os.getenv("USE_AGENT_SANDBOX", "").lower() in ("true", "1")
      if env_flag:
        use_agent_sandbox = True
    self.use_agent_sandbox = use_agent_sandbox
    self.fleet = fleet

    assert scaffold in [
        "r2egym",
        "sweagent",
        "openhands",
    ], (
        f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent',"
        " 'openhands']"
    )
    super().__init__(max_steps=max_steps)

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}

    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _init_agent_sandbox_env(self) -> None:
    sandbox_utils.patch_r2egym_for_agent_sandbox()
    from agent_sandbox_rl import Task  # pytype: disable=import-error
    from agent_sandbox_rl.adapters.r2egym import (  # pytype: disable=import-error
        make_fleet_repo_env,
        r2egym_command_files,
    )

    fleet = self.fleet or sandbox_utils.get_global_fleet()
    msg = "[SWEEnv] Acquiring SandboxHandle from SandboxFleet!"
    logging.info(msg)
    task_id = str(
        self.entry.get("instance_id", self.entry.get("docker_image", "default"))
    )
    task_img = self.entry.get("docker_image", "default")
    if isinstance(task_img, (list, np.ndarray)):
      task_img = task_img[0] if len(task_img) > 0 else "default"
    task_img_str = str(task_img)
    rewrite_fn = _get_image_rewrite_fn(
        getattr(fleet, "_image_rewrite_fn", None)
    )
    if rewrite_fn:
      task_img_str = rewrite_fn(task_img_str)
    task = Task(
        id=task_id,
        image=task_img_str,
        metadata={"ds": self.entry},
    )
    max_acquire_retries = 5
    for attempt in range(max_acquire_retries):
      try:
        self.handle = fleet.acquire(task)
        break
      except Exception as e:
        if attempt < max_acquire_retries - 1:
          logging.warning(
              "[SWEEnv] fleet.acquire failed (attempt %d/%d): %s; retrying in"
              " %ds...",
              attempt + 1,
              max_acquire_retries,
              e,
              5 * (attempt + 1),
          )
          time.sleep(5 * (attempt + 1))
        else:
          raise
    configure_claim_lifecycle(self.handle)

    try:
      if self.scaffold == "openhands":
        from agent_sandbox_rl.adapters.openhands import make_handle_workspace  # pytype: disable=import-error

        ws_kwargs = {}
        if os.getenv("SANDBOX_SESSION_KEY"):
          ws_kwargs["api_key"] = os.getenv("SANDBOX_SESSION_KEY")
        if os.getenv("ROUTER_URL"):
          ws_kwargs["router_url"] = os.getenv("ROUTER_URL")
        if os.getenv("ROUTER_AUTH_TOKEN"):
          ws_kwargs["router_auth_token"] = os.getenv("ROUTER_AUTH_TOKEN")
        ws_kwargs["working_dir"] = os.getenv("OPENHANDS_WORKING_DIR", "/testbed")
        self.workspace = make_handle_workspace(self.handle, **ws_kwargs)
      try:
        cmd_files = r2egym_command_files()
      except Exception:  # pylint: disable=broad-exception-caught
        cmd_files = None
      self.env = make_fleet_repo_env(
          self.handle,
          command_files=cmd_files,
          step_timeout=self.step_timeout,
          reward_timeout=self.reward_timeout,
          verbose=self.verbose,
      )
      if self.scaffold == "openhands":
        openhands_utils.setup_openhands_workspace(self.workspace, self.entry)
    except Exception:
      self.close()
      raise

  def _init_local_repo_env(self) -> None:
    # Initialize standard local Docker RepoEnv
    global EnvArgs, RepoEnv, Action
    if EnvArgs is None:
      from r2egym.agenthub.action import Action  # pytype: disable=import-error
      from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error
    env_args = EnvArgs(ds=self.entry)
    self.env = RepoEnv(
        env_args,
        backend=self.backend,
        step_timeout=self.step_timeout,
        reward_timeout=self.reward_timeout,
        verbose=self.verbose,
    )
    if self.scaffold == "r2egym":
      self.env.add_commands(R2EGYM_COMMAND_FILES)
    elif self.scaffold == "sweagent":
      self.env.add_commands(SWEAGENT_COMMAND_FILES)

  def _initial_observation(self) -> Any:
    if not self.env and not self.workspace:
      if self.use_agent_sandbox:
        self._init_agent_sandbox_env()
      else:
        self._init_local_repo_env()
    elif self.env is not None:
      self.env.reset()

    self.final_reward_fn = self.env.compute_reward  # pytype: disable=attribute-error
    self.total_steps = 0

    if self.workspace is not None:
      return str(
          self.entry.get("problem_statement")
          or self.entry.get("instruction")
          or ""
      )

    # Polls docker runtime to get task instruction.
    return self.env.get_task_instruction()  # pytype: disable=attribute-error

  def _step_impl(self, action: Any) -> EnvStepResult:
    global Action
    if Action is None:
      from r2egym.agenthub.action import Action  # pytype: disable=import-error
    if isinstance(action, str):
      action_obj = Action.from_string(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return EnvStepResult(
          observation="",
          reward=0,
          done=False,
          info={"max_steps": self.max_steps},
      )

    if self.scaffold == "openhands" and self.workspace is not None:
      return openhands_utils.step_openhands(self, action_obj)

    # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
    if not self.env:
      raise ValueError("Environment not initialized")
    obs, reward, done, info = self.env.step(action_obj)

    self.total_steps += 1

    return EnvStepResult(
        observation=str(obs), reward=reward, done=done, info=info
    )

  def close(self) -> None:
    """Close the environment and clean up resources."""
    if (
        self.delete_image
        and not self.use_agent_sandbox
        and getattr(self, "env", None) is not None
        and hasattr(self.env, "runtime")
    ):
      docker_image = getattr(self.env.runtime, "docker_image", None)
      if docker_image:
        os.system(f"docker rmi {docker_image}")

    if self.env is not None:
      try:
        self.env.close()
      except Exception as e:
        logging.warning("[SWEEnv] Error closing underlying env: %s", e)
      self.env = None

    if getattr(self, "workspace", None) is not None:
      try:
        self.workspace.cleanup()  # pytype: disable=attribute-error
      except Exception as e:
        logging.warning("[SWEEnv] Workspace cleanup note: %s", e)
      self.workspace = None

    handle = getattr(self, "handle", None)
    fleet = self.fleet or getattr(sandbox_utils, "_GLOBAL_FLEET", None)
    if handle is not None:
      if fleet is not None:
        try:
          logging.info("[SWEEnv] Releasing SandboxHandle back to SandboxFleet.")
          fleet.release(handle)
        except Exception as e:
          logging.warning("[SWEEnv] Error releasing handle back to fleet: %s", e)
      elif hasattr(handle, "release"):
        try:
          handle.release()
        except Exception as e:
          logging.warning("[SWEEnv] Error calling handle.release(): %s", e)

      try:
        cleanup_k8s_sandbox_handle(handle)
      except Exception as e:
        logging.warning("[SWEEnv] Error during defensive k8s sandbox cleanup: %s", e)

      self.handle = None

  def __enter__(self) -> "SWEEnv":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass

  @staticmethod
  def from_dict(extra_info: dict | str) -> "SWEEnv":  # pyrefly: ignore[bad-override]
    """Create an environment instance from JSON configuration.

    Args:
        extra_info: Dictionary containing configuration parameters. The entire
          dict will be used as 'entry', and any keys matching __init__
          parameters will be extracted and passed.

    Returns:
        Initialized SWEEnv instance
    """
    import inspect

    if isinstance(extra_info, str):
      extra_info = json.loads(extra_info)

    sig = inspect.signature(SWEEnv.__init__)
    init_params = {}
    for param_name, param in sig.parameters.items():
      if param_name == "self":
        continue
      if param_name in extra_info:
        init_params[param_name] = extra_info[param_name]
      # else if param has default value, use the default value
    init_params["entry"] = extra_info
    return SWEEnv(**init_params)
