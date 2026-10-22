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

"""Utility functions for OpenHands workspace and environment setup."""

import logging
import os
from typing import Any, Optional

from tunix.rl.agentic.environments.base_environment import EnvStepResult


def get_image_rewrite_fn(image_rewrite: Any | None = None) -> Any | None:
  """Retrieve or construct the image rewrite function from prefix if configured."""
  if image_rewrite is not None:
    return image_rewrite
  if os.getenv("IMAGE_REWRITE_PREFIX"):
    prefix = os.environ["IMAGE_REWRITE_PREFIX"].rstrip("/")
    return lambda img: f"{prefix}/{img.split('/')[-1]}"
  return None


_get_image_rewrite_fn = get_image_rewrite_fn


def setup_openhands_workspace(
    workspace: Any,
    entry: Optional[Any] = None,
) -> None:
  """Configure repository environment in the OpenHands workspace.

  Task containers already ship the repository pre-cloned at the target commit
  under /testbed, so this only ensures the idempotent git ``safe.directory``
  configuration and the /workspace <-> /testbed symlink.

  Args:
    workspace: The OpenHands workspace instance (or an environment object with
      workspace and entry attributes).
    entry: Optional dataset entry containing repo and commit metadata. Unused;
      retained for call-site compatibility.
  """
  del entry  # The task image already provides the repo at the target commit.
  if workspace is None:
    return

  if hasattr(workspace, "workspace") and not hasattr(
      workspace, "execute_command"
  ):
    workspace = workspace.workspace
    if workspace is None:
      return

  setup_cmds = [
      "git config --global user.email 'openhands@agent.sandbox'",
      "git config --global user.name 'OpenHands Agent'",
      "git config --global --add safe.directory /testbed 2>/dev/null || true",
      "git config --global --add safe.directory /workspace 2>/dev/null || true",
      (
          "([ -d /testbed ] && [ ! -e /workspace ] && ln -s /testbed /workspace"
          " 2>/dev/null || true)"
      ),
      (
          "([ -d /workspace ] && [ ! -e /testbed ] && ln -s /workspace /testbed"
          " 2>/dev/null || true)"
      ),
  ]

  full_setup_cmd = " && ".join(setup_cmds)
  try:
    logging.info("[SWEEnv] Configuring OpenHands workspace...")
    res = workspace.execute_command(full_setup_cmd, timeout=180.0)
    if res.exit_code != 0:
      logging.warning(
          "[SWEEnv] Workspace setup exit code %s: %s",
          res.exit_code,
          res.stderr or res.stdout,
      )
    else:
      logging.info("[SWEEnv] Successfully configured OpenHands workspace")
  except Exception as e:
    logging.warning("[SWEEnv] Failed to set up repository in workspace: %s", e)


def step_openhands(
    env: Any,
    action_obj: Any,
) -> EnvStepResult:
  """Execute an action in an OpenHands-backed environment.

  Handles OpenHands-specific tool dispatch ('finish'/'submit',
  'str_replace_editor', and 'execute_bash') using the environment's workspace
  and bound grading environment.

  Args:
    env: The SWEEnv (or compatible) instance containing `workspace`, `env`,
      `max_steps`, `step_timeout`, and `total_steps`.
    action_obj: The parsed action object with `function_name` and `parameters`.

  Returns:
    EnvStepResult: Result of executing the action.
  """
  max_steps = getattr(env, "max_steps", None)

  if action_obj.function_name in ("finish", "submit"):
    return EnvStepResult(
        observation="Task submitted.",
        reward=0,
        done=True,
        info={"max_steps": max_steps},
    )

  if action_obj.function_name == "str_replace_editor" and env.env is not None:
    # R2E registers this editor as `file_editor` and `RepoEnv.run_action`
    # asserts the tool name is in its registered command list, so translate
    # before delegating. The parameter schemas are identical.
    action_obj.function_name = "file_editor"
    obs, reward, done, info = env.env.step(action_obj)
    if hasattr(env, "total_steps"):
      env.total_steps += 1
    return EnvStepResult(
        observation=str(obs),
        reward=0,
        done=done,
        info={"max_steps": max_steps},
    )

  if action_obj.function_name != "execute_bash":
    return EnvStepResult(
        observation=(
            f"ERROR: Tool '{action_obj.function_name}' is not recognized. "
            "Only 'execute_bash', 'str_replace_editor', and 'submit' are"
            " available."
        ),
        reward=0,
        done=False,
        info={"max_steps": max_steps},
    )

  cmd = (
      action_obj.parameters.get("command") or action_obj.parameters.get("cmd")
      if getattr(action_obj, "parameters", None)
      else None
  )
  if not cmd:
    return EnvStepResult(
        observation="ERROR: No command specified for execute_bash.",
        reward=0,
        done=False,
        info={"max_steps": max_steps},
    )

  try:
    step_timeout = getattr(env, "step_timeout", 30.0)
    wrapped_cmd = f"(cd /testbed 2>/dev/null || cd /workspace) && {cmd}"
    result = env.workspace.execute_command(
        wrapped_cmd, timeout=float(step_timeout)
    )
    obs = (
        result.stdout
        if result.exit_code == 0
        else f"{result.stdout}\n{result.stderr}"
    )
  except Exception as e:
    obs = f"Command execution failed: {e}"
  if hasattr(env, "total_steps"):
    env.total_steps += 1
  return EnvStepResult(
      observation=obs,
      reward=0,
      done=False,
      info={"max_steps": max_steps},
  )


_step_openhands = step_openhands
