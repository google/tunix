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

import base64
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


_HIDE_R2E_TESTS_CMD = (
    "mkdir -p /var/tmp/.r2e_grading_stash && ("
    "for p in /root/run_tests.sh /testbed/run_tests.sh /run_tests.sh; do "
    '[ -f "$p" ] && [ ! -L "$p" ] && cp -a "$p"'
    " /var/tmp/.r2e_grading_stash/run_tests.sh && break; done; "
    "for d in /root/r2e_tests /r2e_tests /testbed/r2e_tests; do "
    '[ -d "$d" ] && [ ! -L "$d" ] && rm -rf'
    ' /var/tmp/.r2e_grading_stash/r2e_tests && cp -a "$d"'
    " /var/tmp/.r2e_grading_stash/r2e_tests && break; done; "
    "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests "
    "/run_tests.sh /root/run_tests.sh /testbed/run_tests.sh"
    ") 2>/dev/null || true"
)

_RESTORE_R2E_TESTS_CMD = (
    "if [ -d /var/tmp/.r2e_grading_stash ]; then "
    "[ -f /var/tmp/.r2e_grading_stash/run_tests.sh ] && "
    "cp -a /var/tmp/.r2e_grading_stash/run_tests.sh /root/run_tests.sh || true; "
    "if [ -d /var/tmp/.r2e_grading_stash/r2e_tests ]; then "
    "rm -rf /root/r2e_tests /testbed/r2e_tests && "
    "cp -a /var/tmp/.r2e_grading_stash/r2e_tests /root/r2e_tests && "
    "ln -s /root/r2e_tests /testbed/r2e_tests || true; "
    "fi; fi"
)


def _exec_in_sandbox(target: Any, cmd: str, timeout: float = 60.0) -> Any:
  """Execute a shell command on either an OpenHands workspace or RepoEnv runtime."""
  if target is None:
    return None
  ws = getattr(target, "workspace", None)
  if ws is not None and not hasattr(target, "execute_command"):
    target = ws
  if hasattr(target, "execute_command"):
    return target.execute_command(cmd, timeout=timeout)
  runtime = getattr(target, "runtime", None)
  if runtime is None and getattr(target, "env", None) is not None:
    runtime = getattr(target.env, "runtime", None)
  if runtime is not None and hasattr(runtime, "run"):
    return runtime.run(cmd, timeout=int(timeout))
  return None


def hide_r2e_tests_for_rollout(target: Any) -> None:
  """Stash and remove /r2e_tests and /run_tests.sh during agent rollout."""
  try:
    _exec_in_sandbox(target, _HIDE_R2E_TESTS_CMD, timeout=60.0)
  except Exception as e:
    logging.warning("[SWEEnv] Failed to hide R2E tests for rollout: %s", e)


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
      _HIDE_R2E_TESTS_CMD,
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


def restore_r2e_tests_for_reward(target: Any) -> None:
  """Restore stashed /root/run_tests.sh and /root/r2e_tests before grading."""
  try:
    _exec_in_sandbox(target, _RESTORE_R2E_TESTS_CMD, timeout=60.0)
  except Exception as e:
    logging.warning("[SWEEnv] Failed to restore R2E tests for reward: %s", e)


def step_openhands(
    env: Any,
    action_obj: Any,
) -> EnvStepResult:
  """Execute an action in an OpenHands-backed environment.

  Handles OpenHands-specific tool dispatch ('finish'/'submit',
  'str_replace_editor'/'file_editor', 'execute_ipython_cell', and
  'execute_bash') using the environment's workspace and bound grading
  environment.

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

  if action_obj.function_name in ("str_replace_editor", "file_editor") and env.env is not None:
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

  if action_obj.function_name in ("execute_ipython_cell", "python", "ipython"):
    code = (
        action_obj.parameters.get("code")
        or action_obj.parameters.get("command")
        or action_obj.parameters.get("cell")
        if getattr(action_obj, "parameters", None)
        else None
    )
    if not code:
      return EnvStepResult(
          observation="ERROR: No code specified for execute_ipython_cell.",
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )

    step_timeout = getattr(env, "step_timeout", 30.0)
    b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    wrapped_cmd = (
        "(cd /testbed 2>/dev/null || cd /workspace) && "
        f"python3 -c \"import base64; exec(base64.b64decode('{b64_code}').decode('utf-8'))\""
    )

    if getattr(env, "workspace", None) is not None:
      try:
        result = env.workspace.execute_command(
            wrapped_cmd, timeout=float(step_timeout)
        )
        if getattr(result, "stdout", None) is not None:
          obs = (
              str(result.stdout)
              if getattr(result, "exit_code", 0) == 0
              else f"{result.stdout}\n{getattr(result, 'stderr', '')}"
          )
        elif getattr(result, "output", None) is not None:
          obs = str(result.output)
        else:
          obs = str(result)
      except Exception as e:
        obs = f"Python execution failed: {e}"
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=obs,
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )
    elif getattr(env, "env", None) is not None:
      from r2egym.agenthub.action.action import Action as SWEAction  # pytype: disable=import-error
      bash_action = SWEAction("execute_bash", {"command": wrapped_cmd})
      obs, reward, done, info = env.env.step(bash_action)
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=str(obs), reward=reward, done=done, info=info
      )
    else:
      raise ValueError("Environment and workspace are not initialized")

  if action_obj.function_name in ("execute_bash", "bash"):
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

    step_timeout = getattr(env, "step_timeout", 30.0)
    wrapped_cmd = f"(cd /testbed 2>/dev/null || cd /workspace) && {cmd}"

    if getattr(env, "workspace", None) is not None:
      try:
        result = env.workspace.execute_command(
            wrapped_cmd, timeout=float(step_timeout)
        )
        if getattr(result, "stdout", None) is not None:
          obs = (
              str(result.stdout)
              if getattr(result, "exit_code", 0) == 0
              else f"{result.stdout}\n{getattr(result, 'stderr', '')}"
          )
        elif getattr(result, "output", None) is not None:
          obs = str(result.output)
        else:
          obs = str(result)
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
    elif getattr(env, "env", None) is not None:
      obs, reward, done, info = env.env.step(action_obj)
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=str(obs), reward=reward, done=done, info=info
      )
    else:
      raise ValueError("Environment and workspace are not initialized")

  return EnvStepResult(
      observation=(
          f"ERROR: Tool '{action_obj.function_name}' is not recognized. "
          "Only 'execute_bash', 'execute_ipython_cell', 'str_replace_editor', "
          "and 'submit' are available."
      ),
      reward=0,
      done=False,
      info={"max_steps": max_steps},
  )


_step_openhands = step_openhands
