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
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Optional

from tunix.rl.agentic.environments.base_environment import EnvStepResult

MAX_RESPONSE_LEN_CHAR = 16000
MAX_LINES_TO_VIEW = 600
SNIPPET_LINES = 4
CLIPPED_NOTICE = "<response clipped>"
FILE_CLIPPED_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of this"
    " file has been shown to you. You should retry this tool after you have"
    " searched inside the file with grep in order to find the line numbers of"
    " what you are looking for.</NOTE>"
)


def _maybe_clip_response(
    text: str,
    max_len: int = MAX_RESPONSE_LEN_CHAR,
    notice: str = CLIPPED_NOTICE,
) -> str:
  """Clips overly long tool outputs matching OpenHands observation truncation."""
  if not text or max_len <= 0 or len(text) <= max_len:
    return text
  return text[:max_len] + f"\n{notice}"


def run_oh_editor_locally(
    params: dict[str, Any],
    history_file: str = "/var/tmp/.oh_editor_history.json",
) -> str:
  """Executes str_replace_editor matching OpenHands OHEditor semantics."""
  cmd = str(params.get("command") or "")
  path = str(params.get("path") or "")

  if not path.startswith("/"):
    return (
        f"Error: The path {path} is not an absolute path, it should start"
        " with '/'."
    )

  def _load_history() -> dict[str, list[str]]:
    if os.path.exists(history_file):
      try:
        with open(history_file, "r", encoding="utf-8") as f:
          data = json.load(f)
          if isinstance(data, dict):
            return data
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return {}

  def _save_history(hist: dict[str, list[str]]) -> None:
    try:
      os.makedirs(os.path.dirname(history_file), exist_ok=True)
      with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def _push_history(file_path: str, old_content: str) -> None:
    hist = _load_history()
    hist.setdefault(file_path, []).append(old_content)
    _save_history(hist)

  def _pop_history(file_path: str) -> Optional[str]:
    hist = _load_history()
    entries = hist.get(file_path) or []
    if not entries:
      return None
    val = entries.pop()
    _save_history(hist)
    return val

  def _format_cat_n(
      lines: list[str], start_line: int = 1, path_label: str = path
  ) -> str:
    numbered = "\n".join(
        f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)
    )
    return f"Here's the result of running `cat -n` on {path_label}:\n{numbered}\n"

  if cmd == "view":
    if os.path.isdir(path):
      if params.get("view_range") is not None:
        return (
            "Error: The `view_range` parameter is not allowed when `path`"
            " points to a directory."
        )
      try:
        proc = subprocess.run(
            ["find", "-L", path, "-maxdepth", "2", "-not", "-path", "*/.*"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        stdout = proc.stdout.strip()
      except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error listing directory {path}: {e}"
      out = (
          "Here's the files and directories up to 2 levels deep in"
          f" {path}, excluding hidden items:\n{stdout}\n"
      )
      return _maybe_clip_response(out, MAX_RESPONSE_LEN_CHAR, FILE_CLIPPED_NOTICE)

    if not os.path.exists(path):
      return (
          f"Error: Invalid `path` parameter: {path}. The path {path} does"
          " not exist. Please provide a valid path."
      )
    try:
      with open(path, "r", encoding="utf-8", errors="replace") as f:
        file_content = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error reading {path}: {e}"

    file_lines = file_content.expandtabs().split("\n")
    n_lines = len(file_lines)
    init_line = 1
    raw_range = params.get("view_range")
    clipped_by_line_limit = False

    if raw_range is not None and raw_range != "":
      if isinstance(raw_range, str):
        try:
          view_range = json.loads(raw_range)
        except Exception:  # pylint: disable=broad-exception-caught
          return (
              f"Error: Invalid `view_range` parameter: {raw_range}. It"
              " should be a list of two integers."
          )
      else:
        view_range = raw_range
      if (
          not isinstance(view_range, list)
          or len(view_range) != 2
          or not all(isinstance(x, int) for x in view_range)
      ):
        return (
            f"Error: Invalid `view_range` parameter: {raw_range}. It should"
            " be a list of two integers."
        )
      init_line, final_line = view_range[0], view_range[1]
      if init_line < 1 or init_line > n_lines:
        return (
            f"Error: Invalid `view_range` {view_range}. Its first element"
            f" `{init_line}` should be within the range of lines of the"
            f" file: {[1, n_lines]}."
        )
      if final_line > n_lines:
        return (
            f"Error: Invalid `view_range` {view_range}. Its second element"
            f" `{final_line}` should be smaller than the number of lines in"
            f" the file: `{n_lines}`."
        )
      if final_line != -1 and final_line < init_line:
        return (
            f"Error: Invalid `view_range` {view_range}. Its second element"
            f" `{final_line}` should be larger or equal than its first"
            f" `{init_line}`."
        )
      if final_line == -1:
        selected = file_lines[init_line - 1 :]
      else:
        selected = file_lines[init_line - 1 : final_line]
    else:
      selected = file_lines
      if len(selected) > MAX_LINES_TO_VIEW:
        selected = selected[:MAX_LINES_TO_VIEW]
        clipped_by_line_limit = True

    out = _format_cat_n(selected, start_line=init_line, path_label=path)
    if clipped_by_line_limit:
      out += FILE_CLIPPED_NOTICE
    return _maybe_clip_response(out, MAX_RESPONSE_LEN_CHAR, FILE_CLIPPED_NOTICE)

  elif cmd == "create":
    if os.path.exists(path):
      return (
          f"Error: File already exists at: {path}. Cannot overwrite files"
          " using command `create`."
      )
    file_text = params.get("file_text")
    if file_text is None:
      return "Error: Parameter `file_text` is required for command: create."
    try:
      with open(path, "w", encoding="utf-8") as f:
        f.write(str(file_text))
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error creating file {path}: {e}"
    return f"File created successfully at: {path}"

  elif cmd == "str_replace":
    if not os.path.isfile(path):
      return (
          f"Error: Invalid `path` parameter: {path}. The path {path} does"
          " not exist. Please provide a valid file path."
      )
    old_str = params.get("old_str")
    if old_str is None:
      return "Error: Parameter `old_str` is required for command: str_replace."
    old_str = str(old_str).expandtabs()
    new_str = (
        str(params.get("new_str")).expandtabs()
        if params.get("new_str") is not None
        else ""
    )
    if old_str == new_str:
      return (
          "Error: No replacement was performed. `new_str` and `old_str` must"
          " be different."
      )
    try:
    try:
      with open(path, "r", encoding="utf-8", errors="replace") as f:
        file_content = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error reading {path}: {e}"

    occurrences = file_content.count(old_str)
    if occurrences == 0:
      return (
          f"Error: No replacement was performed, old_str `{old_str}` did not"
          f" appear verbatim in {path}."
      )
    if occurrences > 1:
      file_lines = file_content.split("\n")
      lines = [
          idx + 1
          for idx, line in enumerate(file_lines)
          if old_str in line
      ]
      return (
          "Error: No replacement was performed. Multiple occurrences of"
          f" old_str `{old_str}` in lines {lines}. Please ensure it is"
          " unique."
      )

    _push_history(path, file_content)
    new_file_content = file_content.replace(old_str, new_str, 1)
    try:
      with open(path, "w", encoding="utf-8") as f:
        f.write(new_file_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error writing {path}: {e}"

    replacement_line = file_content.split(old_str)[0].count("\n")
    start_line = max(0, replacement_line - SNIPPET_LINES)
    end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
    snippet_lines = new_file_content.split("\n")[start_line : end_line + 1]
    snippet_out = _format_cat_n(
        snippet_lines, start_line=start_line + 1, path_label=f"a snippet of {path}"
    )
    return (
        f"The file {path} has been edited. {snippet_out}Review the changes"
        " and make sure they are as expected. Edit the file again if"
        " necessary."
    )

  elif cmd == "insert":
    if not os.path.isfile(path):
      return (
          f"Error: Invalid `path` parameter: {path}. The path {path} does"
          " not exist."
      )
    raw_insert_line = params.get("insert_line")
    if raw_insert_line is None:
      return "Error: Parameter `insert_line` is required for command: insert."
    try:
      insert_line = int(raw_insert_line)
    except (ValueError, TypeError):
      return (
          f"Error: Invalid `insert_line` parameter: {raw_insert_line}. It"
          " should be an integer."
      )
    new_str = params.get("new_str")
    if new_str is None:
      return "Error: Parameter `new_str` is required for command: insert."
    new_str = str(new_str).expandtabs()
    try:
    try:
      with open(path, "r", encoding="utf-8", errors="replace") as f:
        file_content = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error reading {path}: {e}"

    file_lines = file_content.split("\n")
    n_lines = len(file_lines)
    if insert_line < 0 or insert_line > n_lines:
      return (
          f"Error: Invalid `insert_line` parameter: {insert_line}. It should"
          f" be within the range of lines of the file: {[0, n_lines]}."
      )

    _push_history(path, file_content)
    new_str_lines = new_str.split("\n")
    new_file_lines = (
        file_lines[:insert_line] + new_str_lines + file_lines[insert_line:]
    )
    new_file_content = "\n".join(new_file_lines)
    try:
      with open(path, "w", encoding="utf-8") as f:
        f.write(new_file_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error writing {path}: {e}"

    start_line = max(0, insert_line - SNIPPET_LINES)
    end_line = insert_line + len(new_str_lines) + SNIPPET_LINES
    snippet_lines = new_file_lines[start_line:end_line]
    snippet_out = _format_cat_n(
        snippet_lines,
        start_line=start_line + 1,
        path_label="a snippet of the edited file",
    )
    return (
        f"The file {path} has been edited. {snippet_out}Review the changes"
        " and make sure they are as expected (correct indentation, no"
        " duplicate lines, etc). Edit the file again if necessary."
    )

  elif cmd == "undo_edit":
    old_content = _pop_history(path)
    if old_content is None:
      return f"Error: No edit history found for {path}."
    try:
      with open(path, "w", encoding="utf-8") as f:
        f.write(old_content)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Error restoring {path}: {e}"
    lines = old_content.split("\n")
    if len(lines) > MAX_LINES_TO_VIEW:
      lines = lines[:MAX_LINES_TO_VIEW]
    cat_out = _format_cat_n(lines, start_line=1, path_label=path)
    out = f"Last edit to {path} undone successfully. {cat_out}"
    return _maybe_clip_response(out, MAX_RESPONSE_LEN_CHAR, FILE_CLIPPED_NOTICE)

  return (
      f"Error: Unrecognized command {cmd}. The allowed commands for"
      " str_replace_editor are: `view`, `create`, `str_replace`, `insert`,"
      " `undo_edit`."
  )


def _build_oh_editor_remote_cmd(params: dict[str, Any]) -> str:
  """Builds a self-contained python3 command to run OHEditor inside the sandbox."""
  import inspect  # pylint: disable=g-import-not-at-top

  src_clip = inspect.getsource(_maybe_clip_response)
  src_editor = inspect.getsource(run_oh_editor_locally)
  driver = (
      "import base64, json, os, subprocess, sys\n"
      f"MAX_RESPONSE_LEN_CHAR = {MAX_RESPONSE_LEN_CHAR}\n"
      f"MAX_LINES_TO_VIEW = {MAX_LINES_TO_VIEW}\n"
      f"SNIPPET_LINES = {SNIPPET_LINES}\n"
      f"CLIPPED_NOTICE = {CLIPPED_NOTICE!r}\n"
      f"FILE_CLIPPED_NOTICE = {FILE_CLIPPED_NOTICE!r}\n"
      "from typing import Any, Optional\n\n"
      f"{src_clip}\n\n"
      f"{src_editor}\n\n"
      "payload = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))\n"
      "sys.stdout.write(run_oh_editor_locally(payload))\n"
  )
  b64_driver = base64.b64encode(driver.encode("utf-8")).decode("ascii")
  b64_params = base64.b64encode(
      json.dumps(params, ensure_ascii=False).encode("utf-8")
  ).decode("ascii")
  return (
      "python3 -c \"import base64; exec(base64.b64decode('"
      f"{b64_driver}').decode('utf-8'))\" '{b64_params}'"
  )


def _get_swe_action_cls() -> Any:
  try:
    from r2egym.agenthub.action import Action as SWEAction  # pytype: disable=import-error
    return SWEAction
  except ImportError:
    from examples.deepswe import swe_agent
    return swe_agent.SWEAction


def parse_openhands_action_str(action_str: str) -> Any:
  """Parses an XML action string into an Action object without stripping indentation."""
  swe_action_cls = _get_swe_action_cls()

  fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
  function_name = fn_match.group(1).strip() if fn_match else ""

  pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
  param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

  params: dict[str, str] = {}
  for param_key, param_value in param_matches:
    param_key = param_key.strip()
    if param_value.startswith("\r\n"):
      param_value = param_value[2:]
    elif param_value.startswith("\n"):
      param_value = param_value[1:]
    if param_value.endswith("\r\n"):
      param_value = param_value[:-2]
    elif param_value.endswith("\n"):
      param_value = param_value[:-1]
    params[param_key] = param_value

  return swe_action_cls(function_name, params)


def resolve_base_commit(entry: Optional[dict[str, Any]]) -> str:
  """Resolves base_commit from dataset entry metadata if available."""
  if not isinstance(entry, dict):
    return ""
  base_commit = entry.get("base_commit")
  if base_commit:
    return str(base_commit).strip()
  parsed_commit = entry.get("parsed_commit_content")
  if parsed_commit:
    if isinstance(parsed_commit, str):
      try:
        parsed_commit = json.loads(parsed_commit)
      except Exception:
        parsed_commit = None
    if isinstance(parsed_commit, dict) and parsed_commit.get("old_commit_hash"):
      return str(parsed_commit["old_commit_hash"]).strip()
  return ""


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
    "if [ -f /var/tmp/.r2e_grading_stash/run_tests.sh ]; then "
    "cp -a /var/tmp/.r2e_grading_stash/run_tests.sh /root/run_tests.sh && "
    "ln -sf /root/run_tests.sh /run_tests.sh || true; "
    "fi; "
    "if [ -d /var/tmp/.r2e_grading_stash/r2e_tests ]; then "
    "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests && "
    "cp -a /var/tmp/.r2e_grading_stash/r2e_tests /root/r2e_tests && "
    "ln -s /root/r2e_tests /testbed/r2e_tests && "
    "ln -s /root/r2e_tests /r2e_tests || true; "
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
    entry: Optional dataset entry containing repo and commit metadata.
  """
  if workspace is None:
    return

  if entry is None and hasattr(workspace, "entry"):
    entry = getattr(workspace, "entry", None)

  if hasattr(workspace, "workspace") and not hasattr(
      workspace, "execute_command"
  ):
    workspace = workspace.workspace
    if workspace is None:
      return

  setup_cmds = [
      "rm -f /var/tmp/.oh_cwd /var/tmp/.oh_editor_history.json 2>/dev/null || true",
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
      "git -C /testbed rev-parse HEAD 2>/dev/null || true",
  ]

  full_setup_cmd = " && ".join(setup_cmds)
  try:
    logging.info("[SWEEnv] Configuring OpenHands workspace...")
    res = workspace.execute_command(full_setup_cmd, timeout=180.0)
    if getattr(res, "exit_code", 0) != 0:
      logging.warning(
          "[SWEEnv] Workspace setup exit code %s: %s",
          res.exit_code,
          getattr(res, "stderr", None) or getattr(res, "stdout", None),
      )
    else:
      logging.info("[SWEEnv] Successfully configured OpenHands workspace")
      if isinstance(entry, dict) and not entry.get("base_commit"):
        stdout = getattr(res, "stdout", None)
        if isinstance(stdout, str):
          lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
          if lines and re.fullmatch(r"[0-9a-fA-F]{7,40}", lines[-1]):
            entry["base_commit"] = lines[-1]
  except Exception as e:
    logging.warning("[SWEEnv] Failed to set up repository in workspace: %s", e)


def restore_r2e_tests_for_reward(target: Any) -> None:
  """Restore stashed /root/run_tests.sh and /root/r2e_tests before grading."""
  try:
    _exec_in_sandbox(target, _RESTORE_R2E_TESTS_CMD, timeout=60.0)
  except Exception as e:
    logging.warning("[SWEEnv] Failed to restore R2E tests for reward: %s", e)


def _command_timed_out(result: Any, timeout: float, elapsed: float) -> bool:
  """Returns True if an OpenHands CommandResult represents a timeout."""
  if getattr(result, "timeout_occurred", False) is True:
    return True
  exit_code = getattr(result, "exit_code", None)
  return exit_code == -1 and elapsed >= 0.95 * float(timeout)


def _timeout_notice(timeout: float) -> str:
  return (
      f"[Command timed out after {float(timeout):.0f} seconds and was"
      " terminated. Any output above is partial. The environment is still"
      " usable. Avoid long-running, interactive, or blocking commands (servers,"
      " `watch`, prompts waiting for input); run a narrower subset (e.g. a"
      " single test file or test case), add `timeout <secs>` or limit output,"
      " and continue.]"
  )


def _format_command_result(
    result: Any, timeout: float, elapsed: float
) -> tuple[str, bool]:
  """Formats an OpenHands CommandResult into (observation, timed_out)."""
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
  timed_out = _command_timed_out(result, timeout, elapsed)
  if timed_out:
    obs = f"{obs.rstrip()}\n{_timeout_notice(timeout)}".lstrip("\n")
  obs = _maybe_clip_response(obs)
  return obs, timed_out


def _execute_in_workspace(
    env: Any, wrapped_cmd: str, step_timeout: float, failure_prefix: str
) -> EnvStepResult:
  """Runs `wrapped_cmd` in env.workspace; never raises, never ends the episode."""
  max_steps = getattr(env, "max_steps", None)
  info: dict[str, Any] = {"max_steps": max_steps}
  start = time.monotonic()
  try:
    result = env.workspace.execute_command(
        wrapped_cmd, timeout=float(step_timeout)
    )
    obs, timed_out = _format_command_result(
        result, step_timeout, time.monotonic() - start
    )
    if timed_out:
      info["command_timed_out"] = True
      logging.warning(
          "[SWEEnv] Command hit step_timeout=%.0fs (elapsed %.1fs); returning"
          " timeout observation to agent. cmd=%r",
          float(step_timeout),
          time.monotonic() - start,
          wrapped_cmd[:200],
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    obs = f"{failure_prefix}: {e}"
  if hasattr(env, "total_steps"):
    env.total_steps += 1
  return EnvStepResult(observation=obs, reward=0, done=False, info=info)


def step_openhands(
    env: Any,
    action_obj: Any,
) -> EnvStepResult:
  """Execute an action in an OpenHands-backed environment.

  Handles OpenHands-specific tool dispatch ('finish'/'submit', 'think',
  'task_tracker', 'str_replace_editor'/'file_editor', 'execute_ipython_cell',
  and 'execute_bash') using the environment's workspace and bound grading
  environment.

  Args:
    env: The SWEEnv (or compatible) instance containing `workspace`, `env`,
      `max_steps`, `step_timeout`, and `total_steps`.
    action_obj: The parsed action object with `function_name` and `parameters`.

  Returns:
    EnvStepResult: Result of executing the action.
  """
  max_steps = getattr(env, "max_steps", None)
  params = getattr(action_obj, "parameters", None) or {}

  if action_obj.function_name in ("finish", "submit"):
    msg = params.get("message") or params.get("result") or "Task submitted."
    return EnvStepResult(
        observation=str(msg),
        reward=0,
        done=True,
        info={"max_steps": max_steps},
    )

  if action_obj.function_name == "think":
    if hasattr(env, "total_steps"):
      env.total_steps += 1
    return EnvStepResult(
        observation="Your thought has been logged.",
        reward=0,
        done=False,
        info={"max_steps": max_steps},
    )

  if action_obj.function_name == "task_tracker":
    cmd = params.get("command", "view")
    if hasattr(env, "total_steps"):
      env.total_steps += 1
    if cmd == "view":
      content = getattr(env, "_task_list_content", None)
      if not isinstance(content, str) or not content:
        obs = 'No task list found. Use the "plan" command to create one.'
      else:
        obs = content
      return EnvStepResult(
          observation=obs,
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )
    elif cmd == "plan":
      raw_list = params.get("task_list", "[]")
      if isinstance(raw_list, str):
        try:
          task_list = json.loads(raw_list)
        except Exception as e:
          return EnvStepResult(
              observation=f"Error: Failed to parse task_list JSON: {e}",
              reward=0,
              done=False,
              info={"max_steps": max_steps},
          )
      elif isinstance(raw_list, list):
        task_list = raw_list
      else:
        task_list = []

      status_icons = {
          "todo": "⏳",
          "in_progress": "🔄",
          "done": "✅",
      }
      content = "# Task List\n\n"
      for i, item in enumerate(task_list, 1):
        if isinstance(item, dict):
          status_icon = status_icons.get(
              str(item.get("status", "todo")), "⏳"
          )
          title = item.get("title", "")
          notes = item.get("notes", "")
          content += f"{i}. {status_icon} {title}\n{notes}\n"
      env._task_list = task_list
      env._task_list_content = content
      task_file_path = "/workspace/.openhands/TASKS.md"
      obs = (
          f"Task list has been updated with {len(task_list)} items."
          f" Stored in session directory: {task_file_path}"
      )
      return EnvStepResult(
          observation=obs,
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )
    else:
      return EnvStepResult(
          observation=(
              f"Error: Invalid task_tracker command '{cmd}'. Must be 'view'"
              " or 'plan'."
          ),
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )

  if action_obj.function_name in ("str_replace_editor", "file_editor"):
    if getattr(env, "workspace", None) is not None:
      step_timeout = float(getattr(env, "step_timeout", 60.0))
      remote_cmd = _build_oh_editor_remote_cmd(params)
      return _execute_in_workspace(
          env, remote_cmd, step_timeout, "Command execution failed"
      )
    elif getattr(env, "env", None) is not None:
      # Fallback when running against a local RepoEnv without an OpenHands
      # workspace.
      action_obj.function_name = "file_editor"
      if isinstance(getattr(action_obj, "parameters", None), dict):
        action_obj.parameters.pop("security_risk", None)
      try:
        obs, _, done, _ = env.env.step(action_obj)
        obs_str = _maybe_clip_response(
            str(obs), MAX_RESPONSE_LEN_CHAR, FILE_CLIPPED_NOTICE
        )
      except Exception as e:
        obs_str = f"Command execution failed: {e}"
        done = False
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=obs_str,
          reward=0,
          done=done,
          info={"max_steps": max_steps},
      )
    else:
      raise ValueError("Environment and workspace are not initialized")

  if action_obj.function_name in ("execute_ipython_cell", "python", "ipython"):
    code = (
        params.get("code")
        or params.get("command")
        or params.get("cell")
    )
    if not code:
      return EnvStepResult(
          observation="ERROR: No code specified for execute_ipython_cell.",
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )

    step_timeout = getattr(env, "step_timeout", 60.0)
    b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    wrapped_cmd = (
        "(cd /testbed 2>/dev/null || cd /workspace) && "
        f"python3 -c \"import base64; exec(base64.b64decode('{b64_code}').decode('utf-8'))\""
    )

    if getattr(env, "workspace", None) is not None:
      return _execute_in_workspace(
          env, wrapped_cmd, step_timeout, "Python execution failed"
      )
    elif getattr(env, "env", None) is not None:
      swe_action_cls = _get_swe_action_cls()
      bash_action = swe_action_cls("execute_bash", {"command": wrapped_cmd})
      try:
        obs, reward, done, info = env.env.step(bash_action)
        obs_str = _maybe_clip_response(str(obs))
      except Exception as e:
        obs_str = f"Python execution failed: {e}"
        reward, done, info = 0, False, {"max_steps": max_steps}
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=obs_str, reward=reward, done=done, info=info
      )
    else:
      raise ValueError("Environment and workspace are not initialized")

  if action_obj.function_name in ("execute_bash", "bash"):
    cmd = params.get("command")
    if cmd is None:
      cmd = params.get("cmd")
    is_input = str(params.get("is_input", "false")).lower() == "true"
    if not cmd and not is_input:
      return EnvStepResult(
          observation="ERROR: No command specified for execute_bash.",
          reward=0,
          done=False,
          info={"max_steps": max_steps},
      )

    default_timeout = float(getattr(env, "step_timeout", 60.0))
    max_timeout = float(
        os.getenv("COMMAND_EXEC_TIMEOUT", str(int(default_timeout)))
    )
    raw_timeout = params.get("timeout")
    step_timeout = default_timeout
    if raw_timeout is not None:
      try:
        req_timeout = float(raw_timeout)
        if req_timeout > 0:
          step_timeout = min(req_timeout, max(default_timeout, max_timeout))
      except (ValueError, TypeError):
        step_timeout = default_timeout

    wrapped_cmd = (
        "(__oh_cwd=$(cat /var/tmp/.oh_cwd 2>/dev/null); "
        'if [ -n "$__oh_cwd" ] && [ -d "$__oh_cwd" ]; then cd "$__oh_cwd"; '
        "elif [ -d /testbed ]; then cd /testbed; else cd /workspace; fi; "
        "trap 'pwd > /var/tmp/.oh_cwd 2>/dev/null || true' EXIT; "
        f"{cmd})"
        if cmd
        else "true"
    )

    if getattr(env, "workspace", None) is not None:
      return _execute_in_workspace(
          env, wrapped_cmd, step_timeout, "Command execution failed"
      )
    elif getattr(env, "env", None) is not None:
      if isinstance(getattr(action_obj, "parameters", None), dict):
        action_obj.parameters.pop("security_risk", None)
        action_obj.parameters.pop("timeout", None)
        action_obj.parameters.pop("is_input", None)
      try:
        obs, reward, done, info = env.env.step(action_obj)
        obs_str = _maybe_clip_response(str(obs))
      except Exception as e:
        obs_str = f"Command execution failed: {e}"
        reward, done, info = 0, False, {"max_steps": max_steps}
      if hasattr(env, "total_steps"):
        env.total_steps += 1
      return EnvStepResult(
          observation=obs_str, reward=reward, done=done, info=info
      )
    else:
      raise ValueError("Environment and workspace are not initialized")

  return EnvStepResult(
      observation=(
          f"ERROR: Tool '{action_obj.function_name}' is not recognized. "
          "Only 'execute_bash', 'think', 'finish', 'task_tracker', "
          "'str_replace_editor', 'execute_ipython_cell', and 'submit' are"
          " available."
      ),
      reward=0,
      done=False,
      info={"max_steps": max_steps},
  )


_step_openhands = step_openhands

