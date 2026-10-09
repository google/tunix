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
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from typing import Any, Optional

from examples.deepswe.opencode_fuzzy import fuzzy_find_match
from tunix.rl.agentic.environments.base_environment import EnvStepResult

MAX_RESPONSE_LEN_CHAR = 16000
CLIPPED_NOTICE = "<response clipped>"
FILE_CLIPPED_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of this"
    " file has been shown to you. You should retry this tool after you have"
    " searched inside the file with `grep -n` in order to find the line"
    " numbers of what you are looking for.</NOTE>"
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
  """Executes str_replace_editor with openhands-aci 0.3.2 OHEditor semantics.

  The reference harness (OpenHands 0d766ad06) runs OHEditor in its server
  process and shows the model `result.output`, or "ERROR:\n" + the error
  message. This port reproduces that text byte for byte for UTF-8 text files.
  It is shipped into the sandbox and run there by python3, so it must stay
  self-contained and Python 3.7 compatible. Edit history lives in
  `history_file` because every call is a fresh process.
  """
  import pathlib  # pylint: disable=g-import-not-at-top
  import re as _re  # pylint: disable=g-import-not-at-top

  max_chars = 16000  # openhands_aci.editor.config.MAX_RESPONSE_LEN_CHAR
  context_window = 4  # SNIPPET_CONTEXT_WINDOW
  max_file_bytes = 10 * 1024 * 1024  # OHEditor.MAX_FILE_SIZE_MB
  max_history = 10  # FileHistoryManager(max_history_per_file=10)
  text_notice = (
      "<response clipped><NOTE>Due to the max output limit, only part of this"
      " file has been shown to you. You should retry this tool after you have"
      " searched inside the file with `grep -n` in order to find the line"
      " numbers of what you are looking for.</NOTE>"
  )
  dir_notice = (
      "<response clipped><NOTE>Due to the max output limit, only part of this"
      " directory has been shown to you. You should use `ls -la` instead to"
      " view large directories incrementally.</NOTE>"
  )

  class _ToolError(Exception):
    pass

  def _invalid(parameter, value, hint=None):
    if hint:
      return _ToolError(f"Invalid `{parameter}` parameter: {value}. {hint}")
    return _ToolError(f"Invalid `{parameter}` parameter: {value}.")

  def _missing(command, parameter):
    return _ToolError(
        f"Parameter `{parameter}` is required for command: {command}."
    )

  def _truncate(content, notice):
    if len(content) <= max_chars:
      return content
    return content[:max_chars] + notice

  def _make_output(snippet, description, start_line=1):
    snippet = _truncate(snippet, text_notice)
    numbered = "\n".join(
        f"{i + start_line:6}\t{line}"
        for i, line in enumerate(snippet.split("\n"))
    )
    return (
        f"Here's the result of running `cat -n` on {description}:\n"
        + numbered
        + "\n"
    )

  def _read(file_path, start_line=None, end_line=None):
    _validate_file(file_path)
    try:
      with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        if start_line is not None and end_line is not None:
          lines = []
          for i, line in enumerate(f, 1):
            if i > end_line:
              break
            if i >= start_line:
              lines.append(line)
          return "".join(lines)
        return "".join(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise _ToolError(f"Ran into {e} while trying to read {file_path}")

  def _write(file_path, text):
    _validate_file(file_path)
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise _ToolError(f"Ran into {e} while trying to write to {file_path}")

  def _count_lines(file_path):
    with open(file_path, encoding="utf-8", errors="replace") as f:
      return sum(1 for _ in f)

  def _is_binary(file_path):
    # binaryornot.check.is_binary without chardet: the `.pyc` rule, then a NUL
    # byte in the first 1024 bytes.
    if str(file_path).endswith(".pyc"):
      return True
    try:
      if not file_path.is_file():
        return False
      with open(file_path, "rb") as f:
        return b"\x00" in f.read(1024)
    except (OSError, ValueError):
      return False

  def _validate_file(file_path):
    if not file_path.exists() or not file_path.is_file():
      return
    file_size = os.path.getsize(file_path)
    if file_size > max_file_bytes:
      raise _ToolError(
          f"File validation failed for {file_path}: File is too large"
          f" ({file_size / 1024 / 1024:.1f}MB). Maximum allowed size is"
          f" {int(max_file_bytes / 1024 / 1024)}MB."
      )
    if _is_binary(file_path):
      raise _ToolError(
          f"File validation failed for {file_path}: File appears to be binary"
          " and this file type cannot be read or edited by this tool."
      )

  def _load_history():
    if os.path.exists(history_file):
      try:
        with open(history_file, "r", encoding="utf-8") as f:
          data = json.load(f)
          if isinstance(data, dict):
            return data
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return {}

  def _save_history(hist):
    try:
      os.makedirs(os.path.dirname(history_file), exist_ok=True)
      with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def _add_history(file_path, content):
    hist = _load_history()
    entries = hist.setdefault(str(file_path), [])
    entries.append(content)
    del entries[:-max_history]
    _save_history(hist)

  def _pop_history(file_path):
    hist = _load_history()
    entries = hist.get(str(file_path)) or []
    if not entries:
      return None
    content = entries.pop()
    _save_history(hist)
    return content

  def _run_shell(cmd_str, notice):
    proc = subprocess.run(
        cmd_str,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return _truncate(proc.stdout, notice), _truncate(proc.stderr, notice)

  def _view(file_path, view_range):
    if file_path.is_dir():
      if view_range:
        raise _invalid(
            "view_range",
            view_range,
            "The `view_range` parameter is not allowed when `path` points to"
            " a directory.",
        )
      hidden_stdout, _ = _run_shell(
          f"find -L {file_path} -mindepth 1 -maxdepth 1 -name '.*'",
          "<response clipped><NOTE>Due to the max output limit, only part of"
          " the full response has been shown to you.</NOTE>",
      )
      hidden_count = (
          len(hidden_stdout.strip().split("\n")) if hidden_stdout.strip() else 0
      )
      stdout, stderr = _run_shell(
          f"find -L {file_path} -maxdepth 2 -not \\( -path"
          f" '{file_path}/\\.*' -o -path '{file_path}/*/\\.*' \\) | sort",
          dir_notice,
      )
      if stderr:
        raise _ToolError(stderr)
      paths = stdout.strip().split("\n") if stdout.strip() else []
      formatted = [
          f"{entry}/" if pathlib.Path(entry).is_dir() else entry
          for entry in paths
      ]
      msg = [
          "Here's the files and directories up to 2 levels deep in"
          f" {file_path}, excluding hidden items:\n"
          + "\n".join(formatted)
      ]
      if hidden_count > 0:
        msg.append(
            f"\n{hidden_count} hidden files/directories in this directory are"
            f" excluded. You can use 'ls -la {file_path}' to see them."
        )
      return "\n".join(msg)

    _validate_file(file_path)
    num_lines = _count_lines(file_path)
    if not view_range:
      return _make_output(_read(file_path), str(file_path), 1)
    if (
        not isinstance(view_range, (list, tuple))
        or len(view_range) != 2
        or not all(isinstance(i, int) for i in view_range)
    ):
      raise _invalid(
          "view_range", view_range, "It should be a list of two integers."
      )
    start_line, end_line = view_range
    if start_line < 1 or start_line > num_lines:
      raise _invalid(
          "view_range",
          view_range,
          f"Its first element `{start_line}` should be within the range of"
          f" lines of the file: {[1, num_lines]}.",
      )
    warning_message = None
    if end_line == -1:
      end_line = num_lines
    elif end_line > num_lines:
      warning_message = (
          f"We only show up to {num_lines} since there're only {num_lines}"
          " lines in this file."
      )
      end_line = num_lines
    if end_line < start_line:
      raise _invalid(
          "view_range",
          view_range,
          f"Its second element `{end_line}` should be greater than or equal to"
          f" the first element `{start_line}`.",
      )
    content = _read(file_path, start_line=start_line, end_line=end_line)
    output = _make_output(
        "\n".join(content.splitlines()), str(file_path), start_line
    )
    if warning_message:
      output = f"NOTE: {warning_message}\n{output}"
    return output

  def _str_replace(file_path, old_str, new_str):
    # OpenCodeEditor.str_replace (OpenHands 0d766ad06), which the reference
    # runtime uses in place of OHEditor: exact match, then strip(), then the
    # OpenCode fuzzy replacers. Errors quote the original old_str.
    _validate_file(file_path)
    new_str = new_str or ""
    file_content = _read(file_path)

    def _find(needle):
      return [
          (file_content.count("\n", 0, m.start()) + 1, m.group(), m.start())
          for m in _re.finditer(_re.escape(needle), file_content)
      ]

    occurrences = _find(old_str)
    actual_new_str = new_str
    if not occurrences:
      old_str_stripped = old_str.strip()
      occurrences = _find(old_str_stripped)
      if occurrences:
        actual_new_str = new_str.strip()
    if not occurrences:
      fuzzy_match = fuzzy_find_match(file_content, old_str)
      if fuzzy_match:
        idx = file_content.find(fuzzy_match)
        if idx != -1:
          line_num = file_content.count("\n", 0, idx) + 1
          occurrences = [(line_num, fuzzy_match, idx)]
    if not occurrences:
      raise _ToolError(
          f"No replacement was performed, old_str `{old_str}` did not appear"
          f" verbatim in {file_path}."
      )
    if len(occurrences) > 1:
      line_numbers = sorted(set(line for line, _, _ in occurrences))
      raise _ToolError(
          "No replacement was performed. Multiple occurrences of old_str"
          f" `{old_str}` in lines {line_numbers}. Please ensure it is unique."
      )
    replacement_line, matched_text, idx = occurrences[0]
    new_file_content = (
        file_content[:idx]
        + actual_new_str
        + file_content[idx + len(matched_text) :]
    )
    _write(file_path, new_file_content)
    _add_history(file_path, file_content)
    start_line = max(0, replacement_line - context_window)
    end_line = (
        replacement_line + context_window + actual_new_str.count("\n")
    )
    snippet = _read(file_path, start_line=start_line + 1, end_line=end_line)
    return (
        f"The file {file_path} has been edited. "
        + _make_output(snippet, f"a snippet of {file_path}", start_line + 1)
        + "Review the changes and make sure they are as expected. Edit the"
        " file again if necessary."
    )

  def _insert(file_path, insert_line, new_str):
    _validate_file(file_path)
    num_lines = _count_lines(file_path)
    if insert_line < 0 or insert_line > num_lines:
      raise _invalid(
          "insert_line",
          insert_line,
          "It should be within the range of allowed values:"
          f" {[0, num_lines]}",
      )
    new_str_lines = new_str.split("\n")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
      old_lines = list(f)
    history_text = "".join(old_lines)
    new_text = (
        "".join(old_lines[:insert_line])
        + "".join(line + "\n" for line in new_str_lines)
        + "".join(old_lines[insert_line:])
    )
    _write(file_path, new_text)
    start_line = max(0, insert_line - context_window)
    end_line = min(
        num_lines + len(new_str_lines),
        insert_line + context_window + len(new_str_lines),
    )
    snippet = _read(file_path, start_line=start_line + 1, end_line=end_line)
    _add_history(file_path, history_text)
    return (
        f"The file {file_path} has been edited. "
        + _make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - context_window + 1),
        )
        + "Review the changes and make sure they are as expected (correct"
        " indentation, no duplicate lines, etc). Edit the file again if"
        " necessary."
    )

  def _undo_edit(file_path):
    _read(file_path)
    old_text = _pop_history(file_path)
    if old_text is None:
      raise _ToolError(f"No edit history found for {file_path}.")
    _write(file_path, old_text)
    return (
        f"Last edit to {file_path} undone successfully."
        f" {_make_output(old_text, str(file_path))}"
    )

  def _dispatch():
    command = params.get("command")
    path = pathlib.Path(str(params.get("path") or ""))
    view_range = params.get("view_range")
    if isinstance(view_range, str):
      try:
        view_range = json.loads(view_range)
      except ValueError:
        pass
    insert_line = params.get("insert_line")
    if insert_line is not None and isinstance(insert_line, str):
      try:
        insert_line = int(insert_line)
      except ValueError:
        return (
            f"ERROR:\nInvalid insert_line value: '{insert_line}'. Expected an"
            " integer."
        )
    old_str = params.get("old_str")
    new_str = params.get("new_str")
    file_text = params.get("file_text")

    if command == "view" and _is_binary(path):
      # The reference runtime refuses binary files before the editor runs
      # (action_execution_server.read) and renders an ErrorObservation.
      return "ERROR_BINARY_FILE\n[Error occurred in processing last action]"
    if not path.is_absolute():
      raise _invalid(
          "path",
          path,
          "The path should be an absolute path, starting with `/`.",
      )
    if command == "create" and path.exists():
      raise _invalid(
          "path",
          path,
          f"File already exists at: {path}. Cannot overwrite files using"
          " command `create`.",
      )
    if command != "create" and not path.exists():
      raise _invalid(
          "path",
          path,
          f"The path {path} does not exist. Please provide a valid path.",
      )
    if command != "view" and path.is_dir():
      raise _invalid(
          "path",
          path,
          f"The path {path} is a directory and only the `view` command can be"
          " used on directories.",
      )

    if command == "view":
      return _view(path, view_range)
    if command == "create":
      if file_text is None:
        raise _missing(command, "file_text")
      _write(path, file_text)
      _add_history(path, file_text)
      return f"File created successfully at: {path}"
    if command == "str_replace":
      if old_str is None:
        raise _missing(command, "old_str")
      if new_str == old_str:
        raise _invalid(
            "new_str",
            new_str,
            "No replacement was performed. `new_str` and `old_str` must be"
            " different.",
        )
      return _str_replace(path, old_str, new_str)
    if command == "insert":
      if insert_line is None:
        raise _missing(command, "insert_line")
      if new_str is None:
        raise _missing(command, "new_str")
      return _insert(path, insert_line, new_str)
    if command == "undo_edit":
      return _undo_edit(path)
    raise _ToolError(
        f"Unrecognized command {command}. The allowed commands for the"
        " oh_editor tool are: view, create, str_replace, insert, undo_edit"
    )

  try:
    return _dispatch()
  except _ToolError as e:
    return f"ERROR:\n{e}"

# The agent-server is a PyInstaller onefile binary launched with /oh/glibc236
# in LD_LIBRARY_PATH. Its bootloader prepends its bundle directory (/tmp/_MEI*)
# to LD_LIBRARY_PATH, and commands it spawns inherit that (often with a
# trailing colon from "${LD_LIBRARY_PATH:-}"), so binaries in older task images
# load the bundle's newer libstdc++/libc and fail with "GLIBC_2.36 not found".
# Strip any /tmp/_MEI* and /oh/glibc236 entries and trailing colons before
# running anything, and unset LD_LIBRARY_PATH when empty.
_STRIP_PYINSTALLER_LD_PATH = (
    'while :; do case "${LD_LIBRARY_PATH-}" in'
    " /tmp/_MEI*:*|/oh/glibc236:*|:*) LD_LIBRARY_PATH=${LD_LIBRARY_PATH#*:};;"
    " /tmp/_MEI*|/oh/glibc236|'') unset LD_LIBRARY_PATH; break;;"
    " *:) LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:};;"
    " *) export LD_LIBRARY_PATH; break;;"
    " esac; done; "
)


def _build_oh_editor_remote_cmd(params: dict[str, Any]) -> str:
  """Builds a self-contained python3 command that runs the sandbox editor."""
  import inspect  # pylint: disable=g-import-not-at-top

  # pylint: disable-next=g-import-not-at-top
  from examples.deepswe import opencode_fuzzy

  src_fuzzy = inspect.getsource(opencode_fuzzy).replace(
      "from __future__ import annotations\n", ""
  )
  src_editor = inspect.getsource(run_oh_editor_locally)
  # The driver runs under the sandbox's own python3 (3.7 for numpy and pandas,
  # 3.8 for pyramid), where the PEP 585 annotations in the copied source
  # (`dict[str, Any]`) raise TypeError at def time. Postponed evaluation keeps
  # them unevaluated.
  driver = (
      "from __future__ import annotations\n"
      "import base64, json, os, subprocess, sys\n"
      "from typing import Any, Optional\n\n"
      f"{src_fuzzy}\n\n"
      f"{src_editor}\n\n"
      "payload = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))\n"
      "sys.stdout.write(run_oh_editor_locally(payload))\n"
  )
  b64_driver = base64.b64encode(driver.encode("utf-8")).decode("ascii")
  b64_params = base64.b64encode(
      json.dumps(params, ensure_ascii=False).encode("utf-8")
  ).decode("ascii")
  return (
      _STRIP_PYINSTALLER_LD_PATH
      + "python3 -c \"import base64; exec(base64.b64decode('"
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
  """Parses SWEAction.to_xml_string() output back into an Action, losslessly.

  to_xml_string() writes each value as `<parameter=k>{v}</parameter>` with no
  padding, so the value is taken verbatim. The model-text parser
  (parse_openhands_xml_action) already removed the newline after the opening
  tag and before the closing one; removing another here dropped real edge
  newlines, e.g. the final newline of every `create` file_text.
  """
  swe_action_cls = _get_swe_action_cls()

  fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
  function_name = fn_match.group(1).strip() if fn_match else ""

  pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
  param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

  params: dict[str, str] = {}
  for param_key, param_value in param_matches:
    params[param_key.strip()] = param_value

  return swe_action_cls(function_name, params)


def resolve_base_commit(entry: Optional[dict[str, Any]]) -> str:
  """Resolves base_commit from dataset entry metadata if available."""
  if entry is None:
    return ""
  if "base_commit" in entry and entry["base_commit"]:
    return str(entry["base_commit"]).strip()
  if "parsed_commit_content" in entry and entry["parsed_commit_content"]:
    parsed_commit = entry["parsed_commit_content"]
    if isinstance(parsed_commit, str):
      try:
        parsed_commit = json.loads(parsed_commit)
      except json.JSONDecodeError:
        parsed_commit = None
    if (
        parsed_commit is not None
        and "old_commit_hash" in parsed_commit
        and parsed_commit["old_commit_hash"]
    ):
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
    "rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests "
    "/run_tests.sh /root/run_tests.sh /testbed/run_tests.sh "
    "/testbed/expected_test_output.json /root/expected_test_output.json "
    "/var/tmp/.r2e_grading_stash 2>/dev/null || true"
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

REMOVE_BINARY_FILES_CMD = """
git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4- | while IFS= read -r file; do
    file=$(echo "$file" | sed -e 's/^"//' -e 's/"$//')
    if [ -f "$file" ] && (file -b "$file" 2>/dev/null | grep -v -i "text" | grep -q "executable" || git check-attr binary "$file" 2>/dev/null | grep -q "binary: set"); then
        git rm -f "$file" 2>/dev/null || rm -f "$file"
        echo "Removed: $file"
    fi
done
""".strip()

_CLEANUP_CONTAINER0_PROCESSES_CMD = (
    "_self=$$; _ppid=$PPID; "
    "for _p in /proc/[0-9]*; do "
    '  _pid="${_p#/proc/}"; '
    '  [ "$_pid" = "1" ] || [ "$_pid" = "$_self" ] || [ "$_pid" = "$_ppid" ] && continue; '
    '  _cmd=$(tr "\\0" " " < "$_p/cmdline" 2>/dev/null || true); '
    "  case \"$_cmd\" in "
    "    *openhands-agent-server*|*agent-server*|*tini*) continue ;; "
    "  esac; "
    '  kill -TERM "$_pid" 2>/dev/null || true; '
    "done; "
    "sleep 0.2; "
    "for _p in /proc/[0-9]*; do "
    '  _pid="${_p#/proc/}"; '
    '  [ "$_pid" = "1" ] || [ "$_pid" = "$_self" ] || [ "$_pid" = "$_ppid" ] && continue; '
    '  _cmd=$(tr "\\0" " " < "$_p/cmdline" 2>/dev/null || true); '
    "  case \"$_cmd\" in "
    "    *openhands-agent-server*|*agent-server*|*tini*) continue ;; "
    "  esac; "
    '  kill -KILL "$_pid" 2>/dev/null || true; '
    "done; "
    "rm -rf /dev/shm/* /dev/shm/.[!.]* /dev/shm/..?* 2>/dev/null || true"
)


def remove_binary_files_from_git() -> str:
  """Returns bash snippet to remove staged/untracked binary or executable files."""
  return REMOVE_BINARY_FILES_CMD


def remove_binary_diffs(patch_text: str) -> str:
  """Removes binary file diffs from a git patch (matches nv-OpenHands binary_patch_utils.py)."""
  if not patch_text:
    return ""
  lines = patch_text.splitlines()
  cleaned_lines: list[str] = []
  block: list[str] = []
  is_binary_block = False

  for line in lines:
    if line.startswith("diff --git "):
      if block and not is_binary_block:
        cleaned_lines.extend(block)
      block = [line]
      is_binary_block = False
    elif "Binary files" in line or line.startswith("GIT binary patch"):
      is_binary_block = True
      block.append(line)
    else:
      block.append(line)

  if block and not is_binary_block:
    cleaned_lines.extend(block)

  return "\n".join(cleaned_lines)


def _exec_in_sandbox(target: Any, cmd: str, timeout: float = 60.0) -> Any:
  """Execute a shell command on either an OpenHands workspace or RepoEnv runtime."""
  if target is None:
    return None
  ws = getattr(target, "workspace", None)
  if ws is not None and not hasattr(target, "execute_command"):
    target = ws
  if hasattr(target, "execute_command"):
    return target.execute_command(
        _STRIP_PYINSTALLER_LD_PATH + cmd, timeout=timeout
    )
  runtime = getattr(target, "runtime", None)
  if runtime is None and getattr(target, "env", None) is not None:
    runtime = getattr(target.env, "runtime", None)
  if runtime is not None and hasattr(runtime, "run"):
    return runtime.run(f"/bin/sh -c {shlex.quote(cmd)}", timeout=int(timeout))
  return None


def _unpack_exec_output(res: Any) -> tuple[str, int]:
  """Normalizes output and exit code from workspace.execute_command or runtime.run."""
  if res is None:
    return "", -1
  if isinstance(res, tuple) and len(res) == 2:
    out, code = res
    code_str = str(code).strip()
    return str(out or ""), (0 if code_str == "0" else -1)
  exit_code = getattr(res, "exit_code", 0)
  try:
    exit_code_int = int(exit_code)
  except (TypeError, ValueError):
    exit_code_int = 0 if str(exit_code).strip() == "0" else -1
  stdout = getattr(res, "stdout", None)
  if stdout is None:
    stdout = ""
  return str(stdout), exit_code_int


_GIT_EXCLUDE_R2E_AND_OH_CMD = (
    "(if [ -d /testbed/.git/info ]; then "
    "printf '\\n/bash_events\\n/bash_events/\\n/conversations\\n/conversations/\\n"
    "/install.sh\\n/run_tests.sh\\n/r2e_tests\\n/r2e_tests/\\n' "
    ">> /testbed/.git/info/exclude; fi) 2>/dev/null || true"
)


def hide_r2e_tests_for_rollout(
    target: Any, entry: Optional[dict[str, Any]] = None
) -> None:
  """Removes /r2e_tests and /run_tests.sh from the rollout container."""
  try:
    cmd = (
        f"{_HIDE_R2E_TESTS_CMD} && "
        "(git -C /testbed reset --hard 2>/dev/null || true) && "
        f"{_GIT_EXCLUDE_R2E_AND_OH_CMD} && "
        "(git -C /testbed rev-parse HEAD 2>/dev/null || true)"
    )
    res = _exec_in_sandbox(target, cmd, timeout=60.0)
    if entry is not None and ("base_commit" not in entry or not entry["base_commit"]):
      stdout, exit_code = _unpack_exec_output(res)
      if exit_code == 0 and stdout:
        lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
        if lines and re.fullmatch(r"[0-9a-fA-F]{7,40}", lines[-1]):
          entry["base_commit"] = lines[-1]
  except Exception as e:
    logging.warning("[SWEEnv] Failed to hide R2E tests for rollout: %s", e)


def setup_openhands_workspace(
    workspace: Any,
    entry: Optional[dict[str, Any]] = None,
) -> None:
  """Configure repository environment in the OpenHands workspace.

  Task containers already ship the repository pre-cloned at the target commit
  under /testbed, so this only ensures the idempotent git ``safe.directory``
  configuration, resets any dirty build artifacts, removes hidden test files
  from the rollout container, and sets up the /workspace <-> /testbed symlink.

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
      'git config --global core.pager ""',
      "git config --global diff.binary false",
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
      "git -C /testbed reset --hard 2>/dev/null || true",
      (
          "(for r in $(git -C /testbed remote 2>/dev/null); do"
          ' git -C /testbed remote remove "$r" 2>/dev/null || true; done)'
      ),
      _HIDE_R2E_TESTS_CMD,
      _GIT_EXCLUDE_R2E_AND_OH_CMD,
      "git -C /testbed rev-parse HEAD 2>/dev/null || true",
  ]

  full_setup_cmd = _STRIP_PYINSTALLER_LD_PATH + " && ".join(setup_cmds)
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
      if entry is not None and ("base_commit" not in entry or not entry["base_commit"]):
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


def extract_agent_patch(
    target: Any,
    base_commit: str = "",
    workspace_path: str = "/testbed",
    timeout: float = 300.0,
) -> str:
  """Extracts the git diff patch from the rollout container matching nv-OpenHands.

  Follows nv-OpenHands evaluation/benchmarks/swe_bench/run_infer.py:691-797:
  1. cd to workspace_path (/testbed) and disable git pager.
  2. Find and remove any nested .git directories in subdirectories.
  3. Stage all files with ``git add -A``.
  4. Unstage OpenHands server logs / R2E harness scripts and remove binary and
     executable files from git staging.
  5. Run ``git diff --no-color --cached <base_commit>``.
  6. Strip any remaining binary diff blocks via ``remove_binary_diffs``.
  """
  if target is None:
    return ""
  if not base_commit:
    entry = getattr(target, "entry", None)
    base_commit = resolve_base_commit(entry)
  commit_ref = base_commit.strip() if base_commit else "HEAD"
  quoted_commit_ref = shlex.quote(commit_ref)
  extract_cmd = (
      f"cd {shlex.quote(workspace_path)} && "
      "{ "
      'git config --global core.pager ""; '
      'find . -type d -name .git -not -path "./.git" -exec rm -rf {} + 2>/dev/null; '
      "git add -A; "
      f"git reset {quoted_commit_ref} -- bash_events conversations install.sh run_tests.sh r2e_tests 2>/dev/null || true; "
      f"{REMOVE_BINARY_FILES_CMD}; "
      "} >/dev/null 2>&1 && "
      f"git diff --no-color --cached {quoted_commit_ref} 2>/dev/null"
  )
  try:
    res = _exec_in_sandbox(target, extract_cmd, timeout=timeout)
    raw_patch, exit_code = _unpack_exec_output(res)
    if exit_code != 0:
      logging.warning(
          "[SWEEnv] Failed to extract git patch from rollout container (exit=%s)",
          exit_code,
      )
      return ""
    cleaned = remove_binary_diffs(raw_patch)
    if not cleaned or not cleaned.strip():
      return ""
    if not cleaned.endswith("\n"):
      cleaned += "\n"
    return cleaned
  except Exception as e:
    logging.warning("[SWEEnv] Exception extracting agent patch: %s", e)
    return ""


def cleanup_rollout_container_processes(
    target: Any, timeout: float = 15.0
) -> None:
  """Kills background processes spawned in container 0 and clears /dev/shm."""
  if target is None:
    return
  try:
    _exec_in_sandbox(target, _CLEANUP_CONTAINER0_PROCESSES_CMD, timeout=timeout)
  except Exception as e:
    logging.debug("[SWEEnv] Rollout container process cleanup note: %s", e)


def evaluate_patch_in_fresh_container(
    eval_env: Any,
    patch: str,
    orig_compute_reward: Optional[Any] = None,
    *args: Any,
    **kwargs: Any,
) -> float:
  """Applies `patch` in the fresh `eval` container and runs R2E-Gym grading.

  Follows nv-R2E-Gym src/r2egym/agenthub/run/run_local_evaluation.py:36-71:
  1. An empty patch immediately scores 0.0.
  2. Lists untracked files in /testbed (`git ls-files --others --exclude-standard`)
     and excludes them from `git apply`.
  3. Applies patch with `git apply --whitespace=fix`; failed apply scores 0.0.
  4. Runs `runtime.setup_env()` and `_calculate_reward()` in the fresh container.
  """
  if not patch or not patch.strip():
    logging.info("[SWEEnv] Empty git patch extracted; returning reward 0.0.")
    return 0.0

  if eval_env is None:
    return 0.0

  from examples.deepswe import sandbox_utils  # pylint: disable=g-import-not-at-top

  sandbox_utils.set_runtime_container(
      eval_env, sandbox_utils.EVAL_CONTAINER_NAME
  )
  runtime = getattr(eval_env, "runtime", None)
  if runtime is None or not hasattr(runtime, "run"):
    if orig_compute_reward is not None:
      return float(orig_compute_reward(*args, **kwargs))
    return 0.0

  repo_path = getattr(runtime, "repo_path", "/testbed") or "/testbed"
  try:
    git_ls_res = runtime.run(
        "git ls-files --others --exclude-standard",
        timeout=30,
    )
    git_ls_output, git_ls_exit = _unpack_exec_output(git_ls_res)
    if git_ls_exit != 0:
      logging.warning(
          "[SWEEnv] Failed to list untracked files in eval container: %s",
          git_ls_output,
      )
      return 0.0

    untracked_files = [
        f.strip() for f in git_ls_output.splitlines() if f.strip()
    ]
    exclude_str = " ".join(
        shlex.quote(f"--exclude={f}") for f in untracked_files
    )

    patch_path = "/tmp/model.patch"
    b64_patch = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    if len(b64_patch) <= 65536:
      inner_apply = (
          f"printf '%s' {shlex.quote(b64_patch)} | base64 -d > {patch_path} && "
          f"cd {shlex.quote(repo_path)} && "
          f"git apply --whitespace=fix {exclude_str} {patch_path}; "
          f"_rc=$?; rm -f {patch_path}; exit $_rc"
      )
      apply_cmd = f"/bin/sh -c {shlex.quote(inner_apply)}"
    else:
      b64_tmp_path = f"{patch_path}.b64"
      chunk_size = 49152
      for idx in range(0, len(b64_patch), chunk_size):
        chunk = b64_patch[idx : idx + chunk_size]
        redir = ">" if idx == 0 else ">>"
        inner_chunk = f"printf '%s' {shlex.quote(chunk)} {redir} {b64_tmp_path}"
        chunk_res = runtime.run(
            f"/bin/sh -c {shlex.quote(inner_chunk)}", timeout=30
        )
        chunk_out, chunk_exit = _unpack_exec_output(chunk_res)
        if chunk_exit != 0:
          logging.warning(
              "[SWEEnv] Failed to write patch chunk in eval container: %s",
              chunk_out,
          )
          runtime.run(f"rm -f {shlex.quote(b64_tmp_path)}", timeout=15)
          return 0.0
      inner_apply = (
          f"base64 -d {b64_tmp_path} > {patch_path} && "
          f"rm -f {b64_tmp_path} && "
          f"cd {shlex.quote(repo_path)} && "
          f"git apply --whitespace=fix {exclude_str} {patch_path}; "
          f"_rc=$?; rm -f {b64_tmp_path} {patch_path}; exit $_rc"
      )
      apply_cmd = f"/bin/sh -c {shlex.quote(inner_apply)}"

    apply_res = runtime.run(apply_cmd, timeout=60)
    apply_output, apply_exit = _unpack_exec_output(apply_res)
    if apply_exit != 0:
      logging.warning(
          "[SWEEnv] Failed to apply patch in eval container: %s",
          apply_output,
      )
      return 0.0

    runtime._defer_setup_env = False
    if getattr(runtime, "_handle", None) is not None:
      runtime._handle._defer_setup_env = False
    if hasattr(runtime, "setup_env"):
      runtime.setup_env()

    if orig_compute_reward is not None:
      reward = float(orig_compute_reward(*args, **kwargs))
    elif hasattr(runtime, "_calculate_reward"):
      reward = float(runtime._calculate_reward(*args, **kwargs))
    else:
      reward = 0.0
    logging.info(
        "[SWEEnv] Evaluated agent patch in fresh eval container: reward=%.1f"
        " (patch_bytes=%d)",
        reward,
        len(patch),
    )
    return reward
  except Exception as e:
    logging.warning(
        "[SWEEnv] Exception during fresh container evaluation: %s", e
    )
    return 0.0


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


# Reference observation rendering (OpenHands 0d766ad06): commands.py
# CmdOutputObservation truncates content to 30000 chars at creation, and
# conversation_memory.py truncate_content applies max_message_chars=30000 to
# every rendered observation; both keep the head and tail halves.
_MAX_OBSERVATION_CHARS = 30000
_OBSERVATION_TRUNCATED_MARKER = (
    "\n[... Observation truncated due to length ...]\n"
)
# runtime/utils/bash_constants.py TIMEOUT_MESSAGE_TEMPLATE.
_BASH_TIMEOUT_MESSAGE = (
    "You may wait longer to see additional output by sending empty command '',"
    " send other commands to interact with the current process, send keys"
    ' ("C-c", "C-z", "C-d") to interrupt/kill the previous command before'
    " sending your new command, or use the timeout parameter in execute_bash"
    " for future commands."
)
# The execute_bash wrapper prints this marker, the final working directory
# and `which python` on exit, standing in for the reference's PS1 metadata.
_BASH_META_SENTINEL = "__OH_BASH_META__"


def _truncate_observation(content: str) -> str:
  if len(content) <= _MAX_OBSERVATION_CHARS:
    return content
  half = _MAX_OBSERVATION_CHARS // 2
  return content[:half] + _OBSERVATION_TRUNCATED_MARKER + content[-half:]


def _format_command_result(
    result: Any,
    timeout: float,
    elapsed: float,
    bash_observation: bool = False,
    truncate: bool = True,
    command_echo: str = "",
) -> tuple[str, bool]:
  """Formats an OpenHands CommandResult into (observation, timed_out).

  With bash_observation=True the text matches the reference
  CmdOutputObservation.to_agent_observation(): the output stripped, then
  "[The command completed with exit code N.]", the working directory, the
  Python interpreter and "[Command finished with exit code N]"; a command
  killed at its timeout gets the reference hard-timeout suffix instead.
  Like the reference tmux capture, every output line is rstripped, and
  `command_echo` (the command as the pane echoes it) leads the output.
  """
  if getattr(result, "stdout", None) is not None:
    stdout = str(result.stdout)
  elif getattr(result, "output", None) is not None:
    stdout = str(result.output)
  else:
    stdout = str(result)
  stderr = str(getattr(result, "stderr", "") or "")
  exit_code = getattr(result, "exit_code", None)
  timed_out = _command_timed_out(result, timeout, elapsed)

  if not bash_observation:
    obs = stdout
    if stderr and exit_code not in (0, None):
      obs = f"{stdout}\n{stderr}"
    if timed_out:
      obs = f"{obs.rstrip()}\n{_timeout_notice(timeout)}".lstrip("\n")
    return (_truncate_observation(obs) if truncate else obs), timed_out

  working_dir = python_path = ""
  marker = stdout.rfind(_BASH_META_SENTINEL)
  if marker != -1:
    meta = stdout[marker + len(_BASH_META_SENTINEL) :].strip("\n")
    working_dir, _, python_path = meta.partition("\t")
    stdout = stdout[:marker]
    if stdout.endswith("\n"):
      stdout = stdout[:-1]
  if timed_out:
    # Drop the SDK client's own deadline note; the reference renders a timed
    # out command as its partial output plus the timeout suffix below.
    stderr = re.sub(
        r"\n?Command timed out after [0-9.]+ seconds\.?\s*$", "", stderr
    )
  # The reference shell is a PTY, so stderr is part of the output. Without a
  # PTY, stdout and stderr come back separately; stdout first, then stderr.
  merged = stdout
  if stderr:
    if merged and not merged.endswith("\n"):
      merged += "\n"
    merged += stderr
  merged = "\n".join(line.rstrip() for line in merged.split("\n"))
  if command_echo:
    merged = f"{command_echo}\n{merged}"
  content = _truncate_observation(merged.strip())
  if timed_out:
    rendered = (
        f"{content}\n[The command timed out after {float(timeout)} seconds."
        f" {_BASH_TIMEOUT_MESSAGE}]"
    )
  else:
    code = int(exit_code) if exit_code is not None else -1
    rendered = f"{content}\n[The command completed with exit code {code}.]"
    if working_dir:
      rendered += f"\n[Current working directory: {working_dir}]"
    if python_path:
      rendered += f"\n[Python interpreter: {python_path}]"
    if code != -1:
      rendered += f"\n[Command finished with exit code {code}]"
  return _truncate_observation(rendered), timed_out

def _execute_in_workspace(
    env: Any,
    wrapped_cmd: str,
    step_timeout: float,
    failure_prefix: str,
    bash_observation: bool = False,
    truncate: bool = True,
    command_echo: str = "",
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
        result,
        step_timeout,
        time.monotonic() - start,
        bash_observation,
        truncate,
        command_echo,
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


def _openhands_session_id(env: Any) -> str:
  """Per-trajectory id in the format of OpenHands generate_sid (setup.py)."""
  sid = getattr(env, "_oh_session_id", None)
  if not isinstance(sid, str) or not sid:
    session_name = str(uuid.uuid4())
    hash_str = hashlib.sha256(session_name.encode("utf-8")).hexdigest()
    sid = f"{session_name[:16]}-{hash_str[:15]}"
    try:
      env._oh_session_id = sid
    except AttributeError:
      pass
  return sid


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
      task_file_path = f"sessions/{_openhands_session_id(env)}/TASKS.md"
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
      # The reference passes file views to the model whole and truncates only
      # edit results (conversation_memory FileReadObservation vs
      # FileEditObservation).
      return _execute_in_workspace(
          env,
          remote_cmd,
          step_timeout,
          "Command execution failed",
          truncate=params.get("command") != "view",
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
    py_cmd = (
        f"import base64; exec(base64.b64decode('{b64_code}').decode('utf-8'))"
    )
    wrapped_cmd = (
        _STRIP_PYINSTALLER_LD_PATH
        + "if [ -d /testbed/.venv/bin ]; then export"
        ' PATH="/testbed/.venv/bin:${PATH}"; fi; '
        + "if [ -d /testbed ]; then cd /testbed; elif [ -d /workspace ]; then"
        " cd /workspace; fi; "
        + "if [ -x /testbed/.venv/bin/python ]; then"
        f' /testbed/.venv/bin/python -c "{py_cmd}";'
        f' else python3 -c "{py_cmd}"; fi'
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
    # Stripped like the reference (bash.py `action.command.strip()`); the
    # echo rule below depends on it.
    cmd = str(cmd or "").strip()
    # Each command here runs to completion (or is killed at its timeout), so
    # there is never a previous command to read from or send keys to. These
    # are the reference BashSession replies for that state, without metadata.
    if not cmd or is_input:
      return EnvStepResult(
          observation=(
              "ERROR: No previous running command to retrieve logs from."
              if not cmd
              else "ERROR: No previous running command to interact with."
          ),
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

    # The closing parenthesis goes on its own line. Appended to the command's
    # last line, it would be swallowed by a heredoc terminator (`EOF)`) or a
    # trailing `# comment`, and the shell fails with a syntax error.
    # The EXIT trap persists the working directory for the next command and
    # prints the metadata that _format_command_result renders like the
    # reference PS1 block (cwd and `which python`), keeping the exit code.
    wrapped_cmd = (
        "("
        + _STRIP_PYINSTALLER_LD_PATH
        + "if [ -d /testbed/.venv/bin ]; then export"
        ' PATH="/testbed/.venv/bin:${PATH}"; fi; '
        + "__oh_cwd=$(cat /var/tmp/.oh_cwd 2>/dev/null); "
        'if [ -n "$__oh_cwd" ] && [ -d "$__oh_cwd" ]; then cd "$__oh_cwd"; '
        "elif [ -d /testbed ]; then cd /testbed; else cd /workspace; fi; "
        "trap '__oh_rc=$?; pwd > /var/tmp/.oh_cwd 2>/dev/null; "
        f'printf "\\n{_BASH_META_SENTINEL}%s\\t%s\\n" "$(pwd)"'
        ' "$(which python 2>/dev/null || echo "")"; exit $__oh_rc\' EXIT; '
        f"{cmd}\n)"
    )

    # The reference strips the echoed command from the tmux pane only when the
    # pane shows it verbatim. The pane drops empty lines and trailing spaces,
    # so a command with either keeps its echo at the top of the output.
    pane_echo = "\n".join(
        line.rstrip() for line in cmd.split("\n") if line != ""
    )

    if getattr(env, "workspace", None) is not None:
      return _execute_in_workspace(
          env,
          wrapped_cmd,
          step_timeout,
          "Command execution failed",
          bash_observation=True,
          command_echo=pane_echo if pane_echo != cmd else "",
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

