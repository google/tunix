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

"""SWE-Gym task support for the DeepSWE harness.

SWE-Gym (HF dataset ``SWE-Gym/SWE-Gym``) instances are graded SWE-bench style:
the per-instance image ``xingyaoww/sweb.eval.x86_64.<id>`` ships the repo at
``base_commit`` under ``/testbed`` with a ``testbed`` conda env. Grading applies
the agent patch, resets the files touched by ``test_patch`` and applies
``test_patch``, runs the repo/version test command on the test files touched by
``test_patch`` and parses the pytest ``-rA`` summary. The reward is 1.0 iff all
FAIL_TO_PASS and all PASS_TO_PASS tests pass (SWE-bench ``ResolvedStatus.FULL``).

Test commands and log parsers follow the SWE-Gym fork of the SWE-bench harness
(https://github.com/SWE-Gym/SWE-Bench-Fork, swebench/harness/{constants,
log_parsers,test_spec}.py). The upstream ``swebench`` package does not cover the
SWE-Gym repos, so the small pieces needed here are vendored (stdlib only).
Dataset rows built by the SWE-Gym converter carry a precomputed ``test_cmd`` and
``eval_commands``; the vendored table is only a fallback.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shlex
from typing import Any, Optional

TASK_SOURCE = "swegym"
SOURCE_IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64."
_IMAGE_MARKER = "sweb.eval.x86_64."
REPO_PATH = "/testbed"

# Per-repo base test commands from SWE-Bench-Fork MAP_REPO_VERSION_TO_SPECS
# (constant across versions for every SWE-Gym repo except python/mypy, where
# some old versions drop `-n0`; `-n0` is harmless with xdist installed).
_BASE_TEST_CMDS = {
    "getmoto/moto": "pytest -n0 -rA",
    "python/mypy": "pytest -n0 -rA -k",
    "conan-io/conan": "pytest -n0 -rA",
    "iterative/dvc": "pytest -rA",
    "dask/dask": "pytest -n0 -rA  --color=no",
    "pydantic/pydantic": (
        "pytest -rA --tb=short -vv -o console_output_style=classic --no-header"
    ),
    "pandas-dev/pandas": "pytest -rA --tb=long",
    "facebookresearch/hydra": "pytest -rA --tb=long",
    "bokeh/bokeh": "pytest -rA -n0",
    "project-monai/monai": "pytest -rA ",
    "modin-project/modin": "pytest -n0 -rA",
}
_EVAL_COMMANDS = {
    "conan-io/conan": ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"],
}
# From SWE-Bench-Fork swebench/harness/constants.py NON_TEST_EXTS.
_NON_TEST_EXTS = (
    ".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml",
    ".yaml", ".toml",
)
_DIFF_B_PATH_RE = re.compile(r"^diff --git a/.* b/(.*)$", re.MULTILINE)
_MYPY_CASE_RE = re.compile(r"\[case ([^\]]+)\]")
# SWE-bench TestStatus values.
_STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL")
_PASSING = ("PASSED", "XFAIL")

START_TEST_OUTPUT = ">>>>> SWEGYM Start Test Output"
END_TEST_OUTPUT = ">>>>> SWEGYM End Test Output"
TEST_PATCH_FAILED = ">>>>> SWEGYM Test Patch Apply Failed"


def is_swegym_entry(entry: Optional[dict[str, Any]]) -> bool:
  """Returns True if the dataset entry is a SWE-Gym task."""
  if not isinstance(entry, dict):
    return False
  source = str(entry.get("task_source") or "").strip().lower()
  if source:
    return source == TASK_SOURCE
  if entry.get("is_swegym"):
    return True
  img = str(entry.get("docker_image") or entry.get("image_name") or "").lower()
  return SOURCE_IMAGE_PREFIX in img or (_IMAGE_MARKER in img and "_s_" in img)


def source_image_for_instance(instance_id: str) -> str:
  """Returns the Docker Hub image for a SWE-Gym instance id."""
  return (
      SOURCE_IMAGE_PREFIX
      + instance_id.strip().lower().replace("__", "_s_")
      + ":latest"
  )


def parse_test_list(val: Any) -> list[str]:
  """Normalizes FAIL_TO_PASS / PASS_TO_PASS (list or JSON string) to a list."""
  if val is None:
    return []
  if hasattr(val, "tolist"):
    val = val.tolist()
  if isinstance(val, (list, tuple)):
    return [str(x) for x in val if str(x).strip()]
  if isinstance(val, str):
    s = val.strip()
    if not s:
      return []
    if s.startswith("["):
      try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
          return [str(x) for x in parsed if str(x).strip()]
      except ValueError:
        pass
    return [s]
  return []


def get_test_patch_files(test_patch: str) -> list[str]:
  """Returns every file path touched by `test_patch` (post-image paths)."""
  seen: set[str] = set()
  files: list[str] = []
  for path in _DIFF_B_PATH_RE.findall(test_patch or ""):
    path = path.strip()
    if path and path not in seen:
      seen.add(path)
      files.append(path)
  return files


def get_test_directives(entry: dict[str, Any]) -> list[str]:
  """SWE-bench `get_test_directives`: test files touched by test_patch."""
  return [
      d
      for d in get_test_patch_files(str(entry.get("test_patch") or ""))
      if not any(d.endswith(ext) for ext in _NON_TEST_EXTS)
  ]


def get_test_command(entry: dict[str, Any]) -> str:
  """Returns the full test command (precomputed `test_cmd` if present)."""
  precomputed = str(entry.get("test_cmd") or "").strip()
  if precomputed:
    return precomputed
  repo = str(entry.get("repo") or entry.get("repo_name") or "").lower()
  base = _BASE_TEST_CMDS.get(repo, "pytest -rA")
  if repo == "python/mypy":
    keys = _MYPY_CASE_RE.findall(str(entry.get("test_patch") or ""))
    return f'{base} "{" or ".join(keys)}"'
  return " ".join([base, *get_test_directives(entry)])


def get_eval_commands(entry: dict[str, Any]) -> list[str]:
  """Returns extra repo-specific shell commands to run before the tests."""
  raw = entry.get("eval_commands")
  if raw is not None and raw != "":
    return parse_test_list(raw)
  repo = str(entry.get("repo") or entry.get("repo_name") or "").lower()
  return list(_EVAL_COMMANDS.get(repo, []))


def parse_test_log(log: str, repo: str = "") -> dict[str, str]:
  """Parses pytest `-rA` output into {test_id: status}.

  Vendored from SWE-Bench-Fork log_parsers.py: every SWE-Gym repo uses
  `parse_log_pytest` except pydantic, which uses `parse_log_pytest_pydantic`.

  Args:
    log: Raw test output.
    repo: Repository name (e.g. "pydantic/pydantic").

  Returns:
    Mapping from test id to SWE-bench status string.
  """
  status_map: dict[str, str] = {}
  if not log:
    return status_map
  pydantic = repo.lower() == "pydantic/pydantic"
  escapes = "".join(chr(c) for c in range(1, 32))
  translator = str.maketrans("", "", escapes)
  for line in log.split("\n"):
    if pydantic:
      line = re.sub(r"\[(\d+)m", "", line)
      line = line.translate(translator)
      line = re.sub(r"FAILED\s*\[.*?\]", "FAILED", line)
    if any(line.startswith(s) for s in _STATUSES):
      if line.startswith("FAILED"):
        line = line.replace(" - ", " ")
      test_case = line.split()
      if len(test_case) <= 1:
        continue
      status_map[test_case[1]] = test_case[0]
    elif pydantic and any(line.endswith(s) for s in _STATUSES):
      test_case = line.split()
      if len(test_case) >= 2:
        status_map[test_case[0]] = test_case[1]
  return status_map


def extract_test_output(output: str) -> Optional[str]:
  """Returns the text between the test markers, or None if tests did not run."""
  if not output or TEST_PATCH_FAILED in output:
    return None
  if START_TEST_OUTPUT not in output:
    return None
  body = output.split(START_TEST_OUTPUT, 1)[1]
  return body.split(END_TEST_OUTPUT, 1)[0]


def _strip_control_chars(s: str) -> str:
  return "".join(ch for ch in s if ord(ch) >= 32)


def is_resolved(
    status_map: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> bool:
  """SWE-bench FULL resolution: every F2P and P2P test is PASSED/XFAIL.

  Test ids are matched exactly, then with ASCII control characters removed
  (the pydantic parser strips them from log lines, but some dataset ids keep
  them, e.g. parametrize ids containing "\\x12").
  """
  if not fail_to_pass:
    return False
  normalized = None
  for test in list(fail_to_pass) + list(pass_to_pass):
    status = status_map.get(test)
    if status is None:
      if normalized is None:
        normalized = {_strip_control_chars(k): v for k, v in status_map.items()}
      status = normalized.get(_strip_control_chars(test))
    if status not in _PASSING:
      return False
  return True


def build_rollout_setup_cmds(repo_path: str = REPO_PATH) -> list[str]:
  """Shell commands that hide future history in the rollout container.

  SWE-bench-style images are full clones reset to base_commit, so the fix is
  reachable via the reflog (`HEAD@{1}: clone`), ORIG_HEAD and release tags. This
  squashes the working tree into a single orphan commit and prunes everything
  else so the agent cannot recover the upstream fix from git.

  Args:
    repo_path: Repository path in the container.

  Returns:
    List of shell commands (each safe to join with ` && `).
  """
  q = shlex.quote(repo_path)
  return [
      (
          f"(cd {q} && _prev=$(git rev-parse HEAD 2>/dev/null) && "
          "if git checkout -q --orphan _oh_base >/dev/null 2>&1 && "
          "git add -A >/dev/null 2>&1 && "
          "git -c user.email=openhands@agent.sandbox -c 'user.name=OpenHands"
          " Agent' commit -q -m 'Initial commit' --no-verify >/dev/null 2>&1; "
          "then git branch -M _oh_base main >/dev/null 2>&1; "
          "for _ref in $(git for-each-ref --format='%(refname)' 2>/dev/null); "
          'do [ "$_ref" = "refs/heads/main" ] || git update-ref -d "$_ref"'
          " 2>/dev/null || true; done; "
          "rm -rf .git/logs .git/ORIG_HEAD .git/FETCH_HEAD 2>/dev/null; "
          "git reflog expire --expire=now --all >/dev/null 2>&1; "
          "git gc -q --prune=now >/dev/null 2>&1; "
          'else git checkout -q -f "$_prev" >/dev/null 2>&1; '
          "git branch -D _oh_base >/dev/null 2>&1; fi; true)"
      ),
  ]


def _unpack(res: Any) -> tuple[str, int]:
  """Normalizes runtime.run / execute_command results to (output, exit_code)."""
  if res is None:
    return "", -1
  if isinstance(res, tuple) and len(res) == 2:
    out, code = res
    return str(out or ""), (0 if str(code).strip() == "0" else -1)
  code = getattr(res, "exit_code", 0)
  try:
    code_int = int(code)
  except (TypeError, ValueError):
    code_int = 0 if str(code).strip() == "0" else -1
  out = getattr(res, "stdout", None) or ""
  return str(out), code_int


def _sh(runtime: Any, cmd: str, timeout: int) -> tuple[str, int]:
  return _unpack(runtime.run(f"/bin/bash -c {shlex.quote(cmd)}", timeout=timeout))


def write_file_to_container(runtime: Any, content: str, path: str) -> bool:
  """Writes `content` to `path` in the container via base64 (chunked)."""
  b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
  q_path = shlex.quote(path)
  q_tmp = shlex.quote(path + ".b64")
  chunk_size = 49152
  for idx in range(0, max(len(b64), 1), chunk_size):
    chunk = b64[idx : idx + chunk_size]
    redir = ">" if idx == 0 else ">>"
    _, rc = _sh(runtime, f"printf '%s' {shlex.quote(chunk)} {redir} {q_tmp}", 30)
    if rc != 0:
      _sh(runtime, f"rm -f {q_tmp}", 15)
      return False
  _, rc = _sh(runtime, f"base64 -d {q_tmp} > {q_path} && rm -f {q_tmp}", 30)
  return rc == 0


def build_eval_script(entry: dict[str, Any], repo_path: str = REPO_PATH) -> str:
  """Builds the bash script that applies test_patch and runs the tests.

  Mirrors SWE-Bench-Fork `make_eval_script_list` without re-running the
  repo `install` step (images are already installed in develop/editable mode).
  Expects the test patch at /tmp/.swegym_test.patch.

  Args:
    entry: SWE-Gym dataset entry.
    repo_path: Repository path in the container.

  Returns:
    Bash script text.
  """
  q_repo = shlex.quote(repo_path)
  test_files = get_test_patch_files(str(entry.get("test_patch") or ""))
  reset_lines = [
      f"if git cat-file -e HEAD:{shlex.quote(f)} 2>/dev/null; then "
      f"git checkout -q HEAD -- {shlex.quote(f)}; else rm -f -- {shlex.quote(f)}; fi"
      for f in test_files
  ]
  lines = [
      # No `set -u`: conda activate.d scripts (e.g. binutils in the pandas
      # env) reference unset variables.
      "source /opt/miniconda3/bin/activate",
      "conda activate testbed",
      "set -o pipefail",
      f"cd {q_repo}",
      *get_eval_commands(entry),
      f"git config --global --add safe.directory {q_repo} 2>/dev/null || true",
      f"cd {q_repo}",
      *reset_lines,
      (
          "git apply -v /tmp/.swegym_test.patch >/dev/null 2>&1 || "
          "git apply -v --3way /tmp/.swegym_test.patch >/dev/null 2>&1 || "
          f"{{ echo '{TEST_PATCH_FAILED}'; exit 3; }}"
      ),
      "rm -f /tmp/.swegym_test.patch",
      f"echo '{START_TEST_OUTPUT}'",
      get_test_command(entry),
      f"echo '{END_TEST_OUTPUT}'",
      *reset_lines,
  ]
  return "\n".join(lines) + "\n"


def evaluate_swegym_in_fresh_container(
    runtime: Any,
    entry: dict[str, Any],
    patch: str,
    timeout: int = 1200,
) -> float:
  """Applies `patch` in the fresh SWE-Gym `eval` container and grades F2P/P2P.

  Args:
    runtime: R2E-Gym-style runtime exposing `run(cmd, timeout=...)`.
    entry: SWE-Gym dataset entry (needs test_patch, FAIL_TO_PASS, PASS_TO_PASS).
    patch: Agent (or gold) patch as a unified git diff; may be empty.
    timeout: Timeout in seconds for the test run.

  Returns:
    1.0 iff all FAIL_TO_PASS and PASS_TO_PASS tests pass, else 0.0.
  """
  f2p = parse_test_list(entry.get("FAIL_TO_PASS"))
  p2p = parse_test_list(entry.get("PASS_TO_PASS"))
  test_patch = str(entry.get("test_patch") or "")
  repo = str(entry.get("repo") or entry.get("repo_name") or "")
  instance_id = str(entry.get("instance_id") or "")
  if not f2p or not test_patch:
    logging.warning(
        "[SWEEnv] SWE-Gym entry %s missing FAIL_TO_PASS/test_patch; reward=0.0",
        instance_id,
    )
    return 0.0
  repo_path = getattr(runtime, "repo_path", REPO_PATH) or REPO_PATH
  q_repo = shlex.quote(repo_path)

  # The eval container is fresh (agent never touched it): clean any leftovers.
  _sh(runtime, f"cd {q_repo} && git reset -q --hard && git clean -fdq", 120)

  if patch and patch.strip():
    test_files = get_test_patch_files(test_patch)
    out, _ = _sh(
        runtime, f"cd {q_repo} && git ls-files --others --exclude-standard", 60
    )
    untracked = [f.strip() for f in out.splitlines() if f.strip()]
    exclude = " ".join(
        shlex.quote(f"--exclude={f}")
        for f in dict.fromkeys(untracked + test_files)
    )
    if not write_file_to_container(runtime, patch, "/tmp/model.patch"):
      logging.warning("[SWEEnv] Failed to write SWE-Gym model patch")
      return 0.0
    apply_cmd = (
        f"cd {q_repo} && "
        f"(git apply --whitespace=fix {exclude} /tmp/model.patch || "
        f"git apply --whitespace=fix --reject {exclude} /tmp/model.patch || "
        "patch --batch --fuzz=5 -p1 -i /tmp/model.patch); "
        "_rc=$?; rm -f /tmp/model.patch; exit $_rc"
    )
    apply_out, apply_rc = _sh(runtime, apply_cmd, 120)
    if apply_rc != 0:
      logging.warning(
          "[SWEEnv] Failed to apply SWE-Gym patch for %s: %s",
          instance_id,
          apply_out[-2000:],
      )
      return 0.0

  if not write_file_to_container(runtime, test_patch, "/tmp/.swegym_test.patch"):
    logging.warning("[SWEEnv] Failed to write SWE-Gym test patch")
    return 0.0
  if not write_file_to_container(
      runtime, build_eval_script(entry, repo_path), "/tmp/.swegym_eval.sh"
  ):
    logging.warning("[SWEEnv] Failed to write SWE-Gym eval script")
    return 0.0
  output, _ = _sh(
      runtime,
      "bash /tmp/.swegym_eval.sh 2>&1; _rc=$?; rm -f /tmp/.swegym_eval.sh; "
      "exit $_rc",
      int(timeout),
  )
  test_output = extract_test_output(output)
  if test_output is None:
    logging.warning(
        "[SWEEnv] SWE-Gym tests did not run for %s: %s",
        instance_id,
        output[-2000:],
    )
    return 0.0
  status_map = parse_test_log(test_output, repo)
  resolved = is_resolved(status_map, f2p, p2p)
  if not resolved:
    missing = [t for t in f2p + p2p if status_map.get(t) not in _PASSING]
    logging.info(
        "[SWEEnv] SWE-Gym %s unresolved: %d/%d tests not passing (e.g. %s)",
        instance_id,
        len(missing),
        len(f2p) + len(p2p),
        missing[:3],
    )
  return 1.0 if resolved else 0.0
