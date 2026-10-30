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

"""Harbor & SWE-rebench-V2 dataset normalization and late-bound verifier."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import dataclasses
import io
import json
import logging
import math
import os
import pathlib
import posixpath
import re
import shlex
import subprocess
import tarfile
import tomllib
from typing import Any, Callable

from examples.deepswe import score_tests

SWE_GYM = "swe_gym"
SWE_BENCH = "swe_bench"
SWE_REBENCH = "swe_rebench"
SWE_REBENCH_V2 = "swe_rebench_v2"
SWE_SMITH = "swe_smith"
R2E_GYM = "r2e_gym"
ENDLESS_TERMINALS = "endless_terminals"

DEFAULT_TEST_CMD = "pytest --no-header -rA --tb=short -o console_output_style=classic"
R2E_GYM_TEST_CMD = "bash run_tests.sh"
R2E_HARNESS_SNAPSHOT_DIR = "/opt/harbor/r2e_harness"

DEFAULT_REPO_TEST_CMDS: dict[str, str] = {
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "sympy/sympy": "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning bin/test -C --verbose",
    "sphinx-doc/sphinx": "tox --current-env -epy39 -v --",
}

DEFAULT_IMAGE_TEMPLATES: dict[str, str] = {
    SWE_GYM: "xingyaoww/sweb.eval.x86_64.{instance_id_sweb}",
    SWE_BENCH: "swebench/sweb.eval.x86_64.{instance_id_sweb}:latest",
    SWE_REBENCH: "swerebench/sweb.eval.x86_64.{instance_id_sweb}:latest",
    SWE_REBENCH_V2: "swerebench/sweb.eval.x86_64.{instance_id_sweb}:latest",
    SWE_SMITH: "jyangballin/swesmith.x86_64.{repo_owner}_1776_{repo_name}.{short_commit}",
    ENDLESS_TERMINALS: "hamishivi/swerl-endless-terminals:{instance_id}",
}

VERIFIER_LOG_DIR = "/logs/verifier"
REWARD_TXT_PATH = f"{VERIFIER_LOG_DIR}/reward.txt"
REWARD_JSON_PATH = f"{VERIFIER_LOG_DIR}/reward.json"
RESULTS_JSON_PATH = f"{VERIFIER_LOG_DIR}/results.json"
CONTAINER_TESTS_DIR = "/tests"
CONTAINER_TEST_SH = f"{CONTAINER_TESTS_DIR}/test.sh"

_TASK_ID_INVALID_CHARS = re.compile(r"[^a-z0-9._-]+")
_PATCH_PLUS_HEADER = re.compile(r"^\+\+\+\s+(?:b/)?(?P<path>\S+)")
_PATCH_GIT_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)")
_DJANGO_UNITTEST_ID = re.compile(r"^(?P<method>\w+)\s+\((?P<cls>[\w.]+)\)$")

_SYNTHETIC_PARSED_COMMIT_TEMPLATE = {
    "file_diffs": [],
    "old_commit_hash": "0000000",
    "new_commit_hash": "1111111",
    "commit_message": "harbor verifier task",
    "commit_date": "2026-01-01T00:00:00",
}


def sanitize_task_id(raw_id: str) -> str:
  """Normalizes an instance id into a filesystem- and K8s-safe directory name."""
  lowered = raw_id.strip().lower().replace("/", "__").replace(":", "_")
  cleaned = _TASK_ID_INVALID_CHARS.sub("-", lowered).strip("-")
  if not cleaned:
    raise ValueError(f"Cannot derive a non-empty task id from {raw_id!r}")
  return cleaned


def parse_test_list(raw: Any) -> tuple[str, ...]:
  """Parses `FAIL_TO_PASS` / `PASS_TO_PASS` fields into a tuple of test ids."""
  if raw is None:
    return ()
  if isinstance(raw, str):
    stripped = raw.strip()
    if not stripped:
      return ()
    if stripped.startswith("["):
      try:
        decoded = json.loads(stripped)
      except json.JSONDecodeError:
        decoded = None
      if isinstance(decoded, list):
        return tuple(str(item).strip() for item in decoded if str(item).strip())
    return tuple(line.strip() for line in stripped.splitlines() if line.strip())
  if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
    return tuple(str(item).strip() for item in raw if str(item).strip())
  raise TypeError(f"Unsupported test list type {type(raw).__name__}: {raw!r}")


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
  for key in keys:
    value = record.get(key)
    if value is not None and value != "":
      return value
  return None


def _clean_str(value: Any) -> str:
  if value is None:
    return ""
  return str(value).strip()


def _extract_install_config(
    record: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
  """Extracts `(test_cmd, pre_test_commands)` from SWE-rebench `install_config`."""
  raw = record.get("install_config")
  if not raw:
    return "", ()
  if isinstance(raw, str):
    try:
      config = json.loads(raw)
    except json.JSONDecodeError:
      return "", ()
  elif isinstance(raw, Mapping):
    config = raw
  else:
    return "", ()

  test_cmd = _clean_str(
      config.get("test_cmd")
      or config.get("test_command")
      or config.get("eval_cmd")
  )
  pre_commands: list[str] = []
  for key in ("eval_commands", "pre_test_commands", "install"):
    value = config.get(key)
    if isinstance(value, str) and value.strip():
      pre_commands.append(value.strip())
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
      pre_commands.extend(
          str(item).strip() for item in value if str(item).strip()
      )
  return test_cmd, tuple(pre_commands)


@dataclasses.dataclass(frozen=True)
class RealWorldTask:
  """Canonical representation of one real-world software engineering task."""

  instance_id: str
  repo: str
  base_commit: str
  problem_statement: str
  patch: str
  test_patch: str
  fail_to_pass: tuple[str, ...]
  pass_to_pass: tuple[str, ...]
  source_dataset: str
  version: str = ""
  environment_setup_commit: str = ""
  hints_text: str = ""
  created_at: str = ""
  license: str = ""
  language: str = "python"
  difficulty: str = ""
  docker_image: str = ""
  test_cmd: str = ""
  pre_test_commands: tuple[str, ...] = ()
  extra_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def task_id(self) -> str:
    return sanitize_task_id(self.instance_id)

  @property
  def repo_owner(self) -> str:
    owner, _, _ = self.repo.partition("/")
    return owner

  @property
  def repo_name(self) -> str:
    _, _, name = self.repo.partition("/")
    return name or self.repo

  @property
  def uses_r2e_harness(self) -> bool:
    return self.source_dataset == R2E_GYM or bool(
        self.extra_metadata.get("r2e_harness")
    )

  def validation_errors(self) -> list[str]:
    errors: list[str] = []
    if not self.instance_id:
      errors.append("missing instance_id")
    if not self.repo or "/" not in self.repo:
      errors.append(f'repo must be "owner/name", got {self.repo!r}')
    if not self.base_commit:
      errors.append("missing base_commit")
    if not self.problem_statement:
      errors.append("missing problem_statement")
    if not self.fail_to_pass and not self.test_patch:
      errors.append("need FAIL_TO_PASS tests or a test_patch to verify against")
    return errors


def normalize_record(
    record: Mapping[str, Any],
    source_dataset: str = SWE_REBENCH_V2,
) -> RealWorldTask:
  """Converts a raw SWE-bench / SWE-rebench-v2 / SWE-gym row into `RealWorldTask`."""
  source = source_dataset.strip().lower()
  base_commit = _clean_str(
      _first_present(record, ("base_commit", "commit_hash", "commit"))
  )
  raw_repo = _clean_str(_first_present(record, ("repo", "repo_name")))
  if raw_repo and "/" not in raw_repo:
    repo = f"repo/{raw_repo}"
  else:
    repo = raw_repo
  instance_id = _clean_str(
      _first_present(record, ("instance_id", "task_id", "id"))
  )
  if not instance_id and raw_repo and base_commit:
    instance_id = f"{raw_repo.replace('/', '__')}__{base_commit[:12]}"
  problem_statement = _clean_str(
      _first_present(
          record, ("problem_statement", "prompt", "instruction", "issue")
      )
  )
  patch = _clean_str(
      _first_present(record, ("patch", "gold_patch", "model_patch"))
  )
  test_patch = _clean_str(
      _first_present(record, ("test_patch", "eval_patch"))
  )
  fail_to_pass = parse_test_list(
      _first_present(record, ("FAIL_TO_PASS", "fail_to_pass"))
  )
  pass_to_pass = parse_test_list(
      _first_present(record, ("PASS_TO_PASS", "pass_to_pass"))
  )

  config_test_cmd, pre_commands = _extract_install_config(record)
  test_cmd = (
      _clean_str(
          _first_present(record, ("test_cmd", "test_command", "eval_script"))
      )
      or config_test_cmd
  )
  docker_image = _clean_str(
      _first_present(
          record, ("docker_image", "image_name", "image", "container_image")
      )
  )
  language = _clean_str(_first_present(record, ("language", "repo_language", "lang"))) or "python"

  return RealWorldTask(
      instance_id=instance_id,
      repo=repo,
      base_commit=base_commit,
      problem_statement=problem_statement,
      patch=patch,
      test_patch=test_patch,
      fail_to_pass=fail_to_pass,
      pass_to_pass=pass_to_pass,
      source_dataset=source,
      version=_clean_str(record.get("version")),
      environment_setup_commit=_clean_str(record.get("environment_setup_commit")),
      hints_text=_clean_str(record.get("hints_text")),
      created_at=_clean_str(record.get("created_at")),
      license=_clean_str(_first_present(record, ("license", "repo_license"))),
      language=language.lower(),
      difficulty=_clean_str(record.get("difficulty")),
      docker_image=docker_image,
      test_cmd=test_cmd,
      pre_test_commands=pre_commands,
      extra_metadata={},
  )


@dataclasses.dataclass(frozen=True)
class HarborProfile:
  """Configuration for Harbor verifier synthesis."""

  workdir: str = "/testbed"
  default_test_cmd: str = DEFAULT_TEST_CMD
  repo_test_cmds: Mapping[str, str] = dataclasses.field(
      default_factory=lambda: dict(DEFAULT_REPO_TEST_CMDS)
  )
  reward_mode: str = score_tests.BINARY
  strict_pass_to_pass: bool = False

  def test_cmd_for(self, task: RealWorldTask) -> str:
    if task.test_cmd:
      return task.test_cmd
    if task.uses_r2e_harness:
      return R2E_GYM_TEST_CMD
    if task.repo in self.repo_test_cmds:
      return self.repo_test_cmds[task.repo]
    return self.default_test_cmd


def extract_patch_paths(patch_text: str) -> tuple[list[str], list[str]]:
  """Splits a unified diff into `(all_paths, tracked_paths)`."""
  all_paths: list[str] = []
  tracked_paths: list[str] = []
  seen_all: set[str] = set()
  seen_tracked: set[str] = set()
  pending_old_path: str | None = None

  def _record(path: str, is_new_file: bool) -> None:
    cleaned = path.strip()
    if not cleaned or cleaned == "/dev/null":
      return
    if cleaned not in seen_all:
      seen_all.add(cleaned)
      all_paths.append(cleaned)
    if not is_new_file and cleaned not in seen_tracked:
      seen_tracked.add(cleaned)
      tracked_paths.append(cleaned)

  for line in patch_text.splitlines():
    git_match = _PATCH_GIT_HEADER.match(line)
    if git_match:
      _record(git_match.group("b"), is_new_file=False)
      pending_old_path = None
      continue
    if line.startswith("--- "):
      raw_old = line[4:].split("\t", 1)[0].strip()
      if raw_old.startswith("a/"):
        raw_old = raw_old[2:]
      pending_old_path = raw_old
      continue
    plus_match = _PATCH_PLUS_HEADER.match(line)
    if plus_match:
      target = plus_match.group("path").split("\t", 1)[0]
      _record(target, is_new_file=(pending_old_path == "/dev/null"))
      pending_old_path = None
  return all_paths, tracked_paths


def _is_Clean_pytest_target(test_id: str) -> bool:
  try:
    test_id.encode("ascii")
  except UnicodeEncodeError:
    return False
  if "[" in test_id and not test_id.endswith("]"):
    return False
  return " " not in test_id


def _is_test_source_path(path: str) -> bool:
  base = posixpath.basename(path)
  return base.endswith(".py") and (
      base.startswith("test") or base.endswith("_test.py") or "tests" in path
  )


def _django_directives_from_paths(paths: Sequence[str]) -> list[str]:
  directives: list[str] = []
  seen: set[str] = set()
  for path in paths:
    if not path.endswith(".py"):
      continue
    parts = [part for part in path.split("/") if part]
    if parts and parts[0] == "tests":
      parts = parts[1:]
    if not parts:
      continue
    if parts[-1] == "__init__.py":
      parts = parts[:-1]
    else:
      parts[-1] = parts[-1][:-3]
    directive = ".".join(parts)
    if directive and directive not in seen:
      seen.add(directive)
      directives.append(directive)
  return directives


def _django_directives_from_ids(test_ids: Sequence[str]) -> list[str]:
  directives: list[str] = []
  seen: set[str] = set()
  for test_id in test_ids:
    match = _DJANGO_UNITTEST_ID.match(test_id.strip())
    if not match:
      continue
    module, _, _ = match.group("cls").rpartition(".")
    if module.startswith("tests."):
      module = module[len("tests.") :]
    if module and module not in seen:
      seen.add(module)
      directives.append(module)
  return directives


def select_test_targets(task: RealWorldTask) -> list[str]:
  """Selects CLI test targets for `test_cmd`."""
  if task.uses_r2e_harness:
    return []
  patch_paths, _ = extract_patch_paths(task.test_patch)
  test_patch_files = [path for path in patch_paths if _is_test_source_path(path)]
  combined = list(task.fail_to_pass) + list(task.pass_to_pass)
  if task.repo == "django/django":
    from_patch = _django_directives_from_paths(test_patch_files)
    return from_patch or _django_directives_from_ids(combined)
  if not combined:
    return test_patch_files

  whole_files: set[str] = set()
  for raw_id in combined:
    if "::" in raw_id and not _is_Clean_pytest_target(raw_id):
      file_part, _, _ = raw_id.partition("::")
      if file_part:
        whole_files.add(file_part)

  seen: set[str] = set()
  targets: list[str] = []
  for raw_id in combined:
    if "::" not in raw_id:
      continue
    file_part, _, _ = raw_id.partition("::")
    target = file_part if file_part in whole_files else raw_id
    if target and target not in seen:
      seen.add(target)
      targets.append(target)
  return targets or test_patch_files


def render_test_config(task: RealWorldTask, profile: HarborProfile) -> str:
  payload = {
      "instance_id": task.instance_id,
      "repo": task.repo,
      "base_commit": task.base_commit,
      "FAIL_TO_PASS": list(task.fail_to_pass),
      "PASS_TO_PASS": list(task.pass_to_pass),
      "reward_mode": profile.reward_mode,
      "strict_pass_to_pass": profile.strict_pass_to_pass,
      "test_cmd": profile.test_cmd_for(task),
  }
  return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_test_script(task: RealWorldTask, profile: HarborProfile) -> str:
  """Renders `tests/test.sh` for running inside the evaluation container."""
  test_cmd = profile.test_cmd_for(task)
  targets = select_test_targets(task)
  quoted_targets = "\n".join(f"  {shlex.quote(target)}" for target in targets)
  all_patch_paths, tracked_patch_paths = extract_patch_paths(task.test_patch)
  quoted_all_paths = " ".join(shlex.quote(path) for path in all_patch_paths)
  quoted_tracked_paths = " ".join(
      shlex.quote(path) for path in tracked_patch_paths
  )
  pre_commands = "\n".join(task.pre_test_commands) or ": # no pre-test commands"
  if tracked_patch_paths:
    checkout_block = (
        f"for path in {quoted_tracked_paths}; do\n"
        f"  git checkout {shlex.quote(task.base_commit)} -- \"$path\""
        " 2>/dev/null || true\n"
        "done"
    )
  else:
    checkout_block = ": # test patch touches no tracked files"
  if all_patch_paths:
    clean_block = (
        f"  for path in {quoted_all_paths}; do\n"
        '    rm -rf -- "$path" 2>/dev/null || true\n'
        "  done\n"
        f"  {checkout_block}"
    )
  else:
    clean_block = "  : # empty test patch"
  if task.uses_r2e_harness:
    r2e_restore_block = (
        f"for harness_dir in {R2E_HARNESS_SNAPSHOT_DIR} /root /; do\n"
        '  if [[ -d "$harness_dir/r2e_tests" && ! -d "$TESTBED/r2e_tests" ]];'
        " then\n"
        '    cp -r "$harness_dir/r2e_tests" "$TESTBED/r2e_tests" 2>/dev/null ||'
        " true\n"
        "  fi\n"
        '  if [[ -f "$harness_dir/run_tests.sh" && ! -f "$TESTBED/run_tests.sh"'
        " ]]; then\n"
        '    cp "$harness_dir/run_tests.sh" "$TESTBED/run_tests.sh" 2>/dev/null'
        " || true\n"
        "  fi\n"
        "done"
    )
  else:
    r2e_restore_block = ": # not an R2E-Gym task"
  return f"""#!/usr/bin/env bash
# Generated Harbor/SWE-rebench-v2 verifier for {task.instance_id}.
set -uo pipefail

TEST_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TESTBED="${{HARBOR_TESTBED:-{profile.workdir}}}"
LOG_DIR="${{HARBOR_VERIFIER_LOG_DIR:-/logs/verifier}}"
REWARD_FILE="$LOG_DIR/reward.txt"
RESULTS_FILE="$LOG_DIR/results.json"
TEST_LOG="$LOG_DIR/test_output.log"
mkdir -p "$LOG_DIR"
printf '0.0\\n' > "$REWARD_FILE"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_BUILD_ISOLATION=1

for conda_sh in /opt/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [[ -f "$conda_sh" ]]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate testbed 2>/dev/null || conda activate base 2>/dev/null || true
    break
  fi
done

if [[ ! -d "$TESTBED" ]]; then
  for candidate in /testbed /app /workspace "$(pwd)"; do
    if [[ -d "$candidate/.git" ]]; then
      TESTBED="$candidate"
      break
    fi
  done
  if [[ ! -d "$TESTBED" ]]; then
    for candidate in /*; do
      if [[ -d "$candidate/.git" ]]; then
        TESTBED="$candidate"
        break
      fi
    done
  fi
fi

if [[ -f "$TESTBED/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$TESTBED/.venv/bin/activate"
elif [[ -f /root/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/.venv/bin/activate
fi

cd "$TESTBED" || exit 0
{pre_commands}

if [[ -s "$TEST_DIR/test_patch.diff" ]]; then
{clean_block}
  if ! git apply --verbose "$TEST_DIR/test_patch.diff" 2>&1 | tee "$LOG_DIR/patch_apply.log"; then
    git apply --verbose --reject "$TEST_DIR/test_patch.diff" 2>&1 | tee -a "$LOG_DIR/patch_apply.log" || \\
      patch --batch --fuzz=5 -p1 -i "$TEST_DIR/test_patch.diff" 2>&1 | tee -a "$LOG_DIR/patch_apply.log" || true
  fi
fi
{r2e_restore_block}

TEST_TARGETS=(
{quoted_targets}
)
: > "$TEST_LOG"
TEST_EXIT=0
BATCH_SIZE="${{HARBOR_TEST_BATCH_SIZE:-50}}"
if [[ ${{#TEST_TARGETS[@]}} -gt 0 ]]; then
  for ((offset = 0; offset < ${{#TEST_TARGETS[@]}}; offset += BATCH_SIZE)); do
    BATCH=("${{TEST_TARGETS[@]:offset:BATCH_SIZE}}")
    {test_cmd} "${{BATCH[@]}}" 2>&1 | tee -a "$TEST_LOG"
    BATCH_EXIT=${{PIPESTATUS[0]}}
    if [[ $BATCH_EXIT -ne 0 ]]; then
      TEST_EXIT=$BATCH_EXIT
    fi
  done
else
  {test_cmd} 2>&1 | tee -a "$TEST_LOG"
  TEST_EXIT=${{PIPESTATUS[0]}}
fi

SCORER_PYTHON="${{HARBOR_PYTHON:-}}"
if [[ -z "$SCORER_PYTHON" ]]; then
  for candidate in /usr/bin/python3 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      SCORER_PYTHON="$candidate"
      break
    fi
  done
fi
"$SCORER_PYTHON" "$TEST_DIR/score_tests.py" \\
  --log "$TEST_LOG" \\
  --config "$TEST_DIR/config.json" \\
  --test_exit_code "$TEST_EXIT" \\
  --reward_file "$REWARD_FILE" \\
  --results_file "$RESULTS_FILE" || true
exit 0
"""


def _score_tests_source() -> str:
  score_tests_path = pathlib.Path(score_tests.__file__)
  return score_tests_path.read_text(encoding="utf-8")


def synthesize_tests_files(
    task: RealWorldTask,
    profile: HarborProfile | None = None,
) -> dict[str, str]:
  """Builds the base64-encoded `/tests` directory map for `task`."""
  active_profile = profile or HarborProfile()
  raw_files = {
      "test.sh": render_test_script(task, active_profile).encode("utf-8"),
      "config.json": render_test_config(task, active_profile).encode("utf-8"),
      "score_tests.py": _score_tests_source().encode("utf-8"),
      "test_patch.diff": task.test_patch.encode("utf-8"),
  }
  return {
      rel: base64.b64encode(data).decode("ascii")
      for rel, data in sorted(raw_files.items())
  }


def unpack_tests_files(packed: Mapping[str, str] | str) -> dict[str, bytes]:
  """Decodes a base64 `tests_files` mapping (or JSON string) into raw bytes."""
  if isinstance(packed, str):
    packed = json.loads(packed)
  if not isinstance(packed, Mapping):
    raise TypeError(f"Expected Mapping for tests_files, got {type(packed)}")
  return {
      rel: base64.b64decode(b64.encode("ascii")) for rel, b64 in packed.items()
  }


def _has_nonempty_r2e_diffs(entry: Mapping[str, Any]) -> bool:
  raw = entry.get("parsed_commit_content")
  if not raw:
    return False
  if isinstance(raw, str):
    try:
      parsed = json.loads(raw)
    except json.JSONDecodeError:
      return False
  elif isinstance(raw, Mapping):
    parsed = raw
  else:
    return False
  return bool(parsed.get("file_diffs"))


def is_harbor_entry(entry: Mapping[str, Any] | None) -> bool:
  """Returns True when `entry` should be graded via Harbor `/tests/test.sh`."""
  if not entry or not isinstance(entry, Mapping):
    return False
  if entry.get("task_format") in ("harbor", "harbor_in_place"):
    return True
  if entry.get("source_dataset") == ENDLESS_TERMINALS or entry.get("eval_in_place"):
    return True
  if entry.get("tests_files"):
    return True
  if _has_nonempty_r2e_diffs(entry):
    return False
  for key in (
      "FAIL_TO_PASS",
      "fail_to_pass",
      "PASS_TO_PASS",
      "pass_to_pass",
      "test_patch",
      "install_config",
      "image_name",
  ):
    val = entry.get(key)
    if val is not None and val != "" and val != [] and val != "[]":
      return True
  return False


def is_in_place_harbor_entry(entry: Mapping[str, Any] | None) -> bool:
  """Returns True when Harbor grading should run in-place inside `agent-runtime`.

  Terminal / OS-state benchmarks such as Endless Terminals and Terminal-Bench
  modify non-git files, file permissions, symlinks, SQLite databases, archives,
  or running processes in `/home/user` rather than producing a git repository
  diff against a `base_commit`. Those tasks must be verified directly inside the
  rollout container rather than via `git diff` + `git apply` in the `eval`
  sidecar.
  """
  if not entry or not isinstance(entry, Mapping):
    return False
  if entry.get("eval_in_place") or entry.get("task_format") == "harbor_in_place":
    return True
  if str(entry.get("source_dataset") or "").strip().lower() == ENDLESS_TERMINALS:
    return True
  if is_harbor_entry(entry) and not resolve_base_commit(entry) and bool(entry.get("tests_files")):
    return True
  return False


def render_endless_terminals_test_script(
    test_file: str = "test_final_state.py",
    workdir: str = "/home/user",
) -> str:
  """Renders an offline-first, non-root-safe `test.sh` for Endless Terminals."""
  return f"""#!/usr/bin/env bash
set -uo pipefail

TEST_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_DIR="${{HARBOR_VERIFIER_LOG_DIR:-/logs/verifier}}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null || ! [ -w "$LOG_DIR" ]; then
  LOG_DIR="/tmp/harbor_verifier"
  mkdir -p "$LOG_DIR"
fi
REWARD_FILE="$LOG_DIR/reward.txt"
RESULTS_FILE="$LOG_DIR/results.json"
TEST_LOG="$LOG_DIR/test_output.log"
printf '0.0\\n' > "$REWARD_FILE"

cd {shlex.quote(workdir)} 2>/dev/null || true

TEST_TARGET="$TEST_DIR/{test_file}"
if [[ ! -f "$TEST_TARGET" && -f "/tests/{test_file}" ]]; then
  TEST_TARGET="/tests/{test_file}"
fi

: > "$TEST_LOG"
TEST_EXIT=0
if python3 -c "import pytest" >/dev/null 2>&1; then
  python3 -m pytest "$TEST_TARGET" -v -rA --tb=short -o console_output_style=classic >"$TEST_LOG" 2>&1
  TEST_EXIT=$?
elif command -v pytest >/dev/null 2>&1; then
  pytest "$TEST_TARGET" -v -rA --tb=short -o console_output_style=classic >"$TEST_LOG" 2>&1
  TEST_EXIT=$?
else
  apt-get update -y && apt-get install -y curl
  curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || true
  uvx --python 3.12 --with pytest==8.4.1 pytest "$TEST_TARGET" -v -rA --tb=short >"$TEST_LOG" 2>&1
  TEST_EXIT=$?
fi

cat "$TEST_LOG" || true

python3 - "$TEST_LOG" "$TEST_EXIT" "$REWARD_FILE" "$RESULTS_FILE" <<'PYEOF'
import json, re, sys
log_path, exit_code_str, reward_path, results_path = sys.argv[1:5]
exit_code = int(exit_code_str)
text = open(log_path, "r", encoding="utf-8", errors="replace").read()
passed = [m.group(1) for m in re.finditer(r"^PASSED\\s+(\\S+)", text, flags=re.M)]
failed = [m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR)\\s+(\\S+)", text, flags=re.M)]
total = len(passed) + len(failed)
reward = 1.0 if (exit_code == 0 and len(failed) == 0) else 0.0
open(reward_path, "w", encoding="utf-8").write(f"{{reward}}\\n")
payload = {{
    "reward": reward,
    "test_exit_code": exit_code,
    "fail_to_pass_total": total,
    "fail_to_pass_passed": passed,
    "fail_to_pass_failed": failed,
    "pass_to_pass_total": 0,
    "pass_to_pass_passed": [],
    "pass_to_pass_failed": [],
}}
try:
    open(results_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\\n")
except Exception:
    pass
print(
    f"score_tests: reward={{reward}} "
    f"fail_to_pass={{len(passed)}}/{{total}} "
    f"pass_to_pass_failed=0 "
    f"exit={{exit_code}}"
)
PYEOF
exit 0
"""


def ensure_harbor_entry(
    raw_entry: Mapping[str, Any],
    *,
    source_dataset: str = SWE_REBENCH_V2,
    profile: HarborProfile | None = None,
) -> dict[str, Any]:
  """Normalizes a Harbor / SWE-rebench-v2 record for Tunix `SWEEnv` & `eval_deepswe`.

  If `raw_entry` is a standard R2E-Gym record (not a Harbor / SWE-rebench-v2
  record), returns `dict(raw_entry)` untouched.
  """
  entry = dict(raw_entry)
  if not is_harbor_entry(entry):
    return entry

  active_profile = profile or HarborProfile(
      workdir=str(entry.get("workdir") or "/testbed")
  )
  task = normalize_record(entry, source_dataset=source_dataset)
  docker_image = task.docker_image
  if not docker_image:
    template = DEFAULT_IMAGE_TEMPLATES.get(task.source_dataset, "")
    if template and task.instance_id:
      instance_id_sweb = task.instance_id.lower().replace("__", "_s_")
      docker_image = template.format(
          instance_id=task.instance_id,
          instance_id_lower=task.instance_id.lower(),
          instance_id_sweb=instance_id_sweb,
          repo=task.repo,
          repo_owner=task.repo_owner.lower(),
          repo_name=task.repo_name.lower(),
          base_commit=task.base_commit,
          short_commit=task.base_commit[:12],
          version=task.version,
      )

  tests_files = entry.get("tests_files")
  if isinstance(tests_files, str) and tests_files.strip():
    try:
      tests_files = json.loads(tests_files)
    except json.JSONDecodeError:
      tests_files = None
  if not tests_files:
    tests_files = synthesize_tests_files(task, active_profile)

  synthetic_commit = dict(_SYNTHETIC_PARSED_COMMIT_TEMPLATE)
  if task.base_commit:
    synthetic_commit["old_commit_hash"] = task.base_commit
    synthetic_commit["new_commit_hash"] = task.base_commit

  parsed_commit_content = entry.get("parsed_commit_content")
  if not parsed_commit_content:
    parsed_commit_content = json.dumps(synthetic_commit)

  entry.update({
      "instance_id": task.instance_id or docker_image,
      "docker_image": docker_image,
      "problem_statement": task.problem_statement,
      "prompt": entry.get("prompt") or task.problem_statement,
      "repo": task.repo,
      "repo_name": entry.get("repo_name") or task.repo or task.instance_id,
      "base_commit": task.base_commit,
      "commit_hash": entry.get("commit_hash") or task.base_commit,
      "FAIL_TO_PASS": json.dumps(list(task.fail_to_pass)),
      "PASS_TO_PASS": json.dumps(list(task.pass_to_pass)),
      "expected_output_json": (
          entry.get("expected_output_json") or '{"reward": 1.0}'
      ),
      "parsed_commit_content": parsed_commit_content,
      "execution_result_content": entry.get("execution_result_content") or "{}",
      "modified_files": entry.get("modified_files") or "[]",
      "relevant_files": entry.get("relevant_files") or "[]",
      "modified_entity_summaries": (
          entry.get("modified_entity_summaries") or "[]"
      ),
      "num_non_test_files": int(entry.get("num_non_test_files") or 0),
      "num_non_test_func_methods": int(
          entry.get("num_non_test_func_methods") or 0
      ),
      "num_non_test_lines": int(entry.get("num_non_test_lines") or 0),
      "task_format": "harbor",
      "workdir": str(entry.get("workdir") or active_profile.workdir),
      "tests_files": tests_files,
  })
  return entry


def get_tests_files(entry: Mapping[str, Any]) -> dict[str, bytes]:
  """Extracts or synthesizes the `/tests` file map `{rel_path: bytes}` for `entry`."""
  packed = entry.get("tests_files")
  if not packed:
    normalized = ensure_harbor_entry(entry)
    packed = normalized.get("tests_files") or {}
  return unpack_tests_files(packed)


def pack_files_dir(directory: pathlib.Path) -> dict[str, str]:
  """Recursively packs a directory into `{relative_posix_path: base64_bytes}`."""
  packed: dict[str, str] = {}
  if not directory.is_dir():
    return packed
  for path in sorted(directory.rglob("*")):
    if not path.is_file():
      continue
    rel = path.relative_to(directory).as_posix()
    packed[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
  return packed


def is_harbor_task_dir(path: pathlib.Path) -> bool:
  """Returns True when `path` is a Harbor task directory."""
  return (
      path.is_dir()
      and (path / "task.toml").is_file()
      and (path / "instruction.md").is_file()
      and (path / "tests" / "test.sh").is_file()
  )


def discover_harbor_task_dirs(root: pathlib.Path) -> list[pathlib.Path]:
  """Finds Harbor task directories at `root` or one level below it."""
  if is_harbor_task_dir(root):
    return [root]
  if not root.is_dir():
    return []
  return sorted(child for child in root.iterdir() if is_harbor_task_dir(child))


def _endless_o3_difficulty_tier(num_success: int) -> str:
  if num_success >= 16:
    return "L1_o3_16of16"
  if num_success >= 13:
    return "L2_o3_13to15of16"
  if num_success >= 9:
    return "L3_o3_9to12of16"
  if num_success >= 5:
    return "L4_o3_5to8of16"
  return "L5_o3_1to4of16"


def load_harbor_task_dir(task_dir: str | pathlib.Path) -> dict[str, Any]:
  """Loads a Harbor task directory (`task.toml`, `instruction.md`, `tests/`)."""
  task_dir = pathlib.Path(task_dir)
  config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
  instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
  env_cfg = dict(config.get("environment") or {})
  metadata = dict(config.get("metadata") or {})
  tags = metadata.get("tags") or []
  is_endless = "echos" in tags or task_dir.name.startswith("task_")
  docker_image = str(env_cfg.get("docker_image") or "").strip()
  instance_id = str(metadata.get("instance_id") or task_dir.name)
  if not docker_image and is_endless:
    docker_image = DEFAULT_IMAGE_TEMPLATES[ENDLESS_TERMINALS].format(
        instance_id=instance_id
    )
  if not docker_image:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.is_file():
      for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
          tokens = stripped[5:].strip().split()
          for tok in tokens:
            if not tok.startswith("--"):
              docker_image = tok
              break
          if docker_image:
            break
  base_commit = str(metadata.get("base_commit") or "")
  repo = str(metadata.get("repo") or ("endless_terminals/" + instance_id if is_endless else instance_id))
  default_workdir = "/home/user" if is_endless else "/testbed"
  workdir = str(env_cfg.get("workdir") or default_workdir)
  tests_files = pack_files_dir(task_dir / "tests")
  if is_endless and "test_final_state.py" in tests_files:
    tests_files["test.sh"] = base64.b64encode(
        render_endless_terminals_test_script(workdir=workdir).encode("utf-8")
    ).decode("ascii")
  difficulty = str(metadata.get("difficulty") or "")
  o3_num_success = None
  o3_summary_file = task_dir / "solution" / "o3_summary.json"
  if o3_summary_file.is_file():
    try:
      o3_summary = json.loads(o3_summary_file.read_text(encoding="utf-8"))
      if "num_success" in o3_summary:
        o3_num_success = int(o3_summary["num_success"])
        if not difficulty or difficulty == "easy":
          difficulty = _endless_o3_difficulty_tier(o3_num_success)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  raw_entry = {
      "instance_id": instance_id,
      "docker_image": docker_image,
      "problem_statement": instruction,
      "prompt": instruction,
      "repo": repo,
      "repo_name": repo,
      "base_commit": base_commit,
      "commit_hash": base_commit,
      "workdir": workdir,
      "difficulty": difficulty,
      "difficulty_level": difficulty,
      "o3_num_success": o3_num_success,
      "task_format": "harbor",
      "source_dataset": ENDLESS_TERMINALS if is_endless else SWE_REBENCH_V2,
      "eval_in_place": bool(is_endless or not base_commit),
      "tests_files": tests_files,
  }
  return ensure_harbor_entry(
      raw_entry,
      source_dataset=ENDLESS_TERMINALS if is_endless else SWE_REBENCH_V2,
  )


def _iter_jsonl_lines(
    dataset_path: str, max_rows: int | None = None
) -> list[dict[str, Any]]:
  """Reads JSONL records from a local path or `gs://` URI."""
  rows: list[dict[str, Any]] = []

  def _consume_stream(stream) -> None:
    for raw_line in stream:
      if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace").strip()
      else:
        line = str(raw_line).strip()
      if not line:
        continue
      rows.append(json.loads(line))
      if max_rows is not None and max_rows > 0 and len(rows) >= max_rows:
        break

  if dataset_path.startswith("gs://"):
    opened = False
    try:
      import fsspec  # pylint: disable=g-import-not-at-top

      with fsspec.open(dataset_path, "rt", encoding="utf-8") as f:
        _consume_stream(f)
      opened = True
    except Exception as fsspec_err:  # pylint: disable=broad-exception-caught
      logging.debug(
          "fsspec open failed for %s (%s); trying fallbacks.",
          dataset_path,
          fsspec_err,
      )
    if not opened:
      try:
        from tensorflow.io import gfile  # pylint: disable=g-import-not-at-top

        with gfile.GFile(dataset_path, "r") as f:
          _consume_stream(f)
        opened = True
      except Exception as gfile_err:  # pylint: disable=broad-exception-caught
        logging.debug(
            "gfile open failed for %s (%s); trying gsutil.",
            dataset_path,
            gfile_err,
        )
    if not opened:
      proc = subprocess.Popen(
          ["gsutil", "cat", dataset_path],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )
      try:
        assert proc.stdout is not None
        _consume_stream(proc.stdout)
      finally:
        proc.terminate()
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()
      if not rows and proc.returncode not in (0, None, -15):
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(
            f"Failed to read {dataset_path} via gsutil: {stderr.strip()}"
        )
    return rows

  path = pathlib.Path(dataset_path)
  if path.suffix.lower() == ".json":
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
      items = data[:max_rows] if (max_rows and max_rows > 0) else data
      return [dict(item) for item in items]
    raise ValueError(f"Expected JSON list in {dataset_path}")

  with path.open("r", encoding="utf-8") as f:
    _consume_stream(f)
  return rows


def is_jsonl_or_harbor_path(dataset_path: str) -> bool:
  """Returns True when `dataset_path` is a JSONL/JSON file or Harbor directory."""
  if not dataset_path:
    return False
  lower = dataset_path.rstrip("/").lower()
  if lower.endswith(".jsonl") or lower.endswith(".json"):
    return True
  if not dataset_path.startswith("gs://"):
    path = pathlib.Path(dataset_path)
    if discover_harbor_task_dirs(path):
      return True
  return False


def load_dataset_records(
    dataset_path: str,
    *,
    max_rows: int | None = None,
    source_dataset: str = SWE_REBENCH_V2,
) -> list[dict[str, Any]]:
  """Loads and normalizes records from a JSONL/JSON file or Harbor directory."""
  if not dataset_path.startswith("gs://"):
    path = pathlib.Path(dataset_path)
    task_dirs = discover_harbor_task_dirs(path)
    if task_dirs:
      if max_rows is not None and max_rows > 0:
        task_dirs = task_dirs[:max_rows]
      return [load_harbor_task_dir(td) for td in task_dirs]

  raw_rows = _iter_jsonl_lines(dataset_path, max_rows=max_rows)
  return [
      ensure_harbor_entry(row, source_dataset=source_dataset)
      for row in raw_rows
  ]


def build_tests_tar_gz_b64(tests_files: Mapping[str, bytes]) -> str:
  """Packs `tests_files` into a base64-encoded `.tar.gz` archive."""
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for rel_path, content in sorted(tests_files.items()):
      clean = posixpath.normpath(rel_path).lstrip("/")
      if clean.startswith("..") or "/../" in clean:
        raise ValueError(f"Refusing path traversal in tests_files: {rel_path!r}")
      info = tarfile.TarInfo(name=clean)
      info.size = len(content)
      info.mode = 0o755 if clean.endswith(".sh") else 0o644
      tar.addfile(info, io.BytesIO(content))
  return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_reward_output(
    reward_txt: str,
    reward_json: str = "",
) -> float:
  """Parses the verifier's `/logs/verifier/reward.txt` or `reward.json`."""
  text = (reward_txt or "").strip()
  if text:
    first_line = text.splitlines()[0].strip()
    try:
      value = float(first_line)
      if math.isfinite(value):
        return value
    except ValueError:
      pass
  json_text = (reward_json or "").strip()
  if json_text:
    try:
      payload = json.loads(json_text)
    except json.JSONDecodeError:
      payload = None
    if isinstance(payload, Mapping) and "reward" in payload:
      try:
        value = float(payload["reward"])
        if math.isfinite(value):
          return value
      except (TypeError, ValueError):
        pass
  return 0.0


def _unpack_exec_output(res: Any) -> tuple[str, int]:
  """Normalizes `(stdout, exit_code)` from runtime.run or workspace.execute_command."""
  if res is None:
    return "", -1
  if isinstance(res, tuple) and len(res) == 2:
    out, code = res
    code_str = str(code).strip()
    return str(out or ""), (0 if code_str == "0" else -1)
  if isinstance(res, str):
    return res, 0
  exit_code = getattr(res, "exit_code", 0)
  try:
    exit_code_int = int(exit_code)
  except (TypeError, ValueError):
    exit_code_int = 0 if str(exit_code).strip() == "0" else -1
  stdout = getattr(res, "stdout", None)
  if stdout is None:
    stdout = ""
  return str(stdout), exit_code_int


def _extract_stdout(res: Any) -> str:
  """Extracts stdout string from `(output, exit_code)`, command result, or `str`."""
  stdout, _ = _unpack_exec_output(res)
  return stdout


def _runtime_run_sh(runtime: Any, cmd: str, timeout: float = 60.0) -> Any:
  """Executes `cmd` via `runtime.run`, wrapping compound shell commands for k8s."""
  timeout_int = int(max(1, round(timeout)))
  backend = getattr(runtime, "backend", None) or getattr(
      runtime, "_actual_backend", None
  )
  if backend in ("kubernetes", "kubernetes-sandbox"):
    stripped = cmd.strip()
    if not stripped.startswith(
        ("/bin/sh -c ", "/bin/bash -c ", "sh -c ", "bash -c ")
    ):
      cmd = f"/bin/sh -c {shlex.quote(cmd)}"
    try:
      return runtime.run(cmd, timeout=timeout_int, workdir="")
    except TypeError:
      pass
  return runtime.run(cmd, timeout=timeout_int)


_STRIP_PYINSTALLER_LD_PATH = (
    'while :; do case "${LD_LIBRARY_PATH-}" in'
    " /tmp/_MEI*:*|/oh/glibc236:*|:*) LD_LIBRARY_PATH=${LD_LIBRARY_PATH#*:};;"
    " /tmp/_MEI*|/oh/glibc236|'') unset LD_LIBRARY_PATH; break;;"
    " *:) LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:};;"
    " *) export LD_LIBRARY_PATH; break;;"
    " esac; done; "
)


def exec_in_target(target: Any, cmd: str, timeout: float = 60.0) -> tuple[str, int]:
  """Executes `cmd` inside `target` (RepoEnv, DockerRuntime, or Workspace)."""
  if target is None:
    return "", -1
  if hasattr(target, "execute_command"):
    res = target.execute_command(
        _STRIP_PYINSTALLER_LD_PATH + cmd, timeout=float(timeout)
    )
    return _unpack_exec_output(res)
  runtime = getattr(target, "runtime", None) or target
  if hasattr(runtime, "run"):
    res = _runtime_run_sh(runtime, cmd, timeout=timeout)
    return _unpack_exec_output(res)
  return "", -1


def resolve_base_commit(entry: Mapping[str, Any] | None) -> str:
  """Resolves `base_commit` or `commit_hash` from a Harbor task entry."""
  if not entry:
    return ""
  for key in ("base_commit", "commit_hash", "commit"):
    val = entry.get(key)
    if val and str(val).strip():
      return str(val).strip()
  return ""


_ENSURE_TESTBED_SYMLINK_CMD = (
    "if [ ! -d /testbed/.git ]; then "
    "if [ -L /testbed ] || [ ! -e /testbed ]; then rm -f /testbed 2>/dev/null || true; fi; "
    'if [ -d "$(pwd)/.git" ] && [ "$(pwd)" != "/" ] && [ "$(pwd)" != "/testbed" ]; then ln -s "$(pwd)" /testbed; '
    "else for _d in /app /workspace /*; do "
    'if [ "$_d" != "/testbed" ] && [ -d "$_d/.git" ]; then ln -s "$_d" /testbed; break; fi; '
    "done; fi; fi; "
    "git config --global --add safe.directory '*' 2>/dev/null || true"
)


def strip_harbor_verifier_paths(
    target: Any,
    entry: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
) -> None:
  """Ensures `/testbed` points to the repo and strips verifier paths in rollout container."""
  try:
    cmd = (
        f"{_ENSURE_TESTBED_SYMLINK_CMD}; "
        f"rm -rf {CONTAINER_TESTS_DIR} /solution {VERIFIER_LOG_DIR}"
        " /tmp/harbor_tests /tmp/harbor_verifier"
        " /r2e_tests /testbed/r2e_tests /root/run_tests.sh /testbed/run_tests.sh"
        " 2>/dev/null || true"
    )
    exec_in_target(target, cmd, timeout=timeout)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[HarborUtils] strip_harbor_verifier_paths note: %s", e)
  del entry


def configure_harbor_workspace_cwd(
    target: Any,
    entry: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> None:
  """Initializes OpenHands persistent shell working directory (`workdir`) when not `/testbed`."""
  if target is None or not entry or not isinstance(entry, Mapping):
    return
  workdir = str(entry.get("workdir") or "").strip()
  if not workdir or workdir == "/testbed":
    return
  try:
    cmd = (
        f"mkdir -p /var/tmp 2>/dev/null || true; "
        f"if [ -d {shlex.quote(workdir)} ]; then "
        f"printf '%s\\n' {shlex.quote(workdir)} > /var/tmp/.oh_cwd 2>/dev/null || true; "
        "fi"
    )
    exec_in_target(target, cmd, timeout=timeout)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[HarborUtils] configure_harbor_workspace_cwd note: %s", e)


def run_harbor_verifier_in_target(
    target: Any,
    entry: Mapping[str, Any],
    run_cmd_fn: Callable[..., Any] | None = None,
    *,
    timeout: float = 300.0,
) -> float:
  """Uploads `/tests` into `target`, runs `bash /tests/test.sh`, and reads reward.

  Args:
    target: The evaluation environment (`RepoEnv`, runtime, or `Workspace`).
    entry: The task dictionary containing `tests_files` (or raw SWE-rebench-v2
      fields to synthesize `tests_files` from).
    run_cmd_fn: Optional callable `(target, cmd, timeout=...) -> Any`. Defaults
      to `exec_in_target`.
    timeout: Timeout in seconds for running `/tests/test.sh`.

  Returns:
    Float reward in `[0.0, 1.0]` (`0.0` on timeout or verifier error).
  """
  runner = run_cmd_fn or exec_in_target
  try:
    tests_files = get_tests_files(entry)
    b64_tar_gz = build_tests_tar_gz_b64(tests_files)

    # Keep each command well below Linux MAX_ARG_STRLEN (128 KiB).
    chunk_size = 49152
    chunks = [
        b64_tar_gz[i : i + chunk_size]
        for i in range(0, len(b64_tar_gz), chunk_size)
    ]

    select_dirs_snippet = (
        f"if (rm -rf {CONTAINER_TESTS_DIR} {VERIFIER_LOG_DIR} 2>/dev/null && "
        f"mkdir -p {CONTAINER_TESTS_DIR} {VERIFIER_LOG_DIR} 2>/dev/null && "
        f"[ -w {CONTAINER_TESTS_DIR} ] && [ -w {VERIFIER_LOG_DIR} ]); then "
        f"_TDIR={CONTAINER_TESTS_DIR}; _LDIR={VERIFIER_LOG_DIR}; "
        "else "
        "_TDIR=/tmp/harbor_tests; _LDIR=/tmp/harbor_verifier; "
        "rm -rf \"$_TDIR\" \"$_LDIR\" 2>/dev/null || true; "
        "mkdir -p \"$_TDIR\" \"$_LDIR\"; "
        "fi && chmod 0700 \"$_LDIR\""
    )

    if len(chunks) <= 1:
      payload = chunks[0] if chunks else ""
      setup_cmd = (
          f"{select_dirs_snippet} && "
          f"printf '%s' '{payload}' | base64 -d | tar -xzf - -C \"$_TDIR\" && "
          "chmod +x \"$_TDIR/test.sh\""
      )
      runner(target, setup_cmd, timeout=min(60.0, timeout))
    else:
      runner(
          target,
          f"{select_dirs_snippet} && rm -f /tmp/harbor_tests.tgz.b64",
          timeout=30.0,
      )
      for chunk in chunks:
        runner(
            target,
            f"printf '%s' '{chunk}' >> /tmp/harbor_tests.tgz.b64",
            timeout=30.0,
        )
      runner(
          target,
          (
              f"if [ -d {CONTAINER_TESTS_DIR} ] && [ -w {CONTAINER_TESTS_DIR} ]; then _TDIR={CONTAINER_TESTS_DIR}; "
              "else _TDIR=/tmp/harbor_tests; fi && "
              "base64 -d /tmp/harbor_tests.tgz.b64 | tar -xzf - -C \"$_TDIR\" && "
              "rm -f /tmp/harbor_tests.tgz.b64 && "
              "chmod +x \"$_TDIR/test.sh\""
          ),
          timeout=60.0,
      )

    workdir = str(entry.get("workdir") or "/testbed")
    verifier_cmd = (
        f"if [ -f {CONTAINER_TEST_SH} ]; then _TDIR={CONTAINER_TESTS_DIR}; _LDIR={VERIFIER_LOG_DIR}; "
        "else _TDIR=/tmp/harbor_tests; _LDIR=/tmp/harbor_verifier; fi; "
        f"HARBOR_TESTBED={shlex.quote(workdir)} "
        "HARBOR_VERIFIER_LOG_DIR=\"$_LDIR\" "
        "bash \"$_TDIR/test.sh\""
    )
    verifier_out = _extract_stdout(
        runner(target, verifier_cmd, timeout=float(timeout))
    )
    if verifier_out:
      for line in verifier_out.splitlines():
        if "score_tests:" in line:
          logging.info("[HarborUtils] [%s] %s", entry.get("instance_id"), line)

    reward_txt = _extract_stdout(
        runner(
            target,
            f"cat {REWARD_TXT_PATH} /tmp/harbor_verifier/reward.txt 2>/dev/null | head -n 1 || true",
            timeout=30.0,
        )
    )
    reward_json = ""
    if not reward_txt.strip():
      reward_json = _extract_stdout(
          runner(
              target,
              f"cat {REWARD_JSON_PATH} /tmp/harbor_verifier/reward.json 2>/dev/null || true",
              timeout=30.0,
          )
      )
    reward = parse_reward_output(reward_txt, reward_json)
    logging.info(
        "[HarborUtils] [%s] computed reward=%s",
        entry.get("instance_id"),
        reward,
    )
    return float(reward)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "[HarborUtils] [%s] verifier execution failed: %s",
        entry.get("instance_id"),
        e,
    )
    return 0.0


def evaluate_harbor_patch_in_fresh_container(
    eval_env: Any,
    patch: str,
    entry: Mapping[str, Any],
    *,
    reward_timeout: float = 300.0,
) -> float:
  """Applies `patch` in the fresh `eval` container and runs Harbor `/tests/test.sh`."""
  if not patch or not patch.strip():
    logging.info("[HarborUtils] Empty git patch extracted; returning reward 0.0.")
    return 0.0
  if eval_env is None:
    return 0.0

  from examples.deepswe import sandbox_utils  # pylint: disable=g-import-not-at-top

  original_container = sandbox_utils.RUNTIME_CONTAINER_NAME
  runtime_obj = getattr(eval_env, "runtime", None) or eval_env
  if runtime_obj is not None:
    original_container = (
        getattr(runtime_obj, "_container", None)
        or getattr(runtime_obj, "_target_container", None)
        or sandbox_utils.RUNTIME_CONTAINER_NAME
    )

  sandbox_utils.set_runtime_container(
      eval_env, sandbox_utils.EVAL_CONTAINER_NAME
  )
  try:
    runtime = getattr(eval_env, "runtime", None)
    if runtime is None or not hasattr(runtime, "run"):
      return run_harbor_verifier_in_target(
          eval_env, entry, timeout=reward_timeout
      )

    repo_path = str(
        entry.get("workdir") or getattr(runtime, "repo_path", "/testbed") or "/testbed"
    )
    git_ls_cmd = (
        f"{_ENSURE_TESTBED_SYMLINK_CMD}; "
        f"cd {shlex.quote(repo_path)} && "
        "git ls-files --others --exclude-standard"
    )
    git_ls_output, git_ls_exit = _unpack_exec_output(
        _runtime_run_sh(runtime, git_ls_cmd, timeout=30)
    )
    if git_ls_exit != 0:
      logging.warning(
          "[HarborUtils] Failed to list untracked files in eval container: %s",
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
    git_apply_step = (
        f"(git apply --whitespace=nowarn {exclude_str} {patch_path} || "
        f"git apply --whitespace=fix {exclude_str} {patch_path} || "
        f"git apply --ignore-space-change --ignore-whitespace {exclude_str} {patch_path} || "
        "patch --batch --fuzz=5 -p1 -i " + patch_path + ")"
    )
    if len(b64_patch) <= 65536:
      inner_apply = (
          f"printf '%s' {shlex.quote(b64_patch)} | base64 -d > {patch_path} && "
          f"cd {shlex.quote(repo_path)} && "
          f"{git_apply_step}; "
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
        chunk_out, chunk_exit = _unpack_exec_output(
            _runtime_run_sh(
                runtime, f"/bin/sh -c {shlex.quote(inner_chunk)}", timeout=30
            )
        )
        if chunk_exit != 0:
          logging.warning(
              "[HarborUtils] Failed to write patch chunk in eval container: %s",
              chunk_out,
          )
          _runtime_run_sh(runtime, f"rm -f {shlex.quote(b64_tmp_path)}", timeout=15)
          return 0.0
      inner_apply = (
          f"base64 -d {b64_tmp_path} > {patch_path} && "
          f"rm -f {b64_tmp_path} && "
          f"cd {shlex.quote(repo_path)} && "
          f"{git_apply_step}; "
          f"_rc=$?; rm -f {b64_tmp_path} {patch_path}; exit $_rc"
      )
      apply_cmd = f"/bin/sh -c {shlex.quote(inner_apply)}"

    apply_output, apply_exit = _unpack_exec_output(
        _runtime_run_sh(runtime, apply_cmd, timeout=60)
    )
    if apply_exit != 0:
      logging.warning(
          "[HarborUtils] Failed to apply patch in eval container: %s",
          apply_output,
      )
      return 0.0

    reward = run_harbor_verifier_in_target(
        eval_env,
        entry,
        exec_in_target,
        timeout=float(reward_timeout),
    )
    logging.info(
        "[HarborUtils] Evaluated agent patch in fresh eval container:"
        " reward=%.1f (patch_bytes=%d)",
        reward,
        len(patch),
    )
    return reward
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "[HarborUtils] Exception during fresh container evaluation: %s", e
    )
    return 0.0
  finally:
    sandbox_utils.set_runtime_container(eval_env, original_container)
