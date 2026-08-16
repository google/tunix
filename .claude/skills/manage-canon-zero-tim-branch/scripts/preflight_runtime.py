#!/usr/bin/env python3
"""Fail-closed, secret-safe preflight for a canon-zero-tim worktree."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


REQUIRED_PATHS = (
    "canon-zero-tim/START_HERE.md",
    "canon-zero-tim/EVIDENCE.md",
    "canon-zero-tim/RUNBOOK.md",
    "canon-zero-tim/KNOWN_FOOTGUNS.md",
    "canon-zero-tim/cluster/entrypoint.sh",
)
LIVE_CONFIG_ROOTS = (
    "canon-zero-tim/cluster",
    "canon-zero-tim/profiles",
    "examples",
    "tunix",
)
LIVE_CONFIG_SUFFIXES = {".env", ".py", ".sh", ".yaml", ".yml"}
INVALID_RUNTIME_TOKENS = ("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHING",)
SECRET_IN_URL = re.compile(
    r"(?:https?://[^/@\s]+@|ghp_|github_pat_|wandb_v1_)", re.IGNORECASE
)


def git(repo: Path, *args: str) -> str:
  result = subprocess.run(
      ("git", "-C", str(repo), *args),
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  return result.stdout.strip()


def count_invalid_runtime_config(repo: Path) -> int:
  count = 0
  for relative_root in LIVE_CONFIG_ROOTS:
    root = repo / relative_root
    if not root.is_dir():
      continue
    for path in root.rglob("*"):
      if not path.is_file() or path.suffix not in LIVE_CONFIG_SUFFIXES:
        continue
      try:
        text = path.read_text(encoding="utf-8")
      except UnicodeDecodeError:
        continue
      if any(token in text for token in INVALID_RUNTIME_TOKENS):
        count += 1
  return count


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", required=True, type=Path)
  parser.add_argument("--expected-branch")
  parser.add_argument("--require-clean", action="store_true")
  args = parser.parse_args()

  repo = args.repo.resolve()
  failures: list[str] = []
  try:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain")
    remote = git(repo, "remote", "get-url", "origin")
  except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    print(f"CANON_PREFLIGHT FAIL git_error={type(exc).__name__}")
    return 1

  if branch in ("main", "master", ""):
    failures.append("protected_or_detached_branch")
  if args.expected_branch and branch != args.expected_branch:
    failures.append("unexpected_branch")
  if args.require_clean and status:
    failures.append("dirty_worktree")
  remote_has_credentials = bool(SECRET_IN_URL.search(remote))
  if remote_has_credentials:
    failures.append("credential_bearing_remote")
  missing = [path for path in REQUIRED_PATHS if not (repo / path).is_file()]
  if missing:
    failures.append(f"missing_package_paths:{len(missing)}")
  invalid_runtime_config = count_invalid_runtime_config(repo)
  if invalid_runtime_config:
    failures.append(f"invalid_runtime_config:{invalid_runtime_config}")

  dirty_count = len(status.splitlines()) if status else 0
  print(
      "CANON_PREFLIGHT "
      f"branch={branch or 'detached'} head={head} dirty={dirty_count} "
      f"remote_credentials={int(remote_has_credentials)} "
      f"required_paths={len(REQUIRED_PATHS) - len(missing)}/{len(REQUIRED_PATHS)} "
      f"invalid_runtime_config={invalid_runtime_config}"
  )
  if failures:
    print("CANON_PREFLIGHT FAIL reasons=" + ",".join(failures))
    return 1
  print("CANON_PREFLIGHT PASS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
