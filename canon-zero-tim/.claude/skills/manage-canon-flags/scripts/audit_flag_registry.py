#!/usr/bin/env python3
"""Audit the FLAGS.md inventory and CANON_* names added by a Git diff."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys


FLAG_RE = re.compile(r"\bCANON_[A-Z0-9_]+\b")
COUNT_RE = re.compile(r"Count:\s*(\d+)\s+settable names")


def _package_root() -> Path:
  return Path(__file__).resolve().parents[4]


def _inventory(flags_path: Path) -> tuple[list[str], int]:
  text = flags_path.read_text()
  try:
    appendix = text.split("## Appendix", 1)[1]
    block = appendix.split("```", 2)[1]
  except IndexError as exc:
    raise ValueError("FLAGS.md appendix code block is missing") from exc
  names = [line.strip() for line in block.splitlines() if FLAG_RE.fullmatch(line.strip())]
  match = COUNT_RE.search(appendix)
  if match is None:
    raise ValueError("FLAGS.md declared count is missing")
  return names, int(match.group(1))


def _added_flags(repo: Path, base: str) -> set[str]:
  completed = subprocess.run(
      ["git", "diff", "--unified=0", base, "--"],
      cwd=repo,
      check=True,
      text=True,
      capture_output=True,
  )
  added = (
      line[1:]
      for line in completed.stdout.splitlines()
      if line.startswith("+") and not line.startswith("+++")
  )
  return {name for line in added for name in FLAG_RE.findall(line)}


def _prefix(name: str) -> str:
  match = re.match(r"CANON_(P\d+)", name)
  return match.group(1) if match else "BASE"


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--repo",
      type=Path,
      default=_package_root().parent,
      help="Git worktree root (default: inferred from this skill)",
  )
  parser.add_argument(
      "--changed-base",
      help="Optional Git revision whose added CANON_* names must be registered",
  )
  args = parser.parse_args()

  repo = args.repo.resolve()
  flags_path = repo / "canon-zero-tim/FLAGS.md"
  if not flags_path.is_file():
    print(f"FLAG_AUDIT_FAIL missing={flags_path}", file=sys.stderr)
    return 2

  try:
    names, declared = _inventory(flags_path)
  except ValueError as exc:
    print(f"FLAG_AUDIT_FAIL reason={exc}", file=sys.stderr)
    return 2

  duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
  failures = []
  if declared != len(names):
    failures.append(f"declared={declared} actual={len(names)}")
  if duplicates:
    failures.append("duplicates=" + ",".join(duplicates))

  changed = set()
  unregistered = []
  if args.changed_base:
    try:
      changed = _added_flags(repo, args.changed_base)
    except subprocess.CalledProcessError as exc:
      print(exc.stderr, file=sys.stderr, end="")
      print("FLAG_AUDIT_FAIL reason=git-diff", file=sys.stderr)
      return 2
    unregistered = sorted(changed - set(names))
    if unregistered:
      failures.append("unregistered=" + ",".join(unregistered))

  inversions = sum(names[index] > names[index + 1] for index in range(len(names) - 1))
  prefixes = Counter(_prefix(name) for name in names)
  prefix_summary = ",".join(
      f"{name}:{count}" for name, count in sorted(prefixes.items())
  )
  print(
      "FLAG_AUDIT "
      f"declared={declared} actual={len(names)} unique={len(set(names))} "
      f"ordering_inversions={inversions} changed_names={len(changed)} "
      f"prefixes={prefix_summary}"
  )
  if failures:
    print("FLAG_AUDIT_FAIL " + " ".join(failures), file=sys.stderr)
    return 1
  print("FLAG_AUDIT_PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
