#!/usr/bin/env python3
"""Print mechanical facts for the canon-zero-tim delivery commit stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ORIGINAL_COMMITS = (
    "7101b4a5",
    "53c0448b",
    "7748dbeb",
    "53198034",
    "370d00c3",
    "3f037d8d",
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def inspect_commit(repo: Path, revision: str, package_prefix: str) -> dict[str, object]:
    full_sha = git(repo, "rev-parse", f"{revision}^{{commit}}").strip()
    subject = git(repo, "show", "-s", "--format=%s", full_sha).rstrip("\n")
    body = git(repo, "show", "-s", "--format=%b", full_sha).rstrip("\n")
    numstat = git(repo, "show", "--format=", "--numstat", "--no-renames", full_sha)

    files: list[dict[str, object]] = []
    additions = 0
    deletions = 0
    binary_files = 0
    outside_package: list[str] = []

    for line in numstat.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected numstat line for {full_sha}: {line!r}")
        added_text, deleted_text, path = fields
        is_binary = added_text == "-" or deleted_text == "-"
        if is_binary:
            binary_files += 1
            added = None
            deleted = None
        else:
            added = int(added_text)
            deleted = int(deleted_text)
            additions += added
            deletions += deleted
        if not path.startswith(package_prefix):
            outside_package.append(path)
        files.append(
            {
                "path": path,
                "additions": added,
                "deletions": deleted,
                "binary": is_binary,
            }
        )

    body_lines = [line for line in body.splitlines() if line.strip()]
    lowered = body.lower()
    return {
        "revision": revision,
        "sha": full_sha,
        "subject": subject,
        "file_count": len(files),
        "additions": additions,
        "deletions": deletions,
        "binary_files": binary_files,
        "outside_package": outside_package,
        "message_body_nonblank_lines": len(body_lines),
        "message_mentions_downside": "downside" in lowered or "drawback" in lowered,
        "message_mentions_background": "background" in lowered or "context" in lowered,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect read-only Git facts for the canon-zero-tim commit stack. "
            "The output is not a validation verdict."
        )
    )
    parser.add_argument("--repo", required=True, help="Path to the Git repository.")
    parser.add_argument(
        "--commits",
        nargs="+",
        default=list(ORIGINAL_COMMITS),
        help="Commit revisions to inspect, in delivery order.",
    )
    parser.add_argument(
        "--package-prefix",
        default="canon-zero-tim/",
        help="Expected path prefix for every changed file.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()

    try:
        inside = git(repo, "rev-parse", "--is-inside-work-tree").strip()
        if inside != "true":
            raise RuntimeError(f"not a Git worktree: {repo}")
        rows = [
            inspect_commit(repo, revision, args.package_prefix)
            for revision in args.commits
        ]
        branch = git(repo, "branch", "--show-current").strip()
        head = git(repo, "rev-parse", "HEAD").strip()
        dirty = bool(git(repo, "status", "--short").strip())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = {
        "repo": str(repo),
        "branch": branch,
        "head": head,
        "dirty": dirty,
        "package_prefix": args.package_prefix,
        "commit_count": len(rows),
        "total_files_per_commit": sum(int(row["file_count"]) for row in rows),
        "total_additions": sum(int(row["additions"]) for row in rows),
        "total_deletions": sum(int(row["deletions"]) for row in rows),
        "commits_with_outside_paths": sum(bool(row["outside_package"]) for row in rows),
        "commits": rows,
        "interpretation": (
            "Mechanical inventory only. Additive scope and message structure do not prove "
            "CPU, TPU, Pathways, GKE, round-trip, or training validation."
        ),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Repository: {summary['repo']}")
    print(f"Branch: {branch or '(detached)'}")
    print(f"HEAD: {head}")
    print(f"Dirty: {dirty}")
    print()
    for row in rows:
        print(
            f"{str(row['sha'])[:10]}  files={row['file_count']:>2}  "
            f"+{row['additions']:<5} -{row['deletions']:<5}  {row['subject']}"
        )
        print(
            f"  body_lines={row['message_body_nonblank_lines']} "
            f"downside={row['message_mentions_downside']} "
            f"background={row['message_mentions_background']} "
            f"outside_package={len(row['outside_package'])}"
        )
    print()
    print(
        f"Stack totals: commits={summary['commit_count']} "
        f"files-per-commit={summary['total_files_per_commit']} "
        f"+{summary['total_additions']} -{summary['total_deletions']} "
        f"outside-package-commits={summary['commits_with_outside_paths']}"
    )
    print(f"DATA ONLY: {summary['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
