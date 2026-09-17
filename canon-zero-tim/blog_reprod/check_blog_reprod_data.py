#!/usr/bin/env python3
"""Verify that every vendored data file still matches the hashes in data/manifest.json.

Checks, for each of the three arms:
  data/<arm>.csv                    sha256 == runs.<arm>.plotted_csv_sha256
  runs/<run_id>/history.csv         sha256 == runs.<arm>.source_history_sha256
  runs/<run_id>/config.yaml         sha256 == runs.<arm>.source_config_sha256

Exit status 0 when all nine hashes match, 1 otherwise. Standard library only.

Run from this directory:
    python3 check_blog_reprod_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("standard", "importance_sampling", "zero_tim")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    runs = manifest["runs"]
    if tuple(runs) != ARMS:
        print(f"FAIL: manifest arms {tuple(runs)} != {ARMS}")
        return 1
    failures = 0
    for arm in ARMS:
        entry = runs[arm]
        run_dir = ROOT / "runs" / entry["run_id"]
        checks = (
            (ROOT / "data" / entry["plotted_csv"], entry["plotted_csv_sha256"]),
            (run_dir / "history.csv", entry["source_history_sha256"]),
            (run_dir / "config.yaml", entry["source_config_sha256"]),
        )
        for path, expected in checks:
            relative = path.relative_to(ROOT)
            if not path.is_file():
                print(f"FAIL {arm}: missing {relative}")
                failures += 1
                continue
            actual = digest(path)
            if actual != expected:
                print(f"FAIL {arm}: {relative}\n  expected {expected}\n  actual   {actual}")
                failures += 1
            else:
                print(f"OK   {arm}: {relative} sha256={actual}")
    if failures:
        print(f"FAIL: {failures} hash mismatch(es)")
        return 1
    print(f"PASS: 9 sha256 hashes match data/manifest.json (source_commit {manifest['source_commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
