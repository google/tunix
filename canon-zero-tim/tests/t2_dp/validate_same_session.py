#!/usr/bin/env python3
"""Validate persisted P1b and T2 markers without creating a JAX client."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence


T2_MARKERS = ("CONFIG", "MESH", "CHECKS", "OBSERVATIONS", "UPDATE", "DECISION", "VERDICT")


def validate_lines(lines: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for marker in T2_MARKERS:
        prefix = f"[P32.DP] {marker}"
        count = sum(line.startswith(prefix) for line in lines)
        if count != 1:
            reasons.append(f"expected one {marker} line, got {count}")

    canonical_passes = sum(
        line == "[canonical-op] VERDICT: PASS" for line in lines
    )
    if canonical_passes != 1:
        reasons.append(
            f"expected one canonical-op PASS line, got {canonical_passes}"
        )
    t2_passes = sum(line == "[P32.DP] VERDICT PASS" for line in lines)
    if t2_passes != 1:
        reasons.append(f"expected one T2 PASS line, got {t2_passes}")
    return not reasons, tuple(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.log.is_file():
        print(f"[dp-gate] missing same-session artifact: {args.log}", file=sys.stderr)
        return 2
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    passed, reasons = validate_lines(lines)
    if not passed:
        for reason in reasons:
            print(f"[dp-gate] FAIL: {reason}", file=sys.stderr)
        return 1
    print(f"[dp-gate] SAME_SESSION PASS artifact={args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
