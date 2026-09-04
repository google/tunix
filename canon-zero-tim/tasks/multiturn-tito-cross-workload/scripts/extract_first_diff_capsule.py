#!/usr/bin/env python3
"""Extract and verify one P57 token first-diff capsule from a raw log."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


_REPO = Path(__file__).resolve().parents[4]
_TOKEN_PATH = _REPO / "tunix/rl/agentic/token_continuity.py"
_TOKEN_SPEC = importlib.util.spec_from_file_location(
    "p57_token_continuity_extractor", _TOKEN_PATH
)
if _TOKEN_SPEC is None or _TOKEN_SPEC.loader is None:
  raise RuntimeError(f"cannot load token-continuity module: {_TOKEN_PATH}")
token_continuity = importlib.util.module_from_spec(_TOKEN_SPEC)
sys.modules[_TOKEN_SPEC.name] = token_continuity
_TOKEN_SPEC.loader.exec_module(token_continuity)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--capsule-id")
  args = parser.parse_args()
  if not args.log.is_file():
    raise SystemExit(f"raw log does not exist: {args.log}")
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite capsule: {args.output}")
  capsule = token_continuity.debug_capsule_from_receipts(
      args.log.read_text(encoding="utf-8", errors="strict").splitlines(),
      capsule_id=args.capsule_id,
  )
  payload = (
      json.dumps(capsule, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(
      args.output,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL,
      0o600,
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
  except BaseException:
    try:
      args.output.unlink()
    except FileNotFoundError:
      pass
    raise
  digest = hashlib.sha256(payload).hexdigest()
  header = capsule["header"]
  print(
      "P57_TOKEN_FIRST_DIFF_EXTRACT_PASS "
      f"capsule_id={header['capsule_id']} workload={header['workload']} "
      f"turn={header['turn']} first_mismatch={header['first_mismatch']} "
      f"bytes={len(payload)} sha256={digest} output={args.output}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
