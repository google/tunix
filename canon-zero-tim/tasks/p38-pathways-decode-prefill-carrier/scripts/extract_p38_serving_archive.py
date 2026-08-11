#!/usr/bin/env python3
"""Recover and verify one bounded P38 serving archive from a raw pod log."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
from pathlib import Path
import re


_HEADER = re.compile(
    r"^\[CANON_P38_SERVING_ARCHIVE\] path=\S+ bytes=(\d+) "
    r"sha256=([0-9a-f]{64}) encoding=base64$"
)
_PAYLOAD = "[CANON_P38_SERVING_ARCHIVE_B64] "


def extract(log: Path, output: Path) -> dict[str, object]:
  if output.exists():
    raise RuntimeError(f"refusing to overwrite output: {output}")
  headers: list[tuple[int, str]] = []
  chunks: list[str] = []
  for raw_line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    match = _HEADER.fullmatch(raw_line)
    if match:
      headers.append((int(match.group(1)), match.group(2)))
    elif raw_line.startswith(_PAYLOAD):
      chunks.append(raw_line[len(_PAYLOAD):])
  if len(headers) != 1:
    raise RuntimeError(f"expected exactly one serving-archive header, found {len(headers)}")
  if not chunks:
    raise RuntimeError("serving-archive payload is missing")
  try:
    payload = base64.b64decode("".join(chunks), validate=True)
  except binascii.Error as error:
    raise RuntimeError("serving-archive payload is invalid base64") from error
  expected_bytes, expected_sha = headers[0]
  if len(payload) != expected_bytes:
    raise RuntimeError(
        f"serving-archive byte count mismatch: {len(payload)} != {expected_bytes}"
    )
  actual_sha = hashlib.sha256(payload).hexdigest()
  if actual_sha != expected_sha:
    raise RuntimeError(
        f"serving-archive SHA mismatch: {actual_sha} != {expected_sha}"
    )
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb") as stream:
    stream.write(payload)
  return {"bytes": len(payload), "path": str(output), "sha256": actual_sha}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = extract(args.log, args.output)
  print(
      "[P38.SERVING.EXTRACT] VERDICT PASS "
      f"bytes={result['bytes']} sha256={result['sha256']} path={result['path']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
