#!/usr/bin/env python3
"""Recover and verify a P38 mismatch capsule from an immutable pod log."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_ARTIFACT_RE = re.compile(
    r"^\[CANON_P38_CAPSULE_ARTIFACT\] .* bytes=(\d+) "
    r"sha256=([0-9a-f]{64}) encoding=base64$"
)
_PAYLOAD_PREFIX = "[CANON_P38_CAPSULE_B64] "


def recover(log_path: Path, output_path: Path) -> dict[str, object]:
  """Recovers one capsule, verifies transport and embedded array hashes."""
  lines = log_path.read_text(encoding="utf-8").splitlines()
  artifact_matches = [
      match for line in lines if (match := _ARTIFACT_RE.fullmatch(line))
  ]
  if len(artifact_matches) != 1:
    raise ValueError(
        f"expected exactly one P38 capsule artifact line, got {len(artifact_matches)}"
    )
  encoded = "".join(
      line[len(_PAYLOAD_PREFIX):]
      for line in lines
      if line.startswith(_PAYLOAD_PREFIX)
  )
  if not encoded:
    raise ValueError("P38 capsule base64 payload is missing")
  payload = base64.b64decode(encoded, validate=True)
  expected_bytes = int(artifact_matches[0].group(1))
  expected_sha = artifact_matches[0].group(2)
  actual_sha = hashlib.sha256(payload).hexdigest()
  if len(payload) != expected_bytes or actual_sha != expected_sha:
    raise ValueError(
        "P38 capsule transport check failed: "
        f"bytes={len(payload)}/{expected_bytes} sha256={actual_sha}/{expected_sha}"
    )
  if output_path.exists():
    raise FileExistsError(f"refusing to overwrite {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("xb") as output_file:
    output_file.write(payload)
    output_file.flush()
  with np.load(output_path, allow_pickle=False) as capsule:
    metadata = json.loads(capsule["metadata_json"].tobytes())
    if metadata.get("schema") != "p38-frozenlake-mismatch-capsule-v1":
      raise ValueError("unexpected P38 capsule schema")
    for name, expected in metadata.get("arrays", {}).items():
      value = np.ascontiguousarray(capsule[name])
      observed = hashlib.sha256(value.tobytes()).hexdigest()
      if observed != expected.get("sha256"):
        raise ValueError(f"P38 capsule array hash mismatch: {name}")
    selected_rows = capsule["selected_rows"].tolist()
  result = {
      "verdict": "PASS",
      "path": str(output_path),
      "bytes": len(payload),
      "sha256": actual_sha,
      "selected_rows": selected_rows,
      "schema": metadata["schema"],
  }
  print(json.dumps(result, sort_keys=True))
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  recover(args.log, args.output)


if __name__ == "__main__":
  main()
