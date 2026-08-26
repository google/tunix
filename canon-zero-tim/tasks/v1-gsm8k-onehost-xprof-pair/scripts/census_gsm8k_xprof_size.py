#!/usr/bin/env python3
"""Fail-closed logical-byte budget census for a one-host XProf artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat


SOFT_WARNING_BYTES = 1_200_000_000
HARD_MAX_BYTES = 1_500_000_000
SCHEMA = "canon.v1.gsm8k-onehost-xprof.size.v1"


def _kind(path: Path) -> str:
  name = path.name
  if name.endswith(".xplane.pb"):
    return "xplane"
  if name.endswith(".trace.json.gz"):
    return "trace_json_gz"
  return "other"


def build_receipt(run_root: Path) -> dict:
  """Returns a deterministic receipt without reading artifact contents."""
  xprof_root = run_root / "train/xprof"
  reasons: list[str] = []
  files: list[dict] = []
  if not xprof_root.is_dir():
    reasons.append("missing_xprof_directory")
  else:
    for directory, dirnames, filenames in os.walk(
        xprof_root, followlinks=False
    ):
      directory_path = Path(directory)
      for name in sorted(dirnames):
        path = directory_path / name
        if path.is_symlink():
          reasons.append(
              "symlink_not_allowed:" + path.relative_to(xprof_root).as_posix()
          )
      for name in sorted(filenames):
        path = directory_path / name
        relative = path.relative_to(xprof_root).as_posix()
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
          reasons.append("symlink_not_allowed:" + relative)
          continue
        if not stat.S_ISREG(metadata.st_mode):
          reasons.append("non_regular_file:" + relative)
          continue
        files.append({
            "path": relative,
            "kind": _kind(path),
            "bytes": metadata.st_size,
        })

  files.sort(key=lambda row: row["path"])
  total_bytes = sum(row["bytes"] for row in files)
  counts = {
      kind: sum(row["kind"] == kind for row in files)
      for kind in ("xplane", "trace_json_gz", "other")
  }
  if counts["xplane"] != 1:
    reasons.append(f"xplane_files={counts['xplane']} expected=1")
  if counts["trace_json_gz"] != 1:
    reasons.append(
        f"trace_json_gz_files={counts['trace_json_gz']} expected=1"
    )
  for row in files:
    if row["kind"] in ("xplane", "trace_json_gz") and row["bytes"] <= 0:
      reasons.append(f"empty_required_artifact:{row['path']}")
  if total_bytes > HARD_MAX_BYTES:
    reasons.append(
        f"xprof_bytes={total_bytes} exceeds_hard_max={HARD_MAX_BYTES}"
    )

  if reasons:
    status = "FAIL"
  elif total_bytes > SOFT_WARNING_BYTES:
    status = "WARN"
  else:
    status = "PASS"
  return {
      "schema": SCHEMA,
      "status": status,
      "xprof_root": "train/xprof",
      "byte_basis": "sum_of_logical_bytes_for_regular_files",
      "soft_warning_bytes": SOFT_WARNING_BYTES,
      "hard_max_bytes": HARD_MAX_BYTES,
      "total_bytes": total_bytes,
      "file_count": len(files),
      "counts": counts,
      "files": files,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  receipt = build_receipt(args.run_root)
  args.output.write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  marker = (
      "V1_GSM8K_XPROF_SIZE_CENSUS_"
      + ("GREEN" if receipt["status"] in ("PASS", "WARN") else "RED")
  )
  print(
      f"{marker} status={receipt['status']} "
      f"xprof_bytes={receipt['total_bytes']} "
      f"soft_warning_bytes={SOFT_WARNING_BYTES} "
      f"hard_max_bytes={HARD_MAX_BYTES} "
      f"files={receipt['file_count']} "
      f"xplanes={receipt['counts']['xplane']} "
      f"traces={receipt['counts']['trace_json_gz']} "
      f"reasons={json.dumps(receipt['reasons'], separators=(',', ':'))}"
  )
  return 0 if receipt["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
  raise SystemExit(main())
