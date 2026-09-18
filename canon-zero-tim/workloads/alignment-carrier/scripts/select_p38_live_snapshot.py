#!/usr/bin/env python3
"""Select the most complete immutable P38 live snapshot from an object listing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


class SnapshotSelectionError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise SnapshotSelectionError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def select_snapshot(
    listing_path: Path,
    live_root: str,
    min_capsule_rounds: int,
) -> dict[str, Any]:
  _require(listing_path.is_file(), f"object listing is absent: {listing_path}")
  root = live_root.rstrip("/")
  _require(
      root.startswith(
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/")
      and root.endswith("/attempt-0/live"),
      "live root is outside the registered P38 attempt-0 hierarchy",
  )
  _require(min_capsule_rounds > 0, "minimum capsule rounds must be positive")
  prefix = root + "/"
  objects: dict[str, set[str]] = {}
  for raw in listing_path.read_text(encoding="utf-8").splitlines():
    value = raw.strip().rstrip(":")
    if not value.startswith(prefix):
      continue
    relative = value[len(prefix):].lstrip("/")
    parts = relative.split("/", 1)
    if len(parts) != 2 or not re.fullmatch(r"[0-9]{6}", parts[0]):
      continue
    filename = parts[1]
    if not filename or filename.endswith("/"):
      continue
    objects.setdefault(parts[0], set()).add(filename)
  _require(objects, "object listing contains no six-digit live snapshots")

  candidates = []
  for snapshot, names in sorted(objects.items()):
    capsule_rounds = sorted({
        int(match.group(1))
        for name in names
        if (match := re.fullmatch(
            r"p38_frozenlake_mismatch_capsule\.round-([0-9]{6})\.npz",
            name,
        ))
    })
    seam_json = {
        match.group(1) for name in names
        if (match := re.fullmatch(r"p38_seam_([0-9]{6})\.json", name))
    }
    seam_npz = {
        match.group(1) for name in names
        if (match := re.fullmatch(r"p38_seam_([0-9]{6})\.npz", name))
    }
    contiguous_rounds = capsule_rounds == list(range(len(capsule_rounds)))
    paired_seam_records = seam_json == seam_npz
    qualifies = (
        "LIVE.json" in names
        and "SHA256SUMS" in names
        and "run.log" in names
        and "pre-alignment.jsonl" in names
        and len(capsule_rounds) >= min_capsule_rounds
        and contiguous_rounds
        and bool(seam_json)
        and paired_seam_records
    )
    candidates.append({
        "snapshot": snapshot,
        "object_count": len(names),
        "capsule_rounds": capsule_rounds,
        "seam_json_records": len(seam_json),
        "seam_npz_records": len(seam_npz),
        "paired_seam_records": paired_seam_records,
        "has_live": "LIVE.json" in names,
        "has_sha256sums": "SHA256SUMS" in names,
        "has_run_log": "run.log" in names,
        "has_pre_alignment": "pre-alignment.jsonl" in names,
        "contiguous_capsule_rounds": contiguous_rounds,
        "qualifies": qualifies,
    })
  qualified = [item for item in candidates if item["qualifies"]]
  selected = max(
      qualified,
      key=lambda item: (len(item["capsule_rounds"]), int(item["snapshot"])),
      default=None,
  )
  return {
      "schema": "p38-live-snapshot-selection-v1",
      "live_root": root,
      "listing_sha256": _sha256(listing_path),
      "minimum_capsule_rounds": min_capsule_rounds,
      "candidate_count": len(candidates),
      "qualified_candidate_count": len(qualified),
      "candidates": candidates,
      "selected_snapshot": selected["snapshot"] if selected else None,
      "selected_source_gcs_uri": (
          f"{root}/{selected['snapshot']}" if selected else None),
      "selected_capsule_rounds": (
          selected["capsule_rounds"] if selected else []),
      "selection_complete": selected is not None,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--listing", type=Path, required=True)
  parser.add_argument("--live-root", required=True)
  parser.add_argument("--min-capsule-rounds", type=int, default=2)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    report = select_snapshot(
        args.listing, args.live_root, args.min_capsule_rounds)
    _write_json(args.output, report)
  except (SnapshotSelectionError, OSError, ValueError) as error:
    print(f"[P38.SNAPSHOT] REFUSING: {error}", file=sys.stderr)
    return 2
  if not report["selection_complete"]:
    print(
        "[P38.SNAPSHOT] INCONCLUSIVE no qualifying snapshot "
        f"candidates={report['candidate_count']} "
        f"minimum_capsule_rounds={report['minimum_capsule_rounds']}",
        file=sys.stderr,
    )
    return 4
  print(
      "[P38.SNAPSHOT] SELECTED "
      f"snapshot={report['selected_snapshot']} "
      "capsule_rounds="
      + ",".join(str(value) for value in report["selected_capsule_rounds"]),
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
