#!/usr/bin/env python3
"""Fail-closed integrity census for a P59 narrow backward XProf capture."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Any


_BACKWARD = re.compile(r"pullback|adjoint|_precomputed_gradient_step|bwd_")
_DECODE = re.compile(r"run_model|jit_sample|compute_and_gather")


def inspect_profile(profile_data: Any) -> dict[str, Any]:
  """Returns a serializable eight-plane/drop/module verdict."""
  planes = []
  reasons = []
  for plane in profile_data.planes:
    if "TPU" not in plane.name or "SparseCore" in plane.name:
      continue
    module_names = []
    dropped_events = []
    for line in plane.lines:
      if line.name == "XLA Modules":
        module_names.extend(
            re.sub(r"[(.].*", "", event.name) for event in line.events
        )
      if line.name == "XLA TraceMe":
        dropped_events.extend(
            event.name
            for event in line.events
            if "drop" in event.name.lower() or "overflow" in event.name.lower()
        )
    dropped_stat = dict(plane.stats).get("dropped_traces")
    try:
      dropped_count = int(dropped_stat) if dropped_stat is not None else 0
    except ValueError:
      dropped_count = -1
    has_backward = any(_BACKWARD.search(name) for name in module_names)
    has_decode = any(_DECODE.search(name) for name in module_names)
    has_semantic = any(
        name.startswith("jit_zt_tr_dp_parallel_bwd_")
        for name in module_names
    )
    detail = {
        "name": plane.name,
        "distinct_modules": len(set(module_names)),
        "backward_present": has_backward,
        "decode_present": has_decode,
        "semantic_backward_present": has_semantic,
        "dropped_traces_stat": dropped_stat,
        "dropped_events": dropped_events,
    }
    planes.append(detail)
    if not has_backward:
      reasons.append(f"{plane.name}:backward_absent")
    if has_decode:
      reasons.append(f"{plane.name}:decode_present")
    if not has_semantic:
      reasons.append(f"{plane.name}:semantic_backward_absent")
    if dropped_count != 0:
      reasons.append(f"{plane.name}:dropped_traces={dropped_stat!r}")
    if dropped_events:
      reasons.append(f"{plane.name}:dropped_events={dropped_events!r}")
  if len(planes) != 8:
    reasons.append(f"tpu_planes={len(planes)} expected=8")
  return {
      "schema": "canon-p59-xprof-capture-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "tpu_plane_count": len(planes),
      "planes": planes,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite XProf census: {args.output}")
  xplanes = sorted(glob.glob(
      str(args.run_root / "train/xprof/plugins/profile/*/*.xplane.pb")
  ))
  perfetto = sorted(glob.glob(
      str(args.run_root / "train/xprof/plugins/profile/*/*.trace.json.gz")
  ))
  if len(xplanes) != 1 or len(perfetto) != 1:
    record = {
        "schema": "canon-p59-xprof-capture-v1",
        "verdict": "FAIL",
        "reasons": [
            f"xplane_files={len(xplanes)} expected=1",
            f"perfetto_files={len(perfetto)} expected=1",
        ],
    }
  elif Path(xplanes[0]).stat().st_size <= 0 or Path(perfetto[0]).stat().st_size <= 0:
    record = {
        "schema": "canon-p59-xprof-capture-v1",
        "verdict": "FAIL",
        "reasons": ["empty_xprof_artifact"],
    }
  else:
    from xprof.profile_data import ProfileData  # pylint: disable=g-import-not-at-top

    profile = ProfileData.from_file(xplanes[0])
    record = inspect_profile(profile)
    record["xplane_bytes"] = Path(xplanes[0]).stat().st_size
    record["perfetto_bytes"] = Path(perfetto[0]).stat().st_size
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P59_XPROF_INSPECTION "
      f"verdict={record['verdict']} "
      f"planes={record.get('tpu_plane_count', 0)}/8 "
      f"reasons={record['reasons']}"
  )
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
