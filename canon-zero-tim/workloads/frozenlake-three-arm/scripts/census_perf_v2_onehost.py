#!/usr/bin/env python3
"""Parse the P57.1c semantic Perfetto artifact inside the pinned image."""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
import re


def _event_contract_reasons(
    counts: collections.Counter[str], reference_inference: str
) -> list[str]:
  reasons = []
  required = (
      "data_loading",
      "rollout",
      "advantage_computation",
      "peft_train",
      "weight_sync",
  )
  for name in required:
    if counts[name] <= 0:
      reasons.append(f"missing_event={name}")
  if reference_inference == "required":
    if counts["reference_inference"] <= 0:
      reasons.append("missing_event=reference_inference")
  elif counts["reference_inference"] > 0:
    reasons.append("unexpected_event=reference_inference")
  return reasons


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--perf-dir", required=True)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--reference-inference",
      required=True,
      choices=("required", "disabled"),
  )
  args = parser.parse_args()
  from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

  files = glob.glob(f"{args.perf_dir.rstrip('/')}/perfetto_trace_v2_*.pb")
  reasons = []
  counts: collections.Counter[str] = collections.Counter()
  tracks: set[int] = set()
  size = 0
  if len(files) != 1:
    reasons.append(f"files={len(files)}")
  else:
    data = Path(files[0]).read_bytes()
    size = len(data)
    if not data:
      reasons.append("empty_trace")
    else:
      trace = Trace()
      trace.ParseFromString(data)
      for packet in trace.packet:
        if packet.HasField("track_event") and packet.track_event.name:
          name = re.sub(r" \(.*", "", packet.track_event.name)
          counts[name] += 1
          tracks.add(packet.track_event.track_uuid)
  reasons.extend(_event_contract_reasons(counts, args.reference_inference))
  result = {
      "schema": "p57-perf-v2-semantic-census-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "files": len(files),
      "bytes": size,
      "tracks": len(tracks),
      "reference_inference_contract": args.reference_inference,
      "event_counts": dict(sorted(counts.items())),
      "reasons": reasons,
  }
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("P57_PERF_V2_SEMANTIC_JSON " + json.dumps(result, sort_keys=True))
  if reasons:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
