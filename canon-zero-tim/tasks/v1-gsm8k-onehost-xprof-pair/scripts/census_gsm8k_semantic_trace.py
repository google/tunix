#!/usr/bin/env python3
"""Arm-aware semantic Perfetto census for the GSM8K XProf pair."""

from __future__ import annotations

import argparse
import collections
import glob
import re

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace


COMMON_EXACT_COUNTS = {
    "data_loading": 1,
    "reference_inference": 2,
    "advantage_computation": 1,
    "weight_sync": 2,
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero-hp"), required=True)
  parser.add_argument("--run-root", required=True)
  args = parser.parse_args()

  files = glob.glob(
      f"{args.run_root.rstrip('/')}/train/perf/perfetto_trace_v2_*.pb"
  )
  if len(files) != 1:
    raise SystemExit(f"expected exactly one semantic Perfetto, found {len(files)}")
  trace = Trace()
  trace.ParseFromString(open(files[0], "rb").read())
  counts = collections.Counter()
  tracks: set[int] = set()
  for packet in trace.packet:
    if packet.HasField("track_event") and packet.track_event.name:
      name = re.sub(r" \(.*", "", packet.track_event.name)
      counts[name] += 1
      tracks.add(packet.track_event.track_uuid)

  for name, count in sorted(counts.items(), key=lambda item: -item[1]):
    print(f"{count:4d}  {name}")
  errors = []
  for name, expected in COMMON_EXACT_COUNTS.items():
    if counts[name] != expected:
      errors.append(f"{name}={counts[name]}!={expected}")
  # Stock invokes the ordinary train_step once for each of 16 groups, while
  # G6/P59 wraps the complete grouped update once.  Track placement therefore
  # differs by construction; the exact event count is the stable arm contract.
  expected_peft = 32 if args.arm == "native" else 2
  if counts["peft_train"] != expected_peft:
    errors.append(f"peft_train={counts['peft_train']}!={expected_peft}")
  if counts["rollout"] != 128:
    errors.append(f"rollout={counts['rollout']}!=128")
  custom = counts["segmented_value_and_grad"] + counts["gradient_commit"]
  if custom:
    errors.append(f"custom_span_leftovers={custom}")
  print(
      f"file={files[0].split('/')[-1]} arm={args.arm} tracks={len(tracks)} "
      f"peft_train={counts['peft_train']}"
  )
  if errors:
    print("V1_GSM8K_SEMANTIC_CENSUS_RED " + " ".join(errors))
    raise SystemExit(1)
  print(
      f"V1_GSM8K_SEMANTIC_CENSUS_GREEN arm={args.arm} "
      "single_profiled_update=present"
  )


if __name__ == "__main__":
  main()
