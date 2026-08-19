#!/usr/bin/env python3
"""Span census for a P51/P55 semantic trace.

Usage: python3 census_semantic_trace.py <run_root>
Run inside the training image (it carries the perfetto proto bindings).
Prints base-name counts and exits nonzero unless all three training-phase
spans added on the G6 path are present (peft_train, segmented_value_and
_grad, gradient_commit).
"""
import collections
import glob
import re
import sys

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace

run_root = sys.argv[1].rstrip("/")
files = glob.glob(f"{run_root}/train/perf/perfetto_trace_v2_*.pb")
assert len(files) >= 1, f"no v2 trace under {run_root}"
f = sorted(files)[-1]
t = Trace(); t.ParseFromString(open(f, "rb").read())
base = collections.Counter()
for p in t.packet:
    if p.HasField("track_event") and p.track_event.name:
        base[re.sub(r" \(.*", "", p.track_event.name)] += 1
for n, c in sorted(base.items(), key=lambda x: -x[1]):
    print(f"{c:4d}  {n}")
need = ["peft_train", "segmented_value_and_grad", "gradient_commit"]
missing = [n for n in need if base.get(n, 0) == 0]
print(f"file={f.split('/')[-1]}")
if missing:
    print("CENSUS_RED missing=" + ",".join(missing))
    sys.exit(1)
print("CENSUS_GREEN training spans present")
sys.exit(0)
