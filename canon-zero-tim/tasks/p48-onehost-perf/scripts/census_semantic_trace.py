#!/usr/bin/env python3
"""Span census for a P51/P55 semantic trace.

Usage: python3 census_semantic_trace.py <run_root>
Run inside the training image (it carries the perfetto proto bindings).

Verdict (rc=0 only when all hold):
  1. peft_train is on the timeline (2 events per global step); the G6 path
     never enters PeftTrainer.train(), where the built-in span lives.
  2. peft_train occupies exactly the tracks weight_sync occupies -- the
     writer double-writes device-tagged official spans onto the cluster
     lane plus one device lane, and matching the established placement is
     what keeps the picture readable.
  3. On each shared track the peft_train count equals the weight_sync
     count (one named begin per global step): fewer means some steps went
     unrecorded.
  4. The track inventory is exactly the certified one-host vehicle shape
     (19: 8 device lanes + cluster + 10 host thread lanes): more means
     the writer spilled auxiliary lanes.
  5. No leftovers of the reverted custom spans (segmented_value_and_grad,
     gradient_commit): overlapping or invented names are what made the
     writer spill lanes and render an unreadable smear (the p55c failure).
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
EXPECTED_TRACKS = 19  # certified one-host vehicle shape

per_track = collections.defaultdict(collections.Counter)
base = collections.Counter()
for p in t.packet:
    if p.HasField("track_event") and p.track_event.name:
        name = re.sub(r" \(.*", "", p.track_event.name)
        per_track[p.track_event.track_uuid][name] += 1
        base[name] += 1
for n, c in sorted(base.items(), key=lambda x: -x[1]):
    print(f"{c:4d}  {n}")
peft_tracks = {u for u, c in per_track.items() if c.get("peft_train")}
sync_tracks = {u for u, c in per_track.items() if c.get("weight_sync")}
custom = base.get("segmented_value_and_grad", 0) + base.get("gradient_commit", 0)
print(f"file={f.split('/')[-1]} tracks={len(per_track)} "
      f"peft_tracks={sorted(peft_tracks)} sync_tracks={sorted(sync_tracks)}")
red = []
if len(per_track) != EXPECTED_TRACKS:
    red.append(f"tracks={len(per_track)}!={EXPECTED_TRACKS}")
if not base.get("peft_train"):
    red.append("missing=peft_train")
elif peft_tracks != sync_tracks:
    red.append(f"placement peft={sorted(peft_tracks)} != weight_sync={sorted(sync_tracks)}")
else:
    for u in sorted(sync_tracks):
        if per_track[u]["peft_train"] != per_track[u]["weight_sync"]:
            red.append(
                f"count_mismatch track={u} "
                f"peft={per_track[u]['peft_train']} "
                f"sync={per_track[u]['weight_sync']}")
if custom:
    red.append(f"custom_span_leftovers={custom}")
if red:
    print("CENSUS_RED " + " ".join(red))
    sys.exit(1)
print("CENSUS_GREEN peft_train placed like weight_sync, no custom spans")
sys.exit(0)
