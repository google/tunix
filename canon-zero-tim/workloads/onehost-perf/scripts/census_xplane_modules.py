#!/usr/bin/env python3
"""Device-plane module census for a P51 capture: proves what the xplane holds.

Usage:  python3 census_xplane_modules.py <run_root>
Needs:  pip install --user xprof   (host-side; parses the XSpace proto)

Walks the XLA Modules line of EVERY TensorCore plane (SparseCore planes
excluded) and prints one summary line per plane. Verdict semantics:
  - CENSUS_GREEN (rc=0) only when every plane has the backward family
    (block_pullback / pullback_local_* / adjoint / _precomputed_gradient
    _step) present AND the engine decode family (run_model / jit_sample /
    compute_and_gather) absent -- i.e. the capture proves a clean
    phase=update window on all cores.
  - A step-mode capture at the certified geometry is CENSUS_RED (rc=1) by
    definition: the device trace buffer (~2.8M op events per core) fills
    on ~25s of decode and drops the trainer entirely.
Binary-grepping the xplane cannot stand in for this: host planes are not
subject to the device buffer and mention trainer names in every mode.
"""
import glob
import re
import sys

from xprof.profile_data import ProfileData

BACKWARD = re.compile(r"pullback|adjoint|_precomputed_gradient_step")
DECODE = re.compile(r"run_model|jit_sample|compute_and_gather")

run_root = sys.argv[1].rstrip("/")
files = glob.glob(f"{run_root}/train/xprof/plugins/profile/*/*.xplane.pb")
assert files, f"no xplane under {run_root}"
pd = ProfileData.from_file(sorted(files)[-1])

planes_checked = 0
failures = []
detail = None
for plane in pd.planes:
    if "TPU" not in plane.name or "SparseCore" in plane.name:
        continue
    names = {}
    tmin = tmax = None
    for line in plane.lines:
        if line.name != "XLA Modules":
            continue
        for ev in line.events:
            start = ev.start_ns
            end = start + ev.duration_ns
            tmin = start if tmin is None or start < tmin else tmin
            tmax = end if tmax is None or end > tmax else tmax
            base = re.sub(r"[(.].*", "", ev.name)
            names[base] = names.get(base, 0) + 1
    span = (tmax - tmin) / 1e9 if names else 0.0
    has_backward = any(BACKWARD.search(name) for name in names)
    has_decode = any(DECODE.search(name) for name in names)
    print(f"plane={plane.name} distinct_modules={len(names)} span={span:.1f}s "
          f"backward={'present' if has_backward else 'ABSENT'} "
          f"decode={'PRESENT' if has_decode else 'absent'}")
    planes_checked += 1
    if not has_backward or has_decode:
        failures.append(plane.name)
    if detail is None:
        detail = (plane.name, names)
assert planes_checked > 0, "no TensorCore planes in the xplane"
if detail:
    print(f"module detail for {detail[0]}:")
    for name, count in sorted(detail[1].items(), key=lambda item: -item[1]):
        print(f"  {count:7d}  {name}")
if failures:
    print(f"CENSUS_RED {len(failures)}/{planes_checked} planes fail "
          "backward-present + decode-absent: " + ",".join(failures))
    sys.exit(1)
print(f"CENSUS_GREEN all {planes_checked} planes: backward present, decode absent")
sys.exit(0)
