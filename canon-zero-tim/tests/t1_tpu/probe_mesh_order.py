"""Admission probe: device order, and whether this topology is multi-slice.

Two things this answers before any expensive run.

1. DEVICE ORDER.  The numerical class of a run depends on the physical order the mesh
   assigns to devices.  A topology-aware create_device_mesh does not preserve the order you
   passed in -- on the probe host, [0,1,2,3] comes back as [0,2,1,3].  Two processes that
   "use the same expression" therefore do NOT automatically agree: two different mesh SHAPES
   fed to create_device_mesh produce different permutations.  Rollout and trainer must be
   asserted equal, not assumed equal.  Set CANON_EXPECT_MODEL_MESH_IDS on both sides and
   this probe (and the engine's own guard) will fail closed on a mismatch.

2. SLICE STRUCTURE.  Everything the canonical switch set was validated against ran inside a
   single slice, where collectives stay on the inter-chip interconnect.  Across slices XLA
   lowers a hierarchical reduction (intra-slice, then inter-slice).  That is a NEW program
   -splitting mechanism with zero coverage in this work -- not a scaled-up version of an old
   one.  This probe reports the slice structure so the risk is explicit rather than
   discovered halfway through a campaign.

    python3 probe_mesh_order.py

Environment:
    CANON_EXPECT_MODEL_MESH_IDS   comma-separated device ids; mismatch -> nonzero exit
    CANON_MESH_SHAPE              comma-separated mesh shape to build (default: 1D over all)
"""
import os
import sys

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax
import numpy as np
from jax.experimental import mesh_utils


def _attr(d, name):
    v = getattr(d, name, None)
    return v if v is not None else "-"


def main():
    devs = jax.devices()
    nd = len(devs)
    print(f"[mesh] visible_devices={nd} kind={devs[0].device_kind} "
          f"platform={devs[0].platform}", flush=True)

    slices = {}
    for d in devs:
        si = _attr(d, "slice_index")
        slices.setdefault(si, []).append(d.id)
    print(f"[mesh] raw_device_ids={[d.id for d in devs]}", flush=True)
    for d in devs[: min(nd, 8)]:
        print(f"[mesh]   id={d.id} process={_attr(d, 'process_index')} "
              f"slice={_attr(d, 'slice_index')} coords={_attr(d, 'coords')}", flush=True)
    if nd > 8:
        print(f"[mesh]   ... {nd - 8} more", flush=True)

    n_slices = len(slices)
    print(f"[mesh] slice_count={n_slices} slices={ {k: len(v) for k, v in slices.items()} }",
          flush=True)
    if n_slices > 1:
        print("[mesh] MULTI_SLICE=1  -- collectives cross slices, so XLA lowers a "
              "hierarchical (intra-slice then inter-slice) reduction.  The canonical switch "
              "set has NO coverage for that program family; treat every bitwise claim on "
              "this topology as UNVERIFIED until re-measured here.", flush=True)
    else:
        print("[mesh] MULTI_SLICE=0  -- single slice, same collective family as the "
              "validated probe host.", flush=True)

    shape_env = os.environ.get("CANON_MESH_SHAPE", "")
    shape = tuple(int(s) for s in shape_env.split(",") if s.strip()) if shape_env else (nd,)
    if int(np.prod(shape)) != nd:
        print(f"[mesh] REFUSING: CANON_MESH_SHAPE={shape} has {int(np.prod(shape))} devices, "
              f"{nd} visible", file=sys.stderr)
        return 2

    built = mesh_utils.create_device_mesh(shape, devs)
    ids = [int(d.id) for d in np.asarray(built).flatten()]
    print(f"[mesh] create_device_mesh(shape={shape}) -> ids={ids}", flush=True)
    if ids != [d.id for d in devs]:
        print("[mesh] REORDERED=1  -- topology-aware placement permuted the device order. "
              "Reading an earlier log line that shows the pre-mesh order would mis-read this "
              "run; only the post-build order is meaningful.", flush=True)
    else:
        print("[mesh] REORDERED=0", flush=True)

    expect = os.environ.get("CANON_EXPECT_MODEL_MESH_IDS", "")
    if expect:
        want = [int(x) for x in expect.split(",") if x.strip()]
        print(f"[mesh] expected_ids={want}", flush=True)
        if want != ids:
            print(f"[mesh] VERDICT: MISMATCH -- expected {want}, got {ids}", flush=True)
            return 1
        print("[mesh] VERDICT: MATCH", flush=True)
    else:
        print("[mesh] VERDICT: REPORTED (CANON_EXPECT_MODEL_MESH_IDS unset -- nothing was "
              "asserted; set it on BOTH rollout and trainer to make the agreement enforced "
              "rather than hoped for)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
