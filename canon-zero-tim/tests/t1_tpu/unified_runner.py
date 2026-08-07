"""Unified single-session test runner for T1 topology admission gates.

Initializes Pathways once, holding a single client session, and sequentially
executes all 9 admission probes (P0..P4, H1..H4) without multi-session churn
or lease reclamation collisions on the Resource Manager.
"""
from __future__ import annotations

import os
import sys
import traceback

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax
import jax.numpy as jnp
import numpy as np


def run_all_probes() -> int:
    overall_rc = 0
    print("[t1.unified] Pathways initialized in single session. Starting probes...", flush=True)

    # -------------------------------------------------------------------------
    # Overlay verification (Section B)
    # -------------------------------------------------------------------------
    print("\n== Overlay promotion verification ==", flush=True)
    import importlib
    CHECKS = [
        ("tpu_inference.layers.jax.linear", "P22XK_MATMUL_ACTIVE", True),
        ("tpu_inference.layers.jax.linear", "P22XK_LINEAR_BASE",   None),
        ("tpu_inference.layers.jax.embed",  "_CANON_F4E_ANNOUNCED", None),
        ("tpu_inference.models.jax.qwen3",  "P22XK_RMSNORM_ACTIVE", True),
        ("tpu_inference.models.jax.qwen2",  "P22XK_SWIGLU_ACTIVE",  True),
    ]
    for mod, attr, want in CHECKS:
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, attr):
                print(f"[verify]   FAIL {mod}.{attr} absent", file=sys.stderr)
                overall_rc = 1
                continue
            got = getattr(m, attr)
            if want is not None and got is not want and got != want:
                print(f"[verify]   FAIL {mod}.{attr}={got!r}, expected {want!r}", file=sys.stderr)
                overall_rc = 1
                continue
            print(f"[verify]   OK   {mod}.{attr}" + (f"={got!r}" if want is not None else ""), flush=True)
        except Exception as exc:
            print(f"[verify]   FAIL {mod}: {exc}", file=sys.stderr)
            overall_rc = 1

    # -------------------------------------------------------------------------
    # P0: Pathways / JAX registration
    # -------------------------------------------------------------------------
    print("\n== P0  Pathways/JAX registration ==", flush=True)
    try:
        import probe_devices
        rc_p0 = probe_devices.main()
        if rc_p0 != 0:
            print(f"  FAIL: P0 exited {rc_p0}", file=sys.stderr)
            overall_rc = 1
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # P1: way-count scan (NEW)
    # -------------------------------------------------------------------------
    print("\n== P1  way-count scan (NEW) ==", flush=True)
    try:
        import probe_waycount
        rc_p1 = probe_waycount.main()
        if rc_p1 != 0:
            print(f"  FAIL: P1 exited {rc_p1}", file=sys.stderr)
            overall_rc = 1
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # P2: mesh order / slice (NEW)
    # -------------------------------------------------------------------------
    print("\n== P2  mesh order / slice (NEW) ==", flush=True)
    try:
        import probe_mesh_order
        rc_p2 = probe_mesh_order.main()
        if rc_p2 != 0:
            print(f"  FAIL: P2 exited {rc_p2}", file=sys.stderr)
            overall_rc = 1
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # P3: bucket contract (NEW)
    # -------------------------------------------------------------------------
    print("\n== P3  bucket contract (NEW) ==", flush=True)
    try:
        import probe_bucket_contract
        rc_p3 = probe_bucket_contract.main()
        if rc_p3 != 0:
            print(f"  FAIL: P3 exited {rc_p3}", file=sys.stderr)
            overall_rc = 1
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # P4: F4 cost model (NEW)
    # -------------------------------------------------------------------------
    print("\n== P4  F4 cost model (NEW) ==", flush=True)
    try:
        import probe_f4_cost
        rc_p4 = probe_f4_cost.main()
        if rc_p4 != 0:
            print(f"  FAIL: P4 exited {rc_p4}", file=sys.stderr)
            overall_rc = 1
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # H1: minrepro: F4 tree
    # -------------------------------------------------------------------------
    print("\n== H1  minrepro: F4 tree ==", flush=True)
    try:
        import importlib
        import p19_minrepro_f4
        importlib.reload(p19_minrepro_f4)
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # H2: minrepro: third program
    # -------------------------------------------------------------------------
    print("\n== H2  minrepro: third program ==", flush=True)
    try:
        import importlib
        import p19_minrepro_thirdprog
        importlib.reload(p19_minrepro_thirdprog)
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # H3: minrepro: device topology
    # -------------------------------------------------------------------------
    print("\n== H3  minrepro: device topology ==", flush=True)
    try:
        import importlib
        import p19_minrepro_topo
        importlib.reload(p19_minrepro_topo)
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    # -------------------------------------------------------------------------
    # H4: minrepro: mesh geometry
    # -------------------------------------------------------------------------
    print("\n== H4  minrepro: mesh geometry ==", flush=True)
    try:
        import importlib
        import p19_minrepro_mesh2d
        importlib.reload(p19_minrepro_mesh2d)
    except Exception as exc:
        traceback.print_exc()
        overall_rc = 1

    print("\n" + ("=" * 60), flush=True)
    if overall_rc == 0:
        print("===== T1 COMPLETE -- all probes produced measurements =====", flush=True)
        print("NOTE: 'complete' means every probe ran and reported.  Whether the numbers ADMIT this", flush=True)
        print("topology is a judgement against CLUSTER_ADMISSION.md, not an exit code.", flush=True)
    else:
        print("===== T1 FAIL -- a probe did not run or exited nonzero =====", flush=True)
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(run_all_probes())
