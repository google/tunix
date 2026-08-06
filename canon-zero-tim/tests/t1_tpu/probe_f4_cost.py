"""Admission probe: what does the fixed-order tree cost at THIS tensor-parallel width?

CANON_FIXED_AR replaces the compiler's all-reduce with an explicit fixed-order tree, built
from n-1 full-tensor rotations that are then stacked and summed in global-rank order.  The
rank-order sort needs a dynamic index, so the stack is materialised; it does not fold into a
running accumulator.

That buys row-position and third-program invariance.  It costs communication and memory, and
the cost is LINEAR in the width while a ring all-reduce is not:

    ring all-reduce      moves  2(n-1)/n * B   per chip,  ~1 buffer live
    F4 fixed-order tree  moves    (n-1)  * B   per chip,  ~n buffers live
    ratio                        n/2

At n=4 that is 2x -- measured and accepted on the probe host.  At n=8 it is 4x, and no one
has measured it.  This is an analytic model derived from reading the implementation, not a
benchmark: use it to decide whether a width deserves a measurement, never as the measurement.

The tree only has to be rank-ordered, not linear.  Recursive doubling (log2(n) rounds, pairs
chosen by rank bit) yields the same rank-fixed sum order at log2(n)*B communication.  That is
a real optimisation direction -- and it is a NEW implementation, so it would have to clear
the full THIRDPROG and A=B gate set before it could be adopted.

    python3 probe_f4_cost.py

Environment:
    CANON_TP_WIDTHS   comma-separated widths (default 2,4,8,16,32)
    CANON_OUT_BYTES   bytes of one projection output (default 256*5120*2, i.e. M=256,
                      D=5120, bf16)
    CANON_N_LAYERS    layers, for the per-step total (default 64)
    CANON_SITES       reduction sites per layer (default 2: o_proj and down_proj)
"""
import math
import os


def main():
    widths = [int(w) for w in os.environ.get("CANON_TP_WIDTHS", "2,4,8,16,32").split(",")]
    B = int(os.environ.get("CANON_OUT_BYTES", str(256 * 5120 * 2)))
    layers = int(os.environ.get("CANON_N_LAYERS", "64"))
    sites = int(os.environ.get("CANON_SITES", "2"))

    print(f"[f4cost] out_bytes_per_site={B} ({B / 2**20:.2f} MiB) layers={layers} "
          f"sites_per_layer={sites}", flush=True)
    print("[f4cost] model: ring=2(n-1)/n*B moved, 1 buffer live; "
          "F4=(n-1)*B moved, n buffers live", flush=True)
    print(f"[f4cost] {'width':>5} {'ring MiB':>10} {'F4 MiB':>10} {'ratio':>7} "
          f"{'F4 live MiB':>12} {'F4 per-step GiB':>16} {'recdbl MiB':>11}", flush=True)

    for n in widths:
        ring = 2 * (n - 1) / n * B
        tree = (n - 1) * B
        live = n * B
        per_step = tree * layers * sites
        recdbl = math.log2(n) * B if n > 1 and (n & (n - 1)) == 0 else float("nan")
        print(f"[f4cost] {n:5d} {ring / 2**20:10.2f} {tree / 2**20:10.2f} "
              f"{tree / ring if ring else float('nan'):7.2f} {live / 2**20:12.2f} "
              f"{per_step / 2**30:16.2f} {recdbl / 2**20:11.2f}", flush=True)

    print("[f4cost] VERDICT: ANALYTIC -- derived from the implementation, not measured. "
          "A width whose ratio you are unwilling to pay needs either a measurement or the "
          "recursive-doubling variant (which must re-clear THIRDPROG and A=B first).",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
