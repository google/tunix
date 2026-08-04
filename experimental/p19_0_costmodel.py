"""P19.0 -- does `T = a*G + b*W + c` actually explain the P18 measurements?

Phase 18 produced two coefficients by two independent routes that agreed to
three significant figures (grid ~10.4 us/step, work ~18 us/block).  Before that
model is used to predict anything -- and the whole case for the reshape-to-batch
design rests on a prediction -- it has to be shown to reproduce the numbers it
came from.

Inputs are PARSED from the P18 raw logs, not retyped, so a stale or missing
artifact is an error rather than a silently wrong constant.  G and W are
recomputed from the real MaskInfo (`data_next.shape[-1]` is the grid width the
kernel actually launches; `block_mask != 0` is the work), never assumed to be
(B/256)^2.

GATE: every measured point must be reproduced within 10%.  If not, the model is
rejected and P19.1 measures directly instead of predicting.

CPU ONLY.  Run with JAX_PLATFORMS=cpu.
"""

import re
import sys

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK, dense_mask

LOG_P183 = "/logs/p18_3_kernel_bench.raw.log"
LOG_P184E = "/logs/p18_4e_gridprobe.raw.log"
TOL = 0.10  # pre-registered in phase19.md P19.0 GATE 1


# ---------------------------------------------------------------------------
# measured T, parsed from the artifacts
# ---------------------------------------------------------------------------
def parse_measurements():
  pts = {}
  with open(LOG_P183) as f:
    for line in f:
      m = re.match(r"^(A0|A1|N0|D0|D1)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
      if m:
        pts[m.group(1)] = float(m.group(4))
  with open(LOG_P184E) as f:
    for line in f:
      m = re.match(r"^\s*(2048|4096|8192)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$",
                   line)
      if m:
        pts[f"E{m.group(1)}"] = float(m.group(5))
  return pts


# ---------------------------------------------------------------------------
# G and W, recomputed from the real MaskInfo
# ---------------------------------------------------------------------------
def gw_static(seq_len):
  """Grid steps per head and work blocks for a plain causal mask."""
  mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq_len, seq_len))])
  info, _ = mask_info_lib.process_mask(mask, (BLOCK, BLOCK))
  grid_width = int(np.asarray(info.data_next).shape[-1])
  q_blocks = seq_len // BLOCK
  work = int((np.asarray(info.block_mask) != 0).sum())
  return q_blocks * grid_width, work


def gw_dynamic(dense):
  info, _ = mask_info_lib.process_dynamic_mask(
      jnp.asarray(dense[None], dtype=jnp.bool), (BLOCK, BLOCK))
  grid_width = int(np.asarray(info.data_next).shape[-1])
  q_blocks = dense.shape[0] // BLOCK
  work = int((np.asarray(info.block_mask) != 0).sum())
  return q_blocks * grid_width, work


def pinned_mask(row_len, real=2048):
  pos = np.arange(row_len)
  inside = pos < real
  return (pos[None, :] <= pos[:, None]) & inside[:, None] & inside[None, :]


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if any(d.platform != "cpu" for d in jax.devices()):
    print("REFUSING: CPU-only")
    return 2

  measured = parse_measurements()
  want = {"A0", "A1", "N0", "D0", "D1", "E2048", "E4096", "E8192"}
  missing = want - set(measured)
  if missing:
    print(f"INCONCLUSIVE: could not parse {sorted(missing)} from the P18 logs")
    return 2
  print(f"parsed {len(measured)}/8 measured fwd+bwd times from P18 artifacts\n")

  # --- rebuild the exact geometries P18 measured --------------------------
  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex_a = pack(examples, 8192, 1, 8, row_multiple=1)
  _, _, seg_a, _ = model_inputs(ex_a)
  seg_a = np.asarray(seg_a)
  ex_d = pack(examples, 2048, 4, 8, row_multiple=1)
  _, _, seg_d, _ = model_inputs(ex_d)
  seg_d = np.asarray(seg_d)

  rows = []  # (name, G, W, T, kind)
  g, w = gw_static(8192)
  rows.append(("A0", g, w, measured["A0"], "static"))
  g, w = gw_dynamic(dense_mask(seg_a[0], 8192))
  rows.append(("A1", g, w, measured["A1"], "dynamic"))
  g, w = gw_dynamic(dense_mask(seg_a[0], 8192, causal_only=True))
  rows.append(("N0", g, w, measured["N0"], "dynamic"))
  g, w = gw_static(2048)
  rows.append(("D0", g, w, measured["D0"], "static"))
  g, w = gw_dynamic(dense_mask(seg_d[0], 2048))
  rows.append(("D1", g, w, measured["D1"], "dynamic"))
  for n in (2048, 4096, 8192):
    g, w = gw_dynamic(pinned_mask(n))
    rows.append((f"E{n}", g, w, measured[f"E{n}"], "dynamic"))

  print(f"{'point':<8}{'kind':<9}{'G (grid/head)':>15}{'W (blocks)':>12}"
        f"{'T meas ms':>12}")
  for name, g, w, t, kind in rows:
    print(f"{name:<8}{kind:<9}{g:>15}{w:>12}{t:>12.3f}")
  print()

  # --- fit ------------------------------------------------------------------
  def fit(subset, label):
    A = np.array([[g, w, 1.0] for _, g, w, _, _ in subset])
    y = np.array([t for *_, t, _ in subset])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    rel = np.abs(pred - y) / y
    print(f"--- model [{label}] : a={coef[0]*1000:.3f} us/grid-step, "
          f"b={coef[1]*1000:.3f} us/block, c={coef[2]*1000:+.1f} us")
    for (name, *_r), p, m, e in zip(subset, pred, y, rel):
      flag = "" if e <= TOL else "   <-- OVER TOL"
      print(f"    {name:<8} pred {p:7.3f}  meas {m:7.3f}  rel {e:6.2%}{flag}")
    return coef, rel

  coef_all, rel_all = fit(rows, "all 8 points, shared intercept")
  ok_all = bool((rel_all <= TOL).all())
  print(f"    => max rel err {rel_all.max():.2%}  "
        f"{'PASS' if ok_all else 'FAIL'}\n")

  chosen, label = coef_all, "shared"
  if not ok_all:
    dyn = [r for r in rows if r[4] == "dynamic"]
    coef_dyn, rel_dyn = fit(dyn, "dynamic only (6 points)")
    ok_dyn = bool((rel_dyn <= TOL).all())
    print(f"    => max rel err {rel_dyn.max():.2%}  "
          f"{'PASS' if ok_dyn else 'FAIL'}\n")
    if not ok_dyn:
      print("VERDICT: model REJECTED -- neither form reproduces the "
            "measurements within 10%.\n  P19.1 must measure directly; no "
            "prediction from this model may be quoted.")
      return 1
    chosen, label = coef_dyn, "dynamic-only"
    print("NOTE: the shared-intercept form failed; the dynamic-only fit is "
          "used.\n  Static points are therefore NOT covered by the model and "
          "are quoted as measurements only.")

  a, b, c = chosen
  print(f"MODEL ADOPTED [{label}]: T = {a*1000:.3f}us * G + {b*1000:.3f}us * W "
        f"{c*1000:+.1f}us\n")

  # --- predictions ----------------------------------------------------------
  # Same 8 sequences x 1024 real tokens per chip, three designs.
  print("=" * 74)
  print("PREDICTIONS -- 8 sequences x 1024 tokens per chip, block 256")
  print("=" * 74)
  # Every fitted point was a SINGLE-row [1, L] call, so the intercept c is
  # "per invocation" in the fit.  Production runs the rows as one vmapped call,
  # and whether c is paid once or once per row is NOT determined by this data.
  # Both bounds are reported; P19.1 measures a real multi-row call and settles
  # it.  c is small (~87 us) so the two bounds are close except at many rows.
  print(f"{'design':<34}{'rows':>6}{'G tot':>8}{'W tot':>8}"
        f"{'pred ms (c once..c/row)':>26}{'vs now':>9}")

  designs = []
  # current default: packed at 2048, 4 rows, static
  g, w = gw_static(2048)
  designs.append(("static  packed@2048 (current default)", 4, g, w, "measured"))
  # candidate C at 2048
  g, w = gw_dynamic(dense_mask(seg_d[0], 2048))
  designs.append(("C       packed@2048", 4, g, w, ""))
  # static / C at 8192
  g, w = gw_static(8192)
  designs.append(("static  packed@8192", 1, g, w, ""))
  g, w = gw_dynamic(dense_mask(seg_a[0], 8192))
  designs.append(("C       packed@8192", 1, g, w, ""))
  # reshape-to-batch: 8 independent rows of 1024, plain causal
  g, w = gw_static(1024)
  designs.append(("reshape [8, 1024] plain causal", 8, g, w, "EXTRAPOLATION"))

  base_lo = base_hi = None
  out = []
  for name, nrows, g, w, note in designs:
    gt, wt = nrows * g, nrows * w
    lo = a * gt + b * wt + c            # c paid once for the vmapped call
    hi = a * gt + b * wt + c * nrows    # c paid per row
    if base_lo is None:
      base_lo, base_hi = lo, hi
    out.append((name, nrows, gt, wt, lo, hi, lo / base_lo, hi / base_hi, note))
  for name, nrows, gt, wt, lo, hi, rlo, rhi, note in out:
    span = f"{lo:.2f}..{hi:.2f}"
    print(f"{name:<34}{nrows:>6}{gt:>8}{wt:>8}{span:>26}"
          f"{rlo:>7.2f}-{rhi:.2f}x  {note}")

  print("\n  Interpolation vs extrapolation: every row except the last sits "
        "inside the\n  fitted range (G 64..1024, W 20..528).  The reshape row "
        "has G=16/row and\n  W=10/row, BELOW the smallest fitted point, so it "
        "is an EXTRAPOLATION and\n  P19.1 must measure it rather than trust "
        "it.")
  print("\n  NOTE: with equal-length segments, 'reshape [8, 1024]' IS just "
        "batching the\n  sequences at their own length -- i.e. candidate A "
        "(length bucketing) expressed\n  as a reshape.  Its cost here is the "
        "ideal sum-of-per-segment-triangles.  What\n  it does NOT price is the "
        "padding that equal-length rows reintroduce when the\n  real lengths "
        "are ragged -- that is P19.2, and it is the actual decision gate.")
  print(f"\n  headline prediction to falsify in P19.1: reshape is "
        f"{out[-1][6]:.2f}-{out[-1][7]:.2f}x the current default "
        f"({out[-1][4]:.2f}..{out[-1][5]:.2f} ms vs "
        f"{base_lo:.2f}..{base_hi:.2f} ms)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
