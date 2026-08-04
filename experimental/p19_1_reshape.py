"""P19.1 -- reshape-to-batch: does moving segments into the batch dim pay off?

Candidate C (phase18) removed the WORK splash was doing on cross-segment blocks
but left the GRID untouched, and P18.4e showed the grid is the larger term.  The
grid is `q_blocks x grid_width`, and `grid_width` is one number shared by every
q block, so it must cover the worst case of a row holding one full-length
sequence -- even when the row actually holds several short ones.

If the segments in a row are equal length, that constraint disappears without
any kernel work: reshape [heads, S*L, d] -> [S, heads, L, d], run ordinary
causal splash at seq_len=L, reshape back.  Segments never share a row, so no
mask is needed at all, and the grid drops from S^2*(L/b)^2 to S*(L/b)^2.

Three arms on the same 8 sequences x 1024 real tokens per chip, chosen to match
the rows of P19.0's prediction table that have decision value:

  R0  static causal + segment_ids, packed [4, 2048]   <- the current default
  R1  candidate C (dynamic mask),  packed [4, 2048]
  R2  reshape [8, 1024], plain causal, no mask at all

Gates, in this order (phase19.md P19.1):
  1. negative control -- unequal segments must make the reshape path RAISE
  2. numerics -- R2 vs R1, bf16 rel_L2 <= 1e-2 AND fp32 rel_L2 <= 1e-5.
     NOT bitwise: seq_len 1024 vs 2048 is a different program with different
     tiling, and this project established long ago that different programs are
     not bitwise.  fp32 collapse is the discriminator between rounding and a
     structural bug (same method as phase17 P17.1).
  3. timing vs P19.0's predicted 2.73-3.33 ms, +-30%
  4. settle whether the model's intercept is paid once per call or once per row

Run on the v4-8 with the verified image.
"""

import statistics
import sys
import time

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK, dense_mask
from tunix.models.qwen3 import model as model_lib

# Pre-registered in phase19.md before this ran.
PRED_LO, PRED_HI = 2.73, 3.33   # ms, P19.0 prediction for R2 (fwd+bwd)
TIME_TOL = 0.30
BF16_TOL = 1e-2                 # rel_L2
FP32_TOL = 1e-5                 # rel_L2 -- the discriminator


def timed(fn, args, iters=20, warmup=3, inner=5):
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  s = []
  for _ in range(iters):
    t0 = time.perf_counter()
    for _ in range(inner):
      out = fn(*args)
    jax.block_until_ready(out)
    s.append((time.perf_counter() - t0) * 1e3 / inner)
  return statistics.median(s)


def bs_for(n):
  b = min(BLOCK, n)
  return splash.BlockSizes(
      block_q=b, block_kv=b, block_q_dkv=b, block_kv_dkv=b,
      block_kv_dkv_compute=b, block_q_dq=b, block_kv_dq=b)


def segment_lengths(seg_row):
  """Lengths of the contiguous non-zero runs, in order."""
  ids, counts = np.unique(seg_row[seg_row > 0], return_counts=True)
  return [int(counts[list(ids).index(i)]) for i in ids]


def reshape_to_batch(q, k, v, seg_row, seq_len):
  """Split a packed row into one batch entry per segment.

  Requires every segment in the row to have the same length and to tile the row
  exactly.  Anything else RAISES -- silently computing the wrong thing is the
  failure mode this guard exists to prevent (phase19.md P19.1 gate 1).
  """
  lens = segment_lengths(np.asarray(seg_row))
  if not lens:
    raise ValueError("reshape path: row has no real segments")
  if len(set(lens)) != 1:
    raise ValueError(
        f"reshape path requires equal-length segments in a row, got {lens}")
  seg_len = lens[0]
  if seg_len * len(lens) != seq_len:
    raise ValueError(
        f"reshape path requires the segments to tile the row exactly: "
        f"{len(lens)} x {seg_len} != {seq_len}")
  if seg_len % BLOCK:
    raise ValueError(f"segment length {seg_len} not a multiple of {BLOCK}")
  n = len(lens)
  # [heads, S*L, d] -> [S, heads, L, d]
  rs = lambda x: x.reshape(x.shape[0], n, seg_len, x.shape[-1]).transpose(
      1, 0, 2, 3)
  return rs(q), rs(k), rs(v), n, seg_len


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: P19.1 must run on TPU")
    return 2

  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.flash_attention_block_size = BLOCK
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  print(f"qwen3_1p7b {qh}q/{kh}kv x {hd}, block {BLOCK}\n")

  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex_d = pack(examples, 2048, 4, 8, row_multiple=1)
  _, _, seg_d, _ = model_inputs(ex_d)
  seg_d = np.asarray(seg_d)
  rows, L = seg_d.shape
  print(f"packed [{rows}, {L}], segment lengths per row: "
        f"{[segment_lengths(r) for r in seg_d]}\n")

  # ---- GATE 1: negative control, FIRST -----------------------------------
  print("GATE 1 -- negative control: unequal segments must RAISE")
  bad = seg_d[0].copy()
  bad[:100] = 0          # shorten segment 1 -> lengths no longer equal
  dummy = jnp.zeros((qh, L, hd), jnp.bfloat16)
  try:
    reshape_to_batch(dummy, dummy, dummy, bad, L)
    print("  FAIL -- unequal segments were silently accepted")
    return 1
  except ValueError as exc:
    print(f"  PASS -- raised: {str(exc)[:90]}")
  print()

  # ---- build the three arms ----------------------------------------------
  def make_arms(dtype):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (rows, qh, L, hd), dtype)
    k = jax.random.normal(keys[1], (rows, kh, L, hd), dtype)
    v = jax.random.normal(keys[2], (rows, kh, L, hd), dtype)
    seg = jnp.asarray(seg_d, dtype=jnp.int32)

    static_kern = splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask(
            [mask_lib.CausalMask((L, L)) for _ in range(qh)]),
        block_sizes=bs_for(L))

    @jax.jit
    def r0(q, k, v, seg):
      return jax.vmap(lambda a, b, c, s: static_kern(
          a, b, c, splash.SegmentIds(q=s, kv=s)))(q, k, v, seg)

    @jax.jit
    def r1(q, k, v, seg):
      def one(a, b, c, s):
        pos = jnp.arange(L)
        m = (pos[None, :] <= pos[:, None]) & (s[:, None] == s[None, :])
        kern = splash.make_splash_mha_single_device(
            m[None], block_sizes=bs_for(L))
        return splash._splash_attention(  # noqa: SLF001
            kern.fwd_mask_info, kern.dq_mask_info, kern.dkv_mask_info,
            a, b, c, **kern.kwargs)
      return jax.vmap(one)(q, k, v, seg)

    # R2: segments become batch entries; no mask needed at all.
    n_seg, seg_len = None, None
    qs, ks, vs = [], [], []
    for r in range(rows):
      a, b, c, n_seg, seg_len = reshape_to_batch(q[r], k[r], v[r], seg_d[r], L)
      qs.append(a); ks.append(b); vs.append(c)
    q2 = jnp.concatenate(qs, 0); k2 = jnp.concatenate(ks, 0)
    v2 = jnp.concatenate(vs, 0)
    small_kern = splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask(
            [mask_lib.CausalMask((seg_len, seg_len)) for _ in range(qh)]),
        block_sizes=bs_for(seg_len))

    @jax.jit
    def r2(q2, k2, v2):
      return jax.vmap(lambda a, b, c: small_kern(a, b, c))(q2, k2, v2)

    def scatter_back(out2):
      """[S_tot, heads, L_seg, d] -> [rows, heads, L, d] to compare with R0/R1."""
      o = np.asarray(jax.device_get(out2))
      o = o.reshape(rows, n_seg, qh, seg_len, hd).transpose(0, 2, 1, 3, 4)
      return o.reshape(rows, qh, L, hd)

    return ((r0, (q, k, v, seg)), (r1, (q, k, v, seg)), (r2, (q2, k2, v2)),
            scatter_back, n_seg, seg_len)

  # ---- GATE 2: numerics, bf16 then fp32 ----------------------------------
  print("GATE 2 -- numerics (R2 vs R1); NOT bitwise by construction, see "
        "docstring")
  num_fail = []
  for dtype, tol in ((jnp.bfloat16, BF16_TOL), (jnp.float32, FP32_TOL)):
    (a0, x0), (a1, x1), (a2, x2), scat, n_seg, seg_len = make_arms(dtype)
    o1 = np.asarray(jax.device_get(a1(*x1))).astype(np.float32)
    o2 = scat(a2(*x2)).astype(np.float32)
    rel = float(np.linalg.norm(o2 - o1) / (np.linalg.norm(o1) + 1e-30))
    mx = float(np.max(np.abs(o2 - o1)))
    ok = rel <= tol
    print(f"  {np.dtype(dtype).name:<9} rel_L2 {rel:.3e}  max|d| {mx:.3e}"
          f"  (tol {tol:.0e}) {'PASS' if ok else 'FAIL'}")
    if not ok:
      num_fail.append(str(np.dtype(dtype).name))
  if num_fail:
    print(f"\nVOID: numerics FAILED for {num_fail}; per phase19.md every timing "
          "number in this run is discarded.")
    return 1
  print(f"  (reshape produced {n_seg} segments of {seg_len} per row)\n")

  # ---- GATE 3: timing -----------------------------------------------------
  print("TIMING (bf16, median of 20, 5 calls/sample)")
  (a0, x0), (a1, x1), (a2, x2), _, _, _ = make_arms(jnp.bfloat16)
  res = {}
  for name, fn, xs in (("R0 static@2048", a0, x0), ("R1 C@2048", a1, x1),
                       ("R2 reshape[8,1024]", a2, x2)):
    def fb(*args, _fn=fn):
      return jax.grad(
          lambda x: jnp.sum(_fn(x, *args[1:]).astype(jnp.float32)))(args[0])
    t_f = timed(fn, xs)
    t_b = timed(jax.jit(fb), xs)
    res[name] = (t_f, t_b)
    print(f"  {name:<22} fwd {t_f:7.3f}   fwd+bwd {t_b:7.3f}")

  if len(res) * 2 != 6:
    print(f"INCONCLUSIVE: {len(res)*2}/6 numbers produced")
    return 2

  t0 = res["R0 static@2048"][1]
  t1 = res["R1 C@2048"][1]
  t2 = res["R2 reshape[8,1024]"][1]
  print(f"\n  R1/R0 = {t1/t0:.3f}   R2/R0 = {t2/t0:.3f}   R2/R1 = {t2/t1:.3f}")

  lo, hi = PRED_LO * (1 - TIME_TOL), PRED_HI * (1 + TIME_TOL)
  in_band = lo <= t2 <= hi
  print(f"\nGATE 3 -- R2 = {t2:.3f} ms vs P19.0 prediction "
        f"{PRED_LO:.2f}-{PRED_HI:.2f} ms (+-{TIME_TOL:.0%} => [{lo:.2f}, "
        f"{hi:.2f}]): {'PASS' if in_band else 'OUT OF BAND'}")

  # ---- GATE 4: is the intercept per call or per row? ---------------------
  a, b, c = 10.370e-3, 16.400e-3, 86.7e-3
  g_r0, w_r0 = 4 * 64, 4 * 36
  once = a * g_r0 + b * w_r0 + c
  perrow = a * g_r0 + b * w_r0 + 4 * c
  pick = "once per call" if abs(t0 - once) < abs(t0 - perrow) else "once per row"
  print(f"\nGATE 4 -- intercept: R0 measured {t0:.3f}; model says "
        f"{once:.3f} (c once) vs {perrow:.3f} (c per row) => c is paid {pick}")

  print("\n" + "=" * 72)
  if not in_band:
    print("VERDICT: mechanism works numerically but the timing missed the "
          "predicted band -- report measurement, do not quote the model.")
    return 1
  print(f"VERDICT: PASS -- reshape is {t2/t0:.2f}x the current default "
        f"and {t2/t1:.2f}x candidate C, numerics clean.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
