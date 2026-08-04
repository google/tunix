"""P19.3'b -- measure the static block-diagonal template arm on TPU.

The revived candidate B: a STATIC NumpyMask carrying the row's (quantized)
segment layout.  The static path shrinks the grid (grid_width = max non-empty
kv blocks per q row), so this should collect the same schedule the abandoned
custom kernel would have -- with zero kernel code.  segment_ids are still
passed, exactly as production would (for quantized templates they do the exact
masking; here template == true layout, so they are redundant but faithful).

Arms (same harness as P19.1 -- jax.vmap on one chip -- so numbers compare):
  @2048, 4 rows, template (1024, 1024):
    S0  static causal + segment_ids     (current default)
    S2  static blockdiag + segment_ids  (this design)
  @8192, 1 row, template (1024,)*8:
    A0  static causal + segment_ids     (P18.3 measured 20.069 fwd+bwd)
    S2L static blockdiag + segment_ids

Gates in order (phase19.md P19.3'b, written before running):
  1. block counts == pre-registered (S2: width 4, grid 32, work 20;
     S2L: width 4, grid 128, work 80)
  2. numerics: S2 bitwise == S0 at BOTH geometries (skipped blocks are
     floating-point identities -- P18.3; red => timings VOID)
  3. timing: S2/S0 @2048 within +-30% of 0.526; S2L within 2.6-2.8ms +-30%
  4. completeness: 4 arms x {fwd, fwd+bwd}
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
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

EXPECT = {"S2": (4, 32, 20), "S2L": (4, 128, 80)}  # (width, grid, work)
RATIO_2048, RTOL = 0.526, 0.30
PRED_8192 = (2.6, 2.8)


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


def blockdiag_dense(seq_len, seg_lens):
  pos = np.arange(seq_len)
  seg = np.zeros(seq_len, dtype=np.int64)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    seg[p:p + L] = i
    p += L
  causal = pos[None, :] <= pos[:, None]
  return causal & (seg[:, None] == seg[None, :]) & (seg[:, None] > 0), seg


def seg_ids_arr(seq_len, seg_lens):
  seg = np.zeros(seq_len, dtype=np.int32)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    seg[p:p + L] = i
    p += L
  return seg


def check_counts(name, dense):
  m = mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)])
  info, _ = mask_info_lib.process_mask(m, (BLOCK, BLOCK))
  dn = np.asarray(info.data_next)
  width = dn.shape[2]
  grid = dn.shape[1] * width
  work = int((np.asarray(info.block_mask) != 0).sum())
  w_e, g_e, k_e = EXPECT[name]
  ok = (width, grid, work) == (w_e, g_e, k_e)
  print(f"  {name}: width={width} grid={grid} work={work}  "
        f"expected ({w_e},{g_e},{k_e})  {'OK' if ok else '<-- MISMATCH'}")
  return ok


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: TPU only")
    return 2
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  print(f"qwen3_1p7b {qh}q/{kh}kv x {hd}, block {BLOCK}\n")

  d2048, _ = blockdiag_dense(2048, [1024, 1024])
  d8192, _ = blockdiag_dense(8192, [1024] * 8)

  print("GATE 1 -- block counts vs pre-registered")
  if not (check_counts("S2", d2048) and check_counts("S2L", d8192)):
    print("VOID: geometry is not the one pre-registered; no timing follows.")
    return 1
  print()

  def arms_for(seq_len, rows, seg_lens, dense):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (rows, qh, seq_len, hd), jnp.bfloat16)
    k = jax.random.normal(keys[1], (rows, kh, seq_len, hd), jnp.bfloat16)
    v = jax.random.normal(keys[2], (rows, kh, seq_len, hd), jnp.bfloat16)
    seg = jnp.asarray(
        np.tile(seg_ids_arr(seq_len, seg_lens)[None], (rows, 1)))

    causal_kern = splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask(
            [mask_lib.CausalMask((seq_len, seq_len)) for _ in range(qh)]),
        block_sizes=bs_for(seq_len))
    bd_kern = splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)] * qh),
        block_sizes=bs_for(seq_len))

    def call(kern):
      @jax.jit
      def f(q, k, v, s):
        return jax.vmap(lambda a, b, c, t: kern(
            a, b, c, splash.SegmentIds(q=t, kv=t)))(q, k, v, s)
      return f

    return call(causal_kern), call(bd_kern), (q, k, v, seg)

  results = {}
  numerics_fail = []
  for label, seq_len, rows, seg_lens, dense in (
      ("2048", 2048, 4, [1024, 1024], d2048),
      ("8192", 8192, 1, [1024] * 8, d8192)):
    f0, f2, args = arms_for(seq_len, rows, seg_lens, dense)
    o0 = np.asarray(jax.device_get(f0(*args)))
    o2 = np.asarray(jax.device_get(f2(*args)))
    same = np.array_equal(o0.view(np.uint16), o2.view(np.uint16))
    print(f"GATE 2 -- numerics @{label}: "
          f"{'BITWISE IDENTICAL' if same else 'DIFFERS'}")
    if not same:
      n = int((o0.view(np.uint16) != o2.view(np.uint16)).sum())
      mx = float(np.max(np.abs(o0.astype(np.float32) - o2.astype(np.float32))))
      print(f"    bytes {n}/{o0.size}  max|d| {mx:.3e}")
      numerics_fail.append(label)
      continue
    for arm, fn in ((f"S0@{label}", f0), (f"S2@{label}", f2)):
      def fb(*a, _fn=fn):
        return jax.grad(
            lambda x: jnp.sum(_fn(x, *a[1:]).astype(jnp.float32)))(a[0])
      results[arm] = (timed(fn, args), timed(jax.jit(fb), args))

  if numerics_fail:
    print(f"\nVOID: numerics red at {numerics_fail}; timings discarded.")
    return 1

  print("\nTIMING (bf16, median of 20)")
  for arm, (tf, tb) in results.items():
    print(f"  {arm:<10} fwd {tf:7.3f}   fwd+bwd {tb:7.3f}")
  if len(results) != 4:
    print(f"INCONCLUSIVE: {len(results)}/4 arms")
    return 2

  r2048 = results["S2@2048"][1] / results["S0@2048"][1]
  t8192 = results["S2@8192"][1]
  ok_a = abs(r2048 - RATIO_2048) <= RTOL * RATIO_2048 + 1e-9
  lo, hi = PRED_8192[0] * (1 - RTOL), PRED_8192[1] * (1 + RTOL)
  ok_b = lo <= t8192 <= hi
  print(f"\nGATE 3a -- S2/S0 @2048 = {r2048:.3f} vs {RATIO_2048} +-30%: "
        f"{'PASS' if ok_a else 'OUT OF BAND'}")
  print(f"GATE 3b -- S2 @8192 = {t8192:.3f} ms vs {PRED_8192[0]}-"
        f"{PRED_8192[1]} +-30% [{lo:.2f},{hi:.2f}]: "
        f"{'PASS' if ok_b else 'OUT OF BAND'}")
  print(f"\n  headline: blockdiag vs causal -- @2048 {r2048:.3f}x, "
        f"@8192 {t8192 / results['S0@8192'][1]:.3f}x "
        f"({results['S0@8192'][1]:.2f} -> {t8192:.2f} ms)")
  return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
  sys.exit(main())
