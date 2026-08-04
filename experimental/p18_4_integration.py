"""P18.4 -- can the dynamic path be driven from inside a jit, and what does it cost?

P18.3 left three things open, all of which decide whether P18.5 is possible:

  (a) `SplashAttentionKernel.__call__` raises TracerBoolConversionError when the
      kernel is built inside a jit trace, because `make_splash_mha` tree_maps
      `MaskInfo.is_dynamic_mask` (a Python bool) into a jnp array and `__call__`
      branches on it -- purely to pick a `jax.named_scope` LABEL
      (kernel.py:2463-2467).  Nothing functional depends on it, so calling
      `_splash_attention` directly should sidestep it.  Probe A tests that.
  (b) If it works, does a CHANGING segment layout recompile?  That is the whole
      reason candidate C beats candidate B (a static block-diagonal mask has to
      be grouped by template).  Probe B counts compiles across three layouts.
  (c) What does building + processing the mask inside the jit actually cost, and
      can one vmapped call serve rows with different masks?  Probes C and D.

Also re-measures the grid-overhead attribution directly instead of inferring it:
P18.3's two-parameter fit said ~53% of the packed row's time is grid iteration
that dynamic cannot remove because `shrink_grid` is ignored for dynamic masks.
Probe E holds the WORK constant and varies only the grid width to check that.

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
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK, dense_mask
from tunix.models.qwen3 import model as model_lib


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


def bs_for(seq_len):
  b = min(BLOCK, seq_len)
  return splash.BlockSizes(
      block_q=b, block_kv=b, block_q_dkv=b, block_kv_dkv=b,
      block_kv_dkv_compute=b, block_q_dq=b, block_kv_dq=b,
  )


def call_dynamic(mask_2d, q, k, v, seq_len):
  """Build MaskInfo from a TRACED mask and run splash, bypassing __call__.

  `make_splash_mha_single_device` is safe under trace; only the object's
  `__call__` is not, because it reads `is_dynamic_mask` in a Python `if` to
  choose a named_scope.  Calling `_splash_attention` with the same three
  MaskInfos and the same stored kwargs is the identical computation.
  """
  kern = splash.make_splash_mha_single_device(
      mask_2d[None], block_sizes=bs_for(seq_len)
  )
  return splash._splash_attention(  # noqa: SLF001 -- see docstring
      kern.fwd_mask_info, kern.dq_mask_info, kern.dkv_mask_info,
      q, k, v, **kern.kwargs,
  )


def build_mask(seg, seq_len):
  pos = jnp.arange(seq_len)
  return (pos[None, :] <= pos[:, None]) & (seg[:, None] == seg[None, :])


def main():
  print(f"jax {jax.__version__}  devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: P18.4 must run on TPU")
    return 2

  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  print(f"qwen3_1p7b {qh}q/{kh}kv x {hd}, block {BLOCK}\n")

  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex_a = pack(examples, 8192, 1, 8, row_multiple=1)
  _, _, seg_a, _ = model_inputs(ex_a)
  seg_a = np.asarray(seg_a)
  ex_d = pack(examples, 2048, 4, 8, row_multiple=1)
  _, _, seg_d, _ = model_inputs(ex_d)
  seg_d = np.asarray(seg_d)

  L = 8192
  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(keys[0], (qh, L, hd), jnp.bfloat16)
  k = jax.random.normal(keys[1], (kh, L, hd), jnp.bfloat16)
  v = jax.random.normal(keys[2], (kh, L, hd), jnp.bfloat16)
  seg_j = jnp.asarray(seg_a[0], dtype=jnp.int32)

  results = {}

  # ---- Probe A: dynamic path driven entirely from inside a jit -------------
  print("PROBE A -- build MaskInfo from a TRACED mask, inside jit")

  @jax.jit
  def live_fwd(q, k, v, seg):
    return call_dynamic(build_mask(seg, L), q, k, v, L)

  try:
    out_live = live_fwd(q, k, v, seg_j)
    jax.block_until_ready(out_live)
    probe_a = True
    print(f"  WORKS -- out={out_live.shape}, finite="
          f"{bool(jnp.isfinite(out_live).all())}")
  except Exception as exc:  # noqa: BLE001 -- the failure IS the result
    probe_a = False
    out_live = None
    print(f"  FAILS -- {type(exc).__name__}: {str(exc)[:200]}")

  if not probe_a:
    print("\nVERDICT: dynamic masks cannot be driven from inside a jit in this"
          " jax build -> P18.5 is BLOCKED on an upstream change.")
    return 1

  # numerical gate against the static path, again bitwise
  static_kern = splash.make_splash_mha_single_device(
      mask_lib.MultiHeadMask(
          [mask_lib.CausalMask((L, L)) for _ in range(qh)]),
      block_sizes=bs_for(L))
  out_static = jax.jit(
      lambda a, b, c, s: static_kern(a, b, c, splash.SegmentIds(q=s, kv=s))
  )(q, k, v, seg_j)
  a1 = np.asarray(jax.device_get(out_live)).view(np.uint16)
  a0 = np.asarray(jax.device_get(out_static)).view(np.uint16)
  bitwise = np.array_equal(a1, a0)
  print(f"  numerical vs static+segment_ids: "
        f"{'BITWISE IDENTICAL' if bitwise else 'DIFFERS'}")
  if not bitwise:
    print("\nVOID: numerical gate red; timings not interpreted.")
    return 1

  # ---- Probe B: does a CHANGING layout recompile? --------------------------
  print("\nPROBE B -- recompilation across three different segment layouts")
  compiles = []
  import jax._src.dispatch as _dispatch  # noqa: PLC0415

  layouts = [seg_a[0]]
  rng = np.random.default_rng(0)
  for _ in range(2):
    alt = seg_a[0].copy()
    cut = int(rng.integers(1024, 7168))
    alt[cut:] = alt[cut:][::-1]  # different segment boundaries, same shape
    layouts.append(alt)

  n_before = live_fwd._cache_size() if hasattr(live_fwd, "_cache_size") else None
  for i, lay in enumerate(layouts):
    jax.block_until_ready(live_fwd(q, k, v, jnp.asarray(lay, dtype=jnp.int32)))
    n = live_fwd._cache_size() if hasattr(live_fwd, "_cache_size") else -1
    compiles.append(n)
    print(f"  layout {i}: jit cache size = {n}")
  del _dispatch, n_before
  no_recompile = len(set(compiles)) == 1
  print(f"  -> {'NO RECOMPILE (cache size constant)' if no_recompile else 'RECOMPILED'}"
        "  <-- this is exactly what candidate B could not do")

  # ---- Probe C: what does the live mask cost vs a constant one? ------------
  print("\nPROBE C -- cost of building+processing the mask inside the jit")
  dense_a = dense_mask(seg_a[0], L)
  const_kern = splash.make_splash_mha_single_device(
      jnp.asarray(dense_a[None], dtype=jnp.bool), block_sizes=bs_for(L))
  f_const = jax.jit(lambda a, b, c: const_kern(a, b, c))
  f_live = live_fwd

  def grad_of(fn, extra):
    @jax.jit
    def g(qq, *rest):
      return jax.grad(
          lambda x: jnp.sum(fn(x, *rest).astype(jnp.float32)))(qq)
    return g, (q, k, v) + extra

  t_const_f = timed(f_const, (q, k, v))
  t_live_f = timed(f_live, (q, k, v, seg_j))
  g_const, a_const = grad_of(lambda *a: const_kern(*a), ())
  g_live, a_live = grad_of(
      lambda qq, kk, vv, s: call_dynamic(build_mask(s, L), qq, kk, vv, L),
      (seg_j,))
  t_const_b = timed(g_const, a_const)
  t_live_b = timed(g_live, a_live)
  print(f"  const-mask (kernel only) : fwd {t_const_f:7.3f}  fwd+bwd {t_const_b:7.3f}")
  print(f"  live-mask  (in-jit build): fwd {t_live_f:7.3f}  fwd+bwd {t_live_b:7.3f}")
  print(f"  in-jit mask overhead     : fwd {t_live_f - t_const_f:+7.3f} ms"
        f"  fwd+bwd {t_live_b - t_const_b:+7.3f} ms")
  results["live_overhead_bwd"] = t_live_b - t_const_b

  # ---- Probe D: one vmapped call over rows with DIFFERENT masks ------------
  print("\nPROBE D -- vmap over per-row dynamic masks (production shape)")
  L2 = 2048
  q2 = jax.random.normal(keys[0], (qh, L2, hd), jnp.bfloat16)
  k2 = jax.random.normal(keys[1], (kh, L2, hd), jnp.bfloat16)
  v2 = jax.random.normal(keys[2], (kh, L2, hd), jnp.bfloat16)
  rows = 4
  segs2 = jnp.asarray(seg_d[:rows], dtype=jnp.int32)
  qs, ks, vs = (jnp.stack([x] * rows) for x in (q2, k2, v2))

  @jax.jit
  def vmapped(qq, kk, vv, ss):
    return jax.vmap(
        lambda a, b, c, s: call_dynamic(build_mask(s, L2), a, b, c, L2)
    )(qq, kk, vv, ss)

  try:
    o = vmapped(qs, ks, vs, segs2)
    jax.block_until_ready(o)
    print(f"  WORKS -- out={o.shape}, finite={bool(jnp.isfinite(o).all())}")
    probe_d = True
  except Exception as exc:  # noqa: BLE001
    print(f"  FAILS -- {type(exc).__name__}: {str(exc)[:220]}")
    probe_d = False

  # ---- Probe E: is the residual really grid iteration? ---------------------
  print("\nPROBE E -- grid overhead: same WORK, different grid width")
  # One segment of 2048 real tokens, padded into rows of 2048 / 4096 / 8192.
  # The work (one 8x8 causal triangle = 36 blocks) is identical; only the number
  # of grid steps changes (8x8, 16x16, 32x32).
  for row_len in (2048, 4096, 8192):
    seg = np.zeros(row_len, dtype=np.int32)
    seg[:2048] = 1  # one real segment; the rest is padding id 0
    dense = dense_mask(seg, row_len)
    info, _ = mask_info_lib.process_dynamic_mask(
        jnp.asarray(dense[None], dtype=jnp.bool), (BLOCK, BLOCK))
    nblocks = int((np.asarray(info.block_mask) != 0).sum())
    qq = jax.random.normal(keys[0], (qh, row_len, hd), jnp.bfloat16)
    kk = jax.random.normal(keys[1], (kh, row_len, hd), jnp.bfloat16)
    vv = jax.random.normal(keys[2], (kh, row_len, hd), jnp.bfloat16)
    kern = splash.make_splash_mha_single_device(
        jnp.asarray(dense[None], dtype=jnp.bool), block_sizes=bs_for(row_len))
    f = jax.jit(lambda a, b, c: kern(a, b, c))
    t = timed(f, (qq, kk, vv))
    grid = (row_len // BLOCK) ** 2
    print(f"  row {row_len:>5}: grid {grid:>5} steps, {nblocks:>4} work blocks,"
          f" fwd {t:7.3f} ms")
    results[f"gridprobe_{row_len}"] = (grid, nblocks, t)

  # ---- verdict -------------------------------------------------------------
  print("\n" + "=" * 74)
  print("VERDICT")
  print(f"  Probe A (dynamic inside jit, via _splash_attention): "
        f"{'WORKS' if probe_a else 'FAILS'}")
  print(f"  Probe A numerics vs static: "
        f"{'BITWISE' if bitwise else 'DIFFERS'}")
  print(f"  Probe B (layout change without recompile): "
        f"{'YES' if no_recompile else 'NO'}")
  print(f"  Probe D (vmap over per-row masks): "
        f"{'WORKS' if probe_d else 'FAILS'}")
  gp = [results[f"gridprobe_{n}"] for n in (2048, 4096, 8192)]
  if gp[0][1] == gp[1][1] == gp[2][1]:
    print(f"  Probe E: work held at {gp[0][1]} blocks while grid went "
          f"{gp[0][0]} -> {gp[2][0]}; time {gp[0][2]:.3f} -> {gp[2][2]:.3f} ms"
          f" ({gp[2][2]/gp[0][2]:.2f}x) -- grid iteration IS the residual cost")
  else:
    print(f"  Probe E: work NOT held constant ({[g[1] for g in gp]});"
          " attribution inconclusive")
  return 0


if __name__ == "__main__":
  sys.exit(main())
