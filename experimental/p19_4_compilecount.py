"""P19.4 -- how many COMPILES does the template scheme actually cost?

The obvious objection to static template masks is recompilation: a new layout
every step means a new mask means a new trace.  Quantizing lengths bounds the
number of layouts, but for FrozenLake's budget that bound is astronomically
large (10^25 partitions), so "bounded" is not by itself an answer.

The real question is narrower.  A `SplashAttentionKernel` is a registered pytree
whose leaves are the MaskInfo arrays, so it can be passed as a jit ARGUMENT --
and jit caches on shape/dtype, not on value.  So two templates recompile only if
their MaskInfo SHAPES differ.  The shape probe says only `grid_width` varies,
and `grid_width == max_segment_len / block`.  If that holds under jit, the
compile count is bounded by `L_max / block` (8 for gsm8k, 16 for FrozenLake),
regardless of how many layouts exist.

This measures it directly:
  A. many templates SHARING a grid_width  -> expect ONE compile
  B. templates with DIFFERENT grid_widths -> expect one compile each
  C. pad every template up to a common grid_width -> expect ONE compile total
     (the fallback that trades a little grid for a single program)

Run on the v4-8 with the verified image.
"""

import sys

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

BUDGET = 2048


def blockdiag(seq_len, seg_lens):
  pos = np.arange(seq_len)
  seg = np.zeros(seq_len, dtype=np.int64)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    seg[p:p + L] = i
    p += L
  return (pos[None, :] <= pos[:, None]) & (seg[:, None] == seg[None, :]) & (
      seg[:, None] > 0)


def seg_ids(seq_len, seg_lens):
  s = np.zeros(seq_len, dtype=np.int32)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    s[p:p + L] = i
    p += L
  return s


def bs_for(n):
  b = min(BLOCK, n)
  return splash.BlockSizes(
      block_q=b, block_kv=b, block_q_dkv=b, block_kv_dkv=b,
      block_kv_dkv_compute=b, block_q_dq=b, block_kv_dq=b)


def make_kernel(seg_lens, qh, pad_width=None):
  """Kernel for one template.  `pad_width` forces a wider grid (fewer programs).

  Padding is done by OR-ing in extra allowed blocks on the diagonal band, which
  keeps the mask a SUPERSET of the true one -- segment_ids still does the exact
  masking at runtime, so numerics are unchanged.
  """
  dense = blockdiag(BUDGET, seg_lens)
  if pad_width is not None:
    nb = BUDGET // BLOCK
    bm = np.zeros((nb, nb), dtype=bool)
    for i in range(nb):
      lo = max(0, i - pad_width + 1)
      bm[i, lo:i + 1] = True
    expand = np.kron(bm, np.ones((BLOCK, BLOCK), dtype=bool))
    pos = np.arange(BUDGET)
    dense = dense | (expand & (pos[None, :] <= pos[:, None]))
  return splash.make_splash_mha_single_device(
      mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)] * qh),
      block_sizes=bs_for(BUDGET))


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: TPU only")
    return 2
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(keys[0], (qh, BUDGET, hd), jnp.bfloat16)
  k = jax.random.normal(keys[1], (kh, BUDGET, hd), jnp.bfloat16)
  v = jax.random.normal(keys[2], (kh, BUDGET, hd), jnp.bfloat16)

  # The kernel travels as a jit ARGUMENT, exactly as model.py already passes it
  # through shard_map's in_specs.
  @jax.jit
  def run(kernel, q, k, v, s):
    return kernel(q, k, v, splash.SegmentIds(q=s, kv=s))

  def sweep(name, templates, pad_width=None):
    run._clear_cache()
    widths = []
    for t in templates:
      kern = make_kernel(list(t), qh, pad_width=pad_width)
      w = int(np.asarray(kern.fwd_mask_info.data_next).shape[-1])
      widths.append(w)
      s = jnp.asarray(seg_ids(BUDGET, list(t)))
      jax.block_until_ready(run(kern, q, k, v, s))
    n = run._cache_size()
    print(f"  {name:<46} templates={len(templates):>2}  "
          f"grid_widths={sorted(set(widths))}  ->  COMPILES = {n}")
    return n, sorted(set(widths))

  print(f"\nbudget {BUDGET}, block {BLOCK}, qh={qh}\n")
  print("A. templates that SHARE a grid_width (max segment = 1024 -> width 4)")
  same = [(1024, 1024), (1024, 512, 512), (1024, 512, 256, 256),
          (1024, 256, 256, 256, 256)]
  n_a, w_a = sweep("same max-segment, different layouts", same)

  print("\nB. templates with DIFFERENT grid_widths")
  diff = [(2048,), (1792, 256), (1536, 512), (1024, 1024),
          (512, 512, 512, 512), (256,) * 8]
  n_b, w_b = sweep("one per distinct max-segment length", diff)

  print("\nC. same templates, grid_width PADDED to a common 8")
  n_c, w_c = sweep("all padded to width 8", diff, pad_width=8)

  print("\n" + "=" * 74)
  print("VERDICT")
  ok_a = n_a == 1
  ok_b = n_b == len(w_b)
  ok_c = n_c == 1
  print(f"  A: {len(same)} layouts sharing width {w_a} -> {n_a} compile(s)   "
        f"{'PASS (compiles track SHAPE, not layout)' if ok_a else 'FAIL'}")
  print(f"  B: widths {w_b} -> {n_b} compile(s)   "
        f"{'PASS (one per distinct width)' if ok_b else 'FAIL'}")
  print(f"  C: all padded to width 8 -> {n_c} compile(s)   "
        f"{'PASS (single program fallback works)' if ok_c else 'FAIL'}")
  if ok_a and ok_b:
    print(f"\n  => compile count is bounded by the number of distinct "
          f"grid_widths\n     = distinct quantized MAX SEGMENT lengths "
          f"<= L_max/block, NOT by the\n     number of layouts.  gsm8k: <= 8.  "
          f"FrozenLake (L_max 4096): <= 16.")
  return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
  sys.exit(main())
