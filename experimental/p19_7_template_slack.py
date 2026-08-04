"""P19.7 -- how strict does the template have to be?

The whole recompilation story rests on a claim I have asserted but never tested:
a template only has to be a SUPERSET of the row's true layout, because the exact
masking is still done at runtime by segment_ids.  If that holds, one COARSE
template can serve many different real layouts, and the compile count is set by
the number of standard templates we choose -- not by the data.

Tested here, at budget 2048 (8 blocks of 256):

  A. correctness/slack -- run a row whose real layout is (1024, 512, 512) under
     three masks: its EXACT template, the COARSER (1024, 1024), and plain
     causal.  All three are supersets of the true mask, so with the same
     segment_ids all three must give the SAME answer.  If the coarse template
     differs from the exact one, the superset argument is wrong and the whole
     bucketing scheme collapses.

  B. what slack costs -- block counts for the same row under each mask.

  C. the standard-template ladder -- for a fixed set of 4 templates
     {(2048,), (1024,1024), (512,)*4, (256,)*8}, which rows can each serve and
     what is the grid_width, i.e. how many programs a bucketed scheme needs.

A negative control is included: a template that is NOT a superset (segments
straddling a slot boundary) must give a DIFFERENT answer -- otherwise the test
cannot tell supersets from anything else.
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
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

BUDGET = 2048


def layout_arrays(seq_len, seg_lens):
  seg = np.zeros(seq_len, dtype=np.int64)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    seg[p:p + L] = i
    p += L
  return seg


def blockdiag(seq_len, seg_lens):
  pos = np.arange(seq_len)
  seg = layout_arrays(seq_len, seg_lens)
  return (pos[None, :] <= pos[:, None]) & (seg[:, None] == seg[None, :]) & (
      seg[:, None] > 0)


def counts(dense):
  info, _ = mask_info_lib.process_mask(
      mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)]), (BLOCK, BLOCK))
  dn = np.asarray(info.data_next)
  return dn.shape[-1], dn.shape[1] * dn.shape[-1], int(
      (np.asarray(info.block_mask) != 0).sum())


def is_superset(template_dense, true_dense):
  """A template is usable iff it allows everything the true mask allows."""
  return bool((true_dense & ~template_dense).sum() == 0)


def bs_for(n):
  b = min(BLOCK, n)
  return splash.BlockSizes(
      block_q=b, block_kv=b, block_q_dkv=b, block_kv_dkv=b,
      block_kv_dkv_compute=b, block_q_dq=b, block_kv_dq=b)


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  on_tpu = jax.devices() and jax.devices()[0].platform == "tpu"

  TRUE = [1024, 512, 512]                       # the row's real layout
  true_dense = blockdiag(BUDGET, TRUE)
  seg_true = layout_arrays(BUDGET, TRUE)

  masks = {
      "exact   (1024,512,512)": blockdiag(BUDGET, TRUE),
      "coarser (1024,1024)": blockdiag(BUDGET, [1024, 1024]),
      "coarsest(2048,)": blockdiag(BUDGET, [2048]),
      "causal (today)": np.arange(BUDGET)[None, :] <= np.arange(BUDGET)[:, None],
      # negative control: slots that CUT a real segment in half
      "NEGCTL  (768,768,512)": blockdiag(BUDGET, [768, 768, 512]),
  }

  print(f"\ntrue row layout = {tuple(TRUE)}   budget {BUDGET}, block {BLOCK}\n")
  print(f"{'mask':<26}{'superset?':>11}{'grid_w':>8}{'grid':>7}{'work':>7}")
  info = {}
  for name, m in masks.items():
    sup = is_superset(m, true_dense)
    w, g, k = counts(m)
    info[name] = (sup, w, g, k, m)
    print(f"{name:<26}{str(sup):>11}{w:>8}{g:>7}{k:>7}")

  if not on_tpu:
    print("\n(CPU run: block counts only; numerics need TPU)")
    return 0

  # ---- numerics: every SUPERSET must give the identical answer ------------
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(keys[0], (qh, BUDGET, hd), jnp.bfloat16)
  k = jax.random.normal(keys[1], (kh, BUDGET, hd), jnp.bfloat16)
  v = jax.random.normal(keys[2], (kh, BUDGET, hd), jnp.bfloat16)
  s = jnp.asarray(seg_true, dtype=jnp.int32)   # SAME segment_ids everywhere

  def run(dense):
    kern = splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)] * qh),
        block_sizes=bs_for(BUDGET))
    f = jax.jit(lambda a, b, c, t: kern(
        a, b, c, splash.SegmentIds(q=t, kv=t)))
    return np.asarray(jax.device_get(f(q, k, v, s)))

  ref = run(info["exact   (1024,512,512)"][4])
  print(f"\nnumerics vs the EXACT template (same segment_ids for every arm):")
  ok = True
  for name, (sup, w, g, kk, m) in info.items():
    if name.startswith("exact"):
      continue
    out = run(m)
    same = np.array_equal(out.view(np.uint16), ref.view(np.uint16))
    verdict = "BITWISE SAME" if same else "DIFFERS"
    expect = "expected SAME" if sup else "expected DIFFERS (negative control)"
    good = (same == sup)
    ok &= good
    print(f"  {name:<26}{verdict:<14}{expect:<38}"
          f"{'OK' if good else '<-- FAIL'}")

  # ---- C: standard-template ladder ---------------------------------------
  print("\nstandard-template ladder (a bucketed scheme's programs):")
  ladder = [(2048,), (1024, 1024), (512,) * 4, (256,) * 8]
  for t in ladder:
    w, g, kk = counts(blockdiag(BUDGET, list(t)))
    print(f"  {str(t)[:26]:<28} grid_w={w}  grid={g:>4}  work={kk:>3}")
  print(f"  => {len(ladder)} templates -> {len({counts(blockdiag(BUDGET,list(t)))[0] for t in ladder})} "
        f"distinct grid_widths -> that many programs")

  print("\n" + "=" * 70)
  print(f"VERDICT: {'PASS' if ok else 'FAIL'} -- "
        f"{'supersets agree bitwise and the negative control differs'
           if ok else 'the superset argument does not hold as stated'}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
