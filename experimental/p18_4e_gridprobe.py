"""P18.4e -- is the residual cost really grid iteration?  (corrected probe)

P18.3's two-parameter fit said ~53% of a packed [1, 8192] row's time is grid
iteration that the dynamic path cannot remove, because `shrink_grid` is
documented as "currently ignored" for dynamic masks (mask_info.py:354-358).
This measures that directly: hold the WORK constant and vary only the grid.

The first attempt (P18.4 Probe E) failed its own check and said so: it built the
mask from segment ids, and padding carries segment id 0, so pad attends pad
(kernel.py:679 is a bare `q_ids == kv_ids`) and the work grew 36 -> 72 -> 336
instead of staying at 36.  The fix is to build the dense mask directly --
causal AND both positions inside the one real segment -- so the padding
contributes no blocks at all and the work is pinned by construction.
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
    splash_attention_mask_info as mask_info_lib,
)

from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

REAL = 2048  # the one real segment, identical in every arm


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


def pinned_mask(row_len):
  """Causal, and allowed ONLY inside [0, REAL).  Work is 36 blocks by design."""
  pos = np.arange(row_len)
  causal = pos[None, :] <= pos[:, None]
  inside = (pos < REAL)
  return causal & inside[:, None] & inside[None, :]


def main():
  print(f"jax {jax.__version__}  devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: TPU only")
    return 2
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  qh, kh, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  keys = jax.random.split(jax.random.PRNGKey(0), 3)

  print(f"work pinned to one {REAL}-token segment "
        f"({REAL // BLOCK}x{REAL // BLOCK} causal triangle) in every arm\n")
  print(f"{'row':>7}{'grid steps':>12}{'work blocks':>13}{'fwd ms':>10}"
        f"{'fwd+bwd ms':>13}")
  rows = []
  for row_len in (2048, 4096, 8192):
    dense = pinned_mask(row_len)
    info, _ = mask_info_lib.process_dynamic_mask(
        jnp.asarray(dense[None], dtype=jnp.bool), (BLOCK, BLOCK))
    nblocks = int((np.asarray(info.block_mask) != 0).sum())
    q = jax.random.normal(keys[0], (qh, row_len, hd), jnp.bfloat16)
    k = jax.random.normal(keys[1], (kh, row_len, hd), jnp.bfloat16)
    v = jax.random.normal(keys[2], (kh, row_len, hd), jnp.bfloat16)
    kern = splash.make_splash_mha_single_device(
        jnp.asarray(dense[None], dtype=jnp.bool), block_sizes=bs_for(row_len))
    f = jax.jit(lambda a, b, c: kern(a, b, c))

    @jax.jit
    def fb(a, b, c):
      return jax.grad(lambda x: jnp.sum(f(x, b, c).astype(jnp.float32)))(a)

    t_f, t_b = timed(f, (q, k, v)), timed(fb, (q, k, v))
    grid = (row_len // BLOCK) ** 2
    rows.append((row_len, grid, nblocks, t_f, t_b))
    print(f"{row_len:>7}{grid:>12}{nblocks:>13}{t_f:>10.3f}{t_b:>13.3f}")

  print()
  if len({r[2] for r in rows}) != 1:
    print(f"INCONCLUSIVE: work not held constant ({[r[2] for r in rows]})")
    return 2
  print(f"  work held constant at {rows[0][2]} blocks in all "
        f"{len(rows)} arms: PASS")

  # Fit T = a*grid + b*work with work fixed -> a from the slope.
  (g0, t0), (g2, t2) = (rows[0][1], rows[0][4]), (rows[-1][1], rows[-1][4])
  a = (t2 - t0) / (g2 - g0)
  b_total = t0 - a * g0
  print(f"  fwd+bwd grew {t0:.3f} -> {t2:.3f} ms ({t2/t0:.2f}x) while the work "
        f"never changed")
  print(f"  => grid cost  a = {a*1000:.4f} us/step; "
        f"work+fixed cost at grid {g0} = {b_total:.3f} ms")
  print(f"  => at row 8192 the grid accounts for "
        f"{a*g2/t2:.0%} of fwd+bwd time")
  print("\n  This is the cost `shrink_grid` would remove; it is ignored for "
        "dynamic masks\n  (mask_info.py:354-358), which is why P18.3 measured "
        "0.600x instead of the\n  0.152x block ratio.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
