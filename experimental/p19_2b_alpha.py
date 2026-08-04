"""P19.2b -- measure alpha (attention's share of a decoder layer) on production sharding.

P19.2a reduced the whole decision to one unmeasured number.  Every design's
step-level payoff is `alpha * (attention ratio) + (1 - alpha)`, and alpha --
attention's share of the step -- has never been measured in this project.  With
alpha small, even a perfect attention kernel is not worth writing.

This times, on the production mesh (fsdp x tp, one packed row per chip, the
layout P18.5 used):

  attn_only   model_lib.Attention          -- kernel + q/k/v/o projections
  full_layer  the whole DecoderLayer       -- the above + MLP + norms

alpha = attn_only / full_layer, per arm.  Also re-runs the three P19.1 arms
under PRODUCTION SHARDING rather than a single-device vmap, which is the open
caveat from P19.1 (candidate C looked 12.8% SLOWER there because the per-row
mask build was paid 4x on one chip, while P18.5 -- one row per chip -- measured
it 15% faster).

Note the honest ceiling: this is alpha within a TRANSFORMER LAYER.  A full
training step also carries the loss/lm_head, the optimizer and the rollout, so
the step-level alpha is bounded ABOVE by what this reports.
"""

import statistics
import sys
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib


def timed(fn, args, iters=15, warmup=3, inner=3):
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


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: TPU only")
    return 2

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()

  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16
  print(f"mesh fsdp4 x tp1, qwen3_1p7b, block {BLOCK}")

  examples, total = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex = pack(examples, 2048, 4, 8, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  print(f"packed {tuple(shape)} (one row per chip), "
        f"segs/row={[int(r.max()) for r in np.asarray(seg)]}\n")

  rngs = nnx.Rngs(params=0)
  attn = model_lib.Attention(config=cfg, rngs=rngs)
  layer = model_lib.DecoderLayer(config=cfg, rngs=rngs)
  x = jax.random.normal(
      jax.random.PRNGKey(0), (*shape, cfg.embed_dim), jnp.bfloat16)

  @jax.jit
  def attn_fwd_bwd(x, pos, mask, seg):
    def loss(x):
      _, out = attn(x, pos, None, mask, seg)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  @jax.jit
  def layer_fwd_bwd(x, pos, mask, seg):
    def loss(x):
      _, out = layer(x, pos, None, mask, seg)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  args = (x, pos, attn_mask, seg)
  t_attn = timed(attn_fwd_bwd, args)
  t_layer = timed(layer_fwd_bwd, args)
  if t_layer <= t_attn:
    print(f"INCONCLUSIVE: layer ({t_layer:.3f}) not slower than attention "
          f"({t_attn:.3f}); the two are not measuring different things")
    return 2

  alpha = t_attn / t_layer
  print("MEASURED (fwd+bwd, median of 15, production sharding)")
  print(f"  attention module (kernel + qkvo proj): {t_attn:7.3f} ms")
  print(f"  full decoder layer (+ MLP + norms)   : {t_layer:7.3f} ms")
  print(f"  => alpha (attention share of a layer) = {alpha:.3f}")
  print("     ^ this is an UPPER BOUND on the step-level alpha: a real step "
        "also carries\n       lm_head/loss, the optimizer and the rollout.\n")

  # --- what each design is worth at this alpha -----------------------------
  # attention ratios from P19.2a (exact block counting), per distribution.
  designs = {
      "candidate C (universal, no token cost)": {
          "uniform 700-950": 0.823, "uniform 100-2048": 0.925,
          "bimodal 70/30": 0.922, "uniform 1024": 0.791,
          "all at L_max": 1.000, "near-cap": 1.013, "bimodal 50/50": 0.937},
      "segment-grid kernel (P19.3 ceiling)": {
          "uniform 700-950": 0.416, "uniform 100-2048": 0.740,
          "bimodal 70/30": 0.840, "uniform 1024": 0.526,
          "all at L_max": 1.000, "near-cap": 1.000, "bimodal 50/50": 0.858},
  }
  print("=" * 74)
  print(f"STEP-LEVEL PAYOFF at the measured alpha = {alpha:.3f}")
  print("  (layer-level; the true step-level number is SMALLER)")
  print("=" * 74)
  for label, ratios in designs.items():
    gains = {k: (1 - alpha) + alpha * v for k, v in ratios.items()}
    best = 1 - min(gains.values())
    worst = 1 - max(gains.values())
    print(f"\n  {label}")
    for k, v in ratios.items():
      print(f"    {k:<20} attention {v:.3f}x -> layer "
            f"{gains[k]:.3f}x  ({1-gains[k]:+.1%})")
    print(f"    => range {worst:+.1%} .. {best:+.1%}")

  print("\n" + "=" * 74)
  ceiling = 1 - ((1 - alpha) + alpha * 0.416)
  print(f"VERDICT: even a PERFECT attention kernel (attention -> 0) would save "
        f"at most\n  {alpha:.1%} of a layer.  The best case actually on the "
        f"table (segment grid on a\n  narrow distribution) is {ceiling:.1%}.  "
        "Weigh that against several days of\n  Pallas forward+backward work "
        "before starting P19.3.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
