"""P19.6 -- how long does one compile actually take?

P19.4 settled the compile COUNT (one per distinct grid_width, <= L_max/block,
one-off).  It never measured what a compile costs, and that number decides
whether the option-(A) plumbing is worth doing: if a compile is seconds, the
prototype's <=22 compiles are irrelevant; if it is minutes, (A) is mandatory.

Measured here, on TPU, for every grid_width 1..8:
  attn   model_lib.Attention  fwd+bwd   (kernel + q/k/v/o projections)
  layer  model_lib.DecoderLayer fwd+bwd (the above + MLP + norms)
plus the causal baseline that today's code compiles once.

Cold compile is timed as `jax.jit(f).lower(*args).compile()`, which is where
XLA runs.  The negative control checks the other side: calling an already-traced
jit again must NOT recompile (cache size flat, wall time back at run-time
scale).  Without that control a "compile time" could just be run time.

HONEST SCOPE: this is ONE layer.  A real train step compiles 28 of them plus
lm_head and the optimizer, so the totals here are a LOWER BOUND on the real
cold-start cost, never a measurement of it.
"""

import statistics
import sys
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

BUDGET = 2048
WIDTHS = tuple(range(1, BUDGET // BLOCK + 1))   # 1..8


def blockdiag(seq_len, seg_lens):
  pos = np.arange(seq_len)
  seg = np.zeros(seq_len, dtype=np.int64)
  p = 0
  for i, L in enumerate(seg_lens, 1):
    seg[p:p + L] = i
    p += L
  return (pos[None, :] <= pos[:, None]) & (seg[:, None] == seg[None, :]) & (
      seg[:, None] > 0)


def template_for_width(w):
  """A layout whose longest segment is exactly w blocks (so grid_width == w)."""
  seg = w * BLOCK
  n_full, rest = divmod(BUDGET, seg)
  t = [seg] * n_full + ([rest] if rest else [])
  return t


def cold_compile_seconds(fn, args):
  """Time the XLA compile of a freshly lowered program."""
  lowered = jax.jit(fn).lower(*args)
  t0 = time.perf_counter()
  lowered.compile()
  return time.perf_counter() - t0


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: TPU only")
    return 2
  print(f"persistent compilation cache dir = "
        f"{jax.config.jax_compilation_cache_dir}")

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16

  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex = pack(examples, BUDGET, 4, 8, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  x = jax.random.normal(
      jax.random.PRNGKey(0), (*shape, cfg.embed_dim), jnp.bfloat16)
  rngs = nnx.Rngs(params=0)
  attn = model_lib.Attention(config=cfg, rngs=rngs)
  layer = model_lib.DecoderLayer(config=cfg, rngs=rngs)
  args = (x, pos, attn_mask, seg)
  print(f"packed {tuple(shape)}, num_layers in config = {cfg.num_layers}\n")

  def make_fn(module, template):
    def f(x, pos, mask, seg):
      model_lib._SPLASH_SEGMENT_TEMPLATE = template  # noqa: SLF001
      def loss(x):
        _, out = module(x, pos, None, mask, seg)
        return jnp.sum(out.astype(jnp.float32))
      return jax.grad(loss)(x)
    return f

  # ---- GATE 1: negative control, FIRST ------------------------------------
  print("GATE 1 -- negative control: a second call must NOT recompile")
  f = jax.jit(make_fn(attn, template_for_width(4)))
  t0 = time.perf_counter()
  jax.block_until_ready(f(*args))
  first = time.perf_counter() - t0
  n1 = f._cache_size()
  reps = []
  for _ in range(5):
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    reps.append(time.perf_counter() - t0)
  n2 = f._cache_size()
  warm = statistics.median(reps)
  ok_ctl = (n2 == n1) and (warm < first / 10)
  print(f"  first call {first*1000:8.1f} ms (compile+run), warm median "
        f"{warm*1000:8.1f} ms, cache {n1}->{n2}")
  print(f"  {'PASS' if ok_ctl else 'FAIL'} -- warm must be <1/10 of first and "
        f"cache must not grow")
  if not ok_ctl:
    print("\nVOID: cannot separate compile from run; no numbers follow.")
    return 1
  print()

  # ---- cold compiles -------------------------------------------------------
  results = {}
  print(f"{'case':<26}{'grid_w':>8}{'attn compile s':>16}"
        f"{'layer compile s':>17}")
  causal_a = cold_compile_seconds(make_fn(attn, None), args)
  causal_l = cold_compile_seconds(make_fn(layer, None), args)
  results["causal"] = (None, causal_a, causal_l)
  print(f"{'causal (today)':<26}{'8':>8}{causal_a:>16.2f}{causal_l:>17.2f}")

  for w in WIDTHS:
    t = template_for_width(w)
    a = cold_compile_seconds(make_fn(attn, tuple(t)), args)
    l = cold_compile_seconds(make_fn(layer, tuple(t)), args)
    results[f"w{w}"] = (t, a, l)
    print(f"{str(t)[:24]:<26}{w:>8}{a:>16.2f}{l:>17.2f}")

  want = 1 + len(WIDTHS)
  if len(results) != want:
    print(f"\nINCONCLUSIVE: {len(results)}/{want} cases compiled")
    return 2
  print(f"\nGATE 2 -- completeness: {len(results)}/{want} cases "
        f"({2*len(results)} cold compiles): PASS")

  # ---- GATE 3: cold-start cost of each strategy ---------------------------
  per_w_layer = {w: results[f"w{w}"][2] for w in WIDTHS}
  base = results["causal"][2]
  mean_l = statistics.mean(per_w_layer.values())
  strategies = [
      ("today (causal only)", 1, base),
      ("bucketed widths {2,4,8}", 3,
       sum(per_w_layer[w] for w in (2, 4, 8))),
      ("option (A): one per width, <=8", len(WIDTHS),
       sum(per_w_layer.values())),
      ("option (B) prototype: <=22 templates", 22, 22 * mean_l),
  ]
  print(f"\nGATE 3 -- one-layer cold-start cost by strategy")
  print(f"{'strategy':<40}{'compiles':>10}{'1 layer s':>12}"
        f"{'x28 layers s':>14}")
  for name, n, secs in strategies:
    print(f"{name:<40}{n:>10}{secs:>12.1f}{secs*cfg.num_layers:>14.1f}")
  print(f"\n  extra vs today, option (A): "
        f"{sum(per_w_layer.values()) - base:+.1f} s per layer, "
        f"{(sum(per_w_layer.values()) - base)*cfg.num_layers:+.1f} s x28")
  print("  ^ x28 is a LOWER BOUND on a real step (no lm_head, no optimizer, "
        "and\n    XLA does not compile layers independently).  Do not quote it "
        "as measured.")

  # amortisation: how many steps to pay it back
  step_ms = 2.935  # P19.5 measured module-level fwd+bwd for the 'on' arm
  saved_ms = 3.993 - 2.935
  extra_s = (sum(per_w_layer.values()) - base) * cfg.num_layers
  if saved_ms > 0:
    steps = extra_s * 1000 / (saved_ms * cfg.num_layers)
    print(f"\n  amortisation: extra compile {extra_s:.1f}s vs "
          f"{saved_ms:.3f} ms/layer/step saved -> pays back in ~{steps:,.0f} "
          f"steps")
  return 0


if __name__ == "__main__":
  sys.exit(main())
