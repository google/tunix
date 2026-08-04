"""P20.6 -- integration gate on the REAL path, TPU.

Every stage upstream of the model is the production code, not a stand-in:

  TrainExamples with a realistic length spread
    -> rl_utils.pack_sequences(...)          the real packer
    -> next(gen)  ->  list[TrainExample]     the real container
    -> splash_mask.attach(...)               the call the learner makes
    -> common.compute_per_token_logps(...)   the real consumer
    -> Qwen3 on TPU

This exists because the earlier route gate fed a hand-built single
TrainExample. `pack_sequences` actually yields a LIST, `getattr` on a list
returned None, and `attach` silently returned it untouched for a whole
end-to-end run -- no error, no log, no speed-up.

Asserts, in order of what they would catch:

  A  the packer stamps segment_layout, and it MATCHES the input lengths
  B  attach really attaches through the list (stats().attached > 0, skipped 0)
  C  grid_width actually SHRANK vs the causal baseline (else nothing can speed up)
  D  logps are BITWISE IDENTICAL between the two arms
  E  wall-clock for both arms
"""

import argparse
import statistics
import sys
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from tunix.models.qwen3 import model as model_lib
from tunix.rl import common, splash_mask
from tunix.rl import utils as rl_utils

BUDGET, BLOCK, PACK_SIZE = 2048, 256, 4
# A spread that leaves room for several segments per row; if every row held one
# near-budget sequence there would be nothing to drop and no speed-up possible.
LENGTHS = [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150]
PROMPT_FRAC = 0.3


def make_batch(lengths):
  """One TrainExample holding len(lengths) padded rows, as the learner sees."""
  n = len(lengths)
  pmax = max(int(l * PROMPT_FRAC) for l in lengths)
  cmax = max(l - int(l * PROMPT_FRAC) for l in lengths)
  p_ids = np.zeros((n, pmax), np.int32)
  p_mask = np.zeros((n, pmax), np.int32)
  c_ids = np.zeros((n, cmax), np.int32)
  c_mask = np.zeros((n, cmax), np.int32)
  rng = np.random.default_rng(0)
  for i, total in enumerate(lengths):
    p_len = int(total * PROMPT_FRAC)
    c_len = total - p_len
    p_ids[i, :p_len] = rng.integers(1, 1000, p_len)
    p_mask[i, :p_len] = 1
    c_ids[i, :c_len] = rng.integers(1, 1000, c_len)
    c_mask[i, :c_len] = 1
  return common.TrainExample(
      prompt_ids=jnp.asarray(p_ids),
      prompt_mask=jnp.asarray(p_mask),
      completion_ids=jnp.asarray(c_ids),
      completion_mask=jnp.asarray(c_mask),
      advantages=jnp.asarray(rng.normal(size=(n,)), jnp.float32),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )


def timed(fn, iters=6, warmup=2):
  for _ in range(warmup):
    jax.block_until_ready(fn())
  s = []
  for _ in range(iters):
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    s.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(s)


def checksum(a):
  return int(np.asarray(a).view(np.uint32).astype(np.uint64).sum())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--layers", type=int, default=4,
                  help="truncate depth; attention shape per layer is unchanged")
  args = ap.parse_args()

  fails = []
  print(f"jax {jax.__version__} devices={len(jax.devices())} "
        f"{jax.devices()[0].device_kind}")
  print(f"splash_mask.ENABLED = {splash_mask.ENABLED}")
  if not splash_mask.ENABLED:
    raise SystemExit("run me with TUNIX_SPLASH_DOCMASK=1")

  # ---------- A. the REAL packer ----------
  gen = rl_utils.pack_sequences(
      iter([[make_batch(LENGTHS)]]),
      max_token_budget=BUDGET,
      sequences_per_update=None,
      pack_size=PACK_SIZE,
      max_segments_per_packed_row=8,
  )
  chunk = next(gen)
  print(f"\nA. packer yielded {type(chunk).__name__} of "
        f"{len(chunk)} x {type(chunk[0]).__name__}")
  if not isinstance(chunk, list):
    fails.append("packer no longer yields a list -- this gate's premise moved")
  layout = getattr(chunk[0], "segment_layout", None)
  print(f"   segment_layout = {layout}")
  if not layout:
    fails.append("packer did not stamp segment_layout")
  else:
    flat = sorted(l for row in layout for l in row if l > 0)
    # the packer pads the last segment of each row, so stamped >= input
    if not set(sorted(LENGTHS)).issubset(set(flat)):
      print(f"   NOTE stamped {flat}")
      print(f"        input   {sorted(LENGTHS)}")
      fails.append("stamped layout does not contain the input lengths")

  # ---------- B. attach, through the list ----------
  before = splash_mask.stats()
  attached = splash_mask.attach(chunk, seq_len=BUDGET, block=BLOCK,
                                num_heads=model_lib.ModelConfig.qwen3_1p7b().num_heads)
  st = splash_mask.stats()
  kernel = getattr(attached[0], "splash_kernel", None)
  print(f"\nB. attach: {before['attached']}->{st['attached']} attached, "
        f"{before['skipped']}->{st['skipped']} skipped; kernel is "
        f"{'SET' if kernel is not None else 'None'}")
  if kernel is None:
    fails.append("no kernel attached -- the feature is a no-op")
  if st["skipped"] > before["skipped"]:
    fails.append("attach took a silent-skip path")

  # ---------- C. did grid_width actually shrink ----------
  gw = pb = None
  if kernel is not None:
    fi = kernel.fwd_mask_info
    gw = int(np.asarray(fi.data_next).shape[-1])
    pb = int(np.asarray(fi.partial_mask_blocks).shape[0])
    base = BUDGET // BLOCK
    print(f"\nC. grid_width {base} (causal) -> {gw}   partial_mask_blocks={pb}"
          f"   grid slots {base*base} -> {base*gw}  ({gw/base:.3f}x)")
    if gw >= base:
      fails.append(f"grid_width did not shrink ({gw} >= {base}): no speed-up "
                   "is possible for this layout")
    if pb != 1:
      fails.append(f"partial_mask_blocks={pb}, expected 1 after rounding")

  # ---------- D/E. the REAL consumer, both arms ----------
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.num_layers = args.layers
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:PACK_SIZE]).reshape(PACK_SIZE, 1), ("fsdp", "tp"))
  with mesh:
    model = model_lib.Qwen3(cfg, rngs=nnx.Rngs(params=0))
    graphdef, state = nnx.split(model)
    ex = attached[0]

    # `state` is an ARGUMENT, not a closed-over constant: closing over it
    # captured 2.05GB of literals and made every trace a full recompile.
    def run(st, k):
      return common.compute_per_token_logps(
          graphdef, st,
          prompt_tokens=ex.prompt_ids,
          completion_tokens=ex.completion_ids,
          pad_id=0, eos_id=1,
          stop_gradient=True,
          segment_ids=getattr(ex, "segment_ids", None),
          segment_positions=getattr(ex, "segment_positions", None),
          splash_kernel=k,
      )

    # Both jits are built ONCE, outside the timing loop. Building them inside
    # would time the compiler, not the kernel -- an earlier run of this gate
    # reported 34456 ms vs 14 ms for exactly that reason.
    f_off = jax.jit(lambda st: run(st, None))
    f_on = jax.jit(lambda st, k: run(st, k))

    off = jax.device_get(f_off(state))
    on = jax.device_get(f_on(state, kernel))
    same = np.array_equal(np.asarray(off), np.asarray(on))
    print(f"\nD. logps  off={checksum(off)}  on={checksum(on)}  "
          f"BITWISE {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
      d = np.abs(np.asarray(off) - np.asarray(on))
      print(f"   max|diff|={d.max():.3e}  ndiff={(d>0).sum()}/{d.size}")
      fails.append("logps differ -- the mask is not a superset in practice")

    t_off = timed(lambda: f_off(state))
    t_on = timed(lambda: f_on(state, kernel))
    print(f"\nE. {args.layers}-layer fwd (steady state, compile excluded):")
    print(f"     off {t_off:8.2f} ms   on {t_on:8.2f} ms   {t_on/t_off:.3f}x")
    print(f"     per layer: {t_off/args.layers:6.2f} -> {t_on/args.layers:6.2f} ms")
    if t_on > t_off:
      fails.append(f"the doc mask made it SLOWER ({t_on/t_off:.3f}x)")

    # ---------- F. attention ALONE ----------
    # E is dominated by the LM head: [B, 2048, 2048] @ [2048, vocab] is ~5 TFLOP
    # against ~0.07 TFLOP of attention per layer, so a large win on attention
    # barely moves it. This isolates the part the mask can actually change.
    attn = model.layers[0].attn
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (PACK_SIZE, BUDGET, cfg.embed_dim), jnp.bfloat16)
    pos = jnp.asarray(ex.segment_positions) if getattr(
        ex, "segment_positions", None) is not None else jnp.tile(
            jnp.arange(BUDGET)[None], (PACK_SIZE, 1))
    seg = jnp.asarray(ex.segment_ids)
    a_off = jax.jit(lambda v: attn(v, pos, None, None, seg)[1])
    a_on = jax.jit(lambda v, k: attn(v, pos, None, None, seg,
                                     splash_kernel=k)[1])
    o1 = jax.device_get(a_off(x))
    o2 = jax.device_get(a_on(x, kernel))
    asame = np.array_equal(np.asarray(o1), np.asarray(o2))
    ta_off = timed(lambda: a_off(x))
    ta_on = timed(lambda: a_on(x, kernel))
    print(f"\nF. ONE attention block (fwd):")
    print(f"     off {ta_off:8.3f} ms   on {ta_on:8.3f} ms   "
          f"{ta_on/ta_off:.3f}x    bitwise "
          f"{'IDENTICAL' if asame else 'DIFFERENT'}")
    if not asame:
      fails.append("attention output differs between arms")
    saved = (ta_off - ta_on) * args.layers
    print(f"     x{args.layers} layers that is {saved:.1f} ms saved out of "
          f"{t_off:.1f} ms total  ({saved/t_off*100:.1f}% of the step)")
    print(f"     attention share of this fwd: "
          f"{ta_off*args.layers/t_off*100:.1f}%")

  print(f"\nstats: {splash_mask.stats()}")
  print("\nVERDICT:", "PASS" if not fails else "FAIL")
  for x in fails:
    print("  -", x)
  return 0 if not fails else 1


if __name__ == "__main__":
  sys.exit(main())
