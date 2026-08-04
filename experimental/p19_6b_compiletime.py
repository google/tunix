"""P19.6b -- compile time, with the ordering confound removed.

P19.6 reported attn ~32 s and layer ~3.8 s for every grid_width.  A DecoderLayer
CONTAINS an Attention, so it cannot be 8x cheaper; the run was confounded.  For
each width it compiled attn first and layer second, and the Pallas/Mosaic kernel
is cached by signature -- so the attn call paid the Mosaic compile and the layer
call, using the SAME mask, hit that cache.  P19.6's GATE 3 table is therefore
void.

This re-measures with the confound removed and turns the hypothesis into a
test:

  H1  the Mosaic kernel compile is the dominant term and is shared
      => compiling ONLY the layer (no preceding attn) costs ~the attn number
  H2  a second module reusing the SAME mask is cheap
      => after H1's layer compile, an attn compile at the same width is cheap
  H3  compile cost does not depend on grid_width
      => the spread across widths stays small (P19.6 already suggested this)

Each width runs in a FRESH PROCESS so no cross-width cache can leak; the driver
passes --width and this script measures exactly one.

HONEST SCOPE: still one layer.  But if H1 holds, the Mosaic term is paid once
per width for the whole model, not once per layer, which is the number that
actually matters for cold start.
"""

import argparse
import sys
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK
from tunix.models.qwen3 import model as model_lib

BUDGET = 2048


def template_for_width(w):
  if w is None:
    return None
  seg = w * BLOCK
  n_full, rest = divmod(BUDGET, seg)
  return tuple([seg] * n_full + ([rest] if rest else []))


def cold(fn, args):
  lowered = jax.jit(fn).lower(*args)
  t0 = time.perf_counter()
  lowered.compile()
  return time.perf_counter() - t0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--width", type=int, default=0,
                  help="0 = causal baseline; 1..8 = block-diagonal template")
  args_cli = ap.parse_args()
  w = args_cli.width or None
  tmpl = template_for_width(w)

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

  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex = pack(examples, BUDGET, 4, 8, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  x = jax.random.normal(
      jax.random.PRNGKey(0), (*shape, cfg.embed_dim), jnp.bfloat16)
  rngs = nnx.Rngs(params=0)
  args = (x, pos, attn_mask, seg)

  def make_fn(module):
    def f(x, pos, mask, seg):
      model_lib._SPLASH_SEGMENT_TEMPLATE = tmpl  # noqa: SLF001
      def loss(x):
        _, out = module(x, pos, None, mask, seg)
        return jnp.sum(out.astype(jnp.float32))
      return jax.grad(loss)(x)
    return f

  # LAYER FIRST -- nothing has compiled this mask's Mosaic kernel yet.
  t_layer = cold(make_fn(model_lib.DecoderLayer(config=cfg, rngs=rngs)), args)
  # ATTENTION SECOND -- same mask, so the Mosaic kernel should now be cached.
  t_attn = cold(make_fn(model_lib.Attention(config=cfg, rngs=rngs)), args)
  # A SECOND LAYER, same mask -- should also be cheap.
  t_layer2 = cold(make_fn(model_lib.DecoderLayer(config=cfg, rngs=rngs)), args)

  print(f"WIDTH={w if w else 'causal'} TEMPLATE={tmpl} "
        f"LAYER_FIRST={t_layer:.2f} ATTN_SECOND={t_attn:.2f} "
        f"LAYER_SECOND={t_layer2:.2f}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
