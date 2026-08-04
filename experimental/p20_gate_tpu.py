"""P20.3 -- the document mask on TPU: numerics first, then timing, then compiles.

Three arms on the production mesh, one packed chunk:
  orig  the branch's model.py, untouched
  off   patched, no layout declared
  on    patched, the chunk's real layout declared

`orig == off` proves the patch is neutral; `off == on` proves declaring the
layout does not change the answer -- the mask is a superset and `segment_ids`
still masks exactly, so this is expected to be BITWISE, not merely close.

Timing is only interpreted if the numerics pass.  The compile count is checked
separately: distinct layouts sharing a mask shape must NOT trigger a retrace,
which is the property that bounds the extra compiles.
"""

import argparse
import statistics
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

from bench_splash_packed import make_examples, model_inputs, pack
from tunix.models.qwen3 import model as model_lib

BLOCK = 256


def timed(fn, args, iters=10, warmup=3, inner=3):
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
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--expect", required=True, choices=("none", "off", "on"))
  ap.add_argument("--require_mounted", default="/app/tunix")
  args = ap.parse_args()

  patched = hasattr(model_lib, "_SPLASH_SEGMENT_LAYOUT")
  print(f"jax {jax.__version__} devices={len(jax.devices())} "
        f"kind={jax.devices()[0].device_kind}")
  print(f"model.py = {model_lib.__file__}\npatched  = {patched} "
        f"(arm '{args.expect}')")
  if not model_lib.__file__.startswith(args.require_mounted):
    raise SystemExit(f"PREFLIGHT FAIL: {model_lib.__file__}")
  if (args.expect == "none") == patched:
    raise SystemExit(f"PREFLIGHT FAIL: arm {args.expect} but patched={patched}")

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16

  examples, _ = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex = pack(examples, 2048, 4, 8, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  layout = tuple((1024, 1024) for _ in range(int(shape[0])))
  print(f"packed {tuple(shape)}  layout={layout}")

  if args.expect == "on":
    model_lib.set_splash_segment_layout(layout)
    print(f"layout declared -> {model_lib._SPLASH_SEGMENT_LAYOUT}")
  elif patched and model_lib._SPLASH_SEGMENT_LAYOUT is not None:
    raise SystemExit("PREFLIGHT FAIL: arm 'off' but a layout is set")
  print("PREFLIGHT OK")

  attn = model_lib.Attention(config=cfg, rngs=nnx.Rngs(params=0))
  x = jax.random.normal(
      jax.random.PRNGKey(0), (*shape, cfg.embed_dim), jnp.bfloat16)

  @jax.jit
  def fwd(x, pos, mask, seg):
    _, out = attn(x, pos, None, mask, seg)
    return out

  @jax.jit
  def fwd_bwd(x, pos, mask, seg):
    def loss(x):
      _, out = attn(x, pos, None, mask, seg)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  a = (x, pos, attn_mask, seg)
  out = np.asarray(jax.device_get(fwd(*a)))
  grad = np.asarray(jax.device_get(fwd_bwd(*a)))
  t_f, t_b = timed(fwd, a), timed(fwd_bwd, a)
  print(f"fwd {t_f:.3f} ms   fwd+bwd {t_b:.3f} ms")
  print(f"finite: out={bool(np.isfinite(out.astype(np.float32)).all())} "
        f"grad={bool(np.isfinite(grad.astype(np.float32)).all())}")

  # --- compile count: different layouts, same mask shape -> one program -----
  if args.expect == "on":
    n0 = fwd._cache_size()
    for alt in (((512, 512, 512, 512),) * int(shape[0]),
                ((1024, 512, 512),) * int(shape[0]),
                ((1024, 1024),) * int(shape[0])):
      model_lib.set_splash_segment_layout(alt)
      jax.block_until_ready(fwd(*a))
    print(f"compile probe: jit cache {n0} -> {fwd._cache_size()} after 3 more "
          f"layouts  (a layout change must NOT retrace unless the mask shape "
          f"changes)")
    model_lib.set_splash_segment_layout(layout)

  np.savez(args.out, out=out, grad=grad,
           t_fwd=np.array(t_f), t_bwd=np.array(t_b))
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
