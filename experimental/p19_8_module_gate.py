"""P19.5 -- module-level gate for the static block-diagonal template patch.

Same three-arm shape as p18_5_module_gate.py, and the same preflight discipline:
the arm asserts what it is running and refuses to launch otherwise, because the
first run of the P18.5 gate silently exercised the image's baked-in /app/tunix.

  orig  stock model.py (no template attribute at all)
  off   patched model.py, template left None
  on    patched model.py, template set to the row's real layout

  orig == off  -> the patch is neutral when the template is unset
  off  == on   -> declaring the layout does not change the numerics
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

  has_attr = hasattr(model_lib, "_SPLASH_BAND_W")
  print(f"jax {jax.__version__} devices={len(jax.devices())} "
        f"kind={jax.devices()[0].device_kind}")
  print(f"model.py = {model_lib.__file__}")
  print(f"patched  = {has_attr}   (arm '{args.expect}')")
  if not model_lib.__file__.startswith(args.require_mounted):
    raise SystemExit(f"PREFLIGHT FAIL: {model_lib.__file__} not under "
                     f"{args.require_mounted}")
  if (args.expect == "none") == has_attr:
    raise SystemExit(f"PREFLIGHT FAIL: arm '{args.expect}' but patched="
                     f"{has_attr}")

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16

  examples, total = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex = pack(examples, 2048, 4, 8, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  segs_per_row = [int(r.max()) for r in np.asarray(seg)]
  print(f"packed {tuple(shape)}  segs/row={segs_per_row}")

  if args.expect == "on":
    model_lib.set_splash_band_w(1024)
    print(f"band W -> {model_lib._SPLASH_BAND_W}")
  elif has_attr and model_lib._SPLASH_BAND_W is not None:
    raise SystemExit("PREFLIGHT FAIL: arm 'off' but a template is set")
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
  np.savez(args.out, out=out, grad=grad,
           t_fwd=np.array(t_f), t_bwd=np.array(t_b))
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
