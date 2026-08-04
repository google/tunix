"""P18.5 -- production module gate for the dynamic-mask branch.

Runs the real `model_lib.Attention` on the production mesh and dumps its output,
its gradient and its timings, so three arms can be compared BITWISE:

  orig  the pre-patch model.py, bind-mounted read-only over the patched one
  off   the patched model.py with TUNIX_SPLASH_DYNAMIC_MASK unset
  on    the patched model.py with TUNIX_SPLASH_DYNAMIC_MASK=1

  orig == off  proves the patch is neutral when the flag is off.
  off  == on   proves the dynamic branch does not change the numerics.

Inputs come from the production packer and `common.process_ids`, the same path
P18.0/P18.1 counted blocks on, so the geometry is the one already characterised.
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
  ap.add_argument("--expect", required=True, choices=("none", "off", "on"),
                  help="PREFLIGHT: what this arm must be running. 'none' = the "
                       "pre-patch module (no flag attribute at all), 'off'/'on' "
                       "= the patched module with the flag False/True.")
  ap.add_argument("--require_mounted", default="/work/tunix",
                  help="PREFLIGHT: model.py must come from under this prefix.")
  ap.add_argument("--budget", type=int, default=2048,
                  help="2048 = the production default packing budget")
  ap.add_argument("--num_seqs", type=int, default=8)
  ap.add_argument("--seq_tokens", type=int, default=1024)
  args = ap.parse_args()

  # --- PREFLIGHT: refuse to launch if the variable under test is not the one
  # actually in effect.  The first run of this gate silently exercised the
  # image's baked-in /app/tunix because `-w .../experimental` leaves the mount
  # off sys.path, and printing the flag was not enough to stop it.
  flag = getattr(model_lib, "_SPLASH_DYNAMIC_MASK", None)
  print(f"jax {jax.__version__} devices={len(jax.devices())} "
        f"kind={jax.devices()[0].device_kind}")
  print(f"model.py       = {model_lib.__file__}")
  print(f"flag           = {flag}   (expecting arm '{args.expect}')")
  if not model_lib.__file__.startswith(args.require_mounted):
    raise SystemExit(
        f"PREFLIGHT FAIL: model.py is {model_lib.__file__}, not under "
        f"{args.require_mounted}. Set PYTHONPATH={args.require_mounted}."
    )
  want = {"none": None, "off": False, "on": True}[args.expect]
  if flag is not want:
    raise SystemExit(
        f"PREFLIGHT FAIL: arm '{args.expect}' requires flag {want!r} but the "
        f"loaded module has {flag!r}."
    )
  print("PREFLIGHT OK")

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()

  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16

  examples, total_real = make_examples(
      args.num_seqs, 2048, 0, 0, seed=0, seq_tokens=args.seq_tokens)
  rows = max(1, -(-total_real // args.budget))
  ex = pack(examples, args.budget, rows, args.num_seqs, row_multiple=4)
  pos, attn_mask, seg, shape = model_inputs(ex)
  seg = jnp.asarray(seg)
  print(f"packed: shape={tuple(shape)} segs/row="
        f"{[int(r.max()) for r in np.asarray(seg)]}")

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
  print(f"out finite={bool(np.isfinite(out.astype(np.float32)).all())} "
        f"grad finite={bool(np.isfinite(grad.astype(np.float32)).all())}")

  np.savez(args.out, out=out, grad=grad,
           t_fwd=np.array(t_f), t_bwd=np.array(t_b),
           flag=np.array(-1 if flag is None else int(flag)))
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
