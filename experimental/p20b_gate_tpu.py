"""P20.4 gate -- the kernel-as-argument route, including the check that caught the bug.

The earlier three-arm gate passed while the design was silently broken, because
each arm ran in a FRESH PROCESS and therefore traced exactly once.  A global
read inside jit is a trace-time constant, so a second, different layout in the
SAME process was ignored -- and with data the first mask did not cover, the
answer was simply wrong.

So this gate adds the check that was missing:

  G1 neutrality   kernel=None must be bitwise identical to the unpatched model
  G2 correctness  a declared kernel must be bitwise identical to kernel=None
                  (the document mask is a superset; segment_ids still masks)
  G3 SAME-PROCESS SWITCH -- the one that matters.  Run layout A, then layout B
                  in the same process, and compare against B computed in
                  isolation.  If the second run returns A's answer, the channel
                  is baking the mask in and the design is broken.
  G4 compile count  layouts sharing a mask shape must not retrace; layouts with
                  different shapes must.
"""

import argparse
import statistics
import time

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from tunix.models.qwen3 import model as model_lib
from tunix.rl import splash_mask

BLOCK = 256
BUDGET = 2048
ROWS = 4


def build_kernel(cfg, layout, qh):
  """Built by the RL layer, not the model: the model only consumes a kernel."""
  return splash_mask.build_kernel(BUDGET, layout, BLOCK, qh)


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


def checksum(a):
  return int(np.asarray(a).view(np.uint16).astype(np.uint64).sum())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", required=True,
                  choices=("neutral", "A", "B", "A_then_B", "compiles"))
  args = ap.parse_args()

  threaded = "splash_kernel" in model_lib.Attention.block.__code__.co_varnames
  print(f"jax {jax.__version__}  model.py={model_lib.__file__}")
  print(f"threaded (Attention.block takes splash_kernel) = {threaded}")
  if args.mode != "neutral" and not threaded:
    raise SystemExit("PREFLIGHT FAIL: model.py is not the threaded build")

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(4, 1), ("fsdp", "tp"))
  mesh.__enter__()
  cfg = model_lib.ModelConfig.qwen3_1p7b()
  cfg.use_flash_attention = True
  cfg.flash_attention_block_size = BLOCK
  cfg.dtype = jnp.bfloat16
  qh = cfg.num_heads

  x = jax.random.normal(
      jax.random.PRNGKey(0), (ROWS, BUDGET, cfg.embed_dim), jnp.bfloat16)
  pos = jnp.tile(jnp.arange(BUDGET)[None], (ROWS, 1))
  # real layout is ONE segment per row: a (1024,1024) mask would cut it, so the
  # A-vs-B comparison below can actually tell the two masks apart.
  seg = jnp.ones((ROWS, BUDGET), jnp.int32)
  attn = model_lib.Attention(config=cfg, rngs=nnx.Rngs(params=0))

  LAY_A = ((1024, 1024),) * ROWS      # cuts the real segment -> wrong on purpose
  LAY_B = ((2048,),) * ROWS           # covers it

  if not threaded:
    f = jax.jit(lambda x, p, s: attn(x, p, None, None, s)[1])
    out = f(x, pos, seg)
    print(f"mode={args.mode} checksum={checksum(jax.device_get(out))} "
          f"fwd={timed(f, (x, pos, seg)):.3f}ms")
    return 0

  f = jax.jit(lambda k, x, p, s: attn(x, p, None, None, s, splash_kernel=k)[1])

  if args.mode == "neutral":
    out = jax.jit(lambda x, p, s: attn(x, p, None, None, s)[1])(x, pos, seg)
    print(f"mode=neutral checksum={checksum(jax.device_get(out))}")
    return 0

  if args.mode in ("A", "B"):
    lay = LAY_A if args.mode == "A" else LAY_B
    k = build_kernel(cfg, lay, qh)
    out = f(k, x, pos, seg)
    print(f"mode={args.mode} checksum={checksum(jax.device_get(out))} "
          f"cache={f._cache_size()} fwd={timed(f, (k, x, pos, seg)):.3f}ms")
    return 0

  if args.mode == "A_then_B":
    jax.block_until_ready(f(build_kernel(cfg, LAY_A, qh), x, pos, seg))
    out = f(build_kernel(cfg, LAY_B, qh), x, pos, seg)
    print(f"mode=A_then_B checksum={checksum(jax.device_get(out))} "
          f"cache={f._cache_size()}")
    return 0

  # mode == compiles
  n0 = f._cache_size()
  same_shape = [((1024, 1024),) * ROWS, ((1024, 512, 512),) * ROWS,
                ((1024, 256, 256, 256, 256),) * ROWS]
  for lay in same_shape:
    jax.block_until_ready(f(build_kernel(cfg, lay, qh), x, pos, seg))
  n1 = f._cache_size()
  for lay in (((512,) * 4,) * ROWS, ((2048,),) * ROWS):
    jax.block_until_ready(f(build_kernel(cfg, lay, qh), x, pos, seg))
  n2 = f._cache_size()
  print(f"mode=compiles  start={n0}  after 3 same-shape layouts={n1}  "
        f"after 2 different-shape layouts={n2}")
  print(f"  same-shape must not retrace: "
        f"{'PASS' if n1 - n0 == 1 else f'FAIL (+{n1-n0})'}")
  print(f"  different shapes must retrace: "
        f"{'PASS' if n2 > n1 else 'FAIL'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
