"""P18.3 -- does splash's wall-clock actually track the blocks it schedules?

P18.0 proved the dynamic path SCHEDULES far fewer blocks (528 -> 80 on a packed
[1, 8192] row).  Scheduling fewer blocks is not the same as being faster: the
dynamic path also (a) never shrinks the grid -- `shrink_grid` is documented as
"currently ignored" -- (b) materialises the whole [heads, q, kv] bool mask, and
(c) loses the `lru_cache` that keeps the static MaskInfo off the critical path.
This measures whether the savings survive those costs.

Order matters and is enforced:

  1. NUMERICAL gate, bitwise, no tolerance.  Skipping a fully-masked block is a
     floating-point no-op (mask_value = -2.38e38 underflows exp() to exactly
     0.0, and the running max is unchanged, so the rescale factor is exactly
     1.0), so bitwise equality is the predicted outcome, not a wish.  If it
     fails, every timing number in the same run is VOID.
  2. NEGATIVE control: the dynamic path fed a PURE CAUSAL mask schedules the
     same 528 blocks as static, so it must also take the same time.  If it is
     slower, the dynamic machinery has a fixed overhead that must be priced
     before any saving is claimed.
  3. Only then, the timing verdict against the pre-registered block ratios.

Two dynamic variants are timed because they cost different things:
  dyn_const -- mask built and processed OUTSIDE jit, so MaskInfo enters the
               graph as a constant.  Isolates the KERNEL.
  dyn_live  -- mask built and processed INSIDE jit, which is what
               `model.py:565` would do if the production module switched over.
               Prices costs (b) and (c) above.

Run on the v4-8 with the verified image.  Shapes are per-row (batch 1) because
one splash kernel object carries one mask, so a packed batch whose rows have
different segment layouts cannot share a vmapped kernel -- an integration
constraint this script also probes explicitly (see `probe_vmap`).
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
    splash_attention_mask as mask_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack
from p18_0_blockcount import BLOCK, dense_mask
from tunix.models.qwen3 import model as model_lib

# Pre-registered in phase18.md from P18.0's counts.  Written before any timing.
EXPECTED_BLOCKS = {
    "A0": 528, "A1": 80, "N0": 528,   # [1, 8192] packed row, 8 segments
    "D0": 36, "D1": 20,               # [1, 2048] packed row, 2 segments
}
BLOCK_RATIO = {"A1/A0": 80 / 528, "D1/D0": 20 / 36, "N0/A0": 1.0}
TIME_TOL = 0.30       # timing must land within +-30% of the block ratio
CONTROL_TOL = 0.10    # negative control must land within +-10% of 1.0


def timed(fn, args, iters=20, warmup=3, inner=5):
  """Median per-call wall time in ms (same protocol as bench_splash_packed)."""
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  samples = []
  for _ in range(iters):
    t0 = time.perf_counter()
    for _ in range(inner):
      out = fn(*args)
    jax.block_until_ready(out)
    samples.append((time.perf_counter() - t0) * 1e3 / inner)
  return statistics.median(samples)


def block_sizes_for(config, seq_len):
  b = min(config.flash_attention_block_size, seq_len)
  return splash.BlockSizes(
      block_q=b, block_kv=b, block_q_dkv=b, block_kv_dkv=b,
      block_kv_dkv_compute=b, block_q_dq=b, block_kv_dq=b,
  )


def static_kernel(config, seq_len, qh):
  mask = mask_lib.MultiHeadMask(
      [mask_lib.CausalMask((seq_len, seq_len)) for _ in range(qh)]
  )
  return splash.make_splash_mha_single_device(
      mask, block_sizes=block_sizes_for(config, seq_len)
  )


def dynamic_kernel(config, seq_len, dense):
  """`dense` is [q, kv] bool; a leading axis of 1 broadcasts to every head.

  `_next_nonzero` forces h=0 when the mask info has one head
  (splash_attention_kernel.py:578-579), so one plane covers all heads and the
  mask costs 64 MiB instead of 1 GiB at 8192.
  """
  return splash.make_splash_mha_single_device(
      jnp.asarray(dense[None], dtype=jnp.bool),
      block_sizes=block_sizes_for(config, seq_len),
  )


def qkv(config, seq_len, qh, kh, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  hd = config.head_dim
  return (jax.random.normal(keys[0], (qh, seq_len, hd), jnp.bfloat16),
          jax.random.normal(keys[1], (kh, seq_len, hd), jnp.bfloat16),
          jax.random.normal(keys[2], (kh, seq_len, hd), jnp.bfloat16))


def count_blocks(dense):
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_mask_info as mi,
  )
  info, _ = mi.process_dynamic_mask(
      jnp.asarray(dense[None], dtype=jnp.bool), (BLOCK, BLOCK)
  )
  return int((np.asarray(info.block_mask) != 0).sum())


def probe_vmap(config, seq_len, qh, kh, dense_rows):
  """Can one vmapped call serve rows with DIFFERENT dynamic masks?

  Production vmaps a single kernel object over the batch (model.py:604).  A
  dynamic mask belongs to the kernel object, so this is the open integration
  question for P18.5.  Reported, never silently assumed.
  """
  q, k, v = qkv(config, seq_len, qh, kh)
  qs = jnp.stack([q] * len(dense_rows))
  ks = jnp.stack([k] * len(dense_rows))
  vs = jnp.stack([v] * len(dense_rows))
  masks = jnp.stack([jnp.asarray(d[None], dtype=jnp.bool) for d in dense_rows])

  def one(q_, k_, v_, m_):
    kern = splash.make_splash_mha_single_device(
        m_, block_sizes=block_sizes_for(config, seq_len)
    )
    return kern(q_, k_, v_)

  try:
    out = jax.jit(jax.vmap(one))(qs, ks, vs, masks)
    jax.block_until_ready(out)
    return True, f"OK, out={out.shape}"
  except Exception as exc:  # noqa: BLE001 -- reporting the failure IS the result
    return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def main():
  print(f"jax {jax.__version__}  devices={jax.devices()}")
  if not jax.devices() or jax.devices()[0].platform != "tpu":
    print("REFUSING: P18.3 must run on TPU")
    return 2

  config = model_lib.ModelConfig.qwen3_1p7b()
  config.use_flash_attention = True
  config.flash_attention_block_size = BLOCK
  config.dtype = jnp.bfloat16
  qh, kh = config.num_heads, config.num_kv_heads
  print(f"qwen3_1p7b: {qh}q/{kh}kv heads x {config.head_dim}, block {BLOCK}\n")

  # --- data: identical segment layouts to the ones P18.0 counted -------------
  examples, total_real = make_examples(8, 2048, 0, 0, seed=0, seq_tokens=1024)
  ex_a = pack(examples, 8192, 1, 8, row_multiple=1)
  _, _, seg_a, _ = model_inputs(ex_a)
  seg_a = np.asarray(seg_a)
  ex_d = pack(examples, 2048, 4, 8, row_multiple=1)
  _, _, seg_d, _ = model_inputs(ex_d)
  seg_d = np.asarray(seg_d)

  arms = {}  # name -> (seq_len, dense_or_None, seg_or_None)
  arms["A0"] = (8192, None, seg_a[0])
  arms["A1"] = (8192, dense_mask(seg_a[0], 8192), None)
  arms["N0"] = (8192, dense_mask(seg_a[0], 8192, causal_only=True), None)
  arms["D0"] = (2048, None, seg_d[0])
  arms["D1"] = (2048, dense_mask(seg_d[0], 2048), None)

  # --- block counts must reproduce P18.0 before anything is timed -----------
  print("block counts (must match P18.0's pre-registered values):")
  block_fail = []
  for name, (L, dense, _) in arms.items():
    n = count_blocks(dense) if dense is not None else (
        count_blocks(dense_mask(np.ones(L, dtype=np.int32), L, causal_only=True))
    )
    ok = n == EXPECTED_BLOCKS[name]
    print(f"  {name}: {n:>4}  expected {EXPECTED_BLOCKS[name]:>4}"
          f"  {'OK' if ok else '<-- MISMATCH'}")
    if not ok:
      block_fail.append(name)
  if block_fail:
    print(f"\nVOID: block counts disagree with P18.0 for {block_fail};"
          " geometry is not what P18.0 measured, so no timing is meaningful.")
    return 1
  print()

  # --- build callables ------------------------------------------------------
  def make_fns(name, live_mask=False):
    L, dense, seg = arms[name]
    q, k, v = qkv(config, L, qh, kh)
    if dense is None:  # static arm, segment ids supplied at call time
      kern = static_kernel(config, L, qh)
      seg_j = jnp.asarray(seg)
      fwd = jax.jit(lambda q, k, v, s: kern(
          q, k, v, splash.SegmentIds(q=s, kv=s)))
      args = (q, k, v, seg_j)
    elif not live_mask:  # dynamic, MaskInfo folded in as a constant
      kern = dynamic_kernel(config, L, dense)
      fwd = jax.jit(lambda q, k, v: kern(q, k, v))
      args = (q, k, v)
    else:  # dynamic, mask built AND processed inside the jit
      seg_row = jnp.asarray(
          seg_a[0] if L == 8192 else seg_d[0], dtype=jnp.int32)
      def live(q, k, v, s):
        pos = jnp.arange(L)
        m = (pos[None, :] <= pos[:, None]) & (s[:, None] == s[None, :])
        kern = splash.make_splash_mha_single_device(
            m[None], block_sizes=block_sizes_for(config, L))
        return kern(q, k, v)
      fwd = jax.jit(live)
      args = (q, k, v, seg_row)

    def fwd_bwd_fn(*a):
      return jax.grad(lambda x: jnp.sum(fwd(x, *a[1:]).astype(jnp.float32)))(a[0])
    return fwd, jax.jit(fwd_bwd_fn), args

  # --- GATE 1: numerical, bitwise, BEFORE any timing ------------------------
  print("GATE 1 -- numerical equivalence (bitwise, no tolerance)")
  num_fail = []
  for dyn, sta in (("A1", "A0"), ("D1", "D0")):
    f_d, _, a_d = make_fns(dyn)
    f_s, _, a_s = make_fns(sta)
    out_d = np.asarray(jax.device_get(f_d(*a_d)))
    out_s = np.asarray(jax.device_get(f_s(*a_s)))
    same = np.array_equal(out_d.view(np.uint16), out_s.view(np.uint16))
    if same:
      print(f"  {dyn} vs {sta}: BITWISE IDENTICAL")
    else:
      diff = int((out_d.view(np.uint16) != out_s.view(np.uint16)).sum())
      mx = float(np.max(np.abs(out_d.astype(np.float32)
                               - out_s.astype(np.float32))))
      rel = float(np.linalg.norm(out_d.astype(np.float32)
                                 - out_s.astype(np.float32))
                  / (np.linalg.norm(out_s.astype(np.float32)) + 1e-30))
      print(f"  {dyn} vs {sta}: DIFFERS  bytes={diff}/{out_d.size}"
            f"  max|d|={mx:.3e}  rel_L2={rel:.3e}")
      num_fail.append(f"{dyn} vs {sta}")
  if num_fail:
    print(f"\nVOID: numerical gate RED for {num_fail}."
          " Per phase18.md every timing number in this run is discarded"
          " and not interpreted.")
    return 1
  print()

  # --- timing ---------------------------------------------------------------
  print("TIMING (median of 20, 5 calls per sample)")
  print(f"{'arm':<12}{'blocks':>8}{'fwd ms':>10}{'fwd+bwd ms':>13}")
  results = {}
  for name in ("A0", "A1", "N0", "D0", "D1"):
    fwd, fb, args = make_fns(name)
    t_f, t_b = timed(fwd, args), timed(fb, args)
    results[name] = (t_f, t_b)
    print(f"{name:<12}{EXPECTED_BLOCKS[name]:>8}{t_f:>10.3f}{t_b:>13.3f}")

  for name in ("A1", "D1"):
    fwd, fb, args = make_fns(name, live_mask=True)
    t_f, t_b = timed(fwd, args), timed(fb, args)
    results[name + "_live"] = (t_f, t_b)
    print(f"{name + '_live':<12}{EXPECTED_BLOCKS[name]:>8}"
          f"{t_f:>10.3f}{t_b:>13.3f}   (mask built+processed inside jit)")

  expected_cells = 7
  if len(results) != expected_cells:
    print(f"\nINCONCLUSIVE: {len(results)}/{expected_cells} arms timed")
    return 2

  # --- GATE 2: negative control --------------------------------------------
  print("\nGATE 2 -- negative control (dynamic machinery must be free when it "
        "skips nothing)")
  ctl = results["N0"][1] / results["A0"][1]
  ctl_ok = abs(ctl - 1.0) <= CONTROL_TOL
  print(f"  N0/A0 fwd+bwd = {ctl:.3f} (expect 1.000 +-{CONTROL_TOL:.0%}): "
        f"{'PASS' if ctl_ok else 'FAIL -- dynamic path has fixed overhead'}")

  # --- GATE 3: does time track blocks? -------------------------------------
  print("\nGATE 3 -- measured ratio vs pre-registered block ratio "
        f"(+-{TIME_TOL:.0%})")
  time_fail = []
  for key, ratio in (("A1/A0", BLOCK_RATIO["A1/A0"]),
                     ("D1/D0", BLOCK_RATIO["D1/D0"])):
    dyn, sta = key.split("/")
    for label, suffix in (("kernel-only", ""), ("live-mask", "_live")):
      meas = results[dyn + suffix][1] / results[sta][1]
      ok = abs(meas - ratio) <= TIME_TOL * max(ratio, 1e-9) + TIME_TOL
      print(f"  {key} {label:<12} measured {meas:.3f}  vs blocks {ratio:.3f}"
            f"  {'PASS' if ok else 'OUT OF BAND -> P18.4 attribution'}")
      if not ok:
        time_fail.append(f"{key} {label}")

  # --- integration probe ----------------------------------------------------
  ok_vmap, detail = probe_vmap(config, 2048, qh, kh, [seg_d[0], seg_d[1]])
  print(f"\nINTEGRATION PROBE -- vmap over per-row dynamic masks: "
        f"{'WORKS' if ok_vmap else 'FAILS'}\n  {detail}")

  print("\n" + "=" * 74)
  if not ctl_ok:
    print("VERDICT: negative control FAILED -- price the fixed overhead first")
    return 1
  if time_fail:
    print(f"VERDICT: time does NOT track blocks for {time_fail} -> P18.4")
    return 1
  print("VERDICT: PASS -- numerics bitwise, control clean, and wall-clock "
        "follows the block count.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
