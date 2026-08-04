"""P18.0 -- block-count prediction gate for the splash DYNAMIC-mask path.

Question: splash schedules its blocks from a STATIC mask, so a packed row pays
for its whole causal area no matter how many sequences are inside it
(`splash_packing_findings.md`).  JAX also ships a dynamic path --
`make_splash_mha` dispatches to `process_dynamic_mask` when the mask is a
`jax.Array` -- whose docstring claims it "can still populate MaskInfo to skip
fully-masked blocks".  Nobody has checked whether it actually does.

Counting is deterministic arithmetic, not a measurement, so this runs on CPU for
free and can falsify the whole idea before any TPU time is spent.

Each arm is counted TWO INDEPENDENT WAYS:

  Method J -- hand the mask to JAX's own `process_mask` (static) or
              `process_dynamic_mask` (runtime jax.Array) and count
              `block_mask != 0`.  This is the thing under test.
  Method A -- recompute the same number in plain numpy, block by block, from
              position arithmetic.  It never touches JAX's mask_info code, so a
              bug on either side surfaces as a disagreement rather than as a
              confident wrong answer.

Geometry is RUN1's (8 sequences x 1024 real tokens per chip, block 256) and the
segment ids come from the production packer, so the known-answer controls are
comparable digit for digit with `phase11.md:20-23`.

CPU ONLY.  Run with JAX_PLATFORMS=cpu; this script must never touch a TPU.
"""

import sys

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)

from bench_splash_packed import make_examples, model_inputs, pack

BLOCK = 256

# Pre-registered in tasks/cl944_fsdp_packing/phase18.md BEFORE this ran.
# U0/D0/A0 are known answers from phase11.md:20-23; N0/X0 are controls.
EXPECTED = {
    "U0": 288,   # static causal, [8, 2048] unpacked          (known answer)
    "U1": 160,   # dynamic,       [8, 2048]                   10 real + 10 pad*pad, x8
    "D0": 144,   # static causal, [4, 2048] PRODUCTION DEFAULT (known answer)
    "D1": 80,    # dynamic,       [4, 2048]                   2 segs x 10, x4
    "A0": 528,   # static causal, [1, 8192]                   (known answer, 32*33/2)
    "A1": 80,    # dynamic,       [1, 8192]                   8 segs x 10
    "N0": 528,   # dynamic w/ pure causal mask   -- negative control
    "X0": 1024,  # dynamic w/ all-True mask      -- instrument control (full grid)
}
KNOWN_ANSWER_ARMS = ("U0", "D0", "A0")
CONTROL_ARMS = ("N0", "X0")
DYNAMIC_ARMS = ("U1", "D1", "A1")


# ---------------------------------------------------------------------------
# Method J -- what JAX's own mask_info schedules
# ---------------------------------------------------------------------------
def jax_static_blocks(rows, seq_len):
  """Non-zero blocks JAX schedules for a plain causal mask, times `rows`.

  The static path never sees segment ids, so every row gets the identical
  schedule -- which is the whole point being tested.
  """
  mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((seq_len, seq_len))])
  info, _ = mask_info_lib.process_mask(mask, (BLOCK, BLOCK))
  per_row = int((np.asarray(info.block_mask) != 0).sum())
  return per_row * rows, per_row


def jax_dynamic_blocks(dense_masks):
  """Non-zero blocks JAX schedules for a list of per-row dense bool masks."""
  total = 0
  for m in dense_masks:
    info, _ = mask_info_lib.process_dynamic_mask(
        jnp.asarray(m[None], dtype=jnp.bool), (BLOCK, BLOCK)
    )
    total += int((np.asarray(info.block_mask) != 0).sum())
  return total


# ---------------------------------------------------------------------------
# Method A -- independent numpy recount, block by block
# ---------------------------------------------------------------------------
def analytic_blocks(seg_row, seq_len, causal_only=False):
  """Blocks containing at least one allowed (q, kv) pair.

  Deliberately written without materialising the full mask and without any JAX
  call, so it shares nothing with Method J but the definition of the mask.
  """
  nb = seq_len // BLOCK
  count = 0
  for qi in range(nb):
    q_pos = np.arange(qi * BLOCK, (qi + 1) * BLOCK)
    for kj in range(nb):
      if kj > qi:  # strictly above the diagonal: causal forbids the whole block
        continue
      k_pos = np.arange(kj * BLOCK, (kj + 1) * BLOCK)
      allowed = k_pos[None, :] <= q_pos[:, None]
      if not causal_only:
        q_seg = seg_row[q_pos]
        k_seg = seg_row[k_pos]
        allowed = allowed & (q_seg[:, None] == k_seg[None, :])
      if allowed.any():
        count += 1
  return count


def dense_mask(seg_row, seq_len, causal_only=False, all_true=False):
  """The [q, kv] bool mask handed to the dynamic path."""
  if all_true:
    return np.ones((seq_len, seq_len), dtype=bool)
  pos = np.arange(seq_len)
  m = pos[None, :] <= pos[:, None]
  if not causal_only:
    m = m & (seg_row[:, None] == seg_row[None, :])
  return m


# ---------------------------------------------------------------------------
def main():
  print(f"jax {jax.__version__}  devices={jax.devices()}")
  if any(d.platform != "cpu" for d in jax.devices()):
    print("REFUSING: this script is CPU-only (set JAX_PLATFORMS=cpu)")
    return 2
  print(f"block size = {BLOCK}\n")

  # --- data: RUN1 geometry, per-chip share, from the production packer -------
  num_seqs, seq_len, seq_tokens = 8, 2048, 1024
  examples, total_real = make_examples(
      num_seqs, seq_len, 0, 0, seed=0, seq_tokens=seq_tokens
  )
  print(f"{num_seqs} seqs x {seq_tokens} real tokens = {total_real} real tokens")

  _, _, seg_u, shape_u = model_inputs(examples[0])
  seg_u = np.asarray(seg_u)

  ex_d = pack(examples, 2048, total_real // 2048, num_seqs, row_multiple=1)
  _, _, seg_d, shape_d = model_inputs(ex_d)
  seg_d = np.asarray(seg_d)

  ex_a = pack(examples, 8192, total_real // 8192, num_seqs, row_multiple=1)
  _, _, seg_a, shape_a = model_inputs(ex_a)
  seg_a = np.asarray(seg_a)

  for name, seg, shape in (("U", seg_u, shape_u), ("D", seg_d, shape_d),
                           ("A", seg_a, shape_a)):
    segs_per_row = [int(r.max()) for r in seg]
    print(f"  {name}: shape={tuple(shape)} seg_ids/row={segs_per_row}"
          f" distinct_ids={sorted(set(seg.reshape(-1).tolist()))[:12]}")
  print()

  # --- arms -----------------------------------------------------------------
  results = {}
  analytic = {}

  # static arms: JAX never sees the segments
  for arm, seg, shape in (("U0", seg_u, shape_u), ("D0", seg_d, shape_d),
                          ("A0", seg_a, shape_a)):
    rows, L = int(shape[0]), int(shape[1])
    results[arm], per_row = jax_static_blocks(rows, L)
    analytic[arm] = rows * analytic_blocks(seg[0], L, causal_only=True)
    print(f"  {arm}: static causal [{rows}, {L}]  per_row={per_row}")

  # dynamic arms: mask = causal & (same segment)
  for arm, seg, shape in (("U1", seg_u, shape_u), ("D1", seg_d, shape_d),
                          ("A1", seg_a, shape_a)):
    rows, L = int(shape[0]), int(shape[1])
    masks = [dense_mask(seg[r], L) for r in range(rows)]
    results[arm] = jax_dynamic_blocks(masks)
    analytic[arm] = sum(analytic_blocks(seg[r], L) for r in range(rows))
    print(f"  {arm}: dynamic       [{rows}, {L}]")

  # controls, on the A geometry
  L_a = int(shape_a[1])
  results["N0"] = jax_dynamic_blocks(
      [dense_mask(seg_a[0], L_a, causal_only=True)]
  )
  analytic["N0"] = analytic_blocks(seg_a[0], L_a, causal_only=True)
  results["X0"] = jax_dynamic_blocks([dense_mask(seg_a[0], L_a, all_true=True)])
  analytic["X0"] = (L_a // BLOCK) ** 2
  print("  N0: dynamic w/ pure causal mask   (negative control)")
  print("  X0: dynamic w/ all-True mask      (instrument control)\n")

  # --- completeness FIRST: no verdict may come from a missing measurement ----
  missing = [a for a in EXPECTED if a not in results]
  if missing:
    print(f"INCONCLUSIVE: {len(results)}/{len(EXPECTED)} arms produced a number;"
          f" missing {missing}")
    return 2

  # --- report ---------------------------------------------------------------
  print("=" * 78)
  print(f"{'arm':<6}{'JAX (method J)':>16}{'numpy (method A)':>18}"
        f"{'pre-registered':>16}{'':>4}")
  print("=" * 78)
  agree = True
  for arm in EXPECTED:
    j, a, e = results[arm], analytic[arm], EXPECTED[arm]
    ok = (j == a == e)
    agree &= ok
    print(f"{arm:<6}{j:>16}{a:>18}{e:>16}{'  OK' if ok else '  <-- MISMATCH'}")
  print("=" * 78)

  # --- gates ----------------------------------------------------------------
  print("\nGATES")
  fails = []

  for arm in KNOWN_ANSWER_ARMS:
    ok = results[arm] == EXPECTED[arm]
    print(f"  [1] known answer {arm} == {EXPECTED[arm]}: "
          f"{'PASS' if ok else f'FAIL (got {results[arm]})'}")
    if not ok:
      fails.append(f"known-answer {arm}")

  for arm in CONTROL_ARMS:
    ok = results[arm] == EXPECTED[arm]
    label = "negative" if arm == "N0" else "instrument"
    print(f"  [2] {label} control {arm} == {EXPECTED[arm]}: "
          f"{'PASS' if ok else f'FAIL (got {results[arm]})'}")
    if not ok:
      fails.append(f"control {arm}")

  for arm in DYNAMIC_ARMS:
    ok = results[arm] == analytic[arm]
    print(f"  [3] {arm}: JAX {results[arm]} == numpy {analytic[arm]}: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
      fails.append(f"J-vs-A {arm}")

  budget_invariant = results["D1"] == results["A1"]
  print(f"  [4] strong prediction D1 == A1 (budget-independence): "
        f"{results['D1']} vs {results['A1']} -> "
        f"{'HOLDS' if budget_invariant else 'DOES NOT HOLD'}")

  print(f"  [5] completeness: {len(results)}/{len(EXPECTED)} arms: PASS")

  # --- verdict --------------------------------------------------------------
  print("\nSAVINGS (dynamic / static, same geometry)")
  for dyn, sta in (("U1", "U0"), ("D1", "D0"), ("A1", "A0")):
    print(f"  {dyn}/{sta} = {results[dyn]}/{results[sta]} = "
          f"{results[dyn] / results[sta]:.3f}x")

  print()
  if fails:
    print(f"VERDICT: FAIL -- {fails}")
    return 1
  if not agree:
    print("VERDICT: FAIL -- a count disagreed with its pre-registered value")
    return 1
  print("VERDICT: PASS -- the dynamic path really does skip fully-masked "
        "blocks, and both methods agree with the pre-registered numbers.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
