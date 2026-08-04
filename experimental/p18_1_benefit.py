"""P18.1 -- how big is the prize, as a function of the length distribution?

P18.0 established that the dynamic path schedules exactly the ideal number of
blocks (`sum of per-segment causal triangles`), so candidate C's benefit is a
closed-form property of the packing geometry rather than something that has to
be measured.  That matters here because the ONE real end-to-end length record on
this box is unusable: the only wandb run
(`sequence_packing/tunix/wandb/offline-run-20260727_185617-n2p2b9wr`) reports
`generation/train/completions/clip_ratio = 1` at every step and prompts pinned
at min == mean == max == 2048, and carries `logp_diff_mean = 9.003` -- which is
precisely the signature phase16.md:87-101 traced to `TO_HF_MAPPINGS` missing
495/495 weight-sync keys, i.e. vLLM generating from RANDOM WEIGHTS.  Its length
histogram is an artefact of that bug, not a workload property, so it is reported
and then set aside rather than used.

What this script does instead: run the production FFD packer over a family of
length distributions (phase12's three, plus the degenerate all-at-L_max case
that the broken run superficially resembles), count static vs dynamic blocks
with the SAME counter P18.0 validated against three known answers, and report
where candidate C's benefit lands.

CPU ONLY.  Run with JAX_PLATFORMS=cpu.
"""

import sys

import jax
import numpy as np
from jax import numpy as jnp

from bench_splash_packed import model_inputs, pack
from p18_0_blockcount import (
    BLOCK,
    analytic_blocks,
    dense_mask,
    jax_dynamic_blocks,
    jax_static_blocks,
)
from tunix.rl import common

# gsm8k recipe: MAX_PROMPT_LENGTH=1024 + MAX_RESPONSE_LENGTH=1024
# (examples/math_gsm8k/qwen3_grpo_demo.py:228-229, cited in phase11.md).
L_MAX = 2048
NUM_SEQS = 8  # per-chip share of RUN1's 32 seqs over fsdp 4

# The decision threshold, pre-registered in phase18.md P18.1.
MIN_BENEFIT = 0.10


def distributions(seed=0):
  """(name, lengths[]) -- total real tokens per sequence, capped at L_MAX."""
  rng = np.random.default_rng(seed)
  n = NUM_SEQS
  half = n // 2
  return [
      # phase12's three, scaled to this L_max
      ("uniform 700-950 (RUN1-like, narrow)",
       rng.integers(700, 951, size=n)),
      ("uniform 100-2048 (wide)",
       rng.integers(100, L_MAX + 1, size=n)),
      ("bimodal 70% short / 30% long",
       np.concatenate([rng.integers(150, 400, size=n - n * 3 // 10),
                       rng.integers(1600, L_MAX + 1, size=n * 3 // 10)])),
      # the geometry P18.0 measured, as a cross-check anchor
      ("uniform exactly 1024 (P18.0 anchor)",
       np.full(n, 1024)),
      # worst case for C: nothing shares a row, nothing to skip
      ("degenerate: every sequence at L_max",
       np.full(n, L_MAX)),
      ("near-cap 1900-2048 (what the broken run LOOKED like)",
       rng.integers(1900, L_MAX + 1, size=n)),
      ("bimodal 50/50 at 512 and L_max", np.concatenate([
          np.full(half, 512), np.full(n - half, L_MAX)])),
  ]


def build_examples(lengths, seq_len):
  """TrainExamples with the RL producer's layout: left-pad prompt, right-pad completion.

  Same contract as bench_splash_packed.make_examples, but taking the lengths
  explicitly so arbitrary distributions can be swept.
  """
  rng = np.random.default_rng(0)
  n = len(lengths)
  half = seq_len // 2
  p_ids = np.zeros((n, half), dtype=np.int32)
  p_mask = np.zeros((n, half), dtype=np.int32)
  c_ids = np.zeros((n, half), dtype=np.int32)
  c_mask = np.zeros((n, half), dtype=np.int32)
  for i, total in enumerate(lengths):
    total = int(min(total, seq_len))
    p_len, c_len = max(1, total // 2), max(1, total - total // 2)
    p_len, c_len = min(p_len, half), min(c_len, half)
    p_ids[i, -p_len:] = rng.integers(1, 1000, size=p_len)
    p_mask[i, -p_len:] = 1
    c_ids[i, :c_len] = rng.integers(1, 1000, size=c_len)
    c_mask[i, :c_len] = 1
  return [common.TrainExample(
      prompt_ids=jnp.asarray(p_ids), prompt_mask=jnp.asarray(p_mask),
      completion_ids=jnp.asarray(c_ids), completion_mask=jnp.asarray(c_mask),
      advantages=jnp.zeros((n,), dtype=jnp.float32),
      ref_per_token_logps=None, old_per_token_logps=None,
  )], int(sum(min(int(x), seq_len) for x in lengths))


def ideal_blocks(lengths):
  """Blocks a perfect per-sequence schedule would need: sum of causal triangles."""
  total = 0
  for L in lengths:
    nb = -(-int(min(L, L_MAX)) // BLOCK)
    total += nb * (nb + 1) // 2
  return total


def measure(lengths, budget):
  """(static, dynamic, ideal, rows) block counts for one distribution+budget."""
  examples, total_real = build_examples(lengths, L_MAX)
  rows_lb = max(1, -(-total_real // budget))
  ex = pack(examples, budget, rows_lb, len(lengths), row_multiple=1)
  _, _, seg, shape = model_inputs(ex)
  seg = np.asarray(seg)
  rows, L = int(shape[0]), int(shape[1])

  static, _ = jax_static_blocks(rows, L)
  dynamic = jax_dynamic_blocks([dense_mask(seg[r], L) for r in range(rows)])
  # independent recount, same discipline as P18.0
  recount = sum(analytic_blocks(seg[r], L) for r in range(rows))
  if recount != dynamic:
    raise AssertionError(
        f"method J ({dynamic}) != method A ({recount}) at budget {budget}"
    )
  return static, dynamic, ideal_blocks(lengths), rows


def main():
  print(f"jax {jax.__version__}  devices={jax.devices()}")
  if any(d.platform != "cpu" for d in jax.devices()):
    print("REFUSING: CPU-only script (set JAX_PLATFORMS=cpu)")
    return 2
  print(f"L_max = {L_MAX}, {NUM_SEQS} sequences/chip, block = {BLOCK}")
  print(f"pre-registered decision threshold: C must save >= "
        f"{MIN_BENEFIT:.0%} to justify TPU time\n")

  budgets = [L_MAX, 2 * L_MAX, 4 * L_MAX]
  rows_out = []
  for name, lengths in distributions():
    lengths = np.asarray(lengths)
    print(f"--- {name}")
    print(f"    lengths: mean={lengths.mean():.0f} min={lengths.min()} "
          f"max={lengths.max()}  L_avg/L_max={lengths.mean()/L_MAX:.2f}")
    for budget in budgets:
      static, dynamic, ideal, rows = measure(lengths, budget)
      benefit = 1.0 - dynamic / static
      tag = "  <-- production default" if budget == L_MAX else ""
      print(f"    budget {budget:>5}: rows={rows:>2}  static={static:>5}  "
            f"dynamic={dynamic:>5}  ideal={ideal:>5}  "
            f"C saves {benefit:>6.1%}{tag}")
      rows_out.append((name, budget, static, dynamic, ideal, benefit))
    print()

  # --- verdict --------------------------------------------------------------
  expected = len(list(distributions())) * len(budgets)
  if len(rows_out) != expected:
    print(f"INCONCLUSIVE: {len(rows_out)}/{expected} cells measured")
    return 2

  print("=" * 78)
  print("VERDICT")
  print("=" * 78)
  # dynamic must never exceed ideal, and never lose to static
  assert all(d >= i for _, _, _, d, i, _ in rows_out), "dynamic below ideal?!"
  assert all(d <= s for _, _, s, d, _, _ in rows_out), "dynamic worse than static?!"
  print(f"  sanity: ideal <= dynamic <= static in all {len(rows_out)} cells: PASS")

  at_default = [r for r in rows_out if r[1] == L_MAX]
  best = max(rows_out, key=lambda r: r[5])
  worst = min(rows_out, key=lambda r: r[5])
  print(f"  benefit range across all cells: {worst[5]:.1%} "
        f"({worst[0]}, budget {worst[1]}) .. {best[5]:.1%} "
        f"({best[0]}, budget {best[1]})")
  print("\n  At the PRODUCTION DEFAULT budget = L_max:")
  for name, _, s, d, i, b in at_default:
    verdict = "GO" if b >= MIN_BENEFIT else "below threshold"
    print(f"    {b:>6.1%}  {verdict:<16} {name}")

  n_go = sum(1 for r in at_default if r[5] >= MIN_BENEFIT)
  print(f"\n  {n_go}/{len(at_default)} distributions clear the "
        f"{MIN_BENEFIT:.0%} bar at the production budget.")
  print("  NOTE: the prize is a property of L_avg/L_max and the budget, not of "
        "the kernel.\n        The real workload distribution is NOT available "
        "on this box (see module docstring).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
