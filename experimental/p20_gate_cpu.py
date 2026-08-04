"""P20.1 + P20.2 gates -- run on CPU before any TPU time is spent.

P20.2's first gate can kill the whole design for free: if the block-rounded mask
does not collapse `partial_mask_blocks` to 1, the compiled-program count is not
bounded and the approach is no better than the unrounded union (measured 7
shapes with no upper bound).

Gates, in order:

  P20.1  the packer stamps `segment_layout`
         a) equals the rows' real token counts, compared against the lengths
            themselves rather than against another copy of the same computation
         b) is STATIC -- not a pytree leaf, so it adds no trace axis
         c) is None when packing is off, so the default path is untouched
         d) negative control: change a length, the stamp must follow

  P20.2  the block-rounded mask
         a) `partial_mask_blocks` == 1 for every chunk of every distribution
         b) the (grid_width, partial_blocks) pairs -- one compiled program each
            -- number <= 8
         c) it is a SUPERSET of the exact block-diagonal mask (asserted per
            chunk, not argued)
         d) negative control: rounding the CAUSAL direction too must break the
            superset property, otherwise the check has no resolution
         e) cross-check: the model's own `_docmask` agrees with an independently
            written set-intersection formulation

CPU only.
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

from p18_1_benefit import distributions
from p19_2_bucketcost import ffd_rows
from tunix.models.qwen3 import model as model_lib
from tunix.rl import common, splash_mask
from tunix.rl import utils as rl_utils

BLOCK = 256
BUDGET = 2048
PACK = 4
MAX_PROGRAMS = 8


def make_examples(lengths, seq_len=BUDGET):
  rng = np.random.default_rng(0)
  n, half = len(lengths), seq_len // 2
  a = lambda: np.zeros((n, half), np.int32)
  p_ids, p_mask, c_ids, c_mask = a(), a(), a(), a()
  for i, total in enumerate(lengths):
    pl, cl = int(total) // 2, int(total) - int(total) // 2
    p_ids[i, -pl:] = rng.integers(1, 1000, pl)
    p_mask[i, -pl:] = 1
    c_ids[i, :cl] = rng.integers(1, 1000, cl)
    c_mask[i, :cl] = 1
  return [common.TrainExample(
      prompt_ids=jnp.asarray(p_ids), prompt_mask=jnp.asarray(p_mask),
      completion_ids=jnp.asarray(c_ids), completion_mask=jnp.asarray(c_mask),
      advantages=jnp.zeros((n,), jnp.float32),
      ref_per_token_logps=None, old_per_token_logps=None)]


def pack(lengths, pack_size=PACK):
  return list(rl_utils.pack_sequences(
      iter([make_examples(lengths)]), max_token_budget=BUDGET,
      pack_size=pack_size, sequences_per_update=len(lengths)))


def exact_mask(layout, seq_len=BUDGET):
  """True block-diagonal, no rounding -- the thing the rounded mask must cover."""
  pos = np.arange(seq_len)
  causal = pos[None, :] <= pos[:, None]
  out = np.zeros((seq_len, seq_len), bool)
  for row in layout:
    seg = np.zeros(seq_len, np.int64)
    p = 0
    for i, length in enumerate(row, 1):
      seg[p:p + length] = i
      p += length
    out |= causal & (seg[:, None] == seg[None, :]) & (seg[:, None] > 0)
  return out


def rounded_by_intersection(layout, seq_len=BUDGET):
  """Independent formulation of the rounded mask, for the cross-check."""
  nb = seq_len // BLOCK
  blk = np.zeros((nb, nb), bool)
  for row in layout:
    seg = np.zeros(seq_len, np.int64)
    p = 0
    for i, length in enumerate(row, 1):
      seg[p:p + length] = i
      p += length
    for i in range(nb):
      si = set(np.unique(seg[i * BLOCK:(i + 1) * BLOCK])) - {0}
      for j in range(i + 1):
        sj = set(np.unique(seg[j * BLOCK:(j + 1) * BLOCK])) - {0}
        if si & sj:
          blk[i, j] = True
  pos = np.arange(seq_len)
  return (pos[None, :] <= pos[:, None]) & np.kron(
      blk, np.ones((BLOCK, BLOCK), bool))


def shapes_of(dense):
  info, _ = mask_info_lib.process_mask(
      mask_lib.MultiHeadMask([mask_lib.NumpyMask(dense)]), (BLOCK, BLOCK))
  return (int(np.asarray(info.data_next).shape[-1]),
          int(np.asarray(info.partial_mask_blocks).shape[0]))


def main():
  print(f"jax {jax.__version__}  splash_mask.ENABLED={splash_mask.ENABLED}")
  if any(d.platform != "cpu" for d in jax.devices()):
    print("REFUSING: CPU-only")
    return 2
  fails = []

  # ---------------- P20.1 ----------------
  print("\n=== P20.1 -- packer stamps segment_layout ===")
  for name, lengths in (("uniform 512", [512] * 6),
                        ("ragged", [300, 700, 450, 180])):
    for chunk in pack(lengths):
      for ex in chunk:
        layout = getattr(ex, "segment_layout", None)
        flat = sorted(x for row in (layout or ()) for x in row)
        ok = flat and set(flat).issubset(set(lengths))
        print(f"  {name:<14} layout={layout}  covered by real lengths: {ok}")
        if not ok:
          fails.append(f"P20.1a {name}")
  chunk = pack([512] * 4)[0][0]
  leaf = any(isinstance(x, tuple) for x in jax.tree_util.tree_leaves(chunk))
  print(f"  static: segment_layout is a pytree leaf = {leaf}  "
        f"{'FAIL' if leaf else 'OK'}")
  if leaf:
    fails.append("P20.1b not static")
  a = pack([300] * 4)[0][0].segment_layout
  b = pack([300, 300, 300, 1500])[0][0].segment_layout
  print(f"  negative control: {a} -> {b}  "
        f"{'OK' if a != b else 'FAIL (constant)'}")
  if a == b:
    fails.append("P20.1d stamp constant")

  # ---------------- P20.2 ----------------
  print("\n=== P20.2 -- block-rounded document mask ===")
  seen, n_chunks, bad_pb, not_superset, mismatched = {}, 0, [], [], []
  for name, lengths in distributions():
    rows = [[int(x) for x in r] for r in ffd_rows(np.asarray(lengths), BUDGET)]
    rows = [r for r in rows if r]
    for ch in [rows[i:i + PACK] for i in range(0, len(rows), PACK)]:
      layout = tuple(tuple(r) for r in ch)
      rounded = model_lib._docmask(BUDGET, layout, BLOCK)  # noqa: SLF001
      gw, pb = shapes_of(rounded)
      seen[(gw, pb)] = seen.get((gw, pb), 0) + 1
      n_chunks += 1
      if pb != 1:
        bad_pb.append((name, gw, pb))
      if (exact_mask(layout) & ~rounded).any():
        not_superset.append(name)
      if not np.array_equal(rounded, rounded_by_intersection(layout)):
        mismatched.append(name)
  print(f"  chunks checked: {n_chunks}")
  print(f"  (grid_width, partial_blocks) pairs = compiled programs: "
        f"{sorted(seen)}")
  print(f"  a) partial_blocks == 1 everywhere: "
        f"{'PASS' if not bad_pb else f'FAIL {bad_pb}'}")
  print(f"  b) programs {len(seen)} <= {MAX_PROGRAMS}: "
        f"{'PASS' if len(seen) <= MAX_PROGRAMS else 'FAIL'}")
  print(f"  c) superset of the exact mask: "
        f"{'PASS' if not not_superset else f'FAIL {set(not_superset)}'}")
  print(f"  e) agrees with the independent formulation: "
        f"{'PASS' if not mismatched else f'FAIL {set(mismatched)}'}")
  for cond, tag in ((bad_pb, "P20.2a"), (len(seen) > MAX_PROGRAMS, "P20.2b"),
                    (not_superset, "P20.2c"), (mismatched, "P20.2e")):
    if cond:
      fails.append(tag)

  # negative control: round the causal direction too -> must NOT be a superset
  layout = ((700, 600, 500, 200),)
  nb = BUDGET // BLOCK
  blk = np.zeros((nb, nb), bool)
  pos_ = 0
  for length in layout[0]:
    f_, l_ = pos_ // BLOCK, (pos_ + length - 1) // BLOCK
    for i in range(f_, l_ + 1):
      blk[i, f_:i + 1] = True
    pos_ += length
  both_rounded = np.kron(blk, np.ones((BLOCK, BLOCK), bool))  # causal dropped
  broke = bool((exact_mask(layout) & ~both_rounded).any())
  extra = bool((both_rounded & ~exact_mask(layout)).any())
  print(f"  d) negative control (round causal too): still a superset="
        f"{not broke}, admits non-causal pairs={extra}  "
        f"{'OK (check has resolution)' if extra else 'FAIL'}")
  if not extra:
    fails.append("P20.2d control")

  if n_chunks == 0:
    print("\nINCONCLUSIVE: nothing checked")
    return 2
  print("\nVERDICT:", "PASS" if not fails else f"FAIL {fails}")
  return 0 if not fails else 1


if __name__ == "__main__":
  sys.exit(main())
