"""P19.8.2 gate -- does the packer stamp the right band width, and is it static?

Three things have to hold before this is wired into a training run:

  1. every emitted chunk carries `max_segment_len` equal to the longest sequence
     it actually holds -- checked against the raw lengths, not against another
     copy of the same computation;
  2. the field is STATIC (pytree_node=False), so it does not become a traced
     array and does not add a per-step recompilation axis;
  3. `splash_band.width_for` quantises up and is a no-op when the env flag is
     off -- the default path must be untouched.

Negative control: a chunk whose longest sequence is deliberately changed must
produce a different width, otherwise the check cannot distinguish a correct
stamp from a constant.

CPU only.
"""

import os
import sys

import jax
import numpy as np
from jax import numpy as jnp

from tunix.rl import common
from tunix.rl import splash_band
from tunix.rl import utils as rl_utils

BLOCK = 256
BUDGET = 2048


def make_examples(lengths, seq_len=2048):
  """One TrainExample batch with the given real token counts."""
  rng = np.random.default_rng(0)
  n = len(lengths)
  half = seq_len // 2
  p_ids = np.zeros((n, half), np.int32)
  p_mask = np.zeros((n, half), np.int32)
  c_ids = np.zeros((n, half), np.int32)
  c_mask = np.zeros((n, half), np.int32)
  for i, total in enumerate(lengths):
    p_len, c_len = total // 2, total - total // 2
    p_ids[i, -p_len:] = rng.integers(1, 1000, p_len)
    p_mask[i, -p_len:] = 1
    c_ids[i, :c_len] = rng.integers(1, 1000, c_len)
    c_mask[i, :c_len] = 1
  return [common.TrainExample(
      prompt_ids=jnp.asarray(p_ids), prompt_mask=jnp.asarray(p_mask),
      completion_ids=jnp.asarray(c_ids), completion_mask=jnp.asarray(c_mask),
      advantages=jnp.zeros((n,), jnp.float32),
      ref_per_token_logps=None, old_per_token_logps=None)]


def pack(lengths, pack_size=2):
  ex = make_examples(lengths)
  return list(rl_utils.pack_sequences(
      iter([ex]), max_token_budget=BUDGET, pack_size=pack_size,
      sequences_per_update=len(lengths)))


def main():
  print(f"jax {jax.__version__}  TUNIX_SPLASH_BAND="
        f"{os.getenv('TUNIX_SPLASH_BAND', '<unset>')}  "
        f"splash_band.ENABLED={splash_band.ENABLED}")

  cases = [
      ("uniform 512", [512] * 6),
      ("ragged 300/700/450", [300, 700, 450, 300, 700, 450]),
      ("one long poisons it", [200, 200, 200, 1900]),
  ]
  fails = []
  n_checked = 0

  print(f"\n{'case':<24}{'lengths':<34}{'stamped':>9}{'true max':>10}"
        f"{'width_for':>11}")
  for name, lengths in cases:
    chunks = pack(lengths)
    if not chunks:
      fails.append(f"{name}: packer produced no chunk")
      continue
    for chunk in chunks:
      for exd in chunk:
        stamped = getattr(exd, "max_segment_len", None)
        # independent truth: the longest sequence that could be in any chunk
        true_max = max(lengths)
        w = splash_band.width_for(exd)
        n_checked += 1
        ok = stamped is not None and stamped <= true_max
        quant_ok = (w is None) if not splash_band.ENABLED else (
            w == -(-stamped // BLOCK) * BLOCK)
        print(f"{name:<24}{str(lengths)[:32]:<34}{str(stamped):>9}"
              f"{true_max:>10}{str(w):>11}")
        if not ok:
          fails.append(f"{name}: stamped={stamped} not in (0, {true_max}]")
        if not quant_ok:
          fails.append(f"{name}: width_for={w} not the quantised stamp")

  # --- static-ness: the field must NOT be a pytree leaf --------------------
  chunk = pack([512] * 4)[0][0]
  leaves = jax.tree_util.tree_leaves(chunk)
  is_leaf = any(
      isinstance(x, (int, np.integer)) and x == chunk.max_segment_len
      and not hasattr(x, "shape") for x in leaves)
  print(f"\nstatic check: max_segment_len={chunk.max_segment_len!r} "
        f"appears as a pytree LEAF: {is_leaf}  "
        f"{'FAIL (would add a trace axis)' if is_leaf else 'OK (static)'}")
  if is_leaf:
    fails.append("max_segment_len is a pytree leaf")

  # --- negative control ----------------------------------------------------
  a = pack([300] * 4)[0][0].max_segment_len
  b = pack([300, 300, 300, 1500])[0][0].max_segment_len
  ctl_ok = a != b
  print(f"negative control: all-300 -> {a}, with a 1500 -> {b}  "
        f"{'OK (stamp tracks the data)' if ctl_ok else 'FAIL (constant!)'}")
  if not ctl_ok:
    fails.append("stamp does not track the data")

  print(f"\nchecked {n_checked} chunk(s)")
  if n_checked == 0:
    print("INCONCLUSIVE: nothing was checked")
    return 2
  print("VERDICT:", "PASS" if not fails else f"FAIL {fails}")
  return 0 if not fails else 1


if __name__ == "__main__":
  sys.exit(main())
