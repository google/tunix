"""Regression gate: attach must work on what pack_sequences ACTUALLY yields.

`pack_sequences` is typed `-> Iterator[list[TrainExample]]` and `_mark` returns
`[merged.replace(...)]`.  The learner therefore hands `attach` a LIST.  An
earlier version indexed `segment_layout` straight off that list, got None, and
returned it untouched on every step -- the feature was a no-op for a whole
end-to-end run, with no error and no log line.

The P20.5 route gate missed it because it fed a hand-built single TrainExample
instead of the real container.  So this gate asserts on the real one, and on
the loud-failure behaviour that has to exist when nothing gets attached.
"""

import sys

import jax.numpy as jnp

from tunix.rl import common, splash_mask

BUDGET, BLOCK, HEADS = 2048, 256, 16
LAYOUT = ((900, 700, 448), (2048,))


def chunk(layout):
  n = len(layout) if layout else 2
  return common.TrainExample(
      prompt_ids=jnp.zeros((n, 8), jnp.int32),
      prompt_mask=jnp.ones((n, 8), jnp.int32),
      completion_ids=jnp.zeros((n, BUDGET), jnp.int32),
      completion_mask=jnp.ones((n, BUDGET), jnp.int32),
      advantages=jnp.zeros((n,), jnp.float32),
      ref_per_token_logps=None,
      old_per_token_logps=None,
      segment_layout=layout,
  )


fails = []
print(f"ENABLED={splash_mask.ENABLED}")

# 1) the real container: a LIST of one chunk, as _mark returns
out = splash_mask.attach([chunk(LAYOUT)], seq_len=BUDGET, block=BLOCK,
                         num_heads=HEADS)
print(f"1) list in  -> {type(out).__name__} out, len={len(out)}")
if not isinstance(out, list):
  fails.append("list in must give list out")
got = getattr(out[0], "splash_kernel", None) is not None
print(f"   kernel attached inside the list : {got}")
if splash_mask.ENABLED and not got:
  fails.append("ENABLED but no kernel attached to the list element")
if not splash_mask.ENABLED and got:
  fails.append("disabled but a kernel was attached")

# 2) a bare example must still work (the learner is not the only caller)
one = splash_mask.attach(chunk(LAYOUT), seq_len=BUDGET, block=BLOCK,
                         num_heads=HEADS)
bare = getattr(one, "splash_kernel", None) is not None
print(f"2) bare example -> kernel attached  : {bare}")
if bare != splash_mask.ENABLED:
  fails.append("bare example path disagrees with ENABLED")

# 3) doing nothing must be LOUD, never silent
before = splash_mask.stats()["skipped"]
splash_mask.attach([chunk(None)], seq_len=BUDGET, block=BLOCK, num_heads=HEADS)
after = splash_mask.stats()["skipped"]
print(f"3) no-layout chunk -> skipped counter {before} -> {after}")
if splash_mask.ENABLED and after == before:
  fails.append("a silent no-op was not counted -- this is the bug that shipped")

print(f"\nstats: {splash_mask.stats()}")
print("VERDICT:", "PASS" if not fails else "FAIL " + "; ".join(fails))
sys.exit(0 if not fails else 1)
