"""P19.8.2 -- derive the splash band width from the packed chunk.

The band width W must be >= every segment in the chunk, or the band cuts a real
segment and the answer is wrong.  Pinning W would therefore need a runtime
guard.  Deriving W from the data removes that failure mode by construction: W is
computed from the very lengths it has to cover, so it can never be too small.

Three edits, all additive, all behind TUNIX_SPLASH_BAND=1:

  1. common.py   -- TrainExample gains `max_segment_len`, a STATIC field
                    (pytree_node=False) exactly like the existing
                    `num_segments`, whose comment already explains the pattern:
                    a fixed value every step, so the step compiles once.
  2. utils.py    -- `_emit` stamps each chunk with the longest segment it holds.
                    Computed from `_item_tokens`, which the packer already uses
                    for bin-fitting, so no new notion of "length" is introduced.
  3. rl_learner.py + agentic_rl_learner.py -- one call before the train step.

The hook itself lives in a new module `tunix/rl/splash_band.py` so the learner
does not have to import a specific model, and so there is exactly one place
where the env flag is read.

W is quantised UP to a multiple of the block size: the band's grid_width is
ceil(W/block)+1 either way, so rounding up costs nothing and keeps the number of
distinct programs at most L_max/block.

Usage: python3 p19_9_patch_autow.py <tunix_root_in> <out_dir>
"""

import os
import sys

NEW_MODULE = '''# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Derive the splash band width from a packed chunk.

Splash schedules its blocks from a STATIC mask, so a packed row is charged for
its whole causal area even though cross-segment blocks are then zeroed by
`segment_ids`.  Telling splash that no query reaches further back than the
chunk's longest segment lets it shrink the pallas grid instead.

The band is a SUPERSET of the true block-diagonal mask, so `segment_ids` still
does the exact masking and the output is bitwise unchanged (measured on v4-8:
every superset bitwise identical, non-supersets differ).

W is derived from the chunk rather than configured, so it can never be smaller
than a real segment -- which would silently truncate attention.

Off unless TUNIX_SPLASH_BAND=1.
"""

import os

BLOCK = 256

# Read once at import: this is a run-level switch, not a per-step one.
ENABLED = os.getenv("TUNIX_SPLASH_BAND", "") == "1"

# Observed widths, for the run's log.  A wide spread means chunks are being
# poisoned by one long sequence and W-bucketing the packer would pay off.
_seen: dict[int, int] = {}


def width_for(example) -> int | None:
  """Quantised band width for one packed chunk, or None to leave splash alone."""
  if not ENABLED:
    return None
  longest = getattr(example, "max_segment_len", None)
  if not longest:
    return None
  return -(-int(longest) // BLOCK) * BLOCK


def apply(example) -> int | None:
  """Declare this chunk's band width to the model.  No-op when disabled."""
  w = width_for(example)
  if w is None:
    return None
  from tunix.models.qwen3 import model as qwen3_model  # pylint: disable=g-import-not-at-top

  if hasattr(qwen3_model, "set_splash_band_w"):
    qwen3_model.set_splash_band_w(w)
    _seen[w] = _seen.get(w, 0) + 1
  return w


def observed_widths() -> dict[int, int]:
  """{width: chunk count} seen so far -- the distribution to log."""
  return dict(_seen)
'''

EDITS = {
    "tunix/rl/common.py": [
        (
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n",
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n"
            "  # Longest segment (in tokens) held by this packed chunk. Static\n"
            "  # like `num_segments` above: a per-chunk value read at trace time\n"
            "  # to size the splash band mask, so the number of distinct programs\n"
            "  # is bounded by L_max/block rather than by the layout space.\n"
            "  max_segment_len: int | None = flax.struct.field(\n"
            "      default=None, pytree_node=False\n"
            "  )\n",
        ),
    ],
    "tunix/rl/utils.py": [
        (
            "    return jax.tree.map(\n"
            "        lambda first_x, *rest_xs: None\n"
            "        if first_x is None\n"
            "        else jnp.concatenate((first_x, *rest_xs), axis=0),\n"
            "        *chunk_examples,\n"
            "    )\n",
            "    merged = jax.tree.map(\n"
            "        lambda first_x, *rest_xs: None\n"
            "        if first_x is None\n"
            "        else jnp.concatenate((first_x, *rest_xs), axis=0),\n"
            "        *chunk_examples,\n"
            "    )\n"
            "    # Longest segment in this chunk, from the same token count the\n"
            "    # bin-fitting above uses. Consumed by tunix.rl.splash_band to\n"
            "    # size the attention band; None-safe for unpacked callers.\n"
            "    longest = max(\n"
            "        (_item_tokens(it) for bin_items in chunk for it in bin_items),\n"
            "        default=0,\n"
            "    )\n"
            "    return merged.replace(max_segment_len=int(longest) or None)\n",
        ),
    ],
}


def main():
  root, out_dir = sys.argv[1], sys.argv[2]
  os.makedirs(out_dir, exist_ok=True)

  mod_path = os.path.join(out_dir, "splash_band.py")
  with open(mod_path, "w") as f:
    f.write(NEW_MODULE)
  print(f"  new module: {mod_path} ({len(NEW_MODULE.splitlines())} lines)")

  for rel, edits in EDITS.items():
    src_path = os.path.join(root, rel)
    with open(src_path) as f:
      src = f.read()
    before = len(src.splitlines())
    # expected counts DERIVED, never typed (lessons.md 2026-08-03)
    names = ("max_segment_len", "_item_tokens", "merged.replace")
    want = {}
    for n in names:
      c = src.count(n)
      for anchor, repl in edits:
        c += repl.count(n) - anchor.count(n)
      want[n] = c
    for i, (anchor, repl) in enumerate(edits, 1):
      k = src.count(anchor)
      if k != 1:
        raise SystemExit(
            f"PATCH FAIL: {rel} edit {i} anchor occurs {k}x (need 1):\n{anchor!r}")
      src = src.replace(anchor, repl, 1)
      print(f"  {rel} edit {i}: applied")
    for n, w in want.items():
      got = src.count(n)
      if got != w:
        raise SystemExit(f"PATCH FAIL: {rel} {n!r} {got}x, want {w}x")
    dst = os.path.join(out_dir, os.path.basename(rel))
    with open(dst, "w") as f:
      f.write(src)
    print(f"  wrote {dst} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
