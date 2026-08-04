"""P20.1/P20.2 -- block-rounded document mask, with the packer untouched.

Splash schedules its blocks from a STATIC mask, so a packed row is charged for
its whole causal area and `segment_ids` only zeroes the cross-segment blocks
afterwards.  Handing splash the row layout instead lets it drop those blocks
from the schedule.

The layout is rounded OUTWARD to block boundaries before it becomes a mask.
That is the whole trick:

  * rounding the SEGMENT direction means no block is half-in/half-out of a
    segment, so `partial_mask_blocks` collapses to the single causal triangle
    that every diagonal block shares -- measured 11 -> 1 on a ragged row.  The
    number of compiled programs is then bounded by the (grid_width, 1) pairs
    that occur, measured as 4 across seven length distributions;
  * the CAUSAL direction must NOT be rounded.  `segment_ids` enforces
    same-segment, never causality, so rounding it would let a query attend to
    its own future.  The diagonal blocks therefore stay partial -- but they are
    all the same triangle, which is exactly why the count collapses.

Rounding only widens the mask, so it stays a SUPERSET of the true one and
`segment_ids` still does the exact masking: the output is unchanged.  Because
the rounding happens in the mask and not in the data, the packer keeps its
FFD result and no token is added -- measured 0.861x attention at 0% row cost,
versus 0.888x and +2.6% rows for the quantise-the-data variant.

Four edits + one new module:
  1. common.py     TrainExample.segment_layout, a STATIC field like num_segments
  2. utils.py      `_emit` stamps each chunk with its rows' segment lengths
  3. splash_mask.py (new) the only place the env flag is read
  4. rl_learner.py one call before the train step
  5. model.py      CausalMask -> NumpyMask(block-rounded document mask)

Usage: python3 p20_patch_docmask.py <tunix_root> <model_py_in> <out_dir>
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

"""Declare a packed chunk's segment layout to the attention kernel.

Splash builds its block schedule from a STATIC mask, so a packed row pays for
its entire causal area even though `segment_ids` zeroes the cross-segment
blocks right after.  Passing the layout lets splash drop those blocks from the
schedule instead of computing them.

The mask built from the layout is a SUPERSET of the true one (see
`tunix.models.qwen3.model._docmask`), so `segment_ids` still performs the exact
masking and the output is bitwise unchanged.

Off unless TUNIX_SPLASH_DOCMASK=1.
"""

import os

# Read once at import: a run-level switch, not a per-step one.
ENABLED = os.getenv("TUNIX_SPLASH_DOCMASK", "") == "1"

# (grid_width, partial_mask_blocks) pairs seen, for the run's log.  Each pair is
# one compiled program; a long tail here means the real layouts are more ragged
# than the seven distributions this was sized on, and the compile budget needs
# rechecking.
_seen: dict[tuple, int] = {}


def layout_for(example):
  """The chunk's per-row segment lengths, or None to leave splash alone."""
  if not ENABLED:
    return None
  return getattr(example, "segment_layout", None) or None


def apply(example):
  """Declare this chunk's layout to the model.  No-op when disabled."""
  layout = layout_for(example)
  if layout is None:
    return None
  from tunix.models.qwen3 import model as qwen3_model  # pylint: disable=g-import-not-at-top

  if hasattr(qwen3_model, "set_splash_segment_layout"):
    qwen3_model.set_splash_segment_layout(layout)
  return layout


def note_shapes(grid_width, partial_blocks):
  """Record one compiled mask shape, so the run can report how many there were."""
  key = (int(grid_width), int(partial_blocks))
  _seen[key] = _seen.get(key, 0) + 1


def observed_shapes() -> dict:
  """{(grid_width, partial_blocks): count} -- one entry per compiled program."""
  return dict(_seen)
'''

MODEL_EDITS = [
    (
        "env_utils.setup_sharding_environment()\n",
        '''env_utils.setup_sharding_environment()

# Document-mask gate for sequence packing (tasks/cl944_fsdp_packing/phase20.md).
# Splash schedules blocks from a STATIC mask, so a packed row is charged for its
# whole causal area and `segment_ids` only zeroes the cross-segment blocks after
# the fact.  Declaring the row layout lets splash drop them from the schedule.
#
# None => behave exactly as before.  Otherwise a tuple of per-row segment
# lengths, set by tunix.rl.splash_mask before the jitted step.
_SPLASH_SEGMENT_LAYOUT = None


def set_splash_segment_layout(layout):
  """Declare the packed chunk's per-row segment lengths.  None restores causal."""
  global _SPLASH_SEGMENT_LAYOUT
  _SPLASH_SEGMENT_LAYOUT = (
      tuple(tuple(int(x) for x in row) for row in layout)
      if layout is not None
      else None
  )


def _docmask(seq_len, layout, block):
  """causal AND block-rounded same-segment, unioned over the chunk's rows.

  Only the SEGMENT direction is rounded out to block boundaries: that makes
  every block either wholly inside a segment or wholly outside, so the only
  partial blocks left are the diagonal causal triangles -- all identical, hence
  deduplicated to one.  The causal direction is NOT rounded, because
  `segment_ids` enforces same-segment and never causality.

  Rounding only widens the mask, so the result is a superset of the true
  block-diagonal mask and `segment_ids` still masks exactly.
  """
  import numpy as _np  # local: keep the module's import block untouched

  nb = seq_len // block
  blk = _np.zeros((nb, nb), dtype=bool)
  for row in layout:
    pos = 0
    for length in row:
      if length <= 0:
        continue
      first = pos // block
      last = min((pos + length - 1) // block, nb - 1)
      for i in range(first, last + 1):
        blk[i, first : i + 1] = True
      pos += length
  positions = _np.arange(seq_len)
  causal = positions[None, :] <= positions[:, None]
  return causal & _np.kron(blk, _np.ones((block, block), dtype=bool))
''',
    ),
    (
        """      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])
""",
        """      if _SPLASH_SEGMENT_LAYOUT is not None:
        dense = _docmask(
            seq_len,
            _SPLASH_SEGMENT_LAYOUT,
            self.config.flash_attention_block_size,
        )
        multi_head_mask = mask_lib.MultiHeadMask(
            [mask_lib.NumpyMask(dense) for _ in range(qh)]
        )
      else:
        causal_mask = mask_lib.CausalMask((seq_len, seq_len))
        multi_head_mask = mask_lib.MultiHeadMask(
            [causal_mask for _ in range(qh)]
        )
""",
    ),
]

TUNIX_EDITS = {
    "tunix/rl/common.py": [
        (
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n",
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n"
            "  # Per-row segment lengths of this packed chunk. Static like\n"
            "  # `num_segments` above: read at trace time to build the document\n"
            "  # mask, so the number of distinct compiled programs is bounded by\n"
            "  # the mask shapes that occur, not by the layout space.\n"
            "  segment_layout: tuple[tuple[int, ...], ...] | None = (\n"
            "      flax.struct.field(default=None, pytree_node=False)\n"
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
            "    # Per-row segment lengths, from the same token count the\n"
            "    # bin-fitting above uses. Consumed by tunix.rl.splash_mask to\n"
            "    # build the document mask; None for unpacked callers.\n"
            "    layout = tuple(\n"
            "        tuple(_item_tokens(it) for it in bin_items)\n"
            "        for bin_items in chunk\n"
            "    )\n"
            "    return merged.replace(\n"
            "        segment_layout=layout if any(layout) else None\n"
            "    )\n",
        ),
    ],
    "tunix/rl/rl_learner.py": [
        (
            "from tunix.rl import utils as rl_utils\n",
            "from tunix.rl import splash_mask\nfrom tunix.rl import utils as rl_utils\n",
        ),
        (
            "        self.rl_cluster.update_actor(\n            curr_train_ds,\n",
            "        # Document-mask gate (tasks/cl944_fsdp_packing/phase20.md):\n"
            "        # declare this chunk's segment layout so splash can drop the\n"
            "        # cross-segment blocks from its schedule. No-op unless\n"
            "        # TUNIX_SPLASH_DOCMASK=1; the mask is a superset of the true\n"
            "        # one, so `segment_ids` still masks exactly.\n"
            "        splash_mask.apply(curr_train_ds)\n"
            "        self.rl_cluster.update_actor(\n            curr_train_ds,\n",
        ),
    ],
}

CHECKED = ("segment_layout", "_docmask", "_SPLASH_SEGMENT_LAYOUT", "splash_mask",
           "mask_lib.CausalMask((seq_len, seq_len))", "_item_tokens")


def apply_edits(label, src, edits):
  """Apply edits; expected post-counts are DERIVED, never typed (lessons.md)."""
  want = {}
  for name in CHECKED:
    c = src.count(name)
    for anchor, repl in edits:
      c += repl.count(name) - anchor.count(name)
    want[name] = c
  for i, (anchor, repl) in enumerate(edits, 1):
    n = src.count(anchor)
    if n != 1:
      raise SystemExit(
          f"PATCH FAIL: {label} edit {i} anchor occurs {n}x (need 1):\n{anchor!r}")
    src = src.replace(anchor, repl, 1)
  for name, w in want.items():
    got = src.count(name)
    if got != w:
      raise SystemExit(f"PATCH FAIL: {label} {name!r} {got}x, want {w}x")
  print(f"  {label}: {len(edits)} edit(s), post-conditions derived and met")
  return src


def main():
  root, model_in, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
  os.makedirs(out_dir, exist_ok=True)

  with open(os.path.join(out_dir, "splash_mask.py"), "w") as f:
    f.write(NEW_MODULE)
  print(f"  new module: splash_mask.py ({len(NEW_MODULE.splitlines())} lines)")

  for rel, edits in TUNIX_EDITS.items():
    with open(os.path.join(root, rel)) as f:
      src = f.read()
    out = apply_edits(rel, src, edits)
    with open(os.path.join(out_dir, os.path.basename(rel)), "w") as f:
      f.write(out)

  with open(model_in) as f:
    src = f.read()
  out = apply_edits("model.py", src, MODEL_EDITS)
  with open(os.path.join(out_dir, "model.py"), "w") as f:
    f.write(out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
