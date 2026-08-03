"""Packer-side half of the splash document mask: stamp the chunk's layout.

Three edits, none of them semantic on their own -- they only make the layout
that the bin-fitting already computed visible to the RL layer:

  common.py      `TrainExample.segment_layout`, static like `num_segments`
  utils.py       `_emit` stamps it from the same `_item_tokens` the bin-fitting
                 uses, so the layout cannot drift from the actual packing
  rl_learner.py  one call to `splash_mask.attach` before the train step

`attach` is called on the HOST, after packing and outside jit, and it returns a
new example rather than mutating module state.  That is load-bearing: an earlier
version declared the layout through a module-level global, which jit read as a
trace-time constant and baked in, so a later layout was silently ignored and the
step computed a wrong answer with no error and no retrace.  See
splash_docmask_design.md.

Run before p20b_patch_thread.py (model side) and p20c_patch_route.py (routing);
`p20_splash_mask.py` is copied in as `tunix/rl/splash_mask.py` unmodified.

Usage: python3 p20_patch_packer.py <tunix_root> <out_dir>
"""

import os
import sys

EDITS = {
    "tunix/rl/common.py": [
        (
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n",
            "  num_segments: int | None = flax.struct.field("
            "default=None, pytree_node=False)\n"
            "  # Per-row segment lengths of this packed chunk. Static like\n"
            "  # `num_segments` above: read on the host to build the document\n"
            "  # mask, so it never becomes an extra trace axis.\n"
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
            "        # Attach this chunk's document-mask kernel (built on the host from\n"
            "        # the packer's segment_layout) so splash can drop the\n"
            "        # cross-segment blocks from its schedule.  No-op unless\n"
            "        # TUNIX_SPLASH_DOCMASK=1.  Attached here, after packing and\n"
            "        # outside jit: the kernel's MaskInfo leaves must be traced\n"
            "        # ARGUMENTS, not values baked in at trace time.\n"
            "        curr_train_ds = splash_mask.attach(curr_train_ds)\n"
            "        self.rl_cluster.update_actor(\n            curr_train_ds,\n",
        ),
    ],
}

CHECKED = ("segment_layout", "splash_mask", "_item_tokens", "num_segments")


def main():
  root, out_dir = sys.argv[1], sys.argv[2]
  os.makedirs(out_dir, exist_ok=True)
  for rel, edits in EDITS.items():
    with open(os.path.join(root, rel)) as f:
      src = f.read()
    before = len(src.splitlines())
    # Expected post-counts are DERIVED, never typed: three hand-written
    # counts were wrong and this guard refused to write all three times.
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
            f"PATCH FAIL: {rel} edit {i} anchor occurs {n}x:\n{anchor!r}")
      src = src.replace(anchor, repl, 1)
    for name, w in want.items():
      got = src.count(name)
      if got != w:
        raise SystemExit(f"PATCH FAIL: {rel} {name!r} {got}x, want {w}x")
    dst = os.path.join(out_dir, os.path.basename(rel))
    with open(dst, "w") as f:
      f.write(src)
    print(f"  {rel}: {len(edits)} edit(s), post-conditions derived and met "
          f"-> {dst} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
