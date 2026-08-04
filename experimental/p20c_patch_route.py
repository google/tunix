"""P20.5 -- route the splash kernel from the packed chunk to the model.

The model side already accepts a `splash_kernel` argument (p20b) and the packer
already stamps `segment_layout` (p20).  This closes the gap between them:

  learner            builds the kernel on host from the chunk's layout and
                     attaches it to the TrainExample
  algo_core          reads it off the example, next to `num_segments`, which is
                     read the same way two lines above
  common             forwards it into `model_kwargs`, guarded by the existing
                     `model_call_contains` capability check so models that do
                     not take the argument are unaffected

The field is a REGULAR pytree field, not a static one.  That is the whole point:
its leaves are the MaskInfo arrays, so jit sees them as arguments and caches on
their SHAPES -- one program per mask shape.  Making it static would put the mask
values in the cache key and give one program per layout, i.e. one per step.

The kernel is attached AFTER the packer's `jax.tree.map` concatenation (it is
per-chunk, not per-row), so it never participates in that merge.

Usage: python3 p20c_patch_route.py <tunix_root> <out_dir>
"""

import os
import sys

EDITS = {
    "tunix/rl/common.py": [
        # a regular pytree field: leaves are traced, shapes drive the cache key
        (
            "  segment_layout: tuple[tuple[int, ...], ...] | None = (\n"
            "      flax.struct.field(default=None, pytree_node=False)\n"
            "  )\n",
            "  segment_layout: tuple[tuple[int, ...], ...] | None = (\n"
            "      flax.struct.field(default=None, pytree_node=False)\n"
            "  )\n"
            "  # Splash kernel carrying this chunk's document mask, built on the\n"
            "  # host by tunix.rl.splash_mask. A REGULAR pytree field on purpose:\n"
            "  # its leaves are the MaskInfo arrays, so jit caches on their\n"
            "  # shapes (one program per mask shape). Making it static would put\n"
            "  # the mask values in the cache key -- one program per layout.\n"
            "  splash_kernel: object | None = None\n",
        ),
        (
            "def compute_per_token_logps(\n"
            "    graphdef,\n"
            "    state,\n",
            "def compute_per_token_logps(\n"
            "    graphdef,\n"
            "    state,\n",
        ),
        (
            "    temperature: float = 1.0,\n"
            "    chunk_size: int = 0,\n"
            ") -> jax.Array | tuple[jax.Array, jax.Array]:\n",
            "    temperature: float = 1.0,\n"
            "    chunk_size: int = 0,\n"
            "    splash_kernel=None,\n"
            ") -> jax.Array | tuple[jax.Array, jax.Array]:\n",
        ),
        (
            '  if model_call_contains(model, "segment_ids"):\n',
            "  # Forward the packed chunk's document-mask kernel when the model\n"
            "  # takes one; splash falls back to its causal mask otherwise.\n"
            '  if splash_kernel is not None and model_call_contains(\n'
            '      model, "splash_kernel"\n'
            '  ):\n'
            '    model_kwargs["splash_kernel"] = splash_kernel\n'
            '  if model_call_contains(model, "segment_ids"):\n',
        ),
    ],
    "tunix/rl/algo_core.py": [
        (
            "  segment_ids = getattr(train_example, \"segment_ids\", None)\n"
            "  num_segments = getattr(train_example, \"num_segments\", None)\n",
            "  segment_ids = getattr(train_example, \"segment_ids\", None)\n"
            "  num_segments = getattr(train_example, \"num_segments\", None)\n"
            "  # Document-mask kernel for this chunk, attached by the learner.\n"
            "  # None when packing or the feature is off, in which case splash\n"
            "  # keeps its causal mask and nothing below changes.\n"
            "  splash_kernel = getattr(train_example, \"splash_kernel\", None)\n",
        ),
        (
            "      temperature=algo_config.temperature,\n"
            "      chunk_size=kwargs.get(\"compute_logps_chunk_size\", 0),\n"
            "  )\n",
            "      temperature=algo_config.temperature,\n"
            "      chunk_size=kwargs.get(\"compute_logps_chunk_size\", 0),\n"
            "      splash_kernel=splash_kernel,\n"
            "  )\n",
        ),
    ],
}

CHECKED = ("splash_kernel", "model_call_contains", "num_segments",
           "compute_per_token_logps")


def main():
  root, out_dir = sys.argv[1], sys.argv[2]
  os.makedirs(out_dir, exist_ok=True)
  for rel, edits in EDITS.items():
    with open(os.path.join(root, rel)) as f:
      src = f.read()
    before = len(src.splitlines())
    # expected counts DERIVED (tasks/lessons.md 2026-08-03)
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
    print(f"  {rel}: {len(edits)} edits, post-conditions derived and met "
          f"-> {dst} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
