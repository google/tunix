"""P18.5 -- apply the dynamic-mask branch to a copy of qwen3 model.py.

The image's baked-in `/app/tunix` is what actually runs (it is the packing-native
tree with `pack_size` / `max_segments_per_packed_row`; the checked-out
`sequence_packing/tunix` is an older variant whose `pack_sequences` has a
different signature).  So the runtime gate has to patch the IMAGE's model.py and
bind-mount the result read-only, rather than mounting the whole worktree.

Three edits, all additive, all gated on an env var that defaults to off:
  1. `import os`
  2. the `_SPLASH_DYNAMIC_MASK` flag
  3. a new branch ahead of the existing `if segment_ids is not None:`

Anchors are exact strings; a missing anchor is a hard error, never a silent
no-op (tasks/lessons.md 2026-07-30: two patches silently failed on drifted
anchors).

Usage: python3 p18_5_patch_model.py <in.py> <out.py>
"""

import sys

EDITS = [
    (
        "from functools import partial\n",
        "from functools import partial\nimport os\n",
    ),
    (
        "env_utils.setup_sharding_environment()\n",
        """env_utils.setup_sharding_environment()

# Candidate C from tasks/cl944_fsdp_packing/phase18.md.  Splash schedules its
# blocks from a STATIC mask, so a packed row is charged for its entire causal
# area and `segment_ids` only zeroes cross-segment blocks the kernel already
# computed.  Handing splash a runtime `jax.Array` mask instead makes it skip
# those blocks outright.  Measured on v4-8: bitwise-identical output, no
# recompilation when the packing layout changes, +0.9% to build in-jit.
# DEFAULT OFF -- unset, this file behaves exactly as before.
_SPLASH_DYNAMIC_MASK = os.getenv('TUNIX_SPLASH_DYNAMIC_MASK', '') == '1'
""",
    ),
    (
        "      if segment_ids is not None:\n",
        """      if segment_ids is not None and _SPLASH_DYNAMIC_MASK and q_seq_shards == 1:
        # Candidate C: the mask carries the segment structure, so splash can
        # skip cross-segment and padding blocks instead of computing then
        # zeroing them.  Restricted to q_seq_shards == 1 because a sequence-
        # sharded mesh would need the dense mask sliced to match; the static
        # path below still handles that case.
        unsharded_seg_spec = P(shd_b, None)

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(shd_spec, unsharded_seq, unsharded_seq,
                      unsharded_seg_spec),
            out_specs=shd_spec,
            check_rep=False,
        )
        def sharded_splash_attn_dynamic(q_block, k_block, v_block, seg_block):
          def one_row(q_row, k_row, v_row, seg_row):
            pos = jnp.arange(seq_len)
            row_mask = (pos[None, :] <= pos[:, None]) & (
                seg_row[:, None] == seg_row[None, :]
            )
            # A leading axis of 1 is broadcast to every head by the kernel
            # (`_next_nonzero` forces h=0 when the mask info has one head).
            kernel = splash.make_splash_mha_single_device(
                row_mask[None], block_sizes=block_sizes
            )
            # `kernel(...)` branches on `is_dynamic_mask`, which
            # `make_splash_mha` turned into a traced array purely to choose a
            # named_scope label; that raises under jit.  Same computation,
            # without the branch.
            return splash._splash_attention(  # pylint: disable=protected-access
                kernel.fwd_mask_info,
                kernel.dq_mask_info,
                kernel.dkv_mask_info,
                q_row,
                k_row,
                v_row,
                **kernel.kwargs,
            )

          return jax.vmap(one_row)(q_block, k_block, v_block, seg_block)

        qkv = sharded_splash_attn_dynamic(
            query_proj, key_proj, value_proj, segment_ids
        )
      elif segment_ids is not None:
""",
    ),
]


def main():
  src_path, dst_path = sys.argv[1], sys.argv[2]
  with open(src_path) as f:
    src = f.read()

  for i, (anchor, replacement) in enumerate(EDITS, 1):
    n = src.count(anchor)
    if n != 1:
      raise SystemExit(
          f"PATCH FAIL: edit {i} anchor occurs {n} times (need exactly 1):\n"
          f"  {anchor!r}"
      )
    src = src.replace(anchor, replacement, 1)
    print(f"  edit {i}: applied")

  # post-conditions: the flag exists once, the new branch exists, the old
  # branch survived as the elif
  # 3, not 2: the env var name TUNIX_SPLASH_DYNAMIC_MASK contains the flag name
  # as a substring.  (The first run of this script refused to write because the
  # expectation was 2 -- the guard failed in the right direction.)
  for needle, want in (("_SPLASH_DYNAMIC_MASK", 3),
                       ("sharded_splash_attn_dynamic", 2),
                       ("elif segment_ids is not None:", 1)):
    got = src.count(needle)
    if got != want:
      raise SystemExit(
          f"PATCH FAIL: post-condition {needle!r} appears {got}x, want {want}x")

  with open(dst_path, "w") as f:
    f.write(src)
  print(f"wrote {dst_path}"
        f" ({len(src.splitlines())} lines, was {len(open(src_path).read().splitlines())})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
