"""P19.8.1 -- apply the BAND-mask branch to qwen3 model.py.

Design choice recorded in phase19.md P19.8: band, not exact block-diagonal.
The 2x2 measured on 2026-08-03 says they are alternatives, not a stack --
quantisation is the entry ticket for block-diagonal (which otherwise needs one
program per row) and is a pure loss for band (band's grid_width depends only on
W, and rounding W up to 256 usually lands in the same bucket).  Band gives
+7.9% at the layer level with ZERO packing cost; block-diagonal gives +13.5%
but costs 4.8% tokens and three packer changes.

Mechanically this is smaller than the block-diagonal patch: `LocalMask` is a
built-in, so no dense numpy mask has to be materialised.

  W = the longest segment in the chunk, in tokens.
  LocalMask(window_size=(W-1, 0)) lets a query see W-1 positions back, which is
  exactly what a segment of length W needs (its last token must reach its
  first).  That makes the band a SUPERSET of the true block-diagonal mask for
  any layout whose segments are all <= W, so `segment_ids` still does the exact
  masking and the output is unchanged.  P19.7 verified on TPU that every
  superset is bitwise identical and that a non-superset differs.

Anchors are exact strings; a missing anchor is a hard error.

Usage: python3 p19_8_patch_band.py <in.py> <out.py>
"""

import sys

EDITS = [
    (
        "env_utils.setup_sharding_environment()\n",
        '''env_utils.setup_sharding_environment()

# Band-mask gate for sequence packing (tasks/cl944_fsdp_packing/phase19.md).
# Splash schedules blocks from a STATIC mask, so a packed row is charged for its
# whole causal area and `segment_ids` only zeroes cross-segment blocks the
# kernel already computed.  Declaring W -- the longest segment in the chunk --
# lets splash shrink the pallas grid_width from budget/block to about
# W/block, because no query can legally reach further back than W-1.
#
# The band is a SUPERSET of the true block-diagonal mask, so `segment_ids`
# still performs the exact masking and the output is bitwise unchanged
# (verified on v4-8: every superset bitwise identical, non-supersets differ).
#
# None => behave exactly as before.  Set to the chunk's longest segment length
# before the jitted step; W is a compile-time value, and it has at most
# L_max/block distinct values, so the extra compiles are bounded and one-off.
_SPLASH_BAND_W = None


def set_splash_band_w(w):
  """Declare the chunk's longest segment length.  None restores the default."""
  global _SPLASH_BAND_W
  _SPLASH_BAND_W = int(w) if w is not None else None
''',
    ),
    (
        """      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])
""",
        """      if _SPLASH_BAND_W is not None and _SPLASH_BAND_W < seq_len:
        # A query may reach at most W-1 positions back, so the band covers any
        # layout whose segments are all <= W.  grid_width shrinks accordingly.
        band_mask = mask_lib.LocalMask(
            (seq_len, seq_len), (_SPLASH_BAND_W - 1, 0), 0
        )
        multi_head_mask = mask_lib.MultiHeadMask(
            [band_mask for _ in range(qh)]
        )
      else:
        causal_mask = mask_lib.CausalMask((seq_len, seq_len))
        multi_head_mask = mask_lib.MultiHeadMask(
            [causal_mask for _ in range(qh)]
        )
""",
    ),
]

# Names whose post-patch count is checked.  The EXPECTED count is DERIVED from
# the source and the edits, never typed by hand: three patchers in a row refused
# to write because a hand-typed count was wrong (right failure direction, wasted
# run each time).  A hand-typed constant is a second place to make a mistake;
# deriving it removes the error class.
CHECKED_NAMES = (
    "_SPLASH_BAND_W",
    "mask_lib.LocalMask",
    "mask_lib.CausalMask((seq_len, seq_len))",
    "def set_splash_band_w",
)


def expected_counts(src_before):
  """count_after = count_before - (removed by anchors) + (added by replacements)."""
  want = {}
  for name in CHECKED_NAMES:
    n = src_before.count(name)
    for anchor, replacement in EDITS:
      n += replacement.count(name) - anchor.count(name)
    want[name] = n
  return want


def main():
  src_path, dst_path = sys.argv[1], sys.argv[2]
  with open(src_path) as f:
    src = f.read()
  before = len(src.splitlines())
  want_counts = expected_counts(src)

  for i, (anchor, replacement) in enumerate(EDITS, 1):
    n = src.count(anchor)
    if n != 1:
      raise SystemExit(
          f"PATCH FAIL: edit {i} anchor occurs {n} times (need 1):\n{anchor!r}")
    src = src.replace(anchor, replacement, 1)
    print(f"  edit {i}: applied")

  for needle, want in want_counts.items():
    got = src.count(needle)
    if got != want:
      raise SystemExit(
          f"PATCH FAIL: post-condition {needle!r} appears {got}x, want {want}x")
    print(f"  post-condition {needle!r}: {got} (derived)")

  with open(dst_path, "w") as f:
    f.write(src)
  print(f"wrote {dst_path} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
