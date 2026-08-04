"""P19.5 -- apply the static block-diagonal template branch to qwen3 model.py.

Same discipline as p18_5_patch_model.py: exact-string anchors, hard failure on
a missing anchor, post-conditions checked before anything is written.

What the patch does, in one sentence: when the caller declares the row's
(quantized) segment layout, hand splash a static block-diagonal NumpyMask
instead of a CausalMask, so `_shrink_mask_info` can shrink grid_width from
budget/block down to longest_segment/block.  `segment_ids` is still passed and
still does the exact masking, so the template only has to be a SUPERSET of the
true layout and the numerics are untouched.

PROTOTYPE SCOPE -- this is option (B) from the design discussion: the template
is read at trace time, so each distinct template retraces the step (<= 22 for
budget 2048).  The production form is option (A): build the kernel on host and
thread it in as a pytree argument, which collapses to one compile per distinct
grid_width (<= L_max/block = 8).  (A) needs the kernel threaded through
DecoderLayer -> Attention, which is a wider change and is NOT done here.

Usage: python3 p19_5_patch_blockdiag.py <in.py> <out.py>
"""

import sys

EDITS = [
    # 1. the template setter -- module-level so the prototype needs no new
    #    argument threaded through DecoderLayer -> Attention.
    (
        "env_utils.setup_sharding_environment()\n",
        '''env_utils.setup_sharding_environment()

# Candidate B (tasks/cl944_fsdp_packing/phase19.md): declare the packed row's
# segment layout so splash gets a STATIC block-diagonal mask instead of a causal
# one.  The static path then shrinks the pallas grid_width from budget/block to
# longest_segment/block -- the cross-segment blocks are never enumerated, rather
# than enumerated and skipped.  Exact masking stays with runtime `segment_ids`,
# so the template need only be a SUPERSET of the true layout and the output is
# bitwise unchanged (measured on v4-8, 2026-08-03).
#
# None => behave exactly as before.  Set to a tuple of quantized segment
# lengths, e.g. (1024, 1024), before the jitted step; each distinct value
# retraces (prototype; see module docstring for the argument-threaded form).
_SPLASH_SEGMENT_TEMPLATE = None


def set_splash_segment_template(template):
  """Declare the row layout for subsequent traces.  None restores the default."""
  global _SPLASH_SEGMENT_TEMPLATE
  _SPLASH_SEGMENT_TEMPLATE = tuple(template) if template is not None else None


def _blockdiag_mask(seq_len, seg_lens):
  """[seq_len, seq_len] bool: causal AND same-segment.

  A segment of length L spanning positions [p, p+L) may attend only within
  itself, so each q block's non-empty kv span is at most L/block wide -- which
  is exactly what lets `_shrink_mask_info` shrink the grid.
  """
  import numpy as _np  # local: keep the module's import block untouched

  pos = _np.arange(seq_len)
  seg = _np.zeros(seq_len, dtype=_np.int64)
  p = 0
  for i, length in enumerate(seg_lens, 1):
    seg[p:p + length] = i
    p += length
  causal = pos[None, :] <= pos[:, None]
  return causal & (seg[:, None] == seg[None, :]) & (seg[:, None] > 0)
''',
    ),
    # 2. swap the mask handed to make_splash_mha
    (
        """      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])
""",
        """      if (
          _SPLASH_SEGMENT_TEMPLATE is not None
          and sum(_SPLASH_SEGMENT_TEMPLATE) <= seq_len
      ):
        # Static block-diagonal: shrinks grid_width to longest_segment/block.
        dense = _blockdiag_mask(seq_len, _SPLASH_SEGMENT_TEMPLATE)
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


def main():
  src_path, dst_path = sys.argv[1], sys.argv[2]
  with open(src_path) as f:
    src = f.read()
  before = len(src.splitlines())

  for i, (anchor, replacement) in enumerate(EDITS, 1):
    n = src.count(anchor)
    if n != 1:
      raise SystemExit(
          f"PATCH FAIL: edit {i} anchor occurs {n} times (need 1):\n{anchor!r}")
    src = src.replace(anchor, replacement, 1)
    print(f"  edit {i}: applied")

  # 6 = 1 definition + 2 in the setter + 3 in the branch.  (Counted wrong on
  # the first run; the post-condition refused to write the file, which is the
  # correct failure direction.)
  for needle, want in (("_SPLASH_SEGMENT_TEMPLATE", 6),
                       ("_blockdiag_mask", 2),
                       ("mask_lib.NumpyMask(dense)", 1),
                       ("mask_lib.CausalMask((seq_len, seq_len))", 1)):
    got = src.count(needle)
    if got != want:
      raise SystemExit(
          f"PATCH FAIL: post-condition {needle!r} appears {got}x, want {want}x")

  with open(dst_path, "w") as f:
    f.write(src)
  print(f"wrote {dst_path} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
