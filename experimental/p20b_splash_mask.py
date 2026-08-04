# Copyright 2025 Google LLC
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

"""Build the splash attention kernel for a packed chunk's document mask.

Splash schedules its blocks from a STATIC mask, so a packed row is charged for
its entire causal area even though `segment_ids` zeroes the cross-segment blocks
straight after.  Handing splash the chunk's segment layout lets it drop those
blocks from the schedule instead of computing them.

Two properties make this safe and cheap:

* The mask is a SUPERSET of the true block-diagonal one, because the segment
  extents are rounded OUTWARD to block boundaries.  `segment_ids` still performs
  the exact masking, so the output is bitwise unchanged.
* Rounding the SEGMENT direction (never the causal one -- `segment_ids` does not
  enforce causality) leaves the diagonal causal triangles as the only partial
  blocks, and they are all identical, so `partial_mask_blocks` collapses to 1.
  With that, the compiled programs are bounded by the (grid_width, 1) pairs that
  occur -- four across seven length distributions -- rather than by the layout
  space.

The kernel is built HERE, on the host, and passed into the model as a pytree
argument.  It must not be built from a module-level global inside the jitted
function: a global is not an argument, so jit's cache would not see it change
and a later layout would be silently ignored (measured -- the second layout
returned the first one's answer, with a truncated segment and no error).

Off unless TUNIX_SPLASH_DOCMASK=1.
"""

import os

import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

# Read once at import: a run-level switch, not a per-step one.
ENABLED = os.getenv("TUNIX_SPLASH_DOCMASK", "") == "1"

_seen: dict[tuple, int] = {}


def docmask(seq_len: int, layout, block: int) -> np.ndarray:
  """causal AND block-rounded same-segment, unioned over the chunk's rows.

  Args:
    seq_len: tokens per packed row.
    layout: per-row tuples of segment lengths, e.g. ((1024, 1024), (512,) * 4).
    block: splash block size; segment extents are rounded out to it.

  Returns:
    A [seq_len, seq_len] bool mask that is a superset of the true
    block-diagonal mask for every row in the chunk.
  """
  nb = seq_len // block
  blk = np.zeros((nb, nb), dtype=bool)
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
  positions = np.arange(seq_len)
  causal = positions[None, :] <= positions[:, None]
  return causal & np.kron(blk, np.ones((block, block), dtype=bool))


def build_kernel(seq_len, layout, block, num_heads, head_shards=1,
                 q_seq_shards=1):
  """Splash kernel carrying the chunk's document mask, built on the host.

  `head_shards`/`q_seq_shards` must match what the model derives from the mesh,
  or `manual_sharding_spec` will disagree with the shard_map.  Only the
  1x1 case (tp=1, no sequence sharding) has been exercised; callers on a
  sharded mesh must pass the real values.
  """
  block_sizes = splash.BlockSizes(
      block_q=block, block_kv=block, block_q_dkv=block, block_kv_dkv=block,
      block_kv_dkv_compute=block, block_q_dq=block, block_kv_dq=block)
  dense = docmask(seq_len, layout, block)
  return splash.make_splash_mha(
      mask_lib.MultiHeadMask(
          [mask_lib.NumpyMask(dense) for _ in range(num_heads)]),
      block_sizes=block_sizes, head_shards=head_shards,
      q_seq_shards=q_seq_shards)


def kernel_for(example, seq_len, block, num_heads, head_shards=1,
               q_seq_shards=1):
  """Kernel for this packed chunk, or None to leave splash on its causal mask."""
  if not ENABLED:
    return None
  layout = getattr(example, "segment_layout", None)
  if not layout:
    return None
  kernel = build_kernel(seq_len, layout, block, num_heads, head_shards,
                        q_seq_shards)
  info = kernel.fwd_mask_info
  note_shapes(np.asarray(info.data_next).shape[-1],
              np.asarray(info.partial_mask_blocks).shape[0])
  return kernel


def note_shapes(grid_width, partial_blocks):
  """Record one mask shape; each distinct shape is one compiled program."""
  key = (int(grid_width), int(partial_blocks))
  _seen[key] = _seen.get(key, 0) + 1


def observed_shapes() -> dict:
  """{(grid_width, partial_blocks): chunks} -- one entry per compiled program."""
  return dict(_seen)
