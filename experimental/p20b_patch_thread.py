"""P20.4 -- thread the splash kernel through the model as a pytree ARGUMENT.

Why not the module-level global the earlier patches used: it is read inside the
jitted function, so its value is baked in as a trace-time constant.  jit's cache
key is the ARGUMENTS' shapes, and a global is not an argument, so changing it
after the first trace is silently ignored.  Measured on v4-8: declaring layout
A, running, then declaring layout B and running again returned A's answer bit
for bit -- and with data whose real layout was B, A's mask truncated a segment,
so the second call was simply wrong, with no error and no recompile.

Passing the kernel as a normal argument fixes this at the root:
`SplashAttentionKernel` is a registered pytree whose leaves are the MaskInfo
arrays, so the mask VALUES are runtime data while only their SHAPES enter the
cache key.  That is the property that bounds the compiled programs by mask shape
(4 shapes measured across seven length distributions) instead of by layout
(effectively one per step).  P19.4 measured exactly this: four layouts sharing a
grid_width compiled once; six differing grid_widths compiled six times.

Six edits along the call chain, each additive and defaulting to None so the
unpatched behaviour is bit-for-bit unchanged:

  Qwen3.__call__ -> DecoderLayer.__call__ -> .block -> Attention.__call__ -> .block

Usage: python3 p20b_patch_thread.py <model_py_in> <model_py_out>
"""

import sys

PARAM = "      splash_kernel=None,\n"

EDITS = [
    # ---- Attention.block: signature, then use ------------------------------
    (
        "      segment_ids: jaxtyping.Array | None = None,\n"
        "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    seq_len = x.shape[1]\n",
        "      segment_ids: jaxtyping.Array | None = None,\n"
        + PARAM
        + "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    seq_len = x.shape[1]\n",
    ),
    (
        "      splash_attn_kernel = splash.make_splash_mha(\n"
        "          multi_head_mask,\n"
        "          block_sizes=block_sizes,\n"
        "          head_shards=head_shards,\n"
        "          q_seq_shards=q_seq_shards,\n"
        "      )\n",
        "      # A caller-supplied kernel carries the packed chunk's document\n"
        "      # mask.  It arrives as a pytree ARGUMENT, so its MaskInfo values\n"
        "      # are runtime data and only their shapes enter jit's cache key --\n"
        "      # one compiled program per mask shape, not per layout.  Building\n"
        "      # it from a module-level global instead would bake the mask in as\n"
        "      # a trace-time constant and silently ignore later changes.\n"
        "      splash_attn_kernel = (\n"
        "          splash_kernel\n"
        "          if splash_kernel is not None\n"
        "          else splash.make_splash_mha(\n"
        "              multi_head_mask,\n"
        "              block_sizes=block_sizes,\n"
        "              head_shards=head_shards,\n"
        "              q_seq_shards=q_seq_shards,\n"
        "          )\n"
        "      )\n",
    ),
    # ---- Attention.__call__: signature, then both paths --------------------
    (
        "      segment_ids: jaxtyping.Array | None = None,\n"
        "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    if (\n"
        "        self.config.remat_config == RematConfig.BLOCK\n",
        "      segment_ids: jaxtyping.Array | None = None,\n"
        + PARAM
        + "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    if (\n"
        "        self.config.remat_config == RematConfig.BLOCK\n",
    ),
    (
        "          self, x, segment_pos, cache, attn_mask, segment_ids\n"
        "      )\n"
        "    else:\n"
        "      return self.block(x, segment_pos, cache, attn_mask,"
        " segment_ids=segment_ids)\n",
        "          self, x, segment_pos, cache, attn_mask, segment_ids,\n"
        "          splash_kernel\n"
        "      )\n"
        "    else:\n"
        "      return self.block(\n"
        "          x, segment_pos, cache, attn_mask, segment_ids=segment_ids,\n"
        "          splash_kernel=splash_kernel,\n"
        "      )\n",
    ),
    # ---- DecoderLayer.block: signature, then the attn call -----------------
    (
        "      segment_ids: jaxtyping.Array | None = None,\n"
        "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    inputs_normalized = self.input_layernorm(x)\n",
        "      segment_ids: jaxtyping.Array | None = None,\n"
        + PARAM
        + "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    inputs_normalized = self.input_layernorm(x)\n",
    ),
    (
        "    cache, attn_output = self.attn(\n"
        "        inputs_normalized,\n"
        "        segment_pos,\n"
        "        cache,\n"
        "        attn_mask,\n"
        "        segment_ids=segment_ids,\n"
        "    )\n",
        "    cache, attn_output = self.attn(\n"
        "        inputs_normalized,\n"
        "        segment_pos,\n"
        "        cache,\n"
        "        attn_mask,\n"
        "        segment_ids=segment_ids,\n"
        "        splash_kernel=splash_kernel,\n"
        "    )\n",
    ),
    # ---- DecoderLayer.__call__: signature, then both paths -----------------
    (
        "      segment_ids: jaxtyping.Array | None = None,\n"
        "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    if (\n"
        "        self.config.remat_config == RematConfig.DECODER\n",
        "      segment_ids: jaxtyping.Array | None = None,\n"
        + PARAM
        + "  ) -> tuple[LayerCache | None, jaxtyping.Array]:\n"
        "    if (\n"
        "        self.config.remat_config == RematConfig.DECODER\n",
    ),
    (
        "          self, x, segment_pos, cache, attn_mask, segment_ids\n"
        "      )\n"
        "    else:\n"
        "      return self.block(\n"
        "          x, segment_pos, cache, attn_mask, segment_ids=segment_ids\n"
        "      )\n",
        "          self, x, segment_pos, cache, attn_mask, segment_ids,\n"
        "          splash_kernel\n"
        "      )\n"
        "    else:\n"
        "      return self.block(\n"
        "          x, segment_pos, cache, attn_mask, segment_ids=segment_ids,\n"
        "          splash_kernel=splash_kernel,\n"
        "      )\n",
    ),
    # ---- Qwen3.__call__: signature, then the layer loop --------------------
    (
        "      segment_ids: jaxtyping.Array | None = None,  # [B, L]\n"
        "      skip_lm_head: bool = False,\n",
        "      segment_ids: jaxtyping.Array | None = None,  # [B, L]\n"
        "      skip_lm_head: bool = False,\n"
        "      splash_kernel=None,\n",
    ),
    (
        "      layer_cache, x = layer(\n"
        "          x,\n"
        "          positions,\n"
        "          layer_cache,\n"
        "          attention_mask,\n"
        "          segment_ids=segment_ids,\n"
        "      )\n",
        "      layer_cache, x = layer(\n"
        "          x,\n"
        "          positions,\n"
        "          layer_cache,\n"
        "          attention_mask,\n"
        "          segment_ids=segment_ids,\n"
        "          splash_kernel=splash_kernel,\n"
        "      )\n",
    ),
]

CHECKED = ("splash_kernel", "make_splash_mha", "segment_ids")


def main():
  src_path, dst = sys.argv[1], sys.argv[2]
  with open(src_path) as f:
    src = f.read()
  before = len(src.splitlines())

  # Expected post-counts are DERIVED, never typed: three hand-written
  # counts were wrong and this guard refused to write all three times.
  want = {}
  for name in CHECKED:
    c = src.count(name)
    for anchor, repl in EDITS:
      c += repl.count(name) - anchor.count(name)
    want[name] = c

  for i, (anchor, repl) in enumerate(EDITS, 1):
    n = src.count(anchor)
    if n != 1:
      raise SystemExit(
          f"PATCH FAIL: edit {i} anchor occurs {n}x (need 1):\n{anchor!r}")
    src = src.replace(anchor, repl, 1)
    print(f"  edit {i}: applied")

  for name, w in want.items():
    got = src.count(name)
    if got != w:
      raise SystemExit(f"PATCH FAIL: {name!r} {got}x, want {w}x")
    print(f"  post-condition {name!r}: {got} (derived)")

  with open(dst, "w") as f:
    f.write(src)
  print(f"wrote {dst} ({len(src.splitlines())} lines, was {before})")
  return 0


if __name__ == "__main__":
  sys.exit(main())
