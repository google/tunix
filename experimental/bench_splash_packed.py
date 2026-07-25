"""Microbench: what does sequence packing cost on the TPU splash kernel?

Motivating e2e observation (gsm8k, 32 seqs/micro, fsdp=4): packing [4, 8192]
vs unpacking [32, 2048] left total train time FLAT while flash-attention
BACKWARD went 6ms -> 16ms. The JAX source says why: splash's block schedule
comes from a STATIC mask (`_process_mask` takes no segment info; the pallas
grid width is `mask_info.data_next.shape[-1]`), and `segment_ids` is just an
input that zeroes already-computed blocks inside `_apply_mask_and_soft_cap`.
So a packed row costs row_len^2/2 regardless of how many sequences it holds.

That predicts a BUDGET LAW. For a fixed token total T (rows = T / budget):
    attention ~ rows * budget^2 = T * budget      (LINEAR in budget)
    mlp/lm_head ~ rows * budget = T               (independent of budget)
i.e. packing always saves MLP tokens but pays an attention penalty that grows
with the budget -- the optimum is the smallest budget that still fits the
longest single sequence.

The primary measurement REPLICATES PRODUCTION: `model_lib.Attention` on the
production mesh, with the production hidden size and head layout, fed the
global shapes a training step executes. The module builds `make_splash_mha`
with head_shards=mesh['tp'], wraps it in shard_map and vmaps over the batch,
then runs the q/k/v/o projections around it -- nothing is bypassed.

  U        [32, 2048]  unpacked baseline, per-position non-pad segment_ids
  P4096    [8, 4096]   packed
  P8192    [4, 8192]   packed -- the shape the e2e run used
  C8192    [4, 8192]   the same shape with segment_ids dropped (isolates row
                       length from the segment feature itself)

The same 8-per-chip sequences are behind all three, so their effective
attention work is identical and any timing difference is the row geometry.

A second table times the RAW KERNEL on one chip's share (batch/fsdp rows,
heads/tp heads -- what production's head_shards splits off), built through the
entry point JAX's own splash tests use, `make_splash_mha_single_device`, which
is `make_splash_mha` with head_shards=q_seq_shards=1 and therefore exactly the
per-chip kernel when tp=1. Subtracting it from the module time attributes the
cost between attention and the token-linear projections. `--with_layer` adds a
full DecoderLayer (attention + MLP), the configuration whose e2e total went
flat.

Segment ids and packed rows come from the production packer
(`rl_utils.pack_sequences`), and each arm's model inputs from the production
`common.process_ids` -- which is what gives the UNPACKED arm its real padding
and its per-position non-pad segment_ids, so both arms hit the *_segmented
kernel and row length is the only variable. xprof kernel names carry a
`_segmented` suffix (`get_kernel_name(is_segmented=...)`), which is how these
traces are matched against the e2e trace symbol-for-symbol.
"""

import argparse
import statistics
import time

import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax import numpy as jnp
import numpy as np

from flax import nnx

from tunix.models.qwen3 import model as model_lib
from tunix.rl import common
from tunix.rl import utils as rl_utils


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num_seqs", type=int, default=32,
                 help="GLOBAL sequences per micro-batch -- the e2e run's 32."
                      " The mesh shards them over fsdp, so the shapes below are"
                      " the ones the e2e run actually executes.")
  p.add_argument("--mesh_fsdp", type=int, default=4,
                 help="Production mesh: gsm8k runs (fsdp 4, tp 1).")
  p.add_argument("--mesh_tp", type=int, default=1)
  p.add_argument("--seq_len", type=int, default=2048,
                 help="Padded per-sequence length of the unpacked baseline.")
  p.add_argument("--seq_tokens", type=int, default=1024,
                 help="Real tokens per sequence, UNIFORM, so the arms are an"
                      " exact controlled comparison: the same sequences and"
                      " the same total real tokens laid out over different row"
                      " geometries. 0 falls back to sampling min..max_tokens.")
  p.add_argument("--min_tokens", type=int, default=700,
                 help="Real-token sampling range, used only when seq_tokens=0")
  p.add_argument("--max_tokens", type=int, default=950,
                 help="(a ~20%% dummy_ratio, matching the e2e run).")
  p.add_argument("--budgets", type=str, default="4096,8192",
                 help="Packing budgets to bench. With the defaults the three"
                      " arms are [8,2048] unpacked, [2,4096] and [1,8192]"
                      " packed -- the same 8 sequences and therefore the SAME"
                      " effective attention work, differing only in how much"
                      " padding / cross-segment area the kernel runs through.")
  p.add_argument("--iters", type=int, default=20,
                 help="Timed samples per case (median reported).")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--inner", type=int, default=5,
                 help="Calls dispatched per timed sample, to amortize JAX's"
                      " per-dispatch overhead.")
  p.add_argument("--trace_dest", type=str, default="",
                 help="If set, write one xprof trace per case under here.")
  p.add_argument("--trace_iters", type=int, default=3)
  p.add_argument("--segment_sweep", type=str, default="1,2,4,8,16,32",
                 help="Segment counts to sweep at a FIXED row shape (empty to"
                      " skip). Decouples segment structure from row length:"
                      " flat timings prove segment_ids only masks, never"
                      " skips. Includes a skewed control and a cross-check"
                      " against the real packed row of the same geometry.")
  p.add_argument("--sweep_budget", type=int, default=8192,
                 help="Row length held fixed during the segment sweep.")
  p.add_argument("--skip_module", action="store_true",
                 help="Skip the production model_lib.Attention timings.")
  p.add_argument("--with_layer", action="store_true",
                 help="Also time a full DecoderLayer (attention + MLP), the"
                      " configuration whose e2e total went flat.")
  p.add_argument("--skip_kernel", action="store_true",
                 help="Skip the raw per-chip kernel timings (used to attribute"
                      " the module time between attention and projections).")
  p.add_argument("--model_config", type=str, default="qwen3_1p7b",
                 help="ModelConfig factory: qwen3_1p7b (the gsm8k run) or"
                      " qwen3_8b (FrozenLake).")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


# ---------------------------------------------------------------------------
# Data: build TrainExamples and pack them with the production packer.
# ---------------------------------------------------------------------------
def make_examples(num_seqs, seq_len, min_tokens, max_tokens, seed,
                  seq_tokens=0):
  """`num_seqs` TrainExamples, each padded to `seq_len`, real length `seq_tokens`.

  A UNIFORM real length (the default) makes the arms an exact controlled
  comparison: the same sequences and the same total real tokens, laid out over
  different row geometries, so any timing difference is the geometry. Pass
  seq_tokens=0 to sample min..max_tokens instead (a realistic ragged batch).

  Mirrors the RL producer's layout: prompts are LEFT-padded, completions are
  RIGHT-padded, so `unpad_train_example` inside pack_sequences recovers exactly
  the real tokens.
  """
  rng = np.random.default_rng(seed)
  if seq_tokens:
    lens = np.full(num_seqs, seq_tokens, dtype=int)
  else:
    lens = rng.integers(min_tokens, max_tokens + 1, size=num_seqs)
  half = seq_len // 2

  prompt_ids = np.zeros((num_seqs, half), dtype=np.int32)
  prompt_mask = np.zeros((num_seqs, half), dtype=np.int32)
  completion_ids = np.zeros((num_seqs, half), dtype=np.int32)
  completion_mask = np.zeros((num_seqs, half), dtype=np.int32)

  for i, total in enumerate(lens):
    p_len = int(total) // 2
    c_len = int(total) - p_len
    prompt_ids[i, -p_len:] = rng.integers(1, 1000, size=p_len)  # left pad
    prompt_mask[i, -p_len:] = 1
    completion_ids[i, :c_len] = rng.integers(1, 1000, size=c_len)  # right pad
    completion_mask[i, :c_len] = 1

  return [
      common.TrainExample(
          prompt_ids=jnp.asarray(prompt_ids),
          prompt_mask=jnp.asarray(prompt_mask),
          completion_ids=jnp.asarray(completion_ids),
          completion_mask=jnp.asarray(completion_mask),
          advantages=jnp.zeros((num_seqs,), dtype=jnp.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ], int(lens.sum())


def pack(examples, budget, pack_size, num_seqs, row_multiple=1):
  """One packed chunk straight out of the production FFD packer.

  The bench needs ALL sequences in a single [rows, budget] chunk so the case is
  one kernel invocation. FFD may not reach the ceil(total/budget) lower bound,
  so widen the chunk until everything fits rather than silently benching a
  partial chunk.
  """
  pack_size = max(row_multiple, -(-pack_size // row_multiple) * row_multiple)
  for rows in range(pack_size, pack_size + 8 * row_multiple, row_multiple):
    chunks = list(
        rl_utils.pack_sequences(
            iter([examples]),
            max_token_budget=budget,
            pack_size=rows,
            sequences_per_update=num_seqs,
        )
    )
    if len(chunks) == 1:
      return chunks[0][0]
  raise ValueError(
      f"pack_sequences needed more than {pack_size + 8} rows at budget {budget}"
  )


def geometry(example, budget, total_real_tokens):
  """(rows, segments-per-row, dummy_ratio) of a packed chunk."""
  seg = np.asarray(example.segment_ids)
  rows = seg.shape[0]
  segs = [int(seg[r].max()) for r in range(rows)]
  dummy = 1.0 - total_real_tokens / float(rows * budget)
  return rows, segs, dummy


def effective_attention_work(segment_ids):
  """Causal attention work over REAL tokens only: sum of len^2 per segment.

  This is the invariant of the experiment. The same sequences are being
  attended either way, so every arm has the same effective work no matter how
  the rows are arranged -- laying 8 x 1024-token sequences out as [8, 2048],
  [2, 4096] or [1, 8192] does not change what has to be computed. What differs
  is how much padding and how many cross-segment blocks the kernel runs
  through on top, so measured time divided by this number IS the waste factor.
  """
  seg = np.asarray(segment_ids)
  work = 0.0
  for row in seg:
    ids, counts = np.unique(row[row > 0], return_counts=True)
    del ids
    work += float((counts.astype(np.float64) ** 2).sum()) / 2.0
  return work


def synthetic_segment_ids(rows, row_len, num_segments, real_frac=1.0,
                          skew=False):
  """Segment ids with the same structure `pack_sequences` produces.

  Real packing couples the row length and the segment count (a bigger budget
  holds both a longer row AND more sequences), so a budget sweep cannot say
  which one drives the cost. Synthesizing the ids decouples them: hold the
  shape fixed and vary only the number of segments. Structure matches the
  packer's -- ids 1..K in contiguous runs, 0 for the padding tail -- which
  `gate_p10_geometry` asserts, and one synthetic point is timed against the
  real packed row of the same geometry as a cross-check.

  Args:
    rows: Rows in the chunk.
    row_len: Tokens per row (the packing budget).
    num_segments: Sequences packed into each row.
    real_frac: Fraction of the row that is real tokens (the rest is padding,
      segment 0), mirroring a dummy_ratio of 1 - real_frac.
    skew: If True, one segment takes half the real tokens and the rest split
      the remainder -- a control for whether segment SIZE distribution matters.

  Returns:
    An int32 [rows, row_len] array.
  """
  real = int(row_len * real_frac)
  if skew and num_segments > 1:
    first = real // 2
    rest = [(real - first) // (num_segments - 1)] * (num_segments - 1)
    sizes = [first] + rest
  else:
    sizes = [real // num_segments] * num_segments
  ids = np.zeros((rows, row_len), dtype=np.int32)
  for r in range(rows):
    pos = 0
    for i, size in enumerate(sizes, start=1):
      ids[r, pos:pos + size] = i
      pos += size
  return jnp.asarray(ids)


def model_inputs(example, pad_id=0, eos_id=0):
  """(positions, attn_mask, segment_ids, seq_len) exactly as training builds them.

  Delegates to the production `common.process_ids`, so each arm gets what
  `compute_per_token_logps` would pass to the model -- crucially including the
  UNPACKED arm's per-position non-pad `segment_ids` (production feeds the 0/1
  mask to splash so pad positions are suppressed; see common.py's process_ids),
  which means both arms hit the *_segmented kernel and the only difference left
  is row length.
  """
  ids, positions, attn_mask, input_seg_ids = common.process_ids(
      example.prompt_ids,
      example.completion_ids,
      pad_id,
      eos_id,
      example.segment_ids,
      example.segment_positions,
  )
  # Packed: process_ids returns attn_mask=None and defers to the packing ids.
  seg = example.segment_ids if example.segment_ids is not None else input_seg_ids
  return positions, attn_mask, seg, ids.shape


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def timed(fn, args, iters, warmup, inner=5):
  """Median per-call wall time (ms) of `fn(*args)`, warmup excluded.

  Each sample dispatches `inner` calls and synchronizes once, so JAX's
  per-dispatch overhead is amortized instead of being charged to every
  measurement (it would be a visible fraction of a sub-millisecond case).
  """
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  samples = []
  for _ in range(iters):
    t0 = time.perf_counter()
    for _ in range(inner):
      out = fn(*args)
    jax.block_until_ready(out)
    samples.append((time.perf_counter() - t0) * 1e3 / inner)
  return statistics.median(samples)


def maybe_trace(trace_dest, name, fn, args, iters):
  if not trace_dest:
    return
  with jax.profiler.trace(f"{trace_dest}/splash_bench_{name}"):
    for _ in range(iters):
      jax.block_until_ready(fn(*args))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def kernel_fns(config, batch, seq_len, segment_ids, seed, heads_div=1):
  """(forward, forward+backward) closures over the RAW splash kernel.

  Built the way JAX's own splash tests build it -- `make_splash_mha_single_
  device` (which is `make_splash_mha` with head_shards=q_seq_shards=1) called
  under `jax.vmap` over the batch, the same composition `model_lib.Attention`
  performs inside its shard_map. Single-device means no mesh and no sharding
  constraint on the row count, which a packed geometry can easily violate.
  """
  # Production splits the heads over tp (head_shards=mesh['tp']), so one chip
  # sees num_heads/tp of them; dividing here makes the single-device kernel
  # equal to that per-chip share.
  qh = max(1, config.num_heads // heads_div)
  kh = max(1, config.num_kv_heads // heads_div)
  hd = config.head_dim
  mask = mask_lib.MultiHeadMask(
      [mask_lib.CausalMask((seq_len, seq_len)) for _ in range(qh)]
  )
  block = min(config.flash_attention_block_size, seq_len)
  block_sizes = splash.BlockSizes(
      block_q=block, block_kv=block, block_q_dkv=block, block_kv_dkv=block,
      block_kv_dkv_compute=block, block_q_dq=block, block_kv_dq=block,
  )
  kernel = splash.make_splash_mha_single_device(mask, block_sizes=block_sizes)

  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  q = jax.random.normal(keys[0], (batch, qh, seq_len, hd), jnp.bfloat16)
  k = jax.random.normal(keys[1], (batch, kh, seq_len, hd), jnp.bfloat16)
  v = jax.random.normal(keys[2], (batch, kh, seq_len, hd), jnp.bfloat16)

  if segment_ids is None:
    call = jax.vmap(lambda q, k, v: kernel(q, k, v, None))
    args = (q, k, v)
  else:
    call = jax.vmap(
        lambda q, k, v, s: kernel(q, k, v, splash.SegmentIds(q=s, kv=s))
    )
    args = (q, k, v, jnp.asarray(segment_ids))

  fwd = jax.jit(call)

  @jax.jit
  def fwd_bwd(*a):
    return jax.grad(
        lambda q: jnp.sum(call(q, *a[1:]).astype(jnp.float32))
    )(a[0])

  return (fwd, args), (fwd_bwd, args)


def attn_fns(attn, x, segment_pos, attn_mask, segment_ids):
  """(forward, forward+backward) closures over one attention module.

  `attn_mask` is threaded through for fidelity with the production call; the
  splash path ignores it (it only consumes the static causal mask plus
  segment_ids) and the non-flash path uses it.
  """

  @jax.jit
  def fwd(x, segment_pos, attn_mask, segment_ids):
    _, out = attn(x, segment_pos, None, attn_mask, segment_ids)
    return out

  @jax.jit
  def fwd_bwd(x, segment_pos, attn_mask, segment_ids):
    def loss(x):
      _, out = attn(x, segment_pos, None, attn_mask, segment_ids)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  args = (x, segment_pos, attn_mask, segment_ids)
  return (fwd, args), (fwd_bwd, args)


def layer_fns(layer, x, segment_pos, attn_mask, segment_ids):
  @jax.jit
  def fwd(x, segment_pos, attn_mask, segment_ids):
    _, out = layer.block(x, segment_pos, None, attn_mask, segment_ids)
    return out

  @jax.jit
  def fwd_bwd(x, segment_pos, attn_mask, segment_ids):
    def loss(x):
      _, out = layer.block(x, segment_pos, None, attn_mask, segment_ids)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  args = (x, segment_pos, attn_mask, segment_ids)
  return (fwd, args), (fwd_bwd, args)


def main():
  args = parse_args()
  budgets = [int(b) for b in args.budgets.split(",") if b]

  # Production mesh: the module path shards exactly like the e2e run, so the
  # shapes below are the ones training executes (and the row counts divide by
  # fsdp for free -- 32/8/4 rows over 4 chips).
  mesh = jax.sharding.Mesh(
      np.array(jax.devices()[: args.mesh_fsdp * args.mesh_tp]).reshape(
          args.mesh_fsdp, args.mesh_tp
      ),
      ("fsdp", "tp"),
  )
  mesh.__enter__()
  fsdp, tp = args.mesh_fsdp, args.mesh_tp

  print(f"jax devices: {jax.devices()}")
  print(f"production mesh: fsdp={fsdp} tp={tp}")
  print(
      f"geometry (GLOBAL, as the e2e run executes): {args.num_seqs} seqs x "
      f"{args.seq_len} padded, budgets={budgets}"
  )

  examples, total_real = make_examples(
      args.num_seqs, args.seq_len, args.min_tokens, args.max_tokens, args.seed,
      seq_tokens=args.seq_tokens,
  )
  print(f"total real tokens: {total_real}"
        + (f" ({args.num_seqs} x {args.seq_tokens}, uniform)"
           if args.seq_tokens else " (ragged)"))

  config = getattr(model_lib.ModelConfig, args.model_config)()
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.dtype = jnp.bfloat16

  # ---- inputs per case -----------------------------------------------------
  # Every arm's (positions, attn_mask, segment_ids) comes from the production
  # `process_ids`, so the unpacked arm carries its real ~60% padding and the
  # per-position non-pad segment_ids that training actually feeds to splash.
  cases = []  # (name, segment_pos, attn_mask, segment_ids, shape, note)

  pos_u, mask_u, seg_u, shape_u = model_inputs(examples[0])
  real_frac = float(np.asarray(seg_u).mean())
  cases.append((
      "U", pos_u, mask_u, seg_u, shape_u,
      f"{list(shape_u)} unpacked, real={real_frac:.2f}",
  ))

  for budget in budgets:
    if budget < args.seq_len:
      print(f"  skipping budget {budget} < seq_len {args.seq_len}"
            " (a maximal sequence would not fit a row)")
      continue
    rows = -(-total_real // budget)
    # Rows stay an fsdp multiple, as production's pack_size = fsdp * dp does,
    # so the packed batch shards the same way the e2e run's does.
    ex = pack(examples, budget, rows, args.num_seqs, row_multiple=fsdp)
    r, segs, dummy = geometry(ex, budget, total_real)
    pos_p, mask_p, seg_p, shape_p = model_inputs(ex)
    note = f"[{r}, {budget}] packed, segs/row={segs}, dummy={dummy:.3f}"
    cases.append((f"P{budget}", pos_p, mask_p, seg_p, shape_p, note))
    if budget == max(budgets):
      # Control: same shape, segment_ids dropped -> isolates row length from
      # the segment feature itself.
      cases.append((f"C{budget}", pos_p, mask_p, None, shape_p,
                    f"[{r}, {budget}] packed shape, NO segment_ids"))

  # ---- production module (the primary measurement) -------------------------
  # model_lib.Attention on the production mesh: the module builds
  # make_splash_mha with head_shards=mesh['tp'], wraps it in shard_map and
  # vmaps over the batch, then runs the q/k/v/o projections around it. This is
  # literally what a training step executes.
  rngs = nnx.Rngs(params=args.seed)
  d = config.embed_dim
  key = jax.random.PRNGKey(args.seed)
  xs = {
      name: jax.random.normal(key, (*shape, d), jnp.bfloat16)
      for name, _, _, _, shape, _ in cases
  }

  module_results = {}
  if not args.skip_module:
    attn = model_lib.Attention(config=config, rngs=rngs)
    print()
    print("=" * 80)
    print(f"PRODUCTION ATTENTION MODULE -- model_lib.Attention on the"
          f" fsdp{fsdp}xtp{tp} mesh, hidden {d}, {config.num_heads}q/"
          f"{config.num_kv_heads}kv heads x {config.head_dim}")
    print("=" * 80)
    print(f"{'case':<12} {'shape':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")
    for name, pos, mask, seg, _, note in cases:
      (f_fwd, a_fwd), (f_fb, a_fb) = attn_fns(attn, xs[name], pos, mask, seg)
      t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
      t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
      module_results[name] = (t_fwd, t_fb)
      print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
      maybe_trace(args.trace_dest, f"attn_{name}", f_fb, a_fb, args.trace_iters)

  # ---- raw kernel, per chip (attribution) ----------------------------------
  # Same kernel the module runs, minus the projections, on one chip's share:
  # batch/fsdp rows and heads/tp heads (production's head_shards=tp splits the
  # heads, so one chip sees that many). Subtracting gives the projection cost.
  kernel_results = {}
  if not args.skip_kernel:
    print()
    print("=" * 80)
    print(f"RAW SPLASH KERNEL, ONE CHIP'S SHARE (batch/{fsdp} rows,"
          f" {config.num_heads // tp} of {config.num_heads} heads)")
    print("=" * 80)
    print(f"{'case':<12} {'shape/chip':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")
    for name, _, _, seg, shape, _ in cases:
      batch, seq_len = shape
      chip_batch = max(1, batch // fsdp)
      chip_seg = None if seg is None else seg[:chip_batch]
      (f_fwd, a_fwd), (f_fb, a_fb) = kernel_fns(
          config, chip_batch, seq_len, chip_seg, args.seed, heads_div=tp
      )
      t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
      t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
      eff = effective_attention_work(chip_seg) if chip_seg is not None else 0.0
      kernel_results[name] = (t_fwd, t_fb, chip_batch, seq_len, eff)
      note = f"[{chip_batch}, {seq_len}] per chip"
      print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
      maybe_trace(args.trace_dest, f"kernel_{name}", f_fb, a_fb,
                  args.trace_iters)

  # ---- segment-structure sweep at a FIXED shape ----------------------------
  # The budget sweep above varies row length and segment count together. This
  # holds the shape fixed and varies ONLY the segment structure, which is the
  # decisive test: if splash skipped cross-segment blocks, more segments would
  # mean a sparser block-diagonal and a lower cost.
  sweep_results = {}
  sweep_counts = [int(s) for s in args.segment_sweep.split(",") if s]
  if sweep_counts:
    b_sweep = args.sweep_budget
    real_frac = 1.0 - 0.234  # the packed dummy_ratio the geometry gate reports
    print()
    print("=" * 80)
    print(f"SEGMENT SWEEP -- shape fixed at [1, {b_sweep}], only the segment"
          " structure varies")
    print("=" * 80)
    print(f"{'case':<12} {'segments':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")

    variants = [(f"S{k}", k, False) for k in sweep_counts]
    # Skew control: same shape and segment count, wildly uneven segment sizes.
    if max(sweep_counts) > 1:
      variants.append((f"S{max(sweep_counts)}skew", max(sweep_counts), True))

    for name, k, skew in variants:
      seg = synthetic_segment_ids(1, b_sweep, k, real_frac=real_frac, skew=skew)
      (f_fwd, a_fwd), (f_fb, a_fb) = kernel_fns(
          config, 1, b_sweep, seg, args.seed
      )
      t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
      t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
      sweep_results[name] = (t_fwd, t_fb)
      note = f"{k} segments{' (skewed sizes)' if skew else ''}, real={real_frac:.2f}"
      print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
      maybe_trace(args.trace_dest, f"sweep_{name}", f_fb, a_fb,
                  args.trace_iters)

    # Cross-check: the real packed row of this shape must time like the
    # synthetic one with the same segment count.
    real_case = next(
        (c for c in cases
         if c[0] == f"P{b_sweep}" and c[4][0] == 1), None
    )
    if real_case is not None:
      real_segs = int(np.asarray(real_case[3]).max())
      seg = synthetic_segment_ids(1, b_sweep, real_segs, real_frac=real_frac)
      (_, _), (f_fb, a_fb) = kernel_fns(config, 1, b_sweep, seg, args.seed)
      t_syn = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
      t_real = kernel_results[f"P{b_sweep}"][1]
      print(f"{'XCHECK':<12} {f'real packer vs synthetic, {real_segs} segs':<40}"
            f" {'':>9} {t_real:>5.2f} vs {t_syn:.2f}")
      sweep_results["_xcheck"] = (t_real, t_syn)

  # ---- optional: full decoder layer (attention + MLP) -----------------------
  layer_results = {}
  if args.with_layer:
    layer = model_lib.DecoderLayer(config=config, rngs=rngs)
    print()
    print("=" * 80)
    print("FULL DECODER LAYER (attention + MLP) -- the configuration whose e2e"
          " total went flat")
    print("=" * 80)
    print(f"{'case':<12} {'shape':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")
    for name, pos, mask, seg, _, note in cases:
      if name.startswith("C"):
        continue
      (f_fwd, a_fwd), (f_fb, a_fb) = layer_fns(layer, xs[name], pos, mask, seg)
      t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
      t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
      layer_results[name] = (t_fwd, t_fb)
      print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
      maybe_trace(args.trace_dest, f"layer_{name}", f_fb, a_fb,
                  args.trace_iters)

  # ---- verdicts ------------------------------------------------------------
  print()
  print("=" * 80)
  print("VERDICTS")
  print("=" * 80)
  base_fwd, base_fb, base_b, base_t = kernel_results["U"]
  ref_exec = base_b * base_t * base_t / 2.0
  ref_eff = kernel_results["U"][4]
  print("  [V1] Every arm attends the SAME sequences, so the effective work is"
        " identical; the arms differ only in the padding and cross-segment"
        " area the kernel runs through on top:")
  print(f"       {'case':<10} {'shape':>12} {'effective':>11} {'executed':>10}"
        f" {'waste':>7} {'measured':>10}")
  for name, (t_fwd, t_fb, b, t, eff) in kernel_results.items():
    executed = b * t * t / 2.0
    eff_str = f"{eff / ref_eff:>10.2f}x" if eff else f"{'n/a':>11}"
    waste = f"{executed / eff:>6.1f}x" if eff else f"{'n/a':>7}"
    print(f"       {name:<10} {f'[{b}, {t}]':>12} {eff_str}"
          f" {executed / ref_exec:>9.2f}x {waste} {t_fb / base_fb:>9.2f}x")
  print("       ^ effective = sum of per-segment len^2/2 (what the sequences"
        " actually require); executed = rows*len^2/2 (the full causal area the"
        " static mask schedules). If measured tracks EXECUTED rather than"
        " EFFECTIVE, the kernel is computing padding and cross-segment blocks"
        " and segment_ids is only masking them afterwards.")
  if any(n.startswith("C") for n in kernel_results):
    print("       C vs its P twin isolates segment_ids: equal cost means"
          " segment_ids does not save compute, only masks.")
  if sweep_results:
    keys = [k for k in sweep_results if k != "_xcheck" and "skew" not in k]
    sbase = sweep_results[keys[0]][1]
    spread = max(sweep_results[k][1] for k in keys) / min(
        sweep_results[k][1] for k in keys
    )
    print(f"  [V1b] segment sweep at a fixed [1, {args.sweep_budget}] row"
          f" -- spread across {len(keys)} segment counts = {spread:.2f}x")
    for k in keys:
      print(f"       {k:<10} bwd {sweep_results[k][1] / sbase:6.2f}x S1")
    skew_key = next((k for k in sweep_results if "skew" in k), None)
    if skew_key:
      flat = sweep_results[skew_key.replace("skew", "")][1]
      print(f"       {skew_key:<10} bwd"
            f" {sweep_results[skew_key][1] / flat:6.2f}x the even split"
            "   (segment SIZE distribution -- 1.00x kills load balancing)")
    print("       ~1.00x throughout means segment_ids only masks blocks that"
          " were computed anyway; a real skip would fall with more segments.")
    if "_xcheck" in sweep_results:
      t_real, t_syn = sweep_results["_xcheck"]
      print(f"  [XC] real packed row {t_real:.2f} ms vs synthetic same-geometry"
            f" {t_syn:.2f} ms ({t_syn / t_real:.2f}x) -- close means the"
            " synthetic sweep is representative of production packing.")
  if module_results:
    mbase = module_results["U"][1]
    print("  [V1'] PRODUCTION attention module (what a training step pays):")
    for name, (_, t_fb) in module_results.items():
      line = f"       {name:<10} bwd {t_fb / mbase:6.2f}x U"
      if name in kernel_results:
        k = kernel_results[name][1]
        line += (f"   = kernel {k:.2f}ms + projections/norm"
                 f" {max(0.0, t_fb - k):.2f}ms")
      print(line)
    print("       ^ the projections are linear in the token count, so packing"
          " makes them cheaper and they partly cancel the attention penalty --"
          " which is why the e2e total moved less than the attention did.")
  if layer_results:
    lbase = layer_results["U"][1]
    print("  [V3] full layer -- the flat e2e total should reappear at the e2e"
          " budget, and a smaller budget should win:")
    for name, (_, t_fb) in layer_results.items():
      print(f"       {name:<10} bwd {t_fb / lbase:6.2f}x U")
  print()
  print("  [V2] match the xprof kernel names against the e2e trace: packed and"
        " unpacked both use *_segmented_{fwd,dq,dkv}; only the shapes differ.")
  if args.trace_dest:
    print(f"       traces: {args.trace_dest}/splash_bench_*")


if __name__ == "__main__":
  main()
