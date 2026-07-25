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

The primary measurement is the RAW KERNEL, through the same entry point JAX's
own tests use (`splash.make_splash_mha_single_device` + `jax.vmap` over the
batch, the composition `model_lib.Attention` performs inside its shard_map).
Nothing else is in the timing: no q/k/v/o projections (linear in tokens, so
cheaper when packed -- they would partly cancel the very penalty being
measured), no mesh and no shard_map (a packed geometry can have fewer rows
than chips, which cannot shard). Shapes are therefore ONE CHIP's share of the
e2e run: 32 sequences over fsdp 4 = 8.

  U        [8, 2048]   unpacked baseline, per-position non-pad segment_ids
  P<budget>[rows, b]   packed, real pack_sequences segment ids
  C<budget>[rows, b]   the largest packed shape with segment_ids dropped
                       (isolates row length from the segment feature itself)

Optional context, off by default: --with_module adds `model_lib.Attention`
(kernel + projections) and --with_layer adds a full `DecoderLayer`
(attention + MLP), which is what reproduces the flat e2e total.

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
  p.add_argument("--num_seqs", type=int, default=8,
                 help="Sequences per chip (the e2e run's 32 over fsdp 4).")
  p.add_argument("--seq_len", type=int, default=2048,
                 help="Padded per-sequence length of the unpacked baseline.")
  p.add_argument("--min_tokens", type=int, default=700,
                 help="Real-token sampling range; the default 700-950 lands at")
  p.add_argument("--max_tokens", type=int, default=950,
                 help="a ~20%% dummy_ratio, matching the e2e run.")
  p.add_argument("--budgets", type=str, default="2048,4096,8192,16384",
                 help="Packing budgets to bench; several points make the"
                      " linear-in-budget prediction directly visible.")
  p.add_argument("--iters", type=int, default=20,
                 help="Timed samples per case (median reported).")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--inner", type=int, default=5,
                 help="Calls dispatched per timed sample, to amortize JAX's"
                      " per-dispatch overhead.")
  p.add_argument("--trace_dest", type=str, default="",
                 help="If set, write one xprof trace per case under here.")
  p.add_argument("--trace_iters", type=int, default=3)
  p.add_argument("--with_module", action="store_true",
                 help="Also time model_lib.Attention (kernel + projections).")
  p.add_argument("--with_layer", action="store_true",
                 help="Also time a full DecoderLayer (attention + MLP).")
  p.add_argument("--model_config", type=str, default="qwen3_1p7b",
                 help="ModelConfig factory: qwen3_1p7b (the gsm8k run) or"
                      " qwen3_8b (FrozenLake).")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


# ---------------------------------------------------------------------------
# Data: build TrainExamples and pack them with the production packer.
# ---------------------------------------------------------------------------
def make_examples(num_seqs, seq_len, min_tokens, max_tokens, seed):
  """`num_seqs` TrainExamples, each padded to `seq_len` with a random real length.

  Mirrors the RL producer's layout: prompts are LEFT-padded, completions are
  RIGHT-padded, so `unpad_train_example` inside pack_sequences recovers exactly
  the real tokens.
  """
  rng = np.random.default_rng(seed)
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
def kernel_fns(config, batch, seq_len, segment_ids, seed):
  """(forward, forward+backward) closures over the RAW splash kernel.

  Built the way JAX's own splash tests build it -- `make_splash_mha_single_
  device` (which is `make_splash_mha` with head_shards=q_seq_shards=1) called
  under `jax.vmap` over the batch, the same composition `model_lib.Attention`
  performs inside its shard_map. Single-device means no mesh and no sharding
  constraint on the row count, which a packed geometry can easily violate.
  """
  qh, kh, hd = config.num_heads, config.num_kv_heads, config.head_dim
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

  print(f"jax devices: {jax.devices()}")
  print(
      f"geometry (per chip): {args.num_seqs} seqs x {args.seq_len} padded "
      f"(real {args.min_tokens}-{args.max_tokens}), budgets={budgets}"
  )

  examples, total_real = make_examples(
      args.num_seqs, args.seq_len, args.min_tokens, args.max_tokens, args.seed
  )
  print(f"total real tokens: {total_real}")

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
    ex = pack(examples, budget, rows, args.num_seqs)
    r, segs, dummy = geometry(ex, budget, total_real)
    pos_p, mask_p, seg_p, shape_p = model_inputs(ex)
    note = f"[{r}, {budget}] packed, segs/row={segs}, dummy={dummy:.3f}"
    cases.append((f"P{budget}", pos_p, mask_p, seg_p, shape_p, note))
    if budget == max(budgets):
      # Control: same shape, segment_ids dropped -> isolates row length from
      # the segment feature itself.
      cases.append((f"C{budget}", pos_p, mask_p, None, shape_p,
                    f"[{r}, {budget}] packed shape, NO segment_ids"))

  # ---- raw kernel (the primary measurement) --------------------------------
  print()
  print("=" * 80)
  print("RAW SPLASH KERNEL (make_splash_mha_single_device, vmapped over batch)")
  print("=" * 80)
  print(f"{'case':<12} {'shape':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")
  kernel_results = {}
  for name, _, _, seg, shape, note in cases:
    batch, seq_len = shape
    (f_fwd, a_fwd), (f_fb, a_fb) = kernel_fns(
        config, batch, seq_len, seg, args.seed
    )
    t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
    t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
    kernel_results[name] = (t_fwd, t_fb, batch, seq_len)
    print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
    maybe_trace(args.trace_dest, f"kernel_{name}", f_fb, a_fb, args.trace_iters)

  # ---- optional context: module and full layer -----------------------------
  module_results, layer_results = {}, {}
  if args.with_module or args.with_layer:
    # A 1-chip mesh: packed geometries can have fewer rows than chips, which a
    # sharded batch axis (act_btd/act_btnh put batch on fsdp) cannot split.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")
    )
    mesh.__enter__()
    rngs = nnx.Rngs(params=args.seed)
    d = config.embed_dim
    key = jax.random.PRNGKey(args.seed)
    xs = {
        name: jax.random.normal(key, (*shape, d), jnp.bfloat16)
        for name, _, _, _, shape, _ in cases
    }

    if args.with_module:
      attn = model_lib.Attention(config=config, rngs=rngs)
      print()
      print("=" * 80)
      print("ATTENTION MODULE (kernel + q/k/v/o projections)")
      print("=" * 80)
      print(f"{'case':<12} {'shape':<40} {'fwd ms':>9} {'fwd+bwd ms':>11}")
      for name, pos, mask, seg, _, note in cases:
        (f_fwd, a_fwd), (f_fb, a_fb) = attn_fns(attn, xs[name], pos, mask, seg)
        t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup, args.inner)
        t_fb = timed(f_fb, a_fb, args.iters, args.warmup, args.inner)
        module_results[name] = (t_fwd, t_fb)
        print(f"{name:<12} {note:<40} {t_fwd:>9.2f} {t_fb:>11.2f}")
        maybe_trace(args.trace_dest, f"attn_{name}", f_fb, a_fb,
                    args.trace_iters)

    if args.with_layer:
      layer = model_lib.DecoderLayer(config=config, rngs=rngs)
      print()
      print("=" * 80)
      print("FULL DECODER LAYER (attention + MLP)")
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
  print(f"  [V1] budget law -- kernel cost should track rows * budget^2"
        f" (= total tokens * budget), i.e. be LINEAR in the budget:")
  ref_work = base_b * base_t * base_t
  for name, (t_fwd, t_fb, b, t) in kernel_results.items():
    work = b * t * t
    print(f"       {name:<10} bwd {t_fb / base_fb:6.2f}x U   "
          f"predicted {work / ref_work:6.2f}x   (rows*budget^2)")
  if any(n.startswith("C") for n in kernel_results):
    print("       C vs its P twin isolates segment_ids: equal cost means"
          " segment_ids does not save compute, only masks.")
  if module_results:
    mbase = module_results["U"][1]
    print("  [V1'] attention MODULE (mixes in token-linear projections, which"
          " are cheaper when packed and pull the ratio below the kernel's):")
    for name, (_, t_fb) in module_results.items():
      print(f"       {name:<10} bwd {t_fb / mbase:6.2f}x U")
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
